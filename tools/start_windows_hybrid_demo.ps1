[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [ValidateRange(1, 65535)]
    [int]$HarnessPort = 8082,
    [ValidateRange(1, 65535)]
    [int]$RichPort = 8091,
    [ValidateRange(1, 3600)]
    [int]$RichReadyTimeoutSeconds = 900,
    [switch]$NoRichBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $PythonPath) {
    $PythonPath = Join-Path $repoRoot ".venv-directml\Scripts\python.exe"
}
if (-not [System.IO.Path]::IsPathRooted($PythonPath)) {
    $PythonPath = Join-Path $repoRoot $PythonPath
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "DirectML Python was not found at '$PythonPath'. Follow WINDOWS_HYBRID.md first."
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Docker Desktop must be running for the CPU Paddle worker."
}
if ($HarnessPort -eq $RichPort) {
    throw "HarnessPort and RichPort must be different."
}

$providerJson = & $PythonPath -c "import json, onnxruntime as ort; print(json.dumps(ort.get_available_providers()))"
if ($LASTEXITCODE -ne 0) {
    throw "Could not import the DirectML ONNX Runtime environment."
}
if (-not ($providerJson -match 'DmlExecutionProvider')) {
    throw "DmlExecutionProvider is unavailable in this Python environment: $providerJson"
}
Write-Host "InsightFace provider check: $providerJson"

$experimentEnvironmentNames = @(
    "RICH_HOST_PORT",
    "HARNESS_CONFIG",
    "HARNESS_DEPLOYMENT_SETTINGS_PATH",
    "HARNESS_AUTOSTART",
    "HARNESS_DRY_RUN",
    "HARNESS_HOST",
    "HARNESS_PORT",
    "FACE_MODEL_ROOT",
    "FACE_GALLERY_PATH",
    "FACE_PROVIDER",
    "FACE_LIVE_SCHEDULER",
    "RICH_LABEL_ENABLED",
    "RICH_LABEL_ENDPOINT"
)
$originalEnvironment = @{}
foreach ($name in $experimentEnvironmentNames) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable(
        $name,
        "Process"
    )
}

try {
    $env:RICH_HOST_PORT = [string]$RichPort
    $composeArgs = @(
        "compose",
        "-f", (Join-Path $repoRoot "compose.yaml"),
        "-f", (Join-Path $repoRoot "compose.rich-cpu-host.yaml"),
        "--profile", "rich"
    )
    if (-not $NoRichBuild) {
        & docker @composeArgs build rich-label-worker
        if ($LASTEXITCODE -ne 0) {
            throw "The CPU Paddle worker image could not be built."
        }
    }
    $upArgs = @("up", "-d")
    if ($NoRichBuild) {
        $upArgs += "--no-build"
    }
    $upArgs += "rich-label-worker"
    & docker @composeArgs @upArgs
    if ($LASTEXITCODE -ne 0) {
        throw "The CPU Paddle worker could not be started."
    }

    $richReadyUrl = "http://127.0.0.1:$RichPort/health/ready"
    $deadline = [DateTime]::UtcNow.AddSeconds($RichReadyTimeoutSeconds)
    $richReady = $false
    Write-Host "Waiting for the complete CPU Paddle worker at $richReadyUrl ..."
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $richReadyUrl -TimeoutSec 3
            if ($response.StatusCode -eq 200) {
                $richReady = $true
                break
            }
        }
        catch {
            # Model download/warmup and transient connection failures are expected.
        }
        Start-Sleep -Seconds 2
    }
    if (-not $richReady) {
        & docker @composeArgs logs --tail 80 rich-label-worker
        throw "The CPU Paddle worker did not become ready within $RichReadyTimeoutSeconds seconds."
    }

    $modelRoot = Join-Path $repoRoot ".models\insightface"
    $galleryPath = Join-Path $repoRoot ".goodvision\face-gallery.json"
    New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path $galleryPath) | Out-Null

    # Process-level overrides win over .env and make this experiment deterministic.
    $env:HARNESS_CONFIG = Join-Path $repoRoot "config.yaml"
    $env:HARNESS_DEPLOYMENT_SETTINGS_PATH = ""
    $env:HARNESS_AUTOSTART = "false"
    $env:HARNESS_DRY_RUN = "false"
    $env:HARNESS_HOST = "127.0.0.1"
    $env:HARNESS_PORT = [string]$HarnessPort
    $env:FACE_MODEL_ROOT = $modelRoot
    $env:FACE_GALLERY_PATH = $galleryPath
    $env:FACE_PROVIDER = "DmlExecutionProvider"
    $env:FACE_LIVE_SCHEDULER = "full"
    $env:RICH_LABEL_ENABLED = "true"
    $env:RICH_LABEL_ENDPOINT = "http://127.0.0.1:$RichPort/v1/labels"

    Write-Host "CPU Paddle worker: ready"
    Write-Host "InsightFace: DmlExecutionProvider, full live schedule"
    Write-Host "Open http://127.0.0.1:$HarnessPort/ and choose Camera."
    Write-Host "Press Ctrl+C to stop the native Harness; the Paddle container stays warm."

    Push-Location $repoRoot
    try {
        & $PythonPath -m goodvision_harness.service
        if ($LASTEXITCODE -ne 0) {
            throw "The native Harness exited with code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    foreach ($name in $experimentEnvironmentNames) {
        [Environment]::SetEnvironmentVariable(
            $name,
            $originalEnvironment[$name],
            "Process"
        )
    }
}
