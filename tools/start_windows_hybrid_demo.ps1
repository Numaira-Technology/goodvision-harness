[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [ValidateRange(1, 65535)]
    [int]$HarnessPort = 8082,
    [string]$RichEndpoint = ""
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
$providerJson = & $PythonPath -c "import json, onnxruntime as ort; print(json.dumps(ort.get_available_providers()))"
if ($LASTEXITCODE -ne 0) {
    throw "Could not import the DirectML ONNX Runtime environment."
}
if (-not ($providerJson -match 'DmlExecutionProvider')) {
    throw "DmlExecutionProvider is unavailable in this Python environment: $providerJson"
}
Write-Host "InsightFace provider check: $providerJson"

$experimentEnvironmentNames = @(
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
    $richReady = $false
    if ($RichEndpoint) {
        try {
            $endpointUri = [Uri]$RichEndpoint
            if (-not $endpointUri.IsAbsoluteUri -or $endpointUri.Scheme -notin @("http", "https")) {
                throw "The optional rich endpoint must be an HTTP URL."
            }
            $richReadyUrl = [Uri]::new($endpointUri, "/health/ready").AbsoluteUri
            $capability = Invoke-RestMethod -Uri $richReadyUrl -TimeoutSec 3
            $gpuProperty = $capability.PSObject.Properties["gpu_ready"]
            if ($null -ne $gpuProperty -and $gpuProperty.Value -is [bool] -and $gpuProperty.Value) {
                $richReady = $true
            }
        }
        catch {
            # Optional annotation must not delay or prevent face/event testing.
            $richReady = $false
        }
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
    $env:RICH_LABEL_ENABLED = "false"
    $env:RICH_LABEL_ENDPOINT = ""
    if ($richReady) {
        $env:RICH_LABEL_ENABLED = "true"
        $env:RICH_LABEL_ENDPOINT = $RichEndpoint
    }

    if ($richReady) {
        Write-Host "Rich annotation: verified external GPU worker"
    }
    else {
        Write-Host "Rich annotation: disabled (no verified GPU worker)"
    }
    Write-Host "InsightFace: DmlExecutionProvider, full live schedule"
    Write-Host "Open http://127.0.0.1:$HarnessPort/ and choose Camera."
    Write-Host "Press Ctrl+C to stop the native Harness."

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
