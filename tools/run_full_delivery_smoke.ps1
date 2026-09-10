#requires -Version 7.0
<#
.SYNOPSIS
Run the full-image synthetic protocol smoke check in disposable containers.
.DESCRIPTION
Uses existing local images only. Creates an internal network without published
ports or production volumes. The Paddle image runs its actual app.py with the
deterministic test fixture; no Paddle import, model download, or real GPU/model
inference is performed. The Harness is explicitly dry-run with CPU face provider
only for this check. Production image defaults and files are not modified.

Requires PowerShell 7 and Docker Desktop/Linux containers. The two Python smoke
tools must be alongside this script. No image/container IDs or checksum files
are emitted. Existing resources with the reserved names cause an early failure.
#>
[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string] $HarnessImage = 'goodvision-harness:2026.09.10-full.1',

    [ValidateNotNullOrEmpty()]
    [string] $WorkerImage = 'goodvision-rich-label-worker:2026.09.10-full.1'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$dockerCommand = Get-Command docker -CommandType Application -ErrorAction Stop |
    Select-Object -First 1
$dockerExecutable = $dockerCommand.Source
$harnessName = 'goodvision-full-delivery-smoke-harness'
$workerName = 'goodvision-full-delivery-smoke-worker'
$networkName = 'goodvision-full-delivery-smoke-net'
$ownerLabel = 'com.goodvision.delivery-smoke-run'
$runOwner = '{0}-{1}' -f [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffff'), $PID
$networkCreated = $false
$workerCreated = $false
$harnessCreated = $false

function Invoke-SmokeDocker {
    param(
        [Parameter(Mandatory)]
        [string[]] $Arguments,

        [ValidateRange(1, 180)]
        [int] $TimeoutSeconds = 30,

        [AllowEmptyString()]
        [string] $InputText,

        [switch] $AllowFailure,

        [string] $Operation = 'Docker operation'
    )

    $processInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $processInfo.FileName = $dockerExecutable
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    $hasInput = $PSBoundParameters.ContainsKey('InputText')
    $processInfo.RedirectStandardInput = $hasInput
    foreach ($argument in $Arguments) {
        $processInfo.ArgumentList.Add($argument)
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $processInfo
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $timedOut = $false
    $startedSuccessfully = $false
    try {
        if (-not $process.Start()) {
            throw "$Operation could not start."
        }
        $startedSuccessfully = $true
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if ($hasInput) {
            $writeTask = $process.StandardInput.WriteAsync($InputText)
            $writeBudget = [Math]::Min(10000, $TimeoutSeconds * 1000)
            if (-not $writeTask.Wait($writeBudget)) {
                $timedOut = $true
            }
            if (-not $timedOut) {
                $process.StandardInput.Close()
            }
        }
        if (-not $timedOut) {
            $remaining = [Math]::Max(
                1, [int]($TimeoutSeconds * 1000 - $watch.ElapsedMilliseconds)
            )
            $timedOut = -not $process.WaitForExit($remaining)
        }
        if ($timedOut) {
            if (-not $process.HasExited) {
                $process.Kill($true)
            }
            [void] $process.WaitForExit(5000)
            throw "$Operation exceeded its ${TimeoutSeconds}s time limit."
        }
        $result = [pscustomobject]@{
            ExitCode = $process.ExitCode
            Output = $stdoutTask.GetAwaiter().GetResult()
            ErrorOutput = $stderrTask.GetAwaiter().GetResult()
        }
        if ($result.ExitCode -ne 0 -and -not $AllowFailure) {
            # Container/network creation can print internal references. Keep
            # those captured; do not echo raw Docker output into the console.
            throw "$Operation failed with exit code $($result.ExitCode)."
        }
        return $result
    }
    finally {
        if ($startedSuccessfully -and -not $process.HasExited) {
            # A write/read failure must not leave the Docker CLI client behind.
            # The containers themselves are cleaned by the outer finally block.
            try {
                $process.Kill($true)
                [void] $process.WaitForExit(3000)
            }
            catch {
                Write-Warning 'Could not terminate a timed-out Docker client.' `
                    -WarningAction Continue
            }
        }
        $process.Dispose()
    }
}

function Wait-SmokeHealth {
    param(
        [Parameter(Mandatory)] [string] $ContainerName,
        [Parameter(Mandatory)] [int] $Port,
        [Parameter(Mandatory)] [string] $Path
    )

    $probe = 'import urllib.request; r = urllib.request.urlopen(' +
        "'http://127.0.0.1:${Port}${Path}', timeout=2); " +
        'assert r.status == 200; r.close()'
    $deadline = [DateTime]::UtcNow.AddSeconds(60)
    do {
        $result = Invoke-SmokeDocker -Arguments @(
            'exec', $ContainerName, 'python', '-c', $probe
        ) -TimeoutSeconds 6 -AllowFailure -Operation "Health probe for $ContainerName"
        if ($result.ExitCode -eq 0) {
            Write-Host "PASS  $ContainerName is ready for the isolated check."
            return
        }
        Start-Sleep -Milliseconds 400
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Timed out waiting for $ContainerName health."
}

function Remove-OwnedSmokeContainer {
    param(
        [Parameter(Mandatory)] [string] $ContainerName,
        [Parameter(Mandatory)] [bool] $Created
    )

    if (-not $Created) {
        return
    }
    try {
        $template = '{{ index .Config.Labels "' + $ownerLabel + '" }}'
        $inspection = Invoke-SmokeDocker -Arguments @(
            'container', 'inspect', '--format', $template, $ContainerName
        ) -TimeoutSeconds 8 -AllowFailure -Operation 'Inspect temporary container owner'
        if ($inspection.ExitCode -ne 0) {
            # A --rm container may already have exited and removed itself.
            return
        }
        if ($inspection.Output.Trim() -ne $runOwner) {
            Write-Warning "Leaving $ContainerName untouched: ownership changed." `
                -WarningAction Continue
            return
        }
        $stopped = Invoke-SmokeDocker -Arguments @(
            'stop', '--timeout', '10', $ContainerName
        ) -TimeoutSeconds 15 -AllowFailure -Operation 'Stop temporary container'
        if ($stopped.ExitCode -ne 0) {
            # The target was confirmed to belong to this run and has no user
            # volumes; removing this disposable fixture is safe.
            $removed = Invoke-SmokeDocker -Arguments @(
                'container', 'rm', '--force', $ContainerName
            ) -TimeoutSeconds 15 -AllowFailure -Operation 'Remove temporary container'
            if ($removed.ExitCode -ne 0) {
                Write-Warning "Could not confirm cleanup of $ContainerName." `
                    -WarningAction Continue
                return
            }
        }
        Write-Host "CLEAN  Removed temporary container $ContainerName."
    }
    catch {
        Write-Warning "Temporary container cleanup needs attention: $ContainerName." `
            -WarningAction Continue
    }
}

function Remove-OwnedSmokeNetwork {
    if (-not $networkCreated) {
        return
    }
    try {
        $template = '{{ index .Labels "' + $ownerLabel + '" }}'
        $inspection = Invoke-SmokeDocker -Arguments @(
            'network', 'inspect', '--format', $template, $networkName
        ) -TimeoutSeconds 8 -AllowFailure -Operation 'Inspect temporary network owner'
        if ($inspection.ExitCode -ne 0) {
            return
        }
        if ($inspection.Output.Trim() -ne $runOwner) {
            Write-Warning "Leaving $networkName untouched: ownership changed." `
                -WarningAction Continue
            return
        }
        $removed = Invoke-SmokeDocker -Arguments @(
            'network', 'rm', $networkName
        ) -TimeoutSeconds 15 -AllowFailure -Operation 'Remove temporary internal network'
        if ($removed.ExitCode -eq 0) {
            Write-Host "CLEAN  Removed temporary internal network $networkName."
        }
        else {
            Write-Warning "Could not confirm cleanup of $networkName." `
                -WarningAction Continue
        }
    }
    catch {
        Write-Warning "Temporary network cleanup needs attention: $networkName." `
            -WarningAction Continue
    }
}

$workerSmokePath = (Resolve-Path -LiteralPath (
    Join-Path $PSScriptRoot 'paddle_delivery_smoke.py'
)).Path
$harnessSmokePath = (Resolve-Path -LiteralPath (
    Join-Path $PSScriptRoot 'delivery_smoke.py'
)).Path
if ($workerSmokePath.Contains(',')) {
    throw 'The smoke-tool path must not contain a comma for a Docker bind mount.'
}
$harnessSmokeSource = [System.IO.File]::ReadAllText($harnessSmokePath)

Write-Host 'Synthetic full-image protocol check; no real GPU/model inference is tested.'
Write-Host 'Only new disposable containers and an internal network will be used.'

# Read-only preflight: failure cannot delete or replace any existing resource.
$containers = Invoke-SmokeDocker -Arguments @(
    'container', 'ls', '--all', '--format', '{{.Names}}'
) -Operation 'List existing container names'
$existingContainerNames = @($containers.Output -split '\r?\n')
foreach ($reservedName in @($workerName, $harnessName)) {
    if ($existingContainerNames -contains $reservedName) {
        throw "Reserved container name already exists: $reservedName. No changes made."
    }
}
$networks = Invoke-SmokeDocker -Arguments @(
    'network', 'ls', '--format', '{{.Name}}'
) -Operation 'List existing network names'
if (@($networks.Output -split '\r?\n') -contains $networkName) {
    throw "Reserved network name already exists: $networkName. No changes made."
}
foreach ($imageName in @($HarnessImage, $WorkerImage)) {
    $platform = Invoke-SmokeDocker -Arguments @(
        'image', 'inspect', '--format', '{{.Os}}/{{.Architecture}}', $imageName
    ) -Operation 'Inspect local delivery image platform'
    if ($platform.Output.Trim() -ne 'linux/amd64') {
        throw 'Both delivery images must already exist locally as linux/amd64 images.'
    }
}

try {
    $null = Invoke-SmokeDocker -Arguments @(
        'network', 'create', '--internal', '--label', "$ownerLabel=$runOwner", $networkName
    ) -Operation 'Create temporary internal network'
    $networkCreated = $true
    $internal = Invoke-SmokeDocker -Arguments @(
        'network', 'inspect', '--format', '{{.Internal}}', $networkName
    ) -Operation 'Verify internal network isolation'
    if ($internal.Output.Trim() -ne 'true') {
        throw 'The temporary network is not internal; refusing to start containers.'
    }

    $workerArguments = @(
        'run', '--rm', '--detach', '--pull', 'never', '--init',
        '--name', $workerName, '--label', "$ownerLabel=$runOwner",
        '--network', $networkName, '--network-alias', 'rich-label-worker',
        '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges:true',
        '--env', 'NO_PROXY=*', '--env', 'no_proxy=*',
        '--tmpfs', '/tmp:rw,nosuid,nodev,size=256m',
        '--mount', "type=bind,source=$workerSmokePath,target=/smoke/paddle_delivery_smoke.py,readonly",
        '--entrypoint', 'python', $WorkerImage,
        '/smoke/paddle_delivery_smoke.py', '--isolated', '--serve', '--host', '0.0.0.0'
    )
    $null = Invoke-SmokeDocker -Arguments $workerArguments -Operation 'Start synthetic worker'
    $workerCreated = $true
    Wait-SmokeHealth -ContainerName $workerName -Port 8091 -Path '/health/ready'

    $harnessArguments = @(
        'run', '--rm', '--detach', '--pull', 'never', '--init',
        '--name', $harnessName, '--label', "$ownerLabel=$runOwner",
        '--network', $networkName, '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges:true',
        '--env', 'NO_PROXY=*', '--env', 'no_proxy=*',
        '--tmpfs', '/tmp:rw,nosuid,nodev,size=256m',
        '--tmpfs', '/data:rw,nosuid,nodev,size=64m,uid=10001,gid=10001,mode=700',
        '--env', 'HARNESS_CONFIG=/app/config.yaml',
        '--env', 'HARNESS_DEPLOYMENT_SETTINGS_PATH=',
        '--env', 'HARNESS_HOST=0.0.0.0', '--env', 'HARNESS_PORT=8080',
        '--env', 'HARNESS_AUTOSTART=false', '--env', 'HARNESS_DRY_RUN=true',
        '--env', 'HARNESS_STREAM_URL=', '--env', 'RTSP_URL=',
        '--env', 'AICLUB_API_KEY=', '--env', 'INFERENCE_API_KEY=', '--env', 'RESULT_WEBHOOK_URL=',
        '--env', 'HARNESS_WEBHOOK_SECRET=',
        '--env', 'FACE_PROVIDER=CPUExecutionProvider',
        '--env', 'FACE_ACCELERATOR=cpu',
        '--env', 'FACE_ALLOW_MODEL_DOWNLOAD=false',
        '--env', 'FACE_MODEL_ROOT=/models',
        '--env', 'FACE_GALLERY_PATH=/data/face-gallery.json',
        '--env', 'RICH_LABEL_ENABLED=true',
        '--env', 'RICH_LABEL_ENDPOINT=http://rich-label-worker:8091/v1/labels',
        $HarnessImage
    )
    $null = Invoke-SmokeDocker -Arguments $harnessArguments -Operation 'Start dry-run Harness'
    $harnessCreated = $true
    Wait-SmokeHealth -ContainerName $harnessName -Port 8080 -Path '/health/live'

    Write-Host 'Running five-capability, rich-label, JSON/SSE and video protocol checks.'
    $smoke = Invoke-SmokeDocker -Arguments @(
        'exec', '--interactive', $harnessName, 'python', '-', '--isolated', '--with-rich', '--audit-branding'
    ) -InputText $harnessSmokeSource -TimeoutSeconds 150 -AllowFailure `
        -Operation 'Full-image synthetic protocol check'
    # Only the known Python smoke program output is displayed. Docker commands
    # that create/delete resources retain their output inside this script.
    foreach ($line in @($smoke.Output -split '\r?\n')) {
        if ($line -match '^PASS  ') {
            Write-Host $line
        }
    }
    if ($smoke.ExitCode -ne 0) {
        foreach ($line in @($smoke.ErrorOutput -split '\r?\n')) {
            if ($line -match '^FAIL  ') {
                Write-Host $line
            }
        }
        throw "Full-image synthetic protocol check failed (exit $($smoke.ExitCode))."
    }
    Write-Host 'PASS  Both delivery images passed the isolated synthetic HTTP integration.'
    Write-Host 'Real NVIDIA GPU, real models, and model accuracy still need target-host checks.'
}
finally {
    # Cleanup warnings must never replace the original test/startup exception.
    foreach ($cleanupTarget in @(
        @{ Name = $harnessName; Created = $harnessCreated },
        @{ Name = $workerName; Created = $workerCreated }
    )) {
        try {
            Remove-OwnedSmokeContainer -ContainerName $cleanupTarget.Name `
                -Created $cleanupTarget.Created
        }
        catch {
            Write-Warning "Cleanup needs attention: $($cleanupTarget.Name)." `
                -WarningAction Continue
        }
    }
    try {
        Remove-OwnedSmokeNetwork
    }
    catch {
        Write-Warning "Cleanup needs attention: $networkName." -WarningAction Continue
    }
}
