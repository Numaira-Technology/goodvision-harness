#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$PackageDir,
    [ValidatePattern('^[a-z0-9][a-z0-9_-]*$')][string]$Project = 'goodvision-contour',
    [ValidateRange(30, 3600)][int]$TimeoutSeconds = 900,
    [switch]$Rollback,
    [switch]$CheckOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 may bind advanced-script defaults before PSScriptRoot
# is populated. Resolve this in the script body, before calling any functions.
if ([string]::IsNullOrWhiteSpace($PackageDir)) { $PackageDir = $PSScriptRoot }
if ([string]::IsNullOrWhiteSpace($PackageDir)) { throw 'Cannot locate the update package. Pass -PackageDir with its full directory path.' }
$script:Utf8 = New-Object Text.UTF8Encoding($false)

function Invoke-UpdateDocker {
    param([string[]]$Arguments, [switch]$Capture, [switch]$Quiet, [switch]$AllowFailure)
    $ErrorActionPreference = 'Continue'
    if ($Quiet) { & docker @Arguments *> $null; $code = $LASTEXITCODE }
    elseif ($Capture) { $lines = @(& docker @Arguments); $code = $LASTEXITCODE }
    else { & docker @Arguments | Out-Host; $code = $LASTEXITCODE }
    if ($AllowFailure) { return $code }
    if ($code -ne 0) { throw "Docker $($Arguments[0]) failed (exit $code)." }
    if ($Capture) { return ($lines -join "`n") }
}
function Get-UpdateImage([string]$Name) {
    return (Invoke-UpdateDocker -Arguments @('image', 'inspect', '--format', '{{.Id}}', $Name) -Capture).Trim()
}
function Get-UpdateContainer([string]$Service) {
    $ids = @(((Invoke-UpdateDocker -Arguments @('ps', '-a', '-q', '--filter', "label=com.docker.compose.project=$Project", '--filter', "label=com.docker.compose.service=$Service") -Capture) -split '\s+') | Where-Object { $_ })
    if ($ids.Count -ne 1) { throw "Expected exactly one $Project / $Service container; found $($ids.Count)." }
    return @((Invoke-UpdateDocker -Arguments @('inspect', $ids[0]) -Capture | ConvertFrom-Json))[0]
}
function ConvertTo-UpdatePython([string]$Code) {
    return "exec(__import__('base64').b64decode('$([Convert]::ToBase64String($script:Utf8.GetBytes($Code)))'))"
}
function Invoke-UpdateNode([string]$Operation) {
    $code = @'
import json, pathlib, urllib.request
key = pathlib.Path('/run/secrets/node_key').read_text().strip()
op = OPERATION
path = '/api/gpu-memory' if op == 'memory' else '/api/admission' if op in ('pause', 'resume') else '/api/status'
data = json.dumps({'accepting_sessions': op == 'resume'}).encode() if op in ('pause', 'resume') else None
request = urllib.request.Request('http://127.0.0.1:8093' + path, data=data, headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
value = json.load(urllib.request.urlopen(request, timeout=5))
print(json.dumps(value if op == 'memory' else {k: value.get(k) for k in ('gpu_ready','active_sessions','accepting_sessions','requests')}))
'@
    $code = $code.Replace('OPERATION', "'$Operation'")
    return (Invoke-UpdateDocker -Arguments ($script:Compose + @('exec', '-T', 'contour', 'python', '-c', (ConvertTo-UpdatePython $code))) -Capture | ConvertFrom-Json)
}
function Wait-UpdateReady([bool]$RequireMemory) {
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $nextMessage = 0
    do {
        try {
            $status = Invoke-UpdateNode 'status'
            if ($status.gpu_ready -eq $true) {
                if ($RequireMemory) {
                    $memory = Invoke-UpdateNode 'memory'
                    if ($memory.available -ne $true -or $memory.reclaim_count -lt 1 -or $memory.error_code) { throw 'GPU memory initialization pending.' }
                }
                return
            }
        } catch { }
        $engine = Get-UpdateContainer 'engine'
        if ($engine.State.OOMKilled -or $engine.State.Status -in @('exited','dead','restarting')) { throw 'Engine exited or restarted during warmup.' }
        if ($watch.Elapsed.TotalSeconds -ge $nextMessage) {
            Write-Host 'Waiting for the existing model to load and finish GPU warmup...'
            $nextMessage = $watch.Elapsed.TotalSeconds + 30
        }
        Start-Sleep -Seconds 5
    } while ($watch.Elapsed.TotalSeconds -lt $TimeoutSeconds)
    throw 'GPU readiness timed out.'
}
function Set-UpdateImages($Images, [string]$Field) {
    foreach ($item in $Images) { Invoke-UpdateDocker -Arguments @('tag', [string]$item.$Field, [string]$item.image) }
}
function Restart-UpdateServices {
    # Stop both old services before starting the replacement GPU process.
    Invoke-UpdateDocker -Arguments ($script:Compose + @('stop', 'contour', 'engine'))
    Invoke-UpdateDocker -Arguments ($script:Compose + @('up', '-d', '--no-build', '--pull', 'never', '--force-recreate', '--no-deps', 'engine', 'contour'))
}
function Start-ContourUpdate {
    $source = (Resolve-Path -LiteralPath $PackageDir).Path
    $manifest = [IO.File]::ReadAllText((Join-Path $source 'update-manifest.json')) | ConvertFrom-Json
    if ($manifest.product -ne 'GoodVision Contour Code Update' -or $manifest.release -notmatch '^[a-z0-9-]+$' -or @($manifest.images).Count -ne 2) { throw 'Invalid update manifest.' }
    $names = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($item in $manifest.files) {
        $name = [string]$item.path
        if ($name -notmatch '^[a-zA-Z0-9_./-]+$' -or $name.StartsWith('/') -or ($name -split '/') -contains '..' -or -not $names.Add($name)) { throw 'Unsafe or duplicate update path.' }
        $path = Join-Path $source $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
            (Get-Item -LiteralPath $path).Length -ne $item.bytes -or
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.sha256) { throw "Update checksum mismatch: $name" }
    }
    $expected = @{engine='goodvision-contour-runtime:0.1.0';contour='goodvision-contour:0.1.0'}
    $roles = @()
    foreach ($item in $manifest.images) {
        if ($item.service -notin @('engine','contour') -or $item.service -in $roles -or
            $item.image -ne $expected[$item.service] -or $item.base_id -notmatch '^sha256:[0-9a-f]{64}$') { throw 'Unsupported image manifest.' }
        $roles += $item.service
    }
    $context = @((Invoke-UpdateDocker -Arguments @('context', 'inspect') -Capture | ConvertFrom-Json))[0]
    $endpoint = $context.Endpoints.docker.Host
    if ($env:DOCKER_HOST -and -not $env:DOCKER_CONTEXT) { $endpoint = $env:DOCKER_HOST }
    if ($endpoint -notmatch '^npipe:') { throw 'Use the local Docker Desktop context on the GPU PC.' }
    $info = Invoke-UpdateDocker -Arguments @('info', '--format', '{{json .}}') -Capture | ConvertFrom-Json
    if ($info.OSType -ne 'linux' -or $info.Architecture -notin @('amd64','x86_64')) { throw 'Linux amd64 Docker Desktop is required.' }
    $engine = Get-UpdateContainer 'engine'
    $gateway = Get-UpdateContainer 'contour'
    $node = [string]$engine.Config.Labels.'com.docker.compose.project.working_dir'
    $configs = [string]$engine.Config.Labels.'com.docker.compose.project.config_files'
    if (-not $node -or -not $configs -or $node -ne $gateway.Config.Labels.'com.docker.compose.project.working_dir' -or $configs -ne $gateway.Config.Labels.'com.docker.compose.project.config_files') { throw 'Cannot locate matching installed Compose configuration.' }
    $node = (Resolve-Path -LiteralPath $node).Path
    if (-not (Test-Path -LiteralPath (Join-Path $node '.env') -PathType Leaf)) { throw 'Installed .env is required to preserve GPU, ports and model configuration.' }
    $script:Compose = @('compose', '--project-name', $Project, '--project-directory', $node, '--env-file', (Join-Path $node '.env'))
    foreach ($file in ($configs -split ',')) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Compose file is missing: $file" }
        $script:Compose += @('-f', $file)
    }
    $savedEnv = @{}
    foreach ($entry in @(Get-ChildItem Env: | Where-Object { $_.Name -like 'CONTOUR_*' -or $_.Name -eq 'DOCKER_BUILDKIT' })) {
        $savedEnv[$entry.Name] = $entry.Value
        [Environment]::SetEnvironmentVariable($entry.Name, $null, 'Process')
    }
    $paused = $false
    $wasAccepting = $true
    try {
        # Use this Docker context's built-in builder, which shares local image layers.
        $env:DOCKER_BUILDKIT = '1'
        $builder = [string]$context.Name
        if (-not $builder) { throw 'Cannot identify the local Docker builder.' }
        $config = Invoke-UpdateDocker -Arguments ($script:Compose + @('config', '--format', 'json')) -Capture | ConvertFrom-Json
        foreach ($item in $manifest.images) {
            if ($config.services.($item.service).image -ne $item.image) { throw 'Compose image differs from the supported installation.' }
        }
        $stateDir = Join-Path $node '.contour-updates'
        $statePath = Join-Path $stateDir ($manifest.release + '.json')
        if ($Rollback) {
            if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) { throw 'No rollback record exists for this update.' }
            $state = [IO.File]::ReadAllText($statePath) | ConvertFrom-Json
            foreach ($item in $state.images) {
                $original = @($manifest.images | Where-Object service -eq $item.service)[0]
                if ($item.image -ne $original.image -or $item.old_id -ne $original.base_id) { throw 'Rollback record does not match this update.' }
                if ((Get-UpdateImage $item.old_id) -ne $item.old_id) { throw 'Rollback base image is missing.' }
                $current = (Get-UpdateContainer $item.service).Image
                if ($current -notin @($item.old_id, $item.new_id)) { throw 'A newer version is installed; refusing to overwrite it.' }
            }
        } else {
            $images = @()
            $alreadyInstalled = $true
            foreach ($item in $manifest.images) {
                $current = Get-UpdateContainer $item.service
                $newTag = ($item.image -split ':')[0] + ':' + $manifest.release
                $backupTag = ($item.image -split ':')[0] + ':before-' + $manifest.release
                $currentLabels = $current.Config.Labels
                $version = $currentLabels.PSObject.Properties['com.goodvision.code-update']
                if (-not $version -or $version.Value -ne $manifest.release) {
                    $alreadyInstalled = $false
                    if ($current.Image -ne $item.base_id -or (Get-UpdateImage $item.image) -ne $item.base_id) { throw 'Installed image is not the expected base. No services were changed.' }
                }
                $images += [pscustomobject]@{ service=$item.service; image=$item.image; old_id=$item.base_id; new_id=''; new_tag=$newTag; backup_tag=$backupTag }
            }
            if ($alreadyInstalled) {
                Wait-UpdateReady $true
                Write-Host 'This code update is already installed and ready.'
                return
            }
            if (@($images | Where-Object { (Get-UpdateContainer $_.service).Image -ne $_.old_id }).Count) { throw 'Partially applied update detected. Run with -Rollback first.' }
            $state = [pscustomobject]@{release=$manifest.release; project=$Project; accepting_sessions=$true; images=$images}
        }
        if ($CheckOnly) { Write-Host 'Package, installed images and saved configuration checks passed. No changes made.'; return }
        if (-not $Rollback) {
            foreach ($item in $state.images) {
                Invoke-UpdateDocker -Arguments @('build', '--builder', $builder, '--pull=false', '--network', 'none', '--provenance=false', '--build-arg', ('BASE_IMAGE=' + $item.image), '--label', ('com.goodvision.code-update=' + $manifest.release), '-f', (Join-Path $source ('Dockerfile.' + $item.service)), '-t', $item.new_tag, (Join-Path $source 'payload'))
                $item.new_id = Get-UpdateImage $item.new_tag
            }
        }
        if ($Rollback -and $engine.State.Status -in @('created', 'exited', 'dead')) {
            # Recovery must remain possible when the replacement cannot start.
            $wasAccepting = $state.accepting_sessions -eq $true
            $paused = $true
            Write-Host 'The engine is stopped; restoring the previous version.'
        } else {
            $status = Invoke-UpdateNode 'status'
            $wasAccepting = $status.accepting_sessions -eq $true
            $paused = $true
            $null = Invoke-UpdateNode 'pause'
            $status = Invoke-UpdateNode 'status'
            if ($null -eq $status.active_sessions -or $status.active_sessions -ne 0 -or $status.requests.in_flight -ne 0) { throw 'Analysis is still running. Stop analysis in the main service, then run the update again.' }
        }
        if (-not $Rollback) {
            $state.accepting_sessions = $wasAccepting
            [IO.Directory]::CreateDirectory($stateDir) | Out-Null
            foreach ($item in $state.images) { Invoke-UpdateDocker -Arguments @('tag', $item.old_id, $item.backup_tag) }
            [IO.File]::WriteAllText($statePath, ($state | ConvertTo-Json -Depth 8), $script:Utf8)
        }
        try {
            $field = if ($Rollback) { 'old_id' } else { 'new_id' }
            Set-UpdateImages $state.images $field
            Restart-UpdateServices
            Wait-UpdateReady (-not $Rollback)
        } catch {
            $failure = $_
            Write-Host 'Update failed. Restoring the previous images and service...'
            Set-UpdateImages $state.images 'old_id'
            Restart-UpdateServices
            Wait-UpdateReady $false
            throw "Update did not pass readiness; previous version restored. $failure"
        }
        if ($Rollback) { Write-Host 'Previous version restored.' }
        else {
            $memory = Invoke-UpdateNode 'memory'
            Write-Host ('Update ready. Tensor memory: {0:N0} MiB; unused cache: {1:N0} MiB; device free: {2:N0} MiB.' -f ($memory.allocated_bytes/1MB), ($memory.unused_reserved_bytes/1MB), ($memory.device_free_bytes/1MB))
            Write-Host 'The original address, key, camera capacity and model files are unchanged. Analysis remains stopped.'
        }
    } finally {
        if ($paused) {
            try { $null = Invoke-UpdateNode $(if ($wasAccepting) { 'resume' } else { 'pause' }) }
            catch { Write-Warning 'Could not restore admission state. Check the node console before starting analysis.' }
        }
        [Environment]::SetEnvironmentVariable('DOCKER_BUILDKIT', $null, 'Process')
        foreach ($name in $savedEnv.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnv[$name], 'Process') }
    }
}
if ($MyInvocation.InvocationName -ne '.') {
    try { Start-ContourUpdate } catch {
        # Preserve the original failing command and line instead of hiding them
        # behind the outer catch block's Write-Error location.
        Write-Host $_.InvocationInfo.PositionMessage
        Write-Error -ErrorRecord $_ -ErrorAction Continue
        exit 1
    }
}
