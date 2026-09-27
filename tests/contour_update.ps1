#Requires -Version 5.1
param([string]$Archive = (Join-Path (Split-Path $PSScriptRoot -Parent) 'dist/goodvision-contour-update-20260920-memory1.zip'))
$ErrorActionPreference = 'Stop'
$repository = Split-Path $PSScriptRoot -Parent
. (Join-Path $repository 'workers/contour/update_contour.ps1')
Add-Type -AssemblyName System.IO.Compression.FileSystem
$testRoot = Join-Path $repository ('tmp/update-tests-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($testRoot) | Out-Null
[IO.Compression.ZipFile]::ExtractToDirectory($Archive, $testRoot)
$PackageDir = (Get-ChildItem -LiteralPath $testRoot -Directory | Select-Object -First 1).FullName
$script:manifest = [IO.File]::ReadAllText((Join-Path $PackageDir 'update-manifest.json')) | ConvertFrom-Json
function Assert($Value, [string]$Note) { if (-not $Value) { throw "Assertion failed: $Note" } }
function Reset-Fixture {
    $script:node = Join-Path $testRoot ('installed with spaces ' + [Guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($script:node) | Out-Null
    foreach ($name in @('.env', 'compose.yaml', 'compose.gpu.yaml', 'key', 'model')) { [IO.File]::WriteAllText((Join-Path $script:node $name), 'preserve this file') }
    $script:tags = @{}
    $script:containers = @{}
    foreach ($item in $script:manifest.images) {
        $script:tags[$item.image] = $item.base_id
        $script:tags[$item.base_id] = $item.base_id
        $script:containers[$item.service] = [pscustomobject]@{
            Image=$item.base_id; Config=[pscustomobject]@{Labels=[pscustomobject]@{
                'com.docker.compose.project.working_dir'=$script:node
                'com.docker.compose.project.config_files'=((Join-Path $script:node 'compose.yaml') + ',' + (Join-Path $script:node 'compose.gpu.yaml'))
            }}; State=[pscustomobject]@{Status='running';OOMKilled=$false}
        }
    }
    $script:commands = New-Object 'Collections.Generic.List[object]'
    $script:accepting = $true
    $script:active = 0
    $script:failWarmup = $false
    $script:failBuild = $false
    $script:waits = New-Object 'Collections.Generic.List[bool]'
}
function Get-UpdateContainer([string]$Service) { return $script:containers[$Service] }
function Get-UpdateImage([string]$Name) { if (-not $script:tags.ContainsKey($Name)) { throw "Missing image $Name" }; return $script:tags[$Name] }
function Invoke-UpdateDocker {
    param([string[]]$Arguments, [switch]$Capture, [switch]$Quiet, [switch]$AllowFailure)
    $script:commands.Add($Arguments)
    if ($Arguments[0] -eq 'context') { return '[{"Name":"desktop-linux","Endpoints":{"docker":{"Host":"npipe:local"}}}]' }
    if ($Arguments[0] -eq 'info') { return '{"OSType":"linux","Architecture":"amd64"}' }
    if ($Arguments -contains 'config') { return '{"services":{"engine":{"image":"goodvision-contour-runtime:0.1.0"},"contour":{"image":"goodvision-contour:0.1.0"}}}' }
    if ($Arguments[0] -eq 'build') {
        if ($script:failBuild) { throw 'simulated build failure' }
        Assert ($Arguments -contains '--pull=false') 'no image pulls'
        Assert ($Arguments -contains 'none') 'no build networking'
        $tag = $Arguments[[Array]::IndexOf($Arguments, '-t') + 1]
        $script:tags[$tag] = 'sha256:new-' + $tag
        return
    }
    if ($Arguments[0] -eq 'tag') { $script:tags[$Arguments[2]] = $Arguments[1]; return }
    if ($Arguments -contains 'up') {
        Assert ($Arguments -contains '--no-build') 'no dependency rebuilding'
        Assert ($Arguments -contains 'never') 'no pulls on service replacement'
        foreach ($item in $script:manifest.images) {
            $script:containers[$item.service].Image = $script:tags[$item.image]
            $version = if ($script:tags[$item.image] -eq $item.base_id) { '' } else { $script:manifest.release }
            $script:containers[$item.service].Config.Labels | Add-Member NoteProperty 'com.goodvision.code-update' $version -Force
        }
        return
    }
    if ($Arguments -contains 'stop') { return }
    throw ('Unexpected Docker call: ' + ($Arguments -join ' '))
}
function Invoke-UpdateNode([string]$Operation) {
    switch ($Operation) {
        'pause' { $script:accepting = $false }
        'resume' { $script:accepting = $true }
        'memory' { return [pscustomobject]@{available=$true;reclaim_count=1;error_code=$null;allocated_bytes=100;unused_reserved_bytes=0;device_free_bytes=900} }
    }
    return [pscustomobject]@{gpu_ready=$true;accepting_sessions=$script:accepting;active_sessions=$script:active;requests=[pscustomobject]@{in_flight=0}}
}
function Wait-UpdateReady([bool]$RequireMemory) {
    $script:waits.Add($RequireMemory)
    if ($RequireMemory -and $script:failWarmup) { throw 'simulated warmup failure' }
}
function Assert-Preserved {
    foreach ($name in @('.env', 'compose.yaml', 'compose.gpu.yaml', 'key', 'model')) { Assert ([IO.File]::ReadAllText((Join-Path $script:node $name)) -eq 'preserve this file') 'original deployment files preserved' }
}
$env:CONTOUR_PORT = 'test-ambient-value'
$env:DOCKER_BUILDKIT = 'test-ambient-builder'
Reset-Fixture
$CheckOnly = $true
Start-ContourUpdate
Assert (-not @($script:commands | Where-Object { $_[0] -eq 'build' -or $_[0] -eq 'tag' -or $_ -contains 'stop' }).Count) 'check-only has no mutation'
$CheckOnly = $false
Start-ContourUpdate
Assert ($script:waits.Count -eq 1 -and $script:waits[0]) 'new memory endpoint validated'
Assert $script:accepting 'original accepting state restored'
Assert-Preserved
Assert ($env:CONTOUR_PORT -eq 'test-ambient-value' -and $env:DOCKER_BUILDKIT -eq 'test-ambient-builder') 'caller environment restored'
$builds = @($script:commands | Where-Object { $_[0] -eq 'build' }).Count
Start-ContourUpdate
Assert (@($script:commands | Where-Object { $_[0] -eq 'build' }).Count -eq $builds) 'repeat does not rebuild'
$script:containers.engine.State.Status = 'exited'
$Rollback = $true
Start-ContourUpdate
$Rollback = $false
foreach ($item in $script:manifest.images) { Assert ($script:containers[$item.service].Image -eq $item.base_id) 'manual rollback restores base' }
Assert-Preserved
Reset-Fixture
$script:failWarmup = $true
$failed = $false
try { Start-ContourUpdate } catch { $failed = $_.ToString().Contains('previous version restored') }
Assert $failed 'warmup failure is reported after rollback'
foreach ($item in $script:manifest.images) { Assert ($script:containers[$item.service].Image -eq $item.base_id) 'failed warmup rolls back both images' }
Assert ($script:waits.Count -eq 2 -and -not $script:waits[1]) 'rollback uses compatible original health check'
Assert-Preserved
Reset-Fixture
$script:active = 1
$failed = $false
try { Start-ContourUpdate } catch { $failed = $_.ToString().Contains('Analysis is still running') }
Assert $failed 'live sessions reject update'
Assert $script:accepting 'busy rejection resumes original admission'
Assert (-not @($script:commands | Where-Object { $_[0] -eq 'tag' -or $_ -contains 'stop' }).Count) 'busy update never stops or retags services'
Reset-Fixture
$script:failBuild = $true
try { Start-ContourUpdate } catch { }
Assert (-not @($script:commands | Where-Object { $_[0] -eq 'tag' -or $_ -contains 'stop' }).Count) 'build failure leaves original running'
Reset-Fixture
$script:containers.engine.Image = 'sha256:unexpected'
$failed = $false
try { Start-ContourUpdate } catch { $failed = $_.ToString().Contains('not the expected base') }
Assert $failed 'wrong base rejected before build'
Assert (-not @($script:commands | Where-Object { $_[0] -eq 'build' }).Count) 'wrong base never builds'
Reset-Fixture
[IO.File]::AppendAllText((Join-Path $PackageDir 'payload/engine/app.py'), 'corrupt')
$failed = $false
try { Start-ContourUpdate } catch { $failed = $_.ToString().Contains('checksum mismatch') }
Assert $failed 'corrupt payload rejected before Docker'
Assert ($script:commands.Count -eq 0) 'corrupt package never calls Docker'
[Environment]::SetEnvironmentVariable('CONTOUR_PORT', $null, 'Process')
[Environment]::SetEnvironmentVariable('DOCKER_BUILDKIT', $null, 'Process')
Write-Host 'PASS: update, idempotence, manual/automatic rollback, active-session guard, build failure, wrong base, checksums, config preservation.'
