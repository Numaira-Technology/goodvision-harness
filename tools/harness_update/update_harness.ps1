#Requires -Version 5.1
[CmdletBinding()]
param([string]$PackageDir, [string]$Project, [switch]$CheckOnly)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Resolve after parameter binding (Windows PowerShell 5.1 compatibility).
if ([string]::IsNullOrWhiteSpace($PackageDir)) { $PackageDir = $PSScriptRoot }
if ([string]::IsNullOrWhiteSpace($PackageDir)) { throw 'Cannot locate update package.' }

function Invoke-UpdateDocker {
    param([string[]]$Arguments, [switch]$Capture)
    $ErrorActionPreference = 'Continue'
    if ($Capture) { $lines = @(& docker @Arguments) }
    else { & docker @Arguments | Out-Host }
    if ($LASTEXITCODE -ne 0) { throw "Docker $($Arguments[0]) failed (exit $LASTEXITCODE)." }
    if ($Capture) { return ($lines -join "`n") }
}

function Get-ContainerLabel($Container, [string]$Name) {
    $property = $Container.Config.Labels.PSObject.Properties[$Name]
    if ($property) { return [string]$property.Value }
    return ''
}

function Find-InstalledHarness($Manifest, [string]$StateDir) {
    $selectionPath = Join-Path $StateDir 'selection.json'
    if (Test-Path -LiteralPath (Join-Path $StateDir 'update-state.json')) {
        if (-not (Test-Path -LiteralPath $selectionPath)) { throw 'Update selection record is missing.' }
        $saved = Get-Content -LiteralPath $selectionPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($Project -and $Project -ne $saved.project) { throw 'Project differs from the recorded update.' }
        return $saved
    }
    $ids = @(((Invoke-UpdateDocker -Arguments @('ps', '-a', '-q', '--filter', 'label=com.docker.compose.project') -Capture) -split '\s+') | Where-Object { $_ })
    if (-not $ids.Count) { throw 'No installed GoodVision main application containers found.' }
    $containers = Invoke-UpdateDocker -Arguments (@('inspect') + $ids) -Capture | ConvertFrom-Json
    $lanes = @($containers | Where-Object {
        (Get-ContainerLabel $_ 'com.docker.compose.service') -match '^harness(-\d+)?$' -and
        (Get-ContainerLabel $_ 'goodvision.update.helper') -ne 'true' -and
        $_.Name -notmatch '-before-' -and
        (-not $Project -or (Get-ContainerLabel $_ 'com.docker.compose.project') -eq $Project)
    })
    $compatible = @($lanes | Where-Object { $_.Image -in $Manifest.accepted_base_ids })
    if (-not $compatible.Count) { throw 'No supported GoodVision main application version found. Contour-only machines do not need this package.' }
    $projects = @($compatible | ForEach-Object { Get-ContainerLabel $_ 'com.docker.compose.project' } | Select-Object -Unique)
    if ($projects.Count -ne 1) { throw ('Multiple installations found. Run with -Project and one of: ' + ($projects -join ', ')) }
    $selected = @($lanes | Where-Object { (Get-ContainerLabel $_ 'com.docker.compose.project') -eq $projects[0] })
    $bases = @($selected | ForEach-Object { $_.Image } | Select-Object -Unique)
    if ($bases.Count -ne 1 -or $bases[0] -notin $Manifest.accepted_base_ids) { throw 'The selected installation has mixed or unsupported versions. No containers were changed.' }
    $installation = ''
    foreach ($container in $selected) {
        $directory = Get-ContainerLabel $container 'com.docker.compose.project.working_dir'
        if ($directory -and (Test-Path -LiteralPath (Join-Path $directory 'RELEASE.json'))) {
            $resolved = (Resolve-Path -LiteralPath $directory).Path
            if ($installation -and $installation -ne $resolved) { throw 'Camera installation folders do not match.' }
            $installation = $resolved
        }
    }
    if ($bases[0] -in $Manifest.windows_base_ids -and -not $installation) {
        throw 'Cannot locate the original Windows installation folder from Compose labels.'
    }
    $selection = [ordered]@{ project=$projects[0]; base_id=$bases[0]; installation=$installation }
    [IO.File]::WriteAllText($selectionPath, ($selection | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))
    return [pscustomobject]$selection
}

function Start-HarnessUpdate {
    $root = (Resolve-Path -LiteralPath $PackageDir).Path
    $manifest = Get-Content -LiteralPath (Join-Path $root 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.product -ne 'GoodVision Harness Code Update') { throw 'Wrong update package.' }
    foreach ($item in $manifest.files) {
        $file = [IO.Path]::GetFullPath((Join-Path $root $item.path))
        if (-not $file.StartsWith($root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid package path.' }
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $($item.path)" }
        if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.sha256) { throw "Package checksum failed: $($item.path)" }
    }
    $context = @((Invoke-UpdateDocker -Arguments @('context', 'inspect') -Capture | ConvertFrom-Json))[0]
    if ($context.Endpoints.docker.Host -notmatch '^(npipe|unix)://') { throw 'Run this package on the Docker machine itself.' }
    if ((Invoke-UpdateDocker -Arguments @('info', '--format', '{{.OSType}}') -Capture).Trim() -ne 'linux') { throw 'Docker Desktop must use Linux containers.' }
    $stateDir = Join-Path $root 'update-state'
    [IO.Directory]::CreateDirectory($stateDir) | Out-Null
    $selected = Find-InstalledHarness $manifest $stateDir
    $baseId = (Invoke-UpdateDocker -Arguments @('image', 'inspect', '--format', '{{.Id}}', $selected.base_id) -Capture).Trim()
    if ($baseId -ne $selected.base_id) { throw 'The selected installed image is unavailable.' }
    Write-Host "Selected installation: $($selected.project)"
    $runner = @('run', '--rm', '--user', '0:0', '--network', 'none', '--label', 'goodvision.update.helper=true', '--entrypoint', 'python',
        '-v', '/var/run/docker.sock:/var/run/docker.sock', '-v', "${root}:/update:ro", '-v', "${stateDir}:/state")
    if ($selected.installation) { $runner += @('-v', "$($selected.installation):/installation") }
    $runner += @($baseId, '/update/apply_update.py', '--project', [string]$selected.project, '--base-id', $baseId)
    if ($selected.installation) { $runner += @('--installation', '/installation') }
    Invoke-UpdateDocker -Arguments ($runner + @('--check-only'))
    if ($CheckOnly) { return }
    $baseTag = 'goodvision-harness-update-base:' + $manifest.release
    Invoke-UpdateDocker -Arguments @('tag', $baseId, $baseTag)
    $previousBuildkit = $env:DOCKER_BUILDKIT
    try {
        $env:DOCKER_BUILDKIT = '1'
        Invoke-UpdateDocker -Arguments @('build', '--builder', [string]$context.Name, '--pull=false', '--network', 'none', '--provenance=false',
            '--build-arg', "BASE_IMAGE=$baseTag", '-t', [string]$manifest.image, '-f', (Join-Path $root 'Dockerfile'), (Join-Path $root 'payload'))
    } finally { $env:DOCKER_BUILDKIT = $previousBuildkit }
    Invoke-UpdateDocker -Arguments $runner
}
try { Start-HarnessUpdate } catch {
    if ($_.InvocationInfo.PositionMessage) { Write-Host $_.InvocationInfo.PositionMessage }
    Write-Error -ErrorRecord $_ -ErrorAction Continue
    exit 1
}
