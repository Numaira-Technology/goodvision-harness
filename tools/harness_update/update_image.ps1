#Requires -Version 5.1
[CmdletBinding()]
param([string]$PackageDir, [string]$Project, [string]$Containers, [switch]$CheckOnly)
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
        if ($Containers -and (-not $saved.PSObject.Properties['containers'] -or $Containers -ne $saved.containers)) { throw 'Containers differ from the recorded update.' }
        return $saved
    }
    $ids = @(((Invoke-UpdateDocker -Arguments @('ps', '-a', '-q', '--filter', 'label=com.docker.compose.project') -Capture) -split '\s+') | Where-Object { $_ })
    if (-not $ids.Count) { throw 'No installed GoodVision main application containers found.' }
    $installedContainers = Invoke-UpdateDocker -Arguments (@('inspect') + $ids) -Capture | ConvertFrom-Json
    $requestedNames = @()
    if ($Containers) {
        $requestedNames = @($Containers -split ',' | ForEach-Object { $_.Trim().TrimStart('/') })
        if (@($requestedNames | Where-Object { $_ -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_.-]*$' }).Count -or @($requestedNames | Select-Object -Unique).Count -ne $requestedNames.Count) { throw 'Invalid or duplicate container names.' }
    }
    $lanes = @($installedContainers | Where-Object {
        (Get-ContainerLabel $_ 'com.docker.compose.service') -match '^harness(-\d+)?$' -and
        (Get-ContainerLabel $_ 'goodvision.update.helper') -ne 'true' -and
        $_.Name -notmatch '-before-' -and
        (-not $requestedNames.Count -or $_.Name.TrimStart('/') -in $requestedNames) -and
        (-not $Project -or (Get-ContainerLabel $_ 'com.docker.compose.project') -eq $Project)
    })
    if ($requestedNames.Count -and $lanes.Count -ne $requestedNames.Count) { throw 'One or more selected application containers were not found.' }
    if (-not $lanes.Count) { throw 'No GoodVision main application containers found. Contour-only machines do not need this package.' }
    $projects = @($lanes | ForEach-Object { Get-ContainerLabel $_ 'com.docker.compose.project' } | Select-Object -Unique)
    if ($projects.Count -ne 1) { throw ('Multiple installations found. Run with -Project and one of: ' + ($projects -join ', ')) }
    $selected = @($lanes | Where-Object { (Get-ContainerLabel $_ 'com.docker.compose.project') -eq $projects[0] })
    $bases = @($selected | ForEach-Object { $_.Image } | Select-Object -Unique)
    if ($bases.Count -ne 1) { throw 'The selected installation has mixed versions. No containers were changed.' }
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
    $selection = [ordered]@{ project=$projects[0]; base_id=$bases[0]; installation=$installation; containers=($requestedNames -join ',') }
    [IO.File]::WriteAllText($selectionPath, ($selection | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))
    return [pscustomobject]$selection
}

function Get-VerificationBase($Manifest, [string]$BaseId) {
    if ($BaseId -in $Manifest.accepted_base_ids) { return $BaseId }
    $current = @((Invoke-UpdateDocker -Arguments @('image', 'inspect', $BaseId) -Capture | ConvertFrom-Json))[0]
    $release = Get-ContainerLabel $current 'goodvision.ui.release'
    if (-not $release -or -not $Manifest.published_sources.PSObject.Properties[$release]) {
        throw "Unsupported installed main application release '$release' ($BaseId). No containers were changed."
    }
    $available = @((Invoke-UpdateDocker -Arguments @('image', 'ls', '-aq', '--no-trunc') -Capture) -split '\s+' | Where-Object { $_ -in $Manifest.accepted_base_ids } | Select-Object -Unique)
    if ($available.Count) {
        # Windows PowerShell 5.1 emits a JSON array as one pipeline object.
        # Wrapping that pipeline in @() nests the array, so foreach would
        # compare all images' layers together instead of one image at a time.
        $images = Invoke-UpdateDocker -Arguments (@('image', 'inspect') + $available) -Capture | ConvertFrom-Json
        foreach ($candidate in $images) {
            $layers = @($candidate.RootFS.Layers)
            if ($layers.Count -and (($current.RootFS.Layers | Select-Object -First $layers.Count) -join ',') -eq ($layers -join ',')) {
                return [string]$candidate.Id
            }
        }
    }
    throw 'The main application was found, but its verified original base image is unavailable. Keep or restore the original delivery image; no containers were changed.'
}

function Start-HarnessUpdate {
    $root = (Resolve-Path -LiteralPath $PackageDir).Path
    $image = 'goodvision-harness:20260923-image1'
    $archive = Join-Path $root 'application.tar'
    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) { throw 'Missing application.tar. Extract the complete ZIP first.' }
    Invoke-UpdateDocker -Arguments @('load', '-i', $archive)
    $manifest = Invoke-UpdateDocker -Arguments @('run', '--rm', '--network', 'none', '--entrypoint', 'cat', $image, '/opt/goodvision/update/manifest.json') -Capture | ConvertFrom-Json
    if ($manifest.product -ne 'GoodVision Harness Image Update' -or $manifest.image -ne $image) { throw 'Wrong update image.' }
    $context = @((Invoke-UpdateDocker -Arguments @('context', 'inspect') -Capture | ConvertFrom-Json))[0]
    if ($context.Endpoints.docker.Host -notmatch '^(npipe|unix)://') { throw 'Run this package on the Docker machine itself.' }
    if ((Invoke-UpdateDocker -Arguments @('info', '--format', '{{.OSType}}') -Capture).Trim() -ne 'linux') { throw 'Docker Desktop must use Linux containers.' }
    $stateDir = Join-Path $root 'update-state'
    [IO.Directory]::CreateDirectory($stateDir) | Out-Null
    $selected = Find-InstalledHarness $manifest $stateDir
    $baseId = (Invoke-UpdateDocker -Arguments @('image', 'inspect', '--format', '{{.Id}}', $selected.base_id) -Capture).Trim()
    if ($baseId -ne $selected.base_id) { throw 'The selected installed image is unavailable.' }
    $verificationBase = Get-VerificationBase $manifest $baseId
    if ($verificationBase -in $manifest.windows_base_ids -and -not $selected.installation) {
        throw 'Cannot locate the original Windows installation folder from Compose labels.'
    }
    Write-Host "Selected installation: $($selected.project)"
    $runner = @('run', '--rm', '--user', '0:0', '--network', 'none', '--label', 'goodvision.update.helper=true', '--entrypoint', 'python',
        '-v', '/var/run/docker.sock:/var/run/docker.sock', '-v', "${stateDir}:/state")
    if ($selected.installation) { $runner += @('-v', "$($selected.installation):/installation") }
    $runner += @($image, '/opt/goodvision/update/apply_update.py', '--manifest', '/opt/goodvision/update/manifest.json', '--project', [string]$selected.project, '--base-id', $baseId, '--verification-base-id', $verificationBase)
    if ($selected.PSObject.Properties['containers'] -and $selected.containers) { $runner += @('--containers', [string]$selected.containers) }
    if ($selected.installation) { $runner += @('--installation', '/installation') }
    Invoke-UpdateDocker -Arguments ($runner + @('--check-only'))
    if ($CheckOnly) { return }
    $baseTag = 'goodvision-harness-update-base:' + $manifest.release
    Invoke-UpdateDocker -Arguments @('tag', $baseId, $baseTag)
    Invoke-UpdateDocker -Arguments $runner
}
try { Start-HarnessUpdate } catch {
    if ($_.InvocationInfo.PositionMessage) { Write-Host $_.InvocationInfo.PositionMessage }
    Write-Error -ErrorRecord $_ -ErrorAction Continue
    exit 1
}
