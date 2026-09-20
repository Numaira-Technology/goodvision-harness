#Requires -Version 5.1
<#
.SYNOPSIS
Deploy GoodVision Contour on Windows using Docker Desktop (WSL 2).
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\deploy_contour.ps1 -PackageDir D:\Contour
#>
[CmdletBinding()]
param(
    [string]$PackageDir = $PSScriptRoot,
    [string]$InstallDir,
    [string]$ModelsDir,
    [string]$BindAddress = '127.0.0.1',
    [ValidateRange(1, 65535)][int]$Port = 8093,
    [string]$Gpu = '0',
    [ValidateRange(1, 2147483647)][int]$Capacity = 1,
    [switch]$Build,
    [ValidateSet('cu126', 'cu128')][string]$Cuda = 'cu128',
    [ValidateRange(1, 86400)][int]$TimeoutSeconds = 900,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:Utf8 = New-Object System.Text.UTF8Encoding($false)
$script:Images = @('goodvision-contour:0.1.0', 'goodvision-contour-runtime:0.1.0')

function Invoke-ContourDocker {
    param([string[]]$Arguments, [switch]$Capture, [switch]$Quiet, [switch]$AllowFailure)
    # Native stderr must not become a terminating PowerShell 5.1 error record.
    $ErrorActionPreference = 'Continue'
    if ($Quiet) {
        & docker @Arguments *> $null
        $resultCode = $LASTEXITCODE
    } elseif ($Capture) {
        $outputLines = @(& docker @Arguments)
        $resultCode = $LASTEXITCODE
    } else {
        & docker @Arguments | Out-Host
        $resultCode = $LASTEXITCODE
    }
    if ($resultCode -ne 0 -and -not $AllowFailure) {
        throw "Docker command failed (exit $resultCode): $($Arguments[0]). Check the message above."
    }
    if ($AllowFailure) { return $resultCode }
    if ($Capture) { return ($outputLines -join "`n") }
}

function Get-ContourSha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-ContourSidecar([string]$Path) {
    $sidecar = $Path + '.sha256'
    if (Test-Path -LiteralPath $sidecar -PathType Leaf) {
        Write-Host "Verifying archive: $([IO.Path]::GetFileName($Path))"
        $expected = (([IO.File]::ReadAllText($sidecar)).Trim() -split '\s+')[0]
        if ((Get-ContourSha256 $Path) -ne $expected) { throw "SHA-256 mismatch: $Path" }
    }
}

function Get-ContourReleaseRoot([string]$Path) {
    if (Test-Path -LiteralPath (Join-Path $Path 'workers\contour\compose.yaml')) { return $Path }
    if ((Split-Path $Path -Leaf) -eq 'contour' -and
        (Test-Path -LiteralPath (Join-Path $Path 'compose.yaml'))) {
        return (Split-Path (Split-Path $Path -Parent) -Parent)
    }
    return $null
}

function Expand-ContourPackage([string]$Source, [string]$Destination) {
    $archivePath = $null
    foreach ($name in @('goodvision-contour-with-model.zip', 'goodvision-contour-framework.zip')) {
        $candidate = Join-Path $Source $name
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $archivePath = $candidate; break }
    }
    if (-not $archivePath) { throw 'No release ZIP or extracted workers\contour directory found.' }
    Test-ContourSidecar $archivePath
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        $manifestEntry = $zip.GetEntry('release-manifest.json')
        if (-not $manifestEntry) { throw 'Missing release-manifest.json.' }
        $reader = New-Object IO.StreamReader($manifestEntry.Open())
        try { $manifestText = $reader.ReadToEnd() } finally { $reader.Dispose() }
        $manifest = $manifestText | ConvertFrom-Json
        if ($manifest.product -ne 'GoodVision Contour') { throw 'Not a GoodVision Contour package.' }
        $destinationRoot = [IO.Path]::GetFullPath($Destination).TrimEnd('\')
        $prefix = $destinationRoot + '\'
        $names = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
        $items = @()
        foreach ($item in $manifest.files) {
            $name = [string]$item.path
            if (-not $name.StartsWith('workers/') -or $name.Contains('\') -or
                $name.Contains(':') -or ($name -split '/') -contains '..' -or
                -not $names.Add($name)) { throw "Unsafe or duplicate ZIP path: $name" }
            $target = [IO.Path]::GetFullPath((Join-Path $destinationRoot $name))
            if (-not $target.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "ZIP path escapes installation directory: $name"
            }
            $entry = $zip.GetEntry($name)
            if (-not $entry -or $entry.Length -ne $item.bytes -or
                (($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000) {
                throw "Invalid release file: $name"
            }
            $items += [pscustomobject]@{ Entry = $entry; Target = $target; Hash = $item.sha256 }
        }
        if (Test-Path -LiteralPath $destinationRoot) {
            if (-not (Test-Path -LiteralPath (Join-Path $destinationRoot 'release-manifest.json'))) {
                throw 'Installation directory is incomplete. Choose a new -InstallDir.'
            }
            foreach ($item in $items) {
                if (-not (Test-Path -LiteralPath $item.Target -PathType Leaf) -or
                    (Get-ContourSha256 $item.Target) -ne $item.Hash) {
                    throw 'Existing release differs from ZIP. Choose a new -InstallDir; existing files were preserved.'
                }
            }
            Write-Host "Using existing installation: $destinationRoot"
            return $destinationRoot
        }
        [IO.Directory]::CreateDirectory($destinationRoot) | Out-Null
        Write-Host "Extracting release to: $destinationRoot"
        foreach ($item in $items) {
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($item.Target)) | Out-Null
            $inputStream = $item.Entry.Open()
            try {
                $outputStream = [IO.File]::Open($item.Target, [IO.FileMode]::CreateNew)
                try { $inputStream.CopyTo($outputStream, 4MB) } finally { $outputStream.Dispose() }
            } finally { $inputStream.Dispose() }
            if ((Get-ContourSha256 $item.Target) -ne $item.Hash) { throw "Extracted file failed verification: $($item.Target)" }
        }
        [IO.File]::WriteAllText((Join-Path $destinationRoot 'release-manifest.json'), $manifestText, $script:Utf8)
        return $destinationRoot
    } finally { $zip.Dispose() }
}

function Test-ContourModel([string]$Directory) {
    foreach ($name in @('contour.pt', 'model-manifest.json', 'MODEL_LICENSE.txt')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Directory $name) -PathType Leaf)) {
            throw "Missing model file: $name. Use the with-model ZIP, or set -ModelsDir."
        }
    }
    Write-Host 'Verifying model (3.5 GB)...'
    $model = Join-Path $Directory 'contour.pt'
    $origin = [IO.File]::ReadAllText((Join-Path $Directory 'model-manifest.json')) | ConvertFrom-Json
    $expected = '0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6'
    if ($origin.sha256 -ne $expected -or $origin.bytes -ne 3502755717 -or
        (Get-Item -LiteralPath $model).Length -ne 3502755717 -or
        (Get-ContourSha256 $model) -ne $expected) { throw 'Model integrity check failed.' }
}

function Initialize-ContourKey([string]$Node) {
    $directory = Join-Path $Node 'secrets'
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    $key = Join-Path $directory 'node-key'
    if (Test-Path -LiteralPath $key) {
        if (-not ([IO.File]::ReadAllText($key)).Trim()) { throw "Existing key is empty: $key" }
        return $key
    }
    $bytes = New-Object byte[] 36
    $random = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $random.GetBytes($bytes) } finally { $random.Dispose() }
    $value = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    $stream = [IO.File]::Open($key, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try {
        $encoded = $script:Utf8.GetBytes($value + "`n")
        $stream.Write($encoded, 0, $encoded.Length)
    } finally { $stream.Dispose() }
    return $key
}

function Write-ContourEnv([string]$Node, [Collections.IDictionary]$Values) {
    $lines = foreach ($name in $Values.Keys) {
        $value = [string]$Values[$name]
        if ($value.Contains("`n") -or $value.Contains("`r")) { throw 'Newlines are not allowed in configuration.' }
        $name + "='" + $value.Replace("'", "\'") + "'"
    }
    $content = ($lines -join "`n") + "`n"
    $path = Join-Path $Node '.env'
    if (Test-Path -LiteralPath $path) {
        if ([IO.File]::ReadAllText($path) -ne $content) {
            Copy-Item -LiteralPath $path -Destination ($path + '.before-deploy-' + [Guid]::NewGuid().ToString('N'))
        }
    }
    [IO.File]::WriteAllText($path, $content, $script:Utf8)
}

function ConvertTo-ContourPython([string]$Code) {
    $encoded = [Convert]::ToBase64String($script:Utf8.GetBytes($Code))
    return "exec(__import__('base64').b64decode('$encoded'))"
}

function Wait-ContourReady([string[]]$Compose, [int]$Seconds) {
    $probe = ConvertTo-ContourPython @'
import json, pathlib, urllib.request
key = pathlib.Path('/run/secrets/node_key').read_text().strip()
request = urllib.request.Request('http://127.0.0.1:8093/health/ready', headers={'Authorization': 'Bearer ' + key})
response = json.load(urllib.request.urlopen(request, timeout=10))
assert response.get('ready') is True and response.get('gpu_ready') is True
'@
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $nextReport = 0
    while ($watch.Elapsed.TotalSeconds -lt $Seconds) {
        $code = Invoke-ContourDocker -Arguments ($Compose + @('exec', '-T', 'contour', 'python', '-c', $probe)) -Quiet -AllowFailure
        if ($code -eq 0) { return }
        $container = Invoke-ContourDocker -Arguments ($Compose + @('ps', '-a', '-q', 'engine')) -Capture
        if ($container) {
            $state = (Invoke-ContourDocker -Arguments @('inspect', '--format', '{{json .State}}', $container.Trim()) -Capture) | ConvertFrom-Json
            if ($state.OOMKilled -or $state.Status -in @('exited', 'dead', 'restarting')) {
                throw 'Engine exited or restarted. Check engine logs and available GPU memory.'
            }
        }
        if ($watch.Elapsed.TotalSeconds -ge $nextReport) {
            Write-Host 'Waiting for model loading and GPU warmup...'
            $nextReport = $watch.Elapsed.TotalSeconds + 30
        }
        Start-Sleep -Seconds 5
    }
    throw 'Model readiness timed out. A running console does not mean inference is ready.'
}

function Start-ContourDeployment {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT -or
        $env:PROCESSOR_ARCHITECTURE -notin @('AMD64', 'x86')) { throw 'This script requires x64 Windows.' }
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Install and start Docker Desktop, with the WSL 2 backend and Linux containers enabled.'
    }
    if ($Gpu -notmatch '^(\d+|GPU-[0-9a-fA-F-]+)$') { throw 'Use one GPU index (such as 0) or GPU UUID.' }
    $address = [Net.IPAddress]::Parse($BindAddress)
    if ($address.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) { throw 'Use an IPv4 bind address.' }
    Invoke-ContourDocker -Arguments @('compose', 'version')
    $info = (Invoke-ContourDocker -Arguments @('info', '--format', '{{json .}}') -Capture) | ConvertFrom-Json
    if ($info.OSType -ne 'linux' -or $info.Architecture -notin @('x86_64', 'amd64')) {
        throw 'Docker must use Linux amd64 containers. Switch Docker Desktop to Linux containers.'
    }
    if ($info.KernelVersion -notmatch '(?i)wsl2') {
        throw 'Enable Docker Desktop Settings > General > Use the WSL 2 based engine, then restart Docker Desktop.'
    }
    $context = @((Invoke-ContourDocker -Arguments @('context', 'inspect') -Capture) | ConvertFrom-Json)[0]
    $endpoint = $context.Endpoints.docker.Host
    if ($env:DOCKER_HOST -and -not $env:DOCKER_CONTEXT) { $endpoint = $env:DOCKER_HOST }
    if ($endpoint -notmatch '^npipe:') { throw 'Select a local Docker Desktop context; remote Docker hosts are unsupported.' }
    $listener = New-Object Net.Sockets.TcpListener($address, 0)
    try { $listener.Start() } finally { $listener.Stop() }
    $source = (Resolve-Path -LiteralPath $PackageDir).Path
    $root = Get-ContourReleaseRoot $source
    if (-not $root) {
        $target = if ($InstallDir) { [IO.Path]::GetFullPath($InstallDir) } else { Join-Path $source 'goodvision-contour' }
        $root = Expand-ContourPackage $source $target
    }
    $node = Join-Path $root 'workers\contour'
    $models = if ($ModelsDir) { (Resolve-Path -LiteralPath $ModelsDir).Path } else { Join-Path $node 'models' }
    Test-ContourModel $models
    $values = [ordered]@{
        CONTOUR_BIND_ADDRESS = $BindAddress
        CONTOUR_PORT = [string]$Port
        CONTOUR_GPU_ID = $Gpu
        CONTOUR_CAPACITY = [string]$Capacity
        CONTOUR_MODELS_DIR = $models.Replace('\', '/')
        CONTOUR_TORCH_CUDA = $Cuda
    }
    $savedEnvironment = @{}
    foreach ($entry in @(Get-ChildItem Env: | Where-Object Name -like 'CONTOUR_*')) {
        $savedEnvironment[$entry.Name] = $entry.Value
        [Environment]::SetEnvironmentVariable($entry.Name, $null, 'Process')
    }
    foreach ($name in $values.Keys) { [Environment]::SetEnvironmentVariable($name, $values[$name], 'Process') }
    try {
        $compose = @('compose', '--project-name', 'goodvision-contour', '--project-directory', $node,
            '-f', (Join-Path $node 'compose.yaml'), '-f', (Join-Path $node 'compose.gpu.yaml'))
        $archive = Join-Path $source 'goodvision-contour-images.tar.gz'
        if ($Build) {
            Write-Host 'Building images online; model weights will not be downloaded.'
            Invoke-ContourDocker -Arguments ($compose + @('build'))
        } elseif (Test-Path -LiteralPath $archive -PathType Leaf) {
            Test-ContourSidecar $archive
            Write-Host 'Importing offline images...'
            Invoke-ContourDocker -Arguments @('load', '-i', $archive)
        }
        foreach ($image in $script:Images) {
            try {
                $metadata = @((Invoke-ContourDocker -Arguments @('image', 'inspect', $image) -Capture) | ConvertFrom-Json)[0]
            } catch {
                throw "Image unavailable: $image. Copy goodvision-contour-images.tar.gz into -PackageDir, or use -Build for an online build."
            }
            if ($metadata.Architecture -ne 'amd64' -or $metadata.Os -ne 'linux') { throw "Incompatible image: $image" }
        }
        $gpuProbe = ConvertTo-ContourPython @'
from pathlib import Path
import torch
with Path('/models/contour.pt').open('rb') as handle:
    assert handle.read(1), 'Empty model'
assert torch.cuda.is_available(), 'CUDA is unavailable. Check Windows NVIDIA driver and Docker Desktop WSL 2.'
assert torch.ones(64, device='cuda').sum().item() == 64
p = torch.cuda.get_device_properties(0)
print('GPU: %s; VRAM: %.1f GiB; CUDA: %s' % (p.name, p.total_memory / 1024**3, torch.version.cuda))
'@
        Write-Host 'Checking model mount and GPU access inside Docker...'
        Invoke-ContourDocker -Arguments @('run', '--rm', '--pull', 'never', '--network', 'none',
            '--gpus', ('device=' + $Gpu), '--volume', ($models + ':/models:ro'),
            '--entrypoint', 'python', $script:Images[1], '-c', $gpuProbe)
        if ($CheckOnly) { Write-Host 'Preflight passed. Application containers were not started.'; return }
        $key = Initialize-ContourKey $node
        Write-ContourEnv $node $values
        Invoke-ContourDocker -Arguments ($compose + @('config', '--quiet'))
        $ready = $false
        try {
            Invoke-ContourDocker -Arguments ($compose + @('up', '-d', '--no-build', '--pull', 'never'))
            Wait-ContourReady $compose $TimeoutSeconds
            $ready = $true
        } finally {
            if (-not $ready) {
                Write-Host 'Readiness failed. Stopping this deployment engine; preserving console and files.'
                Invoke-ContourDocker -Arguments ($compose + @('stop', '-t', '30', 'engine')) -AllowFailure | Out-Null
                Invoke-ContourDocker -Arguments ($compose + @('logs', '--tail', '60', 'engine')) -AllowFailure | Out-Null
            }
        }
        $displayAddress = if ($BindAddress -eq '0.0.0.0') { '127.0.0.1' } else { $BindAddress }
        Write-Host "`nReady: model loading and GPU warmup passed. Continuous video still needs acceptance testing."
        Write-Host "Console: http://${displayAddress}:$Port/"
        Write-Host "Access key file: $key"
        Write-Host "Manage services in: $node"
    } finally {
        foreach ($name in $values.Keys) { [Environment]::SetEnvironmentVariable($name, $null, 'Process') }
        foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process') }
    }
}

# Dot-sourcing exposes helpers for local tests without deploying anything.
if ($MyInvocation.InvocationName -ne '.') {
    try { Start-ContourDeployment; exit 0 }
    catch { Write-Host ("Deployment failed: " + $_.Exception.Message) -ForegroundColor Red; exit 1 }
}
