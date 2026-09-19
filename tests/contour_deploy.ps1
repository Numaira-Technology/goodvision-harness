#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
$repository = Split-Path $PSScriptRoot -Parent
. (Join-Path $repository 'workers\contour\deploy_contour.ps1')
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$testRoot = Join-Path $repository ('tmp\contour-powershell-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($testRoot) | Out-Null

function Assert-True($Value, [string]$Message) {
    if (-not $Value) { throw "Assertion failed: $Message" }
}

function New-TestBundle([string]$Directory, [string]$EntryName) {
    [IO.Directory]::CreateDirectory($Directory) | Out-Null
    $body = $script:Utf8.GetBytes('services: {}')
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $hash = ([BitConverter]::ToString($sha.ComputeHash($body))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
    $manifest = @{ product = 'GoodVision Contour'; files = @(@{path=$EntryName; bytes=$body.Length; sha256=$hash}) }
    $zip = [IO.Compression.ZipFile]::Open((Join-Path $Directory 'goodvision-contour-framework.zip'), [IO.Compression.ZipArchiveMode]::Create)
    try {
        $stream = ($zip.CreateEntry($EntryName)).Open()
        try { $stream.Write($body, 0, $body.Length) } finally { $stream.Dispose() }
        $stream = ($zip.CreateEntry('release-manifest.json')).Open()
        $json = $script:Utf8.GetBytes(($manifest | ConvertTo-Json -Depth 10))
        try { $stream.Write($json, 0, $json.Length) } finally { $stream.Dispose() }
    } finally { $zip.Dispose() }
}

$source = Join-Path $testRoot 'package with spaces'
New-TestBundle $source 'workers/contour/compose.yaml'
$root = Expand-ContourPackage $source (Join-Path $source 'installed')
$node = Join-Path $root 'workers\contour'
$keyPath = Initialize-ContourKey $node
$key = [IO.File]::ReadAllText($keyPath)
Assert-True ($key.Trim() -match '^[A-Za-z0-9_-]{48}$') 'cryptographic key format'
Assert-True (([IO.File]::ReadAllBytes($keyPath))[0] -ne 239) 'key has no UTF-8 BOM'
$again = Expand-ContourPackage $source $root
$null = Initialize-ContourKey $node
Assert-True ([IO.File]::ReadAllText($keyPath) -eq $key) 'repeated deploy preserves key'
Write-ContourEnv $node ([ordered]@{ CONTOUR_PORT='8093'; CONTOUR_MODELS_DIR='D:/Model Files' })
Write-ContourEnv $node ([ordered]@{ CONTOUR_PORT='18093'; CONTOUR_MODELS_DIR='D:/Model Files' })
Assert-True (@(Get-ChildItem -LiteralPath $node -Force -Filter '.env.before-deploy-*').Count -eq 1) 'config backup'
Assert-True (([IO.File]::ReadAllBytes((Join-Path $node '.env')))[0] -ne 239) 'env has no UTF-8 BOM'
[IO.File]::WriteAllText((Join-Path $node 'compose.yaml'), 'changed')
$rejected = $false
try { $null = Expand-ContourPackage $source $root } catch { $rejected = $true }
Assert-True $rejected 'never overwrite modified installation'
foreach ($unsafePath in @('workers/../../escaped', 'workers/contour/file:stream', '/absolute')) {
    $source = Join-Path $testRoot ([Guid]::NewGuid().ToString('N'))
    New-TestBundle $source $unsafePath
    $rejected = $false
    try { $null = Expand-ContourPackage $source (Join-Path $source 'installed') } catch { $rejected = $true }
    Assert-True $rejected 'unsafe ZIP must be rejected'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $source 'installed'))) 'reject before creating installation'
}

$script:ProbeCount = 0
function Invoke-ContourDocker {
    param([string[]]$Arguments, [switch]$Capture, [switch]$Quiet, [switch]$AllowFailure)
    if ($Arguments -contains 'exec') { $script:ProbeCount++; if ($script:ProbeCount -gt 1) { return 0 }; return 1 }
    if ($Arguments -contains 'inspect') { return '{"Status":"running","OOMKilled":false}' }
    return 'test-container'
}
function Start-Sleep { param($Seconds) }
Wait-ContourReady @('compose') 15
Assert-True ($script:ProbeCount -eq 2) 'must wait for authenticated readiness'
function Invoke-ContourDocker {
    param([string[]]$Arguments, [switch]$Capture, [switch]$Quiet, [switch]$AllowFailure)
    if ($Arguments -contains 'exec') { return 1 }
    if ($Arguments -contains 'inspect') { return '{"Status":"restarting","OOMKilled":true}' }
    return 'test-container'
}
$rejected = $false
try { Wait-ContourReady @('compose') 15 } catch { $rejected = $true }
Assert-True $rejected 'restarting engine cannot be reported as ready'
Write-Host 'PASS: Windows extraction, path validation, key preservation, env backup, readiness and engine failure.'
