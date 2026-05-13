<#
Create a portable ZIP bundle containing the built executable and supporting files.

Usage (PowerShell):
    .\scripts\create_portable.ps1

This will produce `dist\peakfit_portable.zip` containing:
 - dist\peakfit.exe
 - configs\ (default_params.json)
 - README.md
 - an examples folder with the sample data (if present)
#>

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root = Split-Path -Parent $scriptDir
Set-Location $root

$distExe = Join-Path $root 'dist\\peakfit.exe'
if (-not (Test-Path $distExe)) {
    Write-Error "Executable not found at $distExe. Build it first with build_windows.ps1"
    exit 1
}

$portableDir = Join-Path $root 'portable_build'
if (Test-Path $portableDir) { Remove-Item -Recurse -Force $portableDir }
New-Item -ItemType Directory -Path $portableDir | Out-Null

Copy-Item $distExe -Destination $portableDir

# Copy configs
if (Test-Path (Join-Path $root 'configs')) {
    Copy-Item (Join-Path $root 'configs') -Destination $portableDir -Recurse
}

# Copy README
if (Test-Path (Join-Path $root 'README.md')) {
    Copy-Item (Join-Path $root 'README.md') -Destination $portableDir
}

# Copy example data if present
if (Test-Path (Join-Path $root '10nm_5L_peak.txt')) {
    New-Item -ItemType Directory -Path (Join-Path $portableDir 'examples') | Out-Null
    Copy-Item (Join-Path $root '10nm_5L_peak.txt') -Destination (Join-Path $portableDir 'examples')
}

$zipPath = Join-Path $root 'dist\\peakfit_portable.zip'
if (Test-Path $zipPath) { Remove-Item $zipPath }
Compress-Archive -Path (Join-Path $portableDir '*') -DestinationPath $zipPath -Force
Remove-Item -Recurse -Force $portableDir
Write-Host "Created portable ZIP: $zipPath"
