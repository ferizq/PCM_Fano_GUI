param(
    [ValidateSet('Release', 'Debug')]
    [string]$BuildType = 'Release'
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root 'build\c_core'
$outDll = Join-Path $root 'build\peakfit_core.dll'

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw 'cmake is required to build the native core. Install CMake and a C compiler first.'
}

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

Write-Host "Configuring C build in $buildDir"
cmake -S $root -B $buildDir -DCMAKE_BUILD_TYPE=$BuildType

Write-Host "Building peakfit_core ($BuildType)"
cmake --build $buildDir --config $BuildType

$candidates = @(
    (Join-Path $buildDir "$BuildType\peakfit_core.dll"),
    (Join-Path $buildDir 'peakfit_core.dll'),
    (Join-Path $buildDir "$BuildType\libpeakfit_core.dll"),
    (Join-Path $buildDir 'libpeakfit_core.dll')
)

$dllPath = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $dllPath) {
    throw 'Build succeeded but the peakfit_core DLL was not found in expected output paths.'
}

Copy-Item -Path $dllPath -Destination $outDll -Force
Write-Host "Native core copied to $outDll"
