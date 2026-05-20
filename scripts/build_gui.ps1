<#
Build a folder-based Windows distribution for the PeakFit GUI using PyInstaller.

Usage: run from project root in PowerShell with appropriate execution policy.

The script creates/uses a virtualenv `.venv`, installs requirements and PyInstaller,
then runs PyInstaller to produce a folder in `dist\peakfit_gui` containing the
executable and required files (configs and assets).
#>

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root = Split-Path -Parent $scriptDir
Set-Location $root

$venv = Join-Path $root '.venv'
if (-not (Test-Path $venv)) {
    Write-Host "Creating virtual environment..."
    & py -3 -m venv $venv
}

$venvPython = Join-Path $venv 'Scripts\python.exe'
$venvPip = Join-Path $venv 'Scripts\pip.exe'

& $venvPip install --upgrade pip
& $venvPip install -r requirements.txt pyinstaller

# Fail early if the GUI runtime stack is not importable in the venv.
& $venvPython -c "import matplotlib; import matplotlib.backends.backend_qtagg; import PySide6; print('GUI deps OK')"
if ($LASTEXITCODE -ne 0) {
    throw "Required GUI dependencies are missing in the build venv (matplotlib/PySide6)."
}

# Add data for configs and assets
$cfg = Join-Path $root 'configs'
$assets = Join-Path $root 'assets'
$addDataCfg = "$cfg;configs"
if (Test-Path $assets) {
    $addDataAssets = "$assets;assets"
} else {
    $addDataAssets = ''
}

Write-Host "Running PyInstaller to build folder-based GUI (this may take a while)..."

$args = @(
    '--noconfirm',
    '--windowed',
    '--name', 'peakfit_gui',
    '--collect-all', 'matplotlib'
)
if ($addDataCfg) { $args += '--add-data'; $args += $addDataCfg }
if ($addDataAssets) { $args += '--add-data'; $args += $addDataAssets }
$args += '--hidden-import'; $args += 'lmfit'
$args += '--hidden-import'; $args += 'matplotlib'
$args += '--hidden-import'; $args += 'matplotlib.backends.backend_qtagg'
$args += '--hidden-import'; $args += 'matplotlib.backends.qt_compat'
$args += 'scripts\gui_app.py'

& $venvPython -m PyInstaller $args

Write-Host "Build finished. Folder: dist\peakfit_gui"
