<#
Build a standalone Windows executable using PyInstaller.

Usage (from project root in PowerShell):

    powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1

What it does:
- creates a local virtual environment `.venv` (if missing)
- installs project requirements and PyInstaller into the venv
- runs PyInstaller to produce a single-file executable at `dist\peakfit_cli.exe`

The produced executable can be distributed to Windows users without installing Python
or the project's Python libraries.
#>

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root = Split-Path -Parent $scriptDir
Set-Location $root

$venv = Join-Path $root '.venv'
if (Test-Path $venv) {
    $venvPipCheck = Join-Path $venv 'Scripts\\pip.exe'
    if (-not (Test-Path $venvPipCheck)) {
        Write-Host "Found existing incomplete .venv; removing and recreating..."
        Remove-Item -Recurse -Force $venv
    }
}

if (-not (Test-Path $venv)) {
    $pythonCmdInfo = Get-Command python -ErrorAction SilentlyContinue
    $pythonOk = $false
    if ($pythonCmdInfo) {
        try {
            & python -c "import sys; sys.exit(0)" > $null 2>&1
            if ($LASTEXITCODE -eq 0) { $pythonOk = $true }
        } catch { }
    }

    $pyCmdInfo = Get-Command py -ErrorAction SilentlyContinue
    $pyOk = $false
    if (-not $pythonOk -and $pyCmdInfo) {
        try {
            & py -3 -c "import sys; sys.exit(0)" > $null 2>&1
            if ($LASTEXITCODE -eq 0) { $pyOk = $true }
        } catch { }
    }

    if ($pythonOk) {
        Write-Host "Creating venv with 'python'"
        & python -m venv $venv
    } elseif ($pyOk) {
        Write-Host "Creating venv with 'py -3'"
        & py -3 -m venv $venv
    } else {
        Write-Error "Python not found or not functional. Install Python or ensure 'python' or 'py' are available on PATH."
        exit 1
    }
}

$venvPython = Join-Path $venv 'Scripts\\python.exe'
$venvPip = Join-Path $venv 'Scripts\\pip.exe'

& $venvPip install --upgrade pip
& $venvPip install -r requirements.txt pyinstaller

# PyInstaller --add-data expects 'SRC;DEST' on Windows. Use absolute path for SRC.
$cfg = Join-Path $root 'configs\\default_params.json'
$addData = "$cfg;configs"

Write-Host "Running PyInstaller (this may take a minute)..."
# build a windowed (no-console) executable so it can be double-clicked from Explorer
& $venvPython -m PyInstaller --onefile --noconsole --name peakfit --add-data $addData --hidden-import lmfit --hidden-import plotly --hidden-import pkg_resources.py2_warn scripts\fit_peak.py

Write-Host "Build finished. Executable: dist\\peakfit.exe"

# Create a Desktop shortcut to the executable
try {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $exePath = Join-Path $root 'dist\\peakfit.exe'
    if (Test-Path $exePath) {
        $WshShell = New-Object -ComObject WScript.Shell
        $shortcut = $WshShell.CreateShortcut((Join-Path $desktop 'peakfit.lnk'))
        $shortcut.TargetPath = $exePath
        $shortcut.WorkingDirectory = $root
        $shortcut.IconLocation = $exePath
        $shortcut.Save()
        Write-Host "Desktop shortcut created: $desktop\\peakfit.lnk"
    } else {
        Write-Host "Executable not found at $exePath; skipping shortcut creation."
    }
} catch {
    Write-Host "Failed to create desktop shortcut: $_"
}
