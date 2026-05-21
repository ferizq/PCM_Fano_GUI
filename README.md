# peakfit — Run & Build Instructions

Short notes for running the test runs used during development and for building the Windows executable.

Prerequisites
- Windows (instructions below are Windows-focused). Python 3.14 recommended.
- A virtual environment is used at `.venv` (the build script creates/uses it).
- Install Python dependencies from `requirements.txt` inside the venv:

PowerShell (from repo root):
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Quick test run (script)
- From the repository root run the CLI script with the example data used in tests. Replace paths as needed:

```powershell
.\.venv\Scripts\python.exe scripts\fit_peak.py "E:\VS_CODE\PCM_Fano\10nm_5L_peak.txt" \
  -c "E:\VS_CODE\PCM_Fano\configs\default_params.json" \
  -o "E:\VS_CODE\PCM_Fano\fit_10nm_5L_test.html"
```

- Behavior:
  - If you run the script without CLI arguments it will open small file‑selection dialogs to pick data/config/output.
  - While fitting a small "Peakfit — Running" window appears with an indeterminate progress bar and a Cancel button.
  - When fitting finishes a Save As dialog appears to choose the fitted‑data filename. If cancelled a default file `<input>_fitted_func<ext>` is written next to the input data file.
  - The HTML report is written to the `-o` path.

Run the packaged EXE (after building)
- If you prefer the already-built executable, run it like:
```powershell
#.\dist\peakfit.exe "E:\VS_CODE\PCM_Fano\10nm_5L_peak.txt" -c "E:\VS_CODE\PCM_Fano\configs\default_params.json" -o "E:\VS_CODE\PCM_Fano\fit_test.html"
```

Build the Windows executable (recommended)
- The project includes a PowerShell build script that creates the `.venv`, installs dependencies, and runs PyInstaller with the proper options. From the repo root run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```

- Output files are placed in `dist\` (e.g. `dist\peakfit.exe`, `dist\peakfit_cli.exe`, `dist\peakfit_portable.zip`). Use the provided script — it includes the proper data files and PyInstaller options.

Alternative: manual PyInstaller (not recommended)
- If you must run PyInstaller manually, activate the venv and run a one-file build. Example (simple):

```powershell
.\.venv\Scripts\Activate.ps1
pyinstaller --onefile --noconsole --name peakfit --add-data "configs\default_params.json;configs" scripts\fit_peak.py
```

Notes
- The produced EXE is currently unsigned; Windows SmartScreen may show "Unknown publisher" the first time it runs. In that case instruct users: click "More info" → "Run anyway".
- If file dialogs appear behind other windows on some systems, run the EXE from PowerShell to keep dialogs visible in the taskbar.
- The fitted data file is saved as `<input>_fitted_func<ext>` by default (unless you pick a different filename in the Save As dialog).

Troubleshooting
- Missing dependencies: ensure the venv is activated and run `pip install -r requirements.txt`.
- Build failures: open `scripts\build_windows.ps1` to check PyInstaller options; run the script from an elevated PowerShell if you get permission errors.

If you want, I can add a small `README.txt` into `dist\` with the same short user instructions for distribution.# Peak-fitting with integral-defined model

This small package provides a numerical model where the fitted peak is defined
by an integral over a kernel. It supports fitting via `scipy.optimize.curve_fit`
and optionally `lmfit`, and produces interactive Plotly HTML plots with residuals.

Quick start:

1. Install dependencies (for development/build):

```bash
pip install -r requirements.txt
```

2. Run the example CLI:

```bash
python scripts/fit_peak.py path/to/data.txt -c configs/default_params.json -o result.html
```

3. Run a quick benchmark for model and fit runtime (medium workload):

```bash
python scripts/benchmark_fit.py --points 2000 --grid-size 600 --repeats 3
```

Optional: compare accelerator modes explicitly (`auto`, `c`, `numpy`, `numba`):

```bash
python scripts/benchmark_fit.py --points 2000 --grid-size 600 --repeats 3 --accelerator auto
python scripts/benchmark_fit.py --points 2000 --grid-size 600 --repeats 3 --accelerator c
python scripts/benchmark_fit.py --points 2000 --grid-size 600 --repeats 3 --accelerator numpy
```

Experimental native C accelerator (`c`)
--------------------------------------

The repository now includes an optional native C core for the grid integrator.
You can build it into `build/peakfit_core.dll` on Windows using:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_c_core.ps1
```

After building, choose accelerator `c` from the GUI, or pass it in CLI mode:

```bash
python scripts/fit_peak.py path/to/data.txt -c configs/default_params.json -a c -o result.html
```

If the DLL is not present, the app falls back safely to the existing Python paths.

If `numba` is installed, `--accelerator auto` will use the compiled grid path; otherwise it falls back to numpy automatically.

Data must be a two-column whitespace-separated text file with X (iw) in the first
column and Y in the second column.

Packaging for Windows (standalone executable)

To distribute the program to Windows users without requiring them to install Python
or any libraries, build a single-file executable using PyInstaller on a Windows
machine. A convenience PowerShell script is included at [scripts/build_windows.ps1](scripts/build_windows.ps1).

From a PowerShell prompt in the project root, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```

The script will create a `.venv`, install requirements and PyInstaller, and
produce `dist\peakfit_cli.exe`. Distribute that EXE to your users — they won't
need to install Python or the project's Python libraries.

Manual alternative (steps):

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt pyinstaller
.\.venv\Scripts\python.exe -m PyInstaller --onefile --name peakfit_cli --add-data "configs\default_params.json;configs" scripts\fit_peak.py
```

GUI application (new)
---------------------

This repository now includes a PySide6-based GUI. To run it from the repository (development mode):

PowerShell (from repo root, with venv active):

```powershell
.\.venv\Scripts\python.exe -m scripts.gui_app
```

The GUI preview now uses an embedded matplotlib canvas with a native toolbar
(zoom/pan/home) for data+fit and residual plots.

GUI HTML export is disabled. If you still need HTML reports, use the CLI path
(`scripts/fit_peak.py`), which keeps Plotly-based HTML output.

Build a folder-based portable GUI distribution with the helper script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_gui.ps1
```

The generated folder will be placed in `dist\peakfit_gui` and includes the GUI executable plus bundled `configs/` and `assets/` (if present).

