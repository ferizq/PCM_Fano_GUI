# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

matplotlib_datas = collect_data_files('matplotlib')


a = Analysis(
    ['scripts\\gui_app.py'],
    pathex=[],
    binaries=[],
    datas=matplotlib_datas + [('configs', 'configs'), ('assets', 'assets')],
    hiddenimports=['lmfit', 'matplotlib', 'matplotlib.backends.backend_qtagg', 'PySide6', 'PySide6.QtWidgets', 'PySide6.QtGui', 'PySide6.QtCore'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='peakfit_gui2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
