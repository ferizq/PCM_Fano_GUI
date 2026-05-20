# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

plotly_datas = collect_data_files('plotly')


a = Analysis(
    ['scripts\\gui_app.py'],
    pathex=['.'],
    binaries=[],
    datas=plotly_datas + [('configs', 'configs'), ('assets', 'assets')],
    hiddenimports=['lmfit', 'plotly', 'PySide6', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineCore'],
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
    name='peakfit_gui',
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
