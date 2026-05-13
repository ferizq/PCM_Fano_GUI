# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['scripts\\gui_app.py'],
    pathex=[],
    binaries=[],
    datas=[('E:\\VS_CODE\\PCM_Fano_GUI\\configs\\\\default_params.json', 'configs')],
    hiddenimports=['lmfit', 'plotly', 'pkg_resources.py2_warn', 'PySide6', 'PySide6.QtWidgets', 'PySide6.QtGui', 'PySide6.QtCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineCore'],
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
    name='peakfit',
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
