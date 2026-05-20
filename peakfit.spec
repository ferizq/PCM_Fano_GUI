# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_all

datas = [('E:\\VS_CODE\\PCM_Fano_GUI\\configs', 'configs'), ('E:\\VS_CODE\\PCM_Fano_GUI\\assets', 'assets')]
binaries = []
hiddenimports = ['matplotlib', 'matplotlib.backends.backend_qtagg', 'matplotlib.backends.qt_compat', 'lmfit', 'plotly', 'pkg_resources.py2_warn', 'PySide6', 'PySide6.QtWidgets', 'PySide6.QtGui', 'PySide6.QtCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineCore']
datas += collect_data_files('plotly')
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['scripts\\gui_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
