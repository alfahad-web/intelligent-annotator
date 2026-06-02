# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for building Intelligent Annotator (labelImg) Windows executable
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all data files needed
datas = [
    ('data', 'data'),
    ('resources', 'resources'),
    ('intelligence/models', 'intelligence/models'),
]

# Collect PyQt5 data files
try:
    pyqt5_datas = collect_data_files('PyQt5')
    datas.extend(pyqt5_datas)
except:
    pass

# Collect torch data files (if available)
try:
    torch_datas = collect_data_files('torch')
    datas.extend(torch_datas)
except:
    pass

# Hidden imports - modules that PyInstaller might miss
hiddenimports = [
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.QtSvg',
    'lxml',
    'lxml.etree',
    'lxml._elementpath',
    'xml',
    'xml.etree',
    'xml.etree.ElementTree',
    'torch',
    'torch._C',
    'torch._C._fft',
    'ultralytics',
    'ultralytics.models',
    'ultralytics.utils',
    'cv2',  # opencv-python
    'numpy',
    'PIL',
    'PIL._tkinter_finder',
]

# Collect submodules for ultralytics
try:
    ultralytics_submodules = collect_submodules('ultralytics')
    hiddenimports.extend(ultralytics_submodules)
except:
    pass

# Workaround for PyInstaller bytecode analysis issues with Python 3.10
# Exclude problematic modules from bytecode analysis
import sys
if sys.version_info >= (3, 10):
    # For Python 3.10+, we need to be more careful with excludes
    excludes_list = [
        'matplotlib',
        'scipy',
        'pandas',
        'jupyter',
        'notebook',
        'IPython',
        'torch.distributed',
        'torch.multiprocessing',
    ]
else:
    excludes_list = [
        'matplotlib',
        'scipy',
        'pandas',
        'jupyter',
        'notebook',
        'IPython',
    ]

hiddenimports.extend([
    'intelligence',
    'intelligence.paths',
    'intelligence.yolo_inference',
])

a = Analysis(
    ['labelImg.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes_list,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    optimize=0,  # Disable optimization to avoid bytecode issues
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='labelImg',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want to see console output for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='resources/icons/app.ico',  # Uncomment if you have an .ico file
)

