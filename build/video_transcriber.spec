# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import imageio_ffmpeg
import whisper

block_cipher = None

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
whisper_dir = os.path.dirname(whisper.__file__)

a = Analysis(
    ['../src/main.py'],
    pathex=[os.path.abspath('..')],
    binaries=[(ffmpeg_exe, 'imageio_ffmpeg/binaries')],
    datas=[
        (os.path.join(whisper_dir, 'assets'), 'whisper/assets'),
        (os.path.abspath('../assets/icon'), 'assets/icon'),
    ],
    hiddenimports=[
        'whisper',
        'PySide6',
        'numpy',
        'imageio_ffmpeg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'PIL'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VideoTranscriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=os.path.abspath('../assets/icon/app.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VideoTranscriber',
)
