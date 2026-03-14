# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import imageio_ffmpeg
import whisper

block_cipher = None

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
whisper_dir = os.path.dirname(whisper.__file__)

# soundfile의 libsndfile DLL 경로
import soundfile
sf_dir = os.path.dirname(soundfile.__file__)
libsndfile_dir = os.path.join(sf_dir, '_soundfile_data')

a = Analysis(
    ['../src/main.py'],
    pathex=[os.path.abspath('..')],
    binaries=[
        (ffmpeg_exe, 'imageio_ffmpeg/binaries'),
        (os.path.join(libsndfile_dir, 'libsndfile_x64.dll'), '_soundfile_data'),
    ],
    datas=[
        (os.path.join(whisper_dir, 'assets'), 'whisper/assets'),
    ],
    hiddenimports=[
        'whisper',
        'PySide6',
        'numpy',
        'imageio_ffmpeg',
        'soundfile',
        '_soundfile_data',
        'pyannote.audio',
        'pyannote.audio.pipelines',
        'pyannote.audio.pipelines.speaker_diarization',
        'pyannote.core',
        'pyannote.database',
        'pyannote.pipeline',
        'speechbrain',
        'transformers',
        'torchaudio',
        'huggingface_hub',
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
    icon=None,
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
