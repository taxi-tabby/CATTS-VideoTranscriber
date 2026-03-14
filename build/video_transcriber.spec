# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import imageio_ffmpeg
import whisper
from PyInstaller.utils.hooks import collect_all

block_cipher = None

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
whisper_dir = os.path.dirname(whisper.__file__)

# soundfile의 libsndfile DLL 경로
import soundfile
sf_dir = os.path.dirname(soundfile.__file__)
libsndfile_dir = os.path.join(sf_dir, '_soundfile_data')

# collect_all로 패키지 전체 수집 (바이너리, 데이터, hiddenimports)
extra_datas = []
extra_binaries = []
extra_hiddenimports = []

for pkg in ['torchaudio', 'pyannote', 'speechbrain', 'lightning', 'lightning_fabric', 'pytorch_lightning']:
    try:
        datas, binaries, hiddenimports = collect_all(pkg)
        extra_datas.extend(datas)
        extra_binaries.extend(binaries)
        extra_hiddenimports.extend(hiddenimports)
    except Exception:
        pass  # 패키지 없으면 건너뜀

# lightning 패키지들의 version.info 명시적 포함
import site
sp = site.getsitepackages()[0] if site.getsitepackages() else os.path.join(sys.prefix, 'Lib', 'site-packages')
for pkg_name in ['lightning', 'lightning_fabric', 'pytorch_lightning']:
    vi = os.path.join(sp, pkg_name, 'version.info')
    if os.path.exists(vi):
        extra_datas.append((vi, pkg_name))

a = Analysis(
    ['../src/main.py'],
    pathex=[os.path.abspath('..')],
    binaries=[
        (ffmpeg_exe, 'imageio_ffmpeg/binaries'),
        (os.path.join(libsndfile_dir, 'libsndfile_x64.dll'), '_soundfile_data'),
    ] + extra_binaries,
    datas=[
        (os.path.join(whisper_dir, 'assets'), 'whisper/assets'),
    ] + extra_datas,
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
    ] + extra_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],
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
