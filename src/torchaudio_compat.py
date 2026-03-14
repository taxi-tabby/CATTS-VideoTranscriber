"""torchaudio 2.10+ 호환성 shim.

torchaudio 2.10에서 제거된 API(info, AudioMetaData, list_audio_backends)를
soundfile 기반으로 대체하여 pyannote.audio가 동작하도록 한다.
"""

import importlib


def patch_torchaudio():
    """torchaudio에 누락된 API를 주입한다. 이미 있으면 건너뛴다."""
    import torchaudio

    if hasattr(torchaudio, "AudioMetaData") and hasattr(torchaudio, "info"):
        return  # 패치 불필요

    import soundfile as sf
    from dataclasses import dataclass

    @dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int = 16
        encoding: str = "PCM_S"

    def torchaudio_info(filepath, backend=None):
        info = sf.info(str(filepath))
        return AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
        )

    def list_audio_backends():
        return ["soundfile"]

    torchaudio.AudioMetaData = AudioMetaData
    torchaudio.info = torchaudio_info
    torchaudio.list_audio_backends = list_audio_backends
