"""torchaudio 2.10+ 호환성 shim.

torchaudio 2.10에서 제거된 API(info, AudioMetaData, list_audio_backends)를
soundfile 기반으로 대체하여 pyannote.audio가 동작하도록 한다.
torchaudio 자체가 없는 환경(exe 번들 등)에서는 안전하게 건너뛴다.
"""


def patch_torchaudio():
    """torchaudio에 누락된 API를 주입한다. 이미 있거나 torchaudio가 없으면 건너뛴다."""
    try:
        import torchaudio
    except ImportError:
        return  # torchaudio 없음 — 화자 분리 불가, 앱은 정상 실행

    if hasattr(torchaudio, "AudioMetaData") and hasattr(torchaudio, "info"):
        return  # 패치 불필요

    try:
        import soundfile as sf
    except ImportError:
        return  # soundfile 없으면 패치 불가

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
