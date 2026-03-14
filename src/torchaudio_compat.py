"""의존성 호환성 shim.

1. torchaudio 2.10+: 제거된 API(info, AudioMetaData, list_audio_backends)를
   soundfile 기반으로 대체하여 pyannote.audio 및 speechbrain이 동작하도록 한다.
2. huggingface_hub 1.x: 제거된 use_auth_token 파라미터를 token으로 변환하여
   pyannote.audio가 최신 huggingface_hub에서 동작하도록 한다.
"""

_patched = False


def apply_all_patches():
    """모든 호환성 패치를 적용한다. 여러 번 호출해도 안전."""
    global _patched
    if _patched:
        return
    _patched = True
    _patch_torchaudio()
    _patch_huggingface_hub()


def _patch_torchaudio():
    """torchaudio에 누락된 API를 주입한다."""
    try:
        import torchaudio
    except ImportError:
        return

    if hasattr(torchaudio, "AudioMetaData") and hasattr(torchaudio, "info"):
        return

    try:
        import soundfile as sf
    except ImportError:
        return

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


def _patch_huggingface_hub():
    """huggingface_hub 함수에서 use_auth_token kwarg를 제거하고 token으로 변환."""
    try:
        import huggingface_hub
    except ImportError:
        return

    import functools
    import inspect

    fn_names = ["hf_hub_download", "model_info", "list_repo_files"]

    for fn_name in fn_names:
        original = getattr(huggingface_hub, fn_name, None)
        if original is None:
            continue
        # 이미 패치된 함수인지 확인
        if getattr(original, "_compat_patched", False):
            continue
        sig = inspect.signature(original)
        if "use_auth_token" in sig.parameters:
            continue  # 이미 지원하므로 패치 불필요

        @functools.wraps(original)
        def _make_wrapper(orig):
            def wrapper(*args, **kwargs):
                token = kwargs.pop("use_auth_token", None)
                if token is not None and "token" not in kwargs:
                    kwargs["token"] = token
                return orig(*args, **kwargs)
            wrapper._compat_patched = True
            return wrapper

        setattr(huggingface_hub, fn_name, _make_wrapper(original))
