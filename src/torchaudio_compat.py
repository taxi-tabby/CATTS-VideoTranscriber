"""의존성 호환성 shim.

1. torchaudio 2.10+: 제거된 API(info, AudioMetaData, list_audio_backends)를
   soundfile 기반으로 대체하여 pyannote.audio가 동작하도록 한다.
2. huggingface_hub 1.x: 제거된 use_auth_token 파라미터를 token으로 변환하여
   pyannote.audio가 최신 huggingface_hub에서 동작하도록 한다.
"""


def patch_torchaudio():
    """torchaudio에 누락된 API를 주입한다. 이미 있거나 torchaudio가 없으면 건너뛴다."""
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


def patch_huggingface_hub():
    """huggingface_hub의 use_auth_token → token 호환성 패치."""
    try:
        import huggingface_hub
    except ImportError:
        return

    import inspect

    # hf_hub_download 패치
    _original_download = huggingface_hub.hf_hub_download
    sig = inspect.signature(_original_download)
    if "use_auth_token" not in sig.parameters:
        def _patched_download(*args, **kwargs):
            token = kwargs.pop("use_auth_token", None)
            if token is not None and "token" not in kwargs:
                kwargs["token"] = token
            return _original_download(*args, **kwargs)
        huggingface_hub.hf_hub_download = _patched_download

    # hf_hub_download이 호출하는 다른 함수들도 패치
    for fn_name in ["model_info", "list_repo_files"]:
        original = getattr(huggingface_hub, fn_name, None)
        if original is None:
            continue
        sig = inspect.signature(original)
        if "use_auth_token" not in sig.parameters:
            def _make_patch(orig):
                def _patched(*args, **kwargs):
                    token = kwargs.pop("use_auth_token", None)
                    if token is not None and "token" not in kwargs:
                        kwargs["token"] = token
                    return orig(*args, **kwargs)
                return _patched
            setattr(huggingface_hub, fn_name, _make_patch(original))


def apply_all_patches():
    """모든 호환성 패치를 적용한다."""
    patch_torchaudio()
    patch_huggingface_hub()
