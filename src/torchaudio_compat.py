"""의존성 호환성 shim.

1. torchaudio 2.10+: 제거된 API(info, AudioMetaData, list_audio_backends)를
   soundfile 기반으로 대체하여 pyannote.audio가 동작하도록 한다.
2. huggingface_hub 1.x: 제거된 use_auth_token 파라미터를 token으로 변환하여
   pyannote.audio가 최신 huggingface_hub에서 동작하도록 한다.
"""

import inspect
import sys


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


def _wrap_use_auth_token(original_fn):
    """use_auth_token → token 변환 래퍼를 생성한다."""
    sig = inspect.signature(original_fn)
    if "use_auth_token" in sig.parameters:
        return None  # 이미 지원하므로 패치 불필요

    def wrapper(*args, **kwargs):
        token = kwargs.pop("use_auth_token", None)
        if token is not None and "token" not in kwargs:
            kwargs["token"] = token
        return original_fn(*args, **kwargs)

    return wrapper


def patch_huggingface_hub():
    """huggingface_hub의 use_auth_token → token 호환성 패치.

    pyannote가 `from huggingface_hub import hf_hub_download`로 로컬 바인딩하므로,
    huggingface_hub 모듈 자체 + 이미 import된 모든 모듈의 로컬 참조를 패치한다.
    """
    try:
        import huggingface_hub
    except ImportError:
        return

    fn_names = ["hf_hub_download", "model_info", "list_repo_files"]

    # 1단계: huggingface_hub 모듈의 함수 자체를 패치
    patched = {}
    for fn_name in fn_names:
        original = getattr(huggingface_hub, fn_name, None)
        if original is None:
            continue
        wrapper = _wrap_use_auth_token(original)
        if wrapper is not None:
            setattr(huggingface_hub, fn_name, wrapper)
            patched[fn_name] = wrapper

    # 2단계: 이미 import된 모듈에서 로컬 바인딩된 참조도 패치
    # (예: pyannote.audio.core.pipeline에서 `from huggingface_hub import hf_hub_download`)
    for module in list(sys.modules.values()):
        if module is None or module is huggingface_hub:
            continue
        for fn_name, wrapper in patched.items():
            if hasattr(module, fn_name):
                attr = getattr(module, fn_name)
                # huggingface_hub의 원본 함수를 참조하고 있는 경우만 패치
                if callable(attr) and getattr(attr, "__module__", "").startswith("huggingface_hub"):
                    setattr(module, fn_name, wrapper)


def apply_all_patches():
    """모든 호환성 패치를 적용한다."""
    patch_torchaudio()
    patch_huggingface_hub()
