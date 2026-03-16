"""의존성 호환성 shim.

1. torchaudio 2.10+: 제거/변경된 API(info, AudioMetaData, list_audio_backends, load)를
   soundfile 기반으로 대체하여 pyannote.audio 및 speechbrain이 동작하도록 한다.
   torchaudio 2.10부터 load()가 TorchCodec 필수이므로 soundfile로 대체한다.
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
    _patch_torch_load()


def _patch_torchaudio():
    """torchaudio에 누락/변경된 API를 주입한다."""
    try:
        import torchaudio
    except ImportError:
        return

    try:
        import soundfile as sf
    except ImportError:
        return

    import torch
    import numpy as np
    from dataclasses import dataclass

    # AudioMetaData / info — torchaudio 2.10+에서 제거됨
    if not hasattr(torchaudio, "AudioMetaData") or not hasattr(torchaudio, "info"):
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

        torchaudio.AudioMetaData = AudioMetaData
        torchaudio.info = torchaudio_info

    if not hasattr(torchaudio, "list_audio_backends"):
        def list_audio_backends():
            return ["soundfile"]
        torchaudio.list_audio_backends = list_audio_backends

    # load — torchaudio 2.10+에서 TorchCodec 필수로 변경됨
    # soundfile 기반으로 대체하여 TorchCodec 없이도 동작하게 한다.
    _original_load = getattr(torchaudio, "load", None)
    if _original_load is not None and not getattr(_original_load, "_compat_patched", False):
        def _sf_load(
            uri,
            frame_offset=0,
            num_frames=-1,
            normalize=True,
            channels_first=True,
            format=None,
            buffer_size=4096,
            backend=None,
        ):
            # 먼저 원본 시도 (soundfile backend 등이 동작할 수 있음)
            try:
                return _original_load(
                    uri,
                    frame_offset=frame_offset,
                    num_frames=num_frames,
                    normalize=normalize,
                    channels_first=channels_first,
                    format=format,
                    buffer_size=buffer_size,
                    backend=backend,
                )
            except (ImportError, RuntimeError):
                pass

            # 원본 실패 시 soundfile 폴백
            read_kwargs = dict(
                start=frame_offset,
                dtype="float32" if normalize else "int16",
                always_2d=True,
            )
            if num_frames > 0:
                read_kwargs["frames"] = num_frames
            data, sample_rate = sf.read(
                str(uri) if not hasattr(uri, "read") else uri,
                **read_kwargs,
            )
            # data shape: (frames, channels) → torch tensor
            tensor = torch.from_numpy(data)
            if channels_first:
                tensor = tensor.t()  # (channels, frames)
            return tensor, sample_rate

        _sf_load._compat_patched = True
        torchaudio.load = _sf_load


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


def _patch_torch_load():
    """PyTorch 2.6+에서 weights_only=True 기본값 문제를 우회한다.

    pyannote/speechbrain 체크포인트가 TorchVersion, Specifications 등
    다양한 커스텀 타입을 포함하므로 weights_only=False를 강제한다.
    pyannote 모델은 Hugging Face 공식 허브에서 받으므로 신뢰할 수 있다.

    torch.load와 torch.serialization.load 둘 다 패치한다.
    lightning 등 내부 코드가 torch.serialization.load를 직접 호출할 수 있기 때문.
    """
    try:
        import torch
        import torch.serialization
    except ImportError:
        return

    import functools

    # torch.load 패치
    if not getattr(torch.load, "_compat_patched", False):
        _original_load = torch.load

        @functools.wraps(_original_load)
        def _patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)

        _patched_load._compat_patched = True
        torch.load = _patched_load

    # torch.serialization.load도 별도로 패치 (다른 함수 객체일 수 있음)
    if not getattr(torch.serialization.load, "_compat_patched", False):
        _original_ser_load = torch.serialization.load

        @functools.wraps(_original_ser_load)
        def _patched_ser_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_ser_load(*args, **kwargs)

        _patched_ser_load._compat_patched = True
        torch.serialization.load = _patched_ser_load
