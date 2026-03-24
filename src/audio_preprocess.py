"""Whisper 전처리 모듈.

음성 인식 정확도를 높이기 위해 오디오 신호를 정리한다.
적용 순서:
  1. high-pass filter  — 저주파 럼블/험 노이즈 제거
  2. noise reduction   — 배경 소음 제거 (spectral gating)
  3. Silero VAD        — 비음성 구간 무음 처리 (환각 방지)
  4. 앞뒤 무음 트리밍   — 선행/후행 무음 제거
  5. peak normalization — 음량 정규화
"""

import numpy as np

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# 1. High-pass filter
# ---------------------------------------------------------------------------

def highpass_filter(audio: np.ndarray, cutoff_hz: int = 80, order: int = 5) -> np.ndarray:
    """Butterworth high-pass filter로 저주파 럼블/험 노이즈를 제거한다."""
    from scipy.signal import butter, sosfilt

    sos = butter(order, cutoff_hz, btype="high", fs=SAMPLE_RATE, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Noise reduction
# ---------------------------------------------------------------------------

def reduce_noise(audio: np.ndarray) -> np.ndarray:
    """Spectral gating으로 배경 소음을 제거한다."""
    import noisereduce as nr

    return nr.reduce_noise(
        y=audio,
        sr=SAMPLE_RATE,
        stationary=False,
        prop_decrease=0.75,
        n_fft=2048,
        hop_length=512,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Silero VAD — 비음성 구간 무음 처리
# ---------------------------------------------------------------------------

def _load_silero_vad():
    """Silero VAD 모델을 로드한다 (최초 1회만, 이후 캐시)."""
    import torch
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad",
        trust_repo=True,
    )
    return model, utils


def suppress_non_speech(audio: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    """Silero VAD로 음성이 아닌 구간을 무음(0)으로 만든다.

    타이밍을 보존하면서 비음성 구간에서의 Whisper 환각을 방지한다.
    threshold가 낮을수록 더 많은 구간을 음성으로 판단한다.
    """
    import torch

    model, utils = _load_silero_vad()
    get_speech_timestamps = utils[0]

    # Silero VAD는 torch tensor를 기대
    audio_tensor = torch.from_numpy(audio)

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=threshold,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=30,
    )

    # Silero VAD 모델 메모리 해제
    del model, utils, audio_tensor
    import gc as _gc
    _gc.collect()

    if not speech_timestamps:
        # 음성이 전혀 감지되지 않으면 원본 반환
        return audio

    # 비음성 구간을 0으로 채움
    suppressed = np.zeros_like(audio)
    for ts in speech_timestamps:
        start = ts["start"]
        end = ts["end"]
        suppressed[start:end] = audio[start:end]

    return suppressed


# ---------------------------------------------------------------------------
# 4. 앞뒤 무음 트리밍 (타이밍 오프셋 반환)
# ---------------------------------------------------------------------------

def trim_silence(audio: np.ndarray, top_db: float = 30.0) -> tuple[np.ndarray, int]:
    """앞뒤 무음을 제거한다. (trimmed_audio, trimmed_samples_from_start)를 반환.

    Whisper는 선행 무음이 길면 환각을 일으키므로 제거한다.
    trimmed_samples_from_start는 타이밍 보정에 사용할 수 있다.
    """
    # 에너지 기반 무음 감지
    frame_length = int(SAMPLE_RATE * 0.025)  # 25ms
    hop = int(SAMPLE_RATE * 0.010)           # 10ms

    threshold = 10 ** (-top_db / 20.0)

    # 프레임별 RMS 계산
    n_frames = max(1, (len(audio) - frame_length) // hop + 1)
    is_speech = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > threshold:
            is_speech[i] = True

    if not np.any(is_speech):
        return audio, 0

    first = np.argmax(is_speech)
    last = len(is_speech) - 1 - np.argmax(is_speech[::-1])

    start_sample = max(0, first * hop - SAMPLE_RATE // 10)   # 100ms 여유
    end_sample = min(len(audio), (last + 1) * hop + frame_length + SAMPLE_RATE // 10)

    return audio[start_sample:end_sample].copy(), start_sample


# ---------------------------------------------------------------------------
# 5. Peak normalization
# ---------------------------------------------------------------------------

def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """피크 기준으로 음량을 정규화한다."""
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio
    return (audio * (target_peak / peak)).astype(np.float32)


# ---------------------------------------------------------------------------
# 통합 파이프라인
# ---------------------------------------------------------------------------

def preprocess(audio: np.ndarray) -> tuple[np.ndarray, int]:
    """전처리 파이프라인.

    Returns:
        (preprocessed_audio, trim_offset_samples)
        trim_offset_samples: 트리밍으로 제거된 앞부분 샘플 수 (타이밍 보정용)
    """
    audio = highpass_filter(audio)
    audio = reduce_noise(audio)
    audio = suppress_non_speech(audio)
    audio, trim_offset = trim_silence(audio)
    audio = normalize_peak(audio)
    return audio, trim_offset
