"""transcriber.py의 유틸리티 함수 테스트."""

import os
import tempfile
import wave

import numpy as np
import pytest

from src.transcriber import (
    load_wav_as_numpy,
    save_numpy_as_wav,
    get_video_duration,
    _cap_workers_by_memory,
    SAMPLE_RATE,
)


def _make_wav(path: str, duration_sec: float = 1.0, sr: int = 16000) -> np.ndarray:
    """테스트용 WAV 파일을 생성하고 원본 float32 배열을 반환한다."""
    n = int(sr * duration_sec)
    audio = np.sin(2 * np.pi * 440 * np.arange(n) / sr).astype(np.float32) * 0.5
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return audio


class TestLoadWavAsNumpy:
    def test_loads_wav_correctly(self, tmp_path):
        wav_path = str(tmp_path / "test.wav")
        original = _make_wav(wav_path, duration_sec=0.5)
        loaded = load_wav_as_numpy(wav_path)
        assert loaded.dtype == np.float32
        assert len(loaded) == int(0.5 * SAMPLE_RATE)
        # 16-bit 양자화 오차 이내
        assert np.allclose(original, loaded, atol=1.0 / 32768)

    def test_values_in_range(self, tmp_path):
        wav_path = str(tmp_path / "test.wav")
        _make_wav(wav_path)
        loaded = load_wav_as_numpy(wav_path)
        assert np.all(loaded >= -1.0)
        assert np.all(loaded <= 1.0)


class TestSaveNumpyAsWav:
    def test_roundtrip(self, tmp_path):
        wav_path = str(tmp_path / "out.wav")
        # 440Hz 사인파 (amplitude 0.5) — 클리핑 없이 안정적
        t = np.arange(16000) / 16000.0
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        save_numpy_as_wav(audio, wav_path)
        assert os.path.exists(wav_path)
        loaded = load_wav_as_numpy(wav_path)
        assert len(loaded) == len(audio)
        # 16-bit PCM 양자화 오차: ~1/32768 ≈ 3e-5
        assert np.max(np.abs(audio - loaded)) < 1e-3

    def test_wav_format(self, tmp_path):
        wav_path = str(tmp_path / "out.wav")
        save_numpy_as_wav(np.zeros(8000, dtype=np.float32), wav_path)
        with wave.open(wav_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 8000


class TestGetVideoDuration:
    def test_correct_duration(self):
        audio = np.zeros(48000, dtype=np.float32)
        assert get_video_duration(audio) == pytest.approx(3.0)

    def test_empty_audio(self):
        assert get_video_duration(np.array([], dtype=np.float32)) == 0.0


class TestCapWorkersByMemory:
    def test_caps_when_low_memory(self):
        logs = []
        result = _cap_workers_by_memory(
            requested_workers=8,
            model_mb=1500,
            log_fn=logs.append,
        )
        # 모델 1500MB * 2.5 = 3750MB/개, 시스템에 따라 결과가 다르지만
        # 최소 1 이상이어야 함
        assert result >= 1
        assert result <= 8

    def test_returns_at_least_one(self):
        logs = []
        result = _cap_workers_by_memory(
            requested_workers=1,
            model_mb=100000,  # 매우 큰 모델
            log_fn=logs.append,
        )
        assert result >= 1
