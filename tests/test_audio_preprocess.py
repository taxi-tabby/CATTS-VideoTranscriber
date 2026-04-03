import numpy as np
import pytest
from unittest.mock import patch, MagicMock

SAMPLE_RATE = 16000


class TestHighpassFilter:
    def test_output_same_length(self):
        from src.audio_preprocess import highpass_filter
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
        result = highpass_filter(audio)
        assert len(result) == len(audio)
        assert result.dtype == np.float32

    def test_removes_low_frequency(self):
        from src.audio_preprocess import highpass_filter
        # 20Hz 사인파 (cutoff 80Hz 이하)
        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        low_freq = np.sin(2 * np.pi * 20 * t).astype(np.float32)
        result = highpass_filter(low_freq, cutoff_hz=80)
        # 에너지가 크게 감소해야 함
        assert np.std(result) < np.std(low_freq) * 0.3


class TestReduceNoise:
    def test_output_same_length(self):
        from src.audio_preprocess import reduce_noise
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.01
        result = reduce_noise(audio)
        assert len(result) == len(audio)
        assert result.dtype == np.float32


class TestTrimSilence:
    def test_trims_leading_silence(self):
        from src.audio_preprocess import trim_silence
        # 1초 무음 + 1초 신호 + 1초 무음
        audio = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
        audio[SAMPLE_RATE:SAMPLE_RATE * 2] = 0.5
        trimmed, offset = trim_silence(audio)
        assert len(trimmed) < len(audio)
        assert offset > 0

    def test_all_silence_returns_original(self):
        from src.audio_preprocess import trim_silence
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        trimmed, offset = trim_silence(audio)
        assert len(trimmed) == SAMPLE_RATE
        assert offset == 0

    def test_returns_offset_in_samples(self):
        from src.audio_preprocess import trim_silence
        audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        audio[SAMPLE_RATE:] = 0.5  # 후반부에만 신호
        _, offset = trim_silence(audio)
        assert isinstance(offset, (int, np.integer))
        assert offset > 0


class TestNormalizePeak:
    def test_normalizes_to_target(self):
        from src.audio_preprocess import normalize_peak
        audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        result = normalize_peak(audio, target_peak=0.95)
        assert np.max(np.abs(result)) == pytest.approx(0.95, abs=0.01)

    def test_silent_audio_unchanged(self):
        from src.audio_preprocess import normalize_peak
        audio = np.zeros(100, dtype=np.float32)
        result = normalize_peak(audio)
        assert np.all(result == 0)

    def test_output_dtype(self):
        from src.audio_preprocess import normalize_peak
        audio = np.array([0.5, -0.5], dtype=np.float32)
        result = normalize_peak(audio)
        assert result.dtype == np.float32


class TestPreprocess:
    def test_returns_tuple(self):
        from src.audio_preprocess import preprocess
        audio = np.random.randn(SAMPLE_RATE * 2).astype(np.float32) * 0.3
        with patch("src.audio_preprocess.suppress_non_speech", side_effect=lambda a, **kw: a):
            result = preprocess(audio)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], int)

    def test_vocal_separation_called_when_enabled(self):
        from src.audio_preprocess import preprocess
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.3
        with patch("src.audio_preprocess.separate_vocals", return_value=audio) as mock_sep, \
             patch("src.audio_preprocess.suppress_non_speech", side_effect=lambda a, **kw: a):
            preprocess(audio, use_vocal_separation=True)
        mock_sep.assert_called_once()

    def test_vocal_separation_skipped_by_default(self):
        from src.audio_preprocess import preprocess
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.3
        with patch("src.audio_preprocess.separate_vocals") as mock_sep, \
             patch("src.audio_preprocess.suppress_non_speech", side_effect=lambda a, **kw: a):
            preprocess(audio)
        mock_sep.assert_not_called()


class TestGetSpeechSegments:
    def test_returns_segments_in_seconds(self):
        """VAD 결과를 초 단위 세그먼트로 반환해야 한다."""
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        audio[int(0.2 * SAMPLE_RATE):int(0.8 * SAMPLE_RATE)] = 0.5

        mock_timestamps = [
            {"start": int(0.2 * SAMPLE_RATE), "end": int(0.8 * SAMPLE_RATE)}
        ]

        with patch("src.audio_preprocess._load_silero_vad") as mock_load:
            mock_model = MagicMock()
            mock_get_ts = MagicMock(return_value=mock_timestamps)
            mock_load.return_value = (mock_model, [mock_get_ts])

            from src.audio_preprocess import get_speech_segments
            segments = get_speech_segments(audio)

        assert len(segments) == 1
        assert abs(segments[0]["start"] - 0.2) < 0.01
        assert abs(segments[0]["end"] - 0.8) < 0.01

    def test_empty_audio_returns_empty(self):
        """음성이 없으면 빈 리스트를 반환해야 한다."""
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

        with patch("src.audio_preprocess._load_silero_vad") as mock_load:
            mock_model = MagicMock()
            mock_get_ts = MagicMock(return_value=[])
            mock_load.return_value = (mock_model, [mock_get_ts])

            from src.audio_preprocess import get_speech_segments
            segments = get_speech_segments(audio)

        assert segments == []


class TestMergeSpeechSegments:
    def test_merge_short_gap(self):
        """0.5초 미만 간격의 인접 구간을 병합해야 한다."""
        from src.audio_preprocess import merge_speech_segments
        segments = [
            {"start": 0.0, "end": 2.0},
            {"start": 2.3, "end": 5.0},
        ]
        result = merge_speech_segments(segments, min_gap=0.5)
        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 5.0

    def test_keep_large_gap(self):
        """0.5초 이상 간격은 유지해야 한다."""
        from src.audio_preprocess import merge_speech_segments
        segments = [
            {"start": 0.0, "end": 2.0},
            {"start": 3.0, "end": 5.0},
        ]
        result = merge_speech_segments(segments, min_gap=0.5)
        assert len(result) == 2

    def test_drop_short_segments(self):
        """min_duration 미만의 세그먼트는 제거해야 한다."""
        from src.audio_preprocess import merge_speech_segments
        segments = [
            {"start": 0.0, "end": 0.3},
            {"start": 5.0, "end": 8.0},
        ]
        result = merge_speech_segments(segments, min_duration=0.5)
        assert len(result) == 1
        assert result[0]["start"] == 5.0

    def test_empty_input(self):
        from src.audio_preprocess import merge_speech_segments
        assert merge_speech_segments([]) == []


class TestSplitLongSegments:
    def test_split_long_segment(self):
        """max_duration보다 긴 구간을 분할해야 한다."""
        from src.audio_preprocess import split_long_segments
        segments = [{"start": 0.0, "end": 25.0}]
        result = split_long_segments(segments, max_duration=10.0)
        assert len(result) == 3
        assert result[0] == {"start": 0.0, "end": 10.0}
        assert result[1] == {"start": 10.0, "end": 20.0}
        assert result[2] == {"start": 20.0, "end": 25.0}

    def test_short_segment_unchanged(self):
        """max_duration 이하 구간은 그대로 유지해야 한다."""
        from src.audio_preprocess import split_long_segments
        segments = [{"start": 0.0, "end": 5.0}]
        result = split_long_segments(segments, max_duration=10.0)
        assert len(result) == 1
        assert result[0] == {"start": 0.0, "end": 5.0}

    def test_empty_input(self):
        from src.audio_preprocess import split_long_segments
        assert split_long_segments([]) == []
