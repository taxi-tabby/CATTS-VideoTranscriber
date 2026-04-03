import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

SAMPLE_RATE = 16000

# torch가 설치되지 않은 환경에서도 테스트할 수 있도록 mock 처리
mock_torch = MagicMock()
mock_torch.from_numpy = MagicMock(side_effect=lambda x: x)
sys.modules.setdefault("torch", mock_torch)


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
