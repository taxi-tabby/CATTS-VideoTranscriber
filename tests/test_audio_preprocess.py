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
