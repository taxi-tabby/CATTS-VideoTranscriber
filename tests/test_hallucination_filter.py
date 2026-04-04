"""hallucination_filter.py 테스트."""
import pytest


class TestFilterNoSpeech:
    def test_removes_high_no_speech(self):
        from src.hallucination_filter import _filter_no_speech
        segments = [
            {"start": 0, "end": 1, "text": "hello", "no_speech_prob": 0.9},
            {"start": 1, "end": 2, "text": "world", "no_speech_prob": 0.1},
        ]
        result = _filter_no_speech(segments)
        assert len(result) == 1
        assert result[0]["text"] == "world"

    def test_keeps_without_no_speech_key(self):
        from src.hallucination_filter import _filter_no_speech
        segments = [{"start": 0, "end": 1, "text": "ok"}]
        assert len(_filter_no_speech(segments)) == 1


class TestFilterTinySegments:
    def test_removes_tiny_punctuation(self):
        from src.hallucination_filter import _filter_tiny_segments
        segments = [
            {"start": 0.0, "end": 0.05, "text": "..."},
            {"start": 1.0, "end": 3.0, "text": "정상 텍스트"},
        ]
        result = _filter_tiny_segments(segments)
        assert len(result) == 1

    def test_keeps_tiny_with_real_text(self):
        from src.hallucination_filter import _filter_tiny_segments
        segments = [{"start": 0.0, "end": 0.05, "text": "네"}]
        assert len(_filter_tiny_segments(segments)) == 1


class TestFilterRepetitions:
    def test_removes_consecutive_repeats(self):
        from src.hallucination_filter import _filter_repetitions
        segments = [
            {"start": 0, "end": 1, "text": "안녕하세요"},
            {"start": 1, "end": 2, "text": "안녕하세요"},
            {"start": 2, "end": 3, "text": "안녕하세요"},
            {"start": 3, "end": 4, "text": "반갑습니다"},
        ]
        result = _filter_repetitions(segments)
        assert len(result) == 2
        assert result[0]["text"] == "안녕하세요"
        assert result[1]["text"] == "반갑습니다"

    def test_keeps_non_consecutive_same_text(self):
        from src.hallucination_filter import _filter_repetitions
        segments = [
            {"start": 0, "end": 1, "text": "A"},
            {"start": 1, "end": 2, "text": "B"},
            {"start": 2, "end": 3, "text": "A"},
        ]
        result = _filter_repetitions(segments)
        assert len(result) == 3

    def test_empty_input(self):
        from src.hallucination_filter import _filter_repetitions
        assert _filter_repetitions([]) == []


class TestFilterLanguageMismatch:
    def test_removes_english_when_ko(self):
        from src.hallucination_filter import _filter_language_mismatch
        segments = [
            {"start": 0, "end": 1, "text": "Thank you for watching"},
            {"start": 1, "end": 2, "text": "안녕하세요"},
        ]
        result = _filter_language_mismatch(segments, "ko")
        assert len(result) == 1
        assert result[0]["text"] == "안녕하세요"

    def test_keeps_mixed_text(self):
        from src.hallucination_filter import _filter_language_mismatch
        segments = [{"start": 0, "end": 1, "text": "YouTube 채널에 오신 것을"}]
        result = _filter_language_mismatch(segments, "ko")
        assert len(result) == 1

    def test_keeps_numbers_only(self):
        from src.hallucination_filter import _filter_language_mismatch
        segments = [{"start": 0, "end": 1, "text": "123, 456"}]
        result = _filter_language_mismatch(segments, "ko")
        assert len(result) == 1

    def test_skips_for_english_language(self):
        from src.hallucination_filter import _filter_language_mismatch
        segments = [{"start": 0, "end": 1, "text": "Hello world"}]
        result = _filter_language_mismatch(segments, "en")
        assert len(result) == 1

    def test_skips_for_none_language(self):
        from src.hallucination_filter import filter_hallucinations
        segments = [{"start": 0, "end": 1, "text": "random text"}]
        result = filter_hallucinations(segments, language=None)
        assert len(result) == 1


class TestFilterHallucinations:
    def test_combined_filters(self):
        from src.hallucination_filter import filter_hallucinations
        segments = [
            {"start": 0, "end": 1, "text": "안녕하세요", "no_speech_prob": 0.1},
            {"start": 1, "end": 2, "text": "안녕하세요", "no_speech_prob": 0.1},
            {"start": 2, "end": 3, "text": "안녕하세요", "no_speech_prob": 0.1},
            {"start": 3, "end": 4, "text": "Thank you", "no_speech_prob": 0.1},
            {"start": 4, "end": 5, "text": "환각 세그먼트", "no_speech_prob": 0.9},
        ]
        result = filter_hallucinations(segments, language="ko")
        texts = [s["text"] for s in result]
        assert "안녕하세요" in texts
        assert texts.count("안녕하세요") == 1  # 반복 제거
        assert "Thank you" not in texts  # 언어 불일치
        assert "환각 세그먼트" not in texts  # no_speech
