import pytest
from src.diarizer import assign_speakers, map_speaker_labels


class TestAssignSpeakers:
    def test_single_speaker(self):
        diarization_segments = [
            {"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"},
        ]
        whisper_segments = [
            {"start": 0.0, "end": 10.0, "text": "안녕하세요"},
            {"start": 10.0, "end": 20.0, "text": "반갑습니다"},
        ]
        result = assign_speakers(diarization_segments, whisper_segments)
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_00"
        assert result[0]["text"] == "안녕하세요"

    def test_two_speakers(self):
        diarization_segments = [
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
            {"start": 10.0, "end": 20.0, "speaker": "SPEAKER_01"},
        ]
        whisper_segments = [
            {"start": 0.0, "end": 9.0, "text": "첫번째"},
            {"start": 10.0, "end": 19.0, "text": "두번째"},
        ]
        result = assign_speakers(diarization_segments, whisper_segments)
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_maximum_overlap_matching(self):
        """세그먼트가 두 화자 구간에 걸칠 때 더 많이 겹치는 화자에 매칭"""
        diarization_segments = [
            {"start": 0.0, "end": 7.0, "speaker": "SPEAKER_00"},
            {"start": 7.0, "end": 15.0, "speaker": "SPEAKER_01"},
        ]
        whisper_segments = [
            {"start": 5.0, "end": 12.0, "text": "겹치는 구간"},
        ]
        result = assign_speakers(diarization_segments, whisper_segments)
        # 5~7 = 2초 SPEAKER_00, 7~12 = 5초 SPEAKER_01 → SPEAKER_01
        assert result[0]["speaker"] == "SPEAKER_01"

    def test_no_overlap_returns_none(self):
        diarization_segments = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        ]
        whisper_segments = [
            {"start": 10.0, "end": 15.0, "text": "겹침없음"},
        ]
        result = assign_speakers(diarization_segments, whisper_segments)
        assert result[0]["speaker"] is None

    def test_empty_diarization(self):
        whisper_segments = [
            {"start": 0.0, "end": 5.0, "text": "텍스트"},
        ]
        result = assign_speakers([], whisper_segments)
        assert result[0]["speaker"] is None

    def test_preserves_segment_fields(self):
        diarization_segments = [
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
        ]
        whisper_segments = [
            {"start": 1.0, "end": 5.0, "text": "보존 테스트"},
        ]
        result = assign_speakers(diarization_segments, whisper_segments)
        assert result[0]["start"] == 1.0
        assert result[0]["end"] == 5.0
        assert result[0]["text"] == "보존 테스트"


class TestMapSpeakerLabels:
    def test_maps_to_korean_labels(self):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "a", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "text": "b", "speaker": "SPEAKER_01"},
            {"start": 10.0, "end": 15.0, "text": "c", "speaker": "SPEAKER_00"},
        ]
        result = map_speaker_labels(segments)
        assert result[0]["speaker"] == "화자 1"
        assert result[1]["speaker"] == "화자 2"
        assert result[2]["speaker"] == "화자 1"

    def test_none_speaker_preserved(self):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "a", "speaker": None},
        ]
        result = map_speaker_labels(segments)
        assert result[0]["speaker"] is None

    def test_labels_by_first_appearance(self):
        """화자 라벨은 첫 등장 순서로 매핑"""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "a", "speaker": "SPEAKER_02"},
            {"start": 5.0, "end": 10.0, "text": "b", "speaker": "SPEAKER_00"},
        ]
        result = map_speaker_labels(segments)
        assert result[0]["speaker"] == "화자 1"
        assert result[1]["speaker"] == "화자 2"
