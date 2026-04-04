import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.diarizer import assign_speakers, map_speaker_labels, DIAR_PROFILES


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

    def test_weighted_voting_aggregates_overlap(self):
        """같은 화자의 여러 구간 겹침을 합산하여 배정"""
        diarization_segments = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 8.0, "speaker": "SPEAKER_01"},
            {"start": 8.0, "end": 10.0, "speaker": "SPEAKER_00"},
        ]
        whisper_segments = [
            {"start": 0.0, "end": 10.0, "text": "전체 구간"},
        ]
        result = assign_speakers(diarization_segments, whisper_segments)
        # SPEAKER_00: 3+2=5초, SPEAKER_01: 5초 → 동률 → SPEAKER_00 (먼저 등장)
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_weighted_voting_tiebreak_by_earliest(self):
        """동률 시 해당 구간에서 먼저 등장한 화자 선택"""
        diarization_segments = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_01"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_00"},
        ]
        whisper_segments = [
            {"start": 0.0, "end": 10.0, "text": "동률 구간"},
        ]
        result = assign_speakers(diarization_segments, whisper_segments)
        # 둘 다 5초 → SPEAKER_01이 0초에 먼저 등장
        assert result[0]["speaker"] == "SPEAKER_01"

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


class TestEstimateNumSpeakers:
    def test_single_speaker_high_similarity(self):
        """임베딩이 모두 유사하면 화자 1명으로 판정해야 한다."""
        from src.diarizer import _estimate_num_speakers
        rng = np.random.RandomState(42)
        base = rng.randn(256).astype(np.float32)
        base = base / np.linalg.norm(base)
        embeddings = np.array([base + rng.randn(256) * 0.01 for _ in range(5)])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        result = _estimate_num_speakers(embeddings, max_speakers=5)
        assert result == 1

    def test_two_distinct_speakers(self):
        """두 클러스터가 뚜렷하면 화자 2명으로 판정해야 한다."""
        from src.diarizer import _estimate_num_speakers
        rng = np.random.RandomState(42)
        center_a = rng.randn(256).astype(np.float32)
        center_b = -center_a
        embeddings_a = np.array([center_a + rng.randn(256) * 0.05 for _ in range(5)])
        embeddings_b = np.array([center_b + rng.randn(256) * 0.05 for _ in range(5)])
        embeddings = np.vstack([embeddings_a, embeddings_b])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        result = _estimate_num_speakers(embeddings, max_speakers=5)
        assert result == 2

    def test_respects_max_speakers(self):
        """max_speakers를 초과하지 않아야 한다."""
        from src.diarizer import _estimate_num_speakers
        rng = np.random.RandomState(42)
        embeddings = []
        for i in range(3):
            center = np.zeros(256, dtype=np.float32)
            center[i * 80:(i + 1) * 80] = 1.0
            for _ in range(5):
                embeddings.append(center + rng.randn(256) * 0.05)
        embeddings = np.array(embeddings)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        result = _estimate_num_speakers(embeddings, max_speakers=2)
        assert result <= 2

    def test_single_embedding_returns_one(self):
        """임베딩이 1개면 화자 1명이다."""
        from src.diarizer import _estimate_num_speakers
        embeddings = np.random.randn(1, 256).astype(np.float32)
        result = _estimate_num_speakers(embeddings, max_speakers=5)
        assert result == 1


class TestClusterEmbeddings:
    def test_assigns_labels_to_segments(self):
        """각 세그먼트에 speaker 라벨을 할당해야 한다."""
        from src.diarizer import _cluster_embeddings
        rng = np.random.RandomState(42)
        center_a = rng.randn(256).astype(np.float32)
        center_b = -center_a
        embeddings = np.vstack([
            np.array([center_a + rng.randn(256) * 0.05 for _ in range(3)]),
            np.array([center_b + rng.randn(256) * 0.05 for _ in range(3)]),
        ])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        segments = [
            {"start": float(i), "end": float(i + 1)}
            for i in range(6)
        ]

        result = _cluster_embeddings(embeddings, segments, num_speakers=2)
        assert len(result) == 6
        assert result[0]["speaker"] == result[1]["speaker"] == result[2]["speaker"]
        assert result[3]["speaker"] == result[4]["speaker"] == result[5]["speaker"]
        assert result[0]["speaker"] != result[3]["speaker"]
        assert result[0]["speaker"].startswith("SPEAKER_")

    def test_single_speaker(self):
        """num_speakers=1이면 모두 같은 화자여야 한다."""
        from src.diarizer import _cluster_embeddings
        embeddings = np.random.randn(5, 256).astype(np.float32)
        segments = [{"start": float(i), "end": float(i + 1)} for i in range(5)]

        result = _cluster_embeddings(embeddings, segments, num_speakers=1)
        speakers = set(r["speaker"] for r in result)
        assert len(speakers) == 1


class TestRunDiarization:
    def test_returns_segments_with_speaker_labels(self):
        """run_diarization()이 [{start, end, speaker}, ...] 형식을 반환해야 한다."""
        from src.diarizer import run_diarization

        fake_segments = [
            {"start": 0.0, "end": 3.0},
            {"start": 5.0, "end": 8.0},
            {"start": 10.0, "end": 13.0},
        ]
        fake_embeddings = np.random.randn(3, 256).astype(np.float32)

        with patch("src.diarizer._extract_speech_segments", return_value=fake_segments), \
             patch("src.diarizer._extract_embeddings", return_value=fake_embeddings), \
             patch("src.diarizer._estimate_num_speakers", return_value=1), \
             patch("src.diarizer._cluster_embeddings") as mock_cluster:

            mock_cluster.return_value = [
                {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
                {"start": 5.0, "end": 8.0, "speaker": "SPEAKER_00"},
                {"start": 10.0, "end": 13.0, "speaker": "SPEAKER_00"},
            ]

            result = run_diarization(
                audio_path="dummy.wav",
                hf_token="dummy_token",
            )

        assert len(result) == 3
        assert all("start" in r and "end" in r and "speaker" in r for r in result)
        assert all(r["speaker"] == "SPEAKER_00" for r in result)

    def test_num_speakers_skips_estimation(self):
        """num_speakers가 지정되면 _estimate_num_speakers를 호출하지 않아야 한다."""
        from src.diarizer import run_diarization

        fake_segments = [{"start": 0.0, "end": 5.0}]
        fake_embeddings = np.random.randn(1, 256).astype(np.float32)

        with patch("src.diarizer._extract_speech_segments", return_value=fake_segments), \
             patch("src.diarizer._extract_embeddings", return_value=fake_embeddings), \
             patch("src.diarizer._estimate_num_speakers") as mock_estimate, \
             patch("src.diarizer._cluster_embeddings") as mock_cluster:

            mock_cluster.return_value = [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            ]

            run_diarization(
                audio_path="dummy.wav",
                hf_token="dummy_token",
                num_speakers=2,
            )

        mock_estimate.assert_not_called()
        mock_cluster.assert_called_once()

    def test_empty_segments_returns_empty(self):
        """음성 구간이 없으면 빈 리스트를 반환해야 한다."""
        from src.diarizer import run_diarization

        with patch("src.diarizer._extract_speech_segments", return_value=[]):
            result = run_diarization(
                audio_path="dummy.wav",
                hf_token="dummy_token",
            )

        assert result == []


class TestDiarProfiles:
    def test_interview_profile_exists(self):
        assert "interview" in DIAR_PROFILES

    def test_noisy_profile_exists(self):
        assert "noisy" in DIAR_PROFILES

    def test_profiles_have_required_keys(self):
        required = {"label", "description", "vad_threshold", "min_gap",
                     "min_duration", "max_duration", "similarity_threshold"}
        for name, profile in DIAR_PROFILES.items():
            for key in required:
                assert key in profile, f"{name} profile missing {key}"

    def test_noisy_has_higher_vad_threshold(self):
        assert DIAR_PROFILES["noisy"]["vad_threshold"] > DIAR_PROFILES["interview"]["vad_threshold"]

    def test_noisy_has_lower_similarity_threshold(self):
        assert DIAR_PROFILES["noisy"]["similarity_threshold"] < DIAR_PROFILES["interview"]["similarity_threshold"]

    def test_run_diarization_uses_profile(self):
        """noisy 프로파일이 _extract_speech_segments에 전달되는지 확인."""
        from src.diarizer import run_diarization

        with patch("src.diarizer._extract_speech_segments", return_value=[]) as mock_extract:
            run_diarization(
                audio_path="dummy.wav",
                hf_token="dummy_token",
                profile_name="noisy",
            )

        call_kwargs = mock_extract.call_args
        profile_arg = call_kwargs[1].get("profile") or call_kwargs[0][1]
        assert profile_arg["label"] == "영상/영화/노래"
