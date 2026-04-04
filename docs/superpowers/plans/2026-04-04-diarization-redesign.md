# 화자분리 파이프라인 재설계 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** pyannote Pipeline 일체형 화자분리를 Silero VAD + pyannote 임베딩 + silhouette 기반 클러스터링으로 교체하여 화자 수 자동 감지 정확도를 근본적으로 개선한다.

**Architecture:** Silero VAD로 음성 구간을 검출하고, pyannote의 wespeaker 임베딩 모델로 각 구간의 화자 특징 벡터를 추출한 뒤, Agglomerative Clustering + silhouette score로 최적 화자 수를 결정한다. 외부 인터페이스(`run_diarization`, `assign_speakers`, `map_speaker_labels`)는 유지한다.

**Tech Stack:** pyannote.audio (임베딩 모델만), Silero VAD, scikit-learn (AgglomerativeClustering, silhouette_score), torch, numpy

---

## 파일 구조

| 파일 | 역할 |
|------|------|
| `src/audio_preprocess.py` | `get_speech_segments()` 함수 추출, `suppress_non_speech()`가 이를 내부 호출하도록 리팩토링 |
| `src/diarizer.py` | `run_diarization()` 내부를 3단계 파이프라인으로 전면 교체 |
| `tests/test_audio_preprocess.py` | VAD 세그먼트 추출, 병합, 분할 테스트 (새 파일) |
| `tests/test_diarizer.py` | 클러스터링 로직 테스트 추가 (기존 assign_speakers/map_speaker_labels 테스트 유지) |

---

### Task 1: VAD 세그먼트 추출 함수

`audio_preprocess.py`에서 Silero VAD의 speech_timestamps를 반환하는 함수를 추출한다. 기존 `suppress_non_speech()`가 이를 내부적으로 호출하도록 리팩토링한다.

**Files:**
- Modify: `src/audio_preprocess.py:50-100`
- Create: `tests/test_audio_preprocess.py`

- [ ] **Step 1: Write failing test for `get_speech_segments()`**

```python
# tests/test_audio_preprocess.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

SAMPLE_RATE = 16000


class TestGetSpeechSegments:
    def test_returns_segments_in_seconds(self):
        """VAD 결과를 초 단위 세그먼트로 반환해야 한다."""
        # 1초 오디오: 0.2~0.8초 구간에 음성
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audio_preprocess.py::TestGetSpeechSegments -v`
Expected: FAIL with `ImportError` or `cannot import name 'get_speech_segments'`

- [ ] **Step 3: Implement `get_speech_segments()` and refactor `suppress_non_speech()`**

`src/audio_preprocess.py`의 VAD 관련 부분을 다음과 같이 변경한다:

```python
# ---------------------------------------------------------------------------
# 3. Silero VAD — 음성 구간 검출
# ---------------------------------------------------------------------------

def _load_silero_vad():
    """Silero VAD 모델을 로드한다 (최초 1회만, 이후 캐시)."""
    import torch
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad",
        trust_repo=True,
    )
    return model, utils


def get_speech_segments(
    audio: np.ndarray,
    threshold: float = 0.35,
    min_speech_ms: int = 250,
    min_silence_ms: int = 100,
    speech_pad_ms: int = 30,
) -> list[dict]:
    """Silero VAD로 음성 구간을 검출한다.

    Returns:
        [{"start": float(초), "end": float(초)}, ...] 형태의 리스트.
        음성이 없으면 빈 리스트를 반환한다.
    """
    import torch

    model, utils = _load_silero_vad()
    get_speech_timestamps = utils[0]

    audio_tensor = torch.from_numpy(audio)

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )

    del model, utils, audio_tensor
    import gc as _gc
    _gc.collect()

    return [
        {"start": ts["start"] / SAMPLE_RATE, "end": ts["end"] / SAMPLE_RATE}
        for ts in speech_timestamps
    ]


def suppress_non_speech(audio: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    """Silero VAD로 음성이 아닌 구간을 무음(0)으로 만든다.

    타이밍을 보존하면서 비음성 구간에서의 Whisper 환각을 방지한다.
    threshold가 낮을수록 더 많은 구간을 음성으로 판단한다.
    """
    segments = get_speech_segments(audio, threshold=threshold)

    if not segments:
        return audio

    suppressed = np.zeros_like(audio)
    for seg in segments:
        start = int(seg["start"] * SAMPLE_RATE)
        end = int(seg["end"] * SAMPLE_RATE)
        suppressed[start:end] = audio[start:end]

    return suppressed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_audio_preprocess.py::TestGetSpeechSegments -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/audio_preprocess.py tests/test_audio_preprocess.py
git commit -m "refactor: extract get_speech_segments() from suppress_non_speech()"
```

---

### Task 2: 세그먼트 병합/분할 로직

짧은 구간(< 0.5초)은 인접 구간과 병합하고, 긴 구간(> 10초)은 분할하는 로직을 추가한다.

**Files:**
- Modify: `src/audio_preprocess.py`
- Modify: `tests/test_audio_preprocess.py`

- [ ] **Step 1: Write failing tests for merge and split**

```python
# tests/test_audio_preprocess.py에 추가

class TestMergeSpeechSegments:
    def test_merge_short_gap(self):
        """0.5초 미만 간격의 인접 구간을 병합해야 한다."""
        from src.audio_preprocess import merge_speech_segments
        segments = [
            {"start": 0.0, "end": 2.0},
            {"start": 2.3, "end": 5.0},  # 0.3초 간격 → 병합
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
            {"start": 3.0, "end": 5.0},  # 1.0초 간격 → 유지
        ]
        result = merge_speech_segments(segments, min_gap=0.5)
        assert len(result) == 2

    def test_drop_short_segments(self):
        """min_duration 미만의 세그먼트는 제거해야 한다."""
        from src.audio_preprocess import merge_speech_segments
        segments = [
            {"start": 0.0, "end": 0.3},  # 0.3초 → 제거
            {"start": 5.0, "end": 8.0},  # 3.0초 → 유지
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
        assert len(result) == 3  # 0-10, 10-20, 20-25
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_audio_preprocess.py::TestMergeSpeechSegments tests/test_audio_preprocess.py::TestSplitLongSegments -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `merge_speech_segments()` and `split_long_segments()`**

`src/audio_preprocess.py` 하단에 추가:

```python
# ---------------------------------------------------------------------------
# 화자분리용 세그먼트 후처리
# ---------------------------------------------------------------------------

def merge_speech_segments(
    segments: list[dict],
    min_gap: float = 0.5,
    min_duration: float = 0.5,
) -> list[dict]:
    """인접 구간을 병합하고 짧은 구간을 제거한다.

    Args:
        segments: [{"start": float, "end": float}, ...] (초 단위, 정렬되어 있어야 함)
        min_gap: 이 값 미만의 간격은 병합한다 (초)
        min_duration: 이 값 미만의 구간은 제거한다 (초)
    """
    if not segments:
        return []

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        if seg["start"] - merged[-1]["end"] < min_gap:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    return [s for s in merged if s["end"] - s["start"] >= min_duration]


def split_long_segments(
    segments: list[dict],
    max_duration: float = 10.0,
) -> list[dict]:
    """긴 구간을 max_duration 단위로 분할한다.

    Args:
        segments: [{"start": float, "end": float}, ...] (초 단위)
        max_duration: 이 값을 초과하면 분할한다 (초)
    """
    result = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        if duration <= max_duration:
            result.append(seg.copy())
        else:
            t = seg["start"]
            while t < seg["end"]:
                end = min(t + max_duration, seg["end"])
                result.append({"start": t, "end": end})
                t = end
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_audio_preprocess.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/audio_preprocess.py tests/test_audio_preprocess.py
git commit -m "feat: add merge_speech_segments() and split_long_segments()"
```

---

### Task 3: 임베딩 추출 + 클러스터링 코어 로직

`diarizer.py`에 임베딩 추출과 클러스터링 함수를 추가한다. `run_diarization()`은 아직 건드리지 않는다.

**Files:**
- Modify: `src/diarizer.py`
- Modify: `tests/test_diarizer.py`

- [ ] **Step 1: Write failing tests for `_estimate_num_speakers()` and `_cluster_embeddings()`**

```python
# tests/test_diarizer.py에 추가

import numpy as np


class TestEstimateNumSpeakers:
    def test_single_speaker_high_similarity(self):
        """임베딩이 모두 유사하면 화자 1명으로 판정해야 한다."""
        from src.diarizer import _estimate_num_speakers
        # 유사한 임베딩 5개 (약간의 노이즈)
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
        # 두 개의 뚜렷한 클러스터
        center_a = rng.randn(256).astype(np.float32)
        center_b = -center_a  # 반대 방향 → cosine distance 최대
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
        # 3개 클러스터
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
        # 같은 클러스터의 세그먼트는 같은 speaker를 가져야 한다
        assert result[0]["speaker"] == result[1]["speaker"] == result[2]["speaker"]
        assert result[3]["speaker"] == result[4]["speaker"] == result[5]["speaker"]
        assert result[0]["speaker"] != result[3]["speaker"]
        # speaker 라벨 형식 확인
        assert result[0]["speaker"].startswith("SPEAKER_")

    def test_single_speaker(self):
        """num_speakers=1이면 모두 같은 화자여야 한다."""
        from src.diarizer import _cluster_embeddings
        embeddings = np.random.randn(5, 256).astype(np.float32)
        segments = [{"start": float(i), "end": float(i + 1)} for i in range(5)]

        result = _cluster_embeddings(embeddings, segments, num_speakers=1)
        speakers = set(r["speaker"] for r in result)
        assert len(speakers) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_diarizer.py::TestEstimateNumSpeakers tests/test_diarizer.py::TestClusterEmbeddings -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `_estimate_num_speakers()` and `_cluster_embeddings()`**

`src/diarizer.py`에서 기존 import 아래에 추가:

```python
import numpy as np


def _estimate_num_speakers(
    embeddings: np.ndarray,
    max_speakers: int = 10,
    min_speakers: int = 1,
    similarity_threshold: float = 0.7,
) -> int:
    """silhouette score 기반으로 최적 화자 수를 추정한다.

    Args:
        embeddings: (N, D) 형태의 임베딩 배열
        max_speakers: 탐색할 최대 화자 수
        min_speakers: 최소 화자 수
        similarity_threshold: k=1 판정용 cosine similarity 임계값

    Returns:
        추정된 화자 수
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(embeddings)
    if n <= 1:
        return 1

    # k=1 특수 처리: 모든 임베딩 간 cosine similarity가 높으면 화자 1명
    sim_matrix = cosine_similarity(embeddings)
    # 대각선 제외한 평균 유사도
    np.fill_diagonal(sim_matrix, 0)
    mean_sim = sim_matrix.sum() / (n * (n - 1))
    if mean_sim >= similarity_threshold:
        return 1

    # k=2~max_speakers까지 silhouette score 비교
    max_k = min(max_speakers, n)
    if max_k < 2:
        return 1

    best_k = 1
    best_score = -1.0

    for k in range(max(2, min_speakers), max_k + 1):
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        # 모든 라벨이 같으면 skip
        if len(set(labels)) < 2:
            continue

        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _cluster_embeddings(
    embeddings: np.ndarray,
    segments: list[dict],
    num_speakers: int,
) -> list[dict]:
    """임베딩을 클러스터링하여 각 세그먼트에 speaker 라벨을 할당한다.

    Args:
        embeddings: (N, D) 형태의 임베딩 배열
        segments: [{"start": float, "end": float}, ...] 세그먼트 리스트
        num_speakers: 화자 수

    Returns:
        [{"start": float, "end": float, "speaker": str}, ...] 형태의 리스트.
        speaker는 "SPEAKER_00", "SPEAKER_01", ... 형식.
    """
    from sklearn.cluster import AgglomerativeClustering

    n = len(embeddings)

    if num_speakers == 1 or n == 1:
        return [
            {"start": s["start"], "end": s["end"], "speaker": "SPEAKER_00"}
            for s in segments
        ]

    clustering = AgglomerativeClustering(
        n_clusters=num_speakers,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    return [
        {
            "start": segments[i]["start"],
            "end": segments[i]["end"],
            "speaker": f"SPEAKER_{labels[i]:02d}",
        }
        for i in range(n)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_diarizer.py::TestEstimateNumSpeakers tests/test_diarizer.py::TestClusterEmbeddings -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/diarizer.py tests/test_diarizer.py
git commit -m "feat: add _estimate_num_speakers() and _cluster_embeddings()"
```

---

### Task 4: `run_diarization()` 전면 교체

기존 pyannote Pipeline 호출을 새 3단계 파이프라인으로 교체한다. 함수 시그니처와 반환 형식은 유지한다.

**Files:**
- Modify: `src/diarizer.py:28-272`

- [ ] **Step 1: Write failing test for new `run_diarization()` flow**

이 테스트는 전체 파이프라인을 mock으로 검증한다. `tests/test_diarizer.py`에 추가:

```python
from unittest.mock import patch, MagicMock


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
        call_args = mock_cluster.call_args
        assert call_args[1].get("num_speakers") == 2 or call_args[0][2] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_diarizer.py::TestRunDiarization -v`
Expected: FAIL (기존 `run_diarization`은 pyannote Pipeline을 사용하므로 mock 대상이 다름)

- [ ] **Step 3: Rewrite `run_diarization()`**

`src/diarizer.py`의 `run_diarization()` 함수 전체를 다음으로 교체한다. 기존 `_DIAR_STEPS`, `_DIAR_STEP_ORDER`, `DiarizationCancelled`은 새 단계에 맞게 수정한다:

```python
import gc
import os
import threading
import time
from collections import defaultdict

import numpy as np
import torch


# 화자 분석 내부 단계 → (한글 표시, 진행률 범위 내 비율)
_DIAR_STEPS = {
    "vad": ("음성 구간 검출", 0.15),
    "embeddings": ("화자 특징 추출", 0.55),
    "clustering": ("화자 클러스터링", 0.30),
}

_DIAR_STEP_ORDER = list(_DIAR_STEPS.keys())


class DiarizationCancelled(Exception):
    """사용자 취소 시 발생시키는 예외."""
    pass


def _extract_speech_segments(
    audio_path: str,
    log_callback=None,
) -> list[dict]:
    """Silero VAD + 병합/분할로 화자분리용 음성 구간을 추출한다."""
    import soundfile as sf
    from src.audio_preprocess import (
        get_speech_segments,
        merge_speech_segments,
        split_long_segments,
    )

    audio, sr = sf.read(audio_path, dtype="float32")
    if log_callback:
        log_callback(f"VAD 입력: {len(audio) / sr:.1f}초")

    segments = get_speech_segments(audio)
    if log_callback:
        log_callback(f"VAD 검출: {len(segments)}개 구간")

    segments = merge_speech_segments(segments, min_gap=0.5, min_duration=0.5)
    if log_callback:
        log_callback(f"병합 후: {len(segments)}개 구간")

    segments = split_long_segments(segments, max_duration=10.0)
    if log_callback:
        log_callback(f"분할 후: {len(segments)}개 구간")

    return segments


def _extract_embeddings(
    audio_path: str,
    segments: list[dict],
    hf_token: str,
    device: torch.device,
    batch_size: int = 32,
    log_callback=None,
    progress_callback=None,
    cancel_check=None,
) -> np.ndarray:
    """pyannote 임베딩 모델로 각 구간의 화자 특징 벡터를 추출한다.

    Returns:
        (N, D) 형태의 numpy 배열. N=세그먼트 수, D=임베딩 차원.
    """
    os.environ["HF_TOKEN"] = hf_token

    from pyannote.audio import Audio
    from pyannote.core import Segment
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

    if log_callback:
        log_callback("임베딩 모델 로드 중...")

    model = PretrainedSpeakerEmbedding(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        device=device,
    )
    audio_io = Audio(sample_rate=16000, mono="downmix")

    if log_callback:
        log_callback(f"임베딩 추출 시작: {len(segments)}개 구간, batch_size={batch_size}")

    all_embeddings = []
    for batch_start in range(0, len(segments), batch_size):
        if cancel_check and cancel_check():
            raise DiarizationCancelled("사용자가 화자 분석을 취소했습니다.")

        batch_segs = segments[batch_start:batch_start + batch_size]

        # 배치 내 가장 긴 구간에 맞춰 패딩
        waveforms = []
        for seg in batch_segs:
            waveform, _ = audio_io.crop(audio_path, Segment(seg["start"], seg["end"]))
            waveforms.append(waveform.squeeze(0))  # (num_samples,)

        max_len = max(w.shape[0] for w in waveforms)
        padded = torch.zeros(len(waveforms), 1, max_len)
        for i, w in enumerate(waveforms):
            padded[i, 0, :w.shape[0]] = w

        with torch.inference_mode():
            emb = model(padded.to(device))  # (batch_size, D)
        all_embeddings.append(emb)

        if progress_callback:
            done = min(batch_start + batch_size, len(segments))
            progress_callback(done, len(segments))

        if log_callback:
            done = min(batch_start + batch_size, len(segments))
            log_callback(f"임베딩 추출: {done}/{len(segments)}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.vstack(all_embeddings)


def run_diarization(
    audio_path: str,
    hf_token: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress_callback=None,
    cancel_check=None,
    log_callback=None,
    num_threads: int = 1,
) -> list[dict]:
    """Silero VAD + pyannote 임베딩 + 클러스터링으로 화자 분리 실행.

    결과는 [{start, end, speaker}, ...] 리스트.

    Args:
        progress_callback: (percent: int, message: str) 형태의 콜백.
            percent는 0~100 범위이며, 호출측에서 전체 진행률에 매핑해야 한다.
        cancel_check: () -> bool 형태. True를 반환하면 취소.
        log_callback: (message: str) 형태. 상세 로그 출력.
    """

    def _log(msg: str):
        if log_callback:
            log_callback(msg)

    def _progress(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    _progress(0, "화자 분석 시작...")
    diar_start_time = time.time()

    # ── 디바이스 및 배치 크기 설정 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"화자 분석 디바이스: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        _log(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        batch_size = 64 if gpu_mem >= 8 else (32 if gpu_mem >= 4 else 16)
    else:
        if num_threads > 1:
            torch.set_num_threads(num_threads)
            try:
                torch.set_num_interop_threads(min(num_threads, 4))
            except RuntimeError:
                pass
        batch_size = 8
        _log(f"CPU torch 스레드: {num_threads}")

    # ── Step 1: VAD — 음성 구간 검출 ──
    if cancel_check and cancel_check():
        raise RuntimeError("사용자가 화자 분석을 취소했습니다.")

    _progress(5, "화자 분석: 음성 구간 검출 중...")
    segments = _extract_speech_segments(audio_path, log_callback=_log)

    if not segments:
        _log("음성 구간이 검출되지 않았습니다.")
        _progress(100, "화자 분석 완료")
        return []

    _progress(15, f"화자 분석: {len(segments)}개 음성 구간 검출")

    # ── Step 2: 임베딩 추출 ──
    if cancel_check and cancel_check():
        raise RuntimeError("사용자가 화자 분석을 취소했습니다.")

    def _emb_progress(done: int, total: int):
        pct = 15 + int(55 * done / total)
        _progress(pct, f"화자 분석: 화자 특징 추출 [{done}/{total}]")

    embeddings = _extract_embeddings(
        audio_path, segments, hf_token,
        device=device,
        batch_size=batch_size,
        log_callback=_log,
        progress_callback=_emb_progress,
        cancel_check=cancel_check,
    )

    elapsed = time.time() - diar_start_time
    _log(f"임베딩 추출 완료 ({int(elapsed)}초)")

    # ── Step 3: 클러스터링 ──
    if cancel_check and cancel_check():
        raise RuntimeError("사용자가 화자 분석을 취소했습니다.")

    _progress(75, "화자 분석: 화자 클러스터링 중...")

    if num_speakers is not None:
        k = num_speakers
        _log(f"화자 수 지정: {k}명")
    else:
        effective_max = max_speakers if max_speakers is not None else 10
        effective_min = min_speakers if min_speakers is not None else 1
        k = _estimate_num_speakers(
            embeddings,
            max_speakers=effective_max,
            min_speakers=effective_min,
        )
        _log(f"화자 수 추정: {k}명 (silhouette 기반)")

    result = _cluster_embeddings(embeddings, segments, num_speakers=k)

    elapsed_total = time.time() - diar_start_time
    _log(f"화자 분석 완료: {len(result)}개 구간, {k}명 화자 ({int(elapsed_total)}초)")
    _progress(100, "화자 분석 완료")

    # GPU 메모리 해제
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _log("GPU 메모리 해제 완료")

    return result
```

- [ ] **Step 4: Run all diarizer tests**

Run: `python -m pytest tests/test_diarizer.py -v`
Expected: ALL PASS (TestAssignSpeakers, TestMapSpeakerLabels, TestEstimateNumSpeakers, TestClusterEmbeddings, TestRunDiarization)

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/diarizer.py tests/test_diarizer.py
git commit -m "feat: replace pyannote Pipeline with VAD + embedding + clustering pipeline"
```

---

### Task 5: 전체 통합 검증 및 정리

기존 `transcriber.py`에서의 호출이 깨지지 않는지 확인하고, 불필요한 코드를 정리한다.

**Files:**
- Verify: `src/transcriber.py:221-264` (변경 불필요 — 인터페이스 동일)
- Modify: `src/diarizer.py` (불필요한 기존 코드 제거)

- [ ] **Step 1: Verify transcriber.py compatibility**

`transcriber.py:244-253`에서 `run_diarization()`을 호출하는 코드를 읽고, 새 구현과 인터페이스가 일치하는지 확인한다:

```python
# transcriber.py에서의 호출 (변경 없이 동작해야 함):
diarization_segments = run_diarization(
    tmp_clean_wav, self.hf_token,
    num_speakers=self.num_speakers,
    min_speakers=self.min_speakers,
    max_speakers=self.max_speakers,
    progress_callback=_diar_progress,
    cancel_check=lambda: self._cancelled,
    log_callback=self._log,
    num_threads=self.diar_threads,
)
```

시그니처가 동일하므로 변경 불필요. 확인만 한다.

- [ ] **Step 2: Remove unused pyannote Pipeline imports and old `_DIAR_STEPS`**

`src/diarizer.py`에서 기존 코드가 제거되었는지 확인한다. Task 4에서 전체 교체했으므로 아래 항목이 없어야 한다:
- `from pyannote.audio import Pipeline` (run_diarization 내부)
- `Pipeline.from_pretrained(...)` 호출
- 기존 `_hook` 함수
- `pipeline_thread` 관련 heartbeat 로직

이미 Task 4에서 제거되었으면 변경 없음.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit (if any cleanup was needed)**

```bash
git add -u
git commit -m "chore: cleanup unused pyannote Pipeline references"
```

---

### Task 6: 수동 통합 테스트

실제 오디오 파일로 화자분리를 실행하여 정확도를 검증한다. 이 단계는 자동화된 테스트가 아닌 수동 확인이다.

- [ ] **Step 1: 1명 화자 테스트**

실제 1명 화자 오디오(또는 영상)로 앱을 실행하여 화자 수가 1명으로 검출되는지 확인한다.

- [ ] **Step 2: 2명 이상 화자 테스트**

실제 2명 이상 화자 오디오로 앱을 실행하여 화자 수가 올바르게 검출되는지 확인한다.

- [ ] **Step 3: num_speakers 직접 지정 테스트**

UI에서 "직접 지정" 모드로 화자 수를 지정하여 해당 수로 분리되는지 확인한다.

- [ ] **Step 4: 결과에 따라 similarity_threshold 등 파라미터 조정**

실제 테스트 결과에 따라 `_estimate_num_speakers()`의 `similarity_threshold` 값을 조정한다. 기본값 0.7이 적절하지 않으면 0.6~0.8 범위에서 최적값을 찾는다.
