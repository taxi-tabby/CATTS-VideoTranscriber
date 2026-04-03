import gc
import os
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
    """사용자 취소 시 hook에서 발생시키는 예외."""
    pass


# ── 화자분리 프로파일 ──
# 각 프로파일은 VAD, 세그먼트 후처리, 클러스터링 파라미터를 묶은 것.
DIAR_PROFILES = {
    "interview": {
        "label": "인터뷰/대화",
        "description": "깨끗한 음성 위주 (인터뷰, 팟캐스트, 강의)",
        # VAD
        "vad_threshold": 0.35,
        # 세그먼트 후처리
        "min_gap": 0.5,
        "min_duration": 0.5,
        "max_duration": 10.0,
        # 클러스터링
        "similarity_threshold": 0.7,
    },
    "noisy": {
        "label": "영상/영화/노래",
        "description": "배경음악, 효과음, 노래가 포함된 콘텐츠",
        # VAD — 더 엄격하게 음성만 검출
        "vad_threshold": 0.5,
        # 세그먼트 후처리 — 더 공격적으로 병합
        "min_gap": 1.0,
        "min_duration": 1.0,
        "max_duration": 15.0,
        # 클러스터링 — 같은 화자로 판정하는 기준을 낮춤
        "similarity_threshold": 0.55,
    },
}


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
    np.fill_diagonal(sim_matrix, 0)
    mean_sim = sim_matrix.sum() / (n * (n - 1))
    if mean_sim >= similarity_threshold:
        return 1

    # k=2~max_speakers까지 silhouette score 비교
    # silhouette_score는 k < n 일 때만 유효 (k=n이면 각 샘플이 자체 클러스터)
    max_k = min(max_speakers, n - 1)
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


def _extract_speech_segments(
    audio_path: str,
    profile: dict,
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

    segments = get_speech_segments(audio, threshold=profile["vad_threshold"])
    if log_callback:
        log_callback(f"VAD 검출: {len(segments)}개 구간")

    segments = merge_speech_segments(
        segments,
        min_gap=profile["min_gap"],
        min_duration=profile["min_duration"],
    )
    if log_callback:
        log_callback(f"병합 후: {len(segments)}개 구간")

    segments = split_long_segments(segments, max_duration=profile["max_duration"])
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
    profile_name: str = "interview",
) -> list[dict]:
    """Silero VAD + pyannote 임베딩 + 클러스터링으로 화자 분리 실행.

    결과는 [{start, end, speaker}, ...] 리스트.

    Args:
        progress_callback: (percent: int, message: str) 형태의 콜백.
            percent는 0~100 범위이며, 호출측에서 전체 진행률에 매핑해야 한다.
        cancel_check: () -> bool 형태. True를 반환하면 취소.
        log_callback: (message: str) 형태. 상세 로그 출력.
        profile_name: 화자분리 프로파일 ("interview" 또는 "noisy").
    """

    def _log(msg: str):
        if log_callback:
            log_callback(msg)

    def _progress(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    profile = DIAR_PROFILES.get(profile_name, DIAR_PROFILES["interview"])
    _log(f"화자분리 프로파일: {profile['label']}")

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
    segments = _extract_speech_segments(audio_path, profile=profile, log_callback=_log)

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
            similarity_threshold=profile["similarity_threshold"],
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


def _find_speaker_at(sorted_dsegs: list[dict], dstarts: list[float],
                      start: float, end: float) -> str | None:
    """주어진 시간 구간에서 가장 겹침이 큰 화자를 반환한다."""
    import bisect

    speaker_overlap: dict[str, float] = defaultdict(float)
    speaker_earliest: dict[str, float] = {}

    right = bisect.bisect_left(dstarts, end)
    for i in range(right - 1, -1, -1):
        dseg = sorted_dsegs[i]
        if dseg["end"] <= start:
            break
        overlap = min(end, dseg["end"]) - max(start, dseg["start"])
        if overlap > 0:
            speaker = dseg["speaker"]
            speaker_overlap[speaker] += overlap
            if speaker not in speaker_earliest or dseg["start"] < speaker_earliest[speaker]:
                speaker_earliest[speaker] = dseg["start"]

    if not speaker_overlap:
        return None
    return min(
        speaker_overlap,
        key=lambda s: (-speaker_overlap[s], speaker_earliest.get(s, float("inf"))),
    )


def assign_speakers(
    diarization_segments: list[dict],
    whisper_segments: list[dict],
) -> list[dict]:
    """단어 단위 화자 매칭으로 각 Whisper 세그먼트에 화자를 할당.

    word_timestamps가 있으면 단어별로 화자를 판정한 뒤
    세그먼트 내 다수결로 최종 화자를 결정한다.
    word_timestamps가 없으면 세그먼트 단위 가중 투표로 폴백한다.
    """
    import bisect

    if not diarization_segments:
        return [
            {"start": w["start"], "end": w["end"], "text": w["text"], "speaker": None}
            for w in whisper_segments
        ]

    sorted_dsegs = sorted(diarization_segments, key=lambda d: d["start"])
    dstarts = [d["start"] for d in sorted_dsegs]

    result = []
    for wseg in whisper_segments:
        words = wseg.get("words")
        if words:
            # 단어 단위 매칭: 각 단어의 시간 구간으로 화자 판정 → 다수결
            speaker_votes: dict[str, float] = defaultdict(float)
            for w in words:
                w_start = w.get("start", wseg["start"])
                w_end = w.get("end", wseg["end"])
                sp = _find_speaker_at(sorted_dsegs, dstarts, w_start, w_end)
                if sp:
                    # 단어 길이를 가중치로 사용 (긴 단어의 판정이 더 신뢰성 높음)
                    speaker_votes[sp] += (w_end - w_start)
            best_speaker = max(speaker_votes, key=speaker_votes.get) if speaker_votes else None
        else:
            # 폴백: 세그먼트 단위 가중 투표
            best_speaker = _find_speaker_at(sorted_dsegs, dstarts, wseg["start"], wseg["end"])

        result.append({
            "start": wseg["start"],
            "end": wseg["end"],
            "text": wseg["text"],
            "speaker": best_speaker,
        })

    return result


def map_speaker_labels(segments: list[dict]) -> list[dict]:
    """pyannote 라벨(SPEAKER_00)을 한글 라벨(화자 1)로 매핑. 첫 등장 순서 기준."""
    label_map = {}
    counter = 0

    result = []
    for seg in segments:
        raw = seg.get("speaker")
        if raw is None:
            mapped = None
        else:
            if raw not in label_map:
                counter += 1
                label_map[raw] = f"화자 {counter}"
            mapped = label_map[raw]

        result.append({**seg, "speaker": mapped})

    return result
