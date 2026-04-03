import gc
import os
import threading
import time
from collections import defaultdict

import numpy as np
import torch


# 화자 분석 내부 단계 → (한글 표시, 진행률 범위 내 비율)
_DIAR_STEPS = {
    "segmentation": ("음성 구간 분할", 0.3),
    "speaker_counting": ("화자 수 추정", 0.1),
    "embeddings": ("화자 특징 추출", 0.4),
    "discrete_diarization": ("화자 할당", 0.2),
}

_DIAR_STEP_ORDER = list(_DIAR_STEPS.keys())

# 화자 분석 타임아웃 없음 — 오래 걸려도 완수해야 함.


class DiarizationCancelled(Exception):
    """사용자 취소 시 hook에서 발생시키는 예외."""
    pass


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
    """pyannote.audio로 화자 분리 실행. 결과는 [{start, end, speaker}, ...] 리스트.

    Args:
        progress_callback: (percent: int, message: str) 형태의 콜백.
            percent는 0~100 범위이며, 호출측에서 전체 진행률에 매핑해야 한다.
        cancel_check: () -> bool 형태. True를 반환하면 취소.
        log_callback: (message: str) 형태. 상세 로그 출력.
    """

    def _log(msg: str):
        if log_callback:
            log_callback(msg)

    # HF_TOKEN 환경변수로 토큰 전달 — huggingface_hub가 자동으로 읽음.
    os.environ["HF_TOKEN"] = hf_token

    if progress_callback:
        progress_callback(0, "화자 분리 모델 로드 중...")

    from pyannote.audio import Pipeline

    _log("pyannote Pipeline 로드 시작...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
    )

    if pipeline is None:
        raise RuntimeError(
            "화자 분리 모델을 불러올 수 없습니다.\n"
            "1. HuggingFace 토큰이 유효한지 확인하세요.\n"
            "2. https://hf.co/pyannote/speaker-diarization-3.1 에서 라이선스에 동의했는지 확인하세요.\n"
            "3. https://hf.co/pyannote/segmentation-3.0 에서도 라이선스에 동의해야 합니다."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"화자 분석 디바이스: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        _log(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        # GPU VRAM에 따라 batch_size 결정
        if gpu_mem >= 8:
            batch_size = 64
        elif gpu_mem >= 4:
            batch_size = 32
        else:
            batch_size = 16
    else:
        # CPU: 코어 수 기반 batch_size (너무 크면 메모리 부족)
        cpu_count = os.cpu_count() or 4
        if num_threads > 1:
            thread_count = num_threads
            batch_size = min(max(thread_count, 8), 32)
        else:
            thread_count = 1
            batch_size = 8
        torch.set_num_threads(thread_count)
        try:
            torch.set_num_interop_threads(min(thread_count, 4))
        except RuntimeError:
            pass  # 이미 설정된 경우 무시
        _log(f"CPU 코어: {cpu_count}개, torch 스레드: {thread_count}")

    # batch_size 적용
    try:
        pipeline.segmentation.batch_size = batch_size
        _log(f"segmentation batch_size = {batch_size}")
    except Exception:
        pass
    try:
        pipeline.embedding.batch_size = batch_size
        _log(f"embedding batch_size = {batch_size}")
    except Exception:
        pass

    pipeline.to(device)

    if progress_callback:
        progress_callback(10, "화자 분석 시작...")

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    # pyannote hook으로 진행률 보고 + 취소 체크
    completed_pct = 10
    diar_start_time = time.time()
    current_step_name = "시작"  # 현재 진행 중인 단계 (heartbeat용)
    current_chunk_info = ""  # 청크 진행 정보 (heartbeat용)
    step_lock = threading.Lock()

    def _hook(step_name, step_artefact, *args, completed=None, total=None, **kw):
        nonlocal completed_pct, current_step_name, current_chunk_info

        # 취소 체크 — hook은 단계 사이에 호출되므로 여기서 취소 가능
        if cancel_check and cancel_check():
            raise DiarizationCancelled("사용자가 화자 분석을 취소했습니다.")

        step_info = _DIAR_STEPS.get(step_name)
        if not step_info or not progress_callback:
            return
        label, ratio = step_info
        step_idx = _DIAR_STEP_ORDER.index(step_name)
        elapsed = time.time() - diar_start_time

        # 청크 진행률 계산
        chunk_str = ""
        if total and total > 0 and completed is not None:
            chunk_str = f" [{completed}/{total}]"

        # 이전 단계까지의 누적 + 현재 단계 내 청크 진행률 반영
        prev_pct = sum(
            r for name, (_, r) in _DIAR_STEPS.items()
            if _DIAR_STEP_ORDER.index(name) < step_idx
        )
        chunk_fraction = (completed / total) if (total and total > 0 and completed is not None) else 1.0
        completed_pct = 10 + int(90 * (prev_pct + ratio * chunk_fraction))
        completed_pct = min(completed_pct, 95)

        elapsed_str = f" ({int(elapsed)}초 경과)"
        with step_lock:
            current_step_name = label
            current_chunk_info = chunk_str

        progress_callback(completed_pct, f"화자 분석: {label}{chunk_str}{elapsed_str}")
        _log(f"화자 분석: {label}{chunk_str} ({int(elapsed)}초)")

        # GPU 메모리 상태 로깅
        if device.type == "cuda":
            mem_used = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            _log(f"  GPU 메모리: {mem_used:.2f}GB 사용 / {mem_reserved:.2f}GB 예약")

    kwargs["hook"] = _hook

    # ── pipeline을 별도 스레드에서 실행하여 heartbeat + 취소 + 타임아웃 구현 ──
    result_container = [None]
    error_container = [None]
    done_event = threading.Event()

    def _run_pipeline():
        try:
            with torch.inference_mode():
                result_container[0] = pipeline(audio_path, **kwargs)
        except DiarizationCancelled as e:
            error_container[0] = e
        except Exception as e:
            error_container[0] = e
        finally:
            done_event.set()

    pipeline_thread = threading.Thread(target=_run_pipeline, daemon=True)
    pipeline_thread.start()

    # 메인 루프: 주기적으로 heartbeat, 취소 체크, 타임아웃 체크
    heartbeat_interval = 10  # 10초마다 상태 보고
    last_heartbeat = time.time()

    while not done_event.wait(timeout=2):
        elapsed = time.time() - diar_start_time

        # 취소 체크
        if cancel_check and cancel_check():
            _log("사용자 취소 요청 감지 — 현재 단계 완료 후 중단됩니다.")
            # hook에서 예외가 발생할 때까지 대기 (최대 30초)
            done_event.wait(timeout=30)
            if not done_event.is_set():
                _log("경고: 화자 분석 스레드가 응답하지 않아 강제 중단합니다.")
            raise RuntimeError("사용자가 화자 분석을 취소했습니다.")

        # Heartbeat: 주기적으로 진행 상태 보고
        now = time.time()
        if now - last_heartbeat >= heartbeat_interval:
            last_heartbeat = now
            with step_lock:
                step = current_step_name
                chunk_info = current_chunk_info
            elapsed_m, elapsed_s = divmod(int(elapsed), 60)
            if elapsed_m > 0:
                elapsed_str = f"{elapsed_m}분 {elapsed_s}초"
            else:
                elapsed_str = f"{elapsed_s}초"
            heartbeat_msg = f"화자 분석: {step}{chunk_info} 처리 중... ({elapsed_str} 경과)"
            if progress_callback:
                progress_callback(completed_pct, heartbeat_msg)
            _log(f"[heartbeat] {step}{chunk_info} 진행 중 ({elapsed_str})")

            # GPU 메모리 상태
            if device.type == "cuda":
                mem_used = torch.cuda.memory_allocated() / (1024 ** 3)
                _log(f"  GPU 메모리 사용: {mem_used:.2f}GB")

    # 완료 후 에러 체크
    if error_container[0] is not None:
        err = error_container[0]
        if isinstance(err, DiarizationCancelled):
            raise RuntimeError(str(err))
        raise err

    diarization = result_container[0]
    if diarization is None:
        raise RuntimeError("화자 분석 결과가 비어 있습니다.")

    elapsed_total = time.time() - diar_start_time
    _log(f"화자 분석 pipeline 완료 ({int(elapsed_total)}초)")

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    _log(f"화자 분석 결과: {len(segments)}개 구간, "
         f"{len(set(s['speaker'] for s in segments))}명 화자 검출")

    if progress_callback:
        progress_callback(100, "화자 분석 완료")

    # 모델 해제 — GPU 메모리 확보
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _log("GPU 메모리 해제 완료")

    return segments


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
