import gc
import os
import threading
import time
from collections import defaultdict

import torch


# 화자 분석 내부 단계 → (한글 표시, 진행률 범위 내 비율)
_DIAR_STEPS = {
    "segmentation": ("음성 구간 분할", 0.3),
    "speaker_counting": ("화자 수 추정", 0.1),
    "embeddings": ("화자 특징 추출", 0.4),
    "discrete_diarization": ("화자 할당", 0.2),
}

_DIAR_STEP_ORDER = list(_DIAR_STEPS.keys())

# 화자 분석 최대 시간 (초). 이 시간을 초과하면 타임아웃 오류 발생.
DIARIZATION_TIMEOUT = 1800  # 30분


class DiarizationCancelled(Exception):
    """사용자 취소 시 hook에서 발생시키는 예외."""
    pass


def run_diarization(
    audio_path: str,
    hf_token: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress_callback=None,
    cancel_check=None,
    log_callback=None,
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
    step_lock = threading.Lock()

    def _hook(step_name, *args, **kw):
        nonlocal completed_pct, current_step_name

        # 취소 체크 — hook은 단계 사이에 호출되므로 여기서 취소 가능
        if cancel_check and cancel_check():
            raise DiarizationCancelled("사용자가 화자 분석을 취소했습니다.")

        step_info = _DIAR_STEPS.get(step_name)
        if not step_info or not progress_callback:
            return
        label, ratio = step_info
        step_idx = _DIAR_STEP_ORDER.index(step_name)
        elapsed = time.time() - diar_start_time
        elapsed_str = f" ({int(elapsed)}초 경과)"

        completed_pct = 10 + int(90 * sum(
            r for name, (_, r) in _DIAR_STEPS.items()
            if _DIAR_STEP_ORDER.index(name) <= step_idx
        ))
        completed_pct = min(completed_pct, 95)
        progress_callback(completed_pct, f"화자 분석: {label} 완료{elapsed_str}")
        _log(f"화자 분석 단계 완료: {label} ({int(elapsed)}초)")

        # GPU 메모리 상태 로깅
        if device.type == "cuda":
            mem_used = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            _log(f"  GPU 메모리: {mem_used:.2f}GB 사용 / {mem_reserved:.2f}GB 예약")

        # 다음 단계 시작 메시지
        next_idx = step_idx + 1
        if next_idx < len(_DIAR_STEP_ORDER):
            next_step = _DIAR_STEP_ORDER[next_idx]
            next_label = _DIAR_STEPS[next_step][0]
            with step_lock:
                current_step_name = next_label
            progress_callback(completed_pct, f"화자 분석: {next_label} 처리 중...{elapsed_str}")
            _log(f"화자 분석 단계 시작: {next_label}")

    kwargs["hook"] = _hook

    # ── pipeline을 별도 스레드에서 실행하여 heartbeat + 취소 + 타임아웃 구현 ──
    result_container = [None]
    error_container = [None]
    done_event = threading.Event()

    def _run_pipeline():
        try:
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

        # 타임아웃 체크
        if elapsed > DIARIZATION_TIMEOUT:
            _log(f"화자 분석 타임아웃: {int(elapsed)}초 경과 (제한: {DIARIZATION_TIMEOUT}초)")
            raise RuntimeError(
                f"화자 분석이 {DIARIZATION_TIMEOUT // 60}분 이상 걸려 중단되었습니다.\n"
                "오디오가 너무 길거나 GPU 메모리가 부족할 수 있습니다.\n"
                "화자 분리 없이 변환하거나, 더 짧은 파일로 시도해 보세요."
            )

        # Heartbeat: 주기적으로 진행 상태 보고
        now = time.time()
        if now - last_heartbeat >= heartbeat_interval:
            last_heartbeat = now
            with step_lock:
                step = current_step_name
            elapsed_m, elapsed_s = divmod(int(elapsed), 60)
            if elapsed_m > 0:
                elapsed_str = f"{elapsed_m}분 {elapsed_s}초"
            else:
                elapsed_str = f"{elapsed_s}초"

            heartbeat_msg = f"화자 분석: {step} 처리 중... ({elapsed_str} 경과)"
            if progress_callback:
                progress_callback(completed_pct, heartbeat_msg)
            _log(f"[heartbeat] {step} 진행 중 ({elapsed_str})")

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


def assign_speakers(
    diarization_segments: list[dict],
    whisper_segments: list[dict],
) -> list[dict]:
    """가중 투표 방식으로 각 Whisper 세그먼트에 화자를 할당.

    같은 화자의 여러 diarization 세그먼트 겹침을 합산하고,
    동률 시 해당 구간에서 가장 이른 start를 가진 화자를 선택.
    """
    result = []
    for wseg in whisper_segments:
        speaker_overlap: dict[str, float] = defaultdict(float)
        speaker_earliest: dict[str, float] = {}

        for dseg in diarization_segments:
            overlap_start = max(wseg["start"], dseg["start"])
            overlap_end = min(wseg["end"], dseg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > 0:
                speaker = dseg["speaker"]
                speaker_overlap[speaker] += overlap
                if speaker not in speaker_earliest:
                    speaker_earliest[speaker] = dseg["start"]

        best_speaker = None
        if speaker_overlap:
            best_speaker = min(
                speaker_overlap,
                key=lambda s: (-speaker_overlap[s], speaker_earliest.get(s, float("inf"))),
            )

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
