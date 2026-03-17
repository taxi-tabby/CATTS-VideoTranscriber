import gc
import os
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


def run_diarization(
    audio_path: str,
    hf_token: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress_callback=None,
) -> list[dict]:
    """pyannote.audio로 화자 분리 실행. 결과는 [{start, end, speaker}, ...] 리스트.

    Args:
        progress_callback: (percent: int, message: str) 형태의 콜백.
            percent는 0~100 범위이며, 호출측에서 전체 진행률에 매핑해야 한다.
    """

    # HF_TOKEN 환경변수로 토큰 전달 — huggingface_hub가 자동으로 읽음.
    os.environ["HF_TOKEN"] = hf_token

    if progress_callback:
        progress_callback(0, "화자 분리 모델 로드 중...")

    from pyannote.audio import Pipeline

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

    # pyannote hook으로 진행률 보고
    completed_pct = 10  # 모델 로드 후 시작점
    diar_start_time = time.time()
    last_step_idx = -1

    def _hook(step_name, *args, **kwargs):
        nonlocal completed_pct, last_step_idx
        step_info = _DIAR_STEPS.get(step_name)
        if not step_info or not progress_callback:
            return
        label, ratio = step_info
        step_idx = _DIAR_STEP_ORDER.index(step_name)
        elapsed = time.time() - diar_start_time
        elapsed_str = f" ({int(elapsed)}초 경과)"

        # 이 단계가 완료됨 → 다음 단계 시작 메시지도 함께 전달
        completed_pct = 10 + int(90 * sum(
            r for name, (_, r) in _DIAR_STEPS.items()
            if _DIAR_STEP_ORDER.index(name) <= step_idx
        ))
        completed_pct = min(completed_pct, 95)
        progress_callback(completed_pct, f"화자 분석: {label} 완료{elapsed_str}")

        # 다음 단계가 있으면 "시작 중" 메시지 전송
        next_idx = step_idx + 1
        if next_idx < len(_DIAR_STEP_ORDER):
            next_step = _DIAR_STEP_ORDER[next_idx]
            next_label = _DIAR_STEPS[next_step][0]
            progress_callback(completed_pct, f"화자 분석: {next_label} 처리 중...{elapsed_str}")

        last_step_idx = step_idx

    kwargs["hook"] = _hook
    diarization = pipeline(audio_path, **kwargs)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    if progress_callback:
        progress_callback(100, "화자 분석 완료")

    # 모델 해제 — GPU 메모리 확보
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
