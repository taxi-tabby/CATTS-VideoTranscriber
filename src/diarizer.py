import gc
import torch


def run_diarization(audio_path: str, hf_token: str) -> list[dict]:
    """pyannote.audio로 화자 분리 실행. 결과는 [{start, end, speaker}, ...] 리스트."""
    from pyannote.audio import Pipeline

    # pyannote가 import된 후 로컬 바인딩도 패치
    from src.torchaudio_compat import patch_huggingface_hub
    patch_huggingface_hub()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
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

    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

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
    """최대 겹침 방식으로 각 Whisper 세그먼트에 화자를 할당."""
    result = []
    for wseg in whisper_segments:
        best_speaker = None
        best_overlap = 0.0

        for dseg in diarization_segments:
            overlap_start = max(wseg["start"], dseg["start"])
            overlap_end = min(wseg["end"], dseg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg["speaker"]

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
