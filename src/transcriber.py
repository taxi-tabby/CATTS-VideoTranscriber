import os
import re
import subprocess
import tempfile
import time
import wave

import imageio_ffmpeg
import numpy as np
import whisper
from PySide6.QtCore import QObject, Signal, QThread

# Whisper 특수 토큰 패턴 (예: <|ro|>, <|en|>, <|0.00|> 등)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def get_ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path: str, audio_path: str, ffmpeg_path: str) -> None:
    cmd = [
        ffmpeg_path, "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 오류: {r.stderr[-500:]}")


def load_wav_as_numpy(audio_path: str) -> np.ndarray:
    with wave.open(audio_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def get_video_duration(audio: np.ndarray) -> float:
    return len(audio) / 16000.0


SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE


class TranscriberWorker(QObject):
    """Whisper 변환 워커. QThread에서 실행."""

    progress = Signal(int, str)       # (percent, status_message)
    segment_ready = Signal(dict)      # individual segment for real-time display
    finished = Signal(dict)           # transcription result
    error = Signal(str)               # error message

    def __init__(
        self,
        video_path: str,
        use_diarization: bool = False,
        hf_token: str | None = None,
        model_name: str = "medium",
        language: str = "ko",
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
        super().__init__()
        self.video_path = video_path
        self.use_diarization = use_diarization
        self.hf_token = hf_token
        self.model_name = model_name
        self.language = language
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        tmp_wav = None
        try:
            ffmpeg = get_ffmpeg_exe()
            use_diar = self.use_diarization and self.hf_token

            # Step 1: Extract audio
            step_total = 5 if use_diar else 4
            self.progress.emit(5, f"[1/{step_total}] 음성 추출 중...")
            tmp_wav = os.path.join(
                tempfile.gettempdir(),
                f"vt_{os.path.basename(self.video_path)}.wav",
            )
            extract_audio(self.video_path, tmp_wav, ffmpeg)
            if self._cancelled:
                return

            # Step 2: Load audio
            self.progress.emit(8 if use_diar else 10, f"[2/{step_total}] 오디오 로드 중...")
            audio = load_wav_as_numpy(tmp_wav)
            duration = get_video_duration(audio)
            if self._cancelled:
                return

            # Step 2.5: Speaker diarization (optional)
            diarization_segments = None
            if use_diar:
                self.progress.emit(8, f"[3/{step_total}] 화자 분석 중...")
                from src.diarizer import run_diarization
                diarization_segments = run_diarization(
                    tmp_wav, self.hf_token,
                    num_speakers=self.num_speakers,
                    min_speakers=self.min_speakers,
                    max_speakers=self.max_speakers,
                )
                self.progress.emit(18, "화자 분석 완료")
                if self._cancelled:
                    return

            # Step 3 (or 4 with diar): Load Whisper model
            model_pct = 18 if use_diar else 15
            model_step = 4 if use_diar else 3
            self.progress.emit(model_pct, f"[{model_step}/{step_total}] Whisper 모델 로드 중...")
            model = whisper.load_model(self.model_name)
            if self._cancelled:
                return

            # Step 4: Transcribe in chunks
            start_time = time.time()
            total_samples = len(audio)
            all_segments = []
            full_text_parts = []
            prev_text = ""
            transcribe_start = 22 if use_diar else 20

            for chunk_start in range(0, total_samples, CHUNK_SAMPLES):
                if self._cancelled:
                    return

                chunk_end = min(chunk_start + CHUNK_SAMPLES, total_samples)
                chunk = audio[chunk_start:chunk_end]

                time_offset = chunk_start / SAMPLE_RATE
                processed_sec = min(chunk_end / SAMPLE_RATE, duration)

                pct = transcribe_start + int((processed_sec / duration) * (95 - transcribe_start))
                pct = min(pct, 95)

                elapsed = time.time() - start_time
                transcribe_step = step_total
                if processed_sec > 0 and elapsed > 1:
                    eta = elapsed * (duration - processed_sec) / processed_sec
                    eta_min, eta_sec = divmod(int(eta), 60)
                    eta_str = f" (남은 시간: {eta_min}분 {eta_sec}초)" if eta_min > 0 else f" (남은 시간: {eta_sec}초)"
                else:
                    eta_str = ""
                self.progress.emit(pct, f"[{transcribe_step}/{step_total}] 변환 중... {processed_sec:.0f}s / {duration:.0f}s{eta_str}")

                prompt = prev_text[-200:] if prev_text else None
                if prompt:
                    prompt = _SPECIAL_TOKEN_RE.sub("", prompt).strip() or None

                result = model.transcribe(
                    chunk, language=self.language if self.language != "auto" else None, verbose=False,
                    initial_prompt=prompt,
                )

                chunk_text = result.get("text", "").strip()
                if chunk_text:
                    full_text_parts.append(chunk_text)
                    prev_text = chunk_text

                chunk_seg_start = len(all_segments)
                for seg in result.get("segments", []):
                    adjusted = {
                        "start": seg["start"] + time_offset,
                        "end": seg["end"] + time_offset,
                        "text": seg["text"].strip(),
                    }
                    if adjusted["text"]:
                        all_segments.append(adjusted)

                # 이 청크의 세그먼트에 화자 매칭
                if diarization_segments:
                    from src.diarizer import assign_speakers
                    chunk_segs = all_segments[chunk_seg_start:]
                    matched = assign_speakers(diarization_segments, chunk_segs)
                    all_segments[chunk_seg_start:] = matched

                for seg in all_segments[chunk_seg_start:]:
                    self.segment_ready.emit(seg)

            # 최종 한글 라벨 매핑
            if diarization_segments:
                from src.diarizer import map_speaker_labels
                all_segments = map_speaker_labels(all_segments)

            elapsed = time.time() - start_time

            if self._cancelled:
                return

            self.progress.emit(100, "완료!")

            self.finished.emit({
                "filename": os.path.basename(self.video_path),
                "filepath": self.video_path,
                "duration": duration,
                "full_text": " ".join(full_text_parts),
                "segments": all_segments,
                "elapsed": elapsed,
            })

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass
