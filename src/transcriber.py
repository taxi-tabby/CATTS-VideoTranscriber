import os
import subprocess
import tempfile
import time
import wave

import imageio_ffmpeg
import numpy as np
import whisper
from PySide6.QtCore import QObject, Signal, QThread


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


class TranscriberWorker(QObject):
    """Whisper 변환 워커. QThread에서 실행."""

    progress = Signal(int, str)       # (percent, status_message)
    finished = Signal(dict)           # transcription result
    error = Signal(str)               # error message

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        tmp_wav = None
        try:
            ffmpeg = get_ffmpeg_exe()

            # Step 1: Extract audio
            self.progress.emit(5, "음성 추출 중...")
            tmp_wav = os.path.join(
                tempfile.gettempdir(),
                f"vt_{os.path.basename(self.video_path)}.wav",
            )
            extract_audio(self.video_path, tmp_wav, ffmpeg)
            if self._cancelled:
                return

            # Step 2: Load audio
            self.progress.emit(10, "오디오 로드 중...")
            audio = load_wav_as_numpy(tmp_wav)
            duration = get_video_duration(audio)
            if self._cancelled:
                return

            # Step 3: Load model
            self.progress.emit(15, "Whisper 모델 로드 중... (첫 실행 시 다운로드)")
            model = whisper.load_model("medium")
            if self._cancelled:
                return

            # Step 4: Transcribe
            self.progress.emit(20, "텍스트 변환 중...")
            start_time = time.time()
            result = model.transcribe(audio, language="ko", verbose=False)
            elapsed = time.time() - start_time

            if self._cancelled:
                return

            self.progress.emit(100, "완료!")

            segments = []
            for seg in result.get("segments", []):
                segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                })

            self.finished.emit({
                "filename": os.path.basename(self.video_path),
                "filepath": self.video_path,
                "duration": duration,
                "full_text": result.get("text", "").strip(),
                "segments": segments,
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
