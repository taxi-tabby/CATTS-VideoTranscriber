import copy
import gc
import os
import re
import subprocess
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor

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
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        stderr = r.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg 오류: {stderr[-500:]}")


def load_wav_as_numpy(audio_path: str) -> np.ndarray:
    with wave.open(audio_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def get_video_duration(audio: np.ndarray) -> float:
    return len(audio) / 16000.0


SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE


def _get_available_memory_mb() -> int:
    """시스템 가용 메모리를 MB 단위로 반환. 측정 불가 시 0 반환."""
    try:
        import psutil
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except ImportError:
        pass
    # psutil 없을 때 플랫폼별 fallback
    import sys
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullAvailPhys / (1024 * 1024))
        else:
            # Linux / macOS
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def _cap_workers_by_memory(requested_workers: int, model_mb: int, log_fn) -> int:
    """가용 메모리를 확인하여 모델 복사본 수를 안전하게 제한한다."""
    avail_mb = _get_available_memory_mb()
    if avail_mb <= 0:
        # 측정 불가 시 보수적으로 2개까지만 허용
        capped = min(requested_workers, 2)
        if capped < requested_workers:
            log_fn(f"메모리 측정 불가 — 워커 수 제한: {requested_workers} → {capped}")
        return capped

    # 모델 메모리 (디스크 크기의 ~1.5배가 실제 메모리 사용량)
    model_mem_mb = int(model_mb * 1.5)
    # 시스템 여유분 2GB 확보
    reserve_mb = 2048
    usable_mb = max(0, avail_mb - reserve_mb)
    max_copies = max(1, usable_mb // model_mem_mb)
    capped = min(requested_workers, max_copies)

    if capped < requested_workers:
        log_fn(f"메모리 제한으로 워커 축소: {requested_workers} → {capped} "
               f"(가용: {avail_mb}MB, 모델: ~{model_mem_mb}MB/개, 여유: {reserve_mb}MB)")
    else:
        log_fn(f"메모리 확인: {avail_mb}MB 가용 (모델 {capped}개 × ~{model_mem_mb}MB)")

    return capped


class TranscriberWorker(QObject):
    """Whisper 변환 워커. QThread에서 실행."""

    progress = Signal(int, str)       # (percent, status_message)
    log_message = Signal(str)         # detailed log for processing log tab
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
        whisper_workers: int = 1,
        diar_threads: int = 1,
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
        self.whisper_workers = whisper_workers
        self.diar_threads = diar_threads
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _log(self, msg: str):
        """타임스탬프 포함 로그 메시지 전송."""
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_message.emit(f"[{ts}] {msg}")

    def run(self):
        tmp_wav = None
        try:
            ffmpeg = get_ffmpeg_exe()
            use_diar = self.use_diarization and self.hf_token

            # Step 1: Extract audio
            step_total = 5 if use_diar else 4
            self.progress.emit(5, f"[1/{step_total}] 음성 추출 중...")
            self._log(f"음성 추출 시작: {os.path.basename(self.video_path)}")
            tmp_wav = os.path.join(
                tempfile.gettempdir(),
                f"vt_{os.path.basename(self.video_path)}.wav",
            )
            extract_audio(self.video_path, tmp_wav, ffmpeg)
            self._log("음성 추출 완료 (WAV 16kHz mono)")
            if self._cancelled:
                return

            # Step 2: Load audio + preprocess
            self.progress.emit(8 if use_diar else 10, f"[2/{step_total}] 오디오 로드 및 전처리 중...")
            self._log("오디오 로드 중...")
            audio = load_wav_as_numpy(tmp_wav)
            duration = get_video_duration(audio)  # 원본 영상 길이 (결과용)
            self._log(f"오디오 길이: {duration:.1f}초")
            if self._cancelled:
                return

            from src.audio_preprocess import preprocess as preprocess_audio
            self._log("오디오 전처리 중 (노이즈 제거, 트리밍)...")
            audio, trim_offset_samples = preprocess_audio(audio)
            trim_offset_sec = trim_offset_samples / SAMPLE_RATE
            audio_duration = get_video_duration(audio)  # 트리밍 후 길이 (진행률용)
            self._log(f"전처리 완료 (트리밍 후: {audio_duration:.1f}초)")
            if self._cancelled:
                return

            # Step 2.5: Speaker diarization (optional)
            diarization_segments = None
            if use_diar:
                self.progress.emit(8, f"[3/{step_total}] 화자 분석 중...")
                self._log("화자 분리 모델 로드 중...")
                from src.diarizer import run_diarization

                # diarizer의 0~100을 전체 진행률 8~18에 매핑
                def _diar_progress(pct: int, msg: str):
                    mapped = 8 + int(pct * 0.10)  # 8% ~ 18%
                    self.progress.emit(mapped, f"[3/{step_total}] {msg}")

                diarization_segments = run_diarization(
                    tmp_wav, self.hf_token,
                    num_speakers=self.num_speakers,
                    min_speakers=self.min_speakers,
                    max_speakers=self.max_speakers,
                    progress_callback=_diar_progress,
                    cancel_check=lambda: self._cancelled,
                    log_callback=self._log,
                    num_threads=self.diar_threads,
                )
                self._log(f"화자 분석 완료: {len(diarization_segments)}개 구간 검출")
                self.progress.emit(18, "화자 분석 완료")
                if self._cancelled:
                    return

            # Step 3 (or 4 with diar): Load Whisper model
            model_pct = 18 if use_diar else 15
            model_step = 4 if use_diar else 3
            self.progress.emit(model_pct, f"[{model_step}/{step_total}] Whisper 모델 로드 중...")
            self._log(f"Whisper 모델 로드 중: {self.model_name}")
            from src.model_utils import get_whisper_cache_dir
            model = whisper.load_model(self.model_name, download_root=get_whisper_cache_dir())
            self._log("Whisper 모델 로드 완료")
            if self._cancelled:
                return

            # Step 4: Transcribe in chunks
            start_time = time.time()
            total_samples = len(audio)
            total_chunks = (total_samples + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
            all_segments = []
            full_text_parts = []
            prev_text = ""
            transcribe_start = 22 if use_diar else 20
            transcribe_step = step_total
            lang_arg = self.language if self.language != "auto" else None

            # GPU 사용 시 멀티스레딩 비활성화 (CUDA가 이미 병렬 연산 처리)
            import torch as _torch
            effective_workers = self.whisper_workers
            if effective_workers > 1 and _torch.cuda.is_available():
                effective_workers = 1
                self._log("GPU 모드 — 멀티스레드 비활성화 (CUDA 연산 사용)")

            MAX_RETRIES = 3

            # 가용 메모리 기반 워커 수 제한 (모델당 ~3GB 복사 필요)
            if effective_workers > 1:
                from src.model_utils import MODEL_SIZES
                model_mb = MODEL_SIZES.get(self.model_name, 500)
                effective_workers = _cap_workers_by_memory(effective_workers, model_mb, self._log)

            if effective_workers > 1:
                # ── 멀티스레드: 청크 병렬 처리 (CPU 전용) ──
                self._log(f"텍스트 변환 시작 (총 {total_chunks}개 청크, 언어: {self.language}, 워커: {effective_workers})")

                # 워커별 독립 모델 복사본 생성 (PyTorch 모델은 thread-safe하지 않음)
                _thread_local = threading.local()
                # torch.set_num_threads()는 프로세스 전역이므로 풀 생성 전에 한 번만 호출
                _orig_threads = _torch.get_num_threads()
                _torch_threads_per_worker = max(1, (_orig_threads // effective_workers))
                _torch.set_num_threads(_torch_threads_per_worker)
                self._log(f"torch 스레드: {_orig_threads} → {_torch_threads_per_worker} (워커 {effective_workers}개)")

                # 워커 모델 사전 생성 후 원본 즉시 해제 (메모리 절감)
                _worker_models = [model]  # 원본을 첫 번째 워커에 할당
                for i in range(effective_workers - 1):
                    try:
                        _worker_models.append(copy.deepcopy(model))
                    except Exception as e:
                        self._log(f"모델 복사 실패 (워커 {i+2}/{effective_workers}): {e}")
                        self._log(f"워커 수를 {len(_worker_models)}개로 축소합니다")
                        effective_workers = len(_worker_models)
                        break
                del model
                gc.collect()
                self._log(f"워커 모델 {effective_workers}개 준비 완료 (원본 해제)")

                _model_lock = threading.Lock()
                _model_idx = [0]

                def _transcribe_chunk(chunk_audio):
                    if not hasattr(_thread_local, "model"):
                        with _model_lock:
                            idx = _model_idx[0]
                            if idx >= len(_worker_models):
                                idx = len(_worker_models) - 1
                                self._log(f"[경고] 모델 인덱스 범위 초과 — 마지막 모델로 폴백 (idx→{idx})")
                            else:
                                _model_idx[0] += 1
                            _thread_local.model = _worker_models[idx]
                    return _thread_local.model.transcribe(chunk_audio, language=lang_arg, verbose=False)

                # 청크 오디오 데이터 준비 (numpy 슬라이스 = view, 메모리 복사 없음)
                chunks_audio = []
                chunk_metas = []
                for chunk_start in range(0, total_samples, CHUNK_SAMPLES):
                    chunk_end = min(chunk_start + CHUNK_SAMPLES, total_samples)
                    chunks_audio.append(audio[chunk_start:chunk_end].copy())
                    chunk_metas.append((chunk_start, chunk_end))
                # 원본 오디오 배열 해제 (청크 복사본만 유지)
                del audio
                gc.collect()

                results_buf = {}
                completed = 0
                failed_chunks = []

                try:
                    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                        # 초기 제출
                        pending = {}  # future -> (chunk_idx, retry_count)
                        for idx, chunk_audio in enumerate(chunks_audio):
                            f = pool.submit(_transcribe_chunk, chunk_audio)
                            pending[f] = (idx, 0)

                        while pending:
                            if self._cancelled:
                                for f in pending:
                                    f.cancel()
                                return

                            done_futures = [f for f in pending if f.done()]
                            if not done_futures:
                                # 아직 완료된 게 없으면 짧게 대기 후 취소 확인
                                time.sleep(0.3)
                                continue

                            for future in done_futures:
                                chunk_idx, retry_count = pending.pop(future)

                                try:
                                    result = future.result()
                                except Exception as chunk_err:
                                    if retry_count < MAX_RETRIES:
                                        retry_count += 1
                                        self._log(f"청크 {chunk_idx + 1}/{total_chunks} 실패 (재시도 {retry_count}/{MAX_RETRIES}): {chunk_err}")
                                        f = pool.submit(_transcribe_chunk, chunks_audio[chunk_idx])
                                        pending[f] = (chunk_idx, retry_count)
                                        continue
                                    else:
                                        self._log(f"청크 {chunk_idx + 1}/{total_chunks} 최종 실패: {chunk_err}")
                                        failed_chunks.append(chunk_idx + 1)
                                        result = {"text": "", "segments": []}

                                results_buf[chunk_idx] = result
                                completed += 1

                                pct = transcribe_start + int((completed / total_chunks) * (95 - transcribe_start))
                                pct = min(pct, 95)
                                elapsed = time.time() - start_time
                                if completed > 0 and elapsed > 1:
                                    eta = elapsed * (total_chunks - completed) / completed
                                    eta_min, eta_sec = divmod(int(eta), 60)
                                    eta_str = f" (남은 시간: {eta_min}분 {eta_sec}초)" if eta_min > 0 else f" (남은 시간: {eta_sec}초)"
                                else:
                                    eta_str = ""
                                self.progress.emit(pct, f"[{transcribe_step}/{step_total}] 변환 중... {completed}/{total_chunks} 청크{eta_str}")
                                self._log(f"청크 {chunk_idx + 1}/{total_chunks} 완료")
                finally:
                    # 취소/예외 여부와 무관하게 torch 스레드 수 복원
                    _torch.set_num_threads(_orig_threads)

                # 워커 모델 메모리 해제 (pool 종료 후 thread-local은 접근 불가하므로 리스트로 정리)
                del _worker_models
                del _thread_local
                del chunks_audio
                gc.collect()
                self._log("워커 모델 메모리 해제 완료")

                if failed_chunks:
                    self._log(f"⚠ {len(failed_chunks)}개 청크 변환 실패 (청크 번호: {failed_chunks})")

                # 순서대로 결과 처리 (pool 종료 후)
                for idx in range(total_chunks):
                    if self._cancelled:
                        return
                    res = results_buf.get(idx)
                    if res is None:
                        continue
                    chunk_start, chunk_end = chunk_metas[idx]
                    time_offset = chunk_start / SAMPLE_RATE

                    chunk_text = res.get("text", "").strip()
                    if chunk_text:
                        full_text_parts.append(chunk_text)

                    chunk_seg_start = len(all_segments)
                    for seg in res.get("segments", []):
                        adjusted = {
                            "start": seg["start"] + time_offset + trim_offset_sec,
                            "end": seg["end"] + time_offset + trim_offset_sec,
                            "text": seg["text"].strip(),
                        }
                        if adjusted["text"]:
                            all_segments.append(adjusted)

                    if diarization_segments:
                        from src.diarizer import assign_speakers
                        chunk_segs = all_segments[chunk_seg_start:]
                        matched = assign_speakers(diarization_segments, chunk_segs)
                        all_segments[chunk_seg_start:] = matched

                    for seg in all_segments[chunk_seg_start:]:
                        self.segment_ready.emit(seg)
            else:
                # ── 싱글스레드: 순차 처리 (이전 청크 컨텍스트 활용) ──
                self._log(f"텍스트 변환 시작 (총 {total_chunks}개 청크, 언어: {self.language})")

                for chunk_idx, chunk_start in enumerate(range(0, total_samples, CHUNK_SAMPLES)):
                    if self._cancelled:
                        return

                    chunk_end = min(chunk_start + CHUNK_SAMPLES, total_samples)
                    chunk = audio[chunk_start:chunk_end]

                    time_offset = chunk_start / SAMPLE_RATE
                    processed_sec = min(chunk_end / SAMPLE_RATE, audio_duration)
                    self._log(f"청크 {chunk_idx + 1}/{total_chunks} 변환 중 ({time_offset:.0f}s ~ {processed_sec:.0f}s)")

                    pct = transcribe_start + int((processed_sec / audio_duration) * (95 - transcribe_start))
                    pct = min(pct, 95)

                    elapsed = time.time() - start_time
                    if processed_sec > 0 and elapsed > 1:
                        eta = elapsed * (audio_duration - processed_sec) / processed_sec
                        eta_min, eta_sec = divmod(int(eta), 60)
                        eta_str = f" (남은 시간: {eta_min}분 {eta_sec}초)" if eta_min > 0 else f" (남은 시간: {eta_sec}초)"
                    else:
                        eta_str = ""
                    self.progress.emit(pct, f"[{transcribe_step}/{step_total}] 변환 중... {processed_sec:.0f}s / {audio_duration:.0f}s{eta_str}")

                    prompt = prev_text[-200:] if prev_text else None
                    if prompt:
                        prompt = _SPECIAL_TOKEN_RE.sub("", prompt).strip() or None

                    result = model.transcribe(
                        chunk, language=lang_arg, verbose=False,
                        initial_prompt=prompt,
                    )
                    if self._cancelled:
                        return

                    chunk_text = result.get("text", "").strip()
                    if chunk_text:
                        full_text_parts.append(chunk_text)
                        prev_text = chunk_text

                    chunk_seg_start = len(all_segments)
                    for seg in result.get("segments", []):
                        adjusted = {
                            "start": seg["start"] + time_offset + trim_offset_sec,
                            "end": seg["end"] + time_offset + trim_offset_sec,
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

                # 싱글스레드 완료 후 모델/오디오 해제
                del model
                del audio
                gc.collect()

            # 최종 한글 라벨 매핑
            if diarization_segments:
                from src.diarizer import map_speaker_labels
                all_segments = map_speaker_labels(all_segments)

            elapsed = time.time() - start_time
            self._log(f"변환 완료! 총 {len(all_segments)}개 세그먼트, 소요시간: {elapsed:.1f}초")

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
                "model_name": self.model_name,
                "language": self.language,
            })

        except Exception as e:
            self._log(f"오류 발생: {e}")
            self.error.emit(str(e))
        finally:
            # ── 메모리 정리 ──
            # 예외/취소 등으로 정상 경로에서 해제하지 못한 대용량 객체 정리
            gc.collect()
            try:
                import torch as _torch_cleanup
                if _torch_cleanup.cuda.is_available():
                    _torch_cleanup.cuda.empty_cache()
            except Exception:
                pass

            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass
