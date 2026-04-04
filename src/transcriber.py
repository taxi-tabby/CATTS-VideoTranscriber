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


def save_numpy_as_wav(audio: np.ndarray, wav_path: str, sample_rate: int = 16000) -> None:
    """float32 numpy 배열을 16-bit PCM WAV로 저장한다."""
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def get_video_duration(audio: np.ndarray) -> float:
    return len(audio) / 16000.0


SAMPLE_RATE = 16000
CHUNK_SECONDS = 30


def _force_release_ml_memory() -> None:
    """ML 모델/텐서 관련 메모리를 강제로 해제한다.

    Python/PyTorch는 del + gc.collect() 후에도 수 GB의 메모리를 유지한다.
    이는 PyTorch 내부 캐시, C 런타임의 힙 관리, 모듈 레벨 전역 변수 등 때문이다.

    이 함수는:
    1. torch 관련 캐시를 모두 비운다
    2. gc를 여러 차례 실행하여 순환 참조를 해제한다
    3. ML 라이브러리를 sys.modules에서 제거하여 모듈 레벨 참조를 끊는다
    4. OS에 메모리 반환을 요청한다
    """
    import sys

    gc.collect()
    gc.collect()

    # torch 캐시 해제
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    gc.collect()

    # ML 라이브러리를 sys.modules에서 제거 — 모듈 레벨 캐시/전역 변수 해제
    _unload_prefixes = (
        "whisper", "demucs", "openunmix",
        "pyannote", "speechbrain",
        "asteroid_filterbanks",
    )
    for mod_name in list(sys.modules.keys()):
        if any(mod_name.startswith(p) for p in _unload_prefixes):
            del sys.modules[mod_name]

    gc.collect()
    gc.collect()

    # OS에 미사용 힙 메모리 반환 요청
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.c_size_t(-1), ctypes.c_size_t(-1),
            )
        else:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
    except Exception:
        pass
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

    # 모델 메모리: 디스크 크기의 ~2.5배가 추론 시 실제 메모리 사용량
    # (모델 가중치 + 디코더 상태 + 추론 중 임시 텐서)
    model_mem_mb = int(model_mb * 2.5)
    # 시스템 여유분 3GB 확보 (OS, Qt GUI, 기타 프로세스)
    reserve_mb = 3072
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
        skip_seconds: float = 0.0,
        profile: str = "interview",
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
        self.skip_seconds = skip_seconds
        self.profile = profile
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
        tmp_clean_wav = None
        # 에러/취소 시 확실하게 해제하기 위해 대용량 객체를 추적한다.
        _heavy_refs = {}  # name → object, finally에서 일괄 삭제
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
                self.error.emit("사용자가 변환을 취소했습니다.")
                return

            # Step 2: Load audio + preprocess
            self.progress.emit(8 if use_diar else 10, f"[2/{step_total}] 오디오 로드 및 전처리 중...")
            self._log("오디오 로드 중...")
            audio = load_wav_as_numpy(tmp_wav)
            _heavy_refs["audio"] = audio
            duration = get_video_duration(audio)  # 원본 영상 길이 (결과용)
            self._log(f"오디오 길이: {duration:.1f}초")
            if self._cancelled:
                self.error.emit("사용자가 변환을 취소했습니다.")
                return

            from src.audio_preprocess import preprocess as preprocess_audio
            use_vocal_sep = self.profile == "noisy"
            if use_vocal_sep:
                self._log("오디오 전처리 중 (보컬 분리, 노이즈 제거, 트리밍)...")
            else:
                self._log("오디오 전처리 중 (노이즈 제거, 트리밍)...")
            audio, trim_offset_samples = preprocess_audio(
                audio,
                use_vocal_separation=use_vocal_sep,
                log_callback=self._log,
            )
            trim_offset_sec = trim_offset_samples / SAMPLE_RATE
            audio_duration = get_video_duration(audio)  # 트리밍 후 길이 (진행률용)
            self._log(f"전처리 완료 (트리밍 후: {audio_duration:.1f}초)")
            if self._cancelled:
                self.error.emit("사용자가 변환을 취소했습니다.")
                return

            # Step 2.5: Speaker diarization (optional)
            # 전처리된 오디오를 WAV로 저장하여 화자 분석에 전달
            # (노이즈 제거 후 오디오로 분석해야 화자 embedding 정확도 향상)
            diarization_segments = None
            tmp_clean_wav = None
            if use_diar:
                self.progress.emit(8, f"[3/{step_total}] 화자 분석 중...")
                self._log("화자 분리 모델 로드 중...")
                from src.diarizer import run_diarization

                # 전처리된 오디오를 임시 WAV로 저장
                tmp_clean_wav = os.path.join(
                    tempfile.gettempdir(),
                    f"vt_{os.path.basename(self.video_path)}_clean.wav",
                )
                save_numpy_as_wav(audio, tmp_clean_wav)
                self._log("전처리된 오디오를 화자 분석에 사용")

                # diarizer의 0~100을 전체 진행률 8~18에 매핑
                def _diar_progress(pct: int, msg: str):
                    mapped = 8 + int(pct * 0.10)  # 8% ~ 18%
                    self.progress.emit(mapped, f"[3/{step_total}] {msg}")

                diarization_segments = run_diarization(
                    tmp_clean_wav, self.hf_token,
                    num_speakers=self.num_speakers,
                    min_speakers=self.min_speakers,
                    max_speakers=self.max_speakers,
                    progress_callback=_diar_progress,
                    cancel_check=lambda: self._cancelled,
                    log_callback=self._log,
                    num_threads=self.diar_threads,
                    profile_name=self.profile,
                )
                # diarization 타임스탬프에 트리밍 오프셋 적용
                # (전처리된 오디오 기준 → 원본 기준으로 보정)
                if trim_offset_sec > 0:
                    for dseg in diarization_segments:
                        dseg["start"] += trim_offset_sec
                        dseg["end"] += trim_offset_sec
                self._log(f"화자 분석 완료: {len(diarization_segments)}개 구간 검출")
                self.progress.emit(18, "화자 분석 완료")
                if self._cancelled:
                    self.error.emit("사용자가 변환을 취소했습니다.")
                    return

            # Step 3 (or 4 with diar): Load Whisper model
            model_pct = 18 if use_diar else 15
            model_step = 4 if use_diar else 3
            self.progress.emit(model_pct, f"[{model_step}/{step_total}] Whisper 모델 로드 중...")
            self._log(f"Whisper 모델 로드 중: {self.model_name}")
            from src.model_utils import get_whisper_cache_dir
            model = whisper.load_model(self.model_name, download_root=get_whisper_cache_dir())
            _heavy_refs["model"] = model
            self._log("Whisper 모델 로드 완료")
            if self._cancelled:
                self.error.emit("사용자가 변환을 취소했습니다.")
                return

            # Step 4: Transcribe in chunks
            start_time = time.time()
            total_samples = len(audio)
            total_chunks = (total_samples + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
            all_segments = []
            full_text_parts = []
            prev_text = ""

            # 이어하기: skip_seconds 이전 청크는 건너뜀
            skip_samples = int(self.skip_seconds * SAMPLE_RATE) if self.skip_seconds > 0 else 0
            skip_sec = self.skip_seconds  # 세그먼트 필터링용
            if skip_samples > 0:
                skipped = sum(1 for cs in range(0, total_samples, CHUNK_SAMPLES)
                              if min(cs + CHUNK_SAMPLES, total_samples) <= skip_samples)
                self._log(f"이어하기: {self.skip_seconds:.1f}초 이전 청크 {skipped}개 건너뜀")
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
                self._log("멀티스레드 모드: 청크 간 문맥 연결 없이 독립 변환 (속도↑, 경계부 정확도↓)")

                # 워커별 독립 모델 복사본 생성 (PyTorch 모델은 thread-safe하지 않음)
                _thread_local = threading.local()
                # torch.set_num_threads()는 프로세스 전역이므로 풀 생성 전에 한 번만 호출
                _orig_threads = _torch.get_num_threads()
                _torch_threads_per_worker = max(1, (_orig_threads // effective_workers))
                _torch.set_num_threads(_torch_threads_per_worker)
                self._log(f"torch 스레드: {_orig_threads} → {_torch_threads_per_worker} (워커 {effective_workers}개)")

                # 워커 모델 사전 생성 후 원본 즉시 해제 (메모리 절감)
                _worker_models = [model]  # 원본을 첫 번째 워커에 할당
                _heavy_refs["_worker_models"] = _worker_models
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
                                raise RuntimeError(
                                    f"워커 스레드가 모델 수({len(_worker_models)})를 초과했습니다. "
                                    f"동일 모델을 공유하면 결과가 손상되므로 중단합니다."
                                )
                            _model_idx[0] += 1
                            _thread_local.model = _worker_models[idx]
                    return _thread_local.model.transcribe(
                        chunk_audio, language=lang_arg, verbose=False,
                        word_timestamps=use_diar,
                    )

                # 청크 메타데이터만 사전 준비 (오디오 데이터는 필요 시 슬라이스)
                chunk_metas = []
                for chunk_start in range(0, total_samples, CHUNK_SAMPLES):
                    chunk_end = min(chunk_start + CHUNK_SAMPLES, total_samples)
                    # 이어하기: 이미 변환된 청크 건너뜀
                    if skip_samples > 0 and chunk_end <= skip_samples:
                        continue
                    chunk_metas.append((chunk_start, chunk_end))

                total_chunks = len(chunk_metas)  # 스킵 후 실제 처리할 청크 수
                results_buf = {}
                completed = 0
                failed_chunks = []

                # 메모리 안전 임계값: 가용 메모리가 이 이하로 떨어지면 submit 보류
                _MEM_SAFETY_MB = 1024

                try:
                    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                        pending = {}  # future -> (chunk_idx, retry_count)
                        next_submit = 0  # 다음 submit할 청크 인덱스
                        max_pending = effective_workers * 2  # 동시 pending 상한

                        def _submit_chunk(idx):
                            cs, ce = chunk_metas[idx]
                            chunk_audio = audio[cs:ce].copy()
                            f = pool.submit(_transcribe_chunk, chunk_audio)
                            pending[f] = (idx, 0)

                        # 초기 제출: 워커 수 × 2개만 먼저 submit
                        while next_submit < total_chunks and len(pending) < max_pending:
                            _submit_chunk(next_submit)
                            next_submit += 1

                        while pending:
                            if self._cancelled:
                                # 취소: 대기 중인 future만 취소, 이미 완료된 결과는 보존
                                for f in pending:
                                    f.cancel()
                                # 이미 완료된 future의 결과 수거
                                for f, (cidx, _) in list(pending.items()):
                                    if f.done() and not f.cancelled():
                                        try:
                                            results_buf[cidx] = f.result()
                                            completed += 1
                                        except Exception:
                                            pass
                                self._log(f"취소됨 — {completed}/{total_chunks} 청크 완료분 보존")
                                break

                            done_futures = [f for f in pending if f.done()]
                            if not done_futures:
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
                                        cs, ce = chunk_metas[chunk_idx]
                                        chunk_audio = audio[cs:ce].copy()
                                        f = pool.submit(_transcribe_chunk, chunk_audio)
                                        pending[f] = (chunk_idx, retry_count)
                                        continue
                                    else:
                                        self._log(f"청크 {chunk_idx + 1}/{total_chunks} 최종 실패: {chunk_err}")
                                        failed_chunks.append(chunk_idx + 1)
                                        result = {"text": "", "segments": []}

                                results_buf[chunk_idx] = result
                                completed += 1

                                # 주기적 GC: 임시 텐서 해제 (매번은 오버헤드)
                                if completed % 5 == 0:
                                    gc.collect()

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

                            # 슬라이딩 윈도우: 빈 슬롯이 생기면 다음 청크 submit
                            while next_submit < total_chunks and len(pending) < max_pending:
                                avail = _get_available_memory_mb()
                                if avail > 0 and avail < _MEM_SAFETY_MB:
                                    self._log(f"메모리 부족 ({avail}MB) — 추가 submit 보류")
                                    break
                                _submit_chunk(next_submit)
                                next_submit += 1
                finally:
                    _torch.set_num_threads(_orig_threads)

                # 원본 오디오 배열 해제 (finally에서 _force_release_ml_memory 호출됨)
                del audio
                del _worker_models
                del _thread_local
                self._log("워커 모델 메모리 해제 완료")

                if failed_chunks:
                    self._log(f"⚠ {len(failed_chunks)}개 청크 변환 실패 (청크 번호: {failed_chunks})")

                # 순서대로 결과 처리 (pool 종료 후 — 취소 시에도 완료분 처리)
                for idx in range(total_chunks):
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
                        # 단어 단위 타임스탬프 전달 (화자 매칭 정확도 향상용)
                        if seg.get("words"):
                            adjusted["words"] = [
                                {"start": w["start"] + time_offset + trim_offset_sec,
                                 "end": w["end"] + time_offset + trim_offset_sec,
                                 "word": w.get("word", "")}
                                for w in seg["words"]
                            ]
                        if adjusted["text"] and adjusted["end"] > skip_sec:
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
                # 이어하기: 스킵 후 실제 처리할 청크 목록 생성
                single_chunks = []
                for chunk_start in range(0, total_samples, CHUNK_SAMPLES):
                    chunk_end = min(chunk_start + CHUNK_SAMPLES, total_samples)
                    if skip_samples > 0 and chunk_end <= skip_samples:
                        continue
                    single_chunks.append((chunk_start, chunk_end))
                total_chunks = len(single_chunks)

                self._log(f"텍스트 변환 시작 (총 {total_chunks}개 청크, 언어: {self.language})")

                for chunk_idx, (chunk_start, chunk_end) in enumerate(single_chunks):
                    if self._cancelled:
                        self._log(f"취소됨 — {chunk_idx}/{total_chunks} 청크 완료분 보존")
                        break

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
                        word_timestamps=use_diar,
                    )
                    if self._cancelled:
                        self._log(f"취소됨 — {chunk_idx + 1}/{total_chunks} 청크 완료분 보존")
                        # 현재 청크 결과는 저장하고 루프 탈출
                        chunk_text = result.get("text", "").strip()
                        if chunk_text:
                            full_text_parts.append(chunk_text)
                        for seg in result.get("segments", []):
                            adjusted = {
                                "start": seg["start"] + time_offset + trim_offset_sec,
                                "end": seg["end"] + time_offset + trim_offset_sec,
                                "text": seg["text"].strip(),
                            }
                            if adjusted["text"]:
                                all_segments.append(adjusted)
                        break

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
                        if seg.get("words"):
                            adjusted["words"] = [
                                {"start": w["start"] + time_offset + trim_offset_sec,
                                 "end": w["end"] + time_offset + trim_offset_sec,
                                 "word": w.get("word", "")}
                                for w in seg["words"]
                            ]
                        if adjusted["text"] and adjusted["end"] > skip_sec:
                            all_segments.append(adjusted)

                    # 이 청크의 세그먼트에 화자 매칭
                    if diarization_segments:
                        from src.diarizer import assign_speakers
                        chunk_segs = all_segments[chunk_seg_start:]
                        matched = assign_speakers(diarization_segments, chunk_segs)
                        all_segments[chunk_seg_start:] = matched

                    for seg in all_segments[chunk_seg_start:]:
                        self.segment_ready.emit(seg)

                # 싱글스레드 완료 후 모델/오디오 해제 (finally에서 _force_release_ml_memory 호출됨)
                del model
                del audio

            # 최종 한글 라벨 매핑
            if diarization_segments:
                from src.diarizer import map_speaker_labels
                all_segments = map_speaker_labels(all_segments)

            elapsed = time.time() - start_time

            if self._cancelled and all_segments:
                # 취소되었지만 완료된 결과가 있으면 부분 결과로 저장
                self._log(f"취소됨 — 부분 결과 저장: {len(all_segments)}개 세그먼트, 소요시간: {elapsed:.1f}초")
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
                return

            if self._cancelled:
                self._log("취소됨 — 저장할 결과 없음")
                self.error.emit("사용자가 변환을 취소했습니다.")
                return

            self._log(f"변환 완료! 총 {len(all_segments)}개 세그먼트, 소요시간: {elapsed:.1f}초")
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
            _heavy_refs.clear()
            _force_release_ml_memory()
            self._log("메모리 정리 완료")

            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass
            if tmp_clean_wav and os.path.exists(tmp_clean_wav):
                try:
                    os.remove(tmp_clean_wav)
                except OSError:
                    pass
