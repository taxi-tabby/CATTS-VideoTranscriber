import copy
import gc
import multiprocessing as mp
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


def _subprocess_worker(params: dict, msg_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """별도 프로세스에서 전처리 + 화자분리 + 전사를 실행한다.

    프로세스 종료 시 OS가 모든 메모리(모델, 텐서)를 회수한다.
    결과는 msg_queue를 통해 메인 프로세스에 전달한다.

    메시지 형식:
        ("progress", percent, message)
        ("log", message)
        ("segment", segment_dict)
        ("result", result_dict)
        ("error", error_message)
    """
    import gc
    import os
    import time

    def _log(msg):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        msg_queue.put(("log", f"[{ts}] {msg}"))

    def _progress(pct, msg):
        msg_queue.put(("progress", pct, msg))

    def _is_cancelled():
        return cancel_event.is_set()

    try:
        wav_path = params["wav_path"]
        use_diar = params["use_diar"]
        profile = params["profile"]
        model_name = params["model_name"]
        language = params["language"]
        skip_seconds = params["skip_seconds"]

        step_total = 5 if use_diar else 4

        # ── Step 2: 오디오 로드 + 전처리 ──
        _progress(8 if use_diar else 10, f"[2/{step_total}] 오디오 로드 및 전처리 중...")
        _log("오디오 로드 중...")
        audio = load_wav_as_numpy(wav_path)
        duration = get_video_duration(audio)
        _log(f"오디오 길이: {duration:.1f}초")

        if _is_cancelled():
            return

        from src.audio_preprocess import preprocess as preprocess_audio
        use_vocal_sep = profile == "noisy"
        if use_vocal_sep:
            _log("오디오 전처리 중 (보컬 분리, 노이즈 제거, 트리밍)...")
        else:
            _log("오디오 전처리 중 (노이즈 제거, 트리밍)...")
        audio, trim_offset_samples = preprocess_audio(
            audio,
            use_vocal_separation=use_vocal_sep,
            log_callback=_log,
        )
        trim_offset_sec = trim_offset_samples / SAMPLE_RATE
        audio_duration = get_video_duration(audio)
        _log(f"전처리 완료 (트리밍 후: {audio_duration:.1f}초)")

        # VAD로 음성 구간 검출 (청크 분할 + 화자분리 양쪽에서 사용)
        from src.audio_preprocess import get_speech_segments, build_vad_chunks
        speech_segments = get_speech_segments(audio)
        _log(f"VAD 음성 구간: {len(speech_segments)}개")

        if _is_cancelled():
            return

        # ── Step 2.5: 화자 분석 ──
        diarization_segments = None
        tmp_clean_wav = None
        if use_diar:
            _progress(8, f"[3/{step_total}] 화자 분석 중...")
            _log("화자 분리 모델 로드 중...")
            from src.diarizer import run_diarization

            tmp_clean_wav = os.path.join(
                tempfile.gettempdir(),
                f"vt_subprocess_clean.wav",
            )
            save_numpy_as_wav(audio, tmp_clean_wav)
            _log("전처리된 오디오를 화자 분석에 사용")

            def _diar_progress(pct, msg):
                mapped = 8 + int(pct * 0.10)
                _progress(mapped, f"[3/{step_total}] {msg}")

            diarization_segments = run_diarization(
                tmp_clean_wav, params["hf_token"],
                num_speakers=params.get("num_speakers"),
                min_speakers=params.get("min_speakers"),
                max_speakers=params.get("max_speakers"),
                progress_callback=_diar_progress,
                cancel_check=_is_cancelled,
                log_callback=_log,
                num_threads=params.get("diar_threads", 1),
                profile_name=profile,
            )
            if trim_offset_sec > 0:
                for dseg in diarization_segments:
                    dseg["start"] += trim_offset_sec
                    dseg["end"] += trim_offset_sec
            _log(f"화자 분석 완료: {len(diarization_segments)}개 구간 검출")
            _progress(18, "화자 분석 완료")

            if tmp_clean_wav and os.path.exists(tmp_clean_wav):
                try:
                    os.remove(tmp_clean_wav)
                except OSError:
                    pass

            if _is_cancelled():
                return

        # ── Step 3: Whisper 모델 로드 ──
        import whisper as _whisper
        model_pct = 18 if use_diar else 15
        model_step = 4 if use_diar else 3
        _progress(model_pct, f"[{model_step}/{step_total}] Whisper 모델 로드 중...")
        _log(f"Whisper 모델 로드 중: {model_name}")
        from src.model_utils import get_whisper_cache_dir
        model = _whisper.load_model(model_name, download_root=get_whisper_cache_dir())
        _log("Whisper 모델 로드 완료")

        if _is_cancelled():
            return

        # ── Step 4: 전사 (싱글스레드) ──
        start_time = time.time()
        total_samples = len(audio)
        all_segments = []
        full_text_parts = []
        prev_text = ""

        # 교정 사전 로드
        correction_entries = params.get("correction_entries") or []
        if correction_entries:
            # prompt 힌트: 올바른 단어들을 쉼표로 연결 (중복 제거, 400자 제한)
            correct_words = list(dict.fromkeys(e["correct"] for e in correction_entries))
            correction_hint = ", ".join(correct_words)[:400]
            _log(f"교정 사전 적용: {len(correction_entries)}개 항목 ({correction_hint[:80]}...)")
        else:
            correction_hint = ""

        skip_samples = int(skip_seconds * SAMPLE_RATE) if skip_seconds > 0 else 0
        skip_sec = skip_seconds

        transcribe_start = 22 if use_diar else 20
        transcribe_step = step_total
        lang_arg = language if language != "auto" else None

        # VAD 기반 지능형 청크 분할 (무음 지점에서만 분할 → 환각 방지)
        vad_chunks = build_vad_chunks(speech_segments, total_samples)
        single_chunks = []
        for vc in vad_chunks:
            cs, ce = vc["start_sample"], vc["end_sample"]
            if skip_samples > 0 and ce <= skip_samples:
                continue
            single_chunks.append((cs, ce))
        total_chunks = len(single_chunks)

        if skip_samples > 0:
            skipped = len(vad_chunks) - total_chunks
            _log(f"이어하기: {skip_seconds:.1f}초 이전 청크 {skipped}개 건너뜀")

        _log(f"텍스트 변환 시작 (총 {total_chunks}개 청크, 언어: {language})")

        for chunk_idx, (chunk_start, chunk_end) in enumerate(single_chunks):
            if _is_cancelled():
                _log(f"취소됨 — {chunk_idx}/{total_chunks} 청크 완료분 보존")
                break

            chunk = audio[chunk_start:chunk_end]
            time_offset = chunk_start / SAMPLE_RATE
            processed_sec = min(chunk_end / SAMPLE_RATE, audio_duration)
            _log(f"청크 {chunk_idx + 1}/{total_chunks} 변환 중 ({time_offset:.0f}s ~ {processed_sec:.0f}s)")

            pct = transcribe_start + int((processed_sec / audio_duration) * (95 - transcribe_start))
            pct = min(pct, 95)

            elapsed = time.time() - start_time
            if processed_sec > 0 and elapsed > 1:
                eta = elapsed * (audio_duration - processed_sec) / processed_sec
                eta_min, eta_sec = divmod(int(eta), 60)
                eta_str = f" (남은 시간: {eta_min}분 {eta_sec}초)" if eta_min > 0 else f" (남은 시간: {eta_sec}초)"
            else:
                eta_str = ""
            _progress(pct, f"[{transcribe_step}/{step_total}] 변환 중... {processed_sec:.0f}s / {audio_duration:.0f}s{eta_str}")

            # prompt 구성: 교정 힌트 + 이전 청크 텍스트 (224토큰 이내)
            prompt_parts = []
            if correction_hint:
                prompt_parts.append(correction_hint)
            if prev_text:
                ctx = _SPECIAL_TOKEN_RE.sub("", prev_text[-200:]).strip()
                if ctx:
                    prompt_parts.append(ctx)
            prompt = ". ".join(prompt_parts) if prompt_parts else None

            result = model.transcribe(
                chunk, language=lang_arg, verbose=False,
                initial_prompt=prompt,
                word_timestamps=use_diar,
            )

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
                    "no_speech_prob": seg.get("no_speech_prob", 0),
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

            if diarization_segments:
                from src.diarizer import assign_speakers
                chunk_segs = all_segments[chunk_seg_start:]
                matched = assign_speakers(diarization_segments, chunk_segs)
                all_segments[chunk_seg_start:] = matched

            for seg in all_segments[chunk_seg_start:]:
                # 실시간 스트림에도 교정 사전 적용
                if correction_entries:
                    for entry in correction_entries:
                        if entry["wrong"] in seg["text"]:
                            seg["text"] = seg["text"].replace(entry["wrong"], entry["correct"])
                msg_queue.put(("segment", seg))

        # 환각 필터 적용 (반복, 언어 불일치, no_speech 등)
        from src.hallucination_filter import filter_hallucinations
        pre_filter_count = len(all_segments)
        all_segments = filter_hallucinations(all_segments, language=language)
        filtered_count = pre_filter_count - len(all_segments)
        if filtered_count > 0:
            _log(f"환각 필터: {filtered_count}개 세그먼트 제거 ({pre_filter_count} → {len(all_segments)})")

        # 교정 사전 후처리: 잘못된 표현 → 올바른 표현 치환
        if correction_entries:
            replaced_count = 0
            for seg in all_segments:
                original = seg["text"]
                for entry in correction_entries:
                    if entry["wrong"] in seg["text"]:
                        seg["text"] = seg["text"].replace(entry["wrong"], entry["correct"])
                if seg["text"] != original:
                    replaced_count += 1
            if replaced_count > 0:
                _log(f"교정 사전 치환: {replaced_count}개 세그먼트 수정")
            # full_text도 치환
            full_text = " ".join(full_text_parts)
            for entry in correction_entries:
                full_text = full_text.replace(entry["wrong"], entry["correct"])
            full_text_parts = [full_text]

        # 최종 라벨 매핑
        if diarization_segments:
            from src.diarizer import map_speaker_labels
            all_segments = map_speaker_labels(all_segments)

        elapsed = time.time() - start_time

        msg_queue.put(("result", {
            "duration": duration,
            "full_text": " ".join(full_text_parts),
            "segments": all_segments,
            "elapsed": elapsed,
            "cancelled": _is_cancelled(),
        }))

    except Exception as e:
        import traceback
        _log(f"오류 발생: {e}")
        msg_queue.put(("error", str(e)))


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
        import ctypes
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetProcessWorkingSetSize(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.c_size_t(-1), ctypes.c_size_t(-1),
            )
        elif sys.platform == "linux":
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        # macOS: malloc_trim 없음 — subprocess 방식이므로 불필요
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
        elif sys.platform == "darwin":
            # macOS: vm_stat 또는 sysctl
            import subprocess as _sp
            out = _sp.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            total = int(out.strip())
            # vm_stat으로 free + inactive 페이지 계산
            vm = _sp.check_output(["vm_stat"], text=True)
            page_size = 16384  # Apple Silicon default
            free_pages = 0
            for line in vm.splitlines():
                if "page size of" in line:
                    page_size = int(line.split()[-2])
                if "Pages free:" in line:
                    free_pages += int(line.split()[-1].rstrip("."))
                if "Pages inactive:" in line:
                    free_pages += int(line.split()[-1].rstrip("."))
            return int(free_pages * page_size / (1024 * 1024))
        else:
            # Linux
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
        correction_entries: list | None = None,
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
        self.correction_entries = correction_entries or []
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _log(self, msg: str):
        """타임스탬프 포함 로그 메시지 전송."""
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_message.emit(f"[{ts}] {msg}")

    def run(self):
        """전처리 + 화자분리 + 전사를 별도 프로세스에서 실행한다.

        프로세스 종료 시 OS가 모든 ML 모델 메모리를 회수하므로
        메모리 누수가 원천적으로 발생하지 않는다.
        """
        tmp_wav = None
        proc = None
        try:
            ffmpeg = get_ffmpeg_exe()
            use_diar = self.use_diarization and self.hf_token

            # Step 1: Extract audio (가벼움 — 메인 프로세스에서 실행)
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

            # Step 2~4: 무거운 작업 → 별도 프로세스에서 실행
            # Linux의 기본 fork는 PyTorch/CUDA와 충돌 가능 → spawn 명시
            ctx = mp.get_context("spawn")
            msg_queue = ctx.Queue()
            cancel_event = ctx.Event()

            params = {
                "wav_path": tmp_wav,
                "use_diar": use_diar,
                "profile": self.profile,
                "model_name": self.model_name,
                "language": self.language,
                "skip_seconds": self.skip_seconds,
                "hf_token": self.hf_token,
                "num_speakers": self.num_speakers,
                "min_speakers": self.min_speakers,
                "max_speakers": self.max_speakers,
                "diar_threads": self.diar_threads,
                "correction_entries": self.correction_entries,
            }

            proc = ctx.Process(
                target=_subprocess_worker,
                args=(params, msg_queue, cancel_event),
                daemon=True,
            )
            proc.start()
            self._log(f"작업 프로세스 시작 (PID: {proc.pid})")

            # 메시지 루프: 서브프로세스 → Qt 시그널 중계
            result_data = None
            while True:
                # 취소 전파
                if self._cancelled and not cancel_event.is_set():
                    cancel_event.set()
                    self._log("취소 요청을 작업 프로세스에 전달")

                # 큐에서 메시지 수신 (100ms 타임아웃)
                try:
                    msg = msg_queue.get(timeout=0.1)
                except Exception:
                    # 프로세스가 죽었는지 확인
                    if not proc.is_alive():
                        break
                    continue

                msg_type = msg[0]

                if msg_type == "progress":
                    _, pct, text = msg
                    self.progress.emit(pct, text)
                elif msg_type == "log":
                    self.log_message.emit(msg[1])
                elif msg_type == "segment":
                    self.segment_ready.emit(msg[1])
                elif msg_type == "result":
                    result_data = msg[1]
                elif msg_type == "error":
                    self.error.emit(msg[1])
                    result_data = None
                    break

            # 프로세스 종료 대기 (최대 10초)
            proc.join(timeout=10)
            if proc.is_alive():
                self._log("작업 프로세스 강제 종료")
                proc.kill()
                proc.join(timeout=5)

            self._log(f"작업 프로세스 종료 (메모리 자동 회수)")

            # 결과 처리
            if result_data is None:
                if not self._cancelled:
                    self.error.emit("작업 프로세스가 비정상 종료되었습니다.")
                return

            cancelled = result_data.get("cancelled", False)
            all_segments = result_data["segments"]
            elapsed = result_data["elapsed"]
            duration = result_data["duration"]
            full_text = result_data["full_text"]

            if cancelled and all_segments:
                self._log(f"취소됨 — 부분 결과 저장: {len(all_segments)}개 세그먼트")

            if cancelled and not all_segments:
                self.error.emit("사용자가 변환을 취소했습니다.")
                return

            if not cancelled:
                self._log(f"변환 완료! 총 {len(all_segments)}개 세그먼트, 소요시간: {elapsed:.1f}초")
                self.progress.emit(100, "완료!")

            self.finished.emit({
                "filename": os.path.basename(self.video_path),
                "filepath": self.video_path,
                "duration": duration,
                "full_text": full_text,
                "segments": all_segments,
                "elapsed": elapsed,
                "model_name": self.model_name,
                "language": self.language,
            })

        except Exception as e:
            self._log(f"오류 발생: {e}")
            self.error.emit(str(e))
        finally:
            if proc and proc.is_alive():
                proc.kill()
                proc.join(timeout=5)

            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass
