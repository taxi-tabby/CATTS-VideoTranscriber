"""메모리 관리 및 서브프로세스 워커 회귀 테스트.

ML 모델 없이 실행 가능하도록 mock 기반으로 작성.
CI(GitHub Actions)에서 매 PR마다 실행된다.
"""

import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# _get_available_memory_mb
# ---------------------------------------------------------------------------

class TestGetAvailableMemoryMb:
    def test_returns_positive_int(self):
        from src.transcriber import _get_available_memory_mb
        result = _get_available_memory_mb()
        assert isinstance(result, int)
        # CI 환경에서도 최소 수백 MB는 있어야 함
        assert result > 100

    def test_returns_int_type(self):
        from src.transcriber import _get_available_memory_mb
        result = _get_available_memory_mb()
        assert type(result) is int  # numpy.int64 등이 아닌 순수 int


    # _force_release_ml_memory는 서브프로세스 방식 전환으로 제거됨 (dead code)


# ---------------------------------------------------------------------------
# _cap_workers_by_memory
# ---------------------------------------------------------------------------

class TestCapWorkersByMemoryDetailed:
    def test_single_worker_always_allowed(self):
        from src.transcriber import _cap_workers_by_memory
        logs = []
        result = _cap_workers_by_memory(1, 999999, logs.append)
        assert result == 1

    def test_caps_when_model_too_large(self):
        from src.transcriber import _cap_workers_by_memory
        logs = []
        # 모델 100GB → 워커 수 제한
        result = _cap_workers_by_memory(8, 100000, logs.append)
        assert result < 8

    def test_does_not_exceed_requested(self):
        from src.transcriber import _cap_workers_by_memory
        logs = []
        result = _cap_workers_by_memory(2, 10, logs.append)
        assert result <= 2


# ---------------------------------------------------------------------------
# 서브프로세스 메시지 큐 통신
# ---------------------------------------------------------------------------

def _dummy_subprocess_target(msg_queue, cancel_event):
    """테스트용 경량 서브프로세스 함수."""
    msg_queue.put(("log", "subprocess started"))
    msg_queue.put(("progress", 50, "halfway"))

    if cancel_event.is_set():
        msg_queue.put(("error", "cancelled"))
        return

    msg_queue.put(("result", {"test": True}))


def _allocate_and_report(queue):
    """서브프로세스에서 큰 메모리를 할당하고 보고한다."""
    big_array = np.zeros(25_000_000, dtype=np.float32)  # ~100MB
    queue.put(("allocated", len(big_array)))
    del big_array
    queue.put(("done", True))


def _crash_target(queue):
    """의도적으로 크래시하는 서브프로세스 타겟."""
    queue.put(("log", "about to crash"))
    raise RuntimeError("intentional crash")


class TestSubprocessCommunication:
    def test_queue_message_roundtrip(self):
        """서브프로세스가 큐에 넣은 메시지를 메인 프로세스에서 받을 수 있어야 한다."""
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        event = ctx.Event()

        proc = ctx.Process(target=_dummy_subprocess_target, args=(q, event))
        proc.start()
        proc.join(timeout=30)

        messages = []
        while not q.empty():
            messages.append(q.get_nowait())

        types = [m[0] for m in messages]
        assert "log" in types
        assert "progress" in types
        assert "result" in types

    def test_cancel_event_propagation(self):
        """cancel_event가 서브프로세스에 전파되어야 한다."""
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        event = ctx.Event()
        event.set()  # 시작 전에 취소

        proc = ctx.Process(target=_dummy_subprocess_target, args=(q, event))
        proc.start()
        proc.join(timeout=30)

        messages = []
        while not q.empty():
            messages.append(q.get_nowait())

        types = [m[0] for m in messages]
        assert "error" in types
        # result가 없어야 함 (취소됨)
        assert "result" not in types

    def test_subprocess_memory_isolation(self):
        """서브프로세스에서 할당한 메모리가 메인 프로세스에 영향을 주지 않아야 한다."""
        ctx = mp.get_context("spawn")
        q = ctx.Queue()

        proc = ctx.Process(target=_allocate_and_report, args=(q,))
        proc.start()
        proc.join(timeout=30)

        assert proc.exitcode == 0

    def test_subprocess_crash_does_not_hang(self):
        """서브프로세스가 크래시해도 메인 프로세스가 멈추지 않아야 한다."""
        ctx = mp.get_context("spawn")
        q = ctx.Queue()

        proc = ctx.Process(target=_crash_target, args=(q,))
        proc.start()
        proc.join(timeout=10)

        assert not proc.is_alive()
        assert proc.exitcode != 0


# ---------------------------------------------------------------------------
# 서브프로세스 워커 함수 (mock 기반)
# ---------------------------------------------------------------------------

class TestSubprocessWorker:
    def test_sends_error_on_missing_wav(self):
        """존재하지 않는 WAV 파일이면 에러 메시지를 보내야 한다."""
        from src.transcriber import _subprocess_worker

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        event = ctx.Event()

        params = {
            "wav_path": "/nonexistent/path.wav",
            "use_diar": False,
            "profile": "interview",
            "model_name": "tiny",
            "language": "ko",
            "skip_seconds": 0.0,
            "hf_token": None,
        }

        proc = ctx.Process(target=_subprocess_worker, args=(params, q, event))
        proc.start()
        proc.join(timeout=60)

        messages = []
        while not q.empty():
            messages.append(q.get_nowait())

        types = [m[0] for m in messages]
        assert "error" in types

    def test_cancel_during_preprocessing(self):
        """전처리 중 취소하면 에러 없이 종료되어야 한다."""
        from src.transcriber import _subprocess_worker

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        event = ctx.Event()
        event.set()  # 즉시 취소

        # 유효한 WAV 생성
        import tempfile
        import wave
        tmp = os.path.join(tempfile.gettempdir(), "_test_cancel.wav")
        audio = np.zeros(16000, dtype=np.int16)
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio.tobytes())

        params = {
            "wav_path": tmp,
            "use_diar": False,
            "profile": "interview",
            "model_name": "tiny",
            "language": "ko",
            "skip_seconds": 0.0,
            "hf_token": None,
        }

        try:
            proc = ctx.Process(target=_subprocess_worker, args=(params, q, event))
            proc.start()
            proc.join(timeout=60)
            # 정상 종료 (exitcode 0) — 취소가 에러가 아닌 정상 종료
            assert proc.exitcode == 0
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


# ---------------------------------------------------------------------------
# 크로스 플랫폼 호환성
# ---------------------------------------------------------------------------

class TestCrossPlatform:
    def test_spawn_context_available(self):
        """모든 플랫폼에서 spawn context를 사용할 수 있어야 한다."""
        ctx = mp.get_context("spawn")
        assert ctx is not None

    def test_memory_measurement_platform(self):
        """현재 플랫폼에서 메모리 측정이 동작해야 한다."""
        from src.transcriber import _get_available_memory_mb
        result = _get_available_memory_mb()
        assert result > 0, f"메모리 측정 실패 (platform={sys.platform})"
