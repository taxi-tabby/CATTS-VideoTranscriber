"""오류/크래시 발생 시 시스템 정보를 수집하여 GitHub Issue로 보고하는 모듈."""

import os
import platform
import sys
import traceback
import urllib.parse
from datetime import datetime

ISSUE_URL = "https://github.com/taxi-tabby/CATTS-VideoTranscriber/issues/new"


def collect_system_info() -> dict:
    """디버깅에 필요한 시스템 정보를 수집한다."""
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": sys.version.split()[0],
        "frozen": getattr(sys, "frozen", False),
    }

    # 앱 버전
    try:
        from src.main import APP_VERSION
        info["app_version"] = str(APP_VERSION)
    except Exception:
        info["app_version"] = "unknown"

    # 주요 라이브러리 버전
    for mod_name in ("torch", "whisper", "numpy", "PySide6"):
        try:
            mod = __import__(mod_name)
            info[mod_name] = getattr(mod, "__version__", "OK")
        except ImportError:
            info[mod_name] = "not installed"

    # GPU 정보
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            info["gpu_vram"] = f"{vram:.1f}GB"
        else:
            info["gpu"] = "CPU only"
    except Exception:
        info["gpu"] = "unknown"

    # 메모리 정보
    try:
        from src.transcriber import _get_available_memory_mb
        avail = _get_available_memory_mb()
        if avail > 0:
            info["available_ram"] = f"{avail}MB"
    except Exception:
        pass

    return info


def format_system_info(info: dict) -> str:
    """시스템 정보를 마크다운 테이블로 포맷한다."""
    lines = ["| Item | Value |", "|------|-------|"]
    labels = {
        "app_version": "App Version",
        "os": "OS",
        "python": "Python",
        "frozen": "Build",
        "torch": "PyTorch",
        "whisper": "Whisper",
        "numpy": "NumPy",
        "PySide6": "PySide6",
        "gpu": "GPU",
        "gpu_vram": "GPU VRAM",
        "available_ram": "Available RAM",
    }
    for key, label in labels.items():
        val = info.get(key)
        if val is not None:
            if key == "frozen":
                val = "PyInstaller" if val else "Source"
            lines.append(f"| {label} | {val} |")
    return "\n".join(lines)


def build_issue_body(
    error_message: str,
    error_traceback: str = "",
    processing_log: str = "",
    system_info: dict | None = None,
    include_log: bool = True,
    include_system: bool = True,
) -> str:
    """GitHub Issue 본문을 생성한다."""
    if system_info is None:
        system_info = collect_system_info()

    sections = []

    sections.append("## Error\n")
    sections.append(f"```\n{error_message}\n```\n")

    if error_traceback:
        sections.append("<details><summary>Traceback</summary>\n")
        sections.append(f"```\n{error_traceback}\n```\n")
        sections.append("</details>\n")

    sections.append("## Steps to Reproduce\n")
    sections.append("<!-- Please describe what you were doing when the error occurred -->\n")
    sections.append("1. \n2. \n3. \n")

    if include_log and processing_log:
        log_lines = processing_log.strip().splitlines()
        if len(log_lines) > 50:
            log_text = "... (truncated)\n" + "\n".join(log_lines[-50:])
        else:
            log_text = "\n".join(log_lines)
        sections.append("<details><summary>Processing Log</summary>\n")
        sections.append(f"```\n{log_text}\n```\n")
        sections.append("</details>\n")

    if include_system:
        sections.append("## System Information\n")
        sections.append(format_system_info(system_info) + "\n")

    return "\n".join(sections)


def build_issue_title(error_message: str) -> str:
    """오류 메시지에서 간결한 Issue 제목을 생성한다."""
    # 첫 줄만 사용, 80자 제한
    first_line = error_message.strip().splitlines()[0] if error_message.strip() else "Unknown error"
    if len(first_line) > 80:
        first_line = first_line[:77] + "..."
    return f"[Bug] {first_line}"


def open_issue_url(title: str, body: str) -> None:
    """GitHub Issue 작성 페이지를 브라우저에서 연다.

    URL 길이 제한(~8000자)을 초과하면 body를 클립보드에 복사하고
    빈 Issue 페이지를 연다.
    """
    import webbrowser

    params = urllib.parse.urlencode({
        "title": title,
        "body": body,
        "labels": "bug",
    })
    full_url = f"{ISSUE_URL}?{params}"

    # URL 길이 제한: 브라우저마다 다르지만 보수적으로 7500자
    if len(full_url) > 7500:
        # body를 클립보드에 복사하고 빈 Issue 페이지를 연다
        try:
            from PySide6.QtWidgets import QApplication
            clipboard = QApplication.instance().clipboard()
            clipboard.setText(body)
        except Exception:
            pass
        short_params = urllib.parse.urlencode({
            "title": title,
            "body": "(Content too long for URL — copied to clipboard. Please paste with Ctrl+V.)",
            "labels": "bug",
        })
        webbrowser.open(f"{ISSUE_URL}?{short_params}")
    else:
        webbrowser.open(full_url)


def install_global_exception_hook(get_window_fn):
    """전역 예외 훅을 설치한다.

    - sys.excepthook: GUI(메인) 스레드의 미처리 예외
    - threading.excepthook: 워커 스레드의 미처리 예외

    Args:
        get_window_fn: MainWindow 인스턴스를 반환하는 callable.
                       윈도우가 아직 생성되지 않았을 때 None을 반환해도 됨.
    """
    import threading

    _original_hook = sys.excepthook

    def _handle_exception(exc_type, exc_value, exc_tb):
        """공통 예외 처리 로직."""
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        error_msg = f"{exc_type.__name__}: {exc_value}"

        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is None:
                return

            window = get_window_fn()
            log_text = ""
            if window and hasattr(window, "txt_log"):
                log_text = window.txt_log.toPlainText()

            _show_crash_dialog(error_msg, tb_str, log_text, parent=window)
        except Exception as dialog_err:
            # 다이얼로그 표시 실패 시에도 stderr에 기록
            print(f"[crash_reporter] 다이얼로그 표시 실패: {dialog_err}", file=sys.stderr)
            print(f"[crash_reporter] 원본 오류: {error_msg}", file=sys.stderr)
            print(tb_str, file=sys.stderr)

    def _exception_hook(exc_type, exc_value, exc_tb):
        # 기본 stderr 출력은 유지
        _original_hook(exc_type, exc_value, exc_tb)
        _handle_exception(exc_type, exc_value, exc_tb)

    def _threading_exception_hook(args):
        # threading.ExceptHookArgs: (exc_type, exc_value, exc_traceback, thread)
        if args.exc_type is SystemExit:
            return
        print(
            f"[crash_reporter] 스레드 '{args.thread.name if args.thread else '?'}' 예외:",
            file=sys.stderr,
        )
        _handle_exception(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = _exception_hook
    threading.excepthook = _threading_exception_hook


def _show_crash_dialog(
    error_message: str,
    error_traceback: str = "",
    processing_log: str = "",
    parent=None,
):
    """크래시 리포트 다이얼로그를 표시한다."""
    from PySide6.QtWidgets import (
        QCheckBox, QDialog, QDialogButtonBox, QLabel,
        QPlainTextEdit, QVBoxLayout,
    )
    from PySide6.QtCore import Qt

    system_info = collect_system_info()

    dialog = QDialog(parent)
    dialog.setWindowTitle("오류 보고")
    dialog.setMinimumSize(560, 420)
    layout = QVBoxLayout(dialog)

    # 안내 메시지
    lbl = QLabel(
        "오류가 발생했습니다. 아래 정보를 GitHub Issue로 보고하면\n"
        "문제 해결에 큰 도움이 됩니다."
    )
    lbl.setWordWrap(True)
    layout.addWidget(lbl)

    # 포함 항목 체크박스
    chk_log = QCheckBox("처리 로그 포함")
    chk_log.setChecked(bool(processing_log))
    chk_log.setEnabled(bool(processing_log))
    layout.addWidget(chk_log)

    chk_system = QCheckBox("시스템 정보 포함 (OS, GPU, 라이브러리 버전)")
    chk_system.setChecked(True)
    layout.addWidget(chk_system)

    # 미리보기
    preview = QPlainTextEdit()
    preview.setReadOnly(True)
    layout.addWidget(preview)

    def _update_preview():
        body = build_issue_body(
            error_message,
            error_traceback=error_traceback,
            processing_log=processing_log,
            system_info=system_info,
            include_log=chk_log.isChecked(),
            include_system=chk_system.isChecked(),
        )
        preview.setPlainText(body)

    chk_log.toggled.connect(lambda: _update_preview())
    chk_system.toggled.connect(lambda: _update_preview())
    _update_preview()

    # 버튼
    buttons = QDialogButtonBox()
    btn_report = buttons.addButton("GitHub에 보고", QDialogButtonBox.ButtonRole.AcceptRole)
    btn_close = buttons.addButton("닫기", QDialogButtonBox.ButtonRole.RejectRole)
    layout.addWidget(buttons)

    def _on_report():
        title = build_issue_title(error_message)
        body = preview.toPlainText()
        open_issue_url(title, body)
        dialog.accept()

    btn_report.clicked.connect(_on_report)
    btn_close.clicked.connect(dialog.reject)

    dialog.exec()
