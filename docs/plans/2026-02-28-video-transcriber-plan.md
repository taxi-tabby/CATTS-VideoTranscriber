# Video Transcriber Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone Windows GUI application that extracts audio from video files and transcribes them to text using Whisper, with a browsable history of past transcriptions.

**Architecture:** Single-process PySide6 app with QThread-based Whisper worker. SQLite for persistent storage. ffmpeg bundled via imageio-ffmpeg. PyInstaller for exe bundling, Inno Setup for Windows installer.

**Tech Stack:** Python 3.13, PySide6, openai-whisper, imageio-ffmpeg, SQLite, PyInstaller, Inno Setup

---

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/__init__.py` (empty)
- Create: `src/resources/` (directory)
- Create: `tests/__init__.py` (empty)
- Create: `.gitignore`

**Step 1: Create .gitignore**

```gitignore
__pycache__/
*.pyc
*.pyo
.venv/
venv/
dist/
build/output/
*.spec
*.egg-info/
*.wav
*.db
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "video-transcriber"
version = "1.0.0"
description = "영상 음성 추출 및 텍스트 변환 프로그램"
requires-python = ">=3.11"

[project.scripts]
video-transcriber = "src.main:main"
```

**Step 3: Create requirements.txt**

```
PySide6>=6.6.0
openai-whisper>=20231117
imageio-ffmpeg>=0.5.1
numpy>=1.24.0
pyinstaller>=6.0
pytest>=7.0
```

**Step 4: Create virtual environment and install dependencies**

Run:
```bash
cd C:/Users/rkdmf/Videos/video-transcriber
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Expected: All packages install successfully.

**Step 5: Create empty init files and resource directory**

```bash
mkdir -p src/resources tests
touch src/__init__.py tests/__init__.py
```

**Step 6: Commit**

```bash
git add .gitignore pyproject.toml requirements.txt src/__init__.py tests/__init__.py
git commit -m "chore: project setup with dependencies"
```

---

### Task 2: Database Layer

**Files:**
- Create: `src/database.py`
- Create: `tests/test_database.py`

**Step 1: Write the failing tests**

Create `tests/test_database.py`:

```python
import os
import pytest
from src.database import Database


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    database = Database(db_path)
    yield database
    database.close()


class TestDatabase:
    def test_add_transcription(self, db):
        tid = db.add_transcription(
            filename="test.mp4",
            filepath="C:/Videos/test.mp4",
            duration=120.5,
            full_text="안녕하세요",
            segments=[
                {"start": 0.0, "end": 5.0, "text": "안녕"},
                {"start": 5.0, "end": 10.0, "text": "하세요"},
            ],
        )
        assert tid == 1

    def test_get_all_transcriptions(self, db):
        db.add_transcription("a.mp4", "C:/a.mp4", 60.0, "텍스트A", [])
        db.add_transcription("b.mp4", "C:/b.mp4", 120.0, "텍스트B", [])
        results = db.get_all_transcriptions()
        assert len(results) == 2
        assert results[0]["filename"] == "b.mp4"  # newest first

    def test_get_transcription_with_segments(self, db):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "첫번째"},
            {"start": 5.0, "end": 10.0, "text": "두번째"},
        ]
        tid = db.add_transcription("c.mp4", "C:/c.mp4", 30.0, "전체", segments)
        result = db.get_transcription(tid)
        assert result["filename"] == "c.mp4"
        assert result["full_text"] == "전체"
        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "첫번째"

    def test_delete_transcription(self, db):
        tid = db.add_transcription("d.mp4", "C:/d.mp4", 10.0, "삭제", [])
        db.delete_transcription(tid)
        results = db.get_all_transcriptions()
        assert len(results) == 0

    def test_delete_cascades_segments(self, db):
        segments = [{"start": 0.0, "end": 5.0, "text": "세그먼트"}]
        tid = db.add_transcription("e.mp4", "C:/e.mp4", 10.0, "삭제", segments)
        db.delete_transcription(tid)
        result = db.get_transcription(tid)
        assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `cd C:/Users/rkdmf/Videos/video-transcriber && python -m pytest tests/test_database.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'src.database'`

**Step 3: Write the implementation**

Create `src/database.py`:

```python
import sqlite3
from datetime import datetime


class Database:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                filepath    TEXT NOT NULL,
                duration    REAL,
                created_at  TEXT NOT NULL,
                full_text   TEXT
            );
            CREATE TABLE IF NOT EXISTS segments (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                transcription_id  INTEGER NOT NULL
                    REFERENCES transcriptions(id) ON DELETE CASCADE,
                start_time        REAL NOT NULL,
                end_time          REAL NOT NULL,
                text              TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def add_transcription(
        self,
        filename: str,
        filepath: str,
        duration: float,
        full_text: str,
        segments: list[dict],
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO transcriptions (filename, filepath, duration, created_at, full_text)
               VALUES (?, ?, ?, ?, ?)""",
            (filename, filepath, duration, datetime.now().isoformat(), full_text),
        )
        tid = cur.lastrowid
        for seg in segments:
            self._conn.execute(
                """INSERT INTO segments (transcription_id, start_time, end_time, text)
                   VALUES (?, ?, ?, ?)""",
                (tid, seg["start"], seg["end"], seg["text"]),
            )
        self._conn.commit()
        return tid

    def get_all_transcriptions(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, filename, filepath, duration, created_at FROM transcriptions ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_transcription(self, tid: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM transcriptions WHERE id = ?", (tid,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        seg_rows = self._conn.execute(
            "SELECT start_time as start, end_time as end, text FROM segments WHERE transcription_id = ? ORDER BY start_time",
            (tid,),
        ).fetchall()
        result["segments"] = [dict(s) for s in seg_rows]
        return result

    def delete_transcription(self, tid: int):
        self._conn.execute("DELETE FROM transcriptions WHERE id = ?", (tid,))
        self._conn.commit()

    def close(self):
        self._conn.close()
```

**Step 4: Run tests to verify they pass**

Run: `cd C:/Users/rkdmf/Videos/video-transcriber && python -m pytest tests/test_database.py -v`

Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/database.py tests/test_database.py
git commit -m "feat: add SQLite database layer with CRUD operations"
```

---

### Task 3: Transcriber Worker

**Files:**
- Create: `src/transcriber.py`

This module wraps the existing transcribe.py logic into a QThread-compatible worker with progress signals. Testing Whisper directly requires the model and real audio, so this task uses manual verification in Task 6.

**Step 1: Write the transcriber worker**

Create `src/transcriber.py`:

```python
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
```

**Step 2: Commit**

```bash
git add src/transcriber.py
git commit -m "feat: add Whisper transcriber worker with QThread signals"
```

---

### Task 4: Main Window UI

**Files:**
- Create: `src/main_window.py`

This is the core UI with the 2-panel layout: transcription list (left) + detail viewer (right) + progress bar (bottom).

**Step 1: Write the main window**

Create `src/main_window.py`:

```python
import os

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.database import Database
from src.transcriber import TranscriberWorker


def format_duration(seconds: float) -> str:
    if seconds is None:
        return "--:--:--"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_timestamp(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class MainWindow(QMainWindow):
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self._worker = None
        self._thread = None

        self.setWindowTitle("Video Transcriber")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)

        self._build_ui()
        self._load_list()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Splitter: left list + right detail ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, stretch=1)

        # Left panel
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 4, 8)

        self.btn_add = QPushButton("+ 영상 추가")
        self.btn_add.setFixedHeight(36)
        self.btn_add.clicked.connect(self._on_add_video)
        left_layout.addWidget(self.btn_add)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_select_item)
        left_layout.addWidget(self.list_widget)

        self.btn_delete = QPushButton("선택 삭제")
        self.btn_delete.clicked.connect(self._on_delete)
        left_layout.addWidget(self.btn_delete)

        splitter.addWidget(left)

        # Right panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 8, 8, 8)

        # Header info
        self.lbl_title = QLabel("")
        self.lbl_title.setFont(QFont("", 11, QFont.Weight.Bold))
        right_layout.addWidget(self.lbl_title)

        self.lbl_info = QLabel("")
        right_layout.addWidget(self.lbl_info)

        # Tabs: timeline / full text
        self.tabs = QTabWidget()

        # Timeline tab
        self.txt_timeline = QPlainTextEdit()
        self.txt_timeline.setReadOnly(True)
        self.txt_timeline.setFont(QFont("Consolas", 10))
        self.tabs.addTab(self.txt_timeline, "타임라인")

        # Full text tab
        self.txt_fulltext = QPlainTextEdit()
        self.txt_fulltext.setReadOnly(True)
        self.txt_fulltext.setFont(QFont("Malgun Gothic", 10))
        self.tabs.addTab(self.txt_fulltext, "전체 텍스트")

        right_layout.addWidget(self.tabs, stretch=1)

        # Copy button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_copy = QPushButton("텍스트 복사")
        self.btn_copy.clicked.connect(self._on_copy)
        btn_row.addWidget(self.btn_copy)
        right_layout.addLayout(btn_row)

        # Empty state
        self.lbl_empty = QLabel("영상을 추가하면 여기에 결과가 표시됩니다.")
        self.lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_empty.setStyleSheet("color: #888; font-size: 14px;")
        right_layout.addWidget(self.lbl_empty)

        splitter.addWidget(right)
        splitter.setSizes([280, 720])

        # --- Bottom progress bar ---
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(8, 4, 8, 8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(22)
        self.progress_bar.setVisible(False)
        bottom_layout.addWidget(self.progress_bar, stretch=1)

        self.lbl_status = QLabel("")
        self.lbl_status.setVisible(False)
        bottom_layout.addWidget(self.lbl_status)

        layout.addWidget(bottom)

        self._show_detail(False)

    def _show_detail(self, show: bool):
        self.lbl_title.setVisible(show)
        self.lbl_info.setVisible(show)
        self.tabs.setVisible(show)
        self.btn_copy.setVisible(show)
        self.lbl_empty.setVisible(not show)

    # --- Data loading ---

    def _load_list(self):
        self.list_widget.clear()
        self._items = self.db.get_all_transcriptions()
        for t in self._items:
            dur = format_duration(t.get("duration"))
            date = t["created_at"][:10]
            item = QListWidgetItem(f"{t['filename']}\n{date}  {dur}")
            item.setData(Qt.ItemDataRole.UserRole, t["id"])
            self.list_widget.addItem(item)

    def _on_select_item(self, row: int):
        if row < 0 or row >= len(self._items):
            self._show_detail(False)
            return

        tid = self._items[row]["id"]
        data = self.db.get_transcription(tid)
        if data is None:
            self._show_detail(False)
            return

        self._show_detail(True)
        self.lbl_title.setText(data["filename"])
        dur = format_duration(data.get("duration"))
        date = data["created_at"][:10]
        self.lbl_info.setText(f"날짜: {date}    길이: {dur}")

        # Timeline
        lines = []
        for seg in data.get("segments", []):
            ts_start = format_timestamp(seg["start"])
            ts_end = format_timestamp(seg["end"])
            lines.append(f"[{ts_start} ~ {ts_end}]  {seg['text']}")
        self.txt_timeline.setPlainText("\n".join(lines))

        # Full text
        self.txt_fulltext.setPlainText(data.get("full_text", ""))

    # --- Actions ---

    def _on_add_video(self):
        if self._thread and self._thread.isRunning():
            QMessageBox.warning(self, "진행 중", "현재 변환이 진행 중입니다. 완료 후 다시 시도하세요.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "영상 파일 선택",
            "",
            "영상 파일 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;모든 파일 (*)",
        )
        if not path:
            return

        self._start_transcription(path)

    def _start_transcription(self, video_path: str):
        self.btn_add.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setVisible(True)
        self.lbl_status.setText("준비 중...")

        self._thread = QThread()
        self._worker = TranscriberWorker(video_path)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.start()

    def _on_progress(self, percent: int, message: str):
        self.progress_bar.setValue(percent)
        self.lbl_status.setText(message)

    def _on_finished(self, result: dict):
        self.db.add_transcription(
            filename=result["filename"],
            filepath=result["filepath"],
            duration=result["duration"],
            full_text=result["full_text"],
            segments=result["segments"],
        )
        self._load_list()
        self.list_widget.setCurrentRow(0)

        self.btn_add.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setVisible(False)

        elapsed_min = result.get("elapsed", 0) / 60
        QMessageBox.information(
            self, "완료", f"변환 완료! (소요시간: {elapsed_min:.1f}분)"
        )

    def _on_error(self, message: str):
        self.btn_add.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setVisible(False)
        QMessageBox.critical(self, "오류", f"변환 중 오류 발생:\n{message}")

    def _on_delete(self):
        row = self.list_widget.currentRow()
        if row < 0:
            return

        tid = self._items[row]["id"]
        name = self._items[row]["filename"]
        reply = QMessageBox.question(
            self,
            "삭제 확인",
            f"'{name}' 트랜스크립션을 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.db.delete_transcription(tid)
            self._load_list()
            self._show_detail(False)

    def _on_copy(self):
        current_tab = self.tabs.currentIndex()
        if current_tab == 0:
            text = self.txt_timeline.toPlainText()
        else:
            text = self.txt_fulltext.toPlainText()

        if text:
            QApplication.clipboard().setText(text)
            self.lbl_status.setVisible(True)
            self.lbl_status.setText("클립보드에 복사됨!")
            from PySide6.QtCore import QTimer
            QTimer.singleShot(2000, lambda: self.lbl_status.setVisible(False))

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            reply = QMessageBox.question(
                self,
                "종료 확인",
                "변환이 진행 중입니다. 종료하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self._worker.cancel()
            self._thread.quit()
            self._thread.wait(5000)
        event.accept()
```

**Step 2: Commit**

```bash
git add src/main_window.py
git commit -m "feat: add main window with 2-panel layout and transcription viewer"
```

---

### Task 5: App Entry Point

**Files:**
- Create: `src/main.py`

**Step 1: Write the entry point**

Create `src/main.py`:

```python
import os
import sys

from PySide6.QtWidgets import QApplication

from src.database import Database
from src.main_window import MainWindow


def get_db_path() -> str:
    app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    db_dir = os.path.join(app_data, "VideoTranscriber")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "transcriptions.db")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Video Transcriber")
    app.setStyle("Fusion")

    db = Database(get_db_path())
    window = MainWindow(db)
    window.show()

    exit_code = app.exec()
    db.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
```

**Step 2: Verify the app launches**

Run: `cd C:/Users/rkdmf/Videos/video-transcriber && python -m src.main`

Expected: A window titled "Video Transcriber" opens with an empty list on the left and placeholder text on the right.

**Step 3: Commit**

```bash
git add src/main.py
git commit -m "feat: add application entry point"
```

---

### Task 6: Integration Smoke Test

**No files to create.** This task verifies the full flow works end-to-end.

**Step 1: Launch the app**

Run: `cd C:/Users/rkdmf/Videos/video-transcriber && python -m src.main`

**Step 2: Test "영상 추가"**

1. Click "영상 추가" button
2. Select a short video file (ideally under 1 minute for quick testing)
3. Observe the progress bar updating
4. Wait for completion dialog
5. Verify the transcription appears in the list
6. Click on it to see timeline + full text

**Step 3: Test "복사"**

1. Click "텍스트 복사" button
2. Paste in a text editor to verify clipboard contents

**Step 4: Test "삭제"**

1. Select the transcription in the list
2. Click "선택 삭제"
3. Confirm deletion
4. Verify the item is removed

**Step 5: Fix any bugs found**

If issues arise, fix them and commit:
```bash
git add -u
git commit -m "fix: resolve integration issues found during smoke test"
```

---

### Task 7: PyInstaller Build Configuration

**Files:**
- Create: `build/video_transcriber.spec`
- Create: `build.bat`

**Step 1: Create the PyInstaller spec file**

Create `build/video_transcriber.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import imageio_ffmpeg

block_cipher = None

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

a = Analysis(
    ['../src/main.py'],
    pathex=[os.path.abspath('..')],
    binaries=[(ffmpeg_exe, 'imageio_ffmpeg/binaries')],
    datas=[],
    hiddenimports=[
        'whisper',
        'PySide6',
        'numpy',
        'imageio_ffmpeg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'PIL'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VideoTranscriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VideoTranscriber',
)
```

**Step 2: Create the build script**

Create `build.bat`:

```batch
@echo off
echo === Video Transcriber Build ===
echo.

cd /d "%~dp0"

echo [1/2] PyInstaller로 exe 생성 중...
cd build
pyinstaller video_transcriber.spec --distpath output/dist --workpath output/build --clean -y
cd ..

if errorlevel 1 (
    echo 빌드 실패!
    pause
    exit /b 1
)

echo.
echo [2/2] 빌드 완료!
echo 결과물: build\output\dist\VideoTranscriber\
echo.
pause
```

**Step 3: Test the build**

Run: `cd C:/Users/rkdmf/Videos/video-transcriber && ./build.bat`

Expected: Creates `build/output/dist/VideoTranscriber/VideoTranscriber.exe`. Launch it to verify it works independently.

**Step 4: Commit**

```bash
git add build/video_transcriber.spec build.bat
git commit -m "feat: add PyInstaller build configuration"
```

---

### Task 8: Inno Setup Installer Script

**Files:**
- Create: `build/installer.iss`
- Modify: `build.bat` (add Inno Setup step)

**Step 1: Create the Inno Setup script**

Create `build/installer.iss`:

```iss
[Setup]
AppName=Video Transcriber
AppVersion=1.0.0
AppPublisher=Video Transcriber
DefaultDirName={autopf}\VideoTranscriber
DefaultGroupName=Video Transcriber
OutputDir=output\installer
OutputBaseFilename=VideoTranscriber_Setup_1.0.0
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; GroupDescription: "추가 옵션:"; Flags: unchecked

[Files]
Source: "output\dist\VideoTranscriber\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Video Transcriber"; Filename: "{app}\VideoTranscriber.exe"
Name: "{group}\Video Transcriber 제거"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Video Transcriber"; Filename: "{app}\VideoTranscriber.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VideoTranscriber.exe"; Description: "Video Transcriber 실행"; Flags: nowait postinstall skipifsilent
```

**Step 2: Update build.bat to include Inno Setup step**

Replace the contents of `build.bat` with:

```batch
@echo off
echo === Video Transcriber Build ===
echo.

cd /d "%~dp0"

echo [1/3] PyInstaller로 exe 생성 중...
cd build
pyinstaller video_transcriber.spec --distpath output/dist --workpath output/build --clean -y
cd ..

if errorlevel 1 (
    echo PyInstaller 빌드 실패!
    pause
    exit /b 1
)

echo.
echo [2/3] Inno Setup으로 설치 프로그램 생성 중...
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" (
    "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" build\installer.iss
) else (
    echo Inno Setup 6이 설치되어 있지 않습니다.
    echo https://jrsoftware.org/isdl.php 에서 설치 후 다시 시도하세요.
    echo exe 빌드는 완료되었습니다: build\output\dist\VideoTranscriber\
    pause
    exit /b 0
)

if errorlevel 1 (
    echo Inno Setup 빌드 실패!
    pause
    exit /b 1
)

echo.
echo [3/3] 빌드 완료!
echo exe: build\output\dist\VideoTranscriber\VideoTranscriber.exe
echo 설치파일: build\output\installer\VideoTranscriber_Setup_1.0.0.exe
echo.
pause
```

**Step 3: Commit**

```bash
git add build/installer.iss build.bat
git commit -m "feat: add Inno Setup installer script and update build.bat"
```

---

## Summary

| Task | Description | Estimated Complexity |
|------|-------------|---------------------|
| 1 | Project setup | Low |
| 2 | Database layer + tests | Low |
| 3 | Transcriber worker | Medium |
| 4 | Main window UI | Medium-High |
| 5 | App entry point | Low |
| 6 | Integration smoke test | Manual |
| 7 | PyInstaller build | Medium |
| 8 | Inno Setup installer | Low |

Tasks must be done in order (1 → 2 → 3 → 4 → 5 → 6 → 7 → 8).
