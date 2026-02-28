import os

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QFont
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
