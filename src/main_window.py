import os
import webbrowser

from PySide6.QtCore import Qt, QThread, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.config import (
    get_hf_token, set_hf_token, delete_hf_token,
    get_whisper_model, set_whisper_model,
    get_show_startup_guide, set_show_startup_guide,
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


# ────────────────────────────────────────────
# 시작 안내 다이얼로그
# ────────────────────────────────────────────

class StartupGuideDialog(QDialog):
    """첫 실행 시 필요 환경을 안내하는 다이얼로그."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("시작 안내")
        self.setMinimumWidth(500)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Video Transcriber 사용 안내")
        title.setFont(QFont("", 13, QFont.Weight.Bold))
        layout.addWidget(title)

        layout.addSpacing(8)

        info = QLabel(
            "이 프로그램은 영상/음성 파일에서 텍스트를 추출합니다.\n"
            "아래 환경이 필요합니다.\n"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # 필수 환경
        grp_required = QGroupBox("필수")
        req_layout = QVBoxLayout(grp_required)
        req_layout.addWidget(QLabel(
            "- 인터넷 연결: 첫 실행 시 Whisper 음성인식 모델을 다운로드합니다\n"
            "  (모델 크기: tiny 39MB ~ large 1.5GB, 기본값 medium 769MB)\n"
            "- 다운로드된 모델은 저장되므로 이후에는 오프라인으로 사용 가능합니다"
        ))
        layout.addWidget(grp_required)

        # 권장 환경
        grp_recommended = QGroupBox("권장")
        rec_layout = QVBoxLayout(grp_recommended)
        rec_layout.addWidget(QLabel(
            "- NVIDIA GPU (CUDA): 변환 속도가 크게 향상됩니다\n"
            "  (GPU 없이도 CPU로 동작하지만 느릴 수 있습니다)"
        ))
        layout.addWidget(grp_recommended)

        # 화자 분리 기능
        grp_diar = QGroupBox("화자 분리 기능 사용 시 (선택)")
        diar_layout = QVBoxLayout(grp_diar)
        diar_layout.addWidget(QLabel(
            "- HuggingFace 계정 및 토큰 (무료)\n"
            "- pyannote 모델 라이선스 동의 (무료)\n"
            "- 설정 버튼에서 토큰을 등록할 수 있습니다"
        ))
        layout.addWidget(grp_diar)

        layout.addSpacing(8)

        # 다시 보지 않기 체크박스 + 확인 버튼
        bottom = QHBoxLayout()
        self.chk_dont_show = QCheckBox("다시 표시하지 않기")
        bottom.addWidget(self.chk_dont_show)
        bottom.addStretch()
        btn_ok = QPushButton("확인")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self._on_ok)
        bottom.addWidget(btn_ok)
        layout.addLayout(bottom)

    def _on_ok(self):
        if self.chk_dont_show.isChecked():
            set_show_startup_guide(False)
        self.accept()


# ────────────────────────────────────────────
# 설정 다이얼로그
# ────────────────────────────────────────────

class SettingsDialog(QDialog):
    """앱 설정 다이얼로그. 일반 + 화자 분리 탭."""

    HF_MODELS = [
        ("pyannote/speaker-diarization-3.1", "https://hf.co/pyannote/speaker-diarization-3.1"),
        ("pyannote/segmentation-3.0", "https://hf.co/pyannote/segmentation-3.0"),
    ]

    WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "turbo", "large-v3-turbo"]

    def __init__(self, db_path: str, parent=None, initial_tab: int = 0):
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.setMinimumWidth(520)
        self._build_ui(db_path, initial_tab)

    def _build_ui(self, db_path: str, initial_tab: int):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # ── 일반 탭 ──
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)

        grp_whisper = QGroupBox("Whisper 모델")
        whisper_layout = QHBoxLayout(grp_whisper)
        whisper_layout.addWidget(QLabel("기본 모델:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(self.WHISPER_MODELS)
        current_model = get_whisper_model()
        idx = self.combo_model.findText(current_model)
        if idx >= 0:
            self.combo_model.setCurrentIndex(idx)
        whisper_layout.addWidget(self.combo_model)
        whisper_layout.addStretch()
        general_layout.addWidget(grp_whisper)

        grp_data = QGroupBox("데이터 저장 위치")
        data_layout = QVBoxLayout(grp_data)
        db_folder = os.path.dirname(db_path)
        lbl_path = QLabel(db_folder)
        lbl_path.setWordWrap(True)
        lbl_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        data_layout.addWidget(lbl_path)
        btn_open = QPushButton("폴더 열기")
        btn_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(db_folder)))
        data_layout.addWidget(btn_open)
        general_layout.addWidget(grp_data)

        # 모델 캐시 위치
        grp_cache = QGroupBox("모델 캐시 위치")
        cache_layout = QVBoxLayout(grp_cache)
        cache_folder = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        lbl_cache = QLabel(cache_folder)
        lbl_cache.setWordWrap(True)
        lbl_cache.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cache_layout.addWidget(lbl_cache)
        btn_open_cache = QPushButton("폴더 열기")
        btn_open_cache.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(cache_folder)))
        cache_layout.addWidget(btn_open_cache)
        general_layout.addWidget(grp_cache)

        general_layout.addStretch()
        tabs.addTab(general_tab, "일반")

        # ── 화자 분리 탭 ──
        diar_tab = QWidget()
        diar_layout = QVBoxLayout(diar_tab)

        grp_token = QGroupBox("HuggingFace 토큰")
        token_layout = QVBoxLayout(grp_token)

        token_input_row = QHBoxLayout()
        token_input_row.addWidget(QLabel("토큰:"))
        self.edit_token = QLineEdit()
        self.edit_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_token.setPlaceholderText("hf_...")
        existing = get_hf_token()
        if existing:
            self.edit_token.setText(existing)
        token_input_row.addWidget(self.edit_token, stretch=1)
        token_layout.addLayout(token_input_row)

        verify_row = QHBoxLayout()
        self.btn_verify = QPushButton("토큰 검증")
        self.btn_verify.clicked.connect(self._on_verify)
        self.btn_delete_token = QPushButton("토큰 삭제")
        self.btn_delete_token.clicked.connect(self._on_delete_token)
        self.lbl_verify_result = QLabel("")
        verify_row.addWidget(self.btn_verify)
        verify_row.addWidget(self.btn_delete_token)
        verify_row.addWidget(self.lbl_verify_result, stretch=1)
        token_layout.addLayout(verify_row)

        diar_layout.addWidget(grp_token)

        grp_license = QGroupBox("모델 라이선스 동의")
        license_layout = QVBoxLayout(grp_license)
        lbl_license = QLabel("아래 모델 페이지에서 각각 라이선스에 동의해야 합니다.")
        lbl_license.setWordWrap(True)
        license_layout.addWidget(lbl_license)
        for model_name, url in self.HF_MODELS:
            btn = QPushButton(f"  {model_name} 페이지 열기")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, u=url: webbrowser.open(u))
            license_layout.addWidget(btn)
        diar_layout.addWidget(grp_license)

        diar_layout.addStretch()
        tabs.addTab(diar_tab, "화자 분리")

        tabs.setCurrentIndex(initial_tab)
        layout.addWidget(tabs)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_cancel = QPushButton("취소")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        btn_save = QPushButton("저장")
        btn_save.setDefault(True)
        btn_save.clicked.connect(self._on_save)
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)

    def _on_verify(self):
        token = self.edit_token.text().strip()
        if not token:
            self.lbl_verify_result.setText("토큰을 입력하세요.")
            self.lbl_verify_result.setStyleSheet("color: red;")
            return

        self.btn_verify.setEnabled(False)
        self.lbl_verify_result.setText("검증 중...")
        self.lbl_verify_result.setStyleSheet("color: gray;")
        QApplication.processEvents()

        try:
            import huggingface_hub
            api = huggingface_hub.HfApi(token=token)
            api.model_info("pyannote/speaker-diarization-3.1")
            api.model_info("pyannote/segmentation-3.0")
            self.lbl_verify_result.setText("토큰 유효! 두 모델 모두 접근 가능합니다.")
            self.lbl_verify_result.setStyleSheet("color: green;")
        except Exception as e:
            err = str(e)
            if "401" in err or "Unauthorized" in err:
                self.lbl_verify_result.setText("토큰이 유효하지 않습니다.")
            elif "403" in err or "Access" in err or "gated" in err.lower():
                self.lbl_verify_result.setText("모델 라이선스 동의가 필요합니다.")
            elif "404" in err:
                self.lbl_verify_result.setText("모델을 찾을 수 없습니다.")
            else:
                self.lbl_verify_result.setText(f"오류: {err[:100]}")
            self.lbl_verify_result.setStyleSheet("color: red;")
        finally:
            self.btn_verify.setEnabled(True)

    def _on_delete_token(self):
        delete_hf_token()
        self.edit_token.clear()
        self.lbl_verify_result.setText("토큰이 삭제되었습니다.")
        self.lbl_verify_result.setStyleSheet("color: gray;")

    def _on_save(self):
        set_whisper_model(self.combo_model.currentText())
        token = self.edit_token.text().strip()
        if token:
            set_hf_token(token)
        self.accept()


# ────────────────────────────────────────────
# 트랜스크립션 설정 다이얼로그
# ────────────────────────────────────────────

class TranscriptionSettingsDialog(QDialog):
    """트랜스크립션 시작 전 설정 다이얼로그."""

    WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "turbo", "large-v3-turbo"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("트랜스크립션 설정")
        self.setMinimumWidth(450)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        grp_model = QGroupBox("Whisper 모델")
        model_layout = QHBoxLayout(grp_model)
        model_layout.addWidget(QLabel("모델:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(self.WHISPER_MODELS)
        current = get_whisper_model()
        idx = self.combo_model.findText(current)
        if idx >= 0:
            self.combo_model.setCurrentIndex(idx)
        model_layout.addWidget(self.combo_model)
        model_layout.addStretch()
        layout.addWidget(grp_model)

        grp_diar = QGroupBox("화자 분리")
        diar_layout = QVBoxLayout(grp_diar)

        self.chk_diarization = QCheckBox("화자 분리 사용")
        self.chk_diarization.toggled.connect(self._on_diar_toggled)
        diar_layout.addWidget(self.chk_diarization)

        self.speaker_widget = QWidget()
        speaker_layout = QVBoxLayout(self.speaker_widget)
        speaker_layout.setContentsMargins(20, 0, 0, 0)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("화자 수:"))
        self.combo_speaker_mode = QComboBox()
        self.combo_speaker_mode.addItems(["자동 감지", "직접 지정"])
        self.combo_speaker_mode.currentIndexChanged.connect(self._on_speaker_mode_changed)
        mode_row.addWidget(self.combo_speaker_mode)
        mode_row.addStretch()
        speaker_layout.addLayout(mode_row)

        self.manual_widget = QWidget()
        manual_layout = QVBoxLayout(self.manual_widget)
        manual_layout.setContentsMargins(0, 0, 0, 0)

        exact_row = QHBoxLayout()
        self.chk_exact = QCheckBox("정확한 화자 수:")
        self.chk_exact.toggled.connect(self._on_exact_toggled)
        self.spin_exact = QSpinBox()
        self.spin_exact.setRange(1, 20)
        self.spin_exact.setValue(2)
        self.spin_exact.setEnabled(False)
        exact_row.addWidget(self.chk_exact)
        exact_row.addWidget(self.spin_exact)
        exact_row.addStretch()
        manual_layout.addLayout(exact_row)

        self.range_widget = QWidget()
        range_layout = QHBoxLayout(self.range_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.addWidget(QLabel("최소:"))
        self.spin_min = QSpinBox()
        self.spin_min.setRange(1, 20)
        self.spin_min.setValue(1)
        range_layout.addWidget(self.spin_min)
        range_layout.addWidget(QLabel("최대:"))
        self.spin_max = QSpinBox()
        self.spin_max.setRange(1, 20)
        self.spin_max.setValue(10)
        range_layout.addWidget(self.spin_max)
        range_layout.addStretch()
        manual_layout.addWidget(self.range_widget)

        self.lbl_warning = QLabel("")
        self.lbl_warning.setStyleSheet("color: red;")
        manual_layout.addWidget(self.lbl_warning)

        self.manual_widget.setVisible(False)
        speaker_layout.addWidget(self.manual_widget)

        self.speaker_widget.setVisible(False)
        diar_layout.addWidget(self.speaker_widget)

        layout.addWidget(grp_diar)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_cancel = QPushButton("취소")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        self.btn_start = QPushButton("시작")
        self.btn_start.setDefault(True)
        self.btn_start.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_start)
        layout.addLayout(btn_layout)

        self.spin_min.valueChanged.connect(self._validate)
        self.spin_max.valueChanged.connect(self._validate)

    def _on_diar_toggled(self, checked: bool):
        self.speaker_widget.setVisible(checked)

    def _on_speaker_mode_changed(self, index: int):
        self.manual_widget.setVisible(index == 1)

    def _on_exact_toggled(self, checked: bool):
        self.spin_exact.setEnabled(checked)
        self.range_widget.setEnabled(not checked)
        self._validate()

    def _validate(self):
        if self.chk_exact.isChecked():
            self.lbl_warning.setText("")
            self.btn_start.setEnabled(True)
            return
        if self.spin_min.value() > self.spin_max.value():
            self.lbl_warning.setText("최소 화자 수가 최대보다 클 수 없습니다.")
            self.btn_start.setEnabled(False)
        else:
            self.lbl_warning.setText("")
            self.btn_start.setEnabled(True)

    def get_settings(self) -> dict:
        model = self.combo_model.currentText()
        use_diar = self.chk_diarization.isChecked()
        num_speakers = None
        min_speakers = None
        max_speakers = None

        if use_diar and self.combo_speaker_mode.currentIndex() == 1:
            if self.chk_exact.isChecked():
                num_speakers = self.spin_exact.value()
            else:
                min_speakers = self.spin_min.value()
                max_speakers = self.spin_max.value()

        return {
            "model_name": model,
            "use_diarization": use_diar,
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }


# ────────────────────────────────────────────
# 메인 윈도우
# ────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self._worker = None
        self._thread = None
        self._current_tid = None

        self.setWindowTitle("Video Transcriber")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)

        self._build_ui()
        self._load_list()

        if get_show_startup_guide():
            QTimer.singleShot(0, self._show_startup_guide)

    def _show_startup_guide(self):
        StartupGuideDialog(self).exec()

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

        self.btn_add = QPushButton("+ 파일 추가")
        self.btn_add.setFixedHeight(36)
        self.btn_add.clicked.connect(self._on_add_video)
        left_layout.addWidget(self.btn_add)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_select_item)
        left_layout.addWidget(self.list_widget)

        # Bottom buttons
        left_btn_row = QHBoxLayout()
        self.btn_delete = QPushButton("선택 삭제")
        self.btn_delete.clicked.connect(self._on_delete)
        left_btn_row.addWidget(self.btn_delete)

        self.btn_settings = QPushButton("설정")
        self.btn_settings.clicked.connect(self._on_settings)
        left_btn_row.addWidget(self.btn_settings)

        left_layout.addLayout(left_btn_row)

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

        # Copy button + Speaker management button
        btn_row = QHBoxLayout()
        self.btn_speakers = QPushButton("화자 관리")
        self.btn_speakers.clicked.connect(self._on_manage_speakers)
        self.btn_speakers.setVisible(False)
        btn_row.addWidget(self.btn_speakers)
        btn_row.addStretch()
        self.btn_copy = QPushButton("텍스트 복사")
        self.btn_copy.clicked.connect(self._on_copy)
        btn_row.addWidget(self.btn_copy)
        right_layout.addLayout(btn_row)

        # Empty state
        self.lbl_empty = QLabel("영상 또는 음성 파일을 추가하면 여기에 결과가 표시됩니다.")
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
        self.btn_speakers.setVisible(False)
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

        self._current_tid = tid
        self._show_detail(True)
        self.lbl_title.setText(data["filename"])
        dur = format_duration(data.get("duration"))
        date = data["created_at"][:10]
        self.lbl_info.setText(f"날짜: {date}    길이: {dur}")

        # Timeline with speakers
        lines = []
        for seg in data.get("segments", []):
            ts_start = format_timestamp(seg["start"])
            ts_end = format_timestamp(seg["end"])
            speaker = seg.get("speaker")
            if speaker:
                lines.append(f"[{ts_start} ~ {ts_end}]  [{speaker}]  {seg['text']}")
            else:
                lines.append(f"[{ts_start} ~ {ts_end}]  {seg['text']}")
        self.txt_timeline.setPlainText("\n".join(lines))

        # Full text with speaker grouping
        self.txt_fulltext.setPlainText(self._build_full_text(data.get("segments", [])))

        # 화자 관리 버튼: 화자 정보가 있을 때만 표시
        has_speakers = any(s.get("speaker") for s in data.get("segments", []))
        self.btn_speakers.setVisible(has_speakers)

    def _build_full_text(self, segments: list[dict]) -> str:
        if not segments:
            return ""
        has_speakers = any(s.get("speaker") for s in segments)
        if not has_speakers:
            return " ".join(s["text"] for s in segments)

        lines = []
        current_speaker = None
        current_texts = []
        for seg in segments:
            speaker = seg.get("speaker")
            if speaker != current_speaker:
                if current_texts:
                    prefix = f"{current_speaker}: " if current_speaker else ""
                    lines.append(prefix + " ".join(current_texts))
                current_speaker = speaker
                current_texts = [seg["text"]]
            else:
                current_texts.append(seg["text"])
        if current_texts:
            prefix = f"{current_speaker}: " if current_speaker else ""
            lines.append(prefix + " ".join(current_texts))
        return "\n".join(lines)

    # --- Actions ---

    def _on_settings(self):
        dialog = SettingsDialog(self.db.db_path, self)
        dialog.exec()

    def _on_add_video(self):
        if self._thread and self._thread.isRunning():
            QMessageBox.warning(self, "진행 중", "현재 변환이 진행 중입니다. 완료 후 다시 시도하세요.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "미디어 파일 선택",
            "",
            "미디어 파일 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a);;영상 파일 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;음성 파일 (*.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a);;모든 파일 (*)",
        )
        if not path:
            return

        settings_dialog = TranscriptionSettingsDialog(self)
        if settings_dialog.exec() != QDialog.DialogCode.Accepted:
            return

        settings = settings_dialog.get_settings()
        hf_token = None

        if settings["use_diarization"]:
            hf_token = get_hf_token()
            if not hf_token:
                QMessageBox.information(
                    self, "토큰 필요",
                    "화자 분리에는 HuggingFace 토큰이 필요합니다.\n설정에서 토큰을 먼저 등록하세요.",
                )
                dialog = SettingsDialog(self.db.db_path, self, initial_tab=1)
                dialog.exec()
                hf_token = get_hf_token()
                if not hf_token:
                    reply = QMessageBox.question(
                        self, "화자 분리 없이 진행",
                        "토큰이 없어 화자 분리를 사용할 수 없습니다.\n텍스트 변환만 진행하시겠습니까?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return
                    settings["use_diarization"] = False

        self._start_transcription(path, settings, hf_token)

    def _start_transcription(self, video_path: str, settings: dict, hf_token: str | None = None):
        self.btn_add.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setVisible(True)
        self.lbl_status.setText("준비 중...")

        self._show_detail(True)
        self.lbl_title.setText(os.path.basename(video_path))
        self.lbl_info.setText("변환 진행 중...")
        self.txt_timeline.clear()
        self.txt_fulltext.clear()
        self._live_segments = []

        self._thread = QThread()
        self._worker = TranscriberWorker(
            video_path,
            use_diarization=settings["use_diarization"],
            hf_token=hf_token,
            model_name=settings.get("model_name", "medium"),
            num_speakers=settings.get("num_speakers"),
            min_speakers=settings.get("min_speakers"),
            max_speakers=settings.get("max_speakers"),
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.segment_ready.connect(self._on_segment_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.start()

    def _on_progress(self, percent: int, message: str):
        self.progress_bar.setValue(percent)
        self.lbl_status.setText(message)

    def _on_segment_ready(self, seg: dict):
        self._live_segments.append(seg)
        ts_start = format_timestamp(seg["start"])
        ts_end = format_timestamp(seg["end"])
        speaker = seg.get("speaker")
        if speaker:
            line = f"[{ts_start} ~ {ts_end}]  [{speaker}]  {seg['text']}"
        else:
            line = f"[{ts_start} ~ {ts_end}]  {seg['text']}"
        self.txt_timeline.appendPlainText(line)
        # Auto-scroll to bottom
        scrollbar = self.txt_timeline.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        # Update full text tab live
        self.txt_fulltext.setPlainText(self._build_full_text(self._live_segments))

    def _on_finished(self, result: dict):
        tid = self.db.add_transcription(
            filename=result["filename"],
            filepath=result["filepath"],
            duration=result["duration"],
            full_text=result["full_text"],
            segments=result["segments"],
        )
        self._current_tid = tid

        # Update display with final mapped labels
        dur = format_duration(result.get("duration"))
        self.lbl_info.setText(f"길이: {dur}")

        lines = []
        for seg in result["segments"]:
            ts_start = format_timestamp(seg["start"])
            ts_end = format_timestamp(seg["end"])
            speaker = seg.get("speaker")
            if speaker:
                lines.append(f"[{ts_start} ~ {ts_end}]  [{speaker}]  {seg['text']}")
            else:
                lines.append(f"[{ts_start} ~ {ts_end}]  {seg['text']}")
        self.txt_timeline.setPlainText("\n".join(lines))
        self.txt_fulltext.setPlainText(self._build_full_text(result["segments"]))

        # 화자 정보 있으면 버튼 표시
        has_speakers = any(s.get("speaker") for s in result.get("segments", []))
        self.btn_speakers.setVisible(has_speakers)

        self._load_list()
        self.list_widget.blockSignals(True)
        self.list_widget.setCurrentRow(0)
        self.list_widget.blockSignals(False)

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
        self._show_detail(False)
        QMessageBox.critical(self, "오류", f"변환 중 오류 발생:\n{message}")

    def _on_manage_speakers(self):
        if self._current_tid is None:
            return

        speakers = self.db.get_speakers(self._current_tid)
        if not speakers:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("화자 관리")
        dialog.setMinimumWidth(350)
        form = QFormLayout(dialog)

        edits = {}
        for speaker in speakers:
            edit = QLineEdit(speaker)
            edits[speaker] = edit
            form.addRow(f"{speaker}:", edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            renames = {}
            for old_name, edit in edits.items():
                new_name = edit.text().strip()
                if new_name and new_name != old_name:
                    renames[old_name] = new_name

            if renames:
                # Use temp names to avoid conflicts when swapping
                # (e.g. "화자 1"→"화자 2" and "화자 2"→"화자 1")
                import uuid
                temp_map = {}
                for old_name in renames:
                    temp_name = f"__temp_{uuid.uuid4().hex[:8]}"
                    self.db.update_speaker_name(self._current_tid, old_name, temp_name)
                    temp_map[temp_name] = renames[old_name]
                for temp_name, new_name in temp_map.items():
                    self.db.update_speaker_name(self._current_tid, temp_name, new_name)

            # Refresh display
            self._on_select_item(self.list_widget.currentRow())

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
