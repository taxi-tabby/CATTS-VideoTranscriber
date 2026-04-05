import os
import sys
import time
import webbrowser

from PySide6.QtCore import Qt, QThread, QTimer, QUrl, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QAction, QDesktopServices, QFont, QIcon, QShortcut, QKeySequence, QTextCursor, QPainter, QColor, QPen, QLinearGradient
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
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QSystemTrayIcon,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtCore import Signal as _Signal


class _DroppableTreeWidget(QTreeWidget):
    """QTreeWidget that emits itemDropped after a drag-and-drop operation."""
    itemDropped = _Signal()

    def dropEvent(self, event):
        super().dropEvent(event)
        self.itemDropped.emit()


class _GlowOverlay(QWidget):
    """변환 중 창 테두리에 흐르는 그라데이션 글로우 효과."""

    _COLORS = [
        QColor(123, 164, 212, 180),   # #7BA4D4
        QColor(155, 142, 212, 180),   # #9B8ED4
        QColor(125, 212, 212, 180),   # #7DD4D4
        QColor(155, 142, 212, 180),   # #9B8ED4
        QColor(123, 164, 212, 180),   # #7BA4D4
    ]
    _BORDER = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self._offset = 0.0
        self._opacity = 0.0

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(30)
        self._anim_timer.timeout.connect(self._tick)

        self._fade_anim = QPropertyAnimation(self, b"opacity", self)
        self._fade_anim.setDuration(500)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def _get_opacity(self):
        return self._opacity

    def _set_opacity(self, v):
        self._opacity = v
        self.update()

    opacity = Property(float, _get_opacity, _set_opacity)

    def start(self):
        self.show()
        self.raise_()
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self._opacity)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()
        self._anim_timer.start()

    def stop(self):
        self._anim_timer.stop()
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self._opacity)
        self._fade_anim.setEndValue(0.0)
        try:
            self._fade_anim.finished.disconnect(self.hide)
        except RuntimeError:
            pass
        self._fade_anim.finished.connect(self.hide)
        self._fade_anim.start()

    def _tick(self):
        self._offset = (self._offset + 0.008) % 1.0
        self.update()

    def paintEvent(self, _event):
        if self._opacity <= 0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(self._opacity)

        w, h = self.width(), self.height()
        b = self._BORDER
        perimeter = 2 * (w + h)

        def _pos_to_point(t):
            """0~1 비율을 테두리 위의 좌표로 변환."""
            d = t * perimeter
            if d < w:
                return d, 0
            d -= w
            if d < h:
                return w, d
            d -= h
            if d < w:
                return w - d, h
            d -= w
            return 0, h - d

        # 그라데이션 시작/끝 좌표
        t0 = self._offset
        t1 = (self._offset + 0.5) % 1.0
        x0, y0 = _pos_to_point(t0)
        x1, y1 = _pos_to_point(t1)

        grad = QLinearGradient(x0, y0, x1, y1)
        for i, c in enumerate(self._COLORS):
            grad.setColorAt(i / (len(self._COLORS) - 1), c)

        pen = QPen(grad, b)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        half = b / 2
        p.drawRoundedRect(half, half, w - b, h - b, 6, 6)
        p.end()


from src.config import (
    get_hf_token, set_hf_token, delete_hf_token,
    get_whisper_model, set_whisper_model,
    get_show_startup_guide, set_show_startup_guide,
    get_theme, set_theme,
    get_db_dir, set_db_dir,
    get_whisper_cache, set_whisper_cache,
    get_hf_cache, set_hf_cache,
    get_thread_config, set_thread_config,
)
from src.database import Database
from src.model_utils import get_model_display_name
from src.transcriber import TranscriberWorker
from src.version_checker import VersionCheckThread, RELEASES_PAGE


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

        title = QLabel("CATTS - Audio/Video to Text")
        title.setFont(QFont("", 13, QFont.Weight.Bold))
        layout.addWidget(title)

        tabs = QTabWidget()

        # ── 시작 안내 탭 ──
        setup_tab = QWidget()
        setup_layout = QVBoxLayout(setup_tab)

        info = QLabel("이 프로그램은 영상/음성 파일에서 텍스트를 추출합니다.")
        info.setWordWrap(True)
        setup_layout.addWidget(info)

        grp_required = QGroupBox("필수")
        req_layout = QVBoxLayout(grp_required)
        req_layout.addWidget(QLabel(
            "- 기본 모델(large-v3, 약 2.9GB)이 프로그램에 포함되어 있습니다\n"
            "- 다른 모델을 사용하려면 인터넷 연결이 필요합니다 (자동 다운로드)\n"
            "- 다운로드된 모델은 저장되므로 이후에는 오프라인으로 사용 가능합니다"
        ))
        setup_layout.addWidget(grp_required)

        grp_recommended = QGroupBox("권장")
        rec_layout = QVBoxLayout(grp_recommended)
        rec_layout.addWidget(QLabel(
            "- NVIDIA GPU (CUDA): 변환 속도가 크게 향상됩니다\n"
            "  (GPU 없이도 CPU로 동작하지만 느릴 수 있습니다)"
        ))
        setup_layout.addWidget(grp_recommended)

        grp_diar = QGroupBox("화자 분리 기능 사용 시 (선택)")
        diar_layout = QVBoxLayout(grp_diar)
        diar_layout.addWidget(QLabel(
            "- HuggingFace 계정 및 토큰 (무료)\n"
            '- 토큰 권한: "Read access to contents of all public gated repos"\n'
            "- pyannote 모델 라이선스 동의 (무료, 모델 페이지에서 Agree 클릭)\n"
            "- 설정 버튼에서 토큰 등록 및 상세 안내를 확인할 수 있습니다"
        ))
        setup_layout.addWidget(grp_diar)
        setup_layout.addStretch()
        tabs.addTab(setup_tab, "시작 안내")

        # ── 알아두기 탭 ──
        tips_tab = QWidget()
        tips_layout = QVBoxLayout(tips_tab)

        grp_perf = QGroupBox("인식 성능에 대하여")
        perf_layout = QVBoxLayout(grp_perf)
        lbl_perf = QLabel(
            "이 프로그램은 OpenAI Whisper 음성인식 모델을 사용합니다.\n\n"
            "높은 정확도를 보이는 경우:\n"
            "  - 회의, 강의, 인터뷰 등 사람 목소리 위주의 녹음\n"
            "  - 배경 소음이 적고 마이크 품질이 좋은 환경\n"
            "  - 한 명 또는 소수가 번갈아 말하는 대화\n\n"
            "정확도가 낮아질 수 있는 경우:\n"
            "  - 배경 음악이 크거나 노래가 포함된 영상\n"
            "  - 여러 사람이 동시에 말하는 상황\n"
            "  - 심한 소음, 울림, 저음질 녹음\n"
            "  - 속삭임이나 매우 빠른 말투"
        )
        lbl_perf.setWordWrap(True)
        perf_layout.addWidget(lbl_perf)
        tips_layout.addWidget(grp_perf)

        grp_speed = QGroupBox("변환 속도에 대하여")
        speed_layout = QVBoxLayout(grp_speed)
        lbl_speed = QLabel(
            "모든 변환은 클라우드가 아닌 사용자의 PC에서 직접 수행됩니다.\n"
            "따라서 PC 사양에 따라 변환 시간이 달라집니다.\n\n"
            "  - NVIDIA GPU가 있으면 실시간보다 빠르게 처리됩니다\n"
            "  - GPU 없이 CPU만 사용할 경우 영상 길이의 수 배가 걸릴 수 있습니다\n"
            "  - 큰 모델(large-v3)일수록 정확하지만 더 오래 걸립니다\n"
            "  - 빠른 처리가 필요하면 tiny, base 등 작은 모델을 사용해보세요\n\n"
            "변환 중에도 프로그램을 계속 사용할 수 있으며,\n"
            "다른 트랜스크립션을 살펴보다가 변환 중 항목으로 돌아올 수 있습니다."
        )
        lbl_speed.setWordWrap(True)
        speed_layout.addWidget(lbl_speed)
        tips_layout.addWidget(grp_speed)

        tips_layout.addStretch()
        tabs.addTab(tips_tab, "알아두기")

        layout.addWidget(tabs)

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
        self.combo_model.setMinimumWidth(320)
        current_model = get_whisper_model()
        current_idx = 0
        for i, name in enumerate(self.WHISPER_MODELS):
            self.combo_model.addItem(get_model_display_name(name), name)
            if name == current_model:
                current_idx = i
        self.combo_model.setCurrentIndex(current_idx)
        whisper_layout.addWidget(self.combo_model)
        whisper_layout.addStretch()
        general_layout.addWidget(grp_whisper)

        # 테마
        grp_theme = QGroupBox("테마")
        theme_layout = QHBoxLayout(grp_theme)
        theme_layout.addWidget(QLabel("테마:"))
        self.combo_theme = QComboBox()
        self.combo_theme.addItem("라이트", "light")
        self.combo_theme.addItem("다크", "dark")
        current_theme = get_theme()
        idx = self.combo_theme.findData(current_theme)
        if idx >= 0:
            self.combo_theme.setCurrentIndex(idx)
        theme_layout.addWidget(self.combo_theme)
        theme_layout.addStretch()
        general_layout.addWidget(grp_theme)

        # 시작 안내 다시 보기
        btn_show_guide = QPushButton("시작 안내 다시 보기")
        btn_show_guide.clicked.connect(self._on_show_guide)
        general_layout.addWidget(btn_show_guide)

        general_layout.addStretch()
        tabs.addTab(general_tab, "일반")

        # ── 저장 경로 탭 ──
        paths_tab = QWidget()
        paths_layout = QVBoxLayout(paths_tab)

        lbl_restart = QLabel("경로 변경 후 프로그램을 다시 시작해야 적용됩니다.")
        lbl_restart.setStyleSheet("color: #B07040; font-weight: bold;")
        lbl_restart.setWordWrap(True)
        paths_layout.addWidget(lbl_restart)

        # DB 저장 위치
        grp_db = QGroupBox("데이터 저장 위치 (DB)")
        db_layout = QVBoxLayout(grp_db)
        db_row = QHBoxLayout()
        self.edit_db_dir = QLineEdit(get_db_dir())
        db_row.addWidget(self.edit_db_dir, stretch=1)
        btn_db_browse = QPushButton("변경")
        btn_db_browse.clicked.connect(lambda: self._browse_dir(self.edit_db_dir))
        db_row.addWidget(btn_db_browse)
        btn_db_open = QPushButton("열기")
        btn_db_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.edit_db_dir.text())))
        db_row.addWidget(btn_db_open)
        db_layout.addLayout(db_row)
        paths_layout.addWidget(grp_db)

        # Whisper 모델 캐시
        grp_whisper_path = QGroupBox("Whisper 모델 저장 위치")
        whisper_path_layout = QVBoxLayout(grp_whisper_path)
        whisper_row = QHBoxLayout()
        self.edit_whisper_cache = QLineEdit(get_whisper_cache())
        whisper_row.addWidget(self.edit_whisper_cache, stretch=1)
        btn_whisper_browse = QPushButton("변경")
        btn_whisper_browse.clicked.connect(lambda: self._browse_dir(self.edit_whisper_cache))
        whisper_row.addWidget(btn_whisper_browse)
        btn_whisper_open = QPushButton("열기")
        btn_whisper_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.edit_whisper_cache.text())))
        whisper_row.addWidget(btn_whisper_open)
        whisper_path_layout.addLayout(whisper_row)
        paths_layout.addWidget(grp_whisper_path)

        # HuggingFace 모델 캐시 (화자 분리)
        grp_hf_path = QGroupBox("화자 분리 모델 저장 위치 (HuggingFace)")
        hf_path_layout = QVBoxLayout(grp_hf_path)
        hf_row = QHBoxLayout()
        self.edit_hf_cache = QLineEdit(get_hf_cache())
        hf_row.addWidget(self.edit_hf_cache, stretch=1)
        btn_hf_browse = QPushButton("변경")
        btn_hf_browse.clicked.connect(lambda: self._browse_dir(self.edit_hf_cache))
        hf_row.addWidget(btn_hf_browse)
        btn_hf_open = QPushButton("열기")
        btn_hf_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.edit_hf_cache.text())))
        hf_row.addWidget(btn_hf_open)
        hf_path_layout.addLayout(hf_row)
        paths_layout.addWidget(grp_hf_path)

        paths_layout.addStretch()
        tabs.addTab(paths_tab, "저장 경로")

        # ── 화자 분리 탭 ──
        diar_tab = QWidget()
        diar_layout = QVBoxLayout(diar_tab)

        grp_token = QGroupBox("HuggingFace 토큰")
        token_layout = QVBoxLayout(grp_token)

        lbl_token_guide = QLabel(
            "화자 분리에는 HuggingFace 토큰이 필요합니다.\n"
            "토큰 생성 시 아래 권한이 반드시 필요합니다:\n"
            '  - "Read access to contents of all public gated repos you can access"'
        )
        lbl_token_guide.setWordWrap(True)
        token_layout.addWidget(lbl_token_guide)

        btn_create_token = QPushButton("HuggingFace 토큰 생성 페이지 열기")
        btn_create_token.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_create_token.clicked.connect(lambda: webbrowser.open("https://huggingface.co/settings/tokens"))
        token_layout.addWidget(btn_create_token)

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

        grp_license = QGroupBox("모델 라이선스 동의 (필수)")
        license_layout = QVBoxLayout(grp_license)
        lbl_license = QLabel(
            "아래 두 모델 페이지에 각각 접속하여 라이선스에 동의해야 합니다.\n"
            "(HuggingFace 로그인 후 모델 페이지 상단의 'Agree' 버튼 클릭)"
        )
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

        # ── 라이선스 탭 ──
        license_tab = QWidget()
        lic_layout = QVBoxLayout(license_tab)

        lic_table = QTableWidget()
        lic_table.setColumnCount(4)
        lic_table.setHorizontalHeaderLabels(["라이브러리", "용도", "라이선스", "버전"])
        lic_table.horizontalHeader().setStretchLastSection(True)
        lic_table.horizontalHeader().setSectionResizeMode(0, lic_table.horizontalHeader().ResizeMode.ResizeToContents)
        lic_table.horizontalHeader().setSectionResizeMode(1, lic_table.horizontalHeader().ResizeMode.Stretch)
        lic_table.horizontalHeader().setSectionResizeMode(2, lic_table.horizontalHeader().ResizeMode.ResizeToContents)
        lic_table.horizontalHeader().setSectionResizeMode(3, lic_table.horizontalHeader().ResizeMode.ResizeToContents)
        lic_table.setEditTriggers(lic_table.EditTrigger.NoEditTriggers)
        lic_table.setSelectionBehavior(lic_table.SelectionBehavior.SelectRows)
        lic_table.verticalHeader().setVisible(False)

        _licenses = [
            ("OpenAI Whisper", "음성 인식 (Speech-to-Text)", "MIT", "openai-whisper"),
            ("PyTorch", "딥러닝 프레임워크", "BSD-3-Clause", "torch"),
            ("torchaudio", "오디오 처리", "BSD-2-Clause", "torchaudio"),
            ("pyannote.audio", "화자 임베딩 추출", "MIT", "pyannote.audio"),
            ("Demucs", "보컬/배경음 분리", "MIT", "demucs"),
            ("SpeechBrain", "음성 처리 툴킷", "Apache-2.0", "speechbrain"),
            ("Silero VAD", "음성 구간 검출 (VAD)", "MIT", "—"),
            ("PySide6 (Qt)", "GUI 프레임워크", "LGPL-3.0", "PySide6"),
            ("NumPy", "수치 연산", "BSD-3-Clause", "numpy"),
            ("SciPy", "과학 연산", "BSD-3-Clause", "scipy"),
            ("scikit-learn", "클러스터링 (화자 분리)", "BSD-3-Clause", "scikit-learn"),
            ("noisereduce", "배경 소음 제거", "MIT", "noisereduce"),
            ("SoundFile", "오디오 파일 읽기/쓰기", "BSD-3-Clause", "soundfile"),
            ("imageio-ffmpeg", "FFmpeg 영상 처리", "BSD-2-Clause", "imageio-ffmpeg"),
            ("julius", "오디오 DSP 필터", "MIT", "julius"),
            ("OpenUnmix", "음원 분리 모델", "MIT", "openunmix"),
            ("HuggingFace Hub", "모델 다운로드", "Apache-2.0", "huggingface-hub"),
        ]

        lic_table.setRowCount(len(_licenses))
        for row, (name, purpose, lic, pkg) in enumerate(_licenses):
            ver = "—"
            if pkg != "—":
                try:
                    import importlib.metadata
                    ver = importlib.metadata.version(pkg)
                except Exception:
                    ver = "—"
            lic_table.setItem(row, 0, QTableWidgetItem(name))
            lic_table.setItem(row, 1, QTableWidgetItem(purpose))
            lic_table.setItem(row, 2, QTableWidgetItem(lic))
            lic_table.setItem(row, 3, QTableWidgetItem(ver))

        lic_layout.addWidget(lic_table)

        lic_note = QLabel("이 프로그램은 위 오픈소스 라이브러리를 사용하여 제작되었습니다.")
        lic_note.setStyleSheet("color: gray; font-size: 11px;")
        lic_layout.addWidget(lic_note)

        tabs.addTab(license_tab, "라이선스")

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
                self.lbl_verify_result.setText(
                    "토큰이 유효하지 않습니다. 토큰 값을 다시 확인하세요."
                )
            elif "403" in err or "Access" in err or "gated" in err.lower():
                self.lbl_verify_result.setText(
                    "모델 접근 불가: 아래 모델 페이지에서 라이선스에 동의하세요."
                )
            elif "404" in err:
                self.lbl_verify_result.setText("모델을 찾을 수 없습니다.")
            elif "locate" in err.lower() or "connection" in err.lower():
                self.lbl_verify_result.setText(
                    "인터넷 연결을 확인하세요."
                )
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

    def _on_show_guide(self):
        StartupGuideDialog(self).exec()

    def _browse_dir(self, line_edit: QLineEdit):
        current = line_edit.text()
        path = QFileDialog.getExistingDirectory(self, "폴더 선택", current)
        if path:
            line_edit.setText(path)

    def _on_save(self):
        set_whisper_model(self.combo_model.currentData())
        new_theme = self.combo_theme.currentData()
        set_theme(new_theme)
        token = self.edit_token.text().strip()
        if token:
            set_hf_token(token)

        # 경로 저장
        need_restart = False
        new_db = self.edit_db_dir.text().strip()
        new_whisper = self.edit_whisper_cache.text().strip()
        new_hf = self.edit_hf_cache.text().strip()

        if new_db and new_db != get_db_dir():
            set_db_dir(new_db)
            need_restart = True
        if new_whisper and new_whisper != get_whisper_cache():
            set_whisper_cache(new_whisper)
            need_restart = True
        if new_hf and new_hf != get_hf_cache():
            set_hf_cache(new_hf)
            need_restart = True

        if need_restart:
            QMessageBox.information(self, "재시작 필요", "경로 변경은 프로그램을 다시 시작하면 적용됩니다.")
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
        self.combo_model.setMinimumWidth(320)
        current = get_whisper_model()
        current_idx = 0
        for i, name in enumerate(self.WHISPER_MODELS):
            self.combo_model.addItem(get_model_display_name(name), name)
            if name == current:
                current_idx = i
        self.combo_model.setCurrentIndex(current_idx)
        model_layout.addWidget(self.combo_model)
        model_layout.addStretch()
        layout.addWidget(grp_model)

        # 언어 선택
        grp_lang = QGroupBox("언어")
        lang_layout = QHBoxLayout(grp_lang)
        lang_layout.addWidget(QLabel("언어:"))
        self.combo_lang = QComboBox()
        self._languages = [
            ("auto", "자동 감지"),
            ("ko", "한국어"),
            ("en", "English"),
            ("ja", "日本語"),
            ("zh", "中文"),
            ("es", "Español"),
            ("fr", "Français"),
            ("de", "Deutsch"),
            ("ru", "Русский"),
            ("pt", "Português"),
            ("it", "Italiano"),
        ]
        for code, name in self._languages:
            self.combo_lang.addItem(name, code)
        self.combo_lang.setCurrentIndex(1)  # 기본값: 한국어
        lang_layout.addWidget(self.combo_lang)
        lang_layout.addStretch()
        layout.addWidget(grp_lang)

        # ── 분석 프로파일 (전처리 + 화자분리 공통) ──
        grp_profile = QGroupBox("분석 프로파일")
        profile_layout = QHBoxLayout(grp_profile)
        self.combo_profile = QComboBox()
        self.combo_profile.addItem("인터뷰/대화 — 깨끗한 음성 위주", "interview")
        self.combo_profile.addItem("영상/영화/노래 — 배경음악, 효과음 포함", "noisy")
        self.combo_profile.setToolTip(
            "인터뷰/대화: 일반 전처리\n"
            "영상/영화/노래: Demucs 보컬 분리로 배경음 제거 후 처리"
        )
        profile_layout.addWidget(self.combo_profile)
        layout.addWidget(grp_profile)

        # ── 교정 사전 ──
        grp_dict = QGroupBox("교정 사전")
        dict_layout = QHBoxLayout(grp_dict)
        self.combo_dict = QComboBox()
        self.combo_dict.addItem("사용 안 함", None)
        self._load_dict_list()
        dict_layout.addWidget(self.combo_dict)
        btn_manage_dict = QPushButton("관리...")
        btn_manage_dict.clicked.connect(self._open_dict_manager)
        dict_layout.addWidget(btn_manage_dict)
        layout.addWidget(grp_dict)

        # ── 화자 분리 ──
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

        # ── 처리 성능 ──
        tc = get_thread_config()

        grp_thread = QGroupBox("처리 성능")
        thread_layout = QVBoxLayout(grp_thread)

        # 음성 인식 스레드
        wt_row = QHBoxLayout()
        wt_row.addWidget(QLabel("음성 인식:"))
        self.chk_whisper_multi = QCheckBox("멀티스레드")
        self.chk_whisper_multi.setChecked(tc["whisper_mode"] == "multi")
        self.chk_whisper_multi.toggled.connect(lambda c: self.whisper_thread_opts.setVisible(c))
        wt_row.addWidget(self.chk_whisper_multi)
        wt_row.addStretch()
        thread_layout.addLayout(wt_row)

        self.whisper_thread_opts = QWidget()
        wto_layout = QHBoxLayout(self.whisper_thread_opts)
        wto_layout.setContentsMargins(20, 0, 0, 0)
        wto_layout.addWidget(QLabel("최소:"))
        self.spin_wt_min = QSpinBox()
        self.spin_wt_min.setRange(2, 64)
        self.spin_wt_min.setValue(tc["whisper_min"])
        wto_layout.addWidget(self.spin_wt_min)
        wto_layout.addWidget(QLabel("최대:"))
        self.spin_wt_max = QSpinBox()
        self.spin_wt_max.setRange(0, 64)
        self.spin_wt_max.setValue(tc["whisper_max"])
        self.spin_wt_max.setSpecialValueText("무제한")
        wto_layout.addWidget(self.spin_wt_max)
        wto_layout.addStretch()
        self.whisper_thread_opts.setVisible(tc["whisper_mode"] == "multi")
        thread_layout.addWidget(self.whisper_thread_opts)

        # 화자 분리 스레드
        dt_row = QHBoxLayout()
        dt_row.addWidget(QLabel("화자 분리:"))
        self.chk_diar_multi = QCheckBox("멀티스레드")
        self.chk_diar_multi.setChecked(tc["diar_mode"] == "multi")
        self.chk_diar_multi.toggled.connect(lambda c: self.diar_thread_opts.setVisible(c))
        dt_row.addWidget(self.chk_diar_multi)
        dt_row.addStretch()
        thread_layout.addLayout(dt_row)

        self.diar_thread_opts = QWidget()
        dto_layout = QHBoxLayout(self.diar_thread_opts)
        dto_layout.setContentsMargins(20, 0, 0, 0)
        dto_layout.addWidget(QLabel("최소:"))
        self.spin_dt_min = QSpinBox()
        self.spin_dt_min.setRange(2, 64)
        self.spin_dt_min.setValue(tc["diar_min"])
        dto_layout.addWidget(self.spin_dt_min)
        dto_layout.addWidget(QLabel("최대:"))
        self.spin_dt_max = QSpinBox()
        self.spin_dt_max.setRange(0, 64)
        self.spin_dt_max.setValue(tc["diar_max"])
        self.spin_dt_max.setSpecialValueText("무제한")
        dto_layout.addWidget(self.spin_dt_max)
        dto_layout.addStretch()
        self.diar_thread_opts.setVisible(tc["diar_mode"] == "multi")
        thread_layout.addWidget(self.diar_thread_opts)

        layout.addWidget(grp_thread)

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
        model = self.combo_model.currentData()
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

        language = self.combo_lang.currentData()

        # 스레드 설정 저장
        whisper_mode = "multi" if self.chk_whisper_multi.isChecked() else "single"
        diar_mode = "multi" if self.chk_diar_multi.isChecked() else "single"
        tc = {
            "whisper_mode": whisper_mode,
            "whisper_min": self.spin_wt_min.value(),
            "whisper_max": self.spin_wt_max.value(),
            "diar_mode": diar_mode,
            "diar_min": self.spin_dt_min.value(),
            "diar_max": self.spin_dt_max.value(),
        }
        set_thread_config(tc)

        # 실제 워커 수 계산
        cpu_count = os.cpu_count() or 4
        if whisper_mode == "multi":
            wt_max = tc["whisper_max"]
            whisper_workers = wt_max if wt_max > 0 else max(tc["whisper_min"], cpu_count)
        else:
            whisper_workers = 1

        if diar_mode == "multi":
            dt_max = tc["diar_max"]
            diar_threads = dt_max if dt_max > 0 else max(tc["diar_min"], cpu_count)
        else:
            diar_threads = 1

        profile = self.combo_profile.currentData()

        return {
            "model_name": model,
            "language": language,
            "use_diarization": use_diar,
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "whisper_workers": whisper_workers,
            "diar_threads": diar_threads,
            "profile": profile,
            "correction_dict_id": self.combo_dict.currentData(),
        }

    def _load_dict_list(self):
        """교정 사전 목록을 콤보박스에 로드한다."""
        current_id = self.combo_dict.currentData()
        self.combo_dict.clear()
        self.combo_dict.addItem("사용 안 함", None)
        try:
            parent = self.parent()
            if parent and hasattr(parent, "db"):
                for d in parent.db.list_correction_dicts():
                    label = f"{d['name']} ({d['entry_count']}개)"
                    self.combo_dict.addItem(label, d["id"])
        except Exception:
            pass
        for i in range(self.combo_dict.count()):
            if self.combo_dict.itemData(i) == current_id:
                self.combo_dict.setCurrentIndex(i)
                break

    def _open_dict_manager(self):
        """교정 사전 관리 다이얼로그를 연다."""
        parent = self.parent()
        if not parent or not hasattr(parent, "db"):
            return
        dlg = CorrectionDictDialog(parent.db, self)
        dlg.exec()
        self._load_dict_list()


class CorrectionDictDialog(QDialog):
    """교정 사전 관리 다이얼로그."""

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("교정 사전 관리")
        self.setMinimumSize(800, 500)
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # 왼쪽: 사전 목록
        left = QVBoxLayout()
        self.list_dicts = QListWidget()
        self.list_dicts.currentRowChanged.connect(self._on_dict_selected)
        left.addWidget(QLabel("사전 목록"))
        left.addWidget(self.list_dicts)

        btn_row = QHBoxLayout()
        btn_analyze = QPushButton("미디어 분석으로 생성...")
        btn_analyze.clicked.connect(self._analyze_media)
        btn_row.addWidget(btn_analyze)
        left.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        btn_add = QPushButton("+ 빈 사전")
        btn_add.clicked.connect(self._add_dict)
        btn_del = QPushButton("삭제")
        btn_del.clicked.connect(self._delete_dict)
        btn_row2.addWidget(btn_add)
        btn_row2.addWidget(btn_del)
        left.addLayout(btn_row2)
        layout.addLayout(left, 1)

        # 오른쪽: 사전 정보 + 항목
        right = QVBoxLayout()

        # 체크섬 정보
        info_row = QHBoxLayout()
        self.lbl_media = QLabel("미디어: —")
        self.lbl_checksum = QLabel("체크섬: —")
        self.lbl_checksum.setStyleSheet("color: gray; font-size: 11px;")
        btn_change_checksum = QPushButton("체크섬 변경...")
        btn_change_checksum.clicked.connect(self._change_checksum)
        info_row.addWidget(self.lbl_media)
        info_row.addStretch()
        info_row.addWidget(btn_change_checksum)
        right.addLayout(info_row)
        right.addWidget(self.lbl_checksum)

        # 항목 테이블 (화자, 인식된 단어, 교정, 빈도)
        right.addWidget(QLabel("교정 항목 (빈도순 — '교정' 열을 편집하면 변환 시 적용됩니다)"))
        self.table_entries = QTableWidget()
        self.table_entries.setColumnCount(4)
        self.table_entries.setHorizontalHeaderLabels(["화자", "인식된 단어", "교정", "빈도"])
        header = self.table_entries.horizontalHeader()
        header.setSectionResizeMode(0, header.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, header.ResizeMode.Stretch)
        header.setSectionResizeMode(2, header.ResizeMode.Stretch)
        header.setSectionResizeMode(3, header.ResizeMode.ResizeToContents)
        self.table_entries.setSelectionBehavior(self.table_entries.SelectionBehavior.SelectRows)
        self.table_entries.verticalHeader().setVisible(False)
        right.addWidget(self.table_entries)

        entry_btn_row = QHBoxLayout()
        btn_add_entry = QPushButton("+ 항목 추가")
        btn_add_entry.clicked.connect(self._add_entry)
        btn_del_entry = QPushButton("항목 삭제")
        btn_del_entry.clicked.connect(self._delete_entry)
        btn_save = QPushButton("저장")
        btn_save.clicked.connect(self._save_entries)
        entry_btn_row.addWidget(btn_add_entry)
        entry_btn_row.addWidget(btn_del_entry)
        entry_btn_row.addStretch()
        entry_btn_row.addWidget(btn_save)
        right.addLayout(entry_btn_row)
        layout.addLayout(right, 3)

    def _refresh(self):
        self.list_dicts.clear()
        self._dicts = self.db.list_correction_dicts()
        for d in self._dicts:
            media = f" [{d.get('media_filename') or '수동'}]" if d.get('media_filename') else ""
            self.list_dicts.addItem(f"{d['name']}{media} ({d['entry_count']}개)")
        self.table_entries.setRowCount(0)
        self.lbl_media.setText("미디어: —")
        self.lbl_checksum.setText("체크섬: —")

    def _current_dict_id(self):
        row = self.list_dicts.currentRow()
        if row < 0 or row >= len(self._dicts):
            return None
        return self._dicts[row]["id"]

    def _on_dict_selected(self, row):
        dict_id = self._current_dict_id()
        if dict_id is None:
            self.table_entries.setRowCount(0)
            self.lbl_media.setText("미디어: —")
            self.lbl_checksum.setText("체크섬: —")
            return

        d = self.db.get_correction_dict(dict_id)
        if d:
            self.lbl_media.setText(f"미디어: {d.get('media_filename') or '(없음)'}")
            cs = d.get("media_checksum") or "—"
            self.lbl_checksum.setText(f"체크섬: {cs[:16]}..." if len(cs) > 16 else f"체크섬: {cs}")

        entries = self.db.get_correction_entries(dict_id)

        # 고유 단어(화자+단어)별로 그룹화하여 UI에 표시
        # DB에는 모든 등장 위치가 개별 행으로 있지만, UI에서는 1행만 표시
        seen = {}  # (speaker, wrong) → row index
        display_rows = []
        for e in entries:
            key = (e.get("speaker"), e["wrong"])
            if key not in seen:
                seen[key] = len(display_rows)
                display_rows.append(e)

        self.table_entries.setRowCount(len(display_rows))
        for i, e in enumerate(display_rows):
            speaker_item = QTableWidgetItem(e.get("speaker") or "—")
            speaker_item.setFlags(speaker_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            wrong_item = QTableWidgetItem(e["wrong"])
            correct_item = QTableWidgetItem(e["correct"])
            freq_item = QTableWidgetItem(str(e.get("frequency", 1)))
            freq_item.setFlags(freq_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            wrong_item.setData(Qt.ItemDataRole.UserRole, e["id"])
            wrong_item.setData(Qt.ItemDataRole.UserRole + 1, {
                "speaker": e.get("speaker"),
                "frequency": e.get("frequency", 1),
                "is_corrected": e.get("is_corrected", 0),
            })

            self.table_entries.setItem(i, 0, speaker_item)
            self.table_entries.setItem(i, 1, wrong_item)
            self.table_entries.setItem(i, 2, correct_item)
            self.table_entries.setItem(i, 3, freq_item)

    def _analyze_media(self):
        """미디어 파일을 선택하고 분석하여 사전을 생성한다."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "미디어 파일 선택", "",
            "미디어 파일 (*.mp4 *.mkv *.avi *.mov *.webm *.mp3 *.wav *.flac *.m4a);;모든 파일 (*)",
        )
        if not filepath:
            return

        name, ok = QInputDialog.getText(
            self, "사전 이름", "새 사전 이름:",
            text=os.path.splitext(os.path.basename(filepath))[0],
        )
        if not ok or not name.strip():
            return

        # 모델/언어 선택
        from src.config import get_whisper_model
        models = ["tiny", "base", "small", "medium", "large-v3"]
        current_model = get_whisper_model()
        model_idx = models.index(current_model) if current_model in models else 3
        model_name, ok = QInputDialog.getItem(
            self, "분석 설정", "Whisper 모델:", models, model_idx, False,
        )
        if not ok:
            return

        languages = ["ko", "en", "ja", "zh", "auto"]
        lang_labels = ["한국어", "English", "日本語", "中文", "자동 감지"]
        lang, ok = QInputDialog.getItem(
            self, "분석 설정", "언어:", lang_labels, 0, False,
        )
        if not ok:
            return
        language = languages[lang_labels.index(lang)]

        from src.database import compute_file_checksum
        checksum = compute_file_checksum(filepath)

        # 진행 다이얼로그
        from PySide6.QtWidgets import QProgressDialog
        progress = QProgressDialog("미디어 분석 중...", "취소", 0, 100, self)
        progress.setWindowTitle("사전 분석")
        progress.setMinimumDuration(0)
        progress.setValue(0)

        import multiprocessing as mp
        from src.dict_analyzer import _analyze_worker
        from src.transcriber import extract_audio, get_ffmpeg_exe

        # ffmpeg으로 오디오 추출
        import tempfile
        tmp_wav = os.path.join(tempfile.gettempdir(), f"vt_dict_{os.getpid()}_{os.path.basename(filepath)}.wav")
        try:
            extract_audio(filepath, tmp_wav, get_ffmpeg_exe())
        except Exception as e:
            QMessageBox.critical(self, "오류", f"오디오 추출 실패: {e}")
            return

        ctx = mp.get_context("spawn")
        msg_queue = ctx.Queue()
        cancel_event = ctx.Event()

        params = {
            "wav_path": tmp_wav,
            "model_name": model_name,
            "language": language,
            "use_diar": False,
            "profile": "interview",
        }

        proc = ctx.Process(target=_analyze_worker, args=(params, msg_queue, cancel_event), daemon=True)
        proc.start()

        result_data = None
        was_cancelled = False
        error_msg = None
        try:
            while proc.is_alive() or not msg_queue.empty():
                if progress.wasCanceled():
                    was_cancelled = True
                    cancel_event.set()
                    break
                try:
                    msg = msg_queue.get(timeout=0.1)
                except Exception:
                    QApplication.processEvents()
                    continue

                if msg[0] == "progress":
                    progress.setValue(msg[1])
                    progress.setLabelText(msg[2])
                elif msg[0] == "result":
                    result_data = msg[1]
                elif msg[0] == "error":
                    error_msg = msg[1]
                    break
                QApplication.processEvents()
        finally:
            # 프로세스 정리 (취소/에러/정상 종료 모두)
            if proc.is_alive():
                if not cancel_event.is_set():
                    cancel_event.set()
                proc.join(timeout=10)
            if proc.is_alive():
                try:
                    proc.kill()
                except Exception:
                    pass
                proc.join(timeout=3)
            progress.close()
            # 임시 파일 정리
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

        if error_msg:
            QMessageBox.critical(self, "분석 오류", error_msg)
            return
        if was_cancelled or not result_data:
            return

        # DB에 사전 생성 + 항목 저장
        try:
            dict_id = self.db.create_correction_dict(
                name.strip(),
                media_checksum=checksum,
                media_filename=os.path.basename(filepath),
            )

            entries = []
            for w in result_data.get("words", []):
                entries.append({
                    "wrong": w["word"],
                    "correct": w["word"],  # 초기에는 동일 (사용자가 교정)
                    "start_time": w["start"],
                    "end_time": w["end"],
                    "speaker": w.get("speaker"),
                    "frequency": w["frequency"],
                    "is_corrected": False,
                })
            self.db.replace_correction_entries(dict_id, entries)

            self._refresh()
            # 새로 만든 사전 선택
            for i in range(self.list_dicts.count()):
                if i < len(self._dicts) and self._dicts[i]["id"] == dict_id:
                    self.list_dicts.setCurrentRow(i)
                    break
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", f"사전 저장 중 오류가 발생했습니다:\n{e}")

    def _change_checksum(self):
        dict_id = self._current_dict_id()
        if dict_id is None:
            return
        # 파일 선택 또는 직접 입력
        choice = QMessageBox.question(
            self, "체크섬 변경",
            "파일을 선택하여 체크섬을 계산하시겠습니까?\n\n'아니오'를 선택하면 직접 입력할 수 있습니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return
        elif choice == QMessageBox.StandardButton.Yes:
            filepath, _ = QFileDialog.getOpenFileName(self, "미디어 파일 선택")
            if not filepath:
                return
            from src.database import compute_file_checksum
            new_checksum = compute_file_checksum(filepath)
        else:
            new_checksum, ok = QInputDialog.getText(self, "체크섬 입력", "새 체크섬 값:")
            if not ok or not new_checksum.strip():
                return
            new_checksum = new_checksum.strip()

        self.db.update_correction_dict_checksum(dict_id, new_checksum)
        self._on_dict_selected(self.list_dicts.currentRow())

    def _add_dict(self):
        name, ok = QInputDialog.getText(self, "새 사전", "사전 이름:")
        if ok and name.strip():
            self.db.create_correction_dict(name.strip())
            self._refresh()
            self.list_dicts.setCurrentRow(self.list_dicts.count() - 1)

    def _delete_dict(self):
        dict_id = self._current_dict_id()
        if dict_id is None:
            return
        if QMessageBox.question(self, "삭제", "이 사전을 삭제하시겠습니까?") == QMessageBox.StandardButton.Yes:
            self.db.delete_correction_dict(dict_id)
            self._refresh()

    def _add_entry(self):
        dict_id = self._current_dict_id()
        if dict_id is None:
            return
        row = self.table_entries.rowCount()
        self.table_entries.insertRow(row)
        self.table_entries.setItem(row, 0, QTableWidgetItem("—"))
        self.table_entries.setItem(row, 1, QTableWidgetItem(""))
        self.table_entries.setItem(row, 2, QTableWidgetItem(""))
        self.table_entries.setItem(row, 3, QTableWidgetItem("1"))
        self.table_entries.editItem(self.table_entries.item(row, 1))

    def _delete_entry(self):
        rows = set(idx.row() for idx in self.table_entries.selectedIndexes())
        for row in sorted(rows, reverse=True):
            item = self.table_entries.item(row, 1)
            entry_id = item.data(Qt.ItemDataRole.UserRole) if item else None
            if entry_id:
                self.db.delete_correction_entry(entry_id)
            self.table_entries.removeRow(row)

    def _save_entries(self):
        dict_id = self._current_dict_id()
        if dict_id is None:
            return

        # UI에서 편집된 교정 매핑 수집: (speaker, wrong) → correct
        correction_map = {}
        for row in range(self.table_entries.rowCount()):
            wrong_item = self.table_entries.item(row, 1)
            wrong = (wrong_item.text() if wrong_item else "").strip()
            correct_item = self.table_entries.item(row, 2)
            correct = (correct_item.text() if correct_item else "").strip()
            if not wrong or not correct:
                continue
            meta = wrong_item.data(Qt.ItemDataRole.UserRole + 1) or {}
            key = (meta.get("speaker"), wrong)
            correction_map[key] = correct

        # DB의 모든 항목을 가져와서 교정 매핑 적용 (모든 등장 위치에 일괄)
        all_entries = self.db.get_correction_entries(dict_id)
        updated = []
        for e in all_entries:
            key = (e.get("speaker"), e["wrong"])
            if key in correction_map:
                e["correct"] = correction_map[key]
                e["is_corrected"] = e["wrong"] != e["correct"]
            updated.append(e)

        # 수동 추가 항목 (DB에 없는 것) 처리
        existing_keys = {(e.get("speaker"), e["wrong"]) for e in all_entries}
        for row in range(self.table_entries.rowCount()):
            wrong_item = self.table_entries.item(row, 1)
            wrong = (wrong_item.text() if wrong_item else "").strip()
            correct_item = self.table_entries.item(row, 2)
            correct = (correct_item.text() if correct_item else "").strip()
            if not wrong or not correct:
                continue
            meta = wrong_item.data(Qt.ItemDataRole.UserRole + 1) or {}
            key = (meta.get("speaker"), wrong)
            if key not in existing_keys:
                updated.append({
                    "wrong": wrong,
                    "correct": correct,
                    "speaker": meta.get("speaker"),
                    "frequency": meta.get("frequency", 1),
                    "is_corrected": wrong != correct,
                })

        self.db.replace_correction_entries(dict_id, updated)
        self._on_dict_selected(self.list_dicts.currentRow())


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
        self._live_item = None          # 변환 중 목록 항목 (QTreeWidgetItem)
        self._live_filename = ""        # 변환 중 파일명
        self._live_segments = []        # 변환 중 세그먼트 누적
        self._pending_tid = None        # 점진적 저장용 DB ID
        self._seg_buffer = []           # DB 저장 대기 세그먼트 버퍼
        self._last_progress_msg = ""    # 마지막 진행 메시지 (경과 시간 표시용)
        self._elapsed_timer = QTimer()  # 경과 시간 갱신 타이머
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._process_start_time = None
        self._editing = False           # 편집 모드 상태
        self._queue = []                # 변환 대기열: [(path, settings, hf_token), ...]
        self._live_last_speaker = None  # 라이브 텍스트 화자 그룹화용

        version = QApplication.instance().applicationVersion()
        self.setWindowTitle(f"CATTS - Video Transcriber v{version}")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)
        self.setAcceptDrops(True)

        self._build_ui()
        self._setup_tray_icon()
        self._setup_shortcuts()
        self._load_list()

        # 이전 업데이트 잔여 파일 정리
        try:
            from src.updater import cleanup_old_update
            cleanup_old_update()
        except Exception:
            pass

        self._version_thread = VersionCheckThread()
        self._version_thread.finished.connect(self._on_version_checked)
        self._version_thread.start()

        if get_show_startup_guide():
            QTimer.singleShot(0, self._show_startup_guide)

        # 미완료 변환 감지 (크래시 복구)
        QTimer.singleShot(500, self._check_incomplete)

    def _show_startup_guide(self):
        StartupGuideDialog(self).exec()

    # ── 미완료 변환 복구 ──

    def _check_incomplete(self):
        """앱 시작 시 미완료 변환을 감지하고 이어하기를 제안한다."""
        incompletes = self.db.get_incomplete_transcriptions()
        if not incompletes:
            return

        # 세그먼트가 있는 미완료 항목만 (빈 레코드는 정리)
        resumable = []
        for item in incompletes:
            if item["seg_count"] > 0 and os.path.exists(item["filepath"]):
                resumable.append(item)
            elif item["seg_count"] == 0:
                self.db.delete_empty_transcription(item["id"])

        if not resumable:
            self._load_list()  # 빈 레코드 삭제 후 새로고침
            return

        # 다이얼로그: 이어하기 / 부분 결과 유지 / 삭제
        for item in resumable:
            m, s = divmod(int(item["last_end"]), 60)
            time_str = f"{m}분 {s}초" if m > 0 else f"{s}초"
            reply = QMessageBox.question(
                self,
                "미완료 변환 발견",
                f"'{item['filename']}'의 변환이 완료되지 않았습니다.\n"
                f"({item['seg_count']}개 세그먼트, {time_str}까지 변환됨)\n\n"
                f"이어서 변환하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._resume_transcription(item)

    def _resume_transcription(self, item: dict):
        """미완료 레코드를 이어서 변환한다."""
        settings_dialog = TranscriptionSettingsDialog(self)
        # 이전 설정 복원
        if item.get("model_name"):
            idx = settings_dialog.combo_model.findData(item["model_name"])
            if idx >= 0:
                settings_dialog.combo_model.setCurrentIndex(idx)
        if item.get("language"):
            idx = settings_dialog.combo_lang.findData(item["language"])
            if idx >= 0:
                settings_dialog.combo_lang.setCurrentIndex(idx)

        if settings_dialog.exec() != QDialog.DialogCode.Accepted:
            return

        settings = settings_dialog.get_settings()
        hf_token = None
        if settings["use_diarization"]:
            hf_token = get_hf_token()
            if not hf_token:
                settings["use_diarization"] = False

        self._start_transcription(
            item["filepath"], settings, hf_token,
            resume_tid=item["id"],
            skip_seconds=item["last_end"],
        )

    # ── 버전 확인 ──

    def _on_version_checked(self, latest: int):
        try:
            current = int(QApplication.instance().applicationVersion())
        except (ValueError, TypeError):
            self.lbl_version.setToolTip("버전 정보를 읽을 수 없습니다")
            return
        if latest < 0:
            self.lbl_version.setToolTip("버전 확인 실패 (네트워크 오류)")
            return
        if current >= latest:
            self.lbl_version.setText(f"v{current} ✓ 최신")
            self.lbl_version.setStyleSheet("color: #66BB6A; font-size: 11px;")
            self.lbl_version.setToolTip("최신 버전을 사용 중입니다")
        else:
            self.lbl_version.setText(f"v{current} → v{latest} 업데이트")
            self.lbl_version.setStyleSheet("color: #FFA726; font-size: 11px; font-weight: bold;")
            self.lbl_version.setToolTip("클릭하면 다운로드 페이지로 이동합니다")
            self._show_update_dialog(current, latest)

    def _show_update_dialog(self, current: int, latest: int):
        from src.updater import can_auto_update, check_for_update

        msg = QMessageBox(self)
        msg.setWindowTitle("업데이트 안내")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(f"새 버전이 있습니다: v{latest} (현재 v{current})")

        if can_auto_update():
            msg.setInformativeText("지금 자동으로 업데이트하시겠습니까?")
            msg.setStandardButtons(
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Open
            )
            msg.button(QMessageBox.StandardButton.Yes).setText("자동 업데이트")
            msg.button(QMessageBox.StandardButton.Open).setText("다운로드 페이지")
            msg.button(QMessageBox.StandardButton.No).setText("나중에")
            msg.setDefaultButton(QMessageBox.StandardButton.Yes)
        else:
            msg.setInformativeText("다운로드 페이지로 이동하시겠습니까?")
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.button(QMessageBox.StandardButton.Yes).setText("다운로드 페이지 열기")
            msg.button(QMessageBox.StandardButton.No).setText("나중에")
            msg.setDefaultButton(QMessageBox.StandardButton.Yes)

        choice = msg.exec()

        if choice == QMessageBox.StandardButton.Yes and can_auto_update():
            self._run_auto_update()
        elif choice == QMessageBox.StandardButton.Yes or choice == QMessageBox.StandardButton.Open:
            QDesktopServices.openUrl(QUrl(RELEASES_PAGE))

    def _run_auto_update(self):
        """자동 업데이트: 다운로드 → 설치 → 재시작."""
        from src.updater import check_for_update, download_update, prepare_update, apply_update_and_restart
        from PySide6.QtWidgets import QProgressDialog

        update_info = check_for_update()
        if not update_info:
            QMessageBox.information(self, "업데이트", "이미 최신 버전입니다.")
            return

        progress = QProgressDialog(
            f"v{update_info['version']} 다운로드 중...", "취소", 0, 100, self
        )
        progress.setWindowTitle("자동 업데이트")
        progress.setMinimumDuration(0)
        progress.setValue(0)

        cancelled = False

        def _on_progress(pct):
            nonlocal cancelled
            if progress.wasCanceled():
                cancelled = True
            progress.setValue(pct)
            QApplication.processEvents()

        try:
            downloaded = download_update(
                update_info["download_url"],
                update_info["size"],
                progress_callback=_on_progress,
            )

            if cancelled:
                progress.close()
                return

            progress.setLabelText("업데이트 준비 중...")
            progress.setValue(95)
            QApplication.processEvents()

            staging = prepare_update(downloaded)

            progress.close()

            reply = QMessageBox.question(
                self, "업데이트 설치",
                f"v{update_info['version']} 다운로드 완료.\n"
                f"앱을 종료하고 업데이트를 설치하시겠습니까?",
            )
            if reply == QMessageBox.StandardButton.Yes:
                apply_update_and_restart(staging)
            # else: staging은 다음에 cleanup_old_update에서 정리됨

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "업데이트 오류", f"업데이트 실패: {e}")

    # ── 시스템 트레이 아이콘 (Feature 7) ──

    def _setup_tray_icon(self):
        self._tray_icon = QSystemTrayIcon(self)
        app_icon = QApplication.instance().windowIcon()
        if not app_icon.isNull():
            self._tray_icon.setIcon(app_icon)
        v = QApplication.instance().applicationVersion()
        self._tray_icon.setToolTip(f"CATTS - Video Transcriber v{v}")
        self._tray_icon.show()

    def _notify(self, title: str, message: str):
        """OS 토스트 알림을 표시한다."""
        if self._tray_icon.isSystemTrayAvailable():
            self._tray_icon.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 5000)

    # ── 키보드 단축키 ──

    def _setup_shortcuts(self):
        # 메뉴바에 없는 단축키만 여기서 등록 (메뉴 항목은 메뉴에서 setShortcut)
        QShortcut(QKeySequence("Ctrl+F"), self, self._focus_timeline_search)

    def _focus_timeline_search(self):
        self.tabs.setCurrentIndex(0)
        self.timeline_search.setFocus()
        self.timeline_search.selectAll()

    # ── UI 빌드 ──

    def _build_menubar(self):
        menubar = self.menuBar()

        # ── 파일 ──
        file_menu = menubar.addMenu("파일")

        act_add = file_menu.addAction("미디어 파일 추가...")
        act_add.setShortcut("Ctrl+O")
        act_add.triggered.connect(self._on_add_video)

        act_export = file_menu.addAction("내보내기...")
        act_export.setShortcut("Ctrl+S")
        act_export.triggered.connect(self._on_export)

        file_menu.addSeparator()

        act_delete = file_menu.addAction("선택 항목 삭제")
        act_delete.setShortcut("Delete")
        act_delete.triggered.connect(self._on_delete)

        file_menu.addSeparator()

        act_quit = file_menu.addAction("종료")
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)

        # ── 편집 ──
        edit_menu = menubar.addMenu("편집")

        act_copy = edit_menu.addAction("텍스트 복사")
        act_copy.setShortcut("Ctrl+Shift+C")
        act_copy.triggered.connect(self._on_copy)

        act_search = edit_menu.addAction("타임라인 검색")
        act_search.setShortcut("Ctrl+F")
        act_search.triggered.connect(self._focus_timeline_search)

        # ── 도구 ──
        tools_menu = menubar.addMenu("도구")

        act_dict = tools_menu.addAction("교정 사전 관리...")
        act_dict.triggered.connect(self._open_dict_manager)

        tools_menu.addSeparator()

        act_settings = tools_menu.addAction("설정...")
        act_settings.setShortcut("Ctrl+,")
        act_settings.triggered.connect(self._on_settings)

        # ── 도움말 ──
        help_menu = menubar.addMenu("도움말")

        act_update = help_menu.addAction("업데이트 확인...")
        act_update.triggered.connect(self._check_update_manual)

        act_guide = help_menu.addAction("시작 안내")
        act_guide.triggered.connect(self._show_startup_guide)

        help_menu.addSeparator()

        act_about = help_menu.addAction("프로그램 정보")
        act_about.triggered.connect(self._show_about)

    def _open_dict_manager(self):
        """메인 윈도우에서 교정 사전 관리 다이얼로그를 연다."""
        dlg = CorrectionDictDialog(self.db, self)
        dlg.exec()

    def _check_update_manual(self):
        """수동 업데이트 확인."""
        from src.updater import check_for_update, can_auto_update
        try:
            update_info = check_for_update()
        except Exception as e:
            QMessageBox.warning(self, "업데이트 확인", f"확인 실패: {e}")
            return

        if update_info:
            current = int(QApplication.instance().applicationVersion())
            self._show_update_dialog(current, update_info["version"])
        else:
            QMessageBox.information(self, "업데이트 확인", "현재 최신 버전입니다.")

    def _show_about(self):
        version = QApplication.instance().applicationVersion()
        QMessageBox.about(
            self, "프로그램 정보",
            f"<h3>CATTS - Video Transcriber</h3>"
            f"<p>버전: {version}</p>"
            f"<p>영상/음성에서 텍스트를 추출하는 프로그램</p>"
            f"<p>Whisper + pyannote + Demucs 기반</p>"
            f"<hr>"
            f"<p><a href='https://github.com/taxi-tabby/CATTS-VideoTranscriber'>GitHub</a></p>",
        )

    def _build_ui(self):
        self._build_menubar()

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

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("검색...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._on_search)
        left_layout.addWidget(self.search_edit)

        self.tree_widget = _DroppableTreeWidget()
        self.tree_widget.setHeaderHidden(True)
        self.tree_widget.setIndentation(16)
        self.tree_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self._on_tree_context_menu)
        self.tree_widget.currentItemChanged.connect(self._on_tree_item_changed)
        self.tree_widget.setDragDropMode(QTreeWidget.DragDropMode.InternalMove)
        self.tree_widget.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.tree_widget.itemDropped.connect(self._on_rows_moved)
        left_layout.addWidget(self.tree_widget)

        # Bottom buttons
        left_btn_row = QHBoxLayout()
        self.btn_delete = QPushButton("삭제")
        self.btn_delete.clicked.connect(self._on_delete)
        left_btn_row.addWidget(self.btn_delete)

        self.btn_retranscribe = QPushButton("재변환")
        self.btn_retranscribe.clicked.connect(self._on_retranscribe)
        left_btn_row.addWidget(self.btn_retranscribe)

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

        # Tabs: timeline / full text / processing log
        self.tabs = QTabWidget()

        # ── Timeline tab (Feature 10: QTableWidget) ──
        timeline_container = QWidget()
        tl_layout = QVBoxLayout(timeline_container)
        tl_layout.setContentsMargins(0, 4, 0, 0)

        # Search + Speaker filter (Feature 1)
        filter_row = QHBoxLayout()
        self.timeline_search = QLineEdit()
        self.timeline_search.setPlaceholderText("텍스트 검색...")
        self.timeline_search.setClearButtonEnabled(True)
        self.timeline_search.textChanged.connect(self._on_timeline_filter)
        filter_row.addWidget(self.timeline_search, stretch=1)

        self.speaker_filter = QComboBox()
        self.speaker_filter.addItem("전체 화자", "")
        self.speaker_filter.setMinimumWidth(120)
        self.speaker_filter.currentIndexChanged.connect(self._on_timeline_filter)
        self.speaker_filter.setVisible(False)
        filter_row.addWidget(self.speaker_filter)

        tl_layout.addLayout(filter_row)

        self.table_timeline = QTableWidget()
        self.table_timeline.setColumnCount(4)
        self.table_timeline.setHorizontalHeaderLabels(["시작", "종료", "화자", "텍스트"])
        self.table_timeline.verticalHeader().setVisible(False)
        self.table_timeline.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_timeline.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_timeline.setWordWrap(True)
        header = self.table_timeline.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        tl_layout.addWidget(self.table_timeline)
        self.tabs.addTab(timeline_container, "타임라인")

        # ── Full text tab (Feature 5: editable) ──
        fulltext_container = QWidget()
        ft_layout = QVBoxLayout(fulltext_container)
        ft_layout.setContentsMargins(0, 4, 0, 0)

        self.txt_fulltext = QPlainTextEdit()
        self.txt_fulltext.setReadOnly(True)
        self.txt_fulltext.setFont(QFont("Malgun Gothic", 10))
        ft_layout.addWidget(self.txt_fulltext)

        # Edit buttons
        edit_row = QHBoxLayout()
        edit_row.addStretch()
        self.btn_edit = QPushButton("편집")
        self.btn_edit.clicked.connect(self._on_toggle_edit)
        self.btn_edit.setVisible(False)
        edit_row.addWidget(self.btn_edit)

        self.btn_save_edit = QPushButton("저장")
        self.btn_save_edit.clicked.connect(self._on_save_edit)
        self.btn_save_edit.setVisible(False)
        self.btn_save_edit.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        edit_row.addWidget(self.btn_save_edit)

        self.btn_cancel_edit = QPushButton("편집 취소")
        self.btn_cancel_edit.clicked.connect(self._on_cancel_edit)
        self.btn_cancel_edit.setVisible(False)
        edit_row.addWidget(self.btn_cancel_edit)

        ft_layout.addLayout(edit_row)
        self.tabs.addTab(fulltext_container, "전체 텍스트")

        # ── Processing log tab ──
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Consolas", 9))
        self.txt_log.setStyleSheet("QPlainTextEdit { background-color: #1e1e2e; color: #cdd6f4; }")
        self.tabs.addTab(self.txt_log, "처리 로그")

        right_layout.addWidget(self.tabs, stretch=1)

        # Copy button + Speaker management button
        btn_row = QHBoxLayout()
        self.btn_speakers = QPushButton("화자 관리")
        self.btn_speakers.clicked.connect(self._on_manage_speakers)
        self.btn_speakers.setVisible(False)
        btn_row.addWidget(self.btn_speakers)
        btn_row.addStretch()
        self.btn_export = QPushButton("내보내기")
        self.btn_export.clicked.connect(self._on_export)
        btn_row.addWidget(self.btn_export)
        self.btn_copy = QPushButton("텍스트 복사")
        self.btn_copy.clicked.connect(self._on_copy)
        btn_row.addWidget(self.btn_copy)
        right_layout.addLayout(btn_row)

        # Empty state
        self.lbl_empty = QLabel("영상 또는 음성 파일을 추가하면 여기에 결과가 표시됩니다.")
        self.lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_empty.setStyleSheet("color: #8EA4C0; font-size: 14px;")
        right_layout.addWidget(self.lbl_empty)

        splitter.addWidget(right)
        splitter.setSizes([280, 720])

        # --- Bottom progress bar + cancel button (Feature 3) ---
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(8, 4, 8, 8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(28)
        self.progress_bar.setVisible(False)
        bottom_layout.addWidget(self.progress_bar, stretch=1)

        self.btn_cancel = QPushButton("  취소  ")
        self.btn_cancel.setFixedHeight(28)
        self.btn_cancel.setMinimumWidth(70)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._on_cancel_transcription)
        bottom_layout.addWidget(self.btn_cancel)

        self.lbl_status = QLabel("")
        self.lbl_status.setVisible(False)
        bottom_layout.addWidget(self.lbl_status)

        bottom_layout.addStretch()

        self.lbl_version = QLabel(f"v{QApplication.instance().applicationVersion()}")
        self.lbl_version.setStyleSheet("color: #8EA4C0; font-size: 11px;")
        self.lbl_version.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lbl_version.setToolTip("버전 정보 확인 중...")
        self.lbl_version.mousePressEvent = lambda _: QDesktopServices.openUrl(QUrl(RELEASES_PAGE))
        bottom_layout.addWidget(self.lbl_version)

        layout.addWidget(bottom)

        # 변환 중 글로우 오버레이
        self._glow = _GlowOverlay(central)
        self._glow.hide()

        self._show_detail(False)

    def _show_detail(self, show: bool):
        self.lbl_title.setVisible(show)
        self.lbl_info.setVisible(show)
        self.tabs.setVisible(show)
        self.btn_copy.setVisible(show)
        self.btn_export.setVisible(show)
        self.btn_speakers.setVisible(False)
        self.btn_edit.setVisible(False)
        self.btn_save_edit.setVisible(False)
        self.btn_cancel_edit.setVisible(False)
        self.lbl_empty.setVisible(not show)

    # ── 타임라인 테이블 관리 (Feature 10) ──

    def _populate_timeline(self, segments: list[dict]):
        """세그먼트 목록으로 타임라인 테이블을 채운다."""
        self.table_timeline.setRowCount(0)
        self._cancel_edit_mode()

        has_speakers = any(s.get("speaker") for s in segments)
        self.table_timeline.setColumnHidden(2, not has_speakers)

        # 화자 필터 업데이트
        self.speaker_filter.blockSignals(True)
        self.speaker_filter.clear()
        self.speaker_filter.addItem("전체 화자", "")
        if has_speakers:
            speakers = sorted(set(s.get("speaker", "") for s in segments if s.get("speaker")))
            for sp in speakers:
                self.speaker_filter.addItem(sp, sp)
        self.speaker_filter.setVisible(has_speakers)
        self.speaker_filter.blockSignals(False)

        for seg in segments:
            self._add_timeline_row(seg)

    def _add_timeline_row(self, seg: dict):
        """테이블에 세그먼트 한 행을 추가한다."""
        row = self.table_timeline.rowCount()
        self.table_timeline.insertRow(row)

        item_start = QTableWidgetItem(format_timestamp(seg["start"]))
        item_start.setFlags(item_start.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item_start.setData(Qt.ItemDataRole.UserRole, seg.get("id"))  # segment DB id

        item_end = QTableWidgetItem(format_timestamp(seg["end"]))
        item_end.setFlags(item_end.flags() & ~Qt.ItemFlag.ItemIsEditable)

        item_speaker = QTableWidgetItem(seg.get("speaker") or "")
        item_speaker.setFlags(item_speaker.flags() & ~Qt.ItemFlag.ItemIsEditable)

        item_text = QTableWidgetItem(seg.get("text", ""))
        item_text.setFlags(item_text.flags() & ~Qt.ItemFlag.ItemIsEditable)

        self.table_timeline.setItem(row, 0, item_start)
        self.table_timeline.setItem(row, 1, item_end)
        self.table_timeline.setItem(row, 2, item_speaker)
        self.table_timeline.setItem(row, 3, item_text)

    # ── 타임라인 검색/필터 (Feature 1) ──

    def _on_timeline_filter(self, *_args):
        search_text = self.timeline_search.text().strip().lower()
        speaker_data = self.speaker_filter.currentData()

        for row in range(self.table_timeline.rowCount()):
            text_item = self.table_timeline.item(row, 3)
            speaker_item = self.table_timeline.item(row, 2)
            text_match = not search_text or (text_item and search_text in text_item.text().lower())
            speaker_match = not speaker_data or (speaker_item and speaker_item.text() == speaker_data)
            self.table_timeline.setRowHidden(row, not (text_match and speaker_match))

    # ── 편집 기능 (Feature 5) ──

    def _on_toggle_edit(self):
        if self._current_tid is None:
            return
        self._editing = True
        self.btn_edit.setVisible(False)
        self.btn_save_edit.setVisible(True)
        self.btn_cancel_edit.setVisible(True)

        # 타임라인 테이블의 텍스트 컬럼을 편집 가능하게
        self.table_timeline.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.EditKeyPressed
        )
        for row in range(self.table_timeline.rowCount()):
            item = self.table_timeline.item(row, 3)
            if item:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)

        # 전체 텍스트는 편집 비활성화 (타임라인 편집만 지원, 저장 시 자동 재생성)
        self.txt_fulltext.setReadOnly(True)
        self.txt_fulltext.setStyleSheet("QPlainTextEdit { border: 2px solid #888; opacity: 0.7; }")
        self.txt_fulltext.setPlaceholderText("타임라인에서 텍스트를 편집하면 전체 텍스트가 자동으로 갱신됩니다.")

    def _on_save_edit(self):
        if self._current_tid is None:
            return

        # 타임라인 테이블의 변경사항을 DB에 저장
        for row in range(self.table_timeline.rowCount()):
            start_item = self.table_timeline.item(row, 0)
            text_item = self.table_timeline.item(row, 3)
            if start_item and text_item:
                seg_id = start_item.data(Qt.ItemDataRole.UserRole)
                if seg_id is not None:
                    self.db.update_segment_text(seg_id, text_item.text())

        # full_text 재생성 및 저장
        data = self.db.get_transcription(self._current_tid)
        if data:
            new_full = self._build_full_text(data.get("segments", []))
            self.db.update_full_text(self._current_tid, new_full)

        self._cancel_edit_mode()

        # 화면 갱신
        self._on_tree_item_changed(self.tree_widget.currentItem(), None)

        self.lbl_status.setVisible(True)
        self.lbl_status.setText("변경사항이 저장되었습니다.")
        QTimer.singleShot(3000, lambda: self.lbl_status.setVisible(False))

    def _on_cancel_edit(self):
        self._cancel_edit_mode()
        # 원본 데이터로 복원
        self._on_tree_item_changed(self.tree_widget.currentItem(), None)

    def _cancel_edit_mode(self):
        self._editing = False
        self.btn_edit.setVisible(self._current_tid is not None)
        self.btn_save_edit.setVisible(False)
        self.btn_cancel_edit.setVisible(False)
        self.table_timeline.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row in range(self.table_timeline.rowCount()):
            item = self.table_timeline.item(row, 3)
            if item:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.txt_fulltext.setReadOnly(True)
        self.txt_fulltext.setStyleSheet("")

    # ── 테마 실시간 적용 (Feature 9) ──

    def _apply_theme(self, theme: str):
        from src.main import get_stylesheet
        QApplication.instance().setStyleSheet(get_stylesheet(theme))

    # --- Data loading ---

    def _load_list(self):
        self.tree_widget.clear()
        self._items = self.db.get_all_transcriptions()
        folders = self.db.get_all_folders()
        search = self.search_edit.text().strip().lower() if hasattr(self, 'search_edit') else ""

        # 폴더 트리 아이템 맵 생성
        folder_map: dict[int, QTreeWidgetItem] = {}

        def _get_folder_item(fid: int) -> QTreeWidgetItem:
            if fid in folder_map:
                return folder_map[fid]
            folder = next((f for f in folders if f["id"] == fid), None)
            if folder is None:
                return None
            parent_id = folder["parent_id"]
            if parent_id and parent_id in [f["id"] for f in folders]:
                parent_item = _get_folder_item(parent_id)
                item = QTreeWidgetItem(parent_item)
            else:
                item = QTreeWidgetItem(self.tree_widget)
            item.setText(0, folder["name"])
            item.setData(0, Qt.ItemDataRole.UserRole, folder["id"])
            item.setData(0, Qt.ItemDataRole.UserRole + 1, "folder")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsAutoTristate)
            folder_map[fid] = item
            return item

        # 폴더 먼저 생성
        for f in folders:
            _get_folder_item(f["id"])

        # 트랜스크립션 항목 추가
        for t in self._items:
            display_name = t.get("display_name") or t["filename"]
            if search and search not in display_name.lower():
                continue
            dur = format_duration(t.get("duration"))
            date = t["created_at"][:10]
            info_parts = [date, dur]
            model = t.get("model_name") or ""
            lang = t.get("language") or ""
            if model:
                info_parts.append(model)
            if lang:
                info_parts.append(lang)

            folder_id = t.get("folder_id")
            parent = folder_map.get(folder_id) if folder_id else None
            if parent:
                item = QTreeWidgetItem(parent)
            else:
                item = QTreeWidgetItem(self.tree_widget)
            item.setText(0, f"{display_name}\n{'  '.join(info_parts)}")
            item.setData(0, Qt.ItemDataRole.UserRole, t["id"])
            item.setData(0, Qt.ItemDataRole.UserRole + 1, "transcription")
            # 트랜스크립션은 자식을 받지 않음 (드롭 대상 제외)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsDropEnabled)

        self.tree_widget.expandAll()

    def _on_search(self, _text: str):
        self._load_list()

    def _remove_live_item(self):
        if self._live_item:
            idx = self.tree_widget.indexOfTopLevelItem(self._live_item)
            if idx >= 0:
                self.tree_widget.takeTopLevelItem(idx)
            self._live_item = None
            self._glow.stop()

    # ── 컨텍스트 메뉴 ──

    def _on_tree_context_menu(self, pos):
        item = self.tree_widget.itemAt(pos)
        menu = QMenu(self)

        act_new_folder = menu.addAction("새 폴더")
        act_new_folder.triggered.connect(lambda: self._ctx_new_folder(item))

        if item is not None:
            item_type = item.data(0, Qt.ItemDataRole.UserRole + 1)

            if item_type in ("folder", "transcription"):
                menu.addSeparator()
                act_rename = menu.addAction("이름 변경")
                act_rename.triggered.connect(lambda: self._ctx_rename(item))

            if item_type == "transcription":
                folders = self.db.get_all_folders()
                if folders:
                    move_menu = menu.addMenu("폴더로 이동")
                    tid = item.data(0, Qt.ItemDataRole.UserRole)
                    current_fid = None
                    for t in self._items:
                        if t["id"] == tid:
                            current_fid = t.get("folder_id")
                            break

                    if current_fid is not None:
                        act_root = move_menu.addAction("(최상위)")
                        act_root.triggered.connect(lambda: self._ctx_move_to_folder(item, None))
                        move_menu.addSeparator()

                    for f in folders:
                        if f["id"] == current_fid:
                            continue
                        act = move_menu.addAction(f["name"])
                        fid = f["id"]
                        act.triggered.connect(lambda checked=False, fid=fid: self._ctx_move_to_folder(item, fid))

        menu.exec(self.tree_widget.viewport().mapToGlobal(pos))

    def _ctx_new_folder(self, item):
        parent_id = None
        if item and item.data(0, Qt.ItemDataRole.UserRole + 1) == "folder":
            parent_id = item.data(0, Qt.ItemDataRole.UserRole)

        name, ok = QInputDialog.getText(self, "새 폴더", "폴더 이름:")
        if ok and name.strip():
            self.db.create_folder(name.strip(), parent_id)
            self._load_list()

    def _ctx_rename(self, item):
        item_type = item.data(0, Qt.ItemDataRole.UserRole + 1)
        item_id = item.data(0, Qt.ItemDataRole.UserRole)

        if item_type == "folder":
            current_name = item.text(0)
            new_name, ok = QInputDialog.getText(self, "이름 변경", "새 이름:", text=current_name)
            if ok and new_name.strip():
                self.db.rename_folder(item_id, new_name.strip())
                self._load_list()

        elif item_type == "transcription":
            data = self.db.get_transcription(item_id)
            current_name = (data.get("display_name") or data["filename"]) if data else ""
            new_name, ok = QInputDialog.getText(self, "이름 변경", "새 이름:", text=current_name)
            if ok and new_name.strip():
                self.db.rename_transcription(item_id, new_name.strip())
                self._load_list()

    def _ctx_move_to_folder(self, item, folder_id):
        tid = item.data(0, Qt.ItemDataRole.UserRole)
        self.db.move_transcription(tid, folder_id)
        self._load_list()

    def _on_rows_moved(self):
        """드래그 앤 드롭 후 트리 상태를 DB에 반영한다."""
        self._sync_tree_to_db()

    def _sync_tree_to_db(self):
        """현재 트리 위젯의 계층 구조를 DB에 반영한다."""
        def _walk(parent_item, parent_folder_id):
            count = parent_item.childCount() if parent_item else self.tree_widget.topLevelItemCount()
            for i in range(count):
                child = parent_item.child(i) if parent_item else self.tree_widget.topLevelItem(i)
                item_type = child.data(0, Qt.ItemDataRole.UserRole + 1)
                item_id = child.data(0, Qt.ItemDataRole.UserRole)

                if item_type == "folder":
                    self.db.move_folder(item_id, parent_folder_id)
                    _walk(child, item_id)
                elif item_type == "transcription":
                    self.db.move_transcription(item_id, parent_folder_id)

        _walk(None, None)

    def _is_viewing_live(self) -> bool:
        """현재 목록에서 '변환 중' 항목을 보고 있는지 확인."""
        item = self.tree_widget.currentItem()
        return item is not None and item is self._live_item

    def _on_tree_item_changed(self, current, _previous):
        """트리 항목 선택 변경 시 호출."""
        if current is None:
            self._show_detail(False)
            return

        # 편집 중이면 저장 여부 확인
        if self._editing:
            reply = QMessageBox.question(
                self, "편집 중",
                "저장하지 않은 편집 내용이 있습니다. 저장하시겠습니까?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._on_save_edit()
            elif reply == QMessageBox.StandardButton.Cancel:
                self.tree_widget.blockSignals(True)
                self.tree_widget.setCurrentItem(_previous)
                self.tree_widget.blockSignals(False)
                return
            else:
                self._cancel_edit_mode()

        # "변환 중" 항목 선택 → 라이브 데이터 복원
        if current is self._live_item:
            self._current_tid = None
            self._show_detail(True)
            self.btn_speakers.setVisible(False)
            self.btn_edit.setVisible(False)
            self.lbl_title.setText(self._live_filename)
            self.lbl_info.setText("변환 진행 중...")
            self._populate_timeline(self._live_segments)
            self.txt_fulltext.setPlainText(self._build_full_text(self._live_segments))
            # 스크롤 끝으로
            self.table_timeline.scrollToBottom()
            return

        item_type = current.data(0, Qt.ItemDataRole.UserRole + 1)

        # 폴더 선택 시 상세 숨김
        if item_type == "folder":
            self._current_tid = None
            self._show_detail(False)
            return

        tid = current.data(0, Qt.ItemDataRole.UserRole)
        if tid is None:
            self._show_detail(False)
            return

        data = self.db.get_transcription(tid)
        if data is None:
            self._show_detail(False)
            return

        self._current_tid = tid
        self._show_detail(True)
        display_name = data.get("display_name") or data["filename"]
        self.lbl_title.setText(display_name)
        dur = format_duration(data.get("duration"))
        date = data["created_at"][:10]
        self.lbl_info.setText(f"날짜: {date}    길이: {dur}")

        self._populate_timeline(data.get("segments", []))
        self.txt_fulltext.setPlainText(self._build_full_text(data.get("segments", [])))

        has_speakers = any(s.get("speaker") for s in data.get("segments", []))
        self.btn_speakers.setVisible(has_speakers)
        self.btn_edit.setVisible(True)

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
        old_theme = get_theme()
        dialog = SettingsDialog(self.db.db_path, self)
        dialog.exec()
        # Feature 9: 테마 실시간 적용
        new_theme = get_theme()
        if new_theme != old_theme:
            self._apply_theme(new_theme)

    def _on_add_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "미디어 파일 선택",
            "",
            "미디어 파일 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a);;영상 파일 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;음성 파일 (*.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a);;모든 파일 (*)",
        )
        if not path:
            return
        self._add_file(path)

    # ── 변환 취소 (Feature 3) ──

    def _on_cancel_transcription(self):
        if self._worker:
            self._worker.cancel()
            # cancel 플래그만 설정 — 워커가 부분 결과를 finished로 emit하거나,
            # 결과 없이 종료하면 _on_finished/_on_error에서 정리됨.
            # UI는 취소 진행 중임을 표시
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("취소 중... (완료된 부분 저장)")

    # ── 변환 시작/대기열 (Feature 2) ──

    def _load_correction_entries(self, dict_id, video_path: str = "") -> list:
        """교정 사전 ID로 항목을 로드한다. 체크섬 불일치 시 빈 리스트."""
        if dict_id is None:
            return []
        try:
            d = self.db.get_correction_dict(dict_id)
            if not d:
                return []
            # 체크섬 검증 (체크섬이 있는 사전만)
            if d.get("media_checksum") and video_path:
                from src.database import compute_file_checksum
                file_checksum = compute_file_checksum(video_path)
                if file_checksum != d["media_checksum"]:
                    self._on_log_message(
                        f"⚠ 교정 사전 '{d['name']}'의 체크섬이 불일치합니다. 사전을 적용하지 않습니다."
                    )
                    return []
            return self.db.get_correction_entries(dict_id)
        except Exception:
            return []

    def _start_transcription(
        self, video_path: str, settings: dict, hf_token: str | None = None,
        resume_tid: int | None = None, skip_seconds: float = 0.0,
    ):
        self.btn_add.setEnabled(True)  # 대기열 추가를 위해 활성 유지
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_cancel.setVisible(True)
        self.lbl_status.setVisible(True)
        self.lbl_status.setText("이어하기 준비 중..." if resume_tid else "준비 중...")

        # 라이브 상태 초기화
        self._live_filename = os.path.basename(video_path)
        self._live_segments = []
        self._live_last_speaker = None

        # 목록 맨 위에 "변환 중" 항목 삽입
        self._live_item = QTreeWidgetItem()
        label = "이어하기" if resume_tid else "변환 중"
        self._live_item.setText(0, f"[ {label} ] {self._live_filename}")
        self._live_item.setData(0, Qt.ItemDataRole.UserRole, None)
        self._live_item.setData(0, Qt.ItemDataRole.UserRole + 1, "live")
        self.tree_widget.insertTopLevelItem(0, self._live_item)
        self.tree_widget.setCurrentItem(self._live_item)
        self._glow.setGeometry(self.centralWidget().rect())
        self._glow.start()

        self._show_detail(True)
        self.lbl_title.setText(self._live_filename)
        self.lbl_info.setText("이어하기 진행 중..." if resume_tid else "변환 진행 중...")
        self.table_timeline.setRowCount(0)
        self.txt_fulltext.clear()
        self.txt_log.clear()
        self._last_progress_msg = ""

        # 경과 시간 타이머 시작
        self._process_start_time = time.time()
        self._elapsed_timer.start()

        # 처리 로그 탭으로 전환
        self.tabs.setCurrentWidget(self.txt_log)

        # 점진적 저장: 이어하기 시 기존 tid 재사용, 신규 시 새 레코드 생성
        if resume_tid:
            self._pending_tid = resume_tid
        else:
            self._pending_tid = self.db.begin_transcription(
                filename=os.path.basename(video_path),
                filepath=video_path,
                model_name=settings.get("model_name"),
                language=settings.get("language"),
            )
        self._seg_buffer = []

        self._thread = QThread()
        self._worker = TranscriberWorker(
            video_path,
            use_diarization=settings["use_diarization"],
            hf_token=hf_token,
            model_name=settings.get("model_name", "medium"),
            language=settings.get("language", "ko"),
            num_speakers=settings.get("num_speakers"),
            min_speakers=settings.get("min_speakers"),
            max_speakers=settings.get("max_speakers"),
            whisper_workers=settings.get("whisper_workers", 1),
            diar_threads=settings.get("diar_threads", 1),
            skip_seconds=skip_seconds,
            profile=settings.get("profile", "interview"),
            correction_entries=self._load_correction_entries(settings.get("correction_dict_id"), video_path),
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self._on_log_message)
        self._worker.segment_ready.connect(self._on_segment_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        # QThread 안전 정리: thread.finished(이벤트 루프 종료 후)에서
        # Python 참조 해제 + C++ deleteLater.
        # finished 시그널은 run()의 finally 블록까지 완료된 후 발생하므로
        # 워커가 아직 실행 중일 때 참조가 해제되는 race condition을 방지한다.
        self._thread.finished.connect(self._release_thread_refs)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_progress(self, percent: int, message: str):
        self.progress_bar.setValue(percent)
        self.lbl_status.setText(message)
        self._last_progress_msg = message

    def _on_log_message(self, message: str):
        self.txt_log.appendPlainText(message)
        scrollbar = self.txt_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_elapsed(self):
        """1초마다 하단 상태 표시줄에 경과 시간을 갱신하여 프로그램이 동작 중임을 보여줌."""
        if self._process_start_time is None:
            return
        elapsed = int(time.time() - self._process_start_time)
        m, s = divmod(elapsed, 60)
        elapsed_str = f"{m}분 {s}초" if m > 0 else f"{s}초"
        base_msg = self._last_progress_msg
        if " ⏱" in base_msg:
            base_msg = base_msg.split(" ⏱")[0]
        self.lbl_status.setText(f"{base_msg} ⏱ {elapsed_str}")

    def _flush_segments(self):
        """버퍼에 쌓인 세그먼트를 DB에 저장한다."""
        if self._pending_tid and self._seg_buffer:
            try:
                self.db.add_segments_batch(self._pending_tid, self._seg_buffer)
                self._seg_buffer = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[flush_segments] DB 저장 실패 ({len(self._seg_buffer)}개 버퍼 보존): {e}")

    def _on_segment_ready(self, seg: dict):
        self._live_segments.append(seg)

        # 점진적 저장: 10개씩 묶어 DB에 저장
        self._seg_buffer.append(seg)
        if len(self._seg_buffer) >= 10:
            self._flush_segments()

        # "변환 중" 항목을 보고 있을 때만 UI 직접 업데이트
        if self._is_viewing_live():
            self._add_timeline_row(seg)
            self.table_timeline.scrollToBottom()
            # O(1) cursor insert — _build_full_text와 동일한 화자 그룹화 포맷
            cursor = self.txt_fulltext.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            speaker = seg.get("speaker")
            has_content = self.txt_fulltext.toPlainText() != ""
            if speaker and speaker != self._live_last_speaker:
                # 새 화자 → 줄바꿈 후 "화자: 텍스트"
                if has_content:
                    cursor.insertText("\n")
                cursor.insertText(f"{speaker}: {seg['text']}")
                self._live_last_speaker = speaker
            elif speaker:
                # 같은 화자 → 이어붙이기
                cursor.insertText(f" {seg['text']}")
            else:
                # 화자 없음 → 공백 구분
                separator = " " if has_content else ""
                cursor.insertText(f"{separator}{seg['text']}")

    def _release_thread_refs(self):
        """QThread.finished 시그널에 의해 호출 — 스레드 완전 종료 후 참조 해제."""
        self._worker = None
        self._thread = None

    def _cleanup_thread(self, wait_ms: int = 0):
        """QThread/Worker 안전 정리.

        quit()으로 이벤트 루프 종료를 요청한다.
        wait_ms > 0이면 동기 대기 후 참조를 즉시 해제하고,
        그렇지 않으면 thread.finished 시그널에서 비동기로 해제한다.

        Args:
            wait_ms: >0이면 스레드 종료까지 동기 대기 (취소/종료 시 사용).
                     0이면 비동기 — thread.finished에서 참조 해제.
        """
        if self._thread is not None:
            self._thread.quit()
            if wait_ms > 0:
                if not self._thread.wait(wait_ms):
                    print(f"[warning] 워커 스레드가 {wait_ms}ms 내에 종료되지 않음")
                # 동기 대기 완료 — 즉시 해제 안전
                self._worker = None
                self._thread = None

    def _on_finished(self, result: dict):
        try:
            self._on_finished_inner(result)
        except Exception:
            import traceback
            traceback.print_exc()
            from src.crash_reporter import _show_crash_dialog
            tb_str = traceback.format_exc()
            log_text = self.txt_log.toPlainText() if hasattr(self, "txt_log") else ""
            _show_crash_dialog(
                f"변환 완료 처리 중 오류: {traceback.format_exc().splitlines()[-1]}",
                error_traceback=tb_str,
                processing_log=log_text,
                parent=self,
            )

    def _on_finished_inner(self, result: dict):
        was_cancelled = self._cancelled_flag()
        self._cleanup_thread()

        try:
            # 점진적 저장: 남은 버퍼 플러시 + 메타데이터 갱신
            self._flush_segments()
            tid = self._pending_tid
            self._pending_tid = None

            if tid:
                # 취소 시 duration=0 유지 → 이어하기 대상으로 남김
                duration = 0 if was_cancelled else result["duration"]
                self.db.finalize_transcription(tid, result["full_text"], duration)
                self.db.remap_speakers(tid)
            else:
                # fallback: 점진적 저장 실패 시 기존 방식
                tid = self.db.add_transcription(
                    filename=result["filename"],
                    filepath=result["filepath"],
                    duration=result["duration"],
                    full_text=result["full_text"],
                    segments=result["segments"],
                    model_name=result.get("model_name"),
                    language=result.get("language"),
                )
        except Exception as e:
            # DB 저장 실패 시에도 UI 크래시 방지
            tid = self._pending_tid or self._current_tid
            self._pending_tid = None
            import traceback
            traceback.print_exc()

        self._current_tid = tid

        # "변환 중" 항목 제거
        self._remove_live_item()

        # 목록 새로고침
        self._load_list()

        # 최종 결과 표시
        self._show_detail(True)
        self.lbl_title.setText(result["filename"])
        dur = format_duration(result.get("duration"))
        self.lbl_info.setText(f"길이: {dur}")

        # DB에서 세그먼트 재로드 (ID 포함)
        data = self.db.get_transcription(tid) if tid else None
        if data:
            self._populate_timeline(data.get("segments", []))
            self.txt_fulltext.setPlainText(self._build_full_text(data.get("segments", [])))
            has_speakers = any(s.get("speaker") for s in data.get("segments", []))
            self.btn_speakers.setVisible(has_speakers)
        self.btn_edit.setVisible(True)

        self.progress_bar.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setVisible(False)
        self._elapsed_timer.stop()
        self._process_start_time = None

        elapsed_min = result.get("elapsed", 0) / 60
        seg_count = len(result.get("segments", []))

        if was_cancelled:
            self._notify("부분 저장", f"{result['filename']} ({seg_count}개 세그먼트 저장됨)")
            if self._queue:
                self._start_next_in_queue()
            else:
                self._update_queue_display()
                QMessageBox.information(
                    self, "부분 저장",
                    f"취소되었지만 완료된 {seg_count}개 세그먼트가 저장되었습니다."
                )
        else:
            self._notify("변환 완료", f"{result['filename']} (소요시간: {elapsed_min:.1f}분)")
            if self._queue:
                self._start_next_in_queue()
            else:
                self._update_queue_display()
                QMessageBox.information(
                    self, "완료", f"변환 완료! (소요시간: {elapsed_min:.1f}분)"
                )

    def _cancelled_flag(self) -> bool:
        """워커의 취소 상태를 반환한다."""
        if self._worker is not None:
            return self._worker._cancelled
        return False

    def _on_error(self, message: str):
        try:
            self._on_error_inner(message)
        except Exception:
            import traceback
            traceback.print_exc()
            from src.crash_reporter import _show_crash_dialog
            tb_str = traceback.format_exc()
            log_text = self.txt_log.toPlainText() if hasattr(self, "txt_log") else ""
            _show_crash_dialog(
                f"오류 처리 중 추가 오류: {traceback.format_exc().splitlines()[-1]}",
                error_traceback=tb_str,
                processing_log=log_text,
                parent=self,
            )

    def _on_error_inner(self, message: str):
        self._cleanup_thread()

        # 점진적 저장: 에러 시에도 부분 결과 보존
        self._flush_segments()
        tid = self._pending_tid
        self._pending_tid = None
        if tid:
            self.db.remap_speakers(tid)
            self.db.delete_empty_transcription(tid)

        # "변환 중" 항목 제거
        self._remove_live_item()

        self.progress_bar.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setVisible(False)
        self._elapsed_timer.stop()
        self._process_start_time = None
        self._show_detail(False)

        # 부분 결과가 있으면 목록 새로고침
        if tid:
            self._load_list()

        # 사용자 취소 메시지는 보고 대상이 아님
        if "취소" in message:
            self._notify("변환 취소", message)
            self._start_next_in_queue()
            return

        # Feature 7: OS 알림
        self._notify("변환 오류", f"변환 중 오류가 발생했습니다.")

        # 오류 보고 다이얼로그
        log_text = self.txt_log.toPlainText()
        reply = QMessageBox.critical(
            self, "오류",
            f"변환 중 오류 발생:\n{message}\n\n"
            "이 오류를 개발자에게 보고하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            from src.crash_reporter import _show_crash_dialog
            _show_crash_dialog(message, processing_log=log_text, parent=self)

        # Feature 2: 대기열 - 오류 시에도 다음 항목 시작
        self._start_next_in_queue()

    # ── 대기열 관리 (Feature 2) ──

    def _update_queue_display(self):
        if self._queue:
            self.btn_add.setText(f"+ 파일 추가 (대기: {len(self._queue)})")
        else:
            self.btn_add.setText("+ 파일 추가")

    def _start_next_in_queue(self):
        if not self._queue:
            self._update_queue_display()
            return
        path, settings, hf_token = self._queue.pop(0)
        self._update_queue_display()
        if not os.path.exists(path):
            self._notify("파일 없음", f"대기열 파일을 찾을 수 없습니다: {os.path.basename(path)}")
            self._start_next_in_queue()
            return
        self._start_transcription(path, settings, hf_token)

    # ── 기타 액션 ──

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
                import uuid
                temp_map = {}
                for old_name in renames:
                    temp_name = f"__temp_{uuid.uuid4().hex[:8]}"
                    self.db.update_speaker_name(self._current_tid, old_name, temp_name)
                    temp_map[temp_name] = renames[old_name]
                for temp_name, new_name in temp_map.items():
                    self.db.update_speaker_name(self._current_tid, temp_name, new_name)

            # Refresh display
            self._on_tree_item_changed(self.tree_widget.currentItem(), None)

    def _on_retranscribe(self):
        item = self.tree_widget.currentItem()
        if item is None or item.data(0, Qt.ItemDataRole.UserRole + 1) != "transcription":
            return
        tid = item.data(0, Qt.ItemDataRole.UserRole)
        data = self.db.get_transcription(tid)
        if not data:
            return
        filepath = data["filepath"]
        if not os.path.exists(filepath):
            QMessageBox.warning(self, "파일 없음", f"원본 파일을 찾을 수 없습니다:\n{filepath}")
            return
        self._add_file(filepath)

    def _on_delete(self):
        item = self.tree_widget.currentItem()
        if item is None:
            return
        item_type = item.data(0, Qt.ItemDataRole.UserRole + 1)
        item_id = item.data(0, Qt.ItemDataRole.UserRole)

        if item_type == "folder":
            reply = QMessageBox.question(
                self, "삭제 확인",
                f"'{item.text(0)}' 폴더를 삭제하시겠습니까?\n(내부 항목은 상위로 이동됩니다)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.db.delete_folder(item_id)
                self._load_list()
                self._show_detail(False)
        elif item_type == "transcription":
            data = self.db.get_transcription(item_id)
            name = (data.get("display_name") or data["filename"]) if data else "?"
            reply = QMessageBox.question(
                self, "삭제 확인",
                f"'{name}' 트랜스크립션을 삭제하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.db.delete_transcription(item_id)
                self._load_list()
                self._show_detail(False)

    def _on_export(self):
        if self._current_tid is None:
            return
        data = self.db.get_transcription(self._current_tid)
        if not data:
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "내보내기",
            os.path.splitext(data["filename"])[0],
            "SRT 자막 (*.srt);;텍스트 파일 (*.txt)",
        )
        if not path:
            return

        segments = data.get("segments", [])
        if selected_filter.startswith("SRT"):
            content = self._build_srt(segments)
        else:
            content = self._build_full_text(segments)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        self.lbl_status.setVisible(True)
        self.lbl_status.setText(f"저장 완료: {os.path.basename(path)}")
        QTimer.singleShot(3000, lambda: self.lbl_status.setVisible(False))

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _build_srt(self, segments: list[dict]) -> str:
        lines = []
        for i, seg in enumerate(segments, 1):
            start = self._format_srt_time(seg["start"])
            end = self._format_srt_time(seg["end"])
            speaker = seg.get("speaker")
            text = f"[{speaker}] {seg['text']}" if speaker else seg["text"]
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        return "\n".join(lines)

    def _on_copy(self):
        current_tab = self.tabs.currentIndex()
        if current_tab == 0:
            # 타임라인 테이블에서 복사
            lines = []
            for row in range(self.table_timeline.rowCount()):
                if self.table_timeline.isRowHidden(row):
                    continue
                start = self.table_timeline.item(row, 0).text() if self.table_timeline.item(row, 0) else ""
                end = self.table_timeline.item(row, 1).text() if self.table_timeline.item(row, 1) else ""
                speaker = self.table_timeline.item(row, 2).text() if self.table_timeline.item(row, 2) else ""
                text_item = self.table_timeline.item(row, 3)
                text = text_item.text() if text_item else ""
                if speaker and not self.table_timeline.isColumnHidden(2):
                    lines.append(f"[{start} ~ {end}]  [{speaker}]  {text}")
                else:
                    lines.append(f"[{start} ~ {end}]  {text}")
            text = "\n".join(lines)
        else:
            text = self.txt_fulltext.toPlainText()

        if text:
            QApplication.clipboard().setText(text)
            self.lbl_status.setVisible(True)
            self.lbl_status.setText("클립보드에 복사됨!")
            QTimer.singleShot(2000, lambda: self.lbl_status.setVisible(False))

    _MEDIA_EXTENSIONS = {
        ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm",
        ".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a",
    }

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    ext = os.path.splitext(url.toLocalFile())[1].lower()
                    if ext in self._MEDIA_EXTENSIONS:
                        event.acceptProposedAction()
                        return

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in self._MEDIA_EXTENSIONS:
                    self._add_file(path)
                    return

    def _add_file(self, path: str):
        """파일 경로로 트랜스크립션 시작 (버튼/드래그 앤 드롭 공용)."""
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

        # Feature 2: 변환 중이면 대기열에 추가
        if self._thread and self._thread.isRunning():
            self._queue.append((path, settings, hf_token))
            self._update_queue_display()
            filename = os.path.basename(path)
            self.lbl_status.setVisible(True)
            self.lbl_status.setText(f"대기열에 추가됨: {filename} (대기: {len(self._queue)})")
            QTimer.singleShot(3000, lambda: None)  # 상태 유지
            return

        self._start_transcription(path, settings, hf_token)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_glow'):
            self._glow.setGeometry(self.centralWidget().rect())

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            queue_msg = f"\n(대기열: {len(self._queue)}개)" if self._queue else ""
            reply = QMessageBox.question(
                self,
                "종료 확인",
                f"변환이 진행 중입니다. 종료하시겠습니까?{queue_msg}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self._worker.cancel()
            self._cleanup_thread(wait_ms=5000)
            # 스레드 종료 후 최종 플러시 (race condition 방지)
            self._flush_segments()
            if self._pending_tid:
                self.db.remap_speakers(self._pending_tid)
                self.db.delete_empty_transcription(self._pending_tid)
                self._pending_tid = None
        self._elapsed_timer.stop()
        if self._tray_icon:
            self._tray_icon.hide()
        event.accept()
