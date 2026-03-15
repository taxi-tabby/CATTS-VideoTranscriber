# 화자 분석 정확도 개선 + 설정 다이얼로그 구현 계획

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 화자 분석 정확도를 개선하고, 앱 전체 설정을 한 곳에서 관리하는 SettingsDialog를 추가한다.

**Architecture:** config.py에 whisper_model getter/setter 추가 → Database에 db_path 속성 노출 → diarizer.py에 가중 투표 알고리즘 + 화자 수 파라미터 적용 → TranscriberWorker에 model_name/화자 수 파라미터 추가 → SettingsDialog로 기존 DiarizationSetupDialog 대체 → TranscriptionSettingsDialog 추가.

**Tech Stack:** Python 3.11+, PySide6, OpenAI Whisper, pyannote.audio, SQLite3

---

## 파일 구조

| 파일 | 역할 | 변경 유형 |
|------|------|-----------|
| `src/config.py` | whisper_model getter/setter 추가 | 수정 |
| `tests/test_config.py` | whisper_model 테스트 추가 | 수정 |
| `src/database.py` | db_path 속성 노출 | 수정 |
| `src/diarizer.py` | 가중 투표 알고리즘, 화자 수 파라미터 | 수정 |
| `tests/test_diarizer.py` | 가중 투표 + tie-break 테스트 | 수정 |
| `src/transcriber.py` | model_name, 화자 수 파라미터 추가 | 수정 |
| `src/main_window.py` | SettingsDialog, TranscriptionSettingsDialog 추가, DiarizationSetupDialog 제거 | 수정 |

---

## Task 1: config.py — whisper_model getter/setter

**Files:**
- Modify: `src/config.py:20-33`
- Test: `tests/test_config.py`

- [ ] **Step 1: 테스트 작성**

`tests/test_config.py`에 추가:

```python
from src.config import get_whisper_model, set_whisper_model

# TestConfig 클래스 안에:
def test_get_whisper_model_default(self, config_dir):
    assert get_whisper_model() == "medium"

def test_set_and_get_whisper_model(self, config_dir):
    set_whisper_model("large-v3")
    assert get_whisper_model() == "large-v3"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `get_whisper_model` not defined

- [ ] **Step 3: 구현**

`src/config.py` 끝에 추가:

```python
def get_whisper_model() -> str:
    return load_config().get("whisper_model", "medium")


def set_whisper_model(model: str) -> None:
    config = load_config()
    config["whisper_model"] = model
    save_config(config)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_config.py -v`
Expected: ALL PASS

- [ ] **Step 5: 커밋**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add whisper_model getter/setter to config"
```

---

## Task 2: Database — db_path 속성 노출

**Files:**
- Modify: `src/database.py:6-7`

- [ ] **Step 1: 구현**

`src/database.py`의 `Database.__init__` 수정:

```python
def __init__(self, db_path: str):
    self.db_path = db_path
    self._conn = sqlite3.connect(db_path)
```

- [ ] **Step 2: 기존 테스트 통과 확인**

Run: `python -m pytest tests/test_database.py -v`
Expected: ALL PASS

- [ ] **Step 3: 커밋**

```bash
git add src/database.py
git commit -m "feat: expose db_path property on Database"
```

---

## Task 3: diarizer.py — 가중 투표 알고리즘

**Files:**
- Modify: `src/diarizer.py:51-77`
- Test: `tests/test_diarizer.py`

- [ ] **Step 1: 가중 투표 테스트 작성**

`tests/test_diarizer.py`의 `TestAssignSpeakers` 클래스에 추가:

```python
def test_weighted_voting_aggregates_overlap(self):
    """같은 화자의 여러 구간 겹침을 합산하여 배정"""
    diarization_segments = [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 8.0, "speaker": "SPEAKER_01"},
        {"start": 8.0, "end": 10.0, "speaker": "SPEAKER_00"},
    ]
    whisper_segments = [
        {"start": 0.0, "end": 10.0, "text": "전체 구간"},
    ]
    result = assign_speakers(diarization_segments, whisper_segments)
    # SPEAKER_00: 3+2=5초, SPEAKER_01: 5초 → 동률 → SPEAKER_00 (먼저 등장)
    assert result[0]["speaker"] == "SPEAKER_00"

def test_weighted_voting_tiebreak_by_earliest(self):
    """동률 시 해당 구간에서 먼저 등장한 화자 선택"""
    diarization_segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_01"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_00"},
    ]
    whisper_segments = [
        {"start": 0.0, "end": 10.0, "text": "동률 구간"},
    ]
    result = assign_speakers(diarization_segments, whisper_segments)
    # 둘 다 5초 → SPEAKER_01이 0초에 먼저 등장
    assert result[0]["speaker"] == "SPEAKER_01"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_diarizer.py::TestAssignSpeakers::test_weighted_voting_aggregates_overlap -v`
Expected: FAIL — 현재 구현은 단일 최대 겹침만 사용

- [ ] **Step 3: assign_speakers 가중 투표 구현**

`src/diarizer.py`의 `assign_speakers` 함수를 교체:

```python
from collections import defaultdict

def assign_speakers(
    diarization_segments: list[dict],
    whisper_segments: list[dict],
) -> list[dict]:
    """가중 투표 방식으로 각 Whisper 세그먼트에 화자를 할당.

    같은 화자의 여러 diarization 세그먼트 겹침을 합산하고,
    동률 시 해당 구간에서 가장 이른 start를 가진 화자를 선택.
    """
    result = []
    for wseg in whisper_segments:
        speaker_overlap: dict[str, float] = defaultdict(float)
        speaker_earliest: dict[str, float] = {}

        for dseg in diarization_segments:
            overlap_start = max(wseg["start"], dseg["start"])
            overlap_end = min(wseg["end"], dseg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > 0:
                speaker = dseg["speaker"]
                speaker_overlap[speaker] += overlap
                if speaker not in speaker_earliest:
                    speaker_earliest[speaker] = dseg["start"]

        best_speaker = None
        if speaker_overlap:
            best_speaker = min(
                speaker_overlap,
                key=lambda s: (-speaker_overlap[s], speaker_earliest.get(s, float("inf"))),
            )

        result.append({
            "start": wseg["start"],
            "end": wseg["end"],
            "text": wseg["text"],
            "speaker": best_speaker,
        })

    return result
```

- [ ] **Step 4: 모든 diarizer 테스트 통과 확인**

Run: `python -m pytest tests/test_diarizer.py -v`
Expected: ALL PASS (기존 테스트도 새 알고리즘에서 동일한 결과)

- [ ] **Step 5: 커밋**

```bash
git add src/diarizer.py tests/test_diarizer.py
git commit -m "feat: weighted voting speaker assignment algorithm"
```

---

## Task 4: diarizer.py — 화자 수 파라미터

**Files:**
- Modify: `src/diarizer.py:7-48`

- [ ] **Step 1: run_diarization 시그니처 확장**

`src/diarizer.py`의 `run_diarization` 함수:

```python
def run_diarization(
    audio_path: str,
    hf_token: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict]:
```

pipeline 호출 부분을 변경:

```python
    # num_speakers가 있으면 min/max 무시
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    diarization = pipeline(audio_path, **kwargs)
```

- [ ] **Step 2: 기존 테스트 통과 확인**

Run: `python -m pytest tests/test_diarizer.py -v`
Expected: ALL PASS (기본값 None이므로 기존 동작 유지)

- [ ] **Step 3: 커밋**

```bash
git add src/diarizer.py
git commit -m "feat: add speaker count parameters to run_diarization"
```

---

## Task 5: TranscriberWorker — model_name + 화자 수 파라미터

**Files:**
- Modify: `src/transcriber.py:51-105`

- [ ] **Step 1: __init__ 시그니처 확장**

```python
def __init__(
    self,
    video_path: str,
    use_diarization: bool = False,
    hf_token: str | None = None,
    model_name: str = "medium",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
):
    super().__init__()
    self.video_path = video_path
    self.use_diarization = use_diarization
    self.hf_token = hf_token
    self.model_name = model_name
    self.num_speakers = num_speakers
    self.min_speakers = min_speakers
    self.max_speakers = max_speakers
    self._cancelled = False
```

- [ ] **Step 2: run_diarization 호출에 파라미터 전달**

`src/transcriber.py` 97행 부근 변경:

```python
diarization_segments = run_diarization(
    tmp_wav, self.hf_token,
    num_speakers=self.num_speakers,
    min_speakers=self.min_speakers,
    max_speakers=self.max_speakers,
)
```

- [ ] **Step 3: Whisper 모델 로딩에 model_name 사용**

`src/transcriber.py` 105행 변경:

```python
model = whisper.load_model(self.model_name)
```

- [ ] **Step 4: 기존 테스트 통과 확인**

Run: `python -m pytest -v`
Expected: ALL PASS

- [ ] **Step 5: 커밋**

```bash
git add src/transcriber.py
git commit -m "feat: add model_name and speaker count params to TranscriberWorker"
```

---

## Task 6: SettingsDialog — 기존 DiarizationSetupDialog 대체

**Files:**
- Modify: `src/main_window.py:1-201` (DiarizationSetupDialog 제거), `src/main_window.py:253-254` (설정 버튼), `src/main_window.py:414-417` (_on_diarization_settings 교체)

- [ ] **Step 1: import 추가**

`src/main_window.py` import 블록에 추가:

```python
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QCheckBox, QComboBox, QSpinBox
```

`src/config.py` import 확장:

```python
from src.config import get_hf_token, set_hf_token, delete_hf_token, get_whisper_model, set_whisper_model
```

- [ ] **Step 2: DiarizationSetupDialog 클래스를 SettingsDialog로 교체**

`src/main_window.py`에서 `DiarizationSetupDialog` 클래스(52~201행)를 제거하고 다음으로 교체:

```python
class SettingsDialog(QDialog):
    """앱 설정 다이얼로그. 일반 + 화자 분리 탭."""

    HF_MODELS = [
        ("pyannote/speaker-diarization-3.1", "https://hf.co/pyannote/speaker-diarization-3.1"),
        ("pyannote/segmentation-3.0", "https://hf.co/pyannote/segmentation-3.0"),
    ]

    WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

    def __init__(self, db_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.setMinimumWidth(520)
        self._build_ui(db_path)

    def _build_ui(self, db_path: str):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # ── 일반 탭 ──
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)

        # Whisper 모델 선택
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

        # DB 경로
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

        general_layout.addStretch()
        tabs.addTab(general_tab, "일반")

        # ── 화자 분리 탭 ──
        diar_tab = QWidget()
        diar_layout = QVBoxLayout(diar_tab)

        # 토큰 입력
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

        # 검증/삭제 버튼
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

        # 라이선스 링크
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

        layout.addWidget(tabs)

        # 하단 버튼
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
        # Whisper 모델 저장
        set_whisper_model(self.combo_model.currentText())
        # 토큰 저장 (입력된 경우)
        token = self.edit_token.text().strip()
        if token:
            set_hf_token(token)
        self.accept()
```

- [ ] **Step 3: 설정 버튼 텍스트 및 핸들러 변경**

`src/main_window.py`에서 설정 버튼 텍스트를 "화자 분리 설정" → "설정"으로 변경:

```python
self.btn_settings = QPushButton("설정")
self.btn_settings.clicked.connect(self._on_settings)
```

`_on_diarization_settings` 메서드를 `_on_settings`로 교체:

```python
def _on_settings(self):
    dialog = SettingsDialog(self.db.db_path, self)
    dialog.exec()
```

- [ ] **Step 4: 수동 테스트**

앱을 실행하여 설정 버튼 → 일반 탭 (모델 선택, 폴더 열기), 화자 분리 탭 (토큰 관리) 확인.

Run: `python -m src.main`

- [ ] **Step 5: 커밋**

```bash
git add src/main_window.py
git commit -m "feat: replace DiarizationSetupDialog with SettingsDialog"
```

---

## Task 7: TranscriptionSettingsDialog — 트랜스크립션 시작 전 설정

**Files:**
- Modify: `src/main_window.py` (_on_add_video, _start_transcription)

- [ ] **Step 1: TranscriptionSettingsDialog 클래스 추가**

`src/main_window.py`에 `SettingsDialog` 클래스 다음에 추가:

```python
class TranscriptionSettingsDialog(QDialog):
    """트랜스크립션 시작 전 설정 다이얼로그."""

    WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("트랜스크립션 설정")
        self.setMinimumWidth(450)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Whisper 모델
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

        # 화자 분리
        grp_diar = QGroupBox("화자 분리")
        diar_layout = QVBoxLayout(grp_diar)

        self.chk_diarization = QCheckBox("화자 분리 사용")
        self.chk_diarization.toggled.connect(self._on_diar_toggled)
        diar_layout.addWidget(self.chk_diarization)

        # 화자 수 설정
        self.speaker_widget = QWidget()
        speaker_layout = QVBoxLayout(self.speaker_widget)
        speaker_layout.setContentsMargins(20, 0, 0, 0)

        # 자동/직접 선택
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("화자 수:"))
        self.combo_speaker_mode = QComboBox()
        self.combo_speaker_mode.addItems(["자동 감지", "직접 지정"])
        self.combo_speaker_mode.currentIndexChanged.connect(self._on_speaker_mode_changed)
        mode_row.addWidget(self.combo_speaker_mode)
        mode_row.addStretch()
        speaker_layout.addLayout(mode_row)

        # 직접 지정 위젯
        self.manual_widget = QWidget()
        manual_layout = QVBoxLayout(self.manual_widget)
        manual_layout.setContentsMargins(0, 0, 0, 0)

        # 정확한 화자 수
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

        # 최소/최대
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

        # 검증 경고
        self.lbl_warning = QLabel("")
        self.lbl_warning.setStyleSheet("color: red;")
        manual_layout.addWidget(self.lbl_warning)

        self.manual_widget.setVisible(False)
        speaker_layout.addWidget(self.manual_widget)

        self.speaker_widget.setVisible(False)
        diar_layout.addWidget(self.speaker_widget)

        layout.addWidget(grp_diar)

        # 버튼
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_cancel = QPushButton("취소")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        self.btn_start = QPushButton("시작")
        self.btn_start.setDefault(True)
        self.btn_start.clicked.connect(self._on_start)
        btn_layout.addWidget(self.btn_start)
        layout.addLayout(btn_layout)

        # min/max 변경 시 검증
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

    def _on_start(self):
        self.accept()

    def get_settings(self) -> dict:
        """다이얼로그에서 선택한 설정 반환."""
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
```

- [ ] **Step 2: _on_add_video 변경**

기존 `_on_add_video` 메서드를 교체:

```python
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

    # 트랜스크립션 설정 다이얼로그
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
            dialog = SettingsDialog(self.db.db_path, self)
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
```

- [ ] **Step 3: _start_transcription 시그니처 변경**

```python
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
```

- [ ] **Step 4: 수동 테스트**

앱을 실행하여 파일 추가 → 설정 다이얼로그 → 모델 선택 + 화자 분리 옵션 확인.

Run: `python -m src.main`

- [ ] **Step 5: 커밋**

```bash
git add src/main_window.py
git commit -m "feat: add TranscriptionSettingsDialog with model/speaker options"
```

---

## Task 8: 전체 테스트 통과 확인

- [ ] **Step 1: 전체 테스트 실행**

Run: `python -m pytest -v`
Expected: ALL PASS

- [ ] **Step 2: 최종 커밋 (필요시)**

변경 사항 있으면 커밋.
