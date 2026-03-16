# Speaker Diarization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** pyannote.audio 기반 화자 분리 기능을 추가하여 각 세그먼트에 화자 라벨을 부여하고, UI에서 화자 이름을 편집할 수 있게 한다.

**Architecture:** 전체 오디오에 pyannote diarization을 먼저 실행하여 화자별 시간 구간을 얻고, Whisper 청크 변환 시 각 세그먼트를 최대 겹침 방식으로 화자에 매칭한다. pyannote 모델은 처리 후 해제하고 Whisper를 순차 로드하여 GPU 메모리 충돌을 방지한다.

**Tech Stack:** pyannote.audio 3.x, openai-whisper, PySide6, SQLite

**Spec:** `docs/superpowers/specs/2026-03-14-speaker-diarization-design.md`

---

## File Structure

| File | Role |
|------|------|
| `src/config.py` | **Create** — HuggingFace 토큰 로드/저장 (`~/.video-transcriber/config.json`) |
| `src/diarizer.py` | **Create** — pyannote diarization 래퍼 + 세그먼트-화자 매칭 로직 |
| `src/database.py` | **Modify** — speaker 컬럼 마이그레이션, 쿼리 변경, 화자 관련 메서드 |
| `src/transcriber.py` | **Modify** — diarization 단계 통합, segment에 speaker 포함 |
| `src/main_window.py` | **Modify** — 화자 분리 확인 다이얼로그, 화자 표시, 화자 관리 다이얼로그, 전체텍스트 화자별 렌더링 |
| `requirements.txt` | **Modify** — pyannote.audio 추가 |
| `build/video_transcriber.spec` | **Modify** — hidden imports 추가 |
| `tests/test_config.py` | **Create** — config 로드/저장 테스트 |
| `tests/test_diarizer.py` | **Create** — 매칭 로직 테스트 |
| `tests/test_database.py` | **Modify** — speaker 관련 테스트 추가 |

---

## Chunk 1: Dependencies & Config

### Task 1: Add pyannote.audio dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependency**

`requirements.txt`에 다음 줄 추가:
```
pyannote.audio>=3.1,<4.0
```

- [ ] **Step 2: Install and verify**

Run: `pip install pyannote.audio>=3.1,<4.0`
Expected: 설치 성공 (speechbrain, transformers 등 전이 의존성 포함)

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add pyannote.audio dependency for speaker diarization"
```

---

### Task 2: Create config module for HuggingFace token

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

`tests/test_config.py`:
```python
import json
import pytest
from src.config import load_config, save_config, get_hf_token, set_hf_token


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    config_path = tmp_path / ".video-transcriber" / "config.json"
    monkeypatch.setattr("src.config.CONFIG_PATH", str(config_path))
    return config_path


class TestConfig:
    def test_load_empty_config(self, config_dir):
        result = load_config()
        assert result == {}

    def test_save_and_load(self, config_dir):
        save_config({"hf_token": "test_token"})
        result = load_config()
        assert result["hf_token"] == "test_token"

    def test_get_hf_token_missing(self, config_dir):
        assert get_hf_token() is None

    def test_set_and_get_hf_token(self, config_dir):
        set_hf_token("hf_abc123")
        assert get_hf_token() == "hf_abc123"

    def test_save_creates_directory(self, config_dir):
        save_config({"key": "value"})
        assert config_dir.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Implement config module**

`src/config.py`:
```python
import json
import os

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".video-transcriber", "config.json")


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_hf_token() -> str | None:
    return load_config().get("hf_token")


def set_hf_token(token: str) -> None:
    config = load_config()
    config["hf_token"] = token
    save_config(config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add config module for HuggingFace token management"
```

---

## Chunk 2: Diarizer Module

### Task 3: Create diarizer module with speaker matching logic

**Files:**
- Create: `src/diarizer.py`
- Create: `tests/test_diarizer.py`

- [ ] **Step 1: Write failing tests for `assign_speakers`**

`assign_speakers`는 순수 함수로, pyannote 없이 테스트 가능. diarization 결과(시간 구간 + 화자)와 Whisper 세그먼트를 받아 최대 겹침 매칭.

`tests/test_diarizer.py`:
```python
import pytest
from src.diarizer import assign_speakers, map_speaker_labels


class TestAssignSpeakers:
    def test_single_speaker(self):
        diarization_segments = [
            {"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"},
        ]
        whisper_segments = [
            {"start": 0.0, "end": 10.0, "text": "안녕하세요"},
            {"start": 10.0, "end": 20.0, "text": "반갑습니다"},
        ]
        result = assign_speakers(diarization_segments, whisper_segments)
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_00"
        assert result[0]["text"] == "안녕하세요"

    def test_two_speakers(self):
        diarization_segments = [
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
            {"start": 10.0, "end": 20.0, "speaker": "SPEAKER_01"},
        ]
        whisper_segments = [
            {"start": 0.0, "end": 9.0, "text": "첫번째"},
            {"start": 10.0, "end": 19.0, "text": "두번째"},
        ]
        result = assign_speakers(diarization_segments, whisper_segments)
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_maximum_overlap_matching(self):
        """세그먼트가 두 화자 구간에 걸칠 때 더 많이 겹치는 화자에 매칭"""
        diarization_segments = [
            {"start": 0.0, "end": 7.0, "speaker": "SPEAKER_00"},
            {"start": 7.0, "end": 15.0, "speaker": "SPEAKER_01"},
        ]
        whisper_segments = [
            {"start": 5.0, "end": 12.0, "text": "겹치는 구간"},
        ]
        result = assign_speakers(diarization_segments, whisper_segments)
        # 5~7 = 2초 SPEAKER_00, 7~12 = 5초 SPEAKER_01 → SPEAKER_01
        assert result[0]["speaker"] == "SPEAKER_01"

    def test_no_overlap_returns_none(self):
        diarization_segments = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        ]
        whisper_segments = [
            {"start": 10.0, "end": 15.0, "text": "겹침없음"},
        ]
        result = assign_speakers(diarization_segments, whisper_segments)
        assert result[0]["speaker"] is None

    def test_empty_diarization(self):
        whisper_segments = [
            {"start": 0.0, "end": 5.0, "text": "텍스트"},
        ]
        result = assign_speakers([], whisper_segments)
        assert result[0]["speaker"] is None

    def test_preserves_segment_fields(self):
        diarization_segments = [
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
        ]
        whisper_segments = [
            {"start": 1.0, "end": 5.0, "text": "보존 테스트"},
        ]
        result = assign_speakers(diarization_segments, whisper_segments)
        assert result[0]["start"] == 1.0
        assert result[0]["end"] == 5.0
        assert result[0]["text"] == "보존 테스트"


class TestMapSpeakerLabels:
    def test_maps_to_korean_labels(self):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "a", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "text": "b", "speaker": "SPEAKER_01"},
            {"start": 10.0, "end": 15.0, "text": "c", "speaker": "SPEAKER_00"},
        ]
        result = map_speaker_labels(segments)
        assert result[0]["speaker"] == "화자 1"
        assert result[1]["speaker"] == "화자 2"
        assert result[2]["speaker"] == "화자 1"

    def test_none_speaker_preserved(self):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "a", "speaker": None},
        ]
        result = map_speaker_labels(segments)
        assert result[0]["speaker"] is None

    def test_labels_by_first_appearance(self):
        """화자 라벨은 첫 등장 순서로 매핑"""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "a", "speaker": "SPEAKER_02"},
            {"start": 5.0, "end": 10.0, "text": "b", "speaker": "SPEAKER_00"},
        ]
        result = map_speaker_labels(segments)
        assert result[0]["speaker"] == "화자 1"
        assert result[1]["speaker"] == "화자 2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_diarizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.diarizer'`

- [ ] **Step 3: Implement diarizer module**

`src/diarizer.py`:
```python
import gc
import torch


def run_diarization(audio_path: str, hf_token: str) -> list[dict]:
    """pyannote.audio로 화자 분리 실행. 결과는 [{start, end, speaker}, ...] 리스트."""
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    # 모델 해제 — GPU 메모리 확보
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return segments


def assign_speakers(
    diarization_segments: list[dict],
    whisper_segments: list[dict],
) -> list[dict]:
    """최대 겹침 방식으로 각 Whisper 세그먼트에 화자를 할당."""
    result = []
    for wseg in whisper_segments:
        best_speaker = None
        best_overlap = 0.0

        for dseg in diarization_segments:
            overlap_start = max(wseg["start"], dseg["start"])
            overlap_end = min(wseg["end"], dseg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg["speaker"]

        result.append({
            "start": wseg["start"],
            "end": wseg["end"],
            "text": wseg["text"],
            "speaker": best_speaker,
        })

    return result


def map_speaker_labels(segments: list[dict]) -> list[dict]:
    """pyannote 라벨(SPEAKER_00)을 한글 라벨(화자 1)로 매핑. 첫 등장 순서 기준."""
    label_map = {}
    counter = 0

    result = []
    for seg in segments:
        raw = seg["speaker"]
        if raw is None:
            mapped = None
        else:
            if raw not in label_map:
                counter += 1
                label_map[raw] = f"화자 {counter}"
            mapped = label_map[raw]

        result.append({**seg, "speaker": mapped})

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_diarizer.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/diarizer.py tests/test_diarizer.py
git commit -m "feat: add diarizer module with speaker matching logic"
```

---

## Chunk 3: Database Changes

### Task 4: Add speaker column and migration

**Files:**
- Modify: `src/database.py`
- Modify: `tests/test_database.py`

- [ ] **Step 1: Write failing tests**

`tests/test_database.py`에 추가:
```python
    def test_migration_adds_speaker_column(self, db):
        """speaker 컬럼이 마이그레이션으로 추가되는지 확인"""
        tid = db.add_transcription(
            "test.mp4", "C:/test.mp4", 60.0, "텍스트",
            [{"start": 0.0, "end": 5.0, "text": "세그먼트", "speaker": "화자 1"}],
        )
        result = db.get_transcription(tid)
        assert result["segments"][0]["speaker"] == "화자 1"

    def test_segment_without_speaker(self, db):
        """speaker 없는 세그먼트도 정상 동작"""
        tid = db.add_transcription(
            "test.mp4", "C:/test.mp4", 60.0, "텍스트",
            [{"start": 0.0, "end": 5.0, "text": "세그먼트"}],
        )
        result = db.get_transcription(tid)
        assert result["segments"][0]["speaker"] is None

    def test_get_speakers(self, db):
        tid = db.add_transcription(
            "test.mp4", "C:/test.mp4", 60.0, "텍스트",
            [
                {"start": 0.0, "end": 5.0, "text": "a", "speaker": "화자 1"},
                {"start": 5.0, "end": 10.0, "text": "b", "speaker": "화자 2"},
                {"start": 10.0, "end": 15.0, "text": "c", "speaker": "화자 1"},
            ],
        )
        speakers = db.get_speakers(tid)
        assert set(speakers) == {"화자 1", "화자 2"}

    def test_update_speaker_name(self, db):
        tid = db.add_transcription(
            "test.mp4", "C:/test.mp4", 60.0, "텍스트",
            [
                {"start": 0.0, "end": 5.0, "text": "a", "speaker": "화자 1"},
                {"start": 5.0, "end": 10.0, "text": "b", "speaker": "화자 2"},
                {"start": 10.0, "end": 15.0, "text": "c", "speaker": "화자 1"},
            ],
        )
        db.update_speaker_name(tid, "화자 1", "김대리")
        result = db.get_transcription(tid)
        assert result["segments"][0]["speaker"] == "김대리"
        assert result["segments"][1]["speaker"] == "화자 2"
        assert result["segments"][2]["speaker"] == "김대리"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_database.py -v`
Expected: FAIL — 새 테스트 실패

- [ ] **Step 3: Modify database.py**

`src/database.py` 변경 사항:

1. `_create_tables()`에 마이그레이션 추가:
```python
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
    self._migrate_add_speaker_column()

def _migrate_add_speaker_column(self):
    columns = [
        row[1] for row in
        self._conn.execute("PRAGMA table_info(segments)").fetchall()
    ]
    if "speaker" not in columns:
        self._conn.execute("ALTER TABLE segments ADD COLUMN speaker TEXT")
        self._conn.commit()
```

2. `add_transcription()`에서 speaker 포함:
```python
def add_transcription(self, filename, filepath, duration, full_text, segments) -> int:
    cur = self._conn.execute(
        """INSERT INTO transcriptions (filename, filepath, duration, created_at, full_text)
           VALUES (?, ?, ?, ?, ?)""",
        (filename, filepath, duration, datetime.now().isoformat(), full_text),
    )
    tid = cur.lastrowid
    for seg in segments:
        self._conn.execute(
            """INSERT INTO segments (transcription_id, start_time, end_time, text, speaker)
               VALUES (?, ?, ?, ?, ?)""",
            (tid, seg["start"], seg["end"], seg["text"], seg.get("speaker")),
        )
    self._conn.commit()
    return tid
```

3. `get_transcription()`에서 speaker 조회:
```python
seg_rows = self._conn.execute(
    "SELECT start_time as start, end_time as end, text, speaker FROM segments WHERE transcription_id = ? ORDER BY start_time",
    (tid,),
).fetchall()
```

4. 새 메서드 추가:
```python
def get_speakers(self, tid: int) -> list[str]:
    rows = self._conn.execute(
        "SELECT DISTINCT speaker FROM segments WHERE transcription_id = ? AND speaker IS NOT NULL ORDER BY speaker",
        (tid,),
    ).fetchall()
    return [row[0] for row in rows]

def update_speaker_name(self, tid: int, old_name: str, new_name: str) -> None:
    self._conn.execute(
        "UPDATE segments SET speaker = ? WHERE transcription_id = ? AND speaker = ?",
        (new_name, tid, old_name),
    )
    self._conn.commit()
```

- [ ] **Step 4: Run all database tests**

Run: `python -m pytest tests/test_database.py -v`
Expected: All tests PASS (기존 5개 + 새로운 4개)

- [ ] **Step 5: Commit**

```bash
git add src/database.py tests/test_database.py
git commit -m "feat: add speaker column to segments with migration and query methods"
```

---

## Chunk 4: Transcriber Integration

### Task 5: Integrate diarization into TranscriberWorker

**Files:**
- Modify: `src/transcriber.py`

- [ ] **Step 1: Add `use_diarization` and `hf_token` parameters to TranscriberWorker.__init__**

```python
def __init__(self, video_path: str, use_diarization: bool = False, hf_token: str | None = None):
    super().__init__()
    self.video_path = video_path
    self.use_diarization = use_diarization
    self.hf_token = hf_token
    self._cancelled = False
```

- [ ] **Step 2: Add diarization step in run() between audio load and Whisper model load**

`run()` 메서드에서 Step 2 (오디오 로드) 이후, Step 3 (Whisper 모델 로드) 이전에 추가:

```python
# Step 2.5: Speaker diarization (optional)
diarization_segments = None
if self.use_diarization and self.hf_token:
    self.progress.emit(8, "화자 분석 중... (취소 불가)")
    from src.diarizer import run_diarization
    diarization_segments = run_diarization(tmp_wav, self.hf_token)
    self.progress.emit(18, "화자 분석 완료")
    if self._cancelled:
        return
```

- [ ] **Step 3: Adjust progress percentages**

diarization 사용 시:
- 음성 추출: 5%
- 오디오 로드: 8%
- 화자 분석: 8% ~ 18%
- Whisper 모델 로드: 18% ~ 22%
- 텍스트 변환: 22% ~ 95%

diarization 미사용 시 (기존과 동일):
- 음성 추출: 5%
- 오디오 로드: 10%
- Whisper 모델 로드: 15%
- 텍스트 변환: 20% ~ 95%

프로그레스 시작점을 변수로:
```python
transcribe_start_pct = 22 if self.use_diarization else 20
model_load_pct = 18 if self.use_diarization else 15
```

- [ ] **Step 4: Apply speaker matching after each chunk**

청크 루프 내부에서, 세그먼트 생성 후 화자 매칭 적용:

```python
for seg in result.get("segments", []):
    adjusted = {
        "start": seg["start"] + time_offset,
        "end": seg["end"] + time_offset,
        "text": seg["text"].strip(),
        "speaker": None,
    }
    if adjusted["text"]:
        all_segments.append(adjusted)

# 이 청크의 세그먼트에 화자 매칭 적용
if diarization_segments:
    from src.diarizer import assign_speakers, map_speaker_labels
    chunk_segs = all_segments[chunk_seg_start:]
    matched = assign_speakers(diarization_segments, chunk_segs)
    matched = map_speaker_labels(matched)
    all_segments[chunk_seg_start:] = matched

for seg in all_segments[chunk_seg_start:]:
    self.segment_ready.emit(seg)
```

주의: `map_speaker_labels`는 전체 세그먼트에 대해 실행해야 라벨 일관성 유지. 따라서 청크 루프 밖에서 한 번만 호출하도록 변경:

```python
# 청크 루프 내부 — 세그먼트 수집만
for seg in result.get("segments", []):
    adjusted = {
        "start": seg["start"] + time_offset,
        "end": seg["end"] + time_offset,
        "text": seg["text"].strip(),
    }
    if adjusted["text"]:
        all_segments.append(adjusted)

# 청크 루프 내부 — 화자 매칭 후 emit
if diarization_segments:
    from src.diarizer import assign_speakers
    chunk_segs = all_segments[chunk_seg_start:]
    matched = assign_speakers(diarization_segments, chunk_segs)
    all_segments[chunk_seg_start:] = matched

for seg in all_segments[chunk_seg_start:]:
    self.segment_ready.emit(seg)
```

청크 루프 종료 후, 최종 라벨 매핑:
```python
if diarization_segments:
    from src.diarizer import map_speaker_labels
    all_segments = map_speaker_labels(all_segments)
```

**수정된 run() 전체 로직:**

```python
def run(self):
    tmp_wav = None
    try:
        ffmpeg = get_ffmpeg_exe()
        use_diar = self.use_diarization and self.hf_token

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
        self.progress.emit(8 if use_diar else 10, "오디오 로드 중...")
        audio = load_wav_as_numpy(tmp_wav)
        duration = get_video_duration(audio)
        if self._cancelled:
            return

        # Step 2.5: Speaker diarization (optional)
        diarization_segments = None
        if use_diar:
            self.progress.emit(8, "화자 분석 중... (취소 불가)")
            from src.diarizer import run_diarization
            diarization_segments = run_diarization(tmp_wav, self.hf_token)
            self.progress.emit(18, "화자 분석 완료")
            if self._cancelled:
                return

        # Step 3: Load Whisper model
        model_pct = 18 if use_diar else 15
        self.progress.emit(model_pct, "Whisper 모델 로드 중... (첫 실행 시 다운로드)")
        model = whisper.load_model("medium")
        if self._cancelled:
            return

        # Step 4: Transcribe in chunks
        start_time = time.time()
        total_samples = len(audio)
        all_segments = []
        full_text_parts = []
        prev_text = ""
        transcribe_start = 22 if use_diar else 20

        for chunk_start in range(0, total_samples, CHUNK_SAMPLES):
            if self._cancelled:
                return

            chunk_end = min(chunk_start + CHUNK_SAMPLES, total_samples)
            chunk = audio[chunk_start:chunk_end]

            time_offset = chunk_start / SAMPLE_RATE
            processed_sec = min(chunk_end / SAMPLE_RATE, duration)

            pct = transcribe_start + int((processed_sec / duration) * (95 - transcribe_start))
            pct = min(pct, 95)
            self.progress.emit(pct, f"변환 중... {processed_sec:.0f}s / {duration:.0f}s")

            result = model.transcribe(
                chunk, language="ko", verbose=False,
                initial_prompt=prev_text[-200:] if prev_text else None,
            )

            chunk_text = result.get("text", "").strip()
            if chunk_text:
                full_text_parts.append(chunk_text)
                prev_text = chunk_text

            chunk_seg_start = len(all_segments)
            for seg in result.get("segments", []):
                adjusted = {
                    "start": seg["start"] + time_offset,
                    "end": seg["end"] + time_offset,
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

        # 최종 한글 라벨 매핑
        if diarization_segments:
            from src.diarizer import map_speaker_labels
            all_segments = map_speaker_labels(all_segments)

        elapsed = time.time() - start_time

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

- [ ] **Step 5: Commit**

```bash
git add src/transcriber.py
git commit -m "feat: integrate speaker diarization into transcription pipeline"
```

---

## Chunk 5: UI — Diarization Toggle & Speaker Display

### Task 6: Add diarization confirmation dialog and speaker display

**Files:**
- Modify: `src/main_window.py`

- [ ] **Step 1: Add imports**

```python
from PySide6.QtWidgets import (
    # ... 기존 imports에 추가 ...
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QInputDialog,
    QLineEdit,
)
from src.config import get_hf_token, set_hf_token
```

- [ ] **Step 2: Add diarization confirmation in `_on_add_video`**

파일 선택 후, 변환 시작 전에 화자 분리 사용 여부를 묻는 다이얼로그:

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

    # 화자 분리 사용 여부 확인
    use_diarization = False
    hf_token = None
    reply = QMessageBox.question(
        self,
        "화자 분리",
        "화자 분리 기능을 사용하시겠습니까?\n(화자별로 발언을 구분합니다)",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    if reply == QMessageBox.StandardButton.Yes:
        hf_token = get_hf_token()
        if not hf_token:
            token, ok = QInputDialog.getText(
                self,
                "HuggingFace 토큰",
                "화자 분리를 위해 HuggingFace 토큰이 필요합니다.\n"
                "https://huggingface.co/settings/tokens 에서 발급받으세요.\n\n"
                "토큰:",
                QLineEdit.EchoMode.Password,
            )
            if ok and token.strip():
                hf_token = token.strip()
                set_hf_token(hf_token)
            else:
                # 토큰 없으면 화자 분리 없이 진행
                hf_token = None

        if hf_token:
            use_diarization = True

    self._start_transcription(path, use_diarization, hf_token)
```

- [ ] **Step 3: Update `_start_transcription` to accept diarization params**

```python
def _start_transcription(self, video_path: str, use_diarization: bool = False, hf_token: str | None = None):
    self.btn_add.setEnabled(False)
    self.progress_bar.setVisible(True)
    self.progress_bar.setValue(0)
    self.lbl_status.setVisible(True)
    self.lbl_status.setText("준비 중...")

    # Show right panel with live transcription
    self._show_detail(True)
    self.lbl_title.setText(os.path.basename(video_path))
    self.lbl_info.setText("변환 진행 중...")
    self.txt_timeline.clear()
    self.txt_fulltext.clear()
    self._live_segments = []
    self.btn_speakers.setVisible(False)

    self._thread = QThread()
    self._worker = TranscriberWorker(video_path, use_diarization, hf_token)
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

- [ ] **Step 4: Update `_on_segment_ready` to show speaker labels**

```python
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
    scrollbar = self.txt_timeline.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum())
    # Update full text with speaker grouping
    self.txt_fulltext.setPlainText(self._build_full_text(self._live_segments))
```

- [ ] **Step 5: Add `_build_full_text` helper**

연속 같은 화자 세그먼트를 하나의 블록으로 합침:

```python
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
```

- [ ] **Step 6: Update `_on_select_item` to show speaker labels in timeline and full text**

```python
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
```

- [ ] **Step 7: Commit**

```bash
git add src/main_window.py
git commit -m "feat: add diarization toggle and speaker display in UI"
```

---

## Chunk 6: UI — Speaker Management Dialog

### Task 7: Add speaker management button and dialog

**Files:**
- Modify: `src/main_window.py`

- [ ] **Step 1: Add "화자 관리" button in `_build_ui`**

`btn_row` (복사 버튼 레이아웃) 부분에 추가:

```python
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
```

- [ ] **Step 2: Add `_show_detail` update**

`_show_detail()`에 `btn_speakers` 가시성 추가:

```python
def _show_detail(self, show: bool):
    self.lbl_title.setVisible(show)
    self.lbl_info.setVisible(show)
    self.tabs.setVisible(show)
    self.btn_copy.setVisible(show)
    self.btn_speakers.setVisible(False)  # 별도 제어
    self.lbl_empty.setVisible(not show)
```

- [ ] **Step 3: Implement `_on_manage_speakers` dialog**

```python
def _on_manage_speakers(self):
    if not hasattr(self, '_current_tid'):
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
        for old_name, edit in edits.items():
            new_name = edit.text().strip()
            if new_name and new_name != old_name:
                self.db.update_speaker_name(self._current_tid, old_name, new_name)
        # Refresh display
        self._on_select_item(self.list_widget.currentRow())
```

- [ ] **Step 4: Initialize `_current_tid` in `__init__`**

```python
def __init__(self, db: Database):
    super().__init__()
    self.db = db
    self._worker = None
    self._thread = None
    self._current_tid = None
```

- [ ] **Step 5: Update `_on_finished` to set `_current_tid` and show speaker button**

`_on_finished`에서 DB 저장 후:

```python
def _on_finished(self, result: dict):
    tid = self.db.add_transcription(
        filename=result["filename"],
        filepath=result["filepath"],
        duration=result["duration"],
        full_text=result["full_text"],
        segments=result["segments"],
    )
    self._current_tid = tid

    # Update info label
    dur = format_duration(result.get("duration"))
    self.lbl_info.setText(f"길이: {dur}")

    # 화자 정보 있으면 버튼 표시
    has_speakers = any(s.get("speaker") for s in result.get("segments", []))
    self.btn_speakers.setVisible(has_speakers)

    # 최종 라벨 매핑이 적용된 세그먼트로 UI 갱신
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

    self._load_list()
    self.list_widget.blockSignals(True)
    self.list_widget.setCurrentRow(0)
    self.list_widget.blockSignals(False)

    self.btn_add.setEnabled(True)
    self.progress_bar.setVisible(False)
    self.lbl_status.setVisible(False)

    elapsed_min = result.get("elapsed", 0) / 60
    QMessageBox.information(self, "완료", f"변환 완료! (소요시간: {elapsed_min:.1f}분)")
```

- [ ] **Step 6: Commit**

```bash
git add src/main_window.py
git commit -m "feat: add speaker management dialog for renaming speakers"
```

---

## Chunk 7: Build Configuration

### Task 8: Update PyInstaller spec

**Files:**
- Modify: `build/video_transcriber.spec`

- [ ] **Step 1: Add hidden imports for pyannote ecosystem**

`hiddenimports` 리스트에 추가:

```python
hiddenimports=[
    'whisper',
    'PySide6',
    'numpy',
    'imageio_ffmpeg',
    'pyannote.audio',
    'pyannote.audio.pipelines',
    'pyannote.core',
    'pyannote.pipeline',
    'speechbrain',
    'transformers',
    'torchaudio',
],
```

- [ ] **Step 2: Commit**

```bash
git add build/video_transcriber.spec
git commit -m "feat: add pyannote hidden imports to PyInstaller spec"
```

---

## Chunk 8: Final Verification

### Task 9: Run all tests and verify

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Manual smoke test**

1. 앱 실행: `python -m src.main`
2. 파일 추가 → 화자 분리 "아니오" → 기존과 동일하게 동작 확인
3. 파일 추가 → 화자 분리 "예" → 토큰 입력 → 화자 라벨이 타임라인에 표시되는지 확인
4. 화자 관리 버튼 → 이름 변경 → 타임라인/전체텍스트 갱신 확인
5. 텍스트 복사 → 변경된 이름이 반영되는지 확인
6. 기존 트랜스크립션(화자 없음) 선택 → 기존 형식으로 표시되는지 확인

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final verification for speaker diarization feature"
```
