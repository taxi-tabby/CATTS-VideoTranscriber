import hashlib
import sqlite3
from datetime import datetime


def compute_file_checksum(filepath: str) -> str:
    """파일의 SHA256 체크섬을 계산한다 (처음 10MB만 사용하여 빠르게)."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        # 대용량 파일도 빠르게 처리하기 위해 처음 10MB + 파일 크기를 해시
        data = f.read(10 * 1024 * 1024)
        h.update(data)
        f.seek(0, 2)
        h.update(str(f.tell()).encode())
    return h.hexdigest()

# 마이그레이션 목록: (버전, 설명, SQL 목록)
# 버전은 1부터 순차적으로 증가. 새 마이그레이션은 맨 끝에 추가.
MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (1, "segments.speaker 컬럼 추가", [
        "ALTER TABLE segments ADD COLUMN speaker TEXT",
    ]),
    (2, "transcriptions.model_name, language 컬럼 추가", [
        "ALTER TABLE transcriptions ADD COLUMN model_name TEXT",
        "ALTER TABLE transcriptions ADD COLUMN language TEXT",
    ]),
    (3, "transcriptions.display_name 컬럼 추가", [
        "ALTER TABLE transcriptions ADD COLUMN display_name TEXT",
        "UPDATE transcriptions SET display_name = filename WHERE display_name IS NULL",
    ]),
    (4, "folders 테이블 및 transcriptions.folder_id 추가", [
        """CREATE TABLE IF NOT EXISTS folders (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            parent_id INTEGER REFERENCES folders(id) ON DELETE CASCADE
        )""",
        "ALTER TABLE transcriptions ADD COLUMN folder_id INTEGER REFERENCES folders(id) ON DELETE SET NULL",
    ]),
    (5, "교정 사전 테이블 추가", [
        """CREATE TABLE IF NOT EXISTS correction_dicts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS correction_entries (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            dict_id INTEGER NOT NULL
                REFERENCES correction_dicts(id) ON DELETE CASCADE,
            wrong   TEXT NOT NULL,
            correct TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_correction_entries_dict_id ON correction_entries(dict_id)",
    ]),
    (6, "교정 사전 미디어 체크섬 및 항목 타임스탬프 추가", [
        "ALTER TABLE correction_dicts ADD COLUMN media_checksum TEXT",
        "ALTER TABLE correction_dicts ADD COLUMN media_filename TEXT",
        "ALTER TABLE correction_entries ADD COLUMN start_time REAL",
        "ALTER TABLE correction_entries ADD COLUMN end_time REAL",
        "ALTER TABLE correction_entries ADD COLUMN speaker TEXT",
        "ALTER TABLE correction_entries ADD COLUMN frequency INTEGER DEFAULT 1",
        "ALTER TABLE correction_entries ADD COLUMN is_corrected INTEGER DEFAULT 0",
    ]),
]

LATEST_VERSION = MIGRATIONS[-1][0] if MIGRATIONS else 0


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ── 스키마 초기화 & 마이그레이션 ──

    def _init_schema(self):
        """기본 테이블 생성 후 마이그레이션 실행."""
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
        self._run_migrations()

    def _get_version(self) -> int:
        return self._conn.execute("PRAGMA user_version").fetchone()[0]

    def _set_version(self, version: int):
        self._conn.execute(f"PRAGMA user_version = {int(version)}")

    def _detect_legacy_version(self) -> int:
        """user_version=0인 기존 DB의 실제 스키마 상태를 감지."""
        version = 0

        seg_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(segments)").fetchall()}
        if "speaker" in seg_cols:
            version = 1

        trans_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(transcriptions)").fetchall()}
        if "model_name" in trans_cols:
            version = 2
        if "display_name" in trans_cols:
            version = 3
        if "folder_id" in trans_cols:
            version = 4

        return version

    def _run_migrations(self):
        current = self._get_version()

        # 기존 DB: user_version=0이지만 이미 마이그레이션이 적용된 상태
        if current == 0 and MIGRATIONS:
            legacy = self._detect_legacy_version()
            if legacy > 0:
                self._set_version(legacy)
                self._conn.commit()
                current = legacy

        if current >= LATEST_VERSION:
            return

        for version, description, sqls in MIGRATIONS:
            if version <= current:
                continue
            for sql in sqls:
                self._conn.execute(sql)
            self._set_version(version)
            self._conn.commit()

    # ── 점진적 저장 (크래시 복구) ──

    def begin_transcription(
        self,
        filename: str,
        filepath: str,
        model_name: str | None = None,
        language: str | None = None,
        folder_id: int | None = None,
    ) -> int:
        """변환 시작 시 빈 레코드를 생성한다. 세그먼트는 이후 점진적으로 추가."""
        cur = self._conn.execute(
            """INSERT INTO transcriptions
               (filename, filepath, duration, created_at, full_text, model_name, language, display_name, folder_id)
               VALUES (?, ?, 0, ?, '', ?, ?, ?, ?)""",
            (filename, filepath, datetime.now().isoformat(), model_name, language, filename, folder_id),
        )
        self._conn.commit()
        return cur.lastrowid

    def add_segments_batch(self, tid: int, segments: list[dict]) -> None:
        """세그먼트를 일괄 삽입한다."""
        self._conn.executemany(
            "INSERT INTO segments (transcription_id, start_time, end_time, text, speaker) VALUES (?, ?, ?, ?, ?)",
            [(tid, s["start"], s["end"], s["text"], s.get("speaker")) for s in segments],
        )
        self._conn.commit()

    def finalize_transcription(self, tid: int, full_text: str, duration: float) -> None:
        """변환 완료 후 메타데이터를 갱신한다."""
        self._conn.execute(
            "UPDATE transcriptions SET full_text = ?, duration = ? WHERE id = ?",
            (full_text, duration, tid),
        )
        self._conn.commit()

    def remap_speakers(self, tid: int) -> None:
        """pyannote 라벨(SPEAKER_00)을 한글 라벨(화자 1)로 일괄 변환."""
        rows = self._conn.execute(
            "SELECT speaker, MIN(start_time) as first_ts FROM segments "
            "WHERE transcription_id = ? AND speaker IS NOT NULL "
            "GROUP BY speaker ORDER BY first_ts",
            (tid,),
        ).fetchall()
        if not rows:
            return
        label_map: dict[str, str] = {}
        counter = 0
        for row in rows:
            raw = row[0]
            if raw not in label_map:
                counter += 1
                label_map[raw] = f"화자 {counter}"
        for raw, mapped in label_map.items():
            if raw != mapped:
                self._conn.execute(
                    "UPDATE segments SET speaker = ? WHERE transcription_id = ? AND speaker = ?",
                    (mapped, tid, raw),
                )
        self._conn.commit()

    def delete_empty_transcription(self, tid: int) -> None:
        """세그먼트가 없는 레코드를 삭제한다."""
        count = self._conn.execute(
            "SELECT COUNT(*) FROM segments WHERE transcription_id = ?", (tid,)
        ).fetchone()[0]
        if count == 0:
            self._conn.execute("DELETE FROM transcriptions WHERE id = ?", (tid,))
            self._conn.commit()

    def get_incomplete_transcriptions(self) -> list[dict]:
        """미완료(duration=0) 레코드를 반환한다. 각 항목에 last_end(마지막 세그먼트 종료 시각)를 포함."""
        rows = self._conn.execute(
            "SELECT t.id, t.filename, t.filepath, t.model_name, t.language, t.folder_id, "
            "COALESCE(MAX(s.end_time), 0) as last_end, "
            "COUNT(s.id) as seg_count "
            "FROM transcriptions t LEFT JOIN segments s ON s.transcription_id = t.id "
            "WHERE t.duration = 0 GROUP BY t.id"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── 트랜스크립션 ──

    def add_transcription(
        self,
        filename: str,
        filepath: str,
        duration: float,
        full_text: str,
        segments: list[dict],
        model_name: str | None = None,
        language: str | None = None,
        folder_id: int | None = None,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO transcriptions (filename, filepath, duration, created_at, full_text, model_name, language, display_name, folder_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (filename, filepath, duration, datetime.now().isoformat(), full_text, model_name, language, filename, folder_id),
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

    def get_all_transcriptions(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, filename, filepath, duration, created_at, model_name, language, display_name, folder_id FROM transcriptions ORDER BY created_at DESC"
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
            "SELECT id, start_time as start, end_time as end, text, speaker FROM segments WHERE transcription_id = ? ORDER BY start_time",
            (tid,),
        ).fetchall()
        result["segments"] = [dict(s) for s in seg_rows]
        return result

    def rename_transcription(self, tid: int, new_name: str) -> None:
        self._conn.execute(
            "UPDATE transcriptions SET display_name = ? WHERE id = ?",
            (new_name, tid),
        )
        self._conn.commit()

    def move_transcription(self, tid: int, folder_id: int | None) -> None:
        self._conn.execute(
            "UPDATE transcriptions SET folder_id = ? WHERE id = ?",
            (folder_id, tid),
        )
        self._conn.commit()

    def delete_transcription(self, tid: int):
        self._conn.execute("DELETE FROM transcriptions WHERE id = ?", (tid,))
        self._conn.commit()

    # ── 폴더 ──

    def create_folder(self, name: str, parent_id: int | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO folders (name, parent_id) VALUES (?, ?)",
            (name, parent_id),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_all_folders(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, name, parent_id FROM folders ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]

    def rename_folder(self, folder_id: int, new_name: str) -> None:
        self._conn.execute(
            "UPDATE folders SET name = ? WHERE id = ?",
            (new_name, folder_id),
        )
        self._conn.commit()

    def move_folder(self, folder_id: int, new_parent_id: int | None) -> None:
        self._conn.execute(
            "UPDATE folders SET parent_id = ? WHERE id = ?",
            (new_parent_id, folder_id),
        )
        self._conn.commit()

    def delete_folder(self, folder_id: int) -> None:
        # 하위 항목들을 상위 폴더로 이동
        parent_id = self._conn.execute(
            "SELECT parent_id FROM folders WHERE id = ?", (folder_id,)
        ).fetchone()
        parent = parent_id[0] if parent_id else None
        self._conn.execute(
            "UPDATE transcriptions SET folder_id = ? WHERE folder_id = ?",
            (parent, folder_id),
        )
        self._conn.execute(
            "UPDATE folders SET parent_id = ? WHERE parent_id = ?",
            (parent, folder_id),
        )
        self._conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
        self._conn.commit()

    # ── 화자 ──

    def get_speakers(self, tid: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT speaker FROM segments WHERE transcription_id = ? AND speaker IS NOT NULL ORDER BY speaker",
            (tid,),
        ).fetchall()
        return [row[0] for row in rows]

    def update_segment_text(self, segment_id: int, new_text: str) -> None:
        self._conn.execute(
            "UPDATE segments SET text = ? WHERE id = ?",
            (new_text, segment_id),
        )
        self._conn.commit()

    def update_full_text(self, tid: int, full_text: str) -> None:
        self._conn.execute(
            "UPDATE transcriptions SET full_text = ? WHERE id = ?",
            (full_text, tid),
        )
        self._conn.commit()

    def update_speaker_name(self, tid: int, old_name: str, new_name: str) -> None:
        self._conn.execute(
            "UPDATE segments SET speaker = ? WHERE transcription_id = ? AND speaker = ?",
            (new_name, tid, old_name),
        )
        self._conn.commit()

    # ── 교정 사전 ──

    def create_correction_dict(
        self, name: str,
        media_checksum: str | None = None,
        media_filename: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO correction_dicts (name, created_at, media_checksum, media_filename) VALUES (?, ?, ?, ?)",
            (name, datetime.now().isoformat(), media_checksum, media_filename),
        )
        self._conn.commit()
        return cur.lastrowid

    def list_correction_dicts(self) -> list[dict]:
        rows = self._conn.execute(
            """SELECT d.id, d.name, d.created_at, d.media_checksum, d.media_filename,
                      COUNT(e.id) as entry_count
               FROM correction_dicts d
               LEFT JOIN correction_entries e ON e.dict_id = d.id
               GROUP BY d.id
               ORDER BY d.name""",
        ).fetchall()
        return [dict(r) for r in rows]

    def get_correction_dict(self, dict_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, name, created_at, media_checksum, media_filename FROM correction_dicts WHERE id = ?",
            (dict_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_correction_dict_checksum(self, dict_id: int, checksum: str) -> None:
        self._conn.execute(
            "UPDATE correction_dicts SET media_checksum = ? WHERE id = ?",
            (checksum, dict_id),
        )
        self._conn.commit()

    def delete_correction_dict(self, dict_id: int) -> None:
        self._conn.execute("DELETE FROM correction_dicts WHERE id = ?", (dict_id,))
        self._conn.commit()

    def rename_correction_dict(self, dict_id: int, name: str) -> None:
        self._conn.execute(
            "UPDATE correction_dicts SET name = ? WHERE id = ?", (name, dict_id),
        )
        self._conn.commit()

    def add_correction_entry(
        self, dict_id: int, wrong: str, correct: str,
        start_time: float | None = None, end_time: float | None = None,
        speaker: str | None = None, frequency: int = 1,
        is_corrected: bool = False,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO correction_entries
               (dict_id, wrong, correct, start_time, end_time, speaker, frequency, is_corrected)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (dict_id, wrong, correct, start_time, end_time, speaker, frequency, int(is_corrected)),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_correction_entries(self, dict_id: int) -> list[dict]:
        rows = self._conn.execute(
            """SELECT id, wrong, correct, start_time, end_time, speaker, frequency, is_corrected
               FROM correction_entries WHERE dict_id = ?
               ORDER BY frequency DESC, wrong""",
            (dict_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_correction_entries_for_timerange(
        self, dict_id: int, start: float, end: float,
    ) -> list[dict]:
        """시간 범위에 겹치는 교정 항목만 반환한다."""
        rows = self._conn.execute(
            """SELECT id, wrong, correct, start_time, end_time, speaker, frequency, is_corrected
               FROM correction_entries
               WHERE dict_id = ? AND is_corrected = 1
                 AND start_time IS NOT NULL AND end_time IS NOT NULL
                 AND start_time < ? AND end_time > ?
               ORDER BY frequency DESC""",
            (dict_id, end, start),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_correction_entry(self, entry_id: int, wrong: str, correct: str) -> None:
        self._conn.execute(
            "UPDATE correction_entries SET wrong = ?, correct = ? WHERE id = ?",
            (wrong, correct, entry_id),
        )
        self._conn.commit()

    def delete_correction_entry(self, entry_id: int) -> None:
        self._conn.execute("DELETE FROM correction_entries WHERE id = ?", (entry_id,))
        self._conn.commit()

    def replace_correction_entries(self, dict_id: int, entries: list[dict]) -> None:
        """사전의 모든 항목을 원자적으로 교체한다 (트랜잭션)."""
        self._conn.execute("DELETE FROM correction_entries WHERE dict_id = ?", (dict_id,))
        self._conn.executemany(
            """INSERT INTO correction_entries
               (dict_id, wrong, correct, start_time, end_time, speaker, frequency, is_corrected)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (dict_id, e["wrong"], e["correct"],
                 e.get("start_time"), e.get("end_time"),
                 e.get("speaker"), e.get("frequency", 1),
                 int(e.get("is_corrected", False)))
                for e in entries if e.get("wrong") and e.get("correct")
            ],
        )
        self._conn.commit()

    def close(self):
        self._conn.close()
