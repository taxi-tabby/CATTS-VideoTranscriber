import sqlite3
from datetime import datetime


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
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
            CREATE TABLE IF NOT EXISTS folders (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT NOT NULL,
                parent_id INTEGER REFERENCES folders(id) ON DELETE CASCADE
            );
        """)
        self._conn.commit()
        self._run_migrations()

    def _run_migrations(self):
        seg_columns = [
            row[1] for row in
            self._conn.execute("PRAGMA table_info(segments)").fetchall()
        ]
        if "speaker" not in seg_columns:
            self._conn.execute("ALTER TABLE segments ADD COLUMN speaker TEXT")
            self._conn.commit()

        trans_columns = [
            row[1] for row in
            self._conn.execute("PRAGMA table_info(transcriptions)").fetchall()
        ]
        if "model_name" not in trans_columns:
            self._conn.execute("ALTER TABLE transcriptions ADD COLUMN model_name TEXT")
            self._conn.execute("ALTER TABLE transcriptions ADD COLUMN language TEXT")
            self._conn.commit()

        if "display_name" not in trans_columns:
            self._conn.execute("ALTER TABLE transcriptions ADD COLUMN display_name TEXT")
            self._conn.execute("UPDATE transcriptions SET display_name = filename WHERE display_name IS NULL")
            self._conn.commit()

        if "folder_id" not in trans_columns:
            self._conn.execute("ALTER TABLE transcriptions ADD COLUMN folder_id INTEGER REFERENCES folders(id) ON DELETE SET NULL")
            self._conn.commit()

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
            "SELECT start_time as start, end_time as end, text, speaker FROM segments WHERE transcription_id = ? ORDER BY start_time",
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

    def update_speaker_name(self, tid: int, old_name: str, new_name: str) -> None:
        self._conn.execute(
            "UPDATE segments SET speaker = ? WHERE transcription_id = ? AND speaker = ?",
            (new_name, tid, old_name),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()
