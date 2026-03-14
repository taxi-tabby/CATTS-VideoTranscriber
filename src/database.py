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
        self._migrate_add_speaker_column()

    def _migrate_add_speaker_column(self):
        columns = [
            row[1] for row in
            self._conn.execute("PRAGMA table_info(segments)").fetchall()
        ]
        if "speaker" not in columns:
            self._conn.execute("ALTER TABLE segments ADD COLUMN speaker TEXT")
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
                """INSERT INTO segments (transcription_id, start_time, end_time, text, speaker)
                   VALUES (?, ?, ?, ?, ?)""",
                (tid, seg["start"], seg["end"], seg["text"], seg.get("speaker")),
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
            "SELECT start_time as start, end_time as end, text, speaker FROM segments WHERE transcription_id = ? ORDER BY start_time",
            (tid,),
        ).fetchall()
        result["segments"] = [dict(s) for s in seg_rows]
        return result

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

    def delete_transcription(self, tid: int):
        self._conn.execute("DELETE FROM transcriptions WHERE id = ?", (tid,))
        self._conn.commit()

    def close(self):
        self._conn.close()
