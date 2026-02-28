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
