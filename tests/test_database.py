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

    def test_segment_with_speaker(self, db):
        tid = db.add_transcription(
            "test.mp4", "C:/test.mp4", 60.0, "텍스트",
            [{"start": 0.0, "end": 5.0, "text": "세그먼트", "speaker": "화자 1"}],
        )
        result = db.get_transcription(tid)
        assert result["segments"][0]["speaker"] == "화자 1"

    def test_segment_without_speaker(self, db):
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


class TestCorrectionDict:
    def test_create_and_list(self, db):
        dict_id = db.create_correction_dict("my dict")
        dicts = db.list_correction_dicts()
        assert len(dicts) == 1
        assert dicts[0]["id"] == dict_id
        assert dicts[0]["name"] == "my dict"
        assert dicts[0]["entry_count"] == 0

    def test_add_and_get_entries(self, db):
        dict_id = db.create_correction_dict("dict1")
        eid1 = db.add_correction_entry(dict_id, "teh", "the")
        eid2 = db.add_correction_entry(dict_id, "wrold", "world")
        entries = db.get_correction_entries(dict_id)
        assert len(entries) == 2
        assert {e["wrong"] for e in entries} == {"teh", "wrold"}
        assert {e["correct"] for e in entries} == {"the", "world"}

    def test_delete_entry(self, db):
        dict_id = db.create_correction_dict("dict1")
        eid1 = db.add_correction_entry(dict_id, "aaa", "AAA")
        eid2 = db.add_correction_entry(dict_id, "bbb", "BBB")
        db.delete_correction_entry(eid1)
        entries = db.get_correction_entries(dict_id)
        assert len(entries) == 1
        assert entries[0]["wrong"] == "bbb"

    def test_delete_dict_cascades(self, db):
        dict_id = db.create_correction_dict("dict1")
        db.add_correction_entry(dict_id, "x", "y")
        db.add_correction_entry(dict_id, "a", "b")
        db.delete_correction_dict(dict_id)
        dicts = db.list_correction_dicts()
        assert len(dicts) == 0
        entries = db.get_correction_entries(dict_id)
        assert len(entries) == 0

    def test_rename_dict(self, db):
        dict_id = db.create_correction_dict("old name")
        db.rename_correction_dict(dict_id, "new name")
        dicts = db.list_correction_dicts()
        assert dicts[0]["name"] == "new name"

    def test_list_with_entry_count(self, db):
        d1 = db.create_correction_dict("alpha")
        d2 = db.create_correction_dict("beta")
        db.add_correction_entry(d1, "a", "A")
        db.add_correction_entry(d1, "b", "B")
        db.add_correction_entry(d1, "c", "C")
        db.add_correction_entry(d2, "x", "X")
        dicts = db.list_correction_dicts()
        counts = {d["name"]: d["entry_count"] for d in dicts}
        assert counts["alpha"] == 3
        assert counts["beta"] == 1
