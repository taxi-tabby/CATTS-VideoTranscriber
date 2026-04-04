import os
import pytest
from src.database import Database, compute_file_checksum


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


class TestComputeFileChecksum:
    def test_returns_hex_string(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        result = compute_file_checksum(str(f))
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex

    def test_same_file_same_checksum(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"content" * 1000)
        assert compute_file_checksum(str(f)) == compute_file_checksum(str(f))

    def test_different_content_different_checksum(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"aaa")
        f2.write_bytes(b"bbb")
        assert compute_file_checksum(str(f1)) != compute_file_checksum(str(f2))


class TestProgressiveSave:
    def test_begin_and_finalize(self, db):
        tid = db.begin_transcription("test.mp4", "/path/test.mp4", "medium", "ko")
        assert tid > 0
        db.add_segments_batch(tid, [
            {"start": 0.0, "end": 5.0, "text": "hello", "speaker": None},
        ])
        db.finalize_transcription(tid, "hello", 10.0)
        result = db.get_transcription(tid)
        assert result["full_text"] == "hello"
        assert result["duration"] == 10.0

    def test_incomplete_transcription(self, db):
        tid = db.begin_transcription("inc.mp4", "/path/inc.mp4")
        db.add_segments_batch(tid, [
            {"start": 0.0, "end": 1.0, "text": "seg1", "speaker": None},
        ])
        incompletes = db.get_incomplete_transcriptions()
        assert any(i["id"] == tid for i in incompletes)

    def test_delete_empty_transcription(self, db):
        tid = db.begin_transcription("empty.mp4", "/path/empty.mp4")
        db.delete_empty_transcription(tid)
        assert db.get_transcription(tid) is None

    def test_remap_speakers(self, db):
        tid = db.begin_transcription("sp.mp4", "/path/sp.mp4")
        db.add_segments_batch(tid, [
            {"start": 0.0, "end": 1.0, "text": "a", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "b", "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 3.0, "text": "c", "speaker": "SPEAKER_00"},
        ])
        db.remap_speakers(tid)
        result = db.get_transcription(tid)
        speakers = [s["speaker"] for s in result["segments"]]
        assert speakers[0] == "화자 1"
        assert speakers[1] == "화자 2"
        assert speakers[2] == "화자 1"


class TestTranscriptionEditing:
    def test_rename_transcription(self, db):
        tid = db.add_transcription("old.mp4", "/old.mp4", 10.0, "text", [])
        db.rename_transcription(tid, "new name")
        result = db.get_transcription(tid)
        assert result["display_name"] == "new name"

    def test_move_transcription(self, db):
        tid = db.add_transcription("m.mp4", "/m.mp4", 10.0, "text", [])
        fid = db.create_folder("folder1")
        db.move_transcription(tid, fid)
        result = db.get_transcription(tid)
        assert result["folder_id"] == fid

    def test_update_segment_text(self, db):
        tid = db.add_transcription("t.mp4", "/t.mp4", 10.0, "text", [
            {"start": 0.0, "end": 5.0, "text": "original"},
        ])
        result = db.get_transcription(tid)
        seg_id = result["segments"][0]["id"]
        db.update_segment_text(seg_id, "updated")
        result2 = db.get_transcription(tid)
        assert result2["segments"][0]["text"] == "updated"

    def test_update_full_text(self, db):
        tid = db.add_transcription("t.mp4", "/t.mp4", 10.0, "old text", [])
        db.update_full_text(tid, "new text")
        result = db.get_transcription(tid)
        assert result["full_text"] == "new text"


class TestFolders:
    def test_create_and_list(self, db):
        fid = db.create_folder("test folder")
        folders = db.get_all_folders()
        assert any(f["id"] == fid and f["name"] == "test folder" for f in folders)

    def test_create_subfolder(self, db):
        parent = db.create_folder("parent")
        child = db.create_folder("child", parent_id=parent)
        folders = db.get_all_folders()
        child_row = next(f for f in folders if f["id"] == child)
        assert child_row["parent_id"] == parent

    def test_rename_folder(self, db):
        fid = db.create_folder("old")
        db.rename_folder(fid, "new")
        folders = db.get_all_folders()
        assert any(f["id"] == fid and f["name"] == "new" for f in folders)

    def test_move_folder(self, db):
        f1 = db.create_folder("f1")
        f2 = db.create_folder("f2")
        db.move_folder(f2, f1)
        folders = db.get_all_folders()
        moved = next(f for f in folders if f["id"] == f2)
        assert moved["parent_id"] == f1

    def test_delete_folder(self, db):
        fid = db.create_folder("to delete")
        db.delete_folder(fid)
        folders = db.get_all_folders()
        assert not any(f["id"] == fid for f in folders)


class TestCorrectionDictExtended:
    def test_create_with_checksum(self, db):
        dict_id = db.create_correction_dict("media dict",
                                             media_checksum="abc123",
                                             media_filename="video.mp4")
        d = db.get_correction_dict(dict_id)
        assert d["media_checksum"] == "abc123"
        assert d["media_filename"] == "video.mp4"

    def test_update_checksum(self, db):
        dict_id = db.create_correction_dict("dict", media_checksum="old")
        db.update_correction_dict_checksum(dict_id, "new_checksum")
        d = db.get_correction_dict(dict_id)
        assert d["media_checksum"] == "new_checksum"

    def test_get_nonexistent_dict(self, db):
        assert db.get_correction_dict(99999) is None

    def test_add_entry_with_timestamps(self, db):
        dict_id = db.create_correction_dict("ts dict")
        eid = db.add_correction_entry(dict_id, "wrong", "right",
                                       start_time=1.5, end_time=2.0,
                                       speaker="화자 1", frequency=5,
                                       is_corrected=True)
        entries = db.get_correction_entries(dict_id)
        assert len(entries) == 1
        e = entries[0]
        assert e["start_time"] == 1.5
        assert e["end_time"] == 2.0
        assert e["speaker"] == "화자 1"
        assert e["frequency"] == 5
        assert e["is_corrected"] == 1

    def test_get_entries_for_timerange(self, db):
        dict_id = db.create_correction_dict("range dict")
        db.add_correction_entry(dict_id, "a", "A",
                                 start_time=1.0, end_time=2.0,
                                 is_corrected=True)
        db.add_correction_entry(dict_id, "b", "B",
                                 start_time=5.0, end_time=6.0,
                                 is_corrected=True)
        db.add_correction_entry(dict_id, "c", "C",
                                 start_time=10.0, end_time=11.0,
                                 is_corrected=True)

        # 0~3초 범위 → "a"만 매칭
        result = db.get_correction_entries_for_timerange(dict_id, 0.0, 3.0)
        assert len(result) == 1
        assert result[0]["wrong"] == "a"

        # 4~7초 범위 → "b"만 매칭
        result = db.get_correction_entries_for_timerange(dict_id, 4.0, 7.0)
        assert len(result) == 1
        assert result[0]["wrong"] == "b"

        # 0~12초 범위 → 전부 매칭
        result = db.get_correction_entries_for_timerange(dict_id, 0.0, 12.0)
        assert len(result) == 3

    def test_timerange_excludes_uncorrected(self, db):
        dict_id = db.create_correction_dict("filter dict")
        db.add_correction_entry(dict_id, "x", "X",
                                 start_time=1.0, end_time=2.0,
                                 is_corrected=False)
        result = db.get_correction_entries_for_timerange(dict_id, 0.0, 5.0)
        assert len(result) == 0

    def test_update_correction_entry(self, db):
        dict_id = db.create_correction_dict("upd dict")
        eid = db.add_correction_entry(dict_id, "old_wrong", "old_correct")
        db.update_correction_entry(eid, "new_wrong", "new_correct")
        entries = db.get_correction_entries(dict_id)
        assert entries[0]["wrong"] == "new_wrong"
        assert entries[0]["correct"] == "new_correct"

    def test_replace_correction_entries(self, db):
        dict_id = db.create_correction_dict("replace dict")
        db.add_correction_entry(dict_id, "a", "A")
        db.add_correction_entry(dict_id, "b", "B")

        # 전부 교체
        db.replace_correction_entries(dict_id, [
            {"wrong": "x", "correct": "X", "start_time": 1.0, "end_time": 2.0,
             "speaker": "spk", "frequency": 3, "is_corrected": True},
            {"wrong": "y", "correct": "Y"},
        ])
        entries = db.get_correction_entries(dict_id)
        assert len(entries) == 2
        wrongs = {e["wrong"] for e in entries}
        assert wrongs == {"x", "y"}

    def test_replace_skips_empty(self, db):
        dict_id = db.create_correction_dict("skip dict")
        db.replace_correction_entries(dict_id, [
            {"wrong": "a", "correct": "A"},
            {"wrong": "", "correct": "B"},  # wrong 비어있음 → 스킵
            {"wrong": "c", "correct": ""},  # correct 비어있음 → 스킵
        ])
        entries = db.get_correction_entries(dict_id)
        assert len(entries) == 1
        assert entries[0]["wrong"] == "a"

    def test_list_includes_media_info(self, db):
        db.create_correction_dict("no media")
        db.create_correction_dict("with media",
                                   media_checksum="hash123",
                                   media_filename="file.mp4")
        dicts = db.list_correction_dicts()
        info = {d["name"]: d for d in dicts}
        assert info["no media"]["media_checksum"] is None
        assert info["with media"]["media_checksum"] == "hash123"
        assert info["with media"]["media_filename"] == "file.mp4"
