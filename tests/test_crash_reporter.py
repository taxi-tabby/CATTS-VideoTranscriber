"""crash_reporter.py 테스트."""

import pytest


class TestCollectSystemInfo:
    def test_returns_dict(self):
        from src.crash_reporter import collect_system_info
        info = collect_system_info()
        assert isinstance(info, dict)

    def test_contains_required_keys(self):
        from src.crash_reporter import collect_system_info
        info = collect_system_info()
        assert "os" in info
        assert "python" in info
        assert "app_version" in info

    def test_os_is_string(self):
        from src.crash_reporter import collect_system_info
        info = collect_system_info()
        assert isinstance(info["os"], str)
        assert len(info["os"]) > 0


class TestFormatSystemInfo:
    def test_returns_markdown_table(self):
        from src.crash_reporter import format_system_info
        info = {"os": "Windows 11", "python": "3.11.9"}
        result = format_system_info(info)
        assert "| Item | Value |" in result
        assert "Windows 11" in result

    def test_skips_missing_keys(self):
        from src.crash_reporter import format_system_info
        result = format_system_info({})
        assert "| Item | Value |" in result
        lines = result.strip().splitlines()
        assert len(lines) == 2  # header + separator only


class TestBuildIssueBody:
    def test_contains_error_message(self):
        from src.crash_reporter import build_issue_body
        body = build_issue_body(
            error_message="test error",
            include_system=False,
        )
        assert "test error" in body

    def test_contains_processing_log(self):
        from src.crash_reporter import build_issue_body
        body = build_issue_body(
            error_message="err",
            processing_log="log line 1\nlog line 2",
            include_system=False,
        )
        assert "log line 1" in body

    def test_truncates_long_log(self):
        from src.crash_reporter import build_issue_body
        long_log = "\n".join(f"line {i}" for i in range(100))
        body = build_issue_body(
            error_message="err",
            processing_log=long_log,
            include_system=False,
        )
        assert "truncated" in body

    def test_includes_traceback(self):
        from src.crash_reporter import build_issue_body
        body = build_issue_body(
            error_message="err",
            error_traceback="File test.py, line 1",
            include_system=False,
        )
        assert "File test.py" in body


class TestBuildIssueTitle:
    def test_prefixes_with_bug(self):
        from src.crash_reporter import build_issue_title
        title = build_issue_title("something broke")
        assert title.startswith("[Bug]")
        assert "something broke" in title

    def test_truncates_long_message(self):
        from src.crash_reporter import build_issue_title
        long_msg = "x" * 200
        title = build_issue_title(long_msg)
        assert len(title) <= 90  # [Bug] + 80 + buffer

    def test_uses_first_line_only(self):
        from src.crash_reporter import build_issue_title
        title = build_issue_title("first line\nsecond line")
        assert "second line" not in title
        assert "first line" in title
