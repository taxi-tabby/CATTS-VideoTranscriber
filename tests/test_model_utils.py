"""model_utils.py 테스트."""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestModelSizes:
    def test_model_sizes_defined(self):
        from src.model_utils import MODEL_SIZES
        assert isinstance(MODEL_SIZES, dict)
        assert "medium" in MODEL_SIZES
        assert "large-v3" in MODEL_SIZES
        assert all(isinstance(v, int) for v in MODEL_SIZES.values())


class TestGetWhisperCacheDir:
    def test_returns_string(self):
        from src.model_utils import get_whisper_cache_dir
        result = get_whisper_cache_dir()
        assert isinstance(result, str)
        assert len(result) > 0


class TestModelFilename:
    def test_known_model(self):
        from src.model_utils import _model_filename
        fname = _model_filename("medium")
        assert fname.endswith(".pt")

    def test_unknown_model(self):
        from src.model_utils import _model_filename
        fname = _model_filename("nonexistent-model")
        assert fname == "nonexistent-model.pt"


class TestGetModelStatus:
    def test_returns_valid_status(self):
        from src.model_utils import get_model_status
        status = get_model_status("medium")
        assert status in ("bundled", "installed", "not_installed")

    def test_not_frozen_no_bundled_dir(self):
        from src.model_utils import _get_bundled_model_dir
        # 일반 실행 환경에서는 None
        assert _get_bundled_model_dir() is None


class TestGetModelDisplayName:
    def test_returns_string_with_model_name(self):
        from src.model_utils import get_model_display_name
        display = get_model_display_name("medium")
        assert "medium" in display

    def test_shows_size(self):
        from src.model_utils import get_model_display_name
        display = get_model_display_name("medium")
        assert "1457MB" in display


class TestEnsureBundledModel:
    def test_noop_when_not_frozen(self):
        from src.model_utils import ensure_bundled_model
        # 일반 환경에서는 아무 것도 하지 않아야 함
        ensure_bundled_model()  # should not raise
