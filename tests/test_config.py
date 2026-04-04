import pytest
from src.config import (
    load_config, save_config,
    get_hf_token, set_hf_token, delete_hf_token,
    get_whisper_model, set_whisper_model,
    get_show_startup_guide, set_show_startup_guide,
    get_theme, set_theme,
    get_db_dir, set_db_dir,
    get_whisper_cache, set_whisper_cache,
    get_hf_cache, set_hf_cache,
    get_thread_config, set_thread_config,
    _default_db_dir, _default_whisper_cache, _default_hf_cache,
)


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    config_path = str(tmp_path / ".video-transcriber" / "config.json")
    monkeypatch.setattr("src.config.CONFIG_PATH", config_path)
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

    def test_delete_hf_token(self, config_dir):
        set_hf_token("hf_abc123")
        delete_hf_token()
        assert get_hf_token() is None

    def test_save_creates_directory(self, config_dir):
        save_config({"key": "value"})
        import os
        assert os.path.exists(config_dir)

    def test_get_whisper_model_default(self, config_dir):
        assert get_whisper_model() == "large-v3"

    def test_set_and_get_whisper_model(self, config_dir):
        set_whisper_model("large-v3")
        assert get_whisper_model() == "large-v3"

    def test_startup_guide_default(self, config_dir):
        assert get_show_startup_guide() is True

    def test_set_startup_guide(self, config_dir):
        set_show_startup_guide(False)
        assert get_show_startup_guide() is False

    def test_theme_default(self, config_dir):
        assert get_theme() == "light"

    def test_set_theme(self, config_dir):
        set_theme("dark")
        assert get_theme() == "dark"

    def test_db_dir_default(self, config_dir):
        result = get_db_dir()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_set_db_dir(self, config_dir):
        set_db_dir("/tmp/test")
        assert get_db_dir() == "/tmp/test"

    def test_whisper_cache_default(self, config_dir):
        result = get_whisper_cache()
        assert isinstance(result, str)

    def test_set_whisper_cache(self, config_dir):
        set_whisper_cache("/tmp/whisper")
        assert get_whisper_cache() == "/tmp/whisper"

    def test_hf_cache_default(self, config_dir):
        result = get_hf_cache()
        assert isinstance(result, str)

    def test_set_hf_cache(self, config_dir):
        set_hf_cache("/tmp/hf")
        assert get_hf_cache() == "/tmp/hf"

    def test_thread_config_default(self, config_dir):
        tc = get_thread_config()
        assert tc["whisper_mode"] == "single"
        assert tc["diar_mode"] == "single"
        assert isinstance(tc["whisper_min"], int)

    def test_set_thread_config(self, config_dir):
        tc = {
            "whisper_mode": "multi",
            "whisper_min": 4,
            "whisper_max": 8,
            "diar_mode": "multi",
            "diar_min": 2,
            "diar_max": 4,
        }
        set_thread_config(tc)
        loaded = get_thread_config()
        assert loaded["whisper_mode"] == "multi"
        assert loaded["whisper_max"] == 8
        assert loaded["diar_mode"] == "multi"

    def test_default_dirs_return_strings(self, config_dir):
        assert isinstance(_default_db_dir(), str)
        assert isinstance(_default_whisper_cache(), str)
        assert isinstance(_default_hf_cache(), str)
