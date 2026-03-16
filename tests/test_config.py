import pytest
from src.config import load_config, save_config, get_hf_token, set_hf_token, get_whisper_model, set_whisper_model


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

    def test_save_creates_directory(self, config_dir):
        save_config({"key": "value"})
        import os
        assert os.path.exists(config_dir)

    def test_get_whisper_model_default(self, config_dir):
        assert get_whisper_model() == "medium"

    def test_set_and_get_whisper_model(self, config_dir):
        set_whisper_model("large-v3")
        assert get_whisper_model() == "large-v3"
