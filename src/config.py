import json
import os

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".video-transcriber", "config.json")


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_hf_token() -> str | None:
    return load_config().get("hf_token")


def set_hf_token(token: str) -> None:
    config = load_config()
    config["hf_token"] = token
    save_config(config)


def delete_hf_token() -> None:
    config = load_config()
    config.pop("hf_token", None)
    save_config(config)


def get_whisper_model() -> str:
    return load_config().get("whisper_model", "large-v3")


def set_whisper_model(model: str) -> None:
    config = load_config()
    config["whisper_model"] = model
    save_config(config)


def get_show_startup_guide() -> bool:
    return load_config().get("show_startup_guide", True)


def set_show_startup_guide(show: bool) -> None:
    config = load_config()
    config["show_startup_guide"] = show
    save_config(config)


def get_theme() -> str:
    return load_config().get("theme", "light")


def set_theme(theme: str) -> None:
    config = load_config()
    config["theme"] = theme
    save_config(config)


# ── 저장 경로 설정 ──

def _default_db_dir() -> str:
    app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return os.path.join(app_data, "VideoTranscriber")


def _default_whisper_cache() -> str:
    default = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")


def _default_hf_cache() -> str:
    return os.getenv(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    )


def get_db_dir() -> str:
    return load_config().get("db_dir", "") or _default_db_dir()


def set_db_dir(path: str) -> None:
    config = load_config()
    config["db_dir"] = path
    save_config(config)


def get_whisper_cache() -> str:
    return load_config().get("whisper_cache", "") or _default_whisper_cache()


def set_whisper_cache(path: str) -> None:
    config = load_config()
    config["whisper_cache"] = path
    save_config(config)


def get_hf_cache() -> str:
    return load_config().get("hf_cache", "") or _default_hf_cache()


def set_hf_cache(path: str) -> None:
    config = load_config()
    config["hf_cache"] = path
    save_config(config)
