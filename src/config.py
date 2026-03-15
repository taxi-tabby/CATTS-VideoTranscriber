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
    return load_config().get("whisper_model", "medium")


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
