"""Whisper 모델 관리 유틸리티.

번들된 모델(large-v3)의 캐시 배포, 설치 상태 확인 등을 담당한다.
"""

import os
import shutil
import sys

import whisper

from src.config import get_whisper_cache

# 모델별 실제 다운로드 파일 크기 (MB)
MODEL_SIZES: dict[str, int] = {
    "tiny": 72,
    "base": 139,
    "small": 461,
    "medium": 1457,
    "large-v1": 2944,
    "large-v2": 2944,
    "large-v3": 2944,
    "turbo": 1543,
    "large-v3-turbo": 1543,
}

BUNDLED_MODEL = "large-v3"


def get_whisper_cache_dir() -> str:
    """Whisper 모델 캐시 디렉터리를 반환한다."""
    return get_whisper_cache()


def _get_bundled_model_dir() -> str | None:
    """PyInstaller 번들에 포함된 모델 디렉터리를 반환한다."""
    if not getattr(sys, "frozen", False):
        return None
    return os.path.join(sys._MEIPASS, "whisper_models")


def _model_filename(model_name: str) -> str:
    """모델 이름에 대응하는 파일명을 반환한다."""
    url = whisper._MODELS.get(model_name, "")
    if url:
        return os.path.basename(url)
    return f"{model_name}.pt"


def _is_model_cached(fname: str) -> bool:
    """캐시에 유효한 모델 파일이 있는지 확인한다 (10MB 이상)."""
    cache_path = os.path.join(get_whisper_cache_dir(), fname)
    return os.path.exists(cache_path) and os.path.getsize(cache_path) > 10 * 1024 * 1024


def get_model_status(model_name: str) -> str:
    """모델의 설치 상태를 반환한다.

    Returns:
        "bundled"       - 프로그램 번들에 포함되어 있거나 캐시에 있는 기본 모델
        "installed"     - 캐시에 다운로드된 모델
        "not_installed" - 미설치
    """
    fname = _model_filename(model_name)

    # 캐시에 존재하는 경우
    if _is_model_cached(fname):
        if model_name == BUNDLED_MODEL:
            return "bundled"
        return "installed"

    # 캐시에 없지만 PyInstaller 번들에 포함된 경우
    if model_name == BUNDLED_MODEL:
        bundled_dir = _get_bundled_model_dir()
        if bundled_dir and os.path.exists(os.path.join(bundled_dir, fname)):
            return "bundled"

    return "not_installed"


def get_model_display_name(model_name: str) -> str:
    """콤보박스에 표시할 모델 이름 (상태 포함)."""
    status = get_model_status(model_name)
    size = MODEL_SIZES.get(model_name, 0)
    size_str = f"{size}MB" if size else ""

    if status == "bundled":
        return f"{model_name} ({size_str}, 기본 제공)"
    elif status == "installed":
        return f"{model_name} ({size_str}, 설치됨)"
    elif model_name == BUNDLED_MODEL:
        return f"{model_name} ({size_str}, 기본 제공 - 자동 다운로드)"
    else:
        return f"{model_name} ({size_str}, 미설치 - 자동 다운로드)"


def ensure_bundled_model() -> None:
    """번들된 모델이 캐시에 없으면 복사한다.

    PyInstaller 번들 실행 시에만 동작한다.
    """
    bundled_dir = _get_bundled_model_dir()
    if not bundled_dir:
        return

    fname = _model_filename(BUNDLED_MODEL)
    bundled_path = os.path.join(bundled_dir, fname)
    if not os.path.exists(bundled_path):
        return

    cache_dir = get_whisper_cache_dir()
    cache_path = os.path.join(cache_dir, fname)

    # 이미 유효한 캐시가 있으면 스킵
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 10 * 1024 * 1024:
        return

    os.makedirs(cache_dir, exist_ok=True)
    shutil.copy2(bundled_path, cache_path)
