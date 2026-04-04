"""Whisper 환각(hallucination) 탐지 및 제거 필터.

Whisper는 무음/잡음 구간에서 반복 텍스트, 잘못된 언어, 존재하지 않는 발화를
생성하는 경향이 있다. 이 모듈은 그런 환각 패턴을 탐지하고 제거한다.
"""

import re


def filter_hallucinations(
    segments: list[dict],
    language: str | None = None,
) -> list[dict]:
    """Whisper 전사 결과에서 환각 세그먼트를 제거한다.

    Args:
        segments: [{"start": float, "end": float, "text": str, ...}, ...]
            Whisper 세그먼트. no_speech_prob 키가 있을 수 있다.
        language: 지정된 언어 코드 (예: "ko"). None이면 언어 불일치 필터 생략.

    Returns:
        환각이 제거된 세그먼트 리스트. 원본을 수정하지 않는다.
    """
    result = list(segments)
    result = _filter_no_speech(result)
    result = _filter_tiny_segments(result)
    result = _filter_repetitions(result)
    if language and language != "auto":
        result = _filter_language_mismatch(result, language)
    return result


def _filter_no_speech(segments: list[dict]) -> list[dict]:
    """no_speech_prob이 높은 세그먼트를 제거한다."""
    return [s for s in segments if s.get("no_speech_prob", 0) <= 0.6]


def _filter_tiny_segments(segments: list[dict]) -> list[dict]:
    """0.1초 미만이면서 의미 없는 텍스트인 세그먼트를 제거한다."""
    result = []
    for s in segments:
        duration = s["end"] - s["start"]
        text = s.get("text", "").strip()
        if duration < 0.1 and (not text or re.match(r'^[\s\W]+$', text)):
            continue
        result.append(s)
    return result


def _filter_repetitions(segments: list[dict]) -> list[dict]:
    """동일 텍스트가 연속으로 반복되면 1회만 유지한다."""
    if not segments:
        return []

    result = []
    prev_text = None

    for s in segments:
        text = s.get("text", "").strip()
        if text and text == prev_text:
            continue
        prev_text = text
        result.append(s)

    return result


# 한글 범위: AC00-D7AF (완성형), 3131-318E (자모)
_HANGUL_RE = re.compile(r'[\uAC00-\uD7AF\u3131-\u318E]')
# 일본어 히라가나/카타카나
_JAPANESE_RE = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')
# 중국어 한자
_CHINESE_RE = re.compile(r'[\u4E00-\u9FFF]')

# 해당 언어의 문자가 포함되어야 하는 언어 목록
_SCRIPT_CHECKS = {
    "ko": _HANGUL_RE,
    "ja": _JAPANESE_RE,
    "zh": _CHINESE_RE,
}


def _filter_language_mismatch(segments: list[dict], language: str) -> list[dict]:
    """지정된 언어와 불일치하는 세그먼트를 제거한다.

    예: language="ko"인데 텍스트에 한글이 전혀 없고 라틴 문자만 있으면 환각으로 판단.
    영어 등 라틴 문자 언어이거나, _SCRIPT_CHECKS에 없는 언어는 필터링하지 않는다.
    """
    script_re = _SCRIPT_CHECKS.get(language)
    if script_re is None:
        return segments  # 라틴 문자 언어는 필터링 불가

    result = []
    for s in segments:
        text = s.get("text", "").strip()
        if not text:
            result.append(s)
            continue
        # 해당 언어의 문자가 하나라도 있으면 OK
        if script_re.search(text):
            result.append(s)
            continue
        # 숫자/기호만 있는 경우는 유지 (시간, 숫자 등)
        if re.match(r'^[\d\s\W]+$', text):
            result.append(s)
            continue
        # 해당 언어 문자 없고 라틴 문자만 → 환각으로 판단, 제거
    return result
