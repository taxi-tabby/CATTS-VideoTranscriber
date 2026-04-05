"""교정 사전 분석 엔진.

미디어 파일을 Whisper로 분석하여 화자별 단어 빈도 테이블을 생성한다.
결과는 DB의 correction_entries에 저장된다.
"""

import multiprocessing as mp
import os
import tempfile

from src.transcriber import (
    extract_audio,
    get_ffmpeg_exe,
    load_wav_as_numpy,
    save_numpy_as_wav,
    get_video_duration,
    SAMPLE_RATE,
)


def _analyze_worker(params: dict, msg_queue: mp.Queue, cancel_event: mp.Event) -> None:
    """별도 프로세스에서 미디어를 분석하여 단어 목록을 추출한다.

    결과 메시지:
        ("log", message)
        ("progress", percent, message)
        ("result", {"words": [{"word", "start", "end", "speaker", "frequency"}, ...]})
        ("error", message)
    """
    import time
    import datetime

    def _log(msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        msg_queue.put(("log", f"[{ts}] {msg}"))

    def _progress(pct, msg):
        msg_queue.put(("progress", pct, msg))

    def _is_cancelled():
        return cancel_event.is_set()

    try:
        wav_path = params["wav_path"]
        model_name = params.get("model_name", "medium")
        language = params.get("language", "ko")
        use_diar = params.get("use_diar", False)
        hf_token = params.get("hf_token")
        profile = params.get("profile", "interview")

        # ── 오디오 로드 + 전처리 ──
        _progress(5, "오디오 로드 중...")
        audio = load_wav_as_numpy(wav_path)
        duration = get_video_duration(audio)
        _log(f"오디오 길이: {duration:.1f}초")

        if _is_cancelled():
            return

        _progress(10, "전처리 중...")
        from src.audio_preprocess import preprocess as preprocess_audio
        use_vocal_sep = profile == "noisy"
        audio, trim_offset_samples = preprocess_audio(
            audio,
            use_vocal_separation=use_vocal_sep,
            log_callback=_log,
        )
        trim_offset_sec = trim_offset_samples / SAMPLE_RATE

        if _is_cancelled():
            return

        # ── 화자 분리 (선택) ──
        diarization_segments = None
        if use_diar and hf_token:
            _progress(20, "화자 분석 중...")
            from src.diarizer import run_diarization
            tmp_clean = os.path.join(tempfile.gettempdir(), f"vt_dict_clean_{os.getpid()}.wav")
            save_numpy_as_wav(audio, tmp_clean)
            diarization_segments = run_diarization(
                tmp_clean, hf_token,
                cancel_check=_is_cancelled,
                log_callback=_log,
                profile_name=profile,
            )
            if trim_offset_sec > 0:
                for dseg in diarization_segments:
                    dseg["start"] += trim_offset_sec
                    dseg["end"] += trim_offset_sec
            _log(f"화자 분석 완료: {len(diarization_segments)}개 구간")
            try:
                os.remove(tmp_clean)
            except OSError:
                pass

        if _is_cancelled():
            return

        # ── VAD 기반 청크 분할 (취소·진행률 지원) ──
        from src.audio_preprocess import get_speech_segments, build_vad_chunks
        speech_segments = get_speech_segments(audio)
        _log(f"VAD 음성 구간: {len(speech_segments)}개")

        if _is_cancelled():
            return

        # ── Whisper 전사 (word_timestamps=True) ──
        _progress(40, "Whisper 모델 로드 중...")
        import whisper as _whisper
        from src.model_utils import get_whisper_cache_dir
        model = _whisper.load_model(model_name, download_root=get_whisper_cache_dir())

        if _is_cancelled():
            return

        lang_arg = language if language != "auto" else None
        total_samples = len(audio)
        vad_chunks = build_vad_chunks(speech_segments, total_samples)
        single_chunks = [(vc["start_sample"], vc["end_sample"]) for vc in vad_chunks]
        total_chunks = len(single_chunks)

        raw_words = []
        if total_chunks == 0:
            _log("음성 구간이 없습니다.")
        else:
            _log(f"단어 단위 분석 시작 (총 {total_chunks}개 청크)")
            audio_duration = total_samples / SAMPLE_RATE
            start_time = time.time()
            for chunk_idx, (chunk_start, chunk_end) in enumerate(single_chunks):
                if _is_cancelled():
                    _log(f"취소됨 — {chunk_idx}/{total_chunks} 청크 완료분 보존")
                    return
                chunk = audio[chunk_start:chunk_end]
                time_offset = chunk_start / SAMPLE_RATE
                processed_sec = min(chunk_end / SAMPLE_RATE, audio_duration)
                pct = 50 + int((processed_sec / max(audio_duration, 1e-6)) * 30)
                pct = min(pct, 80)

                elapsed = time.time() - start_time
                if processed_sec > 0 and elapsed > 1:
                    eta = elapsed * (audio_duration - processed_sec) / processed_sec
                    eta_min, eta_sec = divmod(int(eta), 60)
                    eta_str = f" (남은 시간: {eta_min}분 {eta_sec}초)" if eta_min > 0 else f" (남은 시간: {eta_sec}초)"
                else:
                    eta_str = ""
                _progress(pct, f"단어 단위 분석 중... {processed_sec:.0f}s / {audio_duration:.0f}s{eta_str}")

                result = model.transcribe(
                    chunk, language=lang_arg, verbose=False,
                    word_timestamps=True,
                )
                for seg in result.get("segments", []):
                    for w in seg.get("words", []):
                        word_text = w.get("word", "").strip()
                        if not word_text:
                            continue
                        start = w["start"] + time_offset + trim_offset_sec
                        end = w["end"] + time_offset + trim_offset_sec
                        raw_words.append({
                            "word": word_text,
                            "start": round(start, 3),
                            "end": round(end, 3),
                        })

        if _is_cancelled():
            return

        # ── 단어 정리 + 화자 매칭 ──
        _progress(80, "단어 정리 중...")

        # 화자 매칭
        if diarization_segments:
            from src.diarizer import _find_speaker_at
            import bisect
            sorted_dsegs = sorted(diarization_segments, key=lambda d: d["start"])
            dstarts = [d["start"] for d in sorted_dsegs]
            for w in raw_words:
                w["speaker"] = _find_speaker_at(sorted_dsegs, dstarts, w["start"], w["end"])
        else:
            for w in raw_words:
                w["speaker"] = None

        # ── 빈도 집계 + 모든 등장 위치 저장 ──
        from collections import Counter, defaultdict
        freq_counter = Counter()
        occurrences = defaultdict(list)  # (speaker, word) → [(start, end), ...]
        for w in raw_words:
            key = (w["speaker"], w["word"])
            freq_counter[key] += 1
            occurrences[key].append((w["start"], w["end"]))

        # 각 등장 위치마다 1개 항목 생성 (UI에서는 빈도로 그룹화)
        word_entries = []
        for (speaker, word), count in freq_counter.most_common():
            for start, end in occurrences[(speaker, word)]:
                word_entries.append({
                    "word": word,
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "frequency": count,  # 총 등장 횟수 (모든 행에 동일)
                })

        _log(f"분석 완료: {len(freq_counter)}개 고유 단어, {len(word_entries)}개 등장 위치")
        _progress(100, "분석 완료")
        msg_queue.put(("result", {"words": word_entries}))

    except Exception as e:
        _log(f"오류 발생: {e}")
        msg_queue.put(("error", str(e)))
