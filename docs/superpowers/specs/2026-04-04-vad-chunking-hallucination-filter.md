# VAD 기반 지능형 청크 분할 + 환각 후처리

## 문제

현재 30초 고정 분할로 Whisper에 전달할 때:
- 발화 중간이 잘려 환각(반복, 갑작스러운 영어 전환) 발생
- 무음 구간이 Whisper에 전달되어 "채워넣기" 환각 유발
- initial_prompt에 이전 환각이 포함되어 환각 전파

## 목표

- 발화 중간을 자르지 않는 지능형 청크 분할
- 환각 텍스트를 탐지/제거하는 후처리 필터
- 기존 인터페이스 유지 (외부 변경 없음)

## 설계

### 1. VAD 기반 청크 분할

`build_vad_chunks(audio, speech_segments, max_chunk_sec=30)` 함수를 `src/audio_preprocess.py`에 추가한다.

- 전처리에서 구한 Silero VAD speech_segments를 입력으로 받는다
- 인접 음성 구간을 병합하되 max_chunk_sec를 넘지 않도록 무음 지점에서 분할한다
- 무음만 있는 구간은 제외한다
- 반환: `[{"start_sample": int, "end_sample": int}, ...]`

분할 규칙:
1. speech_segments를 순회하며 현재 청크에 추가
2. 현재 청크 + 다음 구간이 max_chunk_sec를 초과하면 현재 청크를 확정하고 새 청크 시작
3. 청크 경계에 100ms 패딩을 추가하여 자연스러운 전환

### 2. 환각 후처리 필터

`filter_hallucinations(segments, language)` 함수를 `src/hallucination_filter.py`에 추가한다.

필터 규칙:
- **반복 탐지**: 동일 텍스트가 연속 3회 이상 반복 → 1회만 유지
- **언어 불일치**: language가 "ko"인데 세그먼트가 순수 라틴 문자만 포함 → 제거 (영어 등)
- **no_speech 필터**: Whisper의 no_speech_prob > 0.6인 세그먼트 → 제거
- **극단적 짧은 세그먼트**: 0.1초 미만이면서 특수문자/공백만 → 제거

`language`가 "auto"이거나 None이면 언어 불일치 필터를 건너뛴다.

### 3. 파이프라인 통합

`_subprocess_worker()`에서:
1. 전처리 시 `get_speech_segments()` 결과를 별도 보관
2. 30초 고정 분할 대신 `build_vad_chunks(audio, speech_segments)` 사용
3. Whisper 전사 루프는 새 청크 목록 기반으로 실행
4. 전사 완료 후 `filter_hallucinations(all_segments, language)` 적용
5. 이후 화자 매칭은 기존과 동일

### 4. 변경 파일

| 파일 | 변경 |
|------|------|
| `src/audio_preprocess.py` | `build_vad_chunks()` 추가 |
| `src/hallucination_filter.py` | 새 파일 |
| `src/transcriber.py` | `_subprocess_worker()` 청크 분할 교체 + 후처리 적용 |
| `tests/test_audio_preprocess.py` | VAD 청크 분할 테스트 |
| `tests/test_hallucination_filter.py` | 환각 필터 테스트 |

### 5. 검증

각 단계별:
- `build_vad_chunks()`: 무음 지점 분할, max_chunk_sec 준수, 빈 입력 처리
- `filter_hallucinations()`: 반복 제거, 언어 불일치, no_speech, 짧은 세그먼트
- 통합: 기존 122개 테스트 전체 통과
- 코드 리뷰: 구현 완료 후 코드 리뷰 에이전트 실행
