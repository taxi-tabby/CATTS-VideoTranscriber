# 화자분리 파이프라인 재설계

## 문제

pyannote/speaker-diarization-3.1의 기본 Pipeline을 사용하면 화자가 1명인 오디오에서 7명으로 과다 추정하는 문제가 발생한다. 원인은 pyannote의 segmentation 모델이 지나치게 공격적으로 분할하고, 기본 clustering threshold가 높아 같은 화자를 여러 클러스터로 분리하기 때문이다.

## 목표

- **화자 수 자동 감지 정확도를 근본적으로 개선**한다
- 외부 인터페이스(`run_diarization`, `assign_speakers`, `map_speaker_labels`)는 유지한다
- UI 구조(자동 감지 / 직접 지정)는 변경하지 않는다

## 설계

### 아키텍처

pyannote Pipeline 일체형 실행을 3단계 분리 파이프라인으로 교체한다.

```
1. Silero VAD → 음성 구간 검출
2. pyannote 임베딩 모델 → 화자 특징 벡터 추출
3. Agglomerative Clustering + Silhouette Score → 최적 화자 수 결정 + 할당
```

### 1단계: 음성 구간 검출 (Silero VAD)

- `audio_preprocess.py`의 Silero VAD 로직을 재사용하여 `speech_timestamps`를 반환하는 함수를 추출한다
- 짧은 구간(< 0.5초)은 인접 구간과 병합하여 phantom speaker를 방지한다
- 긴 구간(> 10초)은 일정 간격으로 분할하여 화자 전환을 포착한다

### 2단계: 화자 임베딩 추출

- `pyannote.audio`의 임베딩 모델만 단독 사용한다 (Pipeline의 segmentation 모델은 사용하지 않음)
- 각 음성 구간에서 고정 길이 임베딩 벡터를 추출한다
- GPU/CPU 디바이스 선택 및 batch_size 로직은 기존과 동일하게 유지한다

### 3단계: 클러스터링 + 화자 수 결정

- `num_speakers`가 지정되면 해당 수로 Agglomerative Clustering 실행
- 자동 모드:
  - k=1 특수 처리: 모든 임베딩 간 cosine similarity가 임계값(0.7) 이상이면 화자 1명으로 판정
  - k=2~`max_speakers`(기본 10)까지 각각 클러스터링 후 silhouette score 최고점을 최적 k로 선택
- 결과: 각 음성 구간에 speaker 라벨 할당 → `[{start, end, speaker}, ...]` 반환

### 의존성

- 새 의존성 없음: `scikit-learn`은 `pyannote.audio`가 이미 의존하고 있음
- pyannote는 유지하되 Pipeline 대신 임베딩 모델만 사용

## 변경 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/diarizer.py` | `run_diarization()` 내부 전면 교체 |
| `src/audio_preprocess.py` | VAD speech_timestamps 반환 함수 추출 |
| `tests/test_diarizer.py` | 새 파이프라인에 맞게 테스트 갱신 |

## 검증 계획

### 1단계 검증: VAD 함수 추출 후
- `audio_preprocess.py`에서 추출한 `get_speech_timestamps()` 함수가 기존 `suppress_non_speech()`와 동일한 VAD 결과를 반환하는지 단위 테스트
- 짧은 구간 병합 및 긴 구간 분할 로직의 단위 테스트
- 기존 `suppress_non_speech()`가 추출된 함수를 내부적으로 호출하도록 리팩토링 후 기존 동작 유지 확인

### 2단계 검증: 임베딩 추출 후
- pyannote 임베딩 모델이 정상 로드되는지 확인
- 임베딩 벡터의 shape과 값 범위가 유효한지 확인
- GPU/CPU 양쪽에서 동작하는지 확인

### 3단계 검증: 클러스터링 후
- k=1 판정 로직: 단일 화자 임베딩에서 cosine similarity > 0.7 → 화자 1명 판정 테스트
- silhouette score 기반 k 선택: 알려진 화자 수의 합성 임베딩으로 정확도 테스트
- `num_speakers` 직접 지정 시 해당 수로 클러스터링되는지 테스트
- `min_speakers`/`max_speakers` 제약이 올바르게 적용되는지 테스트

### 통합 검증
- `run_diarization()` 반환 형식이 기존과 동일한지 확인 (`[{start, end, speaker}, ...]`)
- `assign_speakers()`, `map_speaker_labels()`가 새 결과와 정상 동작하는지 확인
- `transcriber.py`에서의 호출 흐름이 깨지지 않는지 확인
- 기존 `tests/test_diarizer.py` 전체 통과 확인
