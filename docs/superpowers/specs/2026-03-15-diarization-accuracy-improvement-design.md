# 화자 분석 정확도 개선 설계

**날짜:** 2026-03-15
**상태:** 승인됨

## 목표

화자 분석(speaker diarization)의 전반적인 정확도를 향상시키고, 사용자가 상황에 맞게 파라미터를 튜닝할 수 있도록 한다.

## 변경 범위

1. 트랜스크립션 설정 다이얼로그 추가
2. 화자 배정 알고리즘 개선 (가중 투표 방식)
3. pyannote 파이프라인에 화자 수 파라미터 전달
4. Whisper 모델 선택 기능

---

## 1. 트랜스크립션 설정 다이얼로그

현재 파일 추가 시 "화자 분리 사용 여부"만 Yes/No로 묻는 방식을 설정 다이얼로그로 확장한다.

### UI 구성

```
┌─ 트랜스크립션 설정 ─────────────────────┐
│                                          │
│  Whisper 모델:  [medium  ▾]              │
│    (tiny / base / small / medium / large) │
│                                          │
│  ☑ 화자 분리 사용                         │
│                                          │
│  화자 수 설정:  [자동 감지  ▾]            │
│    (자동 감지 / 직접 지정)                │
│                                          │
│  ┌─ 직접 지정 시 ──────────────┐         │
│  │ 최소 화자 수: [  2  ]       │         │
│  │ 최대 화자 수: [  4  ]       │         │
│  │  또는                       │         │
│  │ 정확한 화자 수: [    ]      │         │
│  └─────────────────────────────┘         │
│                                          │
│              [취소]  [시작]               │
└──────────────────────────────────────────┘
```

### 동작

- "화자 분리 사용" 체크 해제 시 화자 관련 옵션 비활성화
- "자동 감지" 선택 시 화자 수 입력 필드 비활성화
- "정확한 화자 수"에 값을 입력하면 min/max 필드 비활성화 (상호 배타)
- Whisper 모델 선택값은 config.json에 저장하여 다음에도 유지

### 입력 검증

- 화자 수 입력은 QSpinBox 사용 (정수만 허용, 최소 1, 최대 20)
- `min_speakers > max_speakers`인 경우 "시작" 버튼 비활성화 + 경고 표시
- `num_speakers`와 `min_speakers`/`max_speakers`는 UI에서 상호 배타 (동시 입력 불가)
- `run_diarization()`에서도 방어적 검증: `num_speakers`가 있으면 `min/max` 무시

### 기존 DiarizationSetupDialog과의 관계

- `DiarizationSetupDialog`는 **제거** — 기능이 `SettingsDialog` 화자 분리 탭으로 흡수됨 (settings-dialog 스펙 참조)
- 새로운 `TranscriptionSettingsDialog`가 **트랜스크립션 시작 전 설정**을 담당
- 흐름: 파일 추가 → `TranscriptionSettingsDialog` → 화자 분리 체크 + 토큰 없음 → `SettingsDialog` 화자 분리 탭으로 안내 → 돌아와서 `get_hf_token()` 재확인 후 시작

---

## 2. 화자 배정 알고리즘 개선

### 현재 방식 (단일 최대 겹침)

각 Whisper 세그먼트에 대해 가장 많이 겹치는 단일 diarization 세그먼트의 화자를 배정.

### 개선 방식 (가중 투표)

1. Whisper 세그먼트와 겹치는 **모든** diarization 세그먼트를 수집
2. 화자별로 겹침 시간을 **합산**
3. 합산 겹침이 가장 큰 화자를 배정
4. 동률 시 해당 Whisper 세그먼트 구간 내에서 가장 이른 diarization 세그먼트 start를 가진 화자 선택

### 예시

```
Whisper 세그먼트:     |─────────────────────────|
                      0s                       10s

Diarization:   [화자A 0~3s] [화자B 3~8s] [화자A 8~10s]

현재:   화자B (5초 겹침 > 3초, 2초)
개선:   화자A (3+2=5초) vs 화자B (5초) → 동률 → 화자A (먼저 등장)
```

---

## 3. pyannote 파이프라인 파라미터 전달

### 변경

```python
# 현재
diarization = pipeline(audio_path)

# 개선
diarization = pipeline(
    audio_path,
    num_speakers=num_speakers,      # 정확한 화자 수 (None이면 무시)
    min_speakers=min_speakers,       # 최소 화자 수 (None이면 무시)
    max_speakers=max_speakers,       # 최대 화자 수 (None이면 무시)
)
```

### 데이터 흐름

```
설정 다이얼로그 → _start_transcription() → TranscriberWorker
    → run_diarization(audio_path, hf_token, num_speakers, min_speakers, max_speakers)
        → pipeline(audio_path, ...)
```

- `TranscriberWorker`에 `num_speakers`, `min_speakers`, `max_speakers` 파라미터 추가
- `run_diarization()` 함수 시그니처 확장
- 셋 다 `None`이면 현재와 동일하게 자동 감지

---

## 4. Whisper 모델 선택

### 모델 옵션

| 모델 | 크기 | 상대 속도 | 용도 |
|------|------|-----------|------|
| tiny | ~39MB | 가장 빠름 | 빠른 확인용 |
| base | ~74MB | 빠름 | 간단한 내용 |
| small | ~244MB | 보통 | 일반 용도 |
| medium | ~769MB | 느림 | 높은 정확도 |
| large-v3 | ~1.5GB | 가장 느림 | 최고 정확도 |

### 변경

- `TranscriberWorker`에 `model_name` 파라미터 추가
- `whisper.load_model(self.model_name)` 으로 변경
- 설정 다이얼로그에서 선택한 값 전달
- 기본값은 `medium` 유지
- 마지막 선택값을 config.json에 저장

### config.json 최종 형태

```json
{
  "hf_token": "hf_...",
  "whisper_model": "medium"
}
```

화자 수 설정은 영상마다 다르므로 저장하지 않고, Whisper 모델만 저장한다.

---

## 수정 대상 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/main_window.py` | 설정 다이얼로그 클래스 추가, `_on_add_video` 흐름 변경 |
| `src/transcriber.py` | `model_name`, 화자 수 파라미터 추가 |
| `src/diarizer.py` | `run_diarization` 시그니처 확장, `assign_speakers` 알고리즘 개선 |
| `src/config.py` | `whisper_model` getter/setter 추가 |
| `tests/test_diarizer.py` | 가중 투표 알고리즘 테스트 추가 (동률 tie-break 포함) |
| `tests/test_config.py` | `whisper_model` getter/setter 테스트 추가 |
