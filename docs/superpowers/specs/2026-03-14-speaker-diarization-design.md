# Speaker Diarization Design

## Overview

pyannote.audio를 사용하여 화자 분리(speaker diarization) 기능을 추가한다. 음성의 특성 차이를 기반으로 화자를 자동 감지하고, 각 세그먼트에 화자 라벨을 부여한다. 사용자는 UI에서 화자 이름을 편집할 수 있으며, 복사 시 편집된 이름이 반영된다.

## Requirements

- 화자 수 자동 감지 (사전 지정 불필요)
- 정확도 우선
- 회의/미팅이 주 용도이나 다양한 상황 지원
- 화자 이름 편집 가능, 복사 시 편집된 이름 반영
- 화자 분리는 선택적 기능 (체크박스로 on/off)

## Processing Pipeline

```
파일 입력
  ↓
FFmpeg (음성 추출 → 16kHz mono WAV)                [5%]
  ↓
오디오 로드 (numpy array)                           [8%]
  ↓
[선택] pyannote.audio (전체 오디오 화자 분리)        [8% ~ 18%]
  ↓
pyannote 모델 해제 (메모리 확보)
  ↓
Whisper 모델 로드                                   [18% ~ 22%]
  ↓
Whisper (30초 청크 단위 텍스트 변환 + 화자 매칭)     [22% ~ 95%]
  ↓
DB 저장 + UI 표시                                   [100%]
```

### Diarization 단계

- Whisper 변환 전에 **전체 오디오**에 대해 한 번 실행
- 청크 단위로 나누면 화자 일관성이 깨지므로 전체 처리 필수
- 결과: 화자별 시간 구간 목록 (예: `SPEAKER_00: 0.0s ~ 12.3s`, `SPEAKER_01: 12.3s ~ 28.1s`)
- pyannote 라벨(`SPEAKER_00`) → 표시 라벨(`화자 1`)로 매핑 (0-based → 1-based 한글 라벨)
- diarization 완료 후 pyannote 모델을 메모리에서 해제한 뒤 Whisper 모델 로드 (GPU 메모리 충돌 방지)
- diarization은 blocking call이므로 처리 중 취소 불가 — 프로그레스 바에 "화자 분석 중... (취소 불가)" 표시

### 세그먼트-화자 매칭

- 최대 겹침(maximum overlap) 방식: 각 Whisper 세그먼트와 시간적으로 가장 많이 겹치는 diarization 구간의 화자를 할당
- 겹치는 구간이 없는 경우 `None` (화자 미상)
- 연속으로 같은 화자인 세그먼트는 전체 텍스트 뷰에서 하나의 블록으로 합침

## Database Changes

### segments 테이블 변경

```sql
-- 기존
segments(id, transcription_id, start_time, end_time, text)

-- 변경 후
segments(id, transcription_id, start_time, end_time, text, speaker TEXT)
```

- `speaker`: TEXT, nullable
- 자동 감지 시 `"화자 1"`, `"화자 2"` 등으로 저장
- 사용자가 이름 편집 시 해당 화자의 모든 세그먼트를 일괄 업데이트
- 기존 데이터 호환: nullable이므로 기존 레코드는 `NULL` → 화자 표시 없이 동작

### DB 마이그레이션

- `_create_tables()`에서 `PRAGMA table_info(segments)`로 `speaker` 컬럼 존재 여부 확인
- 없으면 `ALTER TABLE segments ADD COLUMN speaker TEXT` 실행
- 멱등성 보장 (이미 존재하면 skip)

### 기존 쿼리 변경

- `get_transcription()`: SELECT에 `speaker` 컬럼 추가

### 새로운 DB 메서드

- `update_speaker_name(transcription_id, old_name, new_name)`: 특정 트랜스크립션 내 화자 이름 일괄 변경
- `get_speakers(transcription_id)`: 해당 트랜스크립션의 고유 화자 목록 반환

## UI Changes

### 타임라인 표시 형식

```
-- 기존
[00:00:00 ~ 00:00:12]  안녕하세요 오늘은...

-- 변경 후
[00:00:00 ~ 00:00:12]  [화자 1]  안녕하세요 오늘은...
[00:00:12 ~ 00:00:28]  [화자 2]  첫 번째 안건은...
[00:00:28 ~ 00:00:45]  [화자 1]  그래서 결론적으로...
```

화자 정보가 없는 기존 데이터는 화자 라벨 없이 기존 형식 유지.

### 전체 텍스트 표시

화자가 바뀔 때마다 이름을 표시. 연속 같은 화자 세그먼트는 하나의 블록으로 합침:

```
화자 1: 안녕하세요 오늘은... 그래서 결론적으로...
화자 2: 첫 번째 안건은...
```

`full_text` DB 컬럼은 기존처럼 순수 텍스트만 저장. 전체 텍스트 탭은 segments에서 화자 정보를 조합하여 실시간 렌더링.

### 화자 분리 활성화

- 파일 추가 다이얼로그 후 "화자 분리 사용" 체크박스 또는 확인 다이얼로그
- 비활성화 시 기존과 동일하게 Whisper만 실행 (HuggingFace 토큰 불필요)

### 화자 관리 버튼

- 오른쪽 패널 헤더 영역에 "화자 관리" 버튼 추가
- 화자 정보가 있는 트랜스크립션 선택 시에만 표시
- 변환 진행 중에는 비활성
- 클릭 시 다이얼로그:
  - 감지된 화자 목록 표시 (예: `화자 1`, `화자 2`)
  - 각 화자 옆에 텍스트 입력 필드 → 이름 수정 가능
  - 확인 시 DB 업데이트 + 타임라인/전체텍스트 즉시 갱신

### 텍스트 복사

- 타임라인 복사: `[00:00:00 ~ 00:00:12]  [김대리]  안녕하세요...` (편집된 이름 반영)
- 전체 텍스트 복사: `김대리: 안녕하세요... \n박과장: 첫 번째 안건은...`

### 실시간 처리 중

- 기존과 동일하게 세그먼트가 실시간으로 추가되되, 화자 라벨도 함께 표시
- `segment_ready` 시그널에 `speaker` 필드 포함

## Dependencies

### 추가 패키지

```
pyannote.audio>=3.1,<4.0
```

PyTorch는 Whisper가 이미 사용하므로 추가 설치 불필요.
참고: pyannote.audio는 `speechbrain`, `transformers`, `torchaudio` 등을 전이 의존성으로 포함하므로 설치 크기가 증가한다.

### HuggingFace 토큰

pyannote.audio 모델 사용에 HuggingFace 토큰 필요:
- `pyannote/segmentation-3.0`
- `pyannote/speaker-diarization-3.1`
- 두 모델 모두 HuggingFace에서 사용 동의(license agreement) 필요

**토큰 처리 방식:**
- 화자 분리 첫 사용 시 토큰이 없으면 입력 다이얼로그 표시
- 사용자가 다이얼로그를 취소하면 화자 분리 없이 Whisper만 실행
- 토큰이 유효하지 않으면 오류 메시지 표시 후 Whisper만 실행으로 fallback
- `~/.video-transcriber/config.json`에 저장
- 설정 다이얼로그에서 토큰 수정/삭제 가능

### GPU/CPU 전략

- pyannote와 Whisper를 동시에 GPU에 올리지 않음
- pyannote 처리 완료 → 모델 해제 → Whisper 모델 로드 (순차적)
- GPU 사용 가능하면 GPU, 아니면 CPU로 자동 fallback

### PyInstaller 번들

- pyannote 모델은 첫 실행 시 다운로드 (Whisper와 동일 방식)
- spec 파일에 pyannote, speechbrain, transformers 관련 hidden imports 추가 필요

## Affected Files

| File | Changes |
|------|---------|
| `requirements.txt` | `pyannote.audio>=3.1,<4.0` 추가 |
| `src/transcriber.py` | diarization 단계 추가, segment에 speaker 포함, 프로그레스 조정, 모델 순차 로드/해제 |
| `src/database.py` | speaker 컬럼 마이그레이션, get_transcription에 speaker 추가, 화자 관련 메서드 추가 |
| `src/main_window.py` | 화자 분리 체크박스, 화자 표시, 화자 관리 다이얼로그, 전체텍스트 화자별 렌더링, HF 토큰 입력 |
| `build/video_transcriber.spec` | pyannote, speechbrain, transformers hidden imports 추가 |
