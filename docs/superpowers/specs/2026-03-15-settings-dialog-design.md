# 설정 다이얼로그 설계

**날짜:** 2026-03-15
**상태:** 승인됨

## 목표

앱의 모든 설정을 한 곳에서 관리할 수 있는 탭 기반 설정 다이얼로그를 추가한다. 기존 DiarizationSetupDialog를 흡수하여 통합한다.

## 변경 범위

1. `SettingsDialog` 클래스 추가 (탭 기반)
2. 기존 `DiarizationSetupDialog` 제거
3. config.py에 whisper_model getter/setter 추가

---

## 1. SettingsDialog 구조

### 일반 탭

```
┌─ 설정 ──────────────────────────────────┐
│  [일반]  [화자 분리]                     │
│─────────────────────────────────────────│
│                                          │
│  Whisper 모델 기본값:  [medium  ▾]       │
│    (tiny / base / small / medium /       │
│     large-v3)                            │
│                                          │
│  데이터 저장 위치:                        │
│    C:\Users\...\video-transcriber\       │
│    [폴더 열기]                           │
│                                          │
│              [취소]  [저장]               │
└──────────────────────────────────────────┘
```

- Whisper 모델: QComboBox (tiny, base, small, medium, large-v3), 기본값 medium
- 데이터 저장 위치: QLabel로 DB 파일 경로 표시 + "폴더 열기" QPushButton
- "폴더 열기": `QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))`로 OS 탐색기에서 열기

### 화자 분리 탭

```
┌─ 설정 ──────────────────────────────────┐
│  [일반]  [화자 분리]                     │
│─────────────────────────────────────────│
│                                          │
│  HuggingFace 토큰:  [••••••••••]        │
│    [토큰 검증]  [토큰 삭제]              │
│                                          │
│  라이선스 동의 필요:                      │
│    [pyannote/speaker-diarization-3.1]    │
│    [pyannote/segmentation-3.0]           │
│                                          │
│              [취소]  [저장]               │
└──────────────────────────────────────────┘
```

- HuggingFace 토큰: QLineEdit (EchoMode.Password)
- "토큰 검증": huggingface_hub.HfApi로 토큰 유효성 + 모델 접근 권한 확인
- "토큰 삭제": config에서 토큰 제거
- 라이선스 링크: QPushButton으로 브라우저에서 HF 모델 페이지 열기

### 공통 동작

- "저장": 변경된 설정을 config.json에 저장 후 닫기
- "취소": 변경 사항 버리고 닫기

---

## 2. 기존 코드와의 통합

### 제거: DiarizationSetupDialog

기존 `DiarizationSetupDialog` 클래스의 기능이 `SettingsDialog` 화자 분리 탭으로 완전히 흡수되므로 제거한다.

### 흐름 변경

```
현재:
  설정 버튼 → DiarizationSetupDialog (토큰 전용)
  파일 추가 → Yes/No 질문 → (토큰 없으면) DiarizationSetupDialog

변경 후:
  설정 버튼 → SettingsDialog (모든 설정)
  파일 추가 → TranscriptionSettingsDialog (diarization-accuracy 스펙)
           → (화자 분리 체크 + 토큰 없으면) SettingsDialog 화자 분리 탭으로 안내
```

### config.json

```json
{
  "hf_token": "hf_...",
  "whisper_model": "medium"
}
```

기존 구조에 `whisper_model` 키만 추가. DB 경로는 config에 저장하지 않고 실제 DB 경로를 직접 읽어서 표시.

---

## 수정 대상 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/main_window.py` | `DiarizationSetupDialog` 제거, `SettingsDialog` 추가, 설정 버튼 연결 변경 |
| `src/config.py` | `get_whisper_model()` / `set_whisper_model()` 추가 |
| `tests/test_config.py` | whisper_model getter/setter 테스트 추가 |
