# Video Transcriber - Windows GUI Application Design

## Overview

영상 파일에서 음성을 추출하여 텍스트로 변환하는 독립형 Windows GUI 프로그램.
기존 `transcribe.py` (Whisper + imageio_ffmpeg) 기반으로, 설치형 패키지로 배포.

## Tech Stack

- **GUI**: PySide6 (Qt6)
- **음성 인식**: OpenAI Whisper (medium 모델, 한국어 고정)
- **음성 추출**: imageio-ffmpeg (번들된 ffmpeg)
- **데이터 저장**: SQLite (Python 내장 sqlite3)
- **패키징**: PyInstaller (exe) + Inno Setup (설치 프로그램)
- **Python**: 3.11

## Architecture

단일 프로세스, QThread 기반. Whisper 변환은 Worker 스레드에서 실행하고 UI는 메인 스레드에서 유지.

## Project Structure

```
video-transcriber/
├── src/
│   ├── main.py              # 앱 진입점
│   ├── main_window.py       # 메인 윈도우 (목록 + 뷰어)
│   ├── transcriber.py       # Whisper 변환 QThread Worker
│   ├── database.py          # SQLite 데이터 관리
│   └── resources/           # 아이콘 등
├── build/
│   ├── video_transcriber.spec  # PyInstaller spec
│   └── installer.iss           # Inno Setup 스크립트
├── requirements.txt
└── pyproject.toml
```

## UI Layout

좌우 2패널 구조:

```
┌──────────────────────────────────────────────────┐
│  Video Transcriber                         - □ x │
├─────────────────┬────────────────────────────────┤
│  [+ 영상 추가]  │  제목: 20260226_200826.mp4     │
│                 │  날짜: 2026-02-26              │
│  ◉ 회의록_1     │  길이: 01:43:47                │
│    2026-02-26   │────────────────────────────────│
│    01:43:47     │  [타임라인] [전체 텍스트] [복사]│
│                 │────────────────────────────────│
│  ○ 강의_2      │  [00:00:00~00:00:12] 안녕하세   │
│    2026-02-27   │  요 오늘은...                   │
│                 │                                │
│  ○ 미팅_3      │  [00:00:12~00:00:28] 첫 번째   │
│    2026-02-28   │  안건은...                      │
│                 │                                │
│ ─────────────── │  [00:00:28~00:00:45] 그래서    │
│  [선택 삭제]    │  결론적으로...                   │
├─────────────────┴────────────────────────────────┤
│  ■■■■■■■■░░░░ 변환 중... 65%  (예상 잔여: 25분)  │
└──────────────────────────────────────────────────┘
```

- **왼쪽**: 트랜스크립션 목록 (파일명, 날짜, 길이). 선택/삭제 가능.
- **오른쪽**: 상세 뷰. 타임라인 뷰/전체 텍스트 뷰 탭 전환. 복사 버튼.
- **하단**: 변환 진행률 (프로그레스바 + 퍼센트 + 예상 시간)

## Data Model (SQLite)

```sql
CREATE TABLE transcriptions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    filepath    TEXT NOT NULL,
    duration    REAL,
    created_at  TEXT NOT NULL,
    full_text   TEXT
);

CREATE TABLE segments (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    transcription_id  INTEGER NOT NULL REFERENCES transcriptions(id) ON DELETE CASCADE,
    start_time        REAL NOT NULL,
    end_time          REAL NOT NULL,
    text              TEXT NOT NULL
);
```

## Core Flow

1. "영상 추가" 클릭 → 파일 선택
2. TranscriberWorker(QThread) 시작:
   - ffmpeg로 WAV 추출 (16kHz, mono, PCM 16bit)
   - Whisper 모델 로드 (첫 실행 시 자동 다운로드)
   - 변환 실행 → 진행률 시그널
   - 결과 SQLite 저장
   - 임시 WAV 삭제
3. 목록 갱신, 결과 표시

## Model Handling

- 첫 실행 시 Whisper medium 모델 자동 다운로드 (~1.4GB)
- 다운로드 진행률 UI에 표시
- `~/.cache/whisper/` 에 캐시 (Whisper 기본 동작)

## Error Handling

- 미지원 파일 형식 → 오류 다이얼로그
- 모델 다운로드 실패 → 재시도 안내
- 변환 중 종료 → 임시 파일 정리
- 디스크 공간 부족 → 사전 경고

## Packaging

- PyInstaller: `--onedir` 모드로 exe 번들
- Inno Setup: Windows 설치 프로그램 생성 (시작 메뉴, 바탕화면 단축키)
- `build.bat`: 빌드 자동화 스크립트

## Dependencies

- PySide6
- openai-whisper
- imageio-ffmpeg
- numpy
