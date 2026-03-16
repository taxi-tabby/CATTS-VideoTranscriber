# CATTS - Video Transcriber

[OpenAI Whisper](https://github.com/openai/whisper)와 [pyannote.audio](https://github.com/pyannote/pyannote-audio)를 활용한 무료 오픈소스 미디어 텍스트 변환 도구입니다. 모든 처리가 로컬에서 수행됩니다.

> [English README](../README.md)

---

## 주요 기능

- **음성-텍스트 변환** -- Whisper 모델(tiny~large-v3)을 사용한 영상/음성 파일 텍스트 변환
- **화자 분리** -- pyannote.audio를 통한 화자 식별 및 라벨링
- **오디오 전처리** -- 고역 통과 필터, 노이즈 제거(spectral gating), Silero VAD 비음성 구간 무음 처리, 무음 트리밍, 피크 정규화로 인식 정확도 향상
- **타임스탬프 세그먼트** -- 구간별 타임스탬프와 함께 변환 결과 탐색
- **드래그 앤 드롭** -- 미디어 파일을 앱 창에 직접 드롭
- **내보내기** -- 변환 결과 저장
- **변환 이력** -- 모든 변환 결과를 로컬 SQLite DB에 저장, 검색 및 폴더 정리 지원
- **GPU 가속** -- CUDA 지원 시 변환 및 화자 분리 속도 향상
- **Windows 설치 프로그램** -- 독립 실행형 `.exe` 및 Inno Setup 설치 파일 제공

## 지원 형식

**영상** -- mp4, avi, mkv, mov, wmv, flv, webm

**음성** -- mp3, wav, flac, aac, ogg, wma, m4a

## 요구 사항

- Windows 10/11
- Python 3.11+
- NVIDIA GPU + CUDA (선택, 가속용)
- [HuggingFace 토큰](https://huggingface.co/settings/tokens) (화자 분리 사용 시에만 필요)

## 설치

### 소스에서 실행

```bash
git clone https://github.com/<your-username>/video-transcriber.git
cd video-transcriber
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 실행

```bash
python -m src.main
```

### 독립 실행 파일 빌드

```bash
build.bat
```

빌드 결과:
- 포터블 exe: `build\output\dist\VideoTranscriber\VideoTranscriber.exe`
- 설치 파일: `build\output\installer\CATTS_Setup_1.0.0.exe` ([Inno Setup 6](https://jrsoftware.org/isdl.php) 필요)

## 구조

```
src/
  main.py             -- 앱 진입점
  main_window.py      -- PySide6 GUI
  transcriber.py       -- 오디오 추출 및 Whisper 변환 파이프라인
  audio_preprocess.py  -- 오디오 전처리 (필터, 노이즈 제거, VAD, 정규화)
  diarizer.py          -- pyannote.audio 화자 분리
  config.py            -- 사용자 설정 (~/.video-transcriber/config.json)
  database.py          -- SQLite 저장 계층
  model_utils.py       -- Whisper 모델 캐시 관리
```

## 오디오 전처리 파이프라인

변환 전 모든 오디오에 다음 전처리를 적용하여 인식 정확도를 높입니다:

1. **고역 통과 필터** (80 Hz) -- 저주파 럼블/험 노이즈 제거
2. **노이즈 제거** -- noisereduce의 spectral gating으로 배경 소음 억제
3. **Silero VAD** -- 비음성 구간을 무음 처리하여 Whisper 환각 방지
4. **무음 트리밍** -- 앞뒤 무음 제거 (타임스탬프 오프셋 자동 보정)
5. **피크 정규화** -- 음량을 일정 수준으로 정규화

## 기술 스택

| 구성 요소 | 라이브러리 |
|---|---|
| GUI | PySide6 (Qt 6) |
| 음성 인식 | openai-whisper |
| 화자 분리 | pyannote.audio |
| 오디오 추출 | FFmpeg (imageio-ffmpeg) |
| 전처리 | scipy, noisereduce, Silero VAD |
| 데이터베이스 | SQLite |
| 패키징 | PyInstaller, Inno Setup |

## 기여

기여를 환영합니다. 가이드라인을 따르는 대부분의 PR은 승인됩니다. 자세한 내용은 [CONTRIBUTING.md](../CONTRIBUTING.md)를 참고하세요.

이 프로젝트는 [Claude Code](https://claude.ai/claude-code)를 적극 활용하여 개발되고 있으며, 기여 시에도 Claude Code 사용을 강력히 권장합니다. Claude Code는 코드베이스 전체 맥락을 이해하고 있어 기존 패턴과 일관된 코드를 작성하는 데 도움이 됩니다.

모든 제출물은 보안 문제를 검토합니다. 취약점을 도입하는 PR은 예외 없이 반려됩니다.

## Built with Claude

이 프로젝트는 Anthropic의 [Claude](https://claude.ai)를 적극 활용하여 설계, 구현, 유지보수되고 있습니다. 아키텍처 결정, 기능 구현, 코드 리뷰, 디버깅, 문서 작성 모두 Claude Code와의 협업으로 이루어집니다.

## 라이선스

MIT
