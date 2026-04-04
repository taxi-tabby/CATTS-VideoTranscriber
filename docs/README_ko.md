<p align="center">
  <img src="../assets/icon/icon-ui-256.png" alt="CATTS" width="128">
</p>

<h1 align="center">CATTS - Video Transcriber</h1>

<p align="center">
  <strong>AI 기반 무료 오픈소스 미디어 텍스트 변환 도구</strong>
</p>

<p align="center">
  <a href="https://github.com/taxi-tabby/CATTS-VideoTranscriber/actions/workflows/test.yml"><img src="https://github.com/taxi-tabby/CATTS-VideoTranscriber/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/taxi-tabby/CATTS-VideoTranscriber/actions/workflows/build.yml"><img src="https://github.com/taxi-tabby/CATTS-VideoTranscriber/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/taxi-tabby/COVERAGE_GIST_ID/raw/coverage-badge.json" alt="Coverage">
  <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Windows-0078D6?logo=windows&logoColor=white" alt="Windows">
  <img src="https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white" alt="macOS">
  <img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" alt="Linux">
</p>

<p align="center">
  <a href="https://github.com/taxi-tabby/CATTS-VideoTranscriber/releases">다운로드</a> &bull;
  <a href="../README.md">English</a>
</p>

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **음성→텍스트 변환** | OpenAI Whisper (tiny ~ large-v3) 모델로 영상/음성 전사 |
| **화자 분리** | VAD + pyannote 임베딩 + silhouette 클러스터링으로 화자 식별 |
| **보컬 분리** | Demucs로 배경음악/효과음 제거 (영상/영화/노래 프로파일) |
| **분석 프로파일** | 인터뷰/대화 vs 영상/영화/노래 — 콘텐츠에 맞는 전처리 |
| **교정 사전** | 미디어별 단어 교정, 타임스탬프 기반 청크별 prompt 주입 |
| **환각 필터** | Whisper 반복/언어 불일치/no_speech 세그먼트 자동 제거 |
| **VAD 기반 청크 분할** | 무음 지점에서만 분할 — 발화 중간 잘림 방지 |
| **실시간 미리보기** | 변환 진행 중 세그먼트가 나타나는 것을 실시간 확인 |
| **크래시 복구** | 중단된 변환을 이어서 진행 |
| **메모리 안전** | ML 추론을 서브프로세스에서 실행 — 완료 후 OS가 메모리 전량 회수 |
| **내보내기** | SRT 자막, 일반 텍스트 |
| **GPU 가속** | CUDA 지원 시 처리 속도 향상 |

## 지원 형식

**영상** — mp4, avi, mkv, mov, wmv, flv, webm

**음성** — mp3, wav, flac, aac, ogg, wma, m4a

## 플랫폼 지원

| 플랫폼 | 배포 형태 | 비고 |
|--------|-----------|------|
| **Windows** | 포터블 ZIP, 설치 프로그램 (Inno Setup) | CUDA 선택 |
| **macOS** | DMG (ad-hoc 서명, 드래그 앤 드롭 설치) | Apple Silicon + Intel |
| **Linux** | tar.gz, .deb, .rpm, AppImage | CPU 또는 CUDA |

## 빠른 시작

### 다운로드

최신 릴리스: [**Releases**](https://github.com/taxi-tabby/CATTS-VideoTranscriber/releases)

### 소스에서 실행

```bash
git clone https://github.com/taxi-tabby/CATTS-VideoTranscriber.git
cd CATTS-VideoTranscriber

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m src.main
```

## 처리 파이프라인

```
미디어 파일
  │
  ├─ FFmpeg → 오디오 추출 (16kHz mono)
  │
  ├─ [noisy 프로파일] Demucs → 보컬 분리
  │
  ├─ 고역 필터 → 노이즈 제거 → Silero VAD → 트리밍 → 정규화
  │
  ├─ [선택] 화자 분리 (VAD + pyannote 임베딩 + 클러스터링)
  │
  ├─ VAD 기반 청크 분할 (무음 지점에서 분할)
  │
  ├─ Whisper 전사 (교정 사전 prompt 힌트 포함)
  │
  ├─ 환각 필터 (반복, 언어 불일치, no_speech)
  │
  └─ 교정 사전 후처리 (텍스트 치환)
```

## 기술 스택

| 구성 요소 | 라이브러리 |
|-----------|-----------|
| GUI | PySide6 (Qt 6) |
| 음성 인식 | OpenAI Whisper |
| 화자 분리 | pyannote.audio (임베딩), scikit-learn (클러스터링) |
| 보컬 분리 | Demucs (Meta) |
| VAD | Silero VAD |
| 오디오 처리 | scipy, noisereduce |
| 데이터베이스 | SQLite |
| 패키징 | PyInstaller |

## 테스트

```bash
python -m pytest tests/ --cov=src --cov-fail-under=60 -v
```

170개 테스트, 커버리지 71%. GitHub Actions에서 모든 PR마다 자동 실행.

## 기여

기여를 환영합니다. 자세한 내용은 [CONTRIBUTING.md](../CONTRIBUTING.md)를 참고하세요.

이 프로젝트는 [Claude Code](https://claude.ai/claude-code)로 개발되고 있으며, 기여 시에도 사용을 권장합니다.

## 라이선스

MIT

## Built with Claude

Anthropic의 [Claude](https://claude.ai)와 함께 설계, 구현, 유지보수되고 있습니다.
