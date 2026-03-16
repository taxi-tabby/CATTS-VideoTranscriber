# CATTS - Video Transcriber

Free, open-source media transcription tool powered by [OpenAI Whisper](https://github.com/openai/whisper) and [pyannote.audio](https://github.com/pyannote/pyannote-audio). Runs entirely half-offline on your local machine.

> Korean UI only. English UI is planned for a future release.
>
> [한국어 문서 (Korean)](docs/README_ko.md)

---

## Downloads


[CATTS-VideoTranscriber/releases]([https://huggingface.co/settings/tokens](https://github.com/taxi-tabby/CATTS-VideoTranscriber/releases)).

---

## Features

- **Speech-to-Text** -- Transcribe video and audio files using OpenAI Whisper (tiny through large-v3)
- **Speaker Diarization** -- Identify and label individual speakers via pyannote.audio
- **Audio Preprocessing** -- High-pass filter, noise reduction (spectral gating), Silero VAD non-speech suppression, silence trimming, and peak normalization for improved accuracy
- **Timestamped Segments** -- Browse transcription results with per-segment timestamps
- **Drag & Drop** -- Drop media files directly into the application window
- **Export** -- Save transcriptions for external use
- **Transcript History** -- All transcriptions are stored in a local SQLite database with search and folder organization
- **GPU Acceleration** -- CUDA support for faster transcription and diarization when available
- **Windows Installer** -- Ships as a standalone `.exe` with an Inno Setup installer

## Supported Formats

**Video** -- mp4, avi, mkv, mov, wmv, flv, webm

**Audio** -- mp3, wav, flac, aac, ogg, wma, m4a

## Requirements

- Windows 10/11
- Python 3.11+
- NVIDIA GPU with CUDA support (optional, for acceleration)
- [HuggingFace token](https://huggingface.co/settings/tokens) (required only for speaker diarization)

## Installation

### From source

```bash
git clone https://github.com/<your-username>/video-transcriber.git
cd video-transcriber
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
python -m src.main
```

### Build standalone executable

```bash
build.bat
```

This produces:
- Portable exe: `build\output\dist\VideoTranscriber\VideoTranscriber.exe`
- Installer: `build\output\installer\CATTS_Setup_1.0.0.exe` (requires [Inno Setup 6](https://jrsoftware.org/isdl.php))

## Architecture

```
src/
  main.py             -- Application entry point
  main_window.py      -- PySide6 GUI
  transcriber.py       -- Audio extraction and Whisper transcription pipeline
  audio_preprocess.py  -- Audio preprocessing (filter, denoise, VAD, normalize)
  diarizer.py          -- Speaker diarization via pyannote.audio
  config.py            -- User configuration (~/.video-transcriber/config.json)
  database.py          -- SQLite persistence layer
  model_utils.py       -- Whisper model cache management
```

## Audio Preprocessing Pipeline

All audio is preprocessed before transcription to improve recognition accuracy:

1. **High-pass filter** (80 Hz) -- Removes low-frequency rumble and hum noise
2. **Noise reduction** -- Spectral gating via noisereduce to suppress background noise
3. **Silero VAD** -- Zeroes out non-speech regions to prevent Whisper hallucinations
4. **Silence trimming** -- Strips leading and trailing silence (with timestamp offset correction)
5. **Peak normalization** -- Normalizes volume to a consistent level

## Tech Stack

| Component | Library |
|---|---|
| GUI | PySide6 (Qt 6) |
| Transcription | openai-whisper |
| Speaker Diarization | pyannote.audio |
| Audio Extraction | FFmpeg (via imageio-ffmpeg) |
| Preprocessing | scipy, noisereduce, Silero VAD |
| Database | SQLite |
| Packaging | PyInstaller, Inno Setup |

## Contributing

Contributions are welcome. Most PRs will be accepted as long as they follow the guidelines. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

This project is actively developed with [Claude Code](https://claude.ai/claude-code), and using it for contributions is strongly recommended. Claude Code understands the full codebase context and helps you write code that is consistent with existing patterns.

All submissions are reviewed for security concerns. PRs that introduce vulnerabilities will be rejected without exception.

## Built with Claude

This project is designed, implemented, and maintained with heavy use of [Claude](https://claude.ai) by Anthropic. Architecture decisions, feature implementation, code review, debugging, and documentation are all done in collaboration with Claude Code.

## License

MIT
