# PyInstaller runtime hook: torchaudio 호환성 패치를 앱 시작 시 최우선 적용
# 이 훅은 모든 다른 import보다 먼저 실행되어 pyannote/speechbrain이
# torchaudio.AudioMetaData 등 제거된 API를 사용할 수 있게 한다.
import sys
import os

# src 모듈을 찾을 수 있도록 경로 설정
if getattr(sys, 'frozen', False):
    base = sys._MEIPASS
else:
    base = os.path.dirname(os.path.abspath(__file__))

if base not in sys.path:
    sys.path.insert(0, base)

try:
    from src.torchaudio_compat import apply_all_patches
    apply_all_patches()
except Exception:
    pass
