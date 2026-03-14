import os
import sys

# 의존성 호환성 패치 — pyannote import 전에 실행 필수
from src.torchaudio_compat import apply_all_patches
apply_all_patches()

from PySide6.QtWidgets import QApplication

from src.database import Database
from src.main_window import MainWindow


def get_db_path() -> str:
    app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    db_dir = os.path.join(app_data, "VideoTranscriber")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "transcriptions.db")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Video Transcriber")
    app.setStyle("Fusion")

    db = Database(get_db_path())
    window = MainWindow(db)
    window.show()

    exit_code = app.exec()
    db.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
