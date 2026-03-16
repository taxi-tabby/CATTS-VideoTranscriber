import os
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from src.database import Database
from src.main_window import MainWindow


def get_icon_path() -> str:
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "assets", "icon", "app.ico")


def get_db_path() -> str:
    app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    db_dir = os.path.join(app_data, "VideoTranscriber")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "transcriptions.db")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CATTS - Video Transcriber")
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(get_icon_path()))

    db = Database(get_db_path())
    window = MainWindow(db)
    window.show()

    exit_code = app.exec()
    db.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
