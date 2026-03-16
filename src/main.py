import os
import sys

# 의존성 호환성 패치 — pyannote import 전에 실행 필수
from src.torchaudio_compat import apply_all_patches
apply_all_patches()

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication

from src.config import get_theme
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


def apply_dark_palette(app: QApplication):
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
    app.setPalette(palette)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CATTS - Video Transcriber")
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(get_icon_path()))
    app.setStyleSheet("""
        /* ── Base ── */
        QMainWindow, QWidget {
            background-color: #F5F7FB;
            color: #2D3748;
        }

        /* ── Left panel list ── */
        QListWidget {
            background-color: #FFFFFF;
            border: 1px solid #D4DCED;
            border-radius: 6px;
            padding: 4px;
            color: #2D3748;
            font-size: 13px;
        }
        QListWidget::item {
            padding: 8px 10px;
            border-bottom: 1px solid #EDF1F7;
            border-radius: 4px;
        }
        QListWidget::item:selected {
            background-color: #E8EEFB;
            color: #2D3748;
        }
        QListWidget::item:hover:!selected {
            background-color: #F0F4FA;
        }

        /* ── Buttons ── */
        QPushButton {
            background-color: #6B8FD4;
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            padding: 7px 18px;
            font-size: 13px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #5A7DC4;
        }
        QPushButton:pressed {
            background-color: #4A6AB0;
        }
        QPushButton:disabled {
            background-color: #B8C8E0;
            color: #E8EDF5;
        }

        /* ── Tabs ── */
        QTabWidget::pane {
            border: 1px solid #D4DCED;
            border-radius: 6px;
            background-color: #FFFFFF;
        }
        QTabBar::tab {
            background-color: #E8EEFB;
            color: #4A5568;
            padding: 8px 20px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 2px;
            font-size: 13px;
        }
        QTabBar::tab:selected {
            background-color: #FFFFFF;
            color: #5A7DC4;
            font-weight: 600;
            border: 1px solid #D4DCED;
            border-bottom: 1px solid #FFFFFF;
        }
        QTabBar::tab:hover:!selected {
            background-color: #DCE4F5;
        }

        /* ── Text areas ── */
        QPlainTextEdit {
            background-color: #FFFFFF;
            color: #2D3748;
            border: none;
            padding: 8px;
            selection-background-color: #C4D5F0;
            selection-color: #1A202C;
        }

        /* ── Progress bar ── */
        QProgressBar {
            background-color: #E8EEFB;
            border: 1px solid #D4DCED;
            border-radius: 4px;
            text-align: center;
            color: #4A5568;
            font-size: 12px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #7BA4D4, stop:0.5 #9B8ED4, stop:1 #7DD4D4);
            border-radius: 3px;
        }

        /* ── Labels ── */
        QLabel {
            color: #4A5568;
            font-size: 13px;
        }

        /* ── Splitter handle ── */
        QSplitter::handle:horizontal {
            background-color: #D4DCED;
            width: 1px;
        }

        /* ── Scrollbars ── */
        QScrollBar:vertical {
            background: #F0F4FA;
            width: 10px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background: #B8C8E0;
            border-radius: 5px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background: #9BAFC8;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """)

    if get_theme() == "dark":
        apply_dark_palette(app)

    db = Database(get_db_path())
    window = MainWindow(db)
    window.show()

    exit_code = app.exec()
    db.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
