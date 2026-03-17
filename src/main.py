import os
import sys
import warnings

# PyInstaller console=False 시 stdout/stderr가 None → 리다이렉트
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

# 의존성 호환성 패치 — pyannote import 전에 실행 필수
from src.torchaudio_compat import apply_all_patches
apply_all_patches()

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from src.config import get_theme, get_db_dir, get_hf_cache
from src.database import Database
from src.main_window import MainWindow
from src.model_utils import ensure_bundled_model


def get_icon_path() -> str:
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "assets", "icon", "app.ico")


def get_db_path() -> str:
    db_dir = get_db_dir()
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "transcriptions.db")


def apply_custom_paths() -> None:
    """사용자 지정 경로를 환경변수에 반영한다 (HuggingFace 캐시)."""
    hf_cache = get_hf_cache()
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_cache, "hub"))


_COMMON_STYLESHEET = """
    QPushButton {
        border: none;
        border-radius: 6px;
        padding: 7px 18px;
        font-size: 13px;
        font-weight: 500;
    }
    QLabel { font-size: 13px; }
    QProgressBar {
        border-radius: 4px;
        text-align: center;
        font-size: 12px;
    }
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #7BA4D4, stop:0.5 #9B8ED4, stop:1 #7DD4D4);
        border-radius: 3px;
    }
    QTabBar::tab {
        padding: 8px 20px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        margin-right: 2px;
        font-size: 13px;
    }
    QPlainTextEdit { border: none; padding: 8px; }
    QSplitter::handle:horizontal { width: 1px; }
    QScrollBar:vertical { width: 10px; border-radius: 5px; }
    QScrollBar::handle:vertical { border-radius: 5px; min-height: 30px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
    QTreeWidget { border-radius: 6px; padding: 4px; font-size: 13px; }
    QTreeWidget::item { padding: 4px 2px; border-radius: 4px; }
    QTableWidget { border: none; font-size: 13px; }
    QTableWidget::item { padding: 2px 4px; }
    QHeaderView::section { padding: 4px 8px; font-size: 12px; font-weight: 600; }
"""

_LIGHT_STYLESHEET = _COMMON_STYLESHEET + """
    QMainWindow, QWidget { background-color: #F5F7FB; color: #2D3748; }
    QTreeWidget { background-color: #FFFFFF; border: 1px solid #D4DCED; color: #2D3748; }
    QTreeWidget::item:selected { background-color: #E8EEFB; color: #2D3748; }
    QTreeWidget::item:hover:!selected { background-color: #F0F4FA; }
    QPushButton { background-color: #6B8FD4; color: #FFFFFF; }
    QPushButton:hover { background-color: #5A7DC4; }
    QPushButton:pressed { background-color: #4A6AB0; }
    QPushButton:disabled { background-color: #B8C8E0; color: #E8EDF5; }
    QTabWidget::pane { border: 1px solid #D4DCED; border-radius: 6px; background-color: #FFFFFF; }
    QTabBar::tab { background-color: #E8EEFB; color: #4A5568; }
    QTabBar::tab:selected { background-color: #FFFFFF; color: #5A7DC4; font-weight: 600;
        border: 1px solid #D4DCED; border-bottom: 1px solid #FFFFFF; }
    QTabBar::tab:hover:!selected { background-color: #DCE4F5; }
    QPlainTextEdit { background-color: #FFFFFF; color: #2D3748;
        selection-background-color: #C4D5F0; selection-color: #1A202C; }
    QProgressBar { background-color: #E8EEFB; border: 1px solid #D4DCED; color: #4A5568; }
    QLabel { color: #4A5568; }
    QSplitter::handle:horizontal { background-color: #D4DCED; }
    QScrollBar:vertical { background: #F0F4FA; }
    QScrollBar::handle:vertical { background: #B8C8E0; }
    QScrollBar::handle:vertical:hover { background: #9BAFC8; }
    QLineEdit { background-color: #FFFFFF; color: #2D3748; border: 1px solid #D4DCED;
        border-radius: 4px; padding: 4px 8px; }
    QComboBox { background-color: #FFFFFF; color: #2D3748; border: 1px solid #D4DCED;
        border-radius: 4px; padding: 4px 8px; }
    QGroupBox { color: #2D3748; }
    QTableWidget { background-color: #FFFFFF; color: #2D3748; border: 1px solid #D4DCED; }
    QTableWidget::item:selected { background-color: #E8EEFB; color: #2D3748; }
    QHeaderView::section { background-color: #F0F4FA; color: #4A5568; border: none; border-bottom: 1px solid #D4DCED; }
"""

_DARK_STYLESHEET = _COMMON_STYLESHEET + """
    QMainWindow, QWidget { background-color: #1E1E2E; color: #CDD6F4; }
    QTreeWidget { background-color: #181825; border: 1px solid #45475A; color: #CDD6F4; }
    QTreeWidget::item:selected { background-color: #45475A; color: #CDD6F4; }
    QTreeWidget::item:hover:!selected { background-color: #313244; }
    QPushButton { background-color: #5B6EA8; color: #CDD6F4; }
    QPushButton:hover { background-color: #6B7EB8; }
    QPushButton:pressed { background-color: #4A5D97; }
    QPushButton:disabled { background-color: #45475A; color: #6C7086; }
    QTabWidget::pane { border: 1px solid #45475A; border-radius: 6px; background-color: #181825; }
    QTabBar::tab { background-color: #313244; color: #A6ADC8; }
    QTabBar::tab:selected { background-color: #181825; color: #89B4FA; font-weight: 600;
        border: 1px solid #45475A; border-bottom: 1px solid #181825; }
    QTabBar::tab:hover:!selected { background-color: #3B3D50; }
    QPlainTextEdit { background-color: #181825; color: #CDD6F4;
        selection-background-color: #45475A; selection-color: #CDD6F4; }
    QProgressBar { background-color: #313244; border: 1px solid #45475A; color: #A6ADC8; }
    QLabel { color: #A6ADC8; }
    QSplitter::handle:horizontal { background-color: #45475A; }
    QScrollBar:vertical { background: #1E1E2E; }
    QScrollBar::handle:vertical { background: #45475A; }
    QScrollBar::handle:vertical:hover { background: #585B70; }
    QLineEdit { background-color: #313244; color: #CDD6F4; border: 1px solid #45475A;
        border-radius: 4px; padding: 4px 8px; }
    QComboBox { background-color: #313244; color: #CDD6F4; border: 1px solid #45475A;
        border-radius: 4px; padding: 4px 8px; }
    QGroupBox { color: #CDD6F4; }
    QSpinBox { background-color: #313244; color: #CDD6F4; border: 1px solid #45475A; border-radius: 4px; }
    QCheckBox { color: #CDD6F4; }
    QDialog { background-color: #1E1E2E; color: #CDD6F4; }
    QTableWidget { background-color: #181825; color: #CDD6F4; border: 1px solid #45475A; }
    QTableWidget::item:selected { background-color: #45475A; color: #CDD6F4; }
    QHeaderView::section { background-color: #313244; color: #A6ADC8; border: none; border-bottom: 1px solid #45475A; }
"""


def get_stylesheet(theme: str) -> str:
    """테마에 맞는 스타일시트를 반환한다."""
    return _DARK_STYLESHEET if theme == "dark" else _LIGHT_STYLESHEET


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CATTS - Video Transcriber")
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(get_icon_path()))

    app.setStyleSheet(get_stylesheet(get_theme()))

    apply_custom_paths()
    ensure_bundled_model()

    db = Database(get_db_path())
    window = MainWindow(db)
    window.show()

    exit_code = app.exec()
    db.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
