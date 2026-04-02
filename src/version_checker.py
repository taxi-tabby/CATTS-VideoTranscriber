import json
import re
import urllib.request
import urllib.error

from PySide6.QtCore import QThread, Signal

RELEASES_URL = "https://api.github.com/repos/taxi-tabby/CATTS-VideoTranscriber/releases/latest"
RELEASES_PAGE = "https://github.com/taxi-tabby/CATTS-VideoTranscriber/releases"


class VersionCheckThread(QThread):
    """GitHub에서 최신 릴리즈 버전을 조회하는 백그라운드 스레드."""

    finished = Signal(int)  # 최신 버전 번호, 실패 시 -1

    def run(self):
        try:
            req = urllib.request.Request(RELEASES_URL, headers={"Accept": "application/vnd.github.v3+json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            tag = data.get("tag_name", "")
            m = re.search(r"(\d+)", tag)
            self.finished.emit(int(m.group(1)) if m else -1)
        except Exception:
            self.finished.emit(-1)
