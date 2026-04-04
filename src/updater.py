"""앱 자동 업데이트 모듈.

GitHub Releases에서 새 버전을 감지하고, 현재 OS에 맞는 패키지를
다운로드하여 설치한다. 실행 중인 앱을 교체하기 위해 별도 스크립트를
생성하고, 앱 종료 후 교체 → 재시작을 수행한다.
"""

import os
import platform
import shutil
import stat
import sys
import tempfile

import requests

GITHUB_REPO = "taxi-tabby/CATTS-VideoTranscriber"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


def get_current_version() -> int:
    """현재 앱 버전을 반환한다."""
    try:
        from src.main import APP_VERSION
        return int(APP_VERSION)
    except Exception:
        return 0


def detect_install_type() -> str:
    """설치 유형을 감지한다."""
    if not getattr(sys, "frozen", False):
        return "dev"

    exe_path = os.path.dirname(sys.executable)

    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        if exe_path.lower().startswith(program_files.lower()):
            return "windows-installer"
        return "windows-portable"
    elif sys.platform == "darwin":
        return "macos-app"
    else:
        if os.environ.get("APPIMAGE"):
            return "linux-appimage"
        if exe_path.startswith("/opt/"):
            return "linux-system"
        return "linux-portable"


def check_for_update() -> dict | None:
    """GitHub Releases에서 새 버전을 확인한다.

    Returns:
        업데이트 정보 dict 또는 None (최신이면).
        {"version": int, "tag": str, "download_url": str, "size": int, "name": str}
    """
    current = get_current_version()
    if current <= 0:
        return None

    try:
        resp = requests.get(GITHUB_API_URL, timeout=10)
        if resp.status_code != 200:
            return None
        release = resp.json()
    except Exception:
        return None

    tag = release.get("tag_name", "")
    try:
        remote_version = int(tag.lstrip("v"))
    except (ValueError, TypeError):
        return None

    if remote_version <= current:
        return None

    assets = release.get("assets", [])
    asset_name = _match_asset(assets)
    if not asset_name:
        return None

    asset = next(a for a in assets if a["name"] == asset_name)
    return {
        "version": remote_version,
        "tag": tag,
        "download_url": asset["browser_download_url"],
        "size": asset["size"],
        "name": asset["name"],
        "release_url": release.get("html_url", ""),
    }


def _match_asset(assets: list) -> str | None:
    """현재 OS/아키텍처에 맞는 에셋 이름을 반환한다."""
    names = [a["name"] for a in assets]
    system = platform.system()

    if system == "Windows":
        for n in names:
            if "Portable" in n and "Windows" in n and n.endswith(".zip"):
                return n
    elif system == "Darwin":
        for n in names:
            if "macOS" in n and n.endswith(".dmg"):
                return n
    else:  # Linux
        # AppImage 실행 중이면 AppImage 우선
        if os.environ.get("APPIMAGE"):
            for n in names:
                if "Linux" in n and n.endswith(".AppImage"):
                    return n
        # 그 외에는 tar.gz
        for n in names:
            if "Linux" in n and n.endswith(".tar.gz"):
                return n
    return None


def can_auto_update() -> bool:
    """현재 설치 유형에서 자동 업데이트가 가능한지 반환한다."""
    install_type = detect_install_type()
    return install_type in (
        "windows-portable",
        "macos-app",
        "linux-portable",
        "linux-appimage",
    )


def download_update(url: str, expected_size: int,
                    progress_callback=None) -> str:
    """업데이트 파일을 다운로드한다. 레쥬메를 지원한다.

    Returns:
        다운로드된 파일 경로.
    """
    staging_dir = os.path.join(tempfile.gettempdir(), "catts-update")
    os.makedirs(staging_dir, exist_ok=True)

    filename = url.split("/")[-1]
    dest = os.path.join(staging_dir, filename)

    headers = {}
    mode = "wb"
    existing = 0

    if os.path.exists(dest):
        existing = os.path.getsize(dest)
        if existing >= expected_size:
            return dest  # 이미 다운로드 완료
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    resp = requests.get(url, headers=headers, stream=True, timeout=30)
    resp.raise_for_status()

    total = expected_size
    downloaded = existing

    with open(dest, mode) as f:
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total > 0:
                progress_callback(int(downloaded * 100 / total))

    # 사이즈 검증
    actual = os.path.getsize(dest)
    if actual != expected_size:
        os.remove(dest)
        raise RuntimeError(
            f"다운로드 파일 크기 불일치: 예상 {expected_size}, 실제 {actual}"
        )

    return dest


def prepare_update(downloaded_path: str) -> str:
    """다운로드된 파일을 staging 디렉토리에 압축 해제한다.

    Returns:
        압축 해제된 디렉토리 또는 파일 경로.
    """
    staging_dir = os.path.join(tempfile.gettempdir(), "catts-update", "staged")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    if downloaded_path.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(downloaded_path, "r") as zf:
            zf.extractall(staging_dir)
    elif downloaded_path.endswith(".tar.gz"):
        import tarfile
        with tarfile.open(downloaded_path, "r:gz") as tf:
            tf.extractall(staging_dir)
    elif downloaded_path.endswith(".AppImage"):
        # AppImage는 단일 파일 — staging에 복사
        dest = os.path.join(staging_dir, os.path.basename(downloaded_path))
        shutil.copy2(downloaded_path, dest)
        os.chmod(dest, os.stat(dest).st_mode | stat.S_IEXEC)
    elif downloaded_path.endswith(".dmg"):
        # macOS DMG는 나중에 마운트/복사
        dest = os.path.join(staging_dir, os.path.basename(downloaded_path))
        shutil.copy2(downloaded_path, dest)
    else:
        raise RuntimeError(f"지원하지 않는 파일 형식: {downloaded_path}")

    return staging_dir


def apply_update_and_restart(staging_dir: str) -> None:
    """업데이터 스크립트를 생성하고 앱을 종료한다.

    앱 종료 후 스크립트가 파일을 교체하고 앱을 재시작한다.
    """
    install_type = detect_install_type()

    if install_type == "windows-portable":
        _apply_windows_portable(staging_dir)
    elif install_type == "macos-app":
        _apply_macos(staging_dir)
    elif install_type == "linux-appimage":
        _apply_linux_appimage(staging_dir)
    elif install_type == "linux-portable":
        _apply_linux_portable(staging_dir)
    else:
        raise RuntimeError(f"자동 업데이트를 지원하지 않는 설치 유형: {install_type}")


def _apply_windows_portable(staging_dir: str):
    """Windows 포터블: bat 스크립트로 교체."""
    app_dir = os.path.dirname(sys.executable)
    app_exe = sys.executable

    # staging에서 새 앱 디렉토리 찾기
    staged_items = os.listdir(staging_dir)
    if len(staged_items) == 1 and os.path.isdir(os.path.join(staging_dir, staged_items[0])):
        new_dir = os.path.join(staging_dir, staged_items[0])
    else:
        new_dir = staging_dir

    script = os.path.join(tempfile.gettempdir(), "catts_update.bat")
    with open(script, "w", encoding="utf-8") as f:
        f.write(f"""@echo off
chcp 65001 >nul
echo CATTS 업데이트 중...
timeout /t 2 /nobreak >nul
rd /s /q "{app_dir}"
xcopy /e /i /y "{new_dir}" "{app_dir}\\"
rd /s /q "{staging_dir}"
start "" "{app_exe}"
del "%~f0"
""")

    os.startfile(script)
    sys.exit(0)


def _apply_macos(staging_dir: str):
    """macOS: shell 스크립트로 .app 교체."""
    # 현재 .app 번들의 경로 찾기
    app_path = sys.executable
    while app_path and not app_path.endswith(".app"):
        app_path = os.path.dirname(app_path)

    if not app_path or not app_path.endswith(".app"):
        raise RuntimeError("macOS .app 번들을 찾을 수 없습니다.")

    # DMG에서 .app 추출
    dmg_files = [f for f in os.listdir(staging_dir) if f.endswith(".dmg")]
    if dmg_files:
        dmg_path = os.path.join(staging_dir, dmg_files[0])
        script = os.path.join(tempfile.gettempdir(), "catts_update.sh")
        with open(script, "w") as f:
            f.write(f"""#!/bin/bash
sleep 2
MOUNT_POINT=$(hdiutil attach "{dmg_path}" -nobrowse -readonly | tail -1 | awk '{{print $NF}}')
if [ -d "$MOUNT_POINT/CATTS.app" ]; then
    rm -rf "{app_path}"
    cp -R "$MOUNT_POINT/CATTS.app" "{app_path}"
fi
hdiutil detach "$MOUNT_POINT" -quiet
rm -rf "{staging_dir}"
open "{app_path}"
rm "$0"
""")
        os.chmod(script, 0o755)
        import subprocess
        subprocess.Popen(["bash", script])
        sys.exit(0)


def _apply_linux_appimage(staging_dir: str):
    """Linux AppImage: 단일 파일 교체."""
    current_appimage = os.environ.get("APPIMAGE")
    if not current_appimage:
        raise RuntimeError("APPIMAGE 환경변수가 없습니다.")

    appimage_files = [f for f in os.listdir(staging_dir) if f.endswith(".AppImage")]
    if not appimage_files:
        raise RuntimeError("staging에서 AppImage를 찾을 수 없습니다.")

    new_appimage = os.path.join(staging_dir, appimage_files[0])

    script = os.path.join(tempfile.gettempdir(), "catts_update.sh")
    with open(script, "w") as f:
        f.write(f"""#!/bin/bash
sleep 2
cp "{new_appimage}" "{current_appimage}"
chmod +x "{current_appimage}"
rm -rf "{staging_dir}"
"{current_appimage}" &
rm "$0"
""")
    os.chmod(script, 0o755)
    import subprocess
    subprocess.Popen(["bash", script])
    sys.exit(0)


def _apply_linux_portable(staging_dir: str):
    """Linux 포터블: tar.gz 압축 해제 후 교체."""
    app_dir = os.path.dirname(sys.executable)
    app_exe = sys.executable

    staged_items = os.listdir(staging_dir)
    if len(staged_items) == 1 and os.path.isdir(os.path.join(staging_dir, staged_items[0])):
        new_dir = os.path.join(staging_dir, staged_items[0])
    else:
        new_dir = staging_dir

    script = os.path.join(tempfile.gettempdir(), "catts_update.sh")
    with open(script, "w") as f:
        f.write(f"""#!/bin/bash
sleep 2
rm -rf "{app_dir}"
cp -r "{new_dir}" "{app_dir}"
chmod +x "{app_exe}"
rm -rf "{staging_dir}"
"{app_exe}" &
rm "$0"
""")
    os.chmod(script, 0o755)
    import subprocess
    subprocess.Popen(["bash", script])
    sys.exit(0)


def cleanup_old_update():
    """이전 업데이트의 잔여 파일을 정리한다. 앱 시작 시 호출."""
    staging = os.path.join(tempfile.gettempdir(), "catts-update")
    if os.path.exists(staging):
        try:
            shutil.rmtree(staging)
        except Exception:
            pass
