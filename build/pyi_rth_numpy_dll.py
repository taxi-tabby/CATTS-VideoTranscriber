# PyInstaller runtime hook: numpy/scipy DLL 로딩 보장
#
# 문제: numpy 2.x의 delvewheel 패치가 PyInstaller frozen 환경에서
# numpy.libs 경로를 찾지 못할 수 있음 (특히 GitHub Actions 빌드).
#
# 해결:
#   1. 모든 *.libs 디렉토리를 os.add_dll_directory()로 등록
#   2. PATH 환경변수에도 추가 (구버전 Windows 호환)
#   3. .libs/ DLL을 _MEIPASS 루트에 복사 (최후의 보장)
import os
import sys
import shutil

if getattr(sys, 'frozen', False):
    _meipass = sys._MEIPASS

    # 1. _MEIPASS 루트를 DLL 검색 경로에 추가
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(_meipass)
        except OSError:
            pass

    # 2. 모든 *.libs 디렉토리를 DLL 검색 경로에 추가
    _libs_dirs = []
    try:
        for entry in os.listdir(_meipass):
            if entry.endswith('.libs'):
                libs_path = os.path.join(_meipass, entry)
                if os.path.isdir(libs_path):
                    _libs_dirs.append(libs_path)
                    if hasattr(os, 'add_dll_directory'):
                        try:
                            os.add_dll_directory(libs_path)
                        except OSError:
                            pass
    except OSError:
        pass

    # 3. PATH 환경변수에 추가
    _extra_paths = [_meipass] + _libs_dirs
    os.environ['PATH'] = os.pathsep.join(_extra_paths) + os.pathsep + os.environ.get('PATH', '')

    # 4. .libs/ 내 DLL을 _MEIPASS 루트에 복사 (최후의 보장)
    #    일부 환경에서 add_dll_directory와 PATH 모두 실패할 수 있으므로
    #    DLL을 exe와 같은 디렉토리에 두면 Windows가 자동으로 찾음
    for libs_dir in _libs_dirs:
        try:
            for fname in os.listdir(libs_dir):
                if fname.lower().endswith('.dll'):
                    src = os.path.join(libs_dir, fname)
                    dst = os.path.join(_meipass, fname)
                    if not os.path.exists(dst):
                        try:
                            shutil.copy2(src, dst)
                        except (OSError, shutil.SameFileError):
                            pass
        except OSError:
            pass
