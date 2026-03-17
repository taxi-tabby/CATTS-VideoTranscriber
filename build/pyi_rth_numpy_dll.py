# PyInstaller runtime hook: numpy/scipy DLL 로딩 보장
#
# 문제: numpy 2.x의 delvewheel 패치가 PyInstaller frozen 환경에서
# numpy.libs 경로를 찾지 못할 수 있음 (특히 GitHub Actions 빌드).
# 해결: numpy import 전에 모든 *.libs 디렉토리를 DLL 검색 경로에 등록.
import os
import sys

if getattr(sys, 'frozen', False):
    _meipass = sys._MEIPASS

    # 1. _MEIPASS 루트를 DLL 검색 경로에 추가
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(_meipass)
        except OSError:
            pass

    # 2. 모든 *.libs 디렉토리를 DLL 검색 경로에 추가
    #    (numpy.libs, scipy.libs, pandas.libs, llvmlite.libs 등)
    try:
        for entry in os.listdir(_meipass):
            if entry.endswith('.libs'):
                libs_path = os.path.join(_meipass, entry)
                if os.path.isdir(libs_path):
                    if hasattr(os, 'add_dll_directory'):
                        try:
                            os.add_dll_directory(libs_path)
                        except OSError:
                            pass
                    # PATH에도 추가 (add_dll_directory가 없는 구버전 대비)
                    os.environ['PATH'] = libs_path + os.pathsep + os.environ.get('PATH', '')
    except OSError:
        pass

    # 3. PATH에 _MEIPASS도 추가
    os.environ['PATH'] = _meipass + os.pathsep + os.environ.get('PATH', '')
