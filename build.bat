@echo off
echo === Video Transcriber Build ===
echo.

cd /d "%~dp0"

echo [1/3] PyInstaller로 exe 생성 중...
cd build
pyinstaller video_transcriber.spec --distpath output/dist --workpath output/build --clean -y
cd ..

if errorlevel 1 (
    echo PyInstaller 빌드 실패!
    pause
    exit /b 1
)

echo.
echo [2/3] Inno Setup으로 설치 프로그램 생성 중...
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" (
    "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" build\installer.iss
) else (
    echo Inno Setup 6이 설치되어 있지 않습니다.
    echo https://jrsoftware.org/isdl.php 에서 설치 후 다시 시도하세요.
    echo exe 빌드는 완료되었습니다: build\output\dist\VideoTranscriber\
    pause
    exit /b 0
)

if errorlevel 1 (
    echo Inno Setup 빌드 실패!
    pause
    exit /b 1
)

echo.
echo [3/3] 빌드 완료!
echo exe: build\output\dist\VideoTranscriber\VideoTranscriber.exe
echo 설치파일: build\output\installer\VideoTranscriber_Setup_1.0.0.exe
echo.
pause
