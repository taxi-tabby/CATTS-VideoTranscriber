@echo off
echo === Video Transcriber Build ===
echo.

cd /d "%~dp0"

echo [1/1] PyInstaller로 exe 생성 중...
cd build
pyinstaller video_transcriber.spec --distpath output/dist --workpath output/build --clean -y
cd ..

if errorlevel 1 (
    echo PyInstaller 빌드 실패!
    pause
    exit /b 1
)

echo.
echo 빌드 완료!
echo exe: build\output\dist\VideoTranscriber\VideoTranscriber.exe
echo.
pause
