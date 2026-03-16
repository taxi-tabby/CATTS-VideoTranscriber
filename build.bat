@echo off
chcp 65001 >nul 2>&1
echo === CATTS - Video Transcriber Build ===
echo.

cd /d "%~dp0"

echo [1/3] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [2/3] PyInstaller building exe...
cd build
pyinstaller video_transcriber.spec --distpath output\dist --workpath output\build --clean -y
cd ..

if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Inno Setup installer...
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" (
    "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" build\installer.iss
) else (
    echo Inno Setup 6 not found. Install from https://jrsoftware.org/isdl.php
    echo exe build complete: build\output\dist\VideoTranscriber\
    pause
    exit /b 0
)

if errorlevel 1 (
    echo Inno Setup build failed!
    pause
    exit /b 1
)

echo.
echo Build complete!
echo exe: build\output\dist\VideoTranscriber\VideoTranscriber.exe
echo installer: build\output\installer\CATTS_Setup_1.0.0.exe
echo.
pause
