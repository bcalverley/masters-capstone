@echo off
setlocal

echo ============================================================
echo  Pokemon Card Scanner - Build Script
echo ============================================================
echo.

REM ── 1. Export class labels ────────────────────────────────────
echo [1/3] Exporting class labels from training set...
python export_labels.py
if errorlevel 1 (
    echo ERROR: export_labels.py failed. Make sure trainingassets/ is present.
    pause & exit /b 1
)
echo.

REM ── 2. PyInstaller ───────────────────────────────────────────
echo [2/3] Building executable with PyInstaller...
pyinstaller PokemonCardScanner.spec --clean --noconfirm
if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    pause & exit /b 1
)
echo.

REM ── 3. Copy .env to dist folder ──────────────────────────────
echo [3/3] Copying .env to dist folder...
if exist ".env" (
    copy /Y ".env" "dist\PokemonCardScanner\.env" >nul
    echo .env copied.
) else (
    echo WARNING: .env not found in project root - copy it manually to dist\PokemonCardScanner\
)
echo.

echo ============================================================
echo  Build complete!
echo  Executable: dist\PokemonCardScanner\PokemonCardScanner.exe
echo.
echo  To create a desktop shortcut, run:  install_shortcut.bat
echo ============================================================
echo.
pause
