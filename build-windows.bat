@echo off
REM Build script for creating Windows executable using PyInstaller
REM This script builds the Intelligent Annotator application

echo ========================================
echo Building Intelligent Annotator for Windows
echo ========================================
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo ERROR: PyInstaller is not installed!
    echo Please install it using: pip install pyinstaller
    pause
    exit /b 1
)

REM Clean previous builds
echo Cleaning previous builds...
if exist build (
    echo Removing build directory...
    rmdir /s /q build
)
if exist dist (
    echo Removing dist directory...
    rmdir /s /q dist
)

REM Clean all __pycache__ directories (important for PyInstaller)
echo Cleaning Python cache files...
for /d /r . %%d in (__pycache__) do @if exist "%%d" (
    echo Removing %%d...
    rmdir /s /q "%%d"
)
for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f"
for /r . %%f in (*.pyo) do @if exist "%%f" del /q "%%f"

echo.
echo Starting PyInstaller build...
echo This may take several minutes due to large dependencies (PyTorch, etc.)...
echo.

REM Build using the spec file
pyinstaller labelImg.spec

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    echo Check the output above for error messages.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo The executable is located at: dist\labelImg.exe
echo.
echo Note: The executable will be large (200-500 MB) due to bundled dependencies.
echo.
pause

