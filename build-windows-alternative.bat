@echo off
REM Alternative build script that uses --onedir instead of --onefile
REM This can help avoid some PyInstaller bytecode analysis issues

echo ========================================
echo Building Intelligent Annotator (Alternative Method)
echo Using --onedir mode (folder instead of single file)
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
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Clean Python cache
echo Cleaning Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f"

echo.
echo Building with --onedir mode...
echo This creates a folder with the executable and dependencies.
echo.

pyinstaller --name=labelImg ^
    --onedir ^
    --windowed ^
    --add-data "data;data" ^
    --add-data "resources;resources" ^
    --add-data "libs;libs" ^
    --hidden-import=PyQt5.QtCore ^
    --hidden-import=PyQt5.QtGui ^
    --hidden-import=PyQt5.QtWidgets ^
    --hidden-import=lxml ^
    --hidden-import=torch ^
    --hidden-import=ultralytics ^
    --exclude-module=matplotlib ^
    --exclude-module=scipy ^
    --exclude-module=pandas ^
    --exclude-module=jupyter ^
    --exclude-module=notebook ^
    --exclude-module=IPython ^
    labelImg.py

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed!
echo ========================================
echo.
echo The executable is in: dist\labelImg\labelImg.exe
echo All dependencies are in the same folder.
echo.
pause

