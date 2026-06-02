# PyInstaller Build Troubleshooting

## Common Error: "IndexError: tuple index out of range"

This error occurs when PyInstaller tries to analyze bytecode in Python 3.10+ with certain packages (especially torch/ultralytics).

### Solution 1: Clean Python Cache (Try This First)

Before building, clean all Python cache files:

**Windows (PowerShell):**
```powershell
Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Filter "*.pyc" -Recurse -File | Remove-Item -Force
```

**Windows (Command Prompt):**
```cmd
for /d /r . %d in (__pycache__) do @if exist "%d" rmdir /s /q "%d"
for /r . %f in (*.pyc) do @if exist "%f" del /q "%f"
```

Then try building again.

### Solution 2: Use Alternative Build Method

Use the `--onedir` mode instead of `--onefile`:

```powershell
.\build-windows-alternative.bat
```

This creates a folder with the executable and dependencies instead of a single file.

### Solution 3: Update PyInstaller

Update to the latest PyInstaller version:

```powershell
pip install --upgrade pyinstaller
```

### Solution 4: Use Python 3.9 (If Available)

If you have Python 3.9 available, it may work better with PyInstaller:

```powershell
# Create a new venv with Python 3.9
python3.9 -m venv venv39
.\venv39\Scripts\Activate.ps1
pip install -r .\requirements\requirements-linux-python3.txt
pip install pyinstaller
```

### Solution 5: Build Without Torch (If YOLO Not Needed)

If you don't need YOLO functionality in the build, you can exclude torch:

Edit `labelImg.spec` and add to excludes:
```python
excludes=[
    'torch',
    'ultralytics',
    # ... other excludes
]
```

### Solution 6: Manual Build with Specific Options

Try building manually with these options:

```powershell
pyinstaller --name=labelImg ^
    --onedir ^
    --windowed ^
    --add-data "data;data" ^
    --add-data "resources;resources" ^
    --hidden-import=PyQt5.QtCore ^
    --hidden-import=PyQt5.QtGui ^
    --hidden-import=PyQt5.QtWidgets ^
    --hidden-import=lxml ^
    --exclude-module=torch.distributed ^
    --exclude-module=torch.multiprocessing ^
    labelImg.py
```

### Solution 7: Use --collect-all with Caution

If modules are missing, try:

```powershell
pyinstaller --name=labelImg ^
    --onedir ^
    --windowed ^
    --collect-all PyQt5 ^
    --collect-all torch ^
    --add-data "data;data" ^
    --add-data "resources;resources" ^
    labelImg.py
```

## Other Common Issues

### "ModuleNotFoundError" in the built executable

Add the missing module to `hiddenimports` in `labelImg.spec`:
```python
hiddenimports = [
    # ... existing imports
    'missing_module_name',
]
```

### Executable is too large

- Use `--onedir` instead of `--onefile` (already done in alternative script)
- Exclude unnecessary modules
- Consider using UPX compression (already enabled in spec file)

### DLL errors when running

- Ensure Visual C++ Redistributables are installed
- Check that all required DLLs are included
- Try building with `console=True` to see error messages

### Build takes too long

- This is normal (5-15 minutes) due to PyTorch
- Use `--onedir` mode for faster builds
- Consider excluding unnecessary modules

## Getting Help

1. Check PyInstaller logs in `build/` directory
2. Build with `console=True` to see detailed output
3. Check PyInstaller documentation: https://pyinstaller.org/
4. Search for your specific error on PyInstaller GitHub issues

## Recommended Build Order

1. **First try:** Clean cache and use updated spec file
2. **If that fails:** Use alternative build script (`build-windows-alternative.bat`)
3. **If still failing:** Try manual build with specific excludes
4. **Last resort:** Use Python 3.9 or exclude torch if not needed

