# Building Intelligent Annotator for Windows

This guide explains how to build a standalone Windows executable (.exe) for the Intelligent Annotator application using PyInstaller.

## Prerequisites

1. **Python 3.10** (or compatible version)
2. **Virtual Environment** (recommended) - Already set up in `venv/`
3. **All dependencies installed** - Run:
   ```powershell
   .\venv\Scripts\Activate.ps1
   pip install -r .\requirements\requirements-linux-python3.txt
   pip install -r .\intelligence\requirements.txt
   ```

## Step 1: Install PyInstaller

Activate your virtual environment and install PyInstaller:

```powershell
.\venv\Scripts\Activate.ps1
pip install pyinstaller
```

## Step 2: Build the Executable

You have two options:

### Option A: Using the Batch Script (Recommended)

Simply run the provided batch script:

```powershell
.\build-windows.bat
```

### Option B: Using the PowerShell Script

```powershell
.\build-windows.ps1
```

### Option C: Manual Build

If you prefer to build manually:

```powershell
.\venv\Scripts\Activate.ps1
pyinstaller labelImg.spec
```

## Step 3: Find Your Executable

After the build completes (this may take 5-15 minutes), your executable will be located at:

```
dist\labelImg.exe
```

## Build Output

- **`build/`** - Temporary build files (can be deleted)
- **`dist/`** - Contains the final executable
- **`labelImg.spec`** - PyInstaller specification file (configuration)

## Important Notes

### File Size
The executable will be **large (200-500 MB)** because it includes:
- PyQt5 libraries
- PyTorch and all its dependencies
- Ultralytics YOLO
- OpenCV
- All other required libraries

### Antivirus Warnings
Some antivirus software may flag PyInstaller executables as suspicious. This is a false positive. To avoid this:
- Code sign your executable (requires a certificate)
- Submit to antivirus vendors for whitelisting
- Use a trusted build environment

### Testing
Always test the executable on a clean Windows machine (or VM) to ensure:
- All dependencies are included
- DLLs are properly bundled
- The application runs without errors

### Console Mode
If you need to see console output for debugging, edit `labelImg.spec` and change:
```python
console=False,  # Change to True
```

## Troubleshooting

### Build Fails with "Module not found"
- Ensure all dependencies are installed in your virtual environment
- Check that you're using the correct Python environment
- Try adding the missing module to `hiddenimports` in `labelImg.spec`

### Executable is too large
- This is normal due to PyTorch and dependencies
- Consider using `--onedir` mode instead of `--onefile` (modify spec file)
- Remove unnecessary dependencies if possible

### DLL errors when running .exe
- Ensure all required Visual C++ Redistributables are installed
- Check that torch DLLs are properly included
- Try building with `console=True` to see error messages

### Application crashes on startup
- Build with `console=True` to see error messages
- Check Windows Event Viewer for detailed error logs
- Ensure all data files (resources, data folders) are included

## Advanced: Creating an Installer

To create a professional installer, you can use:

### Inno Setup (Recommended)
1. Download Inno Setup: https://jrsoftware.org/isinfo.php
2. Create an `.iss` script to package the executable
3. Include data files and create shortcuts

### NSIS (Alternative)
1. Download NSIS: https://nsis.sourceforge.io/
2. Create an installer script
3. Build the installer

## Distribution

When distributing your application:
1. Test thoroughly on clean Windows systems
2. Include a README with system requirements
3. Mention that users need to load YOLO models via the button
4. Consider code signing for trust
5. Provide uninstall instructions if using an installer

## System Requirements

The built executable requires:
- Windows 10 or later (64-bit)
- Visual C++ Redistributables (usually pre-installed)
- Sufficient disk space (500 MB+ for the executable)

## Support

If you encounter issues:
1. Check the PyInstaller documentation: https://pyinstaller.org/
2. Review build logs in the `build/` directory
3. Test with `console=True` to see detailed error messages

