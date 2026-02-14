# Quick Build Guide

## Fastest Way to Build

1. **Activate virtual environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install PyInstaller (if not already installed):**
   ```powershell
   pip install pyinstaller
   ```

3. **Run the build script:**
   ```powershell
   .\build-windows.bat
   ```
   OR
   ```powershell
   .\build-windows.ps1
   ```

4. **Find your executable:**
   ```
   dist\labelImg.exe
   ```

## That's it!

The build process will take 5-15 minutes. The executable will be in the `dist/` folder.

For detailed instructions and troubleshooting, see `BUILD_INSTRUCTIONS.md`.

