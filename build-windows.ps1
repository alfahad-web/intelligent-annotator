# PowerShell build script for creating Windows executable using PyInstaller
# This script builds the Intelligent Annotator application

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Intelligent Annotator for Windows" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if PyInstaller is installed
try {
    python -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller not found"
    }
} catch {
    Write-Host "ERROR: PyInstaller is not installed!" -ForegroundColor Red
    Write-Host "Please install it using: pip install pyinstaller" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Clean previous builds
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "build") {
    Write-Host "Removing build directory..." -ForegroundColor Gray
    Remove-Item -Recurse -Force "build"
}
if (Test-Path "dist") {
    Write-Host "Removing dist directory..." -ForegroundColor Gray
    Remove-Item -Recurse -Force "dist"
}

# Clean all Python cache files (important for PyInstaller)
Write-Host "Cleaning Python cache files..." -ForegroundColor Yellow
Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory | ForEach-Object {
    Write-Host "Removing $($_.FullName)..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $_.FullName
}
Get-ChildItem -Path . -Filter "*.pyc" -Recurse -File | Remove-Item -Force
Get-ChildItem -Path . -Filter "*.pyo" -Recurse -File | Remove-Item -Force

Write-Host ""
Write-Host "Starting PyInstaller build..." -ForegroundColor Yellow
Write-Host "This may take several minutes due to large dependencies (PyTorch, etc.)..." -ForegroundColor Gray
Write-Host ""

# Build using the spec file
pyinstaller labelImg.spec

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    Write-Host "Check the output above for error messages." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "The executable is located at: dist\labelImg.exe" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: The executable will be large (200-500 MB) due to bundled dependencies." -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to exit"

