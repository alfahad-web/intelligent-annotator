import os
import sys
import platform
import traceback
import ctypes
import importlib.util

print("=" * 60)
print("PYTHON / OS INFO")
print("=" * 60)
print("Python executable :", sys.executable)
print("Python version    :", sys.version)
print("Platform          :", platform.platform())
print("Architecture      :", platform.architecture())
print("Working dir       :", os.getcwd())
print()

print("=" * 60)
print("ENVIRONMENT VARIABLES (critical ones)")
print("=" * 60)
for k in ["PATH", "CUDA_PATH", "CUDA_HOME", "CONDA_PREFIX"]:
    print(f"{k}:")
    print(os.environ.get(k, "<NOT SET>"))
    print()

print("=" * 60)
print("CHECKING TORCH INSTALL LOCATION")
print("=" * 60)
spec = importlib.util.find_spec("torch")
if spec is None:
    print("❌ torch NOT FOUND by Python")
    sys.exit(1)
else:
    print("torch found at:", spec.origin)

torch_dir = os.path.dirname(spec.origin)
torch_lib = os.path.join(torch_dir, "lib")
print("torch lib dir:", torch_lib)
print("Contents of torch/lib:")
try:
    for f in os.listdir(torch_lib):
        print(" ", f)
except Exception as e:
    print("❌ Could not list torch/lib:", e)

print()

print("=" * 60)
print("DIRECT DLL LOAD TEST (c10.dll)")
print("=" * 60)
dll_path = os.path.join(torch_lib, "c10.dll")
print("Attempting to load:", dll_path)

try:
    ctypes.WinDLL(dll_path)
    print("✅ c10.dll loaded successfully via ctypes")
except Exception as e:
    print("❌ FAILED to load c10.dll")
    traceback.print_exc()

print()

print("=" * 60)
print("FULL TORCH IMPORT TEST")
print("=" * 60)
try:
    import torch
    print("✅ torch imported successfully")
    print("torch version:", torch.__version__)
    print("torch config:")
    print(torch.__config__.show())
except Exception:
    print("❌ torch import FAILED")
    traceback.print_exc()

print()

print("=" * 60)
print("MSVC RUNTIME PROBING (common missing DLLs)")
print("=" * 60)
msvc_dlls = [
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "api-ms-win-crt-runtime-l1-1-0.dll",
]

for dll in msvc_dlls:
    try:
        ctypes.WinDLL(dll)
        print(f"✅ {dll} loaded")
    except Exception as e:
        print(f"❌ {dll} FAILED:", e)

print()

print("=" * 60)
print("DONE")
print("=" * 60)
