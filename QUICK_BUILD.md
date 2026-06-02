# Quick Build Guide

Intelligent Annotator is **labelImg** plus YOLO auto-detection under the **`intelligence/`** package.

Use **`python3.12`** to create the virtual environment (the pinned AI stack does not support Python 3.14+). Install it if needed: `sudo pacman -S python312` (Arch / CachyOS).

## Dependencies (one install)

From the repo root, with the venv created (see below):

```bash
venv/bin/python -m pip install -r requirements/requirements-linux-python3.txt
```

That file installs **PyQt5 + lxml** and pulls in **`intelligence/requirements.txt`** (Ultralytics, PyTorch, OpenCV, etc.).

| Piece | Location |
|--------|----------|
| UI + labeling | `requirements/requirements-linux-python3.txt` |
| AI stack | `intelligence/requirements.txt` (included via `-r`) |
| YOLO weights | `intelligence/models/*.pt` (optional; not in git) |
| Qt resources | `libs/resources.py` via `make qt5py3` once per clone |

The app runs **without** a `.pt` file (manual labeling only). Auto-detect needs installed requirements **and** a model in `intelligence/models/` or loaded via the UI button.

For Windows `.exe` details, see `BUILD_INSTRUCTIONS.md`.

---

## YOLO model placement

Put trained weights in **`intelligence/models/`**:

- Preferred name: **`model.pt`**
- Or any other **`*.pt`** (first match by name if `model.pt` is missing)

See **`intelligence/models/README.md`**. You can run immediately without a model; auto-annotation is skipped until weights exist or you use **Load YOLO Model Weights**.

---

## Linux — run from source

### System tools (only if `libs/resources.py` is missing)

- **Arch / CachyOS:** `sudo pacman -S qt5-tools`
- **Debian / Ubuntu:** `sudo apt install pyqt5-dev-tools`

### Setup

```bash
python3.12 -m venv venv
venv/bin/python -m pip install -U pip
venv/bin/python -m pip install -r requirements/requirements-linux-python3.txt
make qt5py3    # skip if libs/resources.py already exists
```

### Start

```bash
venv/bin/python labelImg.py
```

### One-liner

```bash
python3.12 -m venv venv && venv/bin/python -m pip install -U pip && venv/bin/python -m pip install -r requirements/requirements-linux-python3.txt && (test -f libs/resources.py || make qt5py3) && venv/bin/python labelImg.py
```

### Optional Linux binary

```bash
venv/bin/python -m pip install pyinstaller
venv/bin/pyinstaller labelImg.spec
# dist/labelImg
```

---

## Windows — run from source

Use Python 3.12 from [python.org](https://www.python.org/downloads/) or `py -3.12` if the launcher is installed.

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -U pip
.\venv\Scripts\python.exe -m pip install -r .\requirements\requirements-linux-python3.txt
pyrcc5 -o libs\resources.py resources.qrc
.\venv\Scripts\python.exe labelImg.py
```

---

## Windows — build `.exe`

1. Venv: `py -3.12 -m venv venv`, then `.\venv\Scripts\python.exe -m pip install -r .\requirements\requirements-linux-python3.txt`
2. `.\venv\Scripts\python.exe -m pip install pyinstaller`
3. `.\build-windows.bat` or `.\build-windows.ps1`
4. Output: `dist\labelImg.exe` (5–15 min build)

Manual: `pyinstaller labelImg.spec` (with the venv activated or on `PATH`)

---

## Notes

- **`venv/`** is gitignored — always create with `python3.12 -m venv venv` on Linux.
- **`libs/resources.py`** is gitignored — run `make qt5py3` / `pyrcc5` when missing.
- **Wayland:** try `export QT_QPA_PLATFORM=wayland` or `xcb` before starting the app.
