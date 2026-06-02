# Quick Build Guide

Intelligent Annotator is **labelImg** plus YOLO auto-detection under the **`intelligence/`** package.

## Dependencies (one install)

From the repo root, after creating a venv:

```bash
pip install -r requirements/requirements-linux-python3.txt
```

That file installs **PyQt5 + lxml** and pulls in **`intelligence/requirements.txt`** (Ultralytics, PyTorch, OpenCV, etc.).

| Piece | Location |
|--------|----------|
| UI + labeling | `requirements/requirements-linux-python3.txt` |
| AI stack | `intelligence/requirements.txt` (included via `-r`) |
| YOLO weights | `intelligence/models/*.pt` (optional; not in git) |
| Qt resources | `libs/resources.py` via `make qt5py3` once per clone |

Use **Python 3.10–3.12**. The app runs **without** a `.pt` file (manual labeling only). Auto-detect needs installed requirements **and** a model in `intelligence/models/` or loaded via the UI button.

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
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements/requirements-linux-python3.txt
make qt5py3    # skip if libs/resources.py already exists
```

### Start

```bash
source venv/bin/activate
python labelImg.py
```

### One-liner

```bash
python3 -m venv venv && . venv/bin/activate && pip install -U pip && pip install -r requirements/requirements-linux-python3.txt && (test -f libs/resources.py || make qt5py3) && python labelImg.py
```

### Optional Linux binary

```bash
source venv/bin/activate
pip install pyinstaller
pyinstaller labelImg.spec
# dist/labelImg
```

---

## Windows — run from source

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -U pip
pip install -r .\requirements\requirements-linux-python3.txt
pyrcc5 -o libs\resources.py resources.qrc
python labelImg.py
```

---

## Windows — build `.exe`

1. Venv with `pip install -r .\requirements\requirements-linux-python3.txt`
2. `pip install pyinstaller`
3. `.\build-windows.bat` or `.\build-windows.ps1`
4. Output: `dist\labelImg.exe` (5–15 min build)

Manual: `pyinstaller labelImg.spec`

---

## Notes

- **`venv/`** is gitignored — create per machine.
- **`libs/resources.py`** is gitignored — run `make qt5py3` / `pyrcc5` when missing.
- **Wayland:** try `export QT_QPA_PLATFORM=wayland` or `xcb` if the window misbehaves.
