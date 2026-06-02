# YOLO model weights

Place your trained Ultralytics YOLO weights (`.pt` files) in this folder.

## Default lookup

1. `model.pt` — preferred name  
2. Otherwise, the first `*.pt` file in this directory (alphabetical order)

## Running without a model

The annotator **starts and works for manual labeling** without any file here.
Auto-detection runs only when:

- A `.pt` file is present in this folder, or  
- You load weights via **Load YOLO Model Weights** in the app

Weights are **not** committed to git (see `.gitignore`). Copy your trained `best.pt` here and rename to `model.pt` if you like.
