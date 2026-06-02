#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO inference for auto-annotation (Ultralytics YOLOv8).

Model weights live under intelligence/models/ (see intelligence/paths.py).
"""

import os

from intelligence.paths import resolve_model_path

# Don't import ultralytics at module level - lazy loading avoids DLL issues with PyQt5
ULTRALYTICS_AVAILABLE = None


def _check_ultralytics():
    """Lazy check for ultralytics availability without importing it."""
    global ULTRALYTICS_AVAILABLE
    if ULTRALYTICS_AVAILABLE is not None:
        return ULTRALYTICS_AVAILABLE

    import importlib.util
    spec = importlib.util.find_spec("ultralytics")
    if spec is None:
        ULTRALYTICS_AVAILABLE = False
        print("Warning: ultralytics package not found. Install project requirements.")
        return False

    ULTRALYTICS_AVAILABLE = True
    return True


class YOLOInference:
    """Handles YOLO model loading and inference for auto-annotation."""

    def __init__(self, model_path=None):
        """
        Args:
            model_path: Path to .pt file, or None to use intelligence/models/ default.
        """
        self.model = None
        self.model_path = model_path
        self.class_names = {}

    def load_model(self, model_path=None):
        """
        Load YOLO model from file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not _check_ultralytics():
            print("Warning: ultralytics not available. Auto-annotation disabled.")
            return False

        from ultralytics import YOLO

        if model_path is None:
            model_path = self.model_path
        model_path = resolve_model_path(model_path)

        if model_path is None:
            print(
                "Warning: No YOLO model found. Place a .pt file in intelligence/models/ "
                "(e.g. model.pt) or use 'Load YOLO Model Weights'."
            )
            return False

        if not os.path.exists(model_path):
            print(f"Warning: YOLO model file not found at {model_path}.")
            return False

        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
            return False

    def run_inference(self, image_path, conf_threshold=0.25):
        if self.model is None:
            if not self.load_model():
                return None

        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None

        try:
            results = self.model.predict(image_path, conf=conf_threshold, verbose=False)

            if not results or len(results) == 0:
                return []

            result = results[0]
            detections = []

            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(boxes_xyxy)):
                    x_min, y_min, x_max, y_max = boxes_xyxy[i]
                    detections.append({
                        'class_id': int(class_ids[i]),
                        'confidence': float(confidences[i]),
                        'x_min': float(x_min),
                        'y_min': float(y_min),
                        'x_max': float(x_max),
                        'y_max': float(y_max),
                    })

            return detections

        except Exception as e:
            print(f"Error running YOLO inference: {e}")
            return None

    def convert_detections_to_shapes(self, detections, image_width, image_height):
        if not detections:
            return []

        shapes = []

        for det in detections:
            class_id = det['class_id']
            if class_id in self.class_names:
                label = self.class_names[class_id]
            else:
                label = f"class_{class_id}"

            x_min = max(0, min(det['x_min'], image_width))
            x_max = max(0, min(det['x_max'], image_width))
            y_min = max(0, min(det['y_min'], image_height))
            y_max = max(0, min(det['y_max'], image_height))

            if x_max <= x_min or y_max <= y_min:
                continue

            points = [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ]
            shapes.append((label, points, None, None, False))

        return shapes

    def get_class_names(self):
        if self.model is None:
            return {}
        return self.class_names if self.class_names else {}


def load_yolo_model(model_path=None):
    inference = YOLOInference(model_path)
    if inference.load_model():
        return inference
    return None
