#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO Inference Module for Auto-Annotation

This module provides functionality to load YOLO v8 models and run inference
on images to generate automatic annotations.
"""

import os
import sys

# Don't import ultralytics at module level - use lazy loading instead
# This prevents DLL initialization issues when imported by PyQt5 applications
ULTRALYTICS_AVAILABLE = None

def _check_ultralytics():
    """Lazy check for ultralytics availability - only checks importability without importing."""
    global ULTRALYTICS_AVAILABLE
    if ULTRALYTICS_AVAILABLE is not None:
        return ULTRALYTICS_AVAILABLE
    
    # Use importlib to check if module exists without actually importing it
    # This avoids triggering DLL loading at module import time
    import importlib.util
    spec = importlib.util.find_spec("ultralytics")
    if spec is None:
        ULTRALYTICS_AVAILABLE = False
        print("Warning: ultralytics package not found.")
        return False
    
    # Module exists, mark as available
    # Actual import will happen in load_model() when needed
    ULTRALYTICS_AVAILABLE = True
    return True


class YOLOInference:
    """Handles YOLO model loading and inference for auto-annotation."""
    
    def __init__(self, model_path=None):
        """
        Initialize YOLO inference handler.
        
        Args:
            model_path: Path to the YOLO model file (.pt). If None, will use default path.
        """
        self.model = None
        self.model_path = model_path
        self.class_names = {}
        
    def load_model(self, model_path=None):
        """
        Load YOLO model from file.
        
        Args:
            model_path: Path to model file. If None, uses self.model_path or default.
            
        Returns:
            True if model loaded successfully, False otherwise.
        """
        # Lazy import check - only imports ultralytics when actually needed
        if not _check_ultralytics():
            print("Warning: ultralytics package not available. Auto-annotation disabled.")
            return False
        
        # Import YOLO here (lazy loading) - only when actually needed
        from ultralytics import YOLO
            
        if model_path is None:
            model_path = self.model_path
            
        if model_path is None:
            # Default path: intelligence/best.pt relative to project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '..', 'intelligence', 'best.pt')
            model_path = os.path.normpath(model_path)
        
        if not os.path.exists(model_path):
            print(f"Warning: YOLO model file not found at {model_path}. Auto-annotation disabled.")
            return False
            
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            # Get class names from model
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
            return False
    
    def run_inference(self, image_path, conf_threshold=0.25):
        """
        Run YOLO inference on an image.
        
        Args:
            image_path: Path to the image file
            conf_threshold: Confidence threshold for detections (default: 0.25)
            
        Returns:
            List of detections, each as (class_id, confidence, x_min, y_min, x_max, y_max)
            Returns None if inference fails.
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None
            
        try:
            # Run inference
            results = self.model.predict(image_path, conf=conf_threshold, verbose=False)
            
            if not results or len(results) == 0:
                return []
            
            # Extract detections from first result (single image)
            result = results[0]
            detections = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                # Get boxes in xyxy format (absolute coordinates)
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes_xyxy)):
                    x_min, y_min, x_max, y_max = boxes_xyxy[i]
                    confidence = float(confidences[i])
                    class_id = int(class_ids[i])
                    
                    detections.append({
                        'class_id': class_id,
                        'confidence': confidence,
                        'x_min': float(x_min),
                        'y_min': float(y_min),
                        'x_max': float(x_max),
                        'y_max': float(y_max)
                    })
            
            return detections
            
        except Exception as e:
            print(f"Error running YOLO inference: {e}")
            return None
    
    def convert_detections_to_shapes(self, detections, image_width, image_height):
        """
        Convert YOLO detections to Shape format expected by labelImg.
        
        Args:
            detections: List of detection dicts from run_inference()
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            
        Returns:
            List of shapes in format: (label, points, line_color, fill_color, difficult)
            where points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        """
        if not detections:
            return []
        
        shapes = []
        
        for det in detections:
            # Get class name
            class_id = det['class_id']
            if class_id in self.class_names:
                label = self.class_names[class_id]
            else:
                label = f"class_{class_id}"
            
            # Get bounding box coordinates
            x_min = det['x_min']
            y_min = det['y_min']
            x_max = det['x_max']
            y_max = det['y_max']
            
            # Ensure coordinates are within image bounds
            x_min = max(0, min(x_min, image_width))
            x_max = max(0, min(x_max, image_width))
            y_min = max(0, min(y_min, image_height))
            y_max = max(0, min(y_max, image_height))
            
            # Skip invalid boxes
            if x_max <= x_min or y_max <= y_min:
                continue
            
            # Create points for rectangle (4 corners)
            points = [
                (x_min, y_min),  # top-left
                (x_max, y_min),  # top-right
                (x_max, y_max),  # bottom-right
                (x_min, y_max)   # bottom-left
            ]
            
            # Shape format: (label, points, line_color, fill_color, difficult)
            # line_color and fill_color are None to use defaults
            shape = (label, points, None, None, False)
            shapes.append(shape)
        
        return shapes
    
    def get_class_names(self):
        """Get class names dictionary from the model."""
        if self.model is None:
            return {}
        return self.class_names if self.class_names else {}


def load_yolo_model(model_path=None):
    """
    Convenience function to load YOLO model.
    
    Args:
        model_path: Path to model file
        
    Returns:
        YOLOInference instance if successful, None otherwise
    """
    inference = YOLOInference(model_path)
    if inference.load_model():
        return inference
    return None

