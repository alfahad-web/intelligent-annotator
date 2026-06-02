#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backward-compatible re-export; implementation lives in intelligence/."""

from intelligence.yolo_inference import (  # noqa: F401
    YOLOInference,
    load_yolo_model,
    ULTRALYTICS_AVAILABLE,
    _check_ultralytics,
)
