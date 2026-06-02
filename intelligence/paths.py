#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paths for intelligence assets (YOLO weights, etc.)."""

import os

# intelligence/ (this package directory)
INTELLIGENCE_DIR = os.path.dirname(os.path.abspath(__file__))

# Place trained *.pt weights here (see intelligence/models/README.md)
MODELS_DIR = os.path.join(INTELLIGENCE_DIR, 'models')

# Preferred default filename; any other *.pt in MODELS_DIR is used if this is missing
DEFAULT_MODEL_FILENAME = 'model.pt'


def models_dir():
    """Directory where YOLO weight files should be stored."""
    return MODELS_DIR


def ensure_models_dir():
    """Create intelligence/models/ if it does not exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def find_default_model_path():
    """
    Return path to the default YOLO weights file, or None if none are present.

    Lookup order:
      1. intelligence/models/model.pt
      2. First *.pt file in intelligence/models/ (sorted by name)
    """
    if not os.path.isdir(MODELS_DIR):
        return None

    preferred = os.path.join(MODELS_DIR, DEFAULT_MODEL_FILENAME)
    if os.path.isfile(preferred):
        return preferred

    pt_files = sorted(
        f for f in os.listdir(MODELS_DIR)
        if f.lower().endswith('.pt') and os.path.isfile(os.path.join(MODELS_DIR, f))
    )
    if pt_files:
        return os.path.join(MODELS_DIR, pt_files[0])
    return None


def resolve_model_path(explicit_path=None):
    """
    Resolve which .pt file to load.

    Args:
        explicit_path: User-selected path (e.g. from file dialog), or None.

    Returns:
        Absolute path to an existing .pt file, or None.
    """
    if explicit_path and os.path.isfile(explicit_path):
        return os.path.abspath(explicit_path)
    return find_default_model_path()
