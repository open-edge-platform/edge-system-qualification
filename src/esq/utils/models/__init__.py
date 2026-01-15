# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Model utilities for ESQ test suites.

This module provides model preparation utilities including:
- Batch model downloading and preparation
- YOLO model utilities
- OpenVINO model utilities
- KaggleHub integration
- Media and LPR resources
- CLIP model utilities

"""

# Import batch processing utilities
from .batch import download_and_prepare_model, prepare_models_batch

__all__ = [
    # Batch processing
    "download_and_prepare_model",
    "prepare_models_batch",
]
