# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Video Analytics (VA) Test Module.

This module provides video analytics pipeline tests:
- va_light: Lightweight pipeline (YOLO11n + ResNet-50)
- va_medium: Medium pipeline (YOLOv5m + dual classification)
- va_heavy: Heavy pipeline (YOLO11m + dual classification)
- va_common: Shared utilities for all VA tests
"""

from .va_common import (
    VA_CONTAINER_PATH,
    attach_va_artifacts,
    create_va_metrics,
    determine_expected_modes,
    extract_fps_from_log,
    extract_metrics_from_csv,
    generate_va_charts,
    initialize_csv_files,
    prepare_docker_build_context,
    run_va_container,
    setup_x11_display,
)

__all__ = [
    "VA_CONTAINER_PATH",
    "attach_va_artifacts",
    "create_va_metrics",
    "determine_expected_modes",
    "extract_fps_from_log",
    "extract_metrics_from_csv",
    "generate_va_charts",
    "initialize_csv_files",
    "prepare_docker_build_context",
    "run_va_container",
    "setup_x11_display",
]
