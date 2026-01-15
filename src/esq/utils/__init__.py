# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
ESQ-specific utility modules.

This package contains AI and model-specific utilities that extend the core sysagent framework:
- models: Model batch processing, YOLO, OpenVINO, KaggleHub, media resources
- downloads: Model and dataset downloading from HuggingFace and ModelScope
- servers: Server-specific utilities (OVMS, etc.)
"""

# Import ESQ-specific packages
from . import downloads, models, servers

__all__ = [
    # Modules
    "models",
    "downloads",
    "servers",
]
