# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
ESQ-specific utility modules.

This package contains AI and model-specific utilities that extend the core sysagent framework:
- models: Model batch processing, YOLO, OpenVINO, KaggleHub, media resources
- downloads: Model and dataset downloading from HuggingFace and ModelScope
- servers: Server-specific utilities (OVMS, etc.)
- services: Shared multi-service orchestration helpers for suite deployments
- references: Verified reference data handling and filtering
- telemetry: ESQ-specific telemetry modules (e.g., package_power via Intel RAPL)
"""

# Import ESQ-specific packages
from . import downloads, models, references, servers, telemetry

__all__ = [
    # Modules
    "models",
    "downloads",
    "servers",
    "services",
    "references",
    "telemetry",
]
