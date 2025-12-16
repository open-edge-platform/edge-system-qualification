# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Media pipeline utilities for Intel ESQ test suites.

This package provides common utilities shared between media and proxy pipeline tests:
- Pipeline execution and benchmarking
- Telemetry collection
- Validation and pre-checks
- Platform identification and matching
- GPU topology detection
"""

# Use relative imports to support both installed package and Docker container usage
from .pipeline_utils import BaseDLBenchmark, BenchmarkLogger
from .telemetry import Telemetry
from .validation import (
    check_va_plugins,
    validate_options,
    format_log_section,
    detect_platform_type,
    get_render_device,
    normalize_device_name,
)

# Host-only utilities - only available when sysagent is installed (not in containers)
try:
    from .benchmark_platform import get_platform_identifier, match_platform
    from .gpu_topology import get_vdbox_count_for_device
    _HOST_UTILS_AVAILABLE = True
except ImportError:
    # Inside Docker container - these utilities use sysagent imports
    _HOST_UTILS_AVAILABLE = False
    get_platform_identifier = None
    match_platform = None
    get_vdbox_count_for_device = None

__all__ = [
    "BaseDLBenchmark",
    "BenchmarkLogger",
    "Telemetry",
    "check_va_plugins",
    "validate_options",
    "format_log_section",
    "detect_platform_type",
    "get_render_device",
    "normalize_device_name",
]

# Add host-only utilities to __all__ if available
if _HOST_UTILS_AVAILABLE:
    __all__.extend([
        "get_platform_identifier",
        "match_platform",
        "get_vdbox_count_for_device",
    ])
