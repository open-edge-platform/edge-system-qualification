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
- Memory monitoring and OOM prevention
- X11 display detection and Docker volume/environment setup
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

# Memory utilities for OOM prevention
from .memory_utils import (
    get_memory_based_max_streams,
    check_available_memory,
    log_memory_status,
    get_memory_info,
    PSUTIL_AVAILABLE,
)
# X11 display utilities for container-based tests
from .container_utils import (
    detect_display_settings,
    get_x11_volumes,
    get_x11_environment,
    determine_display_output,
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
    # Memory utilities
    "get_memory_based_max_streams",
    "check_available_memory",
    "log_memory_status",
    "get_memory_info",
    "PSUTIL_AVAILABLE",
    # X11 display utilities
    "detect_display_settings",
    "get_x11_volumes",
    "get_x11_environment",
    "determine_display_output",
]

# Add host-only utilities to __all__ if available
if _HOST_UTILS_AVAILABLE:
    __all__.extend([
        "get_platform_identifier",
        "match_platform",
        "get_vdbox_count_for_device",
    ])
