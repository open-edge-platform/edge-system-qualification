# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System information utilities package.

This package provides comprehensive system information collection and caching
capabilities, organized into focused modules for maintainability.
"""

# Import main classes and functions directly from local modules
from ..core import process

# Import secure process execution utilities from core
from ..core.process import (
    ProcessResult,
    ProcessSecurityConfig,
    SecureProcessExecutor,
    check_command_available,
    cleanup_processes,
    configure_security,
    run_command,
    run_command_with_output,
    run_git_command,
)
from . import formatter, hardware, power, software
from .cache import SystemInfoCache
from .formatter import (
    format_hardware_summary,
    format_software_summary,
    generate_simple_report,
)
from .info import collect_hardware_info, collect_software_info, collect_system_info
from .power import collect_power_info

__all__ = [
    # Main classes
    "SystemInfoCache",
    # Main functions
    "collect_system_info",
    "collect_hardware_info",
    "collect_software_info",
    "collect_power_info",
    # Formatting functions
    "generate_simple_report",
    "format_hardware_summary",
    "format_software_summary",
    # Process execution utilities
    "SecureProcessExecutor",
    "ProcessResult",
    "ProcessSecurityConfig",
    "run_command",
    "run_command_with_output",
    "check_command_available",
    "run_git_command",
    "configure_security",
    "cleanup_processes",
    # Submodules
    "hardware",
    "software",
    "power",
    "formatter",
    "process",
]
