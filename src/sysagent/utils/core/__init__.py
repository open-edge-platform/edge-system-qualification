# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Core utilities package.

This package contains core functionality for the framework including:
- Test result handling and caching
- KPI validation and metrics
- Shared state management
- Secure process execution
- Virtual environment management
"""

# Import from local modules
# Make sub-modules available
from . import cache, kpi, process, result, shared_state, venv
from .cache import TestResultCache
from .kpi import (
    KpiOperator,
    KpiSeverity,
    KpiType,
    KpiValidationResult,
    validate_boolean_kpi,
    validate_kpi,
    validate_list_kpi,
    validate_numeric_kpi,
    validate_string_kpi,
)

# Import secure process execution utilities
from .process import (
    ProcessHandle,
    ProcessResult,
    ProcessSecurityConfig,
    SecureProcessExecutor,
    check_command_available,
    cleanup_processes,
    configure_security,
    run_command,
    run_command_with_output,
    run_git_command,
    start_process,
)
from .result import Metrics, Result, get_metric_name_for_device
from .shared_state import INTERRUPT_OCCURRED, INTERRUPT_SIGNAL, INTERRUPT_SIGNAL_NAME

# Import virtual environment management utilities
from .venv import VenvManager, get_suite_python_executable, get_venv_manager, run_pytest_in_suite_venv, setup_suite_venv

__all__ = [
    # Cache functionality
    "TestResultCache",
    # Result handling
    "Result",
    "Metrics",
    "get_metric_name_for_device",
    # KPI validation
    "KpiType",
    "KpiOperator",
    "KpiSeverity",
    "KpiValidationResult",
    "validate_kpi",
    "validate_numeric_kpi",
    "validate_string_kpi",
    "validate_boolean_kpi",
    "validate_list_kpi",
    # Shared state
    "shared_state",
    "INTERRUPT_OCCURRED",
    "INTERRUPT_SIGNAL",
    "INTERRUPT_SIGNAL_NAME",
    # Process execution
    "SecureProcessExecutor",
    "ProcessResult",
    "ProcessSecurityConfig",
    "ProcessHandle",
    "run_command",
    "run_command_with_output",
    "check_command_available",
    "run_git_command",
    "configure_security",
    "cleanup_processes",
    "start_process",
    # Virtual environment management
    "VenvManager",
    "get_venv_manager",
    "setup_suite_venv",
    "get_suite_python_executable",
    "run_pytest_in_suite_venv",
    # Sub-modules
    "cache",
    "result",
    "kpi",
    "process",
    "venv",
]
