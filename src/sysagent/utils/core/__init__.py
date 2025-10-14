# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Core utilities package.

This package contains core functionality for the framework including:
- Test result handling and caching
- KPI validation and metrics
- Shared state management
- Secure process execution
"""

# Import from local modules
from .cache import TestResultCache
from .result import Result, Metrics, get_metric_name_for_device
from .kpi import (
    KpiType, KpiOperator, KpiSeverity, 
    KpiValidationResult, 
    validate_kpi,
    validate_numeric_kpi,
    validate_string_kpi,
    validate_boolean_kpi,
    validate_list_kpi
)
from . import shared_state
from .shared_state import INTERRUPT_OCCURRED, INTERRUPT_SIGNAL, INTERRUPT_SIGNAL_NAME

# Import secure process execution utilities
from .process import (
    SecureProcessExecutor, ProcessResult, ProcessSecurityConfig,
    run_command, run_command_with_output, check_command_available,
    run_git_command, configure_security, cleanup_processes
)

# Make sub-modules available
from . import cache
from . import result  
from . import kpi
from . import process

__all__ = [
    # Cache functionality
    'TestResultCache',
    
    # Result handling
    'Result', 
    'Metrics',
    'get_metric_name_for_device',
    
    # KPI validation
    'KpiType',
    'KpiOperator', 
    'KpiSeverity',
    'KpiValidationResult',
    'validate_kpi',
    'validate_numeric_kpi',
    'validate_string_kpi',
    'validate_boolean_kpi',
    'validate_list_kpi',
    
    # Shared state
    'shared_state',
    'INTERRUPT_OCCURRED',
    'INTERRUPT_SIGNAL', 
    'INTERRUPT_SIGNAL_NAME',
    
    # Process execution
    'SecureProcessExecutor',
    'ProcessResult', 
    'ProcessSecurityConfig',
    'run_command',
    'run_command_with_output',
    'check_command_available',
    'run_git_command',
    'configure_security',
    'cleanup_processes',
    
    # Sub-modules
    'cache',
    'result',
    'kpi',
    'process'
]
