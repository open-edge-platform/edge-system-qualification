# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Core Testing Framework Pytest Plugin Package

This package provides common fixtures and hooks for the core testing framework.
All functionality is automatically registered with pytest through entry points.

Key Features:
- Automatic test parameterization based on config.yml files
- System requirement validation
- KPI validation
- Result summarization
- Test execution management with caching
- Allure reporting enhancements
"""

import logging

# Import all fixtures and hooks from their respective modules
# These imports ensure that pytest can discover all fixtures and hooks
from .pytest_allure import (
    pytest_collection_modifyitems,
    pytest_itemcollected,
    pytest_report_teststatus,
    pytest_runtest_call,
    pytest_runtest_logreport,
    pytest_runtest_logstart,
    pytest_runtest_makereport,
    pytest_runtest_setup,
    pytest_sessionstart,
)
from .pytest_cache import cache_result, cached_result
from .pytest_execution import execute_test_with_cache, prepare_test
from .pytest_parameterization import pytest_generate_tests, request_fixture
from .pytest_suite import get_kpi_config, suite_configs
from .pytest_summarization import summarize_test_results
from .pytest_validation import (
    validate_system_requirements_from_configs,
    validate_test_results,
)

# Initialize logger
logger = logging.getLogger(__name__)
logger.debug("core pytest plugin package loaded")

# Export all fixtures and hooks for public usage
__all__ = [
    # Allure hooks - defined in allure_hooks.py
    "pytest_runtest_setup",  # Unified hook for Allure reporting
    "pytest_runtest_call",  # Hook for setting Allure metadata during test execution
    "pytest_runtest_logstart",
    "pytest_itemcollected",
    "pytest_runtest_logreport",
    "pytest_runtest_makereport",
    "pytest_report_teststatus",
    "pytest_sessionstart",
    "pytest_collection_modifyitems",
    # Validation
    "validate_test_results",
    "validate_system_requirements_from_configs",
    # Summarization
    "summarize_test_results",
    # Execution
    "execute_test_with_cache",
    "prepare_test",
    # Cache
    "cached_result",
    "cache_result",
    # Suite
    "suite_configs",
    "get_kpi_config",
    # Parameterization
    "pytest_generate_tests",
    "request_fixture",
]
