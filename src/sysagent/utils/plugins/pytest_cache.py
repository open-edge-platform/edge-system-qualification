# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Cache utilities for core testing framework.

This module provides fixtures and functions for caching test results
to improve test performance and avoid redundant test executions.
"""

import logging
import os
from typing import Any, Dict

import pytest

from sysagent.utils.core import TestResultCache

logger = logging.getLogger(__name__)

test_cache = TestResultCache()


@pytest.fixture(scope="function")
def cached_result(request, configs):
    """
    Fixture to get cached test result if available.

    Args:
        request: Pytest request object
        configs: Test configurations

    Returns:
        Dict[str, Any] or None: Cached test result if available
    """

    def _cached_result(cache_configs: Dict[str, Any] = None):
        if os.environ.get("CORE_NO_CACHE") == "1":
            return
        test_name = request.node.name.split("[")[
            0
        ]  # Get base test name without parameters
        return test_cache.retrieve(test_name, configs, cache_configs)

    return _cached_result


@pytest.fixture(scope="function")
def cache_result(request, configs):
    """
    Fixture to cache test result.

    Args:
        request: Pytest request object
        configs: Test configurations

    Returns:
        Callable: Function to cache test result
    """

    def _cache_result(result: Dict[str, Any], cache_configs: Dict[str, Any] = None):
        if os.environ.get("CORE_NO_CACHE") == "1":
            return
        test_name = request.node.name.split("[")[
            0
        ]  # Get base test name without parameters
        test_cache.store(
            test_name=test_name,
            test_configs=configs,
            test_result=result,
            cache_configs=cache_configs,
        )

    return _cache_result
