# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for test parameterization and dynamic test generation.
"""

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import pytest

logger = logging.getLogger(__name__)


def generate_test_id(configs: Dict[str, Any]) -> str:
    """
    Generate a test ID from a test configurations.

    Args:
        configs: Test configurations dictionary

    Returns:
        str: A human-readable test ID
    """
    # Use display_name if available, otherwise use other parameters
    if "display_name" in configs:
        return configs["display_name"]

    parts = []
    # Include important parameters in ID
    for key in ["duration", "timeout", "mode"]:
        if key in configs:
            parts.append(f"{key}={configs[key]}")

    return "-".join(parts) if parts else "default"


def parameterize_test(
    params: List[Dict[str, Any]],
    test_func: Optional[Callable] = None,
    id_func: Optional[Callable[[Dict[str, Any]], str]] = None,
    metadata_func: Optional[Callable[[Dict[str, Any], str], None]] = None,
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    Decorator to parameterize tests with configuration dictionaries.

    This decorator wraps pytest.mark.parametrize to provide enhanced functionality:
    - Automatic test ID generation
    - Optional custom metadata injection
    - Configuration validation

    Args:
        params: List of parameter dictionaries for the test
        test_func: Test function to parameterize (when used without parentheses)
        id_func: Optional function to generate test IDs
        metadata_func: Optional function to apply test metadata

    Returns:
        Decorated test function or decorator function

    Examples:
        # As decorator with parameters
        @parameterize_test([
            {"duration": 60, "display_name": "short_test"},
            {"duration": 300, "display_name": "long_test"}
        ])
        def test_example(config):
            pass

        # As decorator without parameters (uses defaults)
        @parameterize_test
        def test_example(config):
            pass
    """

    def decorator(func: Callable) -> Callable:
        # Use provided functions or defaults
        actual_id_func = id_func or generate_test_id
        actual_metadata_func = metadata_func

        # Generate test IDs
        test_ids = [actual_id_func(config) for config in params]

        # Apply pytest parameterization
        parameterized_func = pytest.mark.parametrize("config", params, ids=test_ids)(
            func
        )

        @functools.wraps(parameterized_func)
        def wrapped_func(config, *args, **kwargs):
            try:
                # Apply metadata before test execution if metadata function is provided
                if actual_metadata_func:
                    actual_metadata_func(config, func.__name__)

                # Call the original test function
                return parameterized_func.__wrapped__(config, *args, **kwargs)
            except Exception as e:
                logger.error(f"Test execution failed for config {config}: {e}")
                raise

        # Preserve parameterization marks
        if hasattr(parameterized_func, "pytestmark"):
            wrapped_func.pytestmark = parameterized_func.pytestmark

        return wrapped_func

    # Handle being called with or without parameters
    if test_func is not None:
        # Called as @parameterize_test (without parentheses)
        # In this case, params should actually be the test function
        # and we need to extract params from somewhere else
        if callable(params):
            # This means it was called as @parameterize_test without params
            # We'll need to get params from the function or use defaults
            func = params  # params is actually the function
            # Extract params from function annotations, docstring, or use defaults
            # actual_params = getattr(func, "_test_params", [{}])
            return decorator(func)
        else:
            # Called as @parameterize_test([...])
            return decorator(test_func)
    else:
        # Called as @parameterize_test([...])
        return decorator
