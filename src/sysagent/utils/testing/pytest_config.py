# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration utilities for the core framework.

This module provides centralized pytest configuration and argument preparation,
including verbosity-specific options, log file setup, and test execution utilities.
"""

import os
from typing import List, Optional

import pytest

# Default pytest options for different verbosity levels
DEFAULT_PYTEST_OPTS = ["--tb=no", "--no-summary", "--color=no", "--no-header"]
VERBOSE_PYTEST_OPTS = [
    "--tb=no",
    "--no-summary",
    "--color=no",
    "--no-header",
    "--capture=tee-sys",
]
DEBUG_PYTEST_OPTS = [
    "--tb=short",
    "-v",
    "--capture=tee-sys",
    "--log-level=DEBUG",
    "--log-cli-level=DEBUG",
]


def create_pytest_args(
    data_dir: str,
    verbose: bool = False,
    debug: bool = False,
    extra_args: Optional[List[str]] = None,
    log_file_suffix: str = "test",
) -> List[str]:
    """
    Create standardized pytest arguments for test execution.

    Args:
        data_dir: Directory to store test data and logs
        verbose: Whether to enable verbose output
        debug: Whether to enable debug output
        extra_args: Additional pytest arguments to include
        log_file_suffix: Suffix for the log file name (e.g., "test", "profile_name")

    Returns:
        List[str]: Configured pytest arguments
    """
    from sysagent.utils.config import get_project_name

    # Initialize extra_args if None
    if extra_args is None:
        extra_args = []

    # Base pytest arguments
    pytest_args = [
        "-v",  # Always use verbose pytest output
        f"--alluredir={os.path.join(data_dir, 'results', 'allure')}",
        "-o",
        "addopts=-p no:logging",  # Override pytest.ini addopts to remove marker filters
    ]

    # Determine verbosity level for terminal output
    if debug:
        pytest_args.extend(DEBUG_PYTEST_OPTS)
    elif verbose:
        pytest_args.extend(VERBOSE_PYTEST_OPTS)
    else:
        pytest_args.extend(DEFAULT_PYTEST_OPTS)

    # Configure log file
    app_name = get_project_name()
    log_file = f"{app_name}_{log_file_suffix}.log"
    logs_dir = os.path.join(data_dir, "logs")
    log_file_path = os.path.join(logs_dir, log_file)

    # Ensure logs directory exists
    os.makedirs(logs_dir, exist_ok=True)

    # Add logging arguments
    pytest_args.extend(
        [
            f"--log-file={log_file_path}",
            "--log-file-level=DEBUG",
            "--log-file-format=%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        ]
    )

    # Add extra arguments
    pytest_args.extend(extra_args)

    return pytest_args


def run_pytest_with_args(
    test_paths: List[str],
    data_dir: str,
    verbose: bool = False,
    debug: bool = False,
    extra_args: Optional[List[str]] = None,
    log_file_suffix: str = "test",
) -> int:
    """
    Run pytest with standardized arguments.

    Args:
        test_paths: List of test file/directory paths
        data_dir: Directory to store test data and logs
        verbose: Whether to enable verbose output
        debug: Whether to enable debug output
        extra_args: Additional pytest arguments to include
        log_file_suffix: Suffix for the log file name

    Returns:
        int: Pytest exit code
    """
    pytest_args = create_pytest_args(
        data_dir=data_dir,
        verbose=verbose,
        debug=debug,
        extra_args=extra_args,
        log_file_suffix=log_file_suffix,
    )

    # Combine test paths with pytest arguments
    full_args = test_paths + pytest_args

    return pytest.main(full_args)


def get_pytest_markers() -> List[str]:
    """
    Get list of available pytest markers.

    Returns:
        List[str]: Available pytest markers
    """
    return [
        "smoke",  # Smoke tests
        "regression",  # Regression tests
        "performance",  # Performance tests
        "stability",  # Stability tests
        "compatibility",  # Compatibility tests
        "integration",  # Integration tests
        "unit",  # Unit tests
        "slow",  # Slow running tests
        "fast",  # Fast running tests
        "gpu",  # GPU-related tests
        "cpu",  # CPU-related tests
        "npu",  # NPU-related tests
    ]


def filter_by_marker(marker: str) -> str:
    """
    Create pytest marker filter expression.

    Args:
        marker: Marker name to filter by

    Returns:
        str: Pytest marker expression
    """
    return f"-m {marker}"


def exclude_marker(marker: str) -> str:
    """
    Create pytest marker exclusion expression.

    Args:
        marker: Marker name to exclude

    Returns:
        str: Pytest marker exclusion expression
    """
    return f"-m 'not {marker}'"


def create_marker_expression(
    include_markers: Optional[List[str]] = None,
    exclude_markers: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Create complex pytest marker expression.

    Args:
        include_markers: List of markers to include
        exclude_markers: List of markers to exclude

    Returns:
        str: Complex marker expression, or None if no markers specified
    """
    expressions = []

    if include_markers:
        include_expr = " or ".join(include_markers)
        expressions.append(f"({include_expr})")

    if exclude_markers:
        exclude_expr = " and ".join(f"not {marker}" for marker in exclude_markers)
        expressions.append(f"({exclude_expr})")

    if expressions:
        return " and ".join(expressions)

    return None


class PytestConfigurationError(Exception):
    """Exception raised for pytest configuration errors."""

    pass


def create_profile_pytest_args(
    data_dir: str, profile_name: str, verbose: bool = False, debug: bool = False, live_logging: bool = False
) -> List[str]:
    """
    Create pytest arguments specifically for profile execution.

    Args:
        data_dir: Directory to store test data and logs
        profile_name: Name of the profile being executed
        verbose: Whether to enable verbose output
        debug: Whether to enable debug output
        live_logging: Whether to enable live console logging during test execution

    Returns:
        List[str]: Configured pytest arguments for profile execution
    """
    # Base pytest arguments for profile
    pytest_args = [
        "-v",  # Always use verbose pytest output
        f"--alluredir={os.path.join(data_dir, 'results', 'allure')}",
        "-o",
        "addopts=-p no:logging",  # Override pytest.ini addopts to remove marker filters
    ]

    # Add verbosity-specific options
    if debug:
        pytest_args.extend(DEBUG_PYTEST_OPTS)
        if live_logging:
            pytest_args.append("--log-cli-level=DEBUG")
    elif verbose:
        pytest_args.extend(VERBOSE_PYTEST_OPTS)
        if live_logging:
            pytest_args.append("--log-cli-level=INFO")
    else:
        pytest_args.extend(DEFAULT_PYTEST_OPTS)
        if live_logging:
            pytest_args.append("--log-cli-level=INFO")  # Still log at INFO level

    return pytest_args


def run_pytest(pytest_args: List[str]) -> int:
    """
    Execute pytest with the given arguments.

    Args:
        pytest_args: List of pytest arguments

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    return pytest.main(pytest_args)


def cleanup_pytest_cache() -> None:
    """
    Clean up pytest cache to ensure fresh test runs.

    This function cleans up pytest's internal state and file system cache
    to avoid test collection issues between profile runs.
    """
    import shutil

    try:
        # Clear pytest's internal cache to avoid test collection issues
        if hasattr(pytest, "_cleanup"):
            pytest._cleanup()

        # Clear pytest's file system cache
        if os.path.exists(".pytest_cache"):
            shutil.rmtree(".pytest_cache")
    except Exception:
        # Ignore cleanup errors - they're not critical
        pass


def validate_pytest_args(pytest_args: List[str]) -> bool:
    """
    Validate that pytest arguments contain at least one test to run.

    Args:
        pytest_args: List of pytest arguments

    Returns:
        bool: True if there are tests to run, False otherwise
    """
    # Check if we have any tests to run (non-option arguments)
    return any(arg for arg in pytest_args if not arg.startswith("-"))


def add_test_paths_to_args(pytest_args: List[str], test_paths: List[str]) -> List[str]:
    """
    Add test paths to pytest arguments.

    Args:
        pytest_args: Existing pytest arguments
        test_paths: List of test file or directory paths to add

    Returns:
        List[str]: Updated pytest arguments with test paths
    """
    result_args = pytest_args.copy()
    result_args.extend(test_paths)
    return result_args


def configure_pytest_environment(data_dir: str, verbose: bool = False, debug: bool = False) -> List[str]:
    """
    Configure pytest environment with appropriate settings.

    This is a convenience function that sets up standard pytest configuration
    for test execution.

    Args:
        data_dir: Directory for test data and logs
        verbose: Enable verbose output
        debug: Enable debug output

    Returns:
        List of pytest arguments
    """
    return create_pytest_args(data_dir, verbose=verbose, debug=debug)
