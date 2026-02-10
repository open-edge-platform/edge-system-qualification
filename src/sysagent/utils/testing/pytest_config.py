# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration utilities for the core framework.

This module provides centralized pytest configuration and argument preparation,
including verbosity-specific options, log file setup, and test execution utilities.
"""

import logging
import os
from typing import List, Optional

import pytest

logger = logging.getLogger(__name__)

# Default pytest options for different verbosity levels
DEFAULT_PYTEST_OPTS = [
    "--tb=no",
    "--no-summary",
    "--color=no",
    "--no-header",
    "--capture=tee-sys",  # Enable output capture for log file streaming
    # Note: log_cli is NOT enabled in default mode to keep console silent
    # Logger output goes to log file via the logging file handler
]
VERBOSE_PYTEST_OPTS = [
    "--tb=no",
    "--no-summary",
    "--color=no",
    "--no-header",
    "--capture=tee-sys",
    "-o",
    "log_cli=true",  # Enable live logging for log file and stream capture
    "--log-cli-level=INFO",  # Show INFO level logs
    "--log-cli-format=%(message)s",
]
DEBUG_PYTEST_OPTS = [
    "--tb=short",
    "-v",
    "--capture=tee-sys",
    "--log-level=DEBUG",
    "-o",
    "log_cli=true",  # Enable live logging for log file and stream capture
    "--log-cli-level=DEBUG",  # Show DEBUG level logs
    "--log-cli-format=%(message)s",
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

    # Initialize extra_args if None
    if extra_args is None:
        extra_args = []

    # Base pytest arguments
    pytest_args = [
        "-v",  # Always use verbose pytest output
        f"--alluredir={os.path.join(data_dir, 'results', 'allure')}",
    ]

    # Determine verbosity level for terminal output
    if debug:
        pytest_args.extend(DEBUG_PYTEST_OPTS)
    elif verbose:
        pytest_args.extend(VERBOSE_PYTEST_OPTS)
    else:
        pytest_args.extend(DEFAULT_PYTEST_OPTS)

    # Note: We don't use --log-file here because:
    # 1. When running in isolated venv with stream_output=True, log_cli output
    #    is captured by the process utility and written to the main run log
    # 2. Using --log-file would cause redundant logging (same messages twice)
    # 3. The log_cli feature (enabled via -o log_cli=true in DEBUG_PYTEST_OPTS)
    #    provides all the logging output we need through console capture

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

    Temporarily suppresses root logger console handler to avoid redundant output
    with pytest's log_cli feature.

    Args:
        pytest_args: List of pytest arguments

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    import sys

    # Save and remove root logger console handlers to avoid redundancy with pytest log_cli
    root_logger = logging.getLogger()
    saved_handlers = []
    handlers_to_remove = []

    for handler in root_logger.handlers[:]:  # Copy list to avoid modification during iteration
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
            saved_handlers.append((handler, handler.level))
            handlers_to_remove.append(handler)

    # Remove console handlers temporarily
    for handler in handlers_to_remove:
        root_logger.removeHandler(handler)

    try:
        # Run pytest with log_cli handling console output
        exit_code = pytest.main(pytest_args)
    finally:
        # Restore console handlers
        for handler, level in saved_handlers:
            handler.setLevel(level)
            root_logger.addHandler(handler)

    return exit_code


def run_pytest_with_venv(
    pytest_args: List[str],
    suite_path: str,
    requirements_file: str,
    data_dir: str,
    python_version: Optional[str] = None,
    force: bool = False,
    timeout: Optional[float] = None,
) -> int:
    """
    Execute pytest in an isolated virtual environment.

    This function sets up an isolated venv for the test suite and runs
    pytest using that environment's Python interpreter.

    Args:
        pytest_args: List of pytest arguments
        suite_path: Path to test suite (e.g., "esq/suites/ai/gen")
        requirements_file: Path to requirements.txt file
        data_dir: Base data directory for venv storage
        python_version: Optional Python version for the venv
        force: Whether to recreate venv if it already exists
        timeout: Maximum execution time in seconds (default: 7200 = 2 hours)

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        from sysagent.utils.core.venv import run_pytest_in_suite_venv, setup_suite_venv

        # Setup the isolated venv
        success, venv_name, message = setup_suite_venv(
            suite_path=suite_path,
            requirements_file=requirements_file,
            data_dir=data_dir,
            python_version=python_version,
            force=force,
        )

        if not success:
            logger.error(f"Failed to setup venv: {message}")
            return 1

        logger.info(f"Using isolated venv: {venv_name}")

        # When running in venv (subprocess), we need to enable log_cli so that
        # logger statements are output to stdout where they can be captured
        # by the parent process and logged to file. This is different from
        # non-venv mode where log_cli would show on user's console.
        # In venv mode, the subprocess output is captured and logged by parent.
        venv_pytest_args = pytest_args.copy()

        # Check if log_cli is already configured
        has_log_cli = any(
            "-o" in arg and i + 1 < len(venv_pytest_args) and "log_cli" in venv_pytest_args[i + 1]
            for i, arg in enumerate(venv_pytest_args)
        )

        if not has_log_cli:
            # Add log_cli configuration for subprocess to output logger statements
            # Use DEBUG level to capture all logger output from test code
            venv_pytest_args.extend(
                [
                    "-o",
                    "log_cli=true",
                    "--log-cli-level=DEBUG",
                    "--log-cli-format=%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
                ]
            )
            logger.debug("Added log_cli configuration for venv subprocess execution")

        # Run pytest in the isolated venv with configured timeout
        # Default to 7200 seconds (2 hours) if not specified
        venv_timeout = timeout if timeout is not None else 7200.0
        logger.debug(f"Running pytest in venv with timeout: {venv_timeout}s")

        return run_pytest_in_suite_venv(
            suite_path=suite_path,
            requirements_file=requirements_file,
            pytest_args=venv_pytest_args,
            data_dir=data_dir,
            timeout=venv_timeout,
        )
    except Exception as e:
        logger.error(f"Error running pytest with venv: {e}")
        return 1


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
