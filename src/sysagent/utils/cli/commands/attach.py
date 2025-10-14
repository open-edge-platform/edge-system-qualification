# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Attachment commands implementation.

Handles attaching logs, summaries, and system information to Allure reports
with proper isolation and error handling.
"""

import logging
import os
from pathlib import Path

from sysagent.utils.config import setup_data_dir
from sysagent.utils.config.config_loader import discover_entrypoint_paths
from sysagent.utils.core import shared_state
from sysagent.utils.logging import setup_command_logging
from sysagent.utils.testing import (
    add_test_paths_to_args,
    create_pytest_args,
    run_pytest,
    validate_pytest_args,
)

logger = logging.getLogger(__name__)


def _find_core_cli_test_file(test_filename: str) -> str:
    """
    Dynamically find the core CLI test file

    Args:
        test_filename: Name of the test file to find (e.g., "test_log.py")

    Returns:
        str: Absolute path to the test file

    Raises:
        FileNotFoundError: If the test file cannot be found
    """
    # Use the same discovery logic as the config system
    suites_paths = discover_entrypoint_paths("suites")

    for suites_path in suites_paths:
        # Look for core/cli/{test_filename} in each suites directory
        test_path = suites_path / "core" / "cli" / test_filename
        if test_path.exists():
            logger.debug(f"Found core CLI test file: {test_path}")
            return str(test_path)

    # Fallback to explicit search if discovery fails
    logger.warning(f"Test file {test_filename} not found via discovery, trying fallback search")

    # Check current working directory structure
    cwd = Path.cwd()
    fallback_paths = [
        cwd / "src" / "sysagent" / "suites" / "core" / "cli" / test_filename,
        cwd / "suites" / "core" / "cli" / test_filename,
    ]

    for fallback_path in fallback_paths:
        if fallback_path.exists():
            logger.debug(f"Found core CLI test file via fallback: {fallback_path}")
            return str(fallback_path)

    raise FileNotFoundError(f"Core CLI test file not found: {test_filename}")


def attach_logs(verbose: bool = False, debug: bool = False) -> int:
    """
    Attach CLI log files to Allure report by running the CLI log attachment test.

    This command runs independently after test execution to collect all log files
    generated during CLI operations and attach them to the Allure report for debugging.
    Uses completely isolated logging to avoid interfering with existing logs.

    This function is designed to be interrupt-resilient - it will reset any previous
    interrupt states and execute regardless of previous keyboard interrupts during
    test execution.

    Args:
        verbose: Whether to show more detailed output
        debug: Whether to show debug level logs

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        test_path = _find_core_cli_test_file("test_log.py")
        return _run_attachment_test(
            test_path=test_path,
            operation_name="attach_logs",
            check_dir="logs",
            verbose=verbose,
            debug=debug,
        )
    except FileNotFoundError as e:
        logger.error(f"Failed to find core CLI logs test: {e}")
        return 1


def attach_summaries(verbose: bool = False, debug: bool = False) -> int:
    """
    Attach CLI summary JSON files to Allure report by running the CLI summary test.

    This command runs independently after test execution to collect all summary files
    generated during CLI operations and attach them to the Allure report for analysis.
    Uses completely isolated logging to avoid interfering with existing logs.

    This function is designed to be interrupt-resilient - it will reset any previous
    interrupt states and execute regardless of previous keyboard interrupts during
    test execution.

    Args:
        verbose: Whether to show more detailed output
        debug: Whether to show debug level logs

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        test_path = _find_core_cli_test_file("test_summary.py")
        return _run_attachment_test(
            test_path=test_path,
            operation_name="attach_summaries",
            check_dir="results/core",
            verbose=verbose,
            debug=debug,
        )
    except FileNotFoundError as e:
        logger.error(f"Failed to find core CLI summary test: {e}")
        return 1


def attach_system(verbose: bool = False, debug: bool = False) -> int:
    """
    Attach system information to Allure report by running the system information test.

    This command runs independently after test execution to collect comprehensive system
    information (hardware, software, environment) and attach it to the Allure report for
    analysis and debugging. Uses completely isolated logging to avoid interfering with
    existing logs.

    This function is designed to be interrupt-resilient - it will reset any previous
    interrupt states and execute regardless of previous keyboard interrupts during
    test execution.

    Args:
        verbose: Whether to show more detailed output
        debug: Whether to show debug level logs

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        test_path = _find_core_cli_test_file("test_system.py")
        return _run_attachment_test(
            test_path=test_path,
            operation_name="attach_system",
            check_dir=None,
            verbose=verbose,
            debug=debug,
        )
    except FileNotFoundError as e:
        logger.error(f"Failed to find core CLI system test: {e}")
        return 1


def _run_attachment_test(
    test_path: str,
    operation_name: str,
    check_dir: str = None,
    verbose: bool = False,
    debug: bool = False,
) -> int:
    """
    Run a specific attachment test with proper isolation and cleanup.

    Args:
        test_path: Path to the test file to run
        operation_name: Name of the operation for logging
        check_dir: Directory to check for existence (relative to data_dir)
        verbose: Whether to show verbose output
        debug: Whether to show debug output

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    # Store original file handlers to restore later
    original_file_handlers = []
    try:
        previous_interrupt_state = shared_state.INTERRUPT_OCCURRED
        if previous_interrupt_state:
            logger.debug(
                f"Previous interrupt detected ({shared_state.INTERRUPT_SIGNAL_NAME}), "
                f"but resetting state for {operation_name}"
            )

        # Reset interrupt flags to allow attachment to proceed
        shared_state.INTERRUPT_OCCURRED = False
        shared_state.INTERRUPT_SIGNAL = None
        shared_state.INTERRUPT_SIGNAL_NAME = "Unknown"

        # Setup data directory and environment
        data_dir = setup_data_dir()
        os.environ["CORE_DATA_DIR"] = data_dir
        os.environ["CORE_ATTACHMENT_MODE"] = "1"
        os.environ["ACTIVE_PROFILE"] = "core.cli"
        if "ACTIVE_PROFILE_HIGHEST_TIER" in os.environ:
            del os.environ["ACTIVE_PROFILE_HIGHEST_TIER"]

        # Save and remove existing file handlers to prevent contamination
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:  # Use slice copy to avoid modification during iteration
            if isinstance(handler, logging.FileHandler):
                original_file_handlers.append(handler)
                root_logger.removeHandler(handler)

        # Set up completely isolated logging for this operation
        setup_command_logging(operation_name, verbose=verbose, debug=debug)

        # Check if required directory exists (if specified)
        if check_dir:
            check_path = os.path.join(data_dir, check_dir)
            if not os.path.exists(check_path):
                logger.warning(f"No {check_dir} directory found at: {check_path}")
                return 0

        # Create pytest arguments specifically for the attachment test
        pytest_args = create_pytest_args(data_dir, verbose, debug, [])

        # Add the specific attachment test path
        if os.path.exists(test_path):
            pytest_args = add_test_paths_to_args(pytest_args, [test_path])
        else:
            logger.error(f"{operation_name} test not found: {test_path}")
            return 1

        # Validate pytest arguments
        if not validate_pytest_args(pytest_args):
            logger.error(f"No valid tests found for {operation_name}")
            return 1

        logger.debug(f"Running {operation_name} test")
        logger.debug(f"Pytest arguments: {pytest_args}")

        # Run the attachment test
        result_code = run_pytest(pytest_args)

        if result_code == 0:
            logger.debug(f"{operation_name} completed successfully")
        else:
            logger.error(f"{operation_name} failed")

        return result_code

    except KeyboardInterrupt:
        logger.warning(f"{operation_name} interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Error during {operation_name}: {e}", exc_info=debug)
        return 1
    finally:
        # Remove attachment file handlers to clean up
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:  # Use slice copy
            if isinstance(handler, logging.FileHandler):
                handler.close()
                root_logger.removeHandler(handler)

        # Restore original file handlers to prevent disrupting other operations
        for handler in original_file_handlers:
            root_logger.addHandler(handler)

        # Clean up environment variables
        if "CORE_ATTACHMENT_MODE" in os.environ:
            del os.environ["CORE_ATTACHMENT_MODE"]
        if "ACTIVE_PROFILE" in os.environ:
            del os.environ["ACTIVE_PROFILE"]
        if "ACTIVE_PROFILE_HIGHEST_TIER" in os.environ:
            del os.environ["ACTIVE_PROFILE_HIGHEST_TIER"]
