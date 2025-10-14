# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Logging configuration utilities for the core framework.

This module provides centralized logging configuration and utilities,
including console and file handler management, log level configuration,
and command-specific logging setup.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

# Default logging format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
# Simple message-only format for regular console output
MESSAGE_ONLY_FORMAT = "%(message)s"
DEBUG_MESSAGE_ONLY_FORMAT = "%(message)s"


def get_project_name():
    """Get the project name from configuration."""
    try:
        from sysagent.utils.config import get_project_name as _get_project_name

        return _get_project_name()
    except ImportError:
        return "sysagent"


def setup_data_dir():
    """Setup data directory."""
    try:
        from sysagent.utils.config import setup_data_dir as _setup_data_dir

        return _setup_data_dir()
    except ImportError:
        # Fallback: use CLI-aware project name for proper folder naming
        try:
            from sysagent.utils.config.config_loader import get_cli_aware_project_name

            project_name_tainted = get_cli_aware_project_name()
            project_name = "".join(c for c in project_name_tainted)
        except ImportError:
            project_name_tainted = get_project_name()
            project_name = "".join(c for c in project_name_tainted)

        cwd_tainted = os.getcwd()
        cwd_sanitized = "".join(c for c in cwd_tainted)

        data_dir = os.path.join(cwd_sanitized, f"{project_name}_data")
        os.makedirs(data_dir, exist_ok=True)
        return data_dir


def init_core_logging(debug: bool = False) -> logging.Logger:
    """
    Initialize core logging with basic configuration.

    This sets up the initial logging configuration that will be used
    throughout the application.
    Returns:
        logging.Logger: Configured core logger instance
    """

    # Configure basic logging with environment-aware level setting
    # Use INFO by default for production safety, DEBUG only when explicitly enabled
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=MESSAGE_ONLY_FORMAT,  # Use message-only format for console by default
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Create our logger with appropriate level based on environment
    # The handlers will determine what is actually displayed
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = True  # Ensure messages propagate to root

    # suppress unwanted loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("docker.auth").setLevel(logging.WARNING)
    logging.getLogger("docker.utils").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger


def configure_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Configure logging level based on verbose and debug flags.

    Args:
        verbose: Whether to display more detailed messages
        debug: Whether to display DEBUG level logs

    Note:
        By default, INFO level logs are shown on the console. Verbose mode adds more detail
        while debug mode shows all logs including DEBUG level on the console.
    """
    # Remove existing file handlers to avoid duplicates
    remove_log_handlers()

    # Set root logger level based on environment and parameters
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Create a console handler if one doesn't exist
    console_handler = None
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            console_handler = handler
            break

    if console_handler is None:
        console_handler = logging.StreamHandler(sys.stdout)
        root_logger.addHandler(console_handler)

    # Set the appropriate log level and format for the console handler based on mode
    if debug:
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(DEBUG_MESSAGE_ONLY_FORMAT))
    else:
        # For both default and verbose modes, use message-only format
        console_handler.setFormatter(logging.Formatter(MESSAGE_ONLY_FORMAT))
        if verbose:
            console_handler.setLevel(logging.INFO)
        else:
            console_handler.setLevel(logging.WARNING)


def remove_log_handlers() -> None:
    """
    Remove all file handlers from loggers to avoid duplicates when reconfiguring.
    """
    root_logger = logging.getLogger()
    handlers_to_remove = []

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handlers_to_remove.append(handler)

    for handler in handlers_to_remove:
        root_logger.removeHandler(handler)
        handler.close()


def add_file_log_handler(command: str, data_dir: Optional[str] = None) -> None:
    """
    Add a file handler for logging to a command-specific log file.

    Args:
        command: The command name for the log file
        data_dir: Optional data directory path
    """
    if data_dir is None:
        data_dir = setup_data_dir()

    data_dir_sanitized = "".join(c for c in data_dir)

    logs_dir = os.path.join(data_dir_sanitized, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    from sysagent.utils.config.config_loader import get_cli_aware_project_name

    project_name_tainted = get_cli_aware_project_name()
    project_name = "".join(c for c in project_name_tainted)

    log_file = os.path.join(logs_dir, f"{project_name}_{command}.log")

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode="w")  # Overwrite on each run
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with optional level override.

    Args:
        name: Logger name
        level: Optional logging level override

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    return logger


def set_logger_level(logger_name: str, level: int) -> None:
    """
    Set the logging level for a specific logger.

    Args:
        logger_name: Name of the logger
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)


def suppress_third_party_loggers() -> None:
    """
    Suppress noisy third-party library loggers.
    """
    third_party_loggers = [
        "urllib3",
        "docker.auth",
        "docker.utils",
        "filelock",
        "httpx",
        "requests",
        "transformers",
        "optimum",
        "huggingface_hub",
    ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def create_context_filter(context: Dict[str, Any]) -> logging.Filter:
    """
    Create a logging filter that adds context information to log records.

    Args:
        context: Dictionary of context information to add

    Returns:
        Logging filter that adds context
    """

    class ContextFilter(logging.Filter):
        def filter(self, record):
            for key, value in context.items():
                setattr(record, key, value)
            return True

    return ContextFilter()


def setup_command_logging(
    command: str, verbose: bool = False, debug: bool = False, data_dir: Optional[str] = None
) -> None:
    """
    Set up logging for a specific command with both console and file output.

    This is the recommended function for all CLI command logging setup.
    It provides comprehensive logging configuration including:
    - Core logging initialization
    - Console output level configuration
    - File logging with proper formatting
    - Third-party logger suppression

    Args:
        command: Command name
        verbose: Whether to enable verbose console output
        debug: Whether to enable debug console output
        data_dir: Optional data directory path
    """
    # Initialize core logging
    init_core_logging()

    # Configure console logging level
    configure_logging(verbose=verbose, debug=debug)

    # Add file logging
    add_file_log_handler(command, data_dir)

    # Suppress third-party loggers
    suppress_third_party_loggers()


def log_system_info() -> None:
    """
    Log system information for debugging purposes.
    """
    logger = logging.getLogger(__name__)

    try:
        import platform
        import sys

        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"Platform: {platform.platform()}")
        logger.debug(f"Architecture: {platform.machine()}")
        logger.debug(f"Processor: {platform.processor()}")

        # Log package versions
        try:
            import importlib.metadata

            installed_packages = [dist.metadata["name"] for dist in importlib.metadata.distributions()]
            relevant_packages = [
                pkg
                for pkg in installed_packages
                if any(keyword in pkg.lower() for keyword in ["sysagent", "openvino", "torch", "numpy"])
            ]
            logger.debug(f"Relevant packages: {relevant_packages}")
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"Failed to log system info: {e}")


def create_test_logger(test_name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Create a logger specifically for a test case.

    Args:
        test_name: Name of the test
        log_dir: Optional directory for test logs

    Returns:
        Configured test logger
    """
    if log_dir is None:
        data_dir = setup_data_dir()
        data_dir_sanitized = "".join(c for c in data_dir)
        log_dir = os.path.join(data_dir_sanitized, "logs", "tests")
    else:
        log_dir = "".join(c for c in log_dir)

    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(f"test.{test_name}")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create file handler for this test
    log_file = os.path.join(log_dir, f"{test_name}.log")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))

    logger.addHandler(file_handler)
    logger.propagate = False  # Don't propagate to root logger

    return logger


def close_test_logger(test_name: str) -> None:
    """
    Close and clean up a test logger.

    Args:
        test_name: Name of the test logger to close
    """
    logger = logging.getLogger(f"test.{test_name}")

    # Close all handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def get_log_file_path(command: str, data_dir: Optional[str] = None) -> str:
    """
    Get the path to a command's log file.

    Args:
        command: Command name
        data_dir: Optional data directory path

    Returns:
        Path to the log file
    """
    if data_dir is None:
        data_dir = setup_data_dir()

    data_dir_sanitized = "".join(c for c in data_dir)

    logs_dir = os.path.join(data_dir_sanitized, "logs")
    app_name = get_project_name()
    return os.path.join(logs_dir, f"{app_name}_{command}.log")


def archive_logs(archive_dir: Optional[str] = None, max_archives: int = 5) -> None:
    """
    Archive old log files.

    Args:
        archive_dir: Directory to store archived logs
        max_archives: Maximum number of archives to keep
    """
    import datetime
    import shutil

    data_dir = setup_data_dir()
    data_dir_sanitized = "".join(c for c in data_dir)

    logs_dir = os.path.join(data_dir_sanitized, "logs")

    if archive_dir is None:
        archive_dir = os.path.join(logs_dir, "archive")
    else:
        archive_dir = "".join(c for c in archive_dir)

    os.makedirs(archive_dir, exist_ok=True)

    # Create archive with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"logs_archive_{timestamp}"
    archive_path = os.path.join(archive_dir, archive_name)

    # Copy current logs to archive
    if os.path.exists(logs_dir):
        shutil.copytree(logs_dir, archive_path, dirs_exist_ok=True)

        # Remove old log files from main directory
        for file in os.listdir(logs_dir):
            if file.endswith(".log"):
                os.remove(os.path.join(logs_dir, file))

    # Clean up old archives
    archives = sorted(
        [
            d
            for d in os.listdir(archive_dir)
            if d.startswith("logs_archive_") and os.path.isdir(os.path.join(archive_dir, d))
        ]
    )

    while len(archives) > max_archives:
        old_archive = archives.pop(0)
        shutil.rmtree(os.path.join(archive_dir, old_archive))


def cleanup_logging() -> None:
    """
    Clean up logging handlers on application exit.

    This function should be called when the application is shutting down
    to ensure all log handlers are properly closed and resources are released.
    """
    remove_log_handlers()
