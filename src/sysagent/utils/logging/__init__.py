# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Logging utilities package.

Provides utilities for logging configuration, console and file handlers,
test logging, performance logging, and log management.
"""

# Import local modules directly without backward compatibility
from .logging_config import *

# Re-export all functions and classes
__all__ = [
    # Logging configuration
    'init_core_logging',
    'configure_logging',
    'setup_command_logging',
    
    # Handler management
    'add_file_log_handler',
    'remove_log_handlers',
    'cleanup_logging',
    
    # Logger utilities
    'get_logger',
    'set_logger_level',
    'suppress_third_party_loggers',
    'create_context_filter',
    
    # Test logging
    'create_test_logger',
    'close_test_logger',
    
    # System and utility functions
    'log_system_info',
    'get_log_file_path',
    'archive_logs',
    
    # Constants
    'DEFAULT_LOG_FORMAT',
    'MESSAGE_ONLY_FORMAT',
    'DEBUG_MESSAGE_ONLY_FORMAT',
]
