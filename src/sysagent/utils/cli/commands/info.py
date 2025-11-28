# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System information command implementation.

Handles collecting and displaying comprehensive system information
including hardware and software details with caching support.
"""

import logging
import os

from sysagent.utils.config import setup_data_dir
from sysagent.utils.logging import setup_command_logging

logger = logging.getLogger(__name__)


def run_system_info(verbose: bool = False, debug: bool = False, no_mask: bool = False) -> int:
    """
    Run system information check and display hardware and software details.

    The system information cache is always refreshed when this command is run.

    Args:
        verbose: Whether to show more detailed output
        debug: Whether to show debug level logs
        no_mask: Whether to disable masking of data

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        if no_mask:
            os.environ["CORE_MASK_DATA"] = "false"

        from sysagent.utils.system import SystemInfoCache

        # Setup data directory
        data_dir = setup_data_dir()

        # Configure logging with the helper function
        setup_command_logging("info", verbose=verbose, debug=debug, data_dir=data_dir)

        # Set data directory in environment for tests to use
        os.environ["CORE_DATA_DIR"] = data_dir

        # Initialize system info cache
        logger.debug("Collecting system information...")
        cache_dir = os.path.join(data_dir, "cache")
        system_info_cache = SystemInfoCache(cache_dir)

        # Always refresh the cache when info command is run
        logger.debug("Refreshing system information cache")
        system_info_cache.refresh()

        logger.debug("System information check completed successfully")

        # Generate and display the simple report summary
        report = system_info_cache.generate_simple_report()
        print(report)

        return 0

    except Exception as e:
        logger.error(f"Error running system information check: {e}", exc_info=debug)
        return 1
