# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Main system information collection module.

Provides the primary interface for collecting comprehensive system information
by coordinating hardware and software information collection.
"""

import logging
from typing import Any, Dict

from .hardware import collect_hardware_info
from .software import collect_software_info

logger = logging.getLogger(__name__)


def collect_system_info() -> Dict[str, Any]:
    """
    Collect all system information (hardware and software).

    Returns:
        Dict containing comprehensive system information
    """
    logger.debug("Collecting comprehensive system information")

    system_info = {
        "hardware": collect_hardware_info(),
        "software": collect_software_info(),
    }

    return system_info


# Re-export functions for backward compatibility
__all__ = ["collect_system_info", "collect_hardware_info", "collect_software_info"]
