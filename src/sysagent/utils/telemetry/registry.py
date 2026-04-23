# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Telemetry module registry.

Provides a global registry that maps module name strings (as used in profile YAML)
to their corresponding BaseTelemetryModule subclass implementations.

Core modules (cpu_freq, cpu_usage, memory_usage) are registered by sysagent.
Package-level modules (e.g., package_power from esq) register themselves on import.
"""

import logging
from typing import Dict, Optional, Type

from sysagent.utils.telemetry.base import BaseTelemetryModule

logger = logging.getLogger(__name__)

# Global module registry: module_name -> class
_REGISTRY: Dict[str, Type[BaseTelemetryModule]] = {}


def register(name: str, cls: Type[BaseTelemetryModule]) -> None:
    """
    Register a telemetry module class under the given name.

    Args:
        name: The module identifier used in profile YAML ``modules[].name``.
        cls: The BaseTelemetryModule subclass to register.
    """
    if name in _REGISTRY:
        logger.debug("Telemetry module '%s' already registered; overwriting with %s", name, cls.__name__)
    _REGISTRY[name] = cls
    logger.debug("Registered telemetry module: '%s' -> %s", name, cls.__name__)


def get(name: str) -> Optional[Type[BaseTelemetryModule]]:
    """
    Look up a telemetry module class by name.

    Args:
        name: The module identifier.

    Returns:
        The registered class, or None if not found.
    """
    return _REGISTRY.get(name)


def list_modules() -> Dict[str, Type[BaseTelemetryModule]]:
    """Return a copy of the current registry."""
    return dict(_REGISTRY)
