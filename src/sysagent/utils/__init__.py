# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utility modules for the core framework.

This package has been reorganized into focused sub-packages for better maintainability:
- core: Core framework utilities (cache, result, kpi, shared_state)
- system: System information and hardware detection
- reporting: Report generation (allure, summary, visualization)
- testing: Testing utilities and validation
- infrastructure: Infrastructure management (docker, node, dependencies)
- config: Configuration management
- logging: Logging configuration and utilities

Note: Model-specific utilities have been moved to the esq package (esq.utils.models)
to keep sysagent generic and not AI-specific.

All utilities are now available through their respective hierarchical packages.
"""

# # Import from reorganized packages directly (no backward compatibility)
# from .core import *
# from .system import *
# from .reporting import *
# from .testing import *
# from .infrastructure import *
# from .config import *
# from .logging import *

# Individual module imports for explicit access
from . import config, core, infrastructure, logging, reporting, system, testing

# Export all available utilities
__all__ = [
    # Sub-packages
    "core",
    "system",
    "reporting",
    "testing",
    "infrastructure",
    "config",
    "logging",
]
