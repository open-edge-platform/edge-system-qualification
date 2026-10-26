# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Modular dependency management system.

This package provides a comprehensive dependency management system that supports:
- Package-level dependency configuration merging
- Proper status reporting for different dependency types
- Modular validation and installation management
"""

from .loader import DependencyConfigLoader
from .manager import DependencyManager, get_dependency_manager
from .prerun import run_prerun_checks
from .schema import (
    Dependency,
    DependencyCheckResult,
    DependencyConfig,
    DependencyInstallation,
    DependencyStatus,
    DependencyType,
    OSType,
)
from .validator import DependencyValidator

__all__ = [
    "Dependency",
    "DependencyCheckResult",
    "DependencyConfig",
    "DependencyConfigLoader",
    "DependencyInstallation",
    "DependencyManager",
    "DependencyStatus",
    "DependencyType",
    "DependencyValidator",
    "OSType",
    "get_dependency_manager",
    "run_prerun_checks",
]
