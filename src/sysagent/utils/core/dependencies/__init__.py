# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Modular dependency management system.

This package provides a comprehensive dependency management system that supports:
- Package-level dependency configuration merging
- Proper status reporting for different dependency types
- Modular validation and installation management
"""
from .manager import DependencyManager, get_dependency_manager
from .schema import (
    DependencyConfig, Dependency, DependencyInstallation,
    DependencyType, OSType, DependencyStatus, DependencyCheckResult
)
from .validator import DependencyValidator
from .loader import DependencyConfigLoader

__all__ = [
    'DependencyManager',
    'get_dependency_manager', 
    'DependencyConfig',
    'Dependency',
    'DependencyInstallation',
    'DependencyType',
    'OSType',
    'DependencyStatus',
    'DependencyCheckResult',
    'DependencyValidator',
    'DependencyConfigLoader'
]
