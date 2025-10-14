# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Enhanced dependency management schema with improved status handling.

This module defines the structure for comprehensive dependency management
that supports complex installation requirements, proper status reporting,
and package-level configuration merging.
"""
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class DependencyType(Enum):
    """Types of dependencies supported."""
    SYSTEM_PACKAGE = "system_package"
    BINARY_DOWNLOAD = "binary_download"
    SCRIPT_INSTALL = "script_install"
    MANUAL_SETUP = "manual_setup"
    DOCKER_SETUP = "docker_setup"


class OSType(Enum):
    """Supported operating systems."""
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"


class DependencyStatus(Enum):
    """Status of dependency installation/availability."""
    INSTALLED = "installed"
    MISSING = "missing"
    MANUAL_REQUIRED = "manual_required"
    NEEDS_CONFIGURATION = "needs_configuration"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class InstallCommand:
    """Represents a single installation command."""
    command: str
    description: str
    sudo_required: bool = False
    environment_vars: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None
    expected_output: Optional[str] = None
    retry_count: int = 1


@dataclass
class PostInstallStep:
    """Represents a post-installation step."""
    name: str
    description: str
    commands: List[InstallCommand] = field(default_factory=list)
    validation_command: Optional[str] = None
    skip_if_exists: Optional[str] = None


@dataclass
class DependencyValidation:
    """Validation criteria for a dependency."""
    check_command: str = ""
    expected_return_code: int = 0
    expected_output_contains: Optional[str] = None
    check_file_exists: Optional[str] = None
    check_service_running: Optional[str] = None
    check_commands: List[str] = field(default_factory=list)
    check_method: Optional[str] = None  # "dpkg", "rpm", "command", etc.
    packages_check: List[str] = field(default_factory=list)


@dataclass
class DependencyInstallation:
    """Installation instructions for a specific OS."""
    os_type: OSType
    install_method: DependencyType
    commands: List[InstallCommand] = field(default_factory=list)
    post_install_steps: List[PostInstallStep] = field(default_factory=list)
    validation: Optional[DependencyValidation] = None
    notes: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    packages: List[str] = field(default_factory=list)


@dataclass
class Dependency:
    """Complete dependency definition."""
    name: str
    description: str
    required: bool = True
    priority: int = 100
    installations: List[DependencyInstallation] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    package_source: Optional[str] = None  # Which package this came from


@dataclass
class DependencyGroup:
    """Group of related dependencies."""
    name: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    optional: bool = False


@dataclass
class DependencyConfig:
    """Complete dependency configuration."""
    version: str = "1.0"
    dependencies: Dict[str, Dependency] = field(default_factory=dict)
    groups: Dict[str, DependencyGroup] = field(default_factory=dict)
    install_order: List[str] = field(default_factory=list)
    package_source: Optional[str] = None  # Which package this config came from


@dataclass
class DependencyCheckResult:
    """Result of a dependency check."""
    name: str
    status: DependencyStatus
    message: str
    details: Optional[str] = None
    missing_packages: List[str] = field(default_factory=list)
    total_packages: int = 0
    installed_packages: int = 0
