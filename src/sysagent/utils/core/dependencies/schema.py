# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Enhanced dependency management schema with improved status handling.

This module defines the structure for comprehensive dependency management
that supports complex installation requirements, proper status reporting,
and package-level configuration merging.
"""

from dataclasses import dataclass, field
from enum import Enum


class DependencyType(Enum):
    """Types of dependencies supported."""

    SYSTEM_PACKAGE = "system_package"
    BINARY_DOWNLOAD = "binary_download"
    SCRIPT_INSTALL = "script_install"
    MANUAL_SETUP = "manual_setup"
    DOCKER_SETUP = "docker_setup"
    # Runtime sanity/smoke check: verifies an already-detected component is
    # actually functional (e.g. an OpenVINO device can compile/run inference,
    # not just that it is reported as "available"). Unlike the other types,
    # this is not an installable dependency - it never produces install
    # instructions and is only evaluated by prerun checks, not on every CLI
    # command.
    FUNCTIONAL_CHECK = "functional_check"


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
    # Passed a functional_check (runtime smoke test) - distinct from INSTALLED
    # since "installed" is meaningless for e.g. a network reachability check.
    # Value is "ready" (not "functional") to read naturally in readiness-check output.
    FUNCTIONAL = "ready"


@dataclass
class InstallCommand:
    """Represents a single installation command."""

    command: str
    description: str
    sudo_required: bool = False
    environment_vars: dict[str, str] = field(default_factory=dict)
    working_directory: str | None = None
    expected_output: str | None = None
    retry_count: int = 1


@dataclass
class PostInstallStep:
    """Represents a post-installation step."""

    name: str
    description: str
    commands: list[InstallCommand] = field(default_factory=list)
    validation_command: str | None = None
    skip_if_exists: str | None = None


@dataclass
class DependencyValidation:
    """Validation criteria for a dependency."""

    check_command: str = ""
    expected_return_code: int = 0
    expected_output_contains: str | None = None
    check_file_exists: str | None = None
    check_service_running: str | None = None
    check_commands: list[str] = field(default_factory=list)
    check_method: str | None = None  # "dpkg", "rpm", "command", "network", "openvino_device_smoke", etc.
    packages_check: list[str] = field(default_factory=list)
    # Fields used by functional_check validations (check_method-specific)
    devices: list[str] = field(default_factory=list)  # e.g. OpenVINO device ids; empty = auto-detect
    hosts: list[str] = field(default_factory=list)  # e.g. hostnames for network reachability checks
    # Download-source-dependent hosts for the "network" check_method, e.g.
    # {"model_download": {"huggingface": ["huggingface.co"], "modelscope": ["modelscope.cn"]}}.
    # Only the host(s) for whichever source is actually preferred (see
    # get_preferred_download_source) are checked, alongside the static `hosts` above.
    host_alternatives: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    image: str | None = None  # e.g. container image pulled/run by "docker_runtime_smoke"
    port: int | None = None
    timeout: float | None = None


@dataclass
class DependencyInstallation:
    """Installation instructions for a specific OS."""

    os_type: OSType
    install_method: DependencyType
    commands: list[InstallCommand] = field(default_factory=list)
    post_install_steps: list[PostInstallStep] = field(default_factory=list)
    validation: DependencyValidation | None = None
    notes: list[str] = field(default_factory=list)
    documentation_url: str | None = None
    packages: list[str] = field(default_factory=list)


@dataclass
class Dependency:
    """Complete dependency definition."""

    name: str
    description: str
    required: bool = True
    priority: int = 100
    installations: list[DependencyInstallation] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)
    min_version: str | None = None
    max_version: str | None = None
    package_source: str | None = None  # Which package this came from
    # Restricts a functional_check dependency to runs where this profile requirement
    # flag (e.g. "docker_required") is active; None (default) means always run it.
    requires_flag: str | None = None


@dataclass
class DependencyGroup:
    """Group of related dependencies."""

    name: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    optional: bool = False


@dataclass
class DependencyConfig:
    """Complete dependency configuration."""

    version: str = "1.0"
    dependencies: dict[str, Dependency] = field(default_factory=dict)
    groups: dict[str, DependencyGroup] = field(default_factory=dict)
    install_order: list[str] = field(default_factory=list)
    package_source: str | None = None  # Which package this config came from


@dataclass
class DependencyCheckResult:
    """Result of a dependency check."""

    name: str
    status: DependencyStatus
    message: str
    details: str | None = None
    missing_packages: list[str] = field(default_factory=list)
    total_packages: int = 0
    installed_packages: int = 0
