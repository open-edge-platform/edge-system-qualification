# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Software information collection utilities.

Provides functions to collect detailed software information about the system,
including operating system, Python environment, and installed packages.
"""

import logging
import os
import platform
import sys
from typing import Any, Dict, List, Optional

import yaml

# Import secure process execution utilities
from ..core.process import run_command

logger = logging.getLogger(__name__)


def _load_software_config() -> Dict[str, List[str]]:
    """
    Load software configuration from available config files.

    Returns:
        Dict containing consolidated python_packages and system_packages lists
    """
    consolidated_config = {"python_packages": [], "system_packages": []}

    try:
        # Import config loading functions
        from sysagent.utils.config.config_loader import discover_entrypoint_paths

        # Search for software.yml in system/info subdirectories
        config_paths = discover_entrypoint_paths("configs")

        for config_path in config_paths:
            software_config_path = os.path.join(
                config_path, "system", "info", "software.yml"
            )
            if os.path.exists(software_config_path):
                logger.debug(f"Loading software config from: {software_config_path}")

                try:
                    with open(software_config_path, "r", encoding="utf-8") as f:
                        config_data = yaml.safe_load(f)

                    if config_data:
                        # Extend lists with packages from this config file
                        python_packages = config_data.get("python_packages", [])
                        system_packages = config_data.get("system_packages", [])

                        consolidated_config["python_packages"].extend(python_packages)
                        consolidated_config["system_packages"].extend(system_packages)

                        logger.debug(
                            f"Added {len(python_packages)} Python packages and "
                            f"{len(system_packages)} system packages from "
                            f"{software_config_path}"
                        )

                except Exception as e:
                    logger.warning(
                        (
                            f"Failed to load software config from "
                            f"{software_config_path}: {e}"
                        )
                    )

        # Remove duplicates while preserving order
        consolidated_config["python_packages"] = list(
            dict.fromkeys(consolidated_config["python_packages"])
        )
        consolidated_config["system_packages"] = list(
            dict.fromkeys(consolidated_config["system_packages"])
        )

        logger.debug(
            "Consolidated software config: %d Python packages, %d system packages",
            len(consolidated_config["python_packages"]),
            len(consolidated_config["system_packages"]),
        )

    except Exception as e:
        logger.warning(f"Failed to load software configurations: {e}")

    return consolidated_config


def collect_software_info() -> Dict[str, Any]:
    """
    Collect comprehensive software information.

    Returns:
        Dict containing all software information
    """
    logger.debug("Collecting software system information")

    software_info = {
        "os": collect_os_info(),
        "python": collect_python_info(),
        "python_packages": collect_python_package_info(),
        "system_packages": collect_system_package_info(),
        "environment": collect_environment_info(),
    }

    return software_info


def collect_os_info() -> Dict[str, Any]:
    """
    Collect operating system information.

    Returns:
        Dict containing OS information
    """
    try:
        os_info = {
            "name": platform.system(),
            "version": platform.version(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "platform": platform.platform(),
            "node": platform.node(),
        }

        # Add Linux-specific information
        if platform.system().lower() == "linux":
            linux_info = _collect_linux_info()
            os_info.update(linux_info)

        # Add Windows-specific information
        elif platform.system().lower() == "windows":
            windows_info = _collect_windows_info()
            os_info.update(windows_info)

        return os_info

    except Exception as e:
        logger.warning(f"Failed to collect OS info: {e}")
        return {"error": str(e)}


def collect_python_info() -> Dict[str, Any]:
    """
    Collect Python interpreter and environment information.

    Returns:
        Dict containing Python information
    """
    try:
        python_info = {
            "version": sys.version,
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
                "releaselevel": sys.version_info.releaselevel,
                "serial": sys.version_info.serial,
            },
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": getattr(sys, "base_prefix", sys.prefix),
            "platform": sys.platform,
            "maxsize": sys.maxsize,
            "path": sys.path[:5],  # First 5 paths to avoid too much data
            "modules": len(sys.modules),
            "in_virtualenv": _is_in_virtualenv(),
            "pip_version": _get_pip_version(),
        }

        # Add virtual environment information if applicable
        if python_info["in_virtualenv"]:
            venv_info = _collect_virtualenv_info()
            python_info["virtualenv"] = venv_info

        return python_info

    except Exception as e:
        logger.warning(f"Failed to collect Python info: {e}")
        return {"error": str(e)}


def collect_python_package_info() -> Dict[str, Any]:
    """
    Collect information about installed Python packages.

    Returns:
        Dict containing Python package information with simplified structure
    """
    try:
        # Load configured packages from software config files
        software_config = _load_software_config()
        python_packages = software_config.get("python_packages", [])

        packages = {}

        for package in python_packages:
            version = _get_package_version(package)
            packages[package] = version if version else ""

        package_info = {
            "packages": packages,
            "total_installed": _count_installed_packages(),
            "pip_version": _get_pip_version(),
        }

        return package_info

    except Exception as e:
        logger.warning(f"Failed to collect Python package info: {e}")
        return {"error": str(e)}


def collect_system_package_info() -> Dict[str, Any]:
    """
    Collect information about installed system packages.

    Returns:
        Dict containing system package information with simplified structure
    """
    try:
        # Load configured packages from software config files
        software_config = _load_software_config()
        system_packages = software_config.get("system_packages", [])

        packages = {}

        for package in system_packages:
            version = _get_system_package_version(package)
            packages[package] = version if version else ""

        # Count total installed system packages
        total_installed = _count_installed_system_packages()

        package_info = {
            "packages": packages,
            "total_installed": total_installed,
            "package_manager": _detect_package_manager(),
        }

        return package_info

    except Exception as e:
        logger.warning(f"Failed to collect system package info: {e}")
        return {"error": str(e)}


def collect_environment_info() -> Dict[str, Any]:
    """
    Collect environment variables and system configuration.

    Returns:
        Dict containing environment information
    """
    try:
        # Get important environment variables (avoiding sensitive data)
        important_vars = [
            "PATH",
            "PYTHONPATH",
            "HOME",
            "USER",
            "SHELL",
            "TERM",
            "LANG",
            "LC_ALL",
            "TZ",
            "DISPLAY",
            "XDG_SESSION_TYPE",
            "CORE_DATA_DIR",
            "CORE_SUITES_PATH",
        ]

        env_vars = {}
        for var in important_vars:
            value = os.environ.get(var)
            if value:
                # Truncate very long values (like PATH)
                if len(value) > 500:
                    env_vars[var] = value[:500] + "... (truncated)"
                else:
                    env_vars[var] = value

        environment_info = {
            "variables": env_vars,
            "current_directory": os.getcwd(),
            "user_home": os.path.expanduser("~"),
            "temp_directory": _get_temp_directory(),
        }

        return environment_info

    except Exception as e:
        logger.warning(f"Failed to collect environment info: {e}")
        return {"error": str(e)}


def _collect_linux_info() -> Dict[str, Any]:
    """
    Collect Linux-specific information.

    Returns:
        Dict containing Linux-specific information
    """
    linux_info = {}

    # Try to get distribution information
    try:
        with open("/etc/os-release", "r") as f:
            lines = f.readlines()

        os_release = {}
        for line in lines:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                os_release[key] = value.strip('"')

        linux_info["distribution"] = {
            "name": os_release.get("NAME", "Unknown"),
            "version": os_release.get("VERSION", "Unknown"),
            "id": os_release.get("ID", "Unknown"),
            "version_id": os_release.get("VERSION_ID", "Unknown"),
            "pretty_name": os_release.get("PRETTY_NAME", "Unknown"),
            "home_url": os_release.get("HOME_URL", "Unknown"),
        }
    except Exception:
        linux_info["distribution"] = {"error": "Could not read /etc/os-release"}

    # Get kernel information
    try:
        linux_info["kernel"] = {
            "version": platform.release(),
            "build": platform.version(),
        }

        # Try to get more detailed kernel info
        try:
            with open("/proc/version", "r") as f:
                kernel_version = f.read().strip()
                linux_info["kernel"]["full_version"] = kernel_version
        except Exception:
            pass

    except Exception:
        linux_info["kernel"] = {"error": "Could not get kernel information"}

    # Get uptime
    try:
        with open("/proc/uptime", "r") as f:
            uptime_seconds = float(f.read().split()[0])
            linux_info["uptime_seconds"] = uptime_seconds
    except Exception:
        pass

    return linux_info


def _collect_windows_info() -> Dict[str, Any]:
    """
    Collect Windows-specific information.

    Returns:
        Dict containing Windows-specific information
    """
    windows_info = {}

    try:
        # Get Windows version information
        import platform

        windows_info["edition"] = (
            platform.win32_edition()
            if hasattr(platform, "win32_edition")
            else "Unknown"
        )
        windows_info["version"] = platform.win32_ver()

        # Try to get more detailed Windows information using wmic
        try:
            result = run_command(
                ["wmic", "os", "get", "Caption,Version,BuildNumber", "/format:csv"],
                timeout=10,
            )
            if result.success:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    headers = lines[0].split(",")
                    values = lines[1].split(",")
                    if len(headers) == len(values):
                        wmic_info = dict(zip(headers, values))
                        windows_info["detailed"] = wmic_info
        except Exception:
            pass

    except Exception as e:
        windows_info["error"] = str(e)

    return windows_info


def _is_in_virtualenv() -> bool:
    """
    Check if Python is running in a virtual environment.

    Returns:
        True if in virtual environment, False otherwise
    """
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("VIRTUAL_ENV") is not None
    )


def _collect_virtualenv_info() -> Dict[str, Any]:
    """
    Collect virtual environment information.

    Returns:
        Dict containing virtual environment information
    """
    venv_info = {}

    # Check for VIRTUAL_ENV environment variable
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        venv_info["path"] = virtual_env
        venv_info["name"] = os.path.basename(virtual_env)

    # Check for conda environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        venv_info["conda_env"] = conda_env
        venv_info["conda_prefix"] = os.environ.get("CONDA_PREFIX")

    # Check for pip-tools or pipenv
    if os.path.exists(os.path.join(os.getcwd(), "Pipfile")):
        venv_info["type"] = "pipenv"
    elif os.path.exists(os.path.join(os.getcwd(), "pyproject.toml")):
        venv_info["type"] = "poetry_or_pip-tools"
    elif os.path.exists(os.path.join(os.getcwd(), "requirements.txt")):
        venv_info["type"] = "pip"

    return venv_info


def _get_pip_version() -> Optional[str]:
    """
    Get pip version.

    Returns:
        Pip version string or None if not available
    """
    try:
        import pip

        return pip.__version__
    except (ImportError, AttributeError):
        try:
            result = run_command([sys.executable, "-m", "pip", "--version"], timeout=10)
            if result.success:
                # Extract version from output like "pip 21.3.1 from ..."
                output = result.stdout.strip()
                if output.startswith("pip "):
                    return output.split()[1]
        except Exception:
            pass

    return None


def _get_package_version(package_name: str) -> Optional[str]:
    """
    Get version of an installed package.

    Args:
        package_name: Name of the package

    Returns:
        Package version string or None if not installed
    """
    try:
        import importlib.metadata

        return importlib.metadata.version(package_name)
    except Exception:
        return None


def _count_installed_packages() -> int:
    """
    Count total number of installed Python packages.

    Returns:
        Number of installed packages
    """
    try:
        import importlib.metadata

        return len(list(importlib.metadata.distributions()))
    except Exception:
        return 0


def _count_installed_system_packages() -> int:
    """
    Count total number of installed system packages.

    Returns:
        Number of installed system packages
    """
    try:
        # Try different package managers to get total count
        package_managers = [
            # Debian/Ubuntu - dpkg
            ["dpkg", "--get-selections"],
        ]

        for cmd in package_managers:
            try:
                result = run_command(cmd, timeout=30)
                if result.success:
                    # Count lines (each line is a package)
                    lines = result.stdout.strip().split("\n")
                    return len([line for line in lines if line.strip()])
            except (FileNotFoundError, OSError):
                continue

        return 0
    except Exception:
        return 0


def _get_temp_directory() -> str:
    """
    Get system temporary directory.

    Returns:
        Path to temporary directory
    """
    import tempfile

    return tempfile.gettempdir()


def _get_system_package_version(package_name: str) -> Optional[str]:
    """
    Get version of an installed system package.

    Args:
        package_name: Name of the system package

    Returns:
        Package version string or None if not installed
    """
    try:
        # Try different package managers
        package_managers = [
            # Debian/Ubuntu - dpkg
            (
                ["dpkg", "-l", package_name],
                lambda output: _parse_dpkg_version(output, package_name),
            ),
            # Snap
            (
                ["snap", "list", package_name],
                lambda output: _parse_snap_version(output),
            ),
        ]

        for cmd, parser in package_managers:
            try:
                result = run_command(cmd, timeout=10)
                if result.success:
                    version = parser(result.stdout.strip())
                    if version:
                        logger.debug(
                            f"Found {package_name} version {version} via {cmd[0]}"
                        )
                        return version
            except (FileNotFoundError, OSError):
                continue

    except Exception as e:
        logger.debug(f"Failed to get system package version for {package_name}: {e}")

    return None


def _detect_package_manager() -> str:
    """
    Detect the primary package manager on the system.

    Returns:
        Name of the detected package manager
    """
    managers = [("dpkg", "apt"), ("snap", "snap")]

    for cmd, name in managers:
        try:
            result = run_command([cmd, "--version"], timeout=5)
            if result.success:
                return name
        except (FileNotFoundError, OSError):
            continue

    return "unknown"


def _parse_dpkg_version(output: str, package_name: str) -> Optional[str]:
    """Parse version from dpkg output."""
    lines = output.split("\n")
    for line in lines:
        if package_name in line and line.startswith("ii"):
            parts = line.split()
            if len(parts) >= 3:
                return parts[2]  # Version is typically the 3rd column
    return None


def _parse_snap_version(output: str) -> Optional[str]:
    """Parse version from snap output."""
    lines = output.split("\n")
    if len(lines) > 1:  # Skip header
        parts = lines[1].split()
        if len(parts) >= 2:
            return parts[1]  # Version column
    return None
