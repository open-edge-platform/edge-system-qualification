# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Configuration loader utilities for the core framework.

This module provides utilities for loading configuration from various sources
including pyproject.toml, entrypoints, and package metadata.
"""

import importlib
import importlib.metadata
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    pass  # Fallback for older Python versions

logger = logging.getLogger(__name__)


def get_dist_name() -> Optional[str]:
    """Get the distribution name for the current package."""
    pkg = __name__.split(".", 1)[0]
    mapping = importlib.metadata.packages_distributions()
    return mapping.get(pkg, [None])[0]


def get_dist_version(dist: Optional[str] = None) -> str:
    """Get the version of a distribution."""
    if not dist:
        dist = get_dist_name()
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def discover_project_roots(group: str = "sysagent") -> List[Path]:
    """
    Discover project roots for all entrypoints in the given group.

    Args:
        group: Entrypoint group to search

    Returns:
        List of directories containing pyproject.toml
    """
    roots = []

    # Always include sysagent's own root
    try:
        import sysagent

        sysagent_package_dir = Path(sysagent.__file__).parent  # .../src/sysagent/
        sysagent_root = sysagent_package_dir.parent  # .../src/
        # Check if there's a pyproject.toml in the sysagent package directory
        if (sysagent_package_dir / "pyproject.toml").exists():
            roots.append(sysagent_package_dir)
        else:
            # Fallback to the general project root
            roots.append(sysagent_package_dir.parent.parent)  # project root
    except Exception:
        logger.warning("Could not locate sysagent package root.")

    # Discover additional roots via entrypoints
    try:
        entrypoints = importlib.metadata.entry_points()
        eps = entrypoints.select(group=group) if hasattr(entrypoints, "select") else entrypoints.get(group, [])

        for ep in eps:
            try:
                module = importlib.import_module(ep.value)
                module_dir = Path(module.__file__).parent
                for candidate in [module_dir, *module_dir.parents]:
                    pyproject = candidate / "pyproject.toml"
                    if pyproject.exists():
                        roots.append(candidate)
                        break
            except Exception as e:
                logger.warning(f"Failed to load root from entrypoint {ep.name}: {e}")
    except Exception as e:
        logger.warning(f"Failed to discover entrypoints: {e}")

    # Remove duplicates and non-existent pyproject.toml
    unique_roots = []
    seen = set()
    for root in roots:
        pyproject = root / "pyproject.toml"
        if pyproject.exists() and str(root) not in seen:
            unique_roots.append(root)
            seen.add(str(root))

    return unique_roots


def discover_entrypoint_paths(folder_name: str) -> List[Path]:
    """
    Discover paths for a given folder name from sysagent entrypoints.

    This function is CLI-context-aware: when invoked via a specific CLI command
    (e.g., test-esq), it will only discover paths from that specific package
    plus sysagent (core framework).

    Args:
        folder_name: The folder to look for in each entrypoint module

    Returns:
        List of discovered paths
    """
    paths = []
    cli_package = get_cli_aware_project_name()

    # Add sysagent default path (always included as it's the core framework)
    try:
        import sysagent

        paths.append(Path(sysagent.__file__).parent / folder_name)
        logger.debug(f"Loaded sysagent default {folder_name} path: {paths[-1]}")
    except Exception:
        logger.warning(f"Failed to load sysagent default {folder_name} path. Ensure sysagent is installed.")

    # If a specific CLI package is active, only include that package's paths
    if cli_package and cli_package != "sysagent":
        # Map CLI command to package directory name
        package_dir_name = cli_package.replace("-", "_")  # test-esq -> test_esq

        # Check source directories for the specific package
        cwd = Path.cwd()
        possible_paths = [
            cwd / "src" / package_dir_name / folder_name,
            cwd / "tests" / cli_package / "src" / package_dir_name / folder_name,  # For test packages
        ]

        for pkg_path in possible_paths:
            if pkg_path.exists():
                paths.append(pkg_path)
                logger.debug(f"Added CLI-specific {folder_name} path: {pkg_path}")

        # Also check installed package location
        try:
            module = importlib.import_module(package_dir_name)
            module_path = Path(module.__file__).parent / folder_name
            if module_path.exists() and module_path not in paths:
                paths.append(module_path)
                logger.debug(f"Added installed {folder_name} path: {module_path}")
        except ImportError:
            logger.debug(f"Could not import {package_dir_name}")

        # Return early - we only want sysagent + this specific package
        return list(set(paths))  # Remove duplicates

    # Original behavior when no specific CLI package (show all packages)
    # Add local paths (current working directory and source directories)
    cwd = Path.cwd()
    local_folder_path = cwd / folder_name
    if local_folder_path.exists():
        paths.append(local_folder_path)
        logger.debug(f"Added local {folder_name} path: {local_folder_path}")

    # Check source directories
    src_dirs = [
        cwd / "src" / "sysagent" / folder_name,
    ]

    # Dynamically discover other packages in src/ directory only if we have extensions
    if _has_extensions():
        src_dir = cwd / "src"
        if src_dir.exists():
            for pkg_dir in src_dir.iterdir():
                if pkg_dir.is_dir() and not pkg_dir.name.startswith(".") and pkg_dir.name != "sysagent":
                    pkg_folder_path = pkg_dir / folder_name
                    if pkg_folder_path not in src_dirs:  # Avoid duplicates
                        src_dirs.append(pkg_folder_path)

    for src_path in src_dirs:
        if src_path.exists():
            paths.append(src_path)
            logger.debug(f"Added source {folder_name} path: {src_path}")

    # Discover additional folders via entrypoints only if we have extensions
    if _has_extensions():
        try:
            entrypoints = importlib.metadata.entry_points()
            eps = (
                entrypoints.select(group="sysagent")
                if hasattr(entrypoints, "select")
                else entrypoints.get("sysagent", [])
            )

            # Filter entrypoints by name convention: sysagent_{folder_name}
            target_ep_name = f"sysagent_{folder_name}"
            filtered_eps = [ep for ep in eps if ep.name == target_ep_name]
            logger.debug(f"Found {len(filtered_eps)} entrypoints for sysagent {folder_name}.")

            for ep in filtered_eps:
                logger.debug(f"Processing entrypoint: {ep.name} -> {ep.value}")
                try:
                    # Import the base package, not the config folder
                    module = importlib.import_module(ep.value)
                    module_path = Path(module.__file__).parent / folder_name
                    if module_path.exists():
                        paths.append(module_path)
                        logger.debug(f"Added {folder_name} path: {module_path}")
                    else:
                        logger.warning(f"{folder_name} folder not found at {module_path}")
                except Exception as e:
                    logger.warning(f"Failed to process entrypoint {ep.name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to discover entrypoints: {e}")

    # Remove duplicates while preserving order
    unique_paths = []
    seen = set()
    for path in paths:
        path_str = str(path)
        if path_str not in seen and path.exists():
            unique_paths.append(path)
            seen.add(path_str)

    return unique_paths


def _has_extensions() -> bool:
    """
    Check if this is an extended installation (with extension packages) or minimal (sysagent only).

    This function determines whether the current execution context includes extension packages
    by examining the CLI command being used and the project name. It supports any extension
    package name dynamically without hardcoded references.

    Examples:
        - sysagent CLI: Returns False (minimal installation)
        - esq CLI: Returns True (extension installation)
        - any-other-cli: Returns True (extension installation)

    Returns:
        bool: True if extension packages are installed, False for sysagent-only installation
    """
    # Check environment variable first (set by CLI handlers)
    cli_package = os.environ.get("SYSAGENT_CLI_PACKAGE", None)
    if cli_package:
        return True  # Any CLI package context means we should show only that package's profiles

    # Check command line arguments to see which CLI is being used
    if len(sys.argv) > 0:
        cli_command = sys.argv[0].split("/")[-1].split("\\")[-1]  # Get just the command name
        if cli_command == "sysagent":
            # If we're running the sysagent CLI directly, treat as minimal installation
            return False
        elif cli_command != "python" and cli_command != "python3":
            # If we're running any other CLI (like esq, test-esq), it's an extension
            return True

    # Fallback: Check if current project name indicates an extension
    try:
        project_name = get_project_name()
        if project_name and project_name != "sysagent":
            return True
    except Exception:
        pass

    return False


def load_pyproject_config(project_root: Path) -> Dict[str, Any]:
    """
    Load configuration from pyproject.toml.

    Args:
        project_root: Path to the project root containing pyproject.toml

    Returns:
        Configuration dictionary from pyproject.toml
    """
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        logger.warning(f"pyproject.toml not found at {pyproject_path}")
        return {}

    try:
        # Try tomllib first (Python 3.11+), then tomli
        try:
            import tomllib

            with open(pyproject_path, "rb") as f:
                return tomllib.load(f)
        except ImportError:
            import tomli

            with open(pyproject_path, "rb") as f:
                return tomli.load(f)
    except Exception as e:
        logger.error(f"Failed to load pyproject.toml from {pyproject_path}: {e}")
        return {}


def get_project_name() -> str:
    """Get the project name from configuration."""
    try:
        # Try to get from the current package's pyproject.toml
        project_roots = discover_project_roots()
        for root in project_roots:
            config = load_pyproject_config(root)
            project_name = config.get("project", {}).get("name")
            if project_name:
                return project_name
    except Exception:
        pass

    # Fallback to package name
    dist_name = get_dist_name()
    return dist_name if dist_name else "sysagent"


def get_cli_aware_project_name() -> str:
    """
    Get the project name based on the CLI context.

    When running with extensions (esq, test-esq CLI), returns the extension name.
    When running with minimal installation (sysagent CLI), returns 'sysagent'.

    Returns:
        str: Project name based on CLI context (e.g., 'esq', 'test-esq', 'sysagent')
    """
    import os

    # First check environment variable (most reliable, set by CLI main function)
    cli_package = os.environ.get("SYSAGENT_CLI_PACKAGE", None)
    if cli_package:
        return cli_package

    # Check command line arguments to determine CLI context
    if len(sys.argv) > 0:
        cli_command = os.path.basename(sys.argv[0])

        # Map CLI command names directly to package names
        if cli_command in ["esq", "test-esq", "sysagent"]:
            return cli_command

        # When running pytest, check current working directory FIRST
        # to avoid false matches from paths like /esq/dev/add-test-workflow
        import os

        cwd = os.getcwd()

        # Check if we're in a specific package directory (e.g., tests/esq for test-esq)
        # Use path separators to ensure we're matching directory names, not substrings
        if os.path.sep + "tests" + os.path.sep + "esq" in cwd:
            # In tests/esq directory - this is test-esq package
            return "test-esq"
        elif (
            cwd.endswith(os.path.sep + "tests" + os.path.sep + "esq")
            or cwd.endswith("/tests/esq")
            or cwd.endswith("\\tests\\esq")
        ):
            # Also handle if cwd ends with tests/esq
            return "test-esq"

        # Check for other package contexts based on cwd
        # (e.g., src/esq for esq package, src/sysagent for sysagent)
        if os.path.sep + "src" + os.path.sep + "esq" in cwd and "test" not in cwd.lower():
            return "esq"
        elif os.path.sep + "src" + os.path.sep + "sysagent" in cwd:
            return "sysagent"

        # Fallback: check if any known CLI name is in the argv path
        # IMPORTANT: Check longer names first to avoid substring matches
        # But only check for exact command names, not path substrings
        if "sysagent" in os.path.basename(sys.argv[0]):
            return "sysagent"
        elif "test-esq" in os.path.basename(sys.argv[0]):
            return "test-esq"
        elif os.path.basename(sys.argv[0]) == "esq":
            # Only match if basename is exactly "esq", not if it's in the path
            return "esq"
        else:
            # Running extension CLI - try to get extension project name
            # Detect based on current working directory or pytest context
            try:
                import os

                cwd = os.getcwd()

                # Check if we're in a specific package directory (e.g., tests/esq for test-esq)
                # This helps when running pytest directly without the CLI
                if "tests/esq" in cwd or "tests\\esq" in cwd:
                    return "test-esq"
                elif "/esq" in cwd or "\\esq" in cwd:
                    # Could be either esq or test-esq, check more carefully
                    if "test" in cwd.lower():
                        return "test-esq"
                    else:
                        return "esq"

                # Get the actual project name from configuration
                project_roots = discover_project_roots()

                # Prioritize extension package names over sysagent
                extension_names = []
                sysagent_name = None

                for root in project_roots:
                    config = load_pyproject_config(root)
                    project_name = config.get("project", {}).get("name")
                    if project_name:
                        if project_name == "sysagent":
                            sysagent_name = project_name
                        else:
                            extension_names.append(project_name)

                # Return first extension name found, fallback to sysagent
                if extension_names:
                    return extension_names[0]  # Use first extension (e.g., 'esq')
                elif sysagent_name:
                    return sysagent_name

            except Exception:
                pass

    # Fallback to regular project name detection
    return get_project_name()


def get_allure_version() -> str:
    """Get the Allure version from configuration."""
    try:
        project_roots = discover_project_roots()
        for root in project_roots:
            config = load_pyproject_config(root)
            # Try multiple paths for allure version
            version = config.get("tool", {}).get("sysagent", {}).get("allure_version") or config.get("tool", {}).get(
                "sysagent", {}
            ).get("allure", {}).get("version")
            if version:
                return version
    except Exception:
        pass

    # Fallback version
    return "v3.0.0-beta.15"


def get_node_version() -> str:
    """Get the Node.js version from configuration."""
    try:
        project_roots = discover_project_roots()
        for root in project_roots:
            config = load_pyproject_config(root)
            # Try multiple paths for node version
            version = config.get("tool", {}).get("sysagent", {}).get("node_version") or config.get("tool", {}).get(
                "sysagent", {}
            ).get("node", {}).get("version")
            if version:
                return version
    except Exception:
        pass

    # Fallback version
    return "v22.17.0"


def load_tool_config(section: str = "tool.sysagent", package: str = "sysagent") -> Dict[str, Any]:
    """
    Load the tool configuration from the correct pyproject.toml.
    Args:
        section (str): Section in pyproject.toml (dot notation).
        package (str): Which package's pyproject.toml to load.
    Returns:
        Dict[str, Any]: The configuration as a dictionary.
    """
    config = {}
    roots = discover_project_roots()
    # Try to find the root for the requested package
    selected_root = None
    for root in roots:
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            # Try to match the project name in pyproject.toml
            try:
                # Try tomllib first (Python 3.11+), then tomli
                try:
                    import tomllib

                    with open(pyproject, "rb") as f:
                        data = tomllib.load(f)
                except ImportError:
                    import tomli

                    with open(pyproject, "rb") as f:
                        data = tomli.load(f)
                project_name = data.get("project", {}).get("name", "")
                if project_name == package:
                    selected_root = root
                    break
            except Exception:
                continue
    if not selected_root and roots:
        selected_root = roots[0]  # fallback

    if selected_root:
        pyproject_path = selected_root / "pyproject.toml"
        try:
            # Try tomllib first (Python 3.11+), then tomli
            try:
                import tomllib

                with open(pyproject_path, "rb") as f:
                    pyproject_data = tomllib.load(f)
            except ImportError:
                import tomli

                with open(pyproject_path, "rb") as f:
                    pyproject_data = tomli.load(f)
            section_keys = section.split(".")
            config_section = pyproject_data
            for key in section_keys:
                if isinstance(config_section, dict) and key in config_section:
                    config_section = config_section[key]
                else:
                    config_section = None
                    break
            if config_section and isinstance(config_section, dict):
                config = config_section
        except Exception as e:
            logger.warning(f"Failed to load configuration from {pyproject_path}: {e}")
    else:
        logger.warning("No pyproject.toml found for any known roots.")

    return config


def load_merged_tool_config(section: str = "tool.sysagent") -> Dict[str, Any]:
    """
    Load and merge tool configurations from all packages with dynamic discovery.

    Core packages (sysagent) provide base configurations, while extension packages
    can add or override specific configurations. Merging follows dependency-like
    logic with core packages loaded first.

    Args:
        section: Section in pyproject.toml (dot notation) to load

    Returns:
        Dict[str, Any]: Merged configuration from all packages
    """
    merged_config = {}
    roots = discover_project_roots()

    # Sort packages for consistent merging order (sysagent first)
    package_configs = []
    sysagent_config = None
    extension_configs = []

    for root in roots:
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            try:
                # Try tomllib first (Python 3.11+), then tomli
                try:
                    import tomllib

                    with open(pyproject, "rb") as f:
                        data = tomllib.load(f)
                except ImportError:
                    import tomli

                    with open(pyproject, "rb") as f:
                        data = tomli.load(f)

                project_name = data.get("project", {}).get("name", "")

                # Extract the tool config section
                section_keys = section.split(".")
                config_section = data
                for key in section_keys:
                    if isinstance(config_section, dict) and key in config_section:
                        config_section = config_section[key]
                    else:
                        config_section = None
                        break

                if config_section and isinstance(config_section, dict):
                    if project_name == "sysagent":
                        sysagent_config = config_section
                        logger.debug(f"Loaded sysagent tool config: {len(config_section)} keys")
                    else:
                        extension_configs.append((project_name, config_section))
                        logger.debug(f"Loaded {project_name} tool config: {len(config_section)} keys")

            except Exception as e:
                logger.warning(f"Failed to load tool config from {pyproject}: {e}")

    # Merge configurations: sysagent base first, then extensions
    if sysagent_config:
        merged_config.update(sysagent_config)

    # Sort extension configs by package name for consistent ordering
    extension_configs.sort(key=lambda x: x[0])

    for package_name, config in extension_configs:
        # Merge extension config with special handling for lists
        for key, value in config.items():
            if key in merged_config:
                # Special handling for github_dependencies - merge lists
                if key == "github_dependencies" and isinstance(value, list) and isinstance(merged_config[key], list):
                    # Combine lists, avoiding duplicates based on 'name' field
                    existing_names = {item.get("name") for item in merged_config[key] if isinstance(item, dict)}
                    for item in value:
                        if isinstance(item, dict) and item.get("name") not in existing_names:
                            merged_config[key].append(item)
                            existing_names.add(item.get("name"))
                else:
                    # For other keys, extension values override base values
                    merged_config[key] = value
                    logger.debug(f"Override {key} from {package_name}")
            else:
                merged_config[key] = value
                logger.debug(f"Added {key} from {package_name}")

    logger.debug(
        f"Merged tool config with {len(merged_config)} keys from {1 if sysagent_config else 0} core + {len(extension_configs)} extension packages"
    )
    return merged_config


def get_entrypoint_config(name: str, group: str = "sysagent") -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific entrypoint.

    Args:
        name: Entrypoint name
        group: Entrypoint group

    Returns:
        Configuration dictionary or None if not found
    """
    try:
        entrypoints = importlib.metadata.entry_points()
        eps = entrypoints.select(group=group, name=name) if hasattr(entrypoints, "select") else []

        if not eps:
            # Fallback for older Python versions
            all_eps = entrypoints.get(group, [])
            eps = [ep for ep in all_eps if ep.name == name]

        for ep in eps:
            try:
                module = importlib.import_module(ep.value)
                # Look for configuration in the module
                config = getattr(module, "CONFIG", None)
                if config:
                    return config

                # Try to load from module's pyproject.toml
                module_dir = Path(module.__file__).parent
                for candidate in [module_dir, *module_dir.parents]:
                    pyproject = candidate / "pyproject.toml"
                    if pyproject.exists():
                        return load_pyproject_config(candidate)

            except Exception as e:
                logger.warning(f"Failed to load config from entrypoint {name}: {e}")

    except Exception as e:
        logger.warning(f"Failed to get entrypoint config: {e}")

    return None


def find_config_files(filename: str, search_paths: Optional[List[Path]] = None) -> List[Path]:
    """
    Find configuration files in discovered paths.

    Args:
        filename: Name of the configuration file to find
        search_paths: Optional list of additional paths to search

    Returns:
        List of paths to found configuration files
    """
    found_files = []

    # Default search paths
    if search_paths is None:
        search_paths = []

    # Add discovered config paths
    config_paths = discover_entrypoint_paths("configs")
    search_paths.extend(config_paths)

    # Add current directory and standard locations
    search_paths.extend(
        [
            Path.cwd(),
            Path.cwd() / "configs",
            Path(__file__).parent.parent / "configs",
            Path.home() / ".sysagent",
            Path("/etc/sysagent"),
        ]
    )

    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in search_paths:
        if str(path) not in seen:
            unique_paths.append(path)
            seen.add(str(path))

    # Search for the file
    for search_path in unique_paths:
        if search_path.exists():
            config_file = search_path / filename
            if config_file.exists():
                found_files.append(config_file)

    return found_files


def get_runtime_config() -> Dict[str, Any]:
    """
    Get runtime configuration from various sources.

    Returns:
        Runtime configuration dictionary
    """
    config = {
        "project_name": get_project_name(),
        "dist_name": get_dist_name(),
        "dist_version": get_dist_version(),
        "allure_version": get_allure_version(),
        "node_version": get_node_version(),
        "project_roots": [str(p) for p in discover_project_roots()],
        "config_paths": [str(p) for p in discover_entrypoint_paths("configs")],
        "suite_paths": [str(p) for p in discover_entrypoint_paths("suites")],
    }

    return config


def validate_config_integrity() -> List[str]:
    """
    Validate configuration integrity across all discovered sources.

    Returns:
        List of validation warnings/errors
    """
    issues = []

    try:
        # Check if project roots are accessible
        project_roots = discover_project_roots()
        if not project_roots:
            issues.append("No project roots discovered")

        for root in project_roots:
            if not (root / "pyproject.toml").exists():
                issues.append(f"pyproject.toml missing in {root}")

        # Check if essential paths exist
        config_paths = discover_entrypoint_paths("configs")
        if not config_paths:
            issues.append("No config paths discovered")

        # Validate tool configuration
        tool_config = load_tool_config()
        if not tool_config:
            issues.append("No tool configuration found")

    except Exception as e:
        issues.append(f"Configuration validation failed: {e}")

    return issues
