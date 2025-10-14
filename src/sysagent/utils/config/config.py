# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for handling configuration files in the core framework.
"""

import hashlib
import inspect
import logging
import os
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def get_project_name():
    """Get the project name from configuration."""
    try:
        from sysagent.utils.config.config_loader import (
            get_project_name as _get_project_name,
        )

        return _get_project_name()
    except ImportError:
        return "sysagent"


def update_dict_recursively(base_dict, update_dict):
    """
    Recursively update a dictionary.

    Args:
        base_dict: Dictionary to update
        update_dict: Dictionary with updates to apply
    """
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def list_profiles(include_examples: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """
    List all available profile configurations and their file paths.
    This dynamically discovers profile directories, optionally including examples directory.

    Args:
        include_examples: Whether to include profiles from the examples directory

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping profile types to lists of {'configs', 'path'} dicts
    """
    profiles = {}

    # Use the centralized directory discovery logic
    profile_directories = get_profile_directories(include_examples=include_examples)

    # Process each profile directory to extract configs and file paths
    seen_filenames = set()  # Global set for all directories
    for profile_type, profile_dir in profile_directories:
        profile_items = []

        if not os.path.exists(profile_dir):
            continue

        for item in os.listdir(profile_dir):
            if item.endswith(".yml") or item.endswith(".yaml"):
                file_path = os.path.join(profile_dir, item)

                # Only include profiles if we're in a full installation or if the profile is from sysagent
                if not _should_include_profile(file_path):
                    logger.debug(f"Skipping profile from extension package in minimal installation: {file_path}")
                    continue

                try:
                    config = load_yaml_config(file_path)
                    profile_items.append(
                        {
                            "configs": config,
                            "path": file_path,
                            "name": config.get("name", item.split(".")[0]),
                        }
                    )
                    seen_filenames.add(item)
                except Exception:
                    # Skip invalid config files
                    continue

        if profile_items:
            profiles[profile_type] = profile_items

    return profiles


def _should_include_profile(profile_file: str) -> bool:
    """
    Determine if a profile should be included based on installation type.

    This function filters profiles based on whether the current installation includes
    extension packages. For minimal sysagent-only installations, only sysagent profiles
    are included. For extended installations (any extension package), all profiles are included.

    Args:
        profile_file: Path to the profile file

    Returns:
        bool: True if profile should be included
    """
    from sysagent.utils.config.config_loader import _has_extensions

    # Always include profiles in extended installations
    if _has_extensions():
        return True

    # In minimal installation, only include profiles from sysagent package
    # Check if the profile is in sysagent directory
    sysagent_paths = [
        "/src/sysagent/",
        "\\src\\sysagent\\",  # Windows
        "sysagent/configs/",
        "sysagent\\configs\\",  # Windows
    ]

    for sysagent_path in sysagent_paths:
        if sysagent_path in profile_file:
            return True

    return False


def setup_data_dir(data_dir: str = None) -> str:
    """
    Setup data directory for test results and logs.

    Args:
        data_dir: Optional data directory path

    Returns:
        str: Path to the data directory
    """
    if data_dir is None:
        from sysagent.utils.config.config_loader import get_cli_aware_project_name

        cwd_tainted = os.getcwd()
        cwd_sanitized = "".join(c for c in cwd_tainted)
        project_name_tainted = get_cli_aware_project_name()
        project_name_sanitized = "".join(c for c in project_name_tainted)

        data_dir = os.path.join(cwd_sanitized, f"{project_name_sanitized}_data")

    data_dir_sanitized = "".join(c for c in data_dir)

    os.makedirs(data_dir_sanitized, exist_ok=True)
    os.makedirs(os.path.join(data_dir_sanitized, "logs"), exist_ok=True)
    os.makedirs(os.path.join(data_dir_sanitized, "results", "allure"), exist_ok=True)
    os.makedirs(os.path.join(data_dir_sanitized, "cache"), exist_ok=True)
    os.makedirs(os.path.join(data_dir_sanitized, "data"), exist_ok=True)

    return data_dir_sanitized


def get_thirdparty_dir(data_dir: str = None) -> str:
    """
    Get the path to the third-party tools directory.

    Args:
        data_dir: Base data directory

    Returns:
        str: Path to the third-party tools directory
    """
    if data_dir is None:
        data_dir = setup_data_dir()

    data_dir_sanitized = "".join(c for c in data_dir)

    thirdparty_dir = os.path.join(data_dir_sanitized, "thirdparty")
    os.makedirs(thirdparty_dir, exist_ok=True)
    return thirdparty_dir


def get_cache_dir(data_dir: str = None) -> str:
    """
    Get the path to the cache directory.

    Args:
        data_dir: Base data directory

    Returns:
        str: Path to the cache directory
    """
    if data_dir is None:
        data_dir = setup_data_dir()

    data_dir_sanitized = "".join(c for c in data_dir)

    cache_dir = os.path.join(data_dir_sanitized, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_logs_dir(data_dir: str = None) -> str:
    """
    Get the path to the logs directory.

    Args:
        data_dir: Base data directory

    Returns:
        str: Path to the logs directory
    """
    if data_dir is None:
        data_dir = setup_data_dir()

    data_dir_sanitized = "".join(c for c in data_dir)

    logs_dir = os.path.join(data_dir_sanitized, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def get_results_dir(data_dir: str = None) -> str:
    """
    Get the path to the results directory.

    Args:
        data_dir: Base data directory

    Returns:
        str: Path to the results directory
    """
    if data_dir is None:
        data_dir = setup_data_dir()

    data_dir_sanitized = "".join(c for c in data_dir)

    results_dir = os.path.join(data_dir_sanitized, "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dict containing the configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config if config is not None else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")


def save_yaml_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as file:
        yaml.safe_dump(config, file, default_flow_style=False, indent=2)


def get_config_checksum(config_path: str) -> str:
    """
    Get MD5 checksum of a configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        str: MD5 checksum of the file
    """
    if not os.path.exists(config_path):
        return ""

    hash_md5 = hashlib.md5(usedforsecurity=False)
    with open(config_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def validate_config_schema(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
    """
    Validate configuration against required schema.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    for key in required_keys:
        if key not in config:
            errors.append(f"Required key '{key}' not found in configuration")
        elif config[key] is None:
            errors.append(f"Required key '{key}' cannot be None")

    return errors


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base_config: Base configuration
        override_config: Configuration to override base with

    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def get_environment_config() -> Dict[str, str]:
    """
    Get configuration values from environment variables.

    Returns:
        Dict containing environment-based configuration
    """
    env_config = {}

    # Common environment variables
    env_vars = [
        "CORE_DATA_DIR",
        "CORE_CACHE_DIR",
        "CORE_LOGS_DIR",
        "CORE_RESULTS_DIR",
        "CORE_THIRDPARTY_DIR",
        "CORE_CONFIG_PATH",
        "CORE_DEBUG",
        "CORE_VERBOSE",
    ]

    for var in env_vars:
        value = os.environ.get(var)
        if value is not None:
            # Convert key to lowercase and remove prefix
            key = var.lower().replace("core_", "")
            env_config[key] = value

    return env_config


def apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.

    Args:
        config: Base configuration

    Returns:
        Configuration with environment overrides applied
    """
    env_config = get_environment_config()
    return merge_configs(config, env_config)


def find_config_file(filename: str, search_paths: Optional[List[str]] = None) -> Optional[str]:
    """
    Find a configuration file in standard locations using hierarchical discovery.

    Args:
        filename: Name of the configuration file
        search_paths: Optional list of paths to search

    Returns:
        Path to the configuration file if found, None otherwise
    """
    if search_paths is None:
        from .config_loader import discover_entrypoint_paths

        # Use the new hierarchical discovery system
        config_paths = discover_entrypoint_paths("configs")
        search_paths = [str(p) for p in config_paths]

        # Add traditional fallback paths
        search_paths.extend(
            [
                os.getcwd(),
                os.path.join(os.getcwd(), "configs"),
                os.path.join(os.path.dirname(__file__), "..", "configs"),
                os.path.expanduser("~/.sysagent"),
                "/etc/sysagent",
            ]
        )

    for search_path in search_paths:
        config_path = os.path.join(search_path, filename)
        if os.path.exists(config_path):
            return config_path

    return None


def load_config_with_fallback(filename: str, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration with fallback to default.

    Args:
        filename: Name of the configuration file
        default_config: Default configuration to use if file not found

    Returns:
        Loaded configuration dictionary
    """
    config_path = find_config_file(filename)

    if config_path:
        try:
            config = load_yaml_config(config_path)
            logger.debug(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load configuration from {config_path}: {e}")
            config = default_config or {}
    else:
        logger.debug(f"Configuration file {filename} not found, using default")
        config = default_config or {}

    # Apply environment overrides
    config = apply_environment_overrides(config)

    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., "database.host")
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    keys = key_path.split(".")
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set a configuration value using dot notation.

    Args:
        config: Configuration dictionary to modify
        key_path: Dot-separated path to the value (e.g., "database.host")
        value: Value to set
    """
    keys = key_path.split(".")
    current = config

    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set the value
    current[keys[-1]] = value


def list_tiers() -> Dict[str, Any]:
    """
    List available test tiers from configuration.

    Returns:
        Dict containing tier configurations
    """
    from .config_loader import discover_entrypoint_paths

    # Search for requirement.yml in system/tiers subdirectories
    config_paths = discover_entrypoint_paths("configs")

    for config_path in config_paths:
        tier_config_path = os.path.join(config_path, "system", "tiers", "requirement.yml")
        if os.path.exists(tier_config_path):
            logger.debug(f"Found tier config: {tier_config_path}")
            return load_yaml_config(tier_config_path)

    # Fallback: try direct path search
    tier_config_path = find_config_file("system/tiers/requirement.yml")
    if tier_config_path:
        logger.debug(f"Found tier config via fallback: {tier_config_path}")
        return load_yaml_config(tier_config_path)

    logger.debug("No tier configuration found")
    return {"tiers": {}}


def get_tier_config(tier_name: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific tier.

    Args:
        tier_name: Name of the tier

    Returns:
        Tier configuration or None if not found
    """
    tiers_data = list_tiers()

    if "tiers" in tiers_data and tiers_data["tiers"]:
        # The structure is {"tiers": {tier_name: tier_config, ...}}
        all_tiers = tiers_data["tiers"]
        return all_tiers.get(tier_name)

    return None


def validate_tier_config(tier_config: Dict[str, Any]) -> List[str]:
    """
    Validate tier configuration.

    Args:
        tier_config: Tier configuration to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    required_keys = ["name", "priority", "requirements"]

    for key in required_keys:
        if key not in tier_config:
            errors.append(f"Required tier key '{key}' not found")

    if "requirements" in tier_config:
        reqs = tier_config["requirements"]
        if not isinstance(reqs, dict):
            errors.append("Tier requirements must be a dictionary")
        # Note: Our tier config has direct hardware requirements under "requirements"
        # rather than nested under "requirements.hardware", so no need to check for "hardware" section

    return errors


def get_profile_config(profile_name: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific profile.

    Args:
        profile_name: Name of the profile

    Returns:
        Profile configuration or None if not found
    """
    profile_config_path = find_config_file(f"{profile_name}.yml")
    if profile_config_path:
        return load_yaml_config(profile_config_path)

    # Try generic profile directory
    profile_path = find_config_file(f"profiles/{profile_name}.yml")
    if profile_path:
        return load_yaml_config(profile_path)

    return None


def apply_profile_param_merging(profile_configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply profile-level parameter merging to all test parameters in the profile configuration.

    This function merges profile-level parameters (especially requirements) with test-specific
    parameters for all tests, regardless of whether the profile has tiers or not.

    Args:
        profile_configs: Original profile configuration dictionary

    Returns:
        Dict[str, Any]: Profile configuration with merged parameters
    """
    import copy

    # Create a deep copy to avoid modifying the original
    merged_configs = copy.deepcopy(profile_configs)

    # Get profile-level parameters
    profile_params = merged_configs.get("params", {})
    profile_requirements = profile_params.get("requirements", {})

    # Process all suites and their test parameters
    for suite in merged_configs.get("suites", []):
        for sub_suite in suite.get("sub_suites", []):
            for test_name, test_config in sub_suite.get("tests", {}).items():
                if isinstance(test_config, dict) and "params" in test_config:
                    # Apply profile-level parameter merging to each test parameter
                    merged_params = []

                    for param in test_config["params"]:
                        param_copy = copy.deepcopy(param)

                        # Merge requirements: param-specific overrides profile-level
                        param_reqs = param_copy.get("requirements", {})
                        merged_reqs = copy.deepcopy(profile_requirements)
                        merged_reqs.update(param_reqs)
                        if merged_reqs:
                            param_copy["requirements"] = merged_reqs

                        # Merge other profile-level params (if any)
                        for k, v in profile_params.items():
                            if k != "requirements" and k not in param_copy:
                                param_copy[k] = copy.deepcopy(v)

                        merged_params.append(param_copy)

                    # Update the test config with merged parameters
                    test_config["params"] = merged_params
                    logger.debug(f"Merged profile-level params for {test_name}: {len(merged_params)} params")

    logger.debug("Profile parameter merging complete for profile")
    return merged_configs


def get_active_profile_configs() -> Optional[Dict[str, Any]]:
    """
    Get configuration for the currently active profile.

    Applies parameter merging for all profiles and tier filtering only if
    a highest tier is available and the profile has tiers defined.

    Returns:
        Active profile configuration (with merged params and filtered by tier if applicable)
        or None if no active profile
    """
    import os

    active_profile = os.environ.get("ACTIVE_PROFILE")
    if not active_profile:
        return None

    try:
        profile_configs = load_profile(active_profile)

        # Always apply profile-level parameter merging first
        profile_configs = apply_profile_param_merging(profile_configs)

        # Apply tier filtering only if highest tier is available and profile has tiers
        highest_tier = os.environ.get("ACTIVE_PROFILE_HIGHEST_TIER")
        profile_has_tiers = bool(profile_configs.get("params", {}).get("tiers"))

        if highest_tier and profile_has_tiers:
            logger.debug(f"Applying tier filtering for active profile: {active_profile}, tier: {highest_tier}")
            profile_configs = filter_profile_by_tier(profile_configs, highest_tier)
        elif profile_has_tiers:
            logger.debug(f"Profile {active_profile} has tiers but no highest tier determined - using all tiers")
        else:
            logger.debug(f"Profile {active_profile} has no tiers - using merged parameters as-is")

        return profile_configs
    except ValueError:
        logger.warning(f"Active profile '{active_profile}' not found")
        return None


def filter_profile_by_tier(profile_configs: Dict[str, Any], highest_tier: str) -> Dict[str, Any]:
    """
    Filter profile configuration to only include test parameters that match the system's highest tier.

    This ensures that only appropriate tests are included in the profile configuration,
    preventing lower-tier tests from being collected and executed on higher-tier systems.

    Args:
        profile_configs: Original profile configuration dictionary
        highest_tier: The system's highest qualifying tier

    Returns:
        Dict[str, Any]: Filtered profile configuration with only matching tier tests
    """
    import copy

    if not highest_tier:
        logger.warning("No highest tier provided, returning original profile")
        return profile_configs

    logger.info(f"Filtering profile for tier: {highest_tier}")

    # Create a deep copy to avoid modifying the original
    filtered_configs = copy.deepcopy(profile_configs)

    # Filter suites and their test parameters
    filtered_suites = []
    for suite in filtered_configs.get("suites", []):
        filtered_suite = copy.deepcopy(suite)
        filtered_sub_suites = []

        for sub_suite in suite.get("sub_suites", []):
            filtered_sub_suite = copy.deepcopy(sub_suite)
            filtered_tests = {}

            for test_name, test_config in sub_suite.get("tests", {}).items():
                if isinstance(test_config, dict) and "params" in test_config:
                    # Filter test parameters to only include those matching the highest tier
                    # Note: Parameter merging is already done by apply_profile_param_merging()
                    filtered_params = []

                    for param in test_config["params"]:
                        param_tiers = param.get("tiers", [])
                        if not param_tiers:
                            # If no tiers specified, include the parameter
                            filtered_params.append(param)
                            logger.debug(
                                f"Including {test_name} param {param.get('test_id', 'unknown')} - no tier restriction"
                            )
                        elif highest_tier in param_tiers:
                            # Only include if highest tier matches one of the parameter's tiers
                            filtered_params.append(param)
                            logger.debug(
                                f"Including {test_name} param {param.get('test_id', 'unknown')} - "
                                f"matches tier {highest_tier}"
                            )
                        else:
                            logger.debug(
                                f"Excluding {test_name} param {param.get('test_id', 'unknown')} - "
                                f"tiers {param_tiers} don't match {highest_tier}"
                            )

                    if filtered_params:
                        # Only include the test if it has parameters matching the tier
                        filtered_test_config = copy.deepcopy(test_config)
                        filtered_test_config["params"] = filtered_params
                        filtered_tests[test_name] = filtered_test_config
                        logger.info(
                            f"Test {test_name}: included {len(filtered_params)} of {len(test_config['params'])} "
                            f"parameters for tier {highest_tier}"
                        )
                    else:
                        logger.info(f"Test {test_name}: excluded - no parameters match tier {highest_tier}")
                else:
                    # Include tests without params structure
                    filtered_tests[test_name] = test_config

            if filtered_tests:
                # Only include sub-suite if it has tests with matching parameters
                filtered_sub_suite["tests"] = filtered_tests
                filtered_sub_suites.append(filtered_sub_suite)
            else:
                logger.info(
                    f"Sub-suite {sub_suite.get('name', 'unknown')}: excluded - no tests match tier {highest_tier}"
                )

        if filtered_sub_suites:
            # Only include suite if it has sub-suites with matching tests
            filtered_suite["sub_suites"] = filtered_sub_suites
            filtered_suites.append(filtered_suite)
        else:
            logger.info(f"Suite {suite.get('name', 'unknown')}: excluded - no sub-suites match tier {highest_tier}")

    filtered_configs["suites"] = filtered_suites

    total_original_tests = sum(
        len(sub_suite.get("tests", {}))
        for suite in profile_configs.get("suites", [])
        for sub_suite in suite.get("sub_suites", [])
    )
    total_filtered_tests = sum(
        len(sub_suite.get("tests", {}))
        for suite in filtered_configs.get("suites", [])
        for sub_suite in suite.get("sub_suites", [])
    )

    logger.info(
        f"Profile filtering complete: {total_filtered_tests} of {total_original_tests} "
        f"tests remain for tier {highest_tier}"
    )

    return filtered_configs


def load_profile(profile_name: str) -> Dict[str, Any]:
    """
    Load a profile configuration by name.

    Args:
        profile_name: Name of the profile to load

    Returns:
        Dict[str, Any]: The profile configuration
    """
    profiles_dict = list_profiles(include_examples=True)
    for profile_type, profile_list in profiles_dict.items():
        for profile in profile_list:
            if profile.get("name") == profile_name:
                return profile.get("configs", {})
    raise ValueError(f"Profile '{profile_name}' not found")


def backup_config(config_path: str, backup_dir: Optional[str] = None) -> str:
    """
    Create a backup of a configuration file.

    Args:
        config_path: Path to the configuration file to backup
        backup_dir: Directory to store backup (defaults to same directory)

    Returns:
        Path to the backup file
    """
    if backup_dir is None:
        backup_dir = os.path.dirname(config_path)

    filename = os.path.basename(config_path)
    name, ext = os.path.splitext(filename)

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{name}_backup_{timestamp}{ext}"
    backup_path = os.path.join(backup_dir, backup_filename)

    import shutil

    shutil.copy2(config_path, backup_path)

    logger.info(f"Configuration backed up to {backup_path}")
    return backup_path


def restore_config(backup_path: str, original_path: str) -> None:
    """
    Restore a configuration file from backup.

    Args:
        backup_path: Path to the backup file
        original_path: Path where to restore the file
    """
    import shutil

    shutil.copy2(backup_path, original_path)
    logger.info(f"Configuration restored from {backup_path} to {original_path}")


def get_caller_module_dir() -> str:
    """
    Get the directory of the module that called this function.
    Useful for finding config files relative to the calling module.

    Returns:
        Directory path of the calling module
    """
    frame = inspect.currentframe()
    try:
        # Go up the call stack to find the caller
        caller_frame = frame.f_back.f_back
        caller_file = caller_frame.f_code.co_filename
        return os.path.dirname(os.path.abspath(caller_file))
    finally:
        del frame


# Additional functions migrated from old flat config.py file
def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML file into a dictionary.

    Args:
        file_path: Path to the YAML file

    Returns:
        Dict[str, Any]: The loaded YAML content

    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file contains invalid YAML
    """
    import yaml

    try:
        with open(file_path, "r") as f:
            content = f.read()
            # Check for Python-style docstrings and warn
            if content.lstrip().startswith('"""') or content.lstrip().startswith("'''"):
                logger.warning(
                    f"YAML file {file_path} appears to contain a Python docstring "
                    f"which is invalid YAML syntax. Use '#' for YAML comments."
                )
            return yaml.safe_load(content) or {}
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        # Add more helpful message for common errors
        if "expected '<document start>'" in str(e) and ("'''" in str(e) or '"""' in str(e)):
            logger.error(
                f"Python docstrings (''' or \"\"\") are not valid in YAML files. Use '#' for comments in {file_path}"
            )
        raise


def load_test_config(test_name: str, suite_path: str) -> Dict[str, Any]:
    """
    Load test configuration for a specific test.

    Args:
        test_name: Name of the test
        suite_path: Path to the suite directory

    Returns:
        Dict[str, Any]: Test configuration
    """
    config_file = os.path.join(suite_path, f"{test_name}.yaml")
    if os.path.exists(config_file):
        return load_yaml_file(config_file)
    else:
        return {}


def load_test_configurations(test_name: str, config_path: str = None) -> List[Dict[str, Any]]:
    """
    Load test configurations from config.yml for a specific test.

    This function extracts configurations for a test from the config.yml file in the same
    directory as the test file or the explicitly provided path, and applies any qualification-specific
    overrides if running through a qualification.

    Profile requirements are used as defaults for test params and can be overridden by
    test-specific requirements if defined.

    Args:
        test_name: Name of the test (without extension)
        config_path: Optional explicit path to the config.yml file

    Returns:
        List[Dict[str, Any]]: List of test configurations
    """
    import copy
    import logging
    import os

    logger = logging.getLogger(__name__)

    # Determine the config path to use
    if not config_path:
        # Get the caller's file location to find the config relative to it
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        config_path = os.path.join(caller_dir, "config.yml")
        logger.info(f"Looking for config at {config_path} relative to caller: {caller_file}")
    else:
        logger.info(f"Using provided config path: {config_path}")

    suite_configs = {}
    config_exists = os.path.exists(config_path)
    if config_exists:
        try:
            suite_configs = load_yaml_file(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load or parse config.yml: {e}")
            suite_configs = {}

    # Try to get test configs from config.yml if present
    test_configs = {}
    if suite_configs and "tests" in suite_configs and test_name in suite_configs["tests"]:
        test_configs = suite_configs["tests"][test_name]
        if "params" in test_configs and test_configs["params"]:
            base_configs = copy.deepcopy(test_configs["params"])
            logger.info(f"Loaded {len(base_configs)} base configurations for {test_name} from config.yml")
        else:
            base_configs = []
    else:
        base_configs = []

    # Extract suite and sub-suite from config path
    path_parts = config_path.split(os.sep)
    try:
        suites_index = path_parts.index("suites")
        suite_name = path_parts[suites_index + 1] if suites_index + 1 < len(path_parts) else ""
        sub_suite_name = path_parts[suites_index + 2] if suites_index + 2 < len(path_parts) else ""
        logger.info(f"Determined suite={suite_name}, sub_suite={sub_suite_name} for test {test_name}")
    except ValueError:
        logger.warning(f"Could not determine suite/sub-suite from path: {config_path}")
        suite_name = ""
        sub_suite_name = ""

    # Apply profile overrides if running with an active profile
    profile_configs = get_active_profile_configs()
    if profile_configs and suite_name and sub_suite_name:
        profile_name = os.environ.get("ACTIVE_PROFILE", "unknown")
        logger.info(f"Applying profile overrides from {profile_name} for {test_name} in {suite_name}/{sub_suite_name}")

        # Find suite and sub-suite specific requirements
        for qual_suite in profile_configs.get("suites", []):
            if qual_suite.get("name") != suite_name:
                continue
            for qual_sub_suite in qual_suite.get("sub_suites", []):
                if qual_sub_suite.get("name") != sub_suite_name:
                    continue
                tests_config = qual_sub_suite.get("tests", {})
                if test_name in tests_config:
                    qual_test_config = tests_config[test_name]

                    # Return profile's param list directly - params are already pre-merged during profile filtering
                    # This eliminates redundant merging that was happening later in pytest_parameterization.py
                    if (
                        isinstance(qual_test_config, dict)
                        and "params" in qual_test_config
                        and qual_test_config["params"]
                    ):
                        logger.info(f"Using pre-merged param list for {test_name} from profile {profile_name}")
                        return qual_test_config["params"]
                    else:
                        logger.info(f"No params defined in profile for {test_name}, will use base configs or default.")
                break
            break

    # If no configs found in config.yml or profile, return empty list (or optionally a default config)
    # Merge global profile-level params into each test param if present
    if base_configs:
        profile_params = {}
        profile_requirements = {}
        if suite_configs.get("profiles"):
            # If running without ACTIVE_PROFILE, use default/global profile params if present
            default_profile = suite_configs["profiles"].get("default", {})
            profile_params = default_profile.get("params", {})
            profile_requirements = profile_params.get("requirements", {})
        elif suite_configs.get("params"):
            # Support direct params in config.yml (for suite-level configurations)
            profile_params = suite_configs.get("params", {})
            profile_requirements = profile_params.get("requirements", {})

        if profile_params or profile_requirements:
            merged_base_configs = []
            for param in base_configs:
                param_copy = copy.deepcopy(param)
                # Merge requirements: param-specific overrides profile-level
                param_reqs = param_copy.get("requirements", {})
                merged_reqs = copy.deepcopy(profile_requirements)
                merged_reqs.update(param_reqs)
                if merged_reqs:
                    param_copy["requirements"] = merged_reqs
                # Merge other profile-level params (if any)
                for k, v in profile_params.items():
                    if k != "requirements" and k not in param_copy:
                        param_copy[k] = copy.deepcopy(v)
                merged_base_configs.append(param_copy)
            logger.info(
                f"Final configuration has {len(merged_base_configs)} parameter sets for {test_name} "
                f"with profile param merging"
            )
            return merged_base_configs
        else:
            logger.info(
                f"Final configuration has {len(base_configs)} parameter sets for {test_name} "
                f"without profile param merging"
            )
            return base_configs

    logger.warning(f"No configurations found for test '{test_name}' in config.yml or active profile overrides.")
    return []


def get_profile_directories(include_examples: bool = False) -> List[tuple[str, str]]:
    """
    Get all profile directories.

    Args:
        include_examples: Whether to include example profiles

    Returns:
        List of (profile_path, profile_name) tuples
    """
    from sysagent.utils.config.config_loader import discover_entrypoint_paths

    profile_dirs = []
    entrypoint_paths = discover_entrypoint_paths("configs")

    for base_path in entrypoint_paths:
        profiles_path = os.path.join(base_path, "profiles")
        if os.path.exists(profiles_path):
            for item in os.listdir(profiles_path):
                item_path = os.path.join(profiles_path, item)
                if os.path.isdir(item_path):
                    # Skip examples unless requested
                    if not include_examples and item.lower() == "examples":
                        continue
                    logger.debug(f"Found profile directory: {item_path}")
                    profile_dirs.append((item, item_path))

    return profile_dirs


def get_suite_directory(suite_name: str) -> Optional[str]:
    """
    Get the directory path for a specific test suite.

    Args:
        suite_name: Name of the test suite

    Returns:
        Path to the suite directory or None if not found
    """
    from sysagent.utils.config.config_loader import discover_entrypoint_paths

    entrypoint_paths = discover_entrypoint_paths("suites")
    for base_path in entrypoint_paths:
        suites_path = os.path.join(base_path, suite_name)
        if os.path.exists(suites_path):
            return suites_path

    return None


def get_core_directory() -> str:
    """
    Get core directory.

    Returns:
        str: Path to the core directory or None if not found
    """
    from sysagent.utils.config.config_loader import discover_entrypoint_paths

    entrypoint_paths = discover_entrypoint_paths("configs")

    for base_path in entrypoint_paths:
        core_path = os.path.join(base_path, "core")
        if os.path.exists(core_path):
            return core_path

    return None


def get_sysagent_core_directory() -> str:
    """
    Get the core directory from the sysagent package specifically.

    This function always returns the sysagent core directory, not extension packages.
    Useful for accessing shared resources like allure3 patches that should come from
    the core framework.

    Returns:
        str: Path to the sysagent core directory or None if not found
    """
    import importlib.util

    try:
        # Find sysagent package location
        spec = importlib.util.find_spec("sysagent")
        if spec and spec.origin:
            # Get the sysagent package directory
            sysagent_dir = os.path.dirname(spec.origin)

            # Look for configs/core in sysagent
            core_path = os.path.join(sysagent_dir, "configs", "core")
            if os.path.exists(core_path):
                return core_path

            # Alternative: check if we're in development and look in src/sysagent
            parent_dir = os.path.dirname(sysagent_dir)
            if os.path.basename(parent_dir) == "src":
                core_path = os.path.join(parent_dir, "sysagent", "configs", "core")
                if os.path.exists(core_path):
                    return core_path
    except Exception as e:
        logger.debug(f"Failed to locate sysagent core directory: {e}")

    return None


def get_reports_directory() -> str:
    """
    Get reports directory with prioritized lookup for extensions.

    When extensions are present, prioritizes extension package configurations
    over core sysagent configurations. Falls back to sysagent if extension
    configs are not available.

    Returns:
        str: Path to the reports directory or None if not found
    """
    from sysagent.utils.config.config_loader import (
        _has_extensions,
        discover_entrypoint_paths,
    )

    entrypoint_paths = discover_entrypoint_paths("configs")

    # If we have extensions, prioritize extension packages over sysagent
    if _has_extensions():
        # Check extension packages first (non-sysagent paths)
        logger.debug(f"Extensions detected, prioritizing {len(entrypoint_paths)} extension reports directories:")

        # First pass: Look for extension paths only (exclude sysagent paths)
        for base_path in entrypoint_paths:
            logger.debug(f"Checking for extension reports in: {base_path}")
            # Skip sysagent paths in the first pass
            if "sysagent" in str(base_path):
                continue

            reports_path = os.path.join(base_path, "reports")
            # Check if the allure config specifically exists for complete validation
            allure_config = os.path.join(reports_path, "allure", "allurerc.mjs")
            if os.path.exists(reports_path) and os.path.exists(allure_config):
                logger.debug(f"Using extension reports directory: {reports_path}")
                return reports_path

        # Second pass: Fall back to sysagent if no extension reports found
        for base_path in entrypoint_paths:
            logger.debug(f"Checking for sysagent reports in: {base_path}")
            # Only check sysagent paths in the fallback pass
            if "sysagent" in str(base_path):
                reports_path = os.path.join(base_path, "reports")
                if os.path.exists(reports_path):
                    logger.debug(f"Using fallback sysagent reports directory: {reports_path}")
                    return reports_path
    else:
        # For minimal installations, use standard order (sysagent only)
        logger.debug("No extensions detected, using standard reports directory lookup")
        for base_path in entrypoint_paths:
            reports_path = os.path.join(base_path, "reports")
            if os.path.exists(reports_path):
                logger.debug(f"Using sysagent reports directory: {reports_path}")
                return reports_path

    logger.warning("No reports directory found in any configuration paths")
    return None


def list_suites() -> List[str]:
    """
    List all available test suites.

    Returns:
        List of suite names
    """
    from sysagent.utils.config.config_loader import discover_entrypoint_paths

    suites = []
    entrypoint_paths = discover_entrypoint_paths("suites")

    for base_path in entrypoint_paths:
        suites_path = os.path.join(base_path, "suites")
        if os.path.exists(suites_path):
            for item in os.listdir(suites_path):
                item_path = os.path.join(suites_path, item)
                if os.path.isdir(item_path):
                    suites.append(item)

    return list(set(suites))  # Remove duplicates


def override_test_config(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override test configuration with another configuration.

    Args:
        base_config: Base configuration
        override_config: Configuration to override with

    Returns:
        Merged configuration
    """
    import copy

    result = copy.deepcopy(base_config)

    def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update a dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    deep_update(result, override_config)
    return result


def ensure_dir_permissions(path, uid=None, gid=None, mode=0o775):
    """
    Ensure directory exists with proper permissions.

    Args:
        path: Directory path to create/modify
        uid: User ID to set (optional)
        gid: Group ID to set (optional)
        mode: Permission mode (default: 0o775)
    """
    os.makedirs(path, exist_ok=True)
    os.chmod(path, mode)
    if uid is not None and gid is not None:
        try:
            os.chown(path, uid, gid)
        except PermissionError:
            pass


def get_config_hash(config: dict, keys: list = None) -> str:
    """
    Generate a short hash for a config dictionary using specified keys.

    Args:
        config: Configuration dictionary
        keys: List of keys to include in hash (optional, defaults to all)

    Returns:
        Short hash string
    """
    import hashlib
    import json

    # Use specified keys or all keys if none specified
    if keys is None:
        hash_dict = config
    else:
        hash_dict = {k: config.get(k) for k in keys if k in config}

    # Create a deterministic string representation
    config_str = json.dumps(hash_dict, sort_keys=True, separators=(",", ":"))

    # Generate hash and return first 8 characters
    return hashlib.md5(config_str.encode(), usedforsecurity=False).hexdigest()[:8]


def deep_update(base_dict: dict, update_dict: dict) -> None:
    """
    Recursively update a dictionary.

    Args:
        base_dict: Dictionary to update
        update_dict: Dictionary with updates to apply
    """
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def load_test_configs(test_name: str) -> List[Dict[str, Any]]:
    """
    Load test configurations from config.yml for a specific test.

    This function extracts configurations for a test from the config.yml file,
    combining the test-specific settings with profile settings to generate
    a complete test configuration for each profile.

    Args:
        test_name: Name of the test (without extension)

    Returns:
        List[Dict[str, Any]]: List of test configurations for each profile
    """
    # Find the suite directory based on the test name
    suite_dir = None
    for suite in list_suites():
        logger.debug(f"Checking suite: {suite} for test: {test_name}")
        suite_path = get_suite_directory(suite)
        if suite_path:
            test_path = os.path.join(suite_path, f"{test_name}.py")
            if os.path.exists(test_path):
                suite_dir = suite_path
                break

    if not suite_dir:
        logger.error(f"Suite directory not found for test: {test_name}")
        return []

    # Get the path to the config.yml file in the suite directory
    config_path = os.path.join(suite_dir, "config.yml")

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return []

    # Load the configuration file
    configs = load_yaml_file(config_path)

    # Check if the test is defined in the configs
    if "tests" not in configs or test_name not in configs["tests"]:
        logger.warning(f"Test '{test_name}' not found in config.yml")
        return []

    # Get test-specific configurations
    test_configs = configs["tests"][test_name]

    # Get profiles from configs
    profiles = configs.get("profiles", {})

    # Generate a configurations for each profile
    result = []
    for profile_name, profile_configs in profiles.items():
        # Create a new configurations by combining profile and test configs
        combined_configs = {**profile_configs, **test_configs}

        # Add profile name to display name if not already specified
        if "display_name" in combined_configs:
            combined_configs["display_name"] = (
                f"{combined_configs['display_name']} - {profile_name.capitalize()} Profile"
            )
        else:
            combined_configs["display_name"] = f"{test_name} - {profile_name.capitalize()} Profile"

        result.append(combined_configs)

    return result


def apply_profile_overrides(
    test_configs: List[Dict[str, Any]],
    profile_configs: Dict[str, Any],
    test_name: str,
    suite_name: str,
    sub_suite_name: str,
) -> List[Dict[str, Any]]:
    """
    Apply profile-specific configuration overrides to test configurations.

    Args:
        test_configs: List of base test configurations
        profile_configs: Profile configurations
        test_name: Name of the test
        suite_name: Name of the suite
        sub_suite_name: Name of the sub-suite

    Returns:
        List[Dict[str, Any]]: The overridden test configurations
    """
    import logging

    logger = logging.getLogger(__name__)

    # Skip if profile config is None
    if not profile_configs:
        return test_configs

    # Find the test configuration in the profile config
    for suite in profile_configs.get("suites", []):
        if suite.get("name") != suite_name:
            continue

        for sub_suite in suite.get("sub_suites", []):
            if sub_suite.get("name") != sub_suite_name:
                continue

            # Check if the test exists in the profile config
            tests_config = sub_suite.get("tests", {})
            if not tests_config or test_name not in tests_config:
                logger.debug(f"Test {test_name} not found in profile configs")
                return test_configs

            # If profile has its own params list for the test, use that instead
            test_config = tests_config[test_name]
            if "params" in test_config:
                logger.info(f"Using profile's param list for {test_name} in {suite_name}/{sub_suite_name}")
                return test_config["params"]

    # No complete override found, return original configs
    return test_configs
