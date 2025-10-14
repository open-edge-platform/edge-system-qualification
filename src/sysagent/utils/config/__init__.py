# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Configuration utilities package.

Provides utilities for loading and managing configuration from various sources
including pyproject.toml, YAML files, environment variables, and entrypoints.
"""

# Import local modules directly without backward compatibility
from .config import *
from .config_loader import *

# Re-export all functions and classes
__all__ = [
    # From config
    "setup_data_dir",
    "get_thirdparty_dir",
    "get_cache_dir",
    "get_logs_dir",
    "get_results_dir",
    "load_yaml_config",
    "save_yaml_config",
    "get_config_checksum",
    "validate_config_schema",
    "merge_configs",
    "get_environment_config",
    "apply_environment_overrides",
    "find_config_file",
    "load_config_with_fallback",
    "get_config_value",
    "set_config_value",
    "list_tiers",
    "get_tier_config",
    "validate_tier_config",
    "get_project_name",
    "discover_entrypoint_paths",
    "list_profiles",
    "get_profile_config",
    "get_active_profile_configs",
    "get_suite_directory",
    "list_suites",
    "load_test_config",
    "load_test_configurations",
    "update_dict_recursively",
    "filter_profile_by_tier",
    "get_core_directory",
    "get_sysagent_core_directory",
    "get_reports_directory",
    "override_test_config",
    "get_config_hash",
    "deep_update",
    "load_test_configs",
    "apply_profile_overrides",
    # From config_loader
    "load_yaml_file",
    "get_dist_version",
    "ensure_dir_permissions",
    "load_tool_config",
    "load_merged_tool_config",
    "get_dist_name",
    "discover_project_roots",
    "load_pyproject_config",
    "get_allure_version",
    "get_node_version",
    "get_entrypoint_config",
    "find_config_files",
    "get_runtime_config",
    "validate_config_integrity",
    "get_cli_aware_project_name",
]
