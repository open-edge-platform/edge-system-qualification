# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Suite management fixtures for the core test framework.

These fixtures provide access to suite-level configuration and metadata.
"""

import logging
import os
from typing import Any, Dict

import pytest
import yaml

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def suite_configs(request):
    """
    Load the sub-suite configurations file.

    If running in a qualification, overlay the qualification's KPI overrides.
    The function scope ensures that the configurations is reloaded for each test,
    which is important when running multiple qualifications in sequence.
    """
    # Force reload the config module to ensure fresh configuration
    try:
        import importlib

        import sysagent.utils.config.config

        importlib.reload(sysagent.utils.config.config)
        logger.debug("Reloaded config module for suite_configs fixture")
    except Exception as e:
        logger.warning(f"Failed to reload config module: {e}")

    # Get active profile to ensure we're using the right one
    profile_name = os.environ.get("ACTIVE_PROFILE", "none")
    logger.info(f"Loading suite_configs with active profile: {profile_name}")

    # Determine the suite and sub-suite from the test file path
    if not hasattr(request, "node") or not hasattr(request.node, "fspath"):
        logger.warning("Cannot determine test file path - no node.fspath available")
        return {}

    test_file = request.node.fspath.strpath if hasattr(request.node.fspath, "strpath") else str(request.node.fspath)
    # The path structure is expected to be:
    # .../suites/suite_name/sub_suite_name/test_file.py
    parts = test_file.split(os.sep)

    # Find the 'suites' directory in the path to determine suite and sub-suite
    suite_name = None
    sub_suite_name = None

    try:
        if "suites" in parts:
            suites_index = parts.index("suites")
            if len(parts) > suites_index + 1:
                suite_name = parts[suites_index + 1]
            if len(parts) > suites_index + 2:
                sub_suite_name = parts[suites_index + 2]
    except Exception as e:
        logger.warning(f"Failed to determine suite/sub-suite from path: {e}")

    if not suite_name or not sub_suite_name:
        logger.warning(f"Could not determine suite or sub-suite from path: {test_file}")
        return {}

    logger.debug(f"Determined suite: {suite_name}, sub-suite: {sub_suite_name}")

    # Load base configs from the sub-suite
    config_dir = os.path.dirname(test_file)
    config_path = os.path.join(config_dir, "config.yml")

    if not os.path.exists(config_path):
        logger.debug(f"Config file not found at: {config_path}")
        # Try looking one directory up (for sub-suites)
        parent_dir = os.path.dirname(config_dir)
        config_path = os.path.join(parent_dir, "config.yml")
        if os.path.exists(config_path):
            logger.info(f"Found config file in parent directory: {config_path}")
        else:
            logger.debug(f"Config file not found in parent directory either: {parent_dir}")
            return {}

    logger.debug(f"Using configuration file: {config_path}")
    logger.debug(f"Loading configuration from: {config_path}")

    try:
        from sysagent.utils.config import load_yaml_file

        base_configs = load_yaml_file(config_path)
        logger.debug(f"Loaded configuration keys: {list(base_configs.keys())}")
        if "kpi" in base_configs:
            logger.debug(f"Found KPI definitions: {list(base_configs['kpi'].keys())}")
    except ImportError:
        # Fallback to direct loading if import fails
        try:
            with open(config_path, "r") as f:
                base_configs = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            return {}

    # Get active profile configs if available - using fresh import
    try:
        from sysagent.utils.config import get_active_profile_configs

        profile_configs = get_active_profile_configs()
    except ImportError:
        logger.warning("Could not import get_active_profile_configs - skipping profile overrides")
        profile_configs = None

    if profile_configs:
        logger.info(f"Applying profile overrides for {sub_suite_name} from: {profile_name}")

        # Find the matching sub-suite configuration in the profile
        for suite in profile_configs.get("suites", []):
            if suite.get("name") != suite_name:
                continue

            for sub_suite in suite.get("sub_suites", []):
                if sub_suite.get("name") != sub_suite_name:
                    continue

                # Apply KPI overrides from the profile configs
                if "kpi" in sub_suite:
                    logger.info(f"Found KPI definitions in profile configs: {profile_name}")

                    # For a complete KPI section, replace the entire KPI section
                    base_configs["kpi"] = sub_suite["kpi"]
                    logger.info(f"Replaced KPI definitions with profile from: {profile_name}")

    return base_configs


@pytest.fixture(scope="function")
def get_kpi_config(suite_configs, configs):
    """
    Fixture to get KPI configuration by name.

    Args:
        name: Name of the KPI configuration

    Returns:
        Dict[str, Any]: The KPI configuration with any overrides applied
    """

    def _get_kpi_config(name: str) -> Dict[str, Any]:
        # Get base KPI config from suite_configs
        if name not in suite_configs.get("kpi", {}):
            logger.warning(f"KPI configuration not found: {name}")
            return {}

        # Create a deep copy of the base KPI to avoid modifying the original
        import copy

        base_kpi = copy.deepcopy(suite_configs["kpi"][name])

        # Check for test-specific KPI overrides in the test configs
        # Support both singular (kpi_override) and plural (kpi_overrides) forms
        kpi_overrides = {}

        if "kpi_overrides" in configs and isinstance(configs["kpi_overrides"], dict):
            kpi_overrides = configs["kpi_overrides"]
        elif "kpi_override" in configs and isinstance(configs["kpi_override"], dict):
            kpi_overrides = configs["kpi_override"]

        if name in kpi_overrides:
            logger.info(f"Applying test-specific override for KPI: {name}")
            # Use deep_update for proper nested dictionary updates
            from sysagent.utils.config import deep_update

            deep_update(base_kpi, kpi_overrides[name])
            logger.debug(f"KPI after override: {base_kpi}")

        return base_kpi

    return _get_kpi_config
