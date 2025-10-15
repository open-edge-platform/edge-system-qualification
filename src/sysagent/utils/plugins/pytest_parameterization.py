# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Parameterization utilities for core testing framework.

This module provides fixtures and hooks for automatically parameterizing tests
with configuration data from config.yml files.
"""

import importlib
import inspect
import logging
import os
import traceback
from typing import Any, Callable, Dict, List

import pytest

logger = logging.getLogger(__name__)


def pytest_generate_tests(metafunc):
    test_func = metafunc.function
    test_name = test_func.__name__

    def ensure_param(configs, key, value):
        for config in configs:
            if key not in config:
                config[key] = value

    if not test_name.startswith("test_") or "configs" not in metafunc.fixturenames:
        logger.debug(
            f"[pytest_generate_tests] Skipping auto-parameterization for {test_name} "
            "- no configs parameter or not a test function"
        )
        return

    from sysagent.utils.cli.filters import consolidate_profile_parameters
    from sysagent.utils.config import get_active_profile_configs
    from sysagent.utils.testing.tier_validator import get_highest_matching_tier

    profile_configs = get_active_profile_configs()

    # Check for test filters from environment
    filters = None
    if "CORE_TEST_FILTERS" in os.environ:
        import json

        try:
            filters = json.loads(os.environ["CORE_TEST_FILTERS"])
            logger.debug(f"Loaded test filters from environment: {filters}")
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse test filters from environment: {e}")

    if profile_configs:
        # Get highest tier for tier-based profiles
        highest_tier = get_highest_matching_tier(profile_configs)

        # Use consolidated parameter extraction regardless of tier presence
        logger.debug(f"Merged profile-level params for {test_name}: Consolidating parameters")
        all_params = consolidate_profile_parameters(profile_configs, filters, highest_tier)

        # Filter for this specific test
        test_params = [item for item in all_params if item["test"] == test_name]

        if test_params:
            # Extract just the parameter configurations
            filtered_params = [item["param"] for item in test_params]

            logger.debug("Profile parameter merging complete for profile")
            if highest_tier:
                logger.debug(f"Profile {profile_configs.get('name', 'unknown')} using tier: {highest_tier}")
            else:
                logger.debug(
                    f"Profile {profile_configs.get('name', 'unknown')} has no tiers - using merged parameters as-is"
                )

            # Add filter information to logs
            if filters:
                logger.info(
                    f"Test {test_name}: Applied filters {filters}, found {len(filtered_params)} matching parameters"
                )

            if filtered_params:
                ids = [p.get("test_id", f"config_{i}") for i, p in enumerate(filtered_params)]
                metafunc.parametrize("configs", filtered_params, ids=ids)
                logger.debug(f"Parameterizing {test_name} with {len(filtered_params)} configurations: {ids}")
            else:
                logger.debug(f"Test {test_name}: No parameters match the specified criteria - test will be skipped")
                metafunc.parametrize("configs", [])
        else:
            # No tests found for this test function
            if filters:
                logger.debug(f"Test {test_name}: No parameters found matching filters {filters}")
            else:
                logger.debug(f"No test params found for {test_name} in profile")
            metafunc.parametrize("configs", [])

    else:
        # Non-profile mode: Get test configurations for this test
        configs = get_test_configs_for_test(test_func)

        # If configs were found, apply them as parameters
        if configs:
            # Append additional parameters if needed
            ensure_param(configs, "tier", os.environ.get("ACTIVE_PROFILE_HIGHEST_TIER", "entry"))

            # Extract IDs from config names for better test identification
            ids = [config.get("name", f"config_{i}") for i, config in enumerate(configs)]
            logger.debug(
                f"Final configuration has {len(configs)} parameter sets for {test_name} without profile param merging"
            )
            logger.debug(f"Loaded {len(configs)} configurations for {test_name}")
            logger.debug(f"Parameterizing {test_name} with {len(configs)} configurations: {ids}")

            # Apply the parameterization
            metafunc.parametrize("configs", configs, ids=ids)
        else:
            logger.warning(f"No configurations found for {test_name} - test will run without parameterization")
            # Provide a default empty configs to avoid errors
            metafunc.parametrize("configs", [{"display_name": f"{test_name} - Default"}])


def get_test_configs_for_test(test_func: Callable) -> List[Dict[str, Any]]:
    """
    Get test configurations for a test function by looking up its config.yml file.

    Args:
        test_func: The test function to get configurations for

    Returns:
        List[Dict[str, Any]]: List of test configurations
    """
    test_name = test_func.__name__

    # Force reload the configuration module to clear any cached data
    try:
        import sysagent.utils.config.config

        importlib.reload(sysagent.utils.config.config)
        logger.debug(f"Reloaded config module for {test_name}")
    except Exception as e:
        logger.warning(f"Failed to reload config module: {e}")

    profile_name = os.environ.get("ACTIVE_PROFILE", "none")
    logger.info(f"Loading test configurations for {test_name} with profile: {profile_name}")

    # Get the test file location using inspect
    module = inspect.getmodule(test_func)
    if not module or not hasattr(module, "__file__"):
        logger.error(f"Could not determine module file for {test_name}")
        return [{"name": "error", "display_name": f"{test_name} - Error Finding Configs"}]

    test_file = module.__file__
    test_dir = os.path.dirname(os.path.abspath(test_file))
    logger.info(f"Test file: {test_file}")

    # Look for config.yml in the test directory
    config_path = os.path.join(test_dir, "config.yml")
    logger.info(f"Using config path: {config_path}")

    # Use the load_test_configurations function with the determined config path
    try:
        from sysagent.utils.config import load_test_configurations

        configs = load_test_configurations(test_name, config_path)
        logger.info(f"Loaded {len(configs)} configurations for {test_name}")
        return configs
    except Exception as e:
        logger.error(f"Error loading test configurations for {test_name}: {e}")
        traceback.print_exc()
        # Return an empty list to indicate failure
        return []


@pytest.fixture(scope="function")
def request_fixture(request):
    """
    Provide access to the request object for use in parameterized tests.

    This fixture allows tests to access the request object, which contains
    information about the current test run, including parameters.
    """
    return request
