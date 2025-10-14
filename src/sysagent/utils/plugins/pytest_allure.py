# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Allure integration hooks for core testing framework.

This module provides pytest hooks for Allure reporting that dynamically set
suite names and improve the test output formatting.

Key features:
- Automatic suite hierarchy based on test file path structure
- Dynamic metadata application from test configurations
- Automatic tier labels with human-readable display names
- Profile-based configuration resolution
"""

import logging
import os
import sys
from pathlib import Path

import allure
import pytest

from sysagent.utils.core import shared_state

logger = logging.getLogger(__name__)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """
    Hook to dynamically set Allure suite names and metadata.

    This hook:
    1. Sets suite hierarchy based on test file path structure
    2. Applies comprehensive Allure metadata from test configurations
    """
    # Set suite names based on path structure
    _set_allure_suite_names(item)
    logger.debug(f"Set Allure suite names for test: {item.name}")

    # Apply Allure metadata from configurations
    _apply_allure_metadata_from_configs(item)
    logger.debug(f"Applied Allure metadata for test: {item.name}")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    """
    Hook that runs during test execution.

    Re-apply Allure title during test execution to ensure it's not overridden.
    """
    # Re-apply the title during test execution to ensure it persists
    _apply_allure_title_only(item)

    # Skip all remaining tests if an interrupt occurred
    if shared_state.INTERRUPT_OCCURRED:
        error_message = "Skipping test due to previous interrupt"
        logger.warning(error_message)
        raise RuntimeError(error_message)
    else:
        # Proceed with the test execution
        yield


def _apply_allure_title_only(item):
    """
    Apply only the Allure title from test configurations.

    This is a lightweight version that only sets the title during test execution
    to ensure it's not overridden by pytest's internal processes.

    For profile-based tests, it will resolve the tier and configuration metadata
    even when tests are interrupted to ensure proper title reporting.

    Args:
        item: pytest test item containing configuration data
    """
    try:
        # Extract configs from test function parameters
        configs = None

        # Look for configs in function parameters or fixtures
        if hasattr(item, "callspec") and hasattr(item.callspec, "params"):
            # Try both 'configs' (plural) and 'config' (singular) parameter names
            configs = item.callspec.params.get("configs") or item.callspec.params.get("config")

        # If no configs found in params, resolve from profile for qualification tests
        if not configs:
            configs = _resolve_profile_configs_for_test(item)

        if not configs:
            return

        # Re-apply dynamic Allure title to ensure it persists
        if "display_name" in configs:
            test_id = configs.get("test_id", "T0000")
            display_name = configs["display_name"]
            title = f"{test_id} - {display_name}"

            # Re-set the title during test execution
            allure.dynamic.title(title)

            logger.debug(f"Re-applied Allure title during execution: {title}")

    except ImportError:
        logger.debug("Allure not available, skipping title re-application")
    except Exception as e:
        logger.warning(f"Failed to re-apply Allure title for {item.name}: {e}")


def _resolve_profile_configs_for_test(item):
    """
    Resolve profile-based test configurations for a test item.

    This function replicates the logic from pytest_parameterization.py but is
    designed to work in the Allure metadata context, even when tests are interrupted.

    Args:
        item: pytest test item

    Returns:
        dict: Test configuration dictionary, or None if not found
    """
    try:
        # Get test function name
        test_func = item.function
        test_name = test_func.__name__

        # Only process test functions
        if not test_name.startswith("test_"):
            return None

        # Import the required functions
        from sysagent.utils.config import get_active_profile_configs
        from sysagent.utils.testing.tier_validator import (
            get_highest_matching_tier,
            get_test_params_tiers,
        )

        # Get profile configs and highest tier
        profile_configs = get_active_profile_configs()
        if not profile_configs:
            return None

        highest_tier = get_highest_matching_tier(profile_configs)
        if not highest_tier:
            return None

        # Get all test params for this profile (all passing params)
        all_params = get_test_params_tiers(profile_configs, passed_only=True)

        # Filter for this test only - use best available tier for this test
        test_params = [item_data for item_data in all_params if item_data["test"] == test_name]

        if test_params:
            # Find the highest priority tier (lower priority number = higher priority)
            best_tier_priority = min(item_data["tier_result"]["priority"] for item_data in test_params)
            filtered_params = [
                item_data["param"]
                for item_data in test_params
                if item_data["tier_result"]["priority"] == best_tier_priority
            ]

            if filtered_params:
                # Use the first parameter configuration
                config = filtered_params[0].copy()
                return config

        # No config found
        return None

    except Exception as e:
        logger.warning(f"Failed to resolve profile configs for {item.name}: {e}")
        return None


def _apply_allure_metadata_from_configs(item):
    """
    Apply comprehensive Allure metadata based on test configurations.

    This function handles all metadata setting including title, description, severity,
    labels, and parameters. It's designed to work reliably in the pytest setup phase
    before test execution begins.

    For profile-based tests, it will resolve the tier and configuration metadata
    even when tests are interrupted to ensure proper Allure reporting.

    Args:
        item: pytest test item containing configuration data
    """
    try:
        # Extract configs from test function parameters
        configs = None

        # Look for configs in function parameters or fixtures
        if hasattr(item, "callspec") and hasattr(item.callspec, "params"):
            # Try both 'configs' (plural) and 'config' (singular) parameter names
            configs = item.callspec.params.get("configs") or item.callspec.params.get("config")

        # If no configs found in params, resolve from profile for qualification tests
        if not configs:
            configs = _resolve_profile_configs_for_test(item)

        if not configs:
            logger.debug(f"No configs found for test: {item.name}")
            return

        logger.debug(f"Applying Allure metadata for test: {item.name}")
        logger.debug(f"Config keys: {list(configs.keys())}")

        # Set dynamic Allure title with test ID and display name
        if "display_name" in configs:
            test_id = configs.get("test_id", "T0000")
            display_name = configs["display_name"]
            title = f"{test_id} - {display_name}"

            allure.dynamic.title(title)
            allure.dynamic.label("test_title", title)

            logger.debug(f"Set Allure title to: {title}")

        # Set dynamic Allure description
        if "description" in configs:
            description = configs["description"]
            allure.dynamic.description(description)
            logger.debug(f"Set Allure description to: {description}")

        # Only add tier metadata if explicitly defined in ACTIVE_PROFILE_HIGHEST_TIER
        active_tier = os.environ.get("ACTIVE_PROFILE_HIGHEST_TIER")
        if active_tier:
            # Add tier as a dynamic Allure label for the results table
            allure.dynamic.label("tier", str(active_tier))
            logger.debug(f"Set Allure tier label: {active_tier}")
            tier_display_name = active_tier.replace("_", " ").title()
            allure.dynamic.label("tier_display_name", tier_display_name)
            logger.debug(f"Set Allure tier_display_name label: {tier_display_name}")

        # Add dynamic Allure labels from profile and test configurations
        _apply_dynamic_labels_from_configs(configs)

        # Handle explicit severity setting from configs only
        if "severity" in configs:
            severity_map = {
                "critical": allure.severity_level.CRITICAL,
                "blocker": allure.severity_level.BLOCKER,
                "normal": allure.severity_level.NORMAL,
                "minor": allure.severity_level.MINOR,
                "trivial": allure.severity_level.TRIVIAL,
            }
            severity_name = str(configs["severity"]).lower()
            if severity_name in severity_map:
                allure.dynamic.severity(severity_map[severity_name])
                logger.debug(f"Set Allure severity to: {severity_name}")

        # Legacy label support (no duplicates with dynamic labels)
        if "profile" in configs:
            allure.dynamic.label("profile", configs["profile"])

        if "category" in configs:
            allure.dynamic.label("category", configs["category"])

        # Add dynamic Allure parameters based on configs
        for key, value in configs.items():
            # Skip certain keys that are not suitable as parameters or already handled
            if key in [
                "name",
                "display_name",
                "description",
                "severity",
                "kpi_refs",
                "profile",
                "suite",
                "category",
                "tier",
                "labels",
                "metadata",
            ]:
                continue

            # Format key for display by replacing underscores with spaces & capitalizing
            display_key = key.replace("_", " ").title()

            # Handle different types of values
            if isinstance(value, (str, int, float, bool)):
                allure.dynamic.parameter(display_key, str(value))
            elif isinstance(value, list) and len(value) < 10:  # Only add short lists
                allure.dynamic.parameter(display_key, str(value))
            elif isinstance(value, dict) and len(value) < 5:  # Only add small dicts
                allure.dynamic.parameter(display_key, str(value))

        logger.debug(f"Added Allure parameters for test: {item.name}")

    except ImportError:
        logger.debug("Allure not available, skipping metadata application")
    except Exception as e:
        logger.warning(f"Failed to apply Allure metadata for {item.name}: {e}")


def _apply_dynamic_labels_from_configs(configs):
    """
    Apply dynamic Allure labels from profile and test configurations.

    This function processes the 'labels' section from both profile-level and test-level
    configurations and applies them as Allure dynamic labels for enhanced reporting.

    Args:
        configs: Test configuration dictionary containing labels
    """
    try:
        # Handle labels from test configuration
        if "labels" in configs and isinstance(configs["labels"], dict):
            test_labels = configs["labels"]
            for label_key, label_value in test_labels.items():
                if label_value is not None:
                    # Convert value to string for Allure compatibility
                    label_str = str(label_value)
                    allure.dynamic.label(label_key, label_str)
                    logger.debug(f"Applied test-level Allure label: {label_key}={label_str}")

        # Get profile-level labels from active profile configurations
        try:
            from sysagent.utils.config import get_active_profile_configs

            profile_configs = get_active_profile_configs()

            if profile_configs and "params" in profile_configs and "labels" in profile_configs["params"]:
                profile_labels = profile_configs["params"]["labels"]

                for label_key, label_value in profile_labels.items():
                    if label_value is not None:
                        # Skip if already set by test-level config
                        if "labels" in configs and label_key in configs["labels"]:
                            continue

                        # Convert value to string for Allure compatibility
                        label_str = str(label_value)
                        allure.dynamic.label(label_key, label_str)
                        logger.debug(f"Applied profile-level Allure label: {label_key}={label_str}")

        except Exception as e:
            logger.debug(f"Could not load profile-level labels: {e}")

        logger.debug("Dynamic labels application completed")

    except ImportError:
        logger.debug("Allure not available, skipping dynamic labels")
    except Exception as e:
        logger.warning(f"Failed to apply dynamic labels: {e}")


def _set_allure_suite_names(item):
    """
    Set Allure suite names based on test file path structure.

    This function automatically determines and sets:
    - Parent suite: Based on active profile or the folder 3 levels up from test file
    - Suite: Based on the folder 2 levels up from test file (e.g., 'connectivity')
    - Sub-suite: Based on the folder 1 level up from test file (e.g., 'ethernet'),
      combined with the test function name (item.originalname) using a dot.
    """
    try:
        # Get the test file path
        test_file = str(item.fspath)
        logger.debug(f"Setting Allure suite names for test: {test_file}")

        # Parse the path to extract suite hierarchy information
        path_parts = test_file.split(os.sep)

        # Remove the test file name from the path to get directory structure
        dir_parts = path_parts[:-1]  # Exclude the test file itself

        # Ensure we have at least 3 directory levels for suite hierarchy
        if len(dir_parts) < 3:
            logger.warning(f"Test file path too shallow for suite hierarchy (need at least 3 folders): {test_file}")
            return

        # Extract suite hierarchy from the last 3 folders before the test file
        suite_name = dir_parts[-2]  # Two levels up (e.g., 'connectivity')
        parent_suite_folder = dir_parts[-3]  # Three levels up (e.g., 'suites')

        # Combine sub_suite_name with item.originalname using a dot
        sub_suite_name = dir_parts[-1]
        if hasattr(item, "originalname") and item.originalname:
            sub_suite_name = f"{sub_suite_name}.{item.originalname}"

        # Set parent suite based on active profile
        profile_name = os.environ.get("ACTIVE_PROFILE")
        if profile_name:
            logger.debug(f"Using active profile for parent suite: {profile_name}")
            parent_suite = profile_name
        else:
            parent_suite = parent_suite_folder

        tier_name = os.environ.get("ACTIVE_PROFILE_HIGHEST_TIER")
        if tier_name:
            tier_part = tier_name.strip().replace("_", "-")
            if parent_suite:
                parent_suite = f"{parent_suite}.{tier_part}"
            else:
                parent_suite = tier_part
            logger.debug(f"Using active profile highest tier for parent suite: {parent_suite}")

        # Apply Allure suite metadata
        allure.dynamic.parent_suite(parent_suite)
        allure.dynamic.suite(suite_name)
        allure.dynamic.sub_suite(sub_suite_name)

        logger.debug("Set Allure suite hierarchy from path structure:")
        logger.debug(f"  Parent suite: {parent_suite}")
        logger.debug(f"  Suite: {suite_name}")
        logger.debug(f"  Sub-suite: {sub_suite_name}")
        logger.info(f"Allure suite hierarchy: {parent_suite} > {suite_name} > {sub_suite_name}")

    except Exception as e:
        logger.warning(f"Failed to set Allure suite names for {item.fspath}: {e}")


def pytest_runtest_logstart(nodeid, location):
    """
    Called before each test to print a custom header.
    """
    module, _, rest = nodeid.partition("::")
    module = Path(module).name  # shorten path
    formatted = "\n"
    sys.stdout.write(formatted)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting {Path(module).name}")


def pytest_itemcollected(item):
    """
    Shortens the node IDs to make test output more readable.
    """
    parts = item.nodeid.split("::")
    module = Path(parts[0]).name
    rest = parts[1:]
    item._nodeid = "::".join([module] + rest)


# Session tracking hooks
def pytest_sessionstart(session):
    """Initialize test session counters."""
    session.tests_run = 0
    session.total_tests = 0


def pytest_collection_modifyitems(session, config, items):
    """Store the total number of tests collected."""
    session.total_tests = len(items)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Attach session to each report."""
    report = yield
    rep = report.get_result()
    # only attach once
    if not hasattr(rep, "session"):
        rep.session = item.session


# Argument 'config' is pytest specific. Do not rename it.
def pytest_report_teststatus(report, config):
    """Hide the inline dot/F progress indicators."""
    if report.when == "call":
        return


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    """Print custom progress indicators with percentage."""
    if report.when != "call":
        return

    session = report.session  # now available
    session.tests_run += 1
    total = session.total_tests
    current = session.tests_run
    percent = f"{(current / total * 100):5.1f}%" if total else "  ?  "

    parts = report.nodeid.split("::")
    # module = Path(parts[0]).name
    rest = "::".join(parts[1:])

    sys.stdout.write("\n")
    logger = logging.getLogger(__name__)
    logger.info(f"Completed [{percent}] {rest}: {report.outcome.upper()}")
