# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Profile parameter consolidation utilities.

Provides unified parameter consolidation functionality that works consistently
for all profile types regardless of whether they have tier configurations.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def consolidate_profile_parameters(
    profile_configs: Dict[str, Any], filters: Optional[Dict[str, Any]] = None, highest_tier: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Consolidate and extract test parameters from a profile configuration.

    This unified function handles both tier-based and non-tier profiles consistently,
    applying profile-level parameter merging and filtering in a single place.

    Args:
        profile_configs: Profile configuration dictionary
        filters: Optional filters to apply to test parameters
        highest_tier: Optional highest tier for tier-based profiles

    Returns:
        List of consolidated test parameters
    """

    consolidated_params = []

    # Get profile-level parameters that should be merged into all test parameters
    profile_params = profile_configs.get("params", {})
    profile_requirements = profile_params.get("requirements", {})
    profile_labels = profile_params.get("labels", {})
    profile_tiers = profile_params.get("tiers", [])

    logger.debug(f"Consolidating parameters for profile with tiers: {profile_tiers}")
    logger.debug(f"Profile requirements: {list(profile_requirements.keys()) if profile_requirements else 'None'}")
    logger.debug(f"Profile labels: {list(profile_labels.keys()) if profile_labels else 'None'}")

    # Determine if this is a tier-based profile
    has_tiers = bool(profile_tiers)

    if has_tiers:
        logger.debug(f"Processing tier-based profile with {len(profile_tiers)} tiers")
        # For tier-based profiles, use tier validation to get parameters
        consolidated_params = _consolidate_tier_based_parameters(
            profile_configs, filters, highest_tier, profile_requirements, profile_labels
        )
    else:
        logger.debug("Processing non-tier profile with direct parameter merging")
        # For non-tier profiles, directly merge parameters from suites
        consolidated_params = _consolidate_direct_parameters(
            profile_configs, filters, profile_requirements, profile_labels
        )

    logger.debug(f"Consolidated {len(consolidated_params)} parameters from profile")
    return consolidated_params


def _consolidate_tier_based_parameters(
    profile_configs: Dict[str, Any],
    filters: Optional[Dict[str, Any]],
    highest_tier: Optional[str],
    profile_requirements: Dict[str, Any],
    profile_labels: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Consolidate parameters for tier-based profiles."""
    from sysagent.utils.testing.tier_validator import get_test_params_tiers

    # Use existing tier validation system but with consolidated filtering
    tier_params = get_test_params_tiers(profile_configs, passed_only=True, filters=filters)

    consolidated_params = []
    for param_item in tier_params:
        param_config = param_item.get("param", {})

        # Merge profile-level requirements and labels
        merged_param = _merge_profile_level_params(param_config, profile_requirements, profile_labels)

        # Add tier information
        if highest_tier:
            merged_param["tier"] = highest_tier

        consolidated_params.append(
            {
                "test": param_item.get("test"),
                "suite": param_item.get("suite"),
                "sub_suite": param_item.get("sub_suite"),
                "param": merged_param,
            }
        )

    return consolidated_params


def _consolidate_direct_parameters(
    profile_configs: Dict[str, Any],
    filters: Optional[Dict[str, Any]],
    profile_requirements: Dict[str, Any],
    profile_labels: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Consolidate parameters for non-tier profiles."""
    consolidated_params = []

    # Extract test configurations directly from profile suites
    for suite in profile_configs.get("suites", []):
        suite_name = suite.get("name", "unknown_suite")

        for sub_suite in suite.get("sub_suites", []):
            sub_suite_name = sub_suite.get("name", "unknown_sub_suite")

            # Handle tests as dictionary (test_name -> test_config)
            tests = sub_suite.get("tests", {})
            if isinstance(tests, dict):
                for test_name, test_config in tests.items():
                    # Process each parameter configuration
                    if isinstance(test_config, dict) and "params" in test_config:
                        for param_config in test_config["params"]:
                            # Apply filters early if provided
                            if filters and not _match_param_filters(param_config, filters):
                                continue

                            # Merge profile-level parameters
                            merged_param = _merge_profile_level_params(
                                param_config, profile_requirements, profile_labels
                            )

                            # Ensure proper metadata
                            merged_param["suite"] = suite_name
                            merged_param["sub_suite"] = sub_suite_name

                            # Generate proper test ID and display name if not present
                            if "test_id" not in merged_param:
                                merged_param["test_id"] = f"{test_name}_param_{len(consolidated_params)}"
                            if "display_name" not in merged_param:
                                merged_param["display_name"] = (
                                    f"{test_name} - {merged_param.get('test_id', 'Parameter')}"
                                )

                            # Allow param to override test function name (for multiple tests per file)
                            # If param has "test" field, use it; otherwise use YAML key (test_name)
                            target_test_name = merged_param.get("test", test_name)

                            consolidated_params.append(
                                {
                                    "test": target_test_name,
                                    "suite": suite_name,
                                    "sub_suite": sub_suite_name,
                                    "param": merged_param,
                                }
                            )

    return consolidated_params


def _merge_profile_level_params(
    param_config: Dict[str, Any], profile_requirements: Dict[str, Any], profile_labels: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge profile-level requirements and labels into test parameters."""
    merged_param = param_config.copy()

    # Merge profile requirements (test-level requirements take precedence)
    if profile_requirements:
        param_requirements = merged_param.get("requirements", {})
        merged_requirements = profile_requirements.copy()
        merged_requirements.update(param_requirements)  # Test requirements override profile
        merged_param["requirements"] = merged_requirements

    # Merge profile labels (test-level labels take precedence)
    if profile_labels:
        param_labels = merged_param.get("labels", {})
        merged_labels = profile_labels.copy()
        merged_labels.update(param_labels)  # Test labels override profile
        merged_param["labels"] = merged_labels

    return merged_param


def _match_param_filters(param_config: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Check if a test parameter matches the provided filters."""
    from .matcher import match_filter_value

    for filter_key, filter_value in filters.items():
        param_value = param_config.get(filter_key)

        if not match_filter_value(param_value, filter_value):
            return False

    return True
