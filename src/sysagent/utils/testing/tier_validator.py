# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tier validation utilities.

This module provides utilities to validate system requirements against test tiers,
determining which tiers a system qualifies for based on hardware capabilities.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_all_tiers() -> Dict[str, Any]:
    """Load all tier configurations."""
    from sysagent.utils.config import list_tiers

    all_tiers_data = list_tiers()
    # The structure is {"tiers": {tier_name: tier_config, ...}}
    if "tiers" in all_tiers_data and all_tiers_data["tiers"]:
        logger.debug((f"Loaded {len(all_tiers_data['tiers'])} tiers: {list(all_tiers_data['tiers'].keys())}"))
        return all_tiers_data["tiers"]

    logger.debug("No tiers found in tier configuration")
    return {}


def _get_tier_priority_map(tiers: List[str], all_tiers: Dict[str, Any]) -> Dict[str, int]:
    """Get priority mapping for tiers."""
    return {tier: all_tiers[tier]["priority"] for tier in tiers if tier in all_tiers}


def _merge_requirements(
    tier_obj: Dict[str, Any],
    profile_reqs: Optional[Dict[str, Any]],
    test_reqs: Optional[Dict[str, Any]],
    use_profile: bool,
) -> Dict[str, Any]:
    """
    Merge requirements according to use_profile flag:
    - If use_profile is False: only tier's requirements.
    - If use_profile is True: tier + profile + test param
      (test param overrides profile, which overrides tier).
    """
    merged = dict(tier_obj.get("requirements", {}))
    if use_profile:
        if profile_reqs:
            merged.update(profile_reqs)
        if test_reqs:
            merged.update(test_reqs)
    return merged


def _validate_tiers(
    tiers_to_check: List[str],
    all_tiers: Dict[str, Any],
    validator: Any,
    profile_reqs: Optional[Dict[str, Any]] = None,
    test_reqs: Optional[Dict[str, Any]] = None,
    use_profile_reqs: bool = False,
) -> List[Dict[str, Any]]:
    """Validate multiple tiers against system requirements."""
    results = []
    for tier_name in tiers_to_check:
        logger.debug(f"Validating test tier: '{tier_name}'")
        tier_obj = all_tiers.get(tier_name)
        if not tier_obj or "requirements" not in tier_obj:
            results.append(
                {
                    "tier": tier_name,
                    "passed": False,
                    "priority": 9999,
                    "reason": "Tier definition not found or missing requirements",
                    "checks": [],
                }
            )
            continue
        merged_req = _merge_requirements(tier_obj, profile_reqs, test_reqs, use_profile_reqs)
        # Don't log suggestions for tier validation - tier mismatches are expected and handled internally
        result = validator.validate_requirements({"hardware": merged_req, "software": {}}, log_suggestions=False)
        results.append(
            {
                "tier": tier_name,
                "passed": result.get("passed", False),
                "priority": tier_obj.get("priority", 0),
                "name": tier_obj.get("name", tier_name),
                "description": tier_obj.get("description", ""),
                "checks": result.get("checks", []),
            }
        )
    return results


def validate_profile_tiers(profile_configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate system against profile tier requirements

    Args:
        profile_configs: Profile configuration dictionary

    Returns:
        Dict containing validation results
    """
    from sysagent.utils.config import setup_data_dir
    from sysagent.utils.testing.system_validator import SystemValidator

    data_dir = setup_data_dir()
    cache_dir = os.path.join(data_dir, "cache")
    validator = SystemValidator(cache_dir)
    all_tiers = _load_all_tiers()
    profile_params = profile_configs.get("params", {})
    profile_tiers = profile_params.get("tiers", [])

    if not profile_tiers:
        logger.debug("No tiers defined in profile - skipping tier validation")
        return {"passed": True, "matched_tiers": [], "failed_tiers": [], "checks": []}

    # Validate profile tier configuration
    tier_errors = validate_profile_tier_configuration(profile_configs)
    if tier_errors:
        logger.error("Profile tier configuration validation failed with the following errors:")
        for err in tier_errors:
            logger.error(f"  ✗ {err}")
        return {"passed": False, "matched_tiers": [], "failed_tiers": [], "checks": []}

    results = _validate_tiers(profile_tiers, all_tiers, validator)
    matched = [r["tier"] for r in results if r["passed"]]
    failed = [r["tier"] for r in results if not r["passed"]]

    def tier_sort_key(t):
        return t.get("priority", 0)

    prioritized = None
    if matched:
        prioritized = sorted([r for r in results if r["passed"]], key=tier_sort_key)[0]
    elif failed:
        prioritized = sorted([r for r in results if not r["passed"]], key=tier_sort_key)[0]

    return {
        "passed": bool(matched),
        "matched_tiers": matched,
        "failed_tiers": failed,
        "prioritized_tier": prioritized,
        "all_results": results,
        "checks": [check for result in results for check in result.get("checks", [])],
    }


def validate_test_tiers(
    test_tiers: List[str],
    profile_configs: Optional[Dict[str, Any]] = None,
    test_requirements: Optional[Dict[str, Any]] = None,
    use_profile_requirements: bool = True,
) -> Dict[str, Any]:
    """
    Validate system against specific test tier requirements

    Args:
        test_tiers: List of tier names to validate
        profile_configs: Optional profile configuration
        test_requirements: Optional test-specific requirements
        use_profile_requirements: Whether to include profile requirements

    Returns:
        Dict containing validation results
    """
    from sysagent.utils.config import setup_data_dir
    from sysagent.utils.testing.system_validator import SystemValidator

    data_dir = setup_data_dir()
    cache_dir = os.path.join(data_dir, "cache")
    validator = SystemValidator(cache_dir)
    all_tiers = _load_all_tiers()

    if not test_tiers:
        logger.info("No test tiers specified - skipping tier validation")
        return {"passed": True, "matched_tiers": [], "failed_tiers": [], "checks": []}

    profile_reqs = None
    if profile_configs and use_profile_requirements:
        profile_params = profile_configs.get("params", {})
        profile_reqs = profile_params.get("requirements", {})

    results = _validate_tiers(
        test_tiers,
        all_tiers,
        validator,
        profile_reqs,
        test_requirements,
        use_profile_requirements,
    )

    matched = [r["tier"] for r in results if r["passed"]]
    failed = [r["tier"] for r in results if not r["passed"]]

    return {
        "passed": bool(matched),
        "matched_tiers": matched,
        "failed_tiers": failed,
        "all_results": results,
        "checks": [check for result in results for check in result.get("checks", [])],
    }


def get_highest_matching_tier(profile_configs: Dict[str, Any]) -> Optional[str]:
    """
    Get the highest priority tier that the system matches

    Args:
        profile_configs: Profile configuration dictionary

    Returns:
        str: Name of highest matching tier, or None if no matches
    """
    try:
        results = validate_profile_tiers(profile_configs)

        if results.get("passed") and results.get("matched_tiers"):
            # Return the highest priority tier from matched results
            prioritized = results.get("prioritized_tier")
            if prioritized:
                return prioritized.get("tier")

        return None
    except Exception as e:
        logger.error(f"Error getting highest matching tier: {e}")
        return None


def validate_profile_tier_configuration(profile_configs: Dict[str, Any]) -> List[str]:
    """
    Validate the tier configuration in a profile.

    Args:
        profile_configs: Profile configuration dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    all_tiers = _load_all_tiers()
    profile_params = profile_configs.get("params", {})
    profile_tiers = profile_params.get("tiers", [])

    if not isinstance(profile_tiers, list):
        errors.append("Profile 'tiers' must be a list")
        return errors

    for tier in profile_tiers:
        if not isinstance(tier, str):
            errors.append(f"Tier name must be a string, got: {type(tier)}")
            continue

        if tier not in all_tiers:
            available_tiers = list(all_tiers.keys())
            errors.append(f"Unknown tier '{tier}'. Available tiers: {available_tiers}")

    return errors


def log_tier_validation_results(results: Dict[str, Any], context: str = "") -> None:
    """
    Log detailed tier validation results using latest system info structure.

    Args:
        results: Tier validation results dictionary
        context: Optional context string for logging
    """
    context_prefix = f"{context}: " if context else ""

    if results.get("passed"):
        matched_tiers = results.get("matched_tiers", [])
        logger.info(f"{context_prefix}Tier validation PASSED - Matched tiers: {matched_tiers}")

        prioritized = results.get("prioritized_tier")
        if prioritized:
            tier_name = prioritized.get("tier", "unknown")
            tier_desc = prioritized.get("description", "")
            logger.info(f"{context_prefix}Highest priority matching tier: {tier_name} - {tier_desc}")
    else:
        failed_tiers = results.get("failed_tiers", [])
        logger.warning(f"{context_prefix}Tier validation FAILED - Failed tiers: {failed_tiers}")

        # Log details of failed checks using new structure
        for result in results.get("all_results", []):
            if not result.get("passed"):
                tier_name = result.get("tier", "unknown")
                logger.warning(f"{context_prefix}Tier '{tier_name}' requirements not met:")
                for check in result.get("checks", []):
                    if not check.get("passed", True):
                        check_name = check.get("name", "Unknown check")
                        actual = check.get("actual", "Unknown")
                        required = check.get("required", "Unknown")
                        logger.warning(f"{context_prefix}  ✗ {check_name}: actual={actual}, required={required}")


def get_test_params_tiers(
    profile_configs: Dict[str, Any],
    passed_only: bool = True,
    filters: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Extract test parameters for tiers using latest system info structure.

    Args:
        profile_configs: Profile configuration dictionary
        passed_only: If True, only return params for tiers that passed validation
        filters: Optional dictionary of filters to apply to test parameters

    Returns:
        List of test parameter configurations with tier information
    """

    test_params = []

    # Get tier validation results using new structure
    tier_results = validate_profile_tiers(profile_configs)

    if not tier_results or "all_results" not in tier_results:
        return test_params

    # Create a lookup dictionary for tier results
    tier_lookup = {result["tier"]: result for result in tier_results["all_results"]}

    # Extract test configurations from profile
    for suite in profile_configs.get("suites", []):
        suite_name = suite.get("name", "unknown_suite")

        for sub_suite in suite.get("sub_suites", []):
            sub_suite_name = sub_suite.get("name", "unknown_sub_suite")

            # Handle tests as dictionary (test_name -> test_config)
            tests = sub_suite.get("tests", {})
            if isinstance(tests, dict):
                for test_name, test_config in tests.items():
                    # If test has params array, process each param
                    if isinstance(test_config, dict) and "params" in test_config:
                        for param_config in test_config["params"]:
                            # Apply filters early if provided
                            if filters and not _match_test_param_filters(param_config, filters):
                                continue

                            # Get tiers for this param, default to all tiers
                            param_tiers = param_config.get("tiers", list(tier_lookup.keys()))

                            # Create parameter entry for each tier this param supports
                            for tier_name in param_tiers:
                                if tier_name not in tier_lookup:
                                    continue

                                tier_result = tier_lookup[tier_name]
                                if passed_only and not tier_result.get("passed", False):
                                    continue

                                test_param = {
                                    "test": test_name,
                                    "suite": suite_name,
                                    "sub_suite": sub_suite_name,
                                    "tier_result": tier_result,
                                    "param": param_config.copy(),
                                }

                                # Ensure proper tier labeling
                                test_param["param"]["tier"] = tier_name
                                test_param["param"]["suite"] = suite_name
                                test_param["param"]["sub_suite"] = sub_suite_name

                                # Generate proper test ID and display name
                                if "test_id" not in test_param["param"]:
                                    test_param["param"]["test_id"] = f"{test_name}_{tier_name}"
                                if "display_name" not in test_param["param"]:
                                    test_param["param"]["display_name"] = f"{test_name} - {tier_name}"

                                test_params.append(test_param)

    return test_params


def _match_test_param_filters(param_config: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Check if a test parameter configuration matches the provided filters.

    Args:
        param_config: Test parameter configuration
        filters: Dictionary of filter criteria

    Returns:
        True if parameter matches all filters, False otherwise
    """
    from sysagent.utils.cli.filters.matcher import match_filter_value

    for filter_key, filter_value in filters.items():
        param_value = param_config.get(filter_key)

        if not match_filter_value(param_value, filter_value):
            return False

    return True


class TierValidationError(Exception):
    """Exception raised when tier validation fails."""

    def __init__(self, failed_tiers: List[str], context: str = ""):
        self.failed_tiers = failed_tiers
        self.context = context
        message = f"Tier validation failed for: {', '.join(failed_tiers)}"
        if context:
            message = f"{context} - {message}"
        super().__init__(message)
