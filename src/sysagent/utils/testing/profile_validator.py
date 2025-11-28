# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Profile validator utilities.

This module provides utilities to validate system requirements for test profiles,
converting profile configuration formats to system requirement validation.
"""

import logging
import os
from typing import Any, Dict, Optional

# Setup logger
logger = logging.getLogger(__name__)


def validate_profile_requirements(
    profile_configs: Dict[str, Any], profile_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate system requirements for a profile using latest system info structure.

    This function converts profile configuration requirements into the format
    expected by SystemValidator and performs the validation.

    Args:
        profile_configs: Profile configuration dictionary containing requirements
        profile_name: Optional profile name for better context in validation messages

    Returns:
        Dict with validation results containing:
        - passed: bool indicating if all requirements are met
        - checks: list of individual check results
        - profile_name: profile name if provided

    Example:
        profile_configs = {
            "params": {
                "requirements": {
                    "cpu_min_cores": 4,
                    "memory_min_gb": 8.0,
                    "os_type": ["linux"],
                    "docker_required": true
                }
            }
        }
        results = validate_profile_requirements(profile_configs, "my-profile")
        if results["passed"]:
            print("All requirements met")
        else:
            for check in results["checks"]:
                if not check["passed"]:
                    print(f"Failed: {check['name']}")
    """
    from sysagent.utils.config import setup_data_dir
    from sysagent.utils.testing.system_validator import SystemValidator

    data_dir = setup_data_dir()
    cache_dir = os.path.join(data_dir, "cache")
    validator = SystemValidator(cache_dir)

    # Extract requirements from profile config params using new structure
    profile_params = profile_configs.get("params", {})
    requirements = profile_params.get("requirements", {})

    # Convert profile requirements to system validator format
    system_requirements = _convert_profile_requirements_to_system_format(requirements)

    # Perform validation with profile-specific context
    context = f"profile: {profile_name}" if profile_name else "system validator"
    results = validator.validate_requirements(system_requirements, log_suggestions=True, context=context)

    # Add profile name to results for later reference
    if profile_name:
        results["profile_name"] = profile_name

    return results


def validate_filtered_profile_requirements(
    profile_configs: Dict[str, Any], filters: Optional[Dict[str, Any]] = None, profile_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate system requirements for a profile considering both profile-level and
    test-specific requirements from filtered test parameters.

    Args:
        profile_configs: Profile configuration dictionary containing requirements
        filters: Optional test filters to determine which test parameters to validate
        profile_name: Optional profile name for better context in validation messages

    Returns:
        Dict with validation results containing:
        - passed: bool indicating if all requirements are met
        - checks: list of individual check results
        - profile_checks: list of profile-level requirement checks
        - test_checks: list of test-specific requirement checks
    """
    from sysagent.utils.cli.filters.consolidator import consolidate_profile_parameters
    from sysagent.utils.config import setup_data_dir
    from sysagent.utils.testing.system_validator import SystemValidator

    logger.debug("Validating requirements for profile with filtered tests")

    data_dir = setup_data_dir()
    cache_dir = os.path.join(data_dir, "cache")
    validator = SystemValidator(cache_dir)

    # Validate profile-level requirements
    profile_params = profile_configs.get("params", {})
    profile_requirements = profile_params.get("requirements", {})
    profile_system_requirements = _convert_profile_requirements_to_system_format(profile_requirements)

    # Don't log suggestions here - will be consolidated below
    profile_results = validator.validate_requirements(profile_system_requirements, log_suggestions=False)

    # Get consolidated test parameters that would actually run based on filters
    try:
        consolidated_params = consolidate_profile_parameters(profile_configs, filters)
        logger.debug(f"Found {len(consolidated_params)} test parameters after filtering")
    except Exception as e:
        logger.warning(f"Could not consolidate profile parameters: {e}")
        consolidated_params = []

    # Collect all unique test-specific requirements
    all_test_requirements = {}
    test_requirements_list = []

    for param in consolidated_params:
        test_requirements = param.get("param", {}).get("requirements", {})
        if test_requirements:
            test_id = param.get("param", {}).get("test_id", "unknown")
            logger.debug(f"Test {test_id} has requirements: {list(test_requirements.keys())}")

            # Merge test requirements into combined set
            for key, value in test_requirements.items():
                if key not in all_test_requirements:
                    all_test_requirements[key] = value
                    test_requirements_list.append((test_id, key, value))
                elif all_test_requirements[key] != value:
                    # If different tests have conflicting requirements, use most restrictive
                    logger.debug(f"Conflicting requirement {key}: {all_test_requirements[key]} vs {value}")
                    if isinstance(value, (int, float)) and isinstance(all_test_requirements[key], (int, float)):
                        all_test_requirements[key] = max(all_test_requirements[key], value)

    # Validate test-specific requirements if any exist
    test_results = {"passed": True, "checks": []}
    if all_test_requirements:
        test_system_requirements = _convert_profile_requirements_to_system_format(all_test_requirements)
        test_results = validator.validate_requirements(test_system_requirements, log_suggestions=False)

        # Add context to test requirement checks
        for check in test_results.get("checks", []):
            check["context"] = "test-specific"
    else:
        logger.info("No test-specific requirements found in filtered parameters")

    # Add context to profile requirement checks
    for check in profile_results.get("checks", []):
        check["context"] = "profile-level"

    # Combine results
    all_checks = profile_results.get("checks", []) + test_results.get("checks", [])
    overall_passed = profile_results.get("passed", False) and test_results.get("passed", False)

    combined_results = {
        "passed": overall_passed,
        "checks": all_checks,
        "profile_checks": profile_results.get("checks", []),
        "test_checks": test_results.get("checks", []),
        "profile_passed": profile_results.get("passed", False),
        "test_passed": test_results.get("passed", False),
    }

    # Log summary with profile context
    if not overall_passed:
        failed_checks = [check for check in all_checks if not check.get("passed", False)]

        # Provide consolidated fix suggestions without duplicates, with profile context
        from sysagent.utils.testing.validation_suggestions import log_validation_fix_suggestions

        context = f"profile: {profile_name}" if profile_name else "system validator"
        log_validation_fix_suggestions(failed_checks, context=context, deduplicate_by_category=True)

    # Add profile name to results for later reference
    if profile_name:
        combined_results["profile_name"] = profile_name

    return combined_results


def _convert_profile_requirements_to_system_format(
    profile_requirements: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert profile requirements format to system validator format
    No backward compatibility - direct mapping only.

    Args:
        profile_requirements: Requirements in profile format

    Returns:
        Dict in system validator format
    """
    system_format = {}

    # Direct hardware requirements mapping
    if profile_requirements:
        system_format["hardware"] = {}

        # CPU requirements - direct mapping
        for cpu_field in [
            "cpu_min_cores",
            "cpu_xeon_required",
            "cpu_core_required",
            "cpu_ultra_required",
            "cpu_ultra_mobile_required",
            "cpu_entry_required",
        ]:
            if cpu_field in profile_requirements:
                system_format["hardware"][cpu_field] = profile_requirements[cpu_field]

        # Memory requirements - direct mapping
        for memory_field in ["memory_min_gb", "memory_total_min_gb"]:
            if memory_field in profile_requirements:
                system_format["hardware"][memory_field] = profile_requirements[memory_field]

        # Storage requirements - direct mapping
        for storage_field in ["storage_min_gb", "storage_total_min_gb"]:
            if storage_field in profile_requirements:
                system_format["hardware"][storage_field] = profile_requirements[storage_field]

        # GPU requirements - direct mapping
        for gpu_field in [
            "igpu_required",
            "dgpu_required",
            "dgpu_min_devices",
            "dgpu_min_vram_gb",
        ]:
            if gpu_field in profile_requirements:
                system_format["hardware"][gpu_field] = profile_requirements[gpu_field]

        # NPU requirements - direct mapping
        for npu_field in ["npu_required", "npu_min_devices"]:
            if npu_field in profile_requirements:
                system_format["hardware"][npu_field] = profile_requirements[npu_field]

    # Direct software requirements mapping
    if profile_requirements:
        system_format["software"] = {}

        # Direct OS requirements
        for os_field in ["os_type"]:
            if os_field in profile_requirements:
                system_format["software"][os_field] = profile_requirements[os_field]

        # Direct Python requirements
        for python_field in ["min_python_version", "max_python_version"]:
            if python_field in profile_requirements:
                system_format["software"][python_field] = profile_requirements[python_field]

        # Direct package requirements
        for package_field in [
            "docker_required",
            "required_system_packages",
            "required_python_packages",
        ]:
            if package_field in profile_requirements:
                system_format["software"][package_field] = profile_requirements[package_field]

    return system_format


def check_profile_compatibility(profile_configs: Dict[str, Any]) -> bool:
    """
    Quick check if current system is compatible with profile requirements.

    Args:
        profile_configs: Profile configuration dictionary

    Returns:
        bool: True if system meets all requirements, False otherwise
    """
    try:
        results = validate_profile_requirements(profile_configs)
        return results.get("passed", False)
    except Exception as e:
        logger.error(f"Error checking profile compatibility: {e}")
        return False


def get_failed_requirements(profile_configs: Dict[str, Any]) -> list:
    """
    Get list of failed requirements for a profile.

    Args:
        profile_configs: Profile configuration dictionary

    Returns:
        List of failed requirement descriptions
    """
    try:
        results = validate_profile_requirements(profile_configs)
        failed_checks = [check for check in results.get("checks", []) if not check.get("passed", True)]
        return [check.get("name", "Unknown requirement") for check in failed_checks]
    except Exception as e:
        logger.error(f"Error getting failed requirements: {e}")
        return [f"Error checking requirements: {str(e)}"]


def log_profile_validation_results(profile_name: str, profile_configs: Dict[str, Any]) -> None:
    """
    Log detailed profile validation results using latest system info structure.

    Args:
        profile_name: Name of the profile being validated
        profile_configs: Profile configuration dictionary
    """
    try:
        results = validate_profile_requirements(profile_configs)

        if results.get("passed", False):
            logger.info(f"Profile '{profile_name}' requirements validation: PASSED")
        else:
            logger.warning(f"Profile '{profile_name}' requirements validation: FAILED")

            failed_checks = [check for check in results.get("checks", []) if not check.get("passed", True)]

            for check in failed_checks:
                requirement = check.get("name", "Unknown")
                actual = check.get("actual", "Unknown")
                required = check.get("required", "Unknown")
                logger.warning(f"  ✗ {requirement}: actual={actual}, required={required}")

        # Log summary
        total_checks = len(results.get("checks", []))
        pass_checks = len([check for check in results.get("checks", []) if check.get("passed", True)])
        logger.info(f"Profile validation summary: {pass_checks}/{total_checks} requirements met")

    except Exception as e:
        logger.error(f"Error logging profile validation results for '{profile_name}': {e}")


class ProfileValidationError(Exception):
    """Exception raised when profile validation fails."""

    def __init__(self, profile_name: str, failed_requirements: list):
        self.profile_name = profile_name
        self.failed_requirements = failed_requirements
        message = f"Profile '{profile_name}' validation failed. Requirements not met: {', '.join(failed_requirements)}"
        super().__init__(message)
