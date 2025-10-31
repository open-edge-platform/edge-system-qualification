# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Profile validator utilities.

This module provides utilities to validate system requirements for test profiles,
converting profile configuration formats to system requirement validation.
"""

import logging
import os
from typing import Any, Dict

# Setup logger
logger = logging.getLogger(__name__)


def validate_profile_requirements(profile_configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate system requirements for a profile using latest system info structure.

    This function converts profile configuration requirements into the format
    expected by SystemValidator and performs the validation.

    Args:
        profile_configs: Profile configuration dictionary containing requirements

    Returns:
        Dict with validation results containing:
        - passed: bool indicating if all requirements are met
        - checks: list of individual check results

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
        results = validate_profile_requirements(profile_configs)
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

    # Perform validation
    results = validator.validate_requirements(system_requirements)

    return results


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
        for memory_field in ["memory_min_gb"]:
            if memory_field in profile_requirements:
                system_format["hardware"][memory_field] = profile_requirements[memory_field]

        # Storage requirements - direct mapping
        for storage_field in ["storage_min_gb"]:
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
