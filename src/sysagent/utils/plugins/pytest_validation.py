# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Validation utilities for core testing framework.

This module provides fixtures and functions for validating test results
against KPI configurations.
"""

import json
import logging
import os
from typing import Any, Dict

import allure
import pytest
from pytest_check import check

from sysagent.utils.config import setup_data_dir
from sysagent.utils.core import Result

logger = logging.getLogger(__name__)


def mark_step_as_failed(step_name: str, error_msg: str, validation_mode: str = "all", result_json=None):
    """
    Mark an Allure step as failed but continue test execution.

    This function:
    1. Logs the error
    2. Creates a broken step in Allure that's clearly visible
    3. Uses pytest-check to fail the test but continue execution

    Args:
        step_name: Name of the failed step
        error_msg: Error message explaining the failure
        result_json: Optional JSON result to attach
    """
    logger.error(f"Step failed: {error_msg}")

    # Create a step with a clear failure indicator in the name
    with allure.step(f"🔴 {step_name}"):
        # Attach JSON result if provided
        if result_json:
            allure.attach(
                json.dumps(result_json, indent=2),
                name="KPI Validation Result",
                attachment_type=allure.attachment_type.JSON,
            )

        # Log the error in the test logs too
        logger.error(f"FAILED: {error_msg}")

    # Use pytest-check to fail the test but continue execution
    if validation_mode == "all":
        check.equal(False, True, error_msg)


@pytest.fixture
def validate_test_results():
    """
    Fixture that returns a function to validate test results against KPIs.

    Can be used across all test suites to standardize the validation process
    and reporting in Allure. It handles all logging internally so individual tests don't
    need to add additional logging around this function call.

    Usage in tests:
        validation_results = validate_test_results(
            results=results,
            configs=configs,
            get_kpi_config=get_kpi_config,
            test_name=test_name
        )

    Returns:
        A function that performs validation and creates appropriate Allure report steps
    """
    logger = logging.getLogger(__name__)

    def _validate_results(
        results: Result,
        configs: Dict[str, Any],
        get_kpi_config: callable,
        test_name: str = "Unknown",
        mode: str = "all",
    ) -> Dict[str, Any]:
        """
        Validate test results against KPI configurations.
        Accepts results in the generic structure:
            { "metric_name": { "value": ..., "unit": ... }, ... }
        or flat { "metric_name": value, ... } for backward compatibility.
        The function will extract the value for KPI validation and
        pass the full dict for reporting.
        """
        """
        Validate test results against KPI configurations.
        
        Args:
            results: Dictionary containing test results metrics
            configs: Test configurations dictionary (KPI refs extracted from here)
            get_kpi_config: Function to retrieve KPI configuration by name
            test_name: Name of the test for logging purposes
            
        Returns:
            Dictionary containing validation results with the following structure:
            {
                "passed": bool,
                "validations": {
                    "kpi_name": {
                        "passed": bool,
                        "actual_value": float,
                        "expected_value": str,
                        "unit": str,
                        ...
                    },
                    ...
                },
                "skipped": bool,
                "skip_reason": Optional[str]
            }
        """
        # Extract KPI references from configs
        current_kpi_refs = configs.get("kpi_refs", [])
        # Log all config keys and values for debugging
        for key, value in configs.items():
            if isinstance(value, dict):
                logger.debug(
                    f"Config key '{key}', Value: (See the test parameters section in the test report for details)"
                )
            else:
                logger.debug(f"Config Key: '{key}', Value: {value}")

        # Log the start of validation at INFO level
        logger.info(f"Validating test results against KPIs for test: {test_name}")

        validation_results = {
            "passed": True,
            "validations": {},
            "skipped": False,
            "skip_reason": None,
        }

        # Pre-check phase: Determine validation status before creating the step
        logger.debug(f"Pre-checking KPI validations for test: {test_name}")

        # First, determine if we need to skip validation
        should_skip = False
        skip_reason = None

        # Skip the entire validation step if no KPI references are defined
        if not current_kpi_refs:
            should_skip = True
            skip_reason = "No KPI references defined"
            validation_results["skipped"] = should_skip
            validation_results["skip_reason"] = skip_reason
            logger.info(f"KPI validation was skipped: {skip_reason}")
        else:
            # Check if any defined KPIs exist
            has_defined_kpis = False
            for kpi_name in current_kpi_refs:
                if get_kpi_config(kpi_name):
                    has_defined_kpis = True
                    break

            # Skip validation if none of the referenced KPIs are defined
            if not has_defined_kpis:
                should_skip = True
                skip_reason = "None of the referenced KPIs are defined"
                validation_results["skipped"] = should_skip
                validation_results["skip_reason"] = skip_reason
                logger.info(f"KPI validation was skipped: {skip_reason}")
            else:
                # Check if all defined KPIs have validation disabled
                all_disabled = True
                for kpi_name in current_kpi_refs:
                    kpi_config = get_kpi_config(kpi_name)
                    if kpi_config and kpi_config.get("validation", {}).get("enabled", True):
                        all_disabled = False
                        break
                if all_disabled:
                    should_skip = True
                    skip_reason = "All referenced KPIs have validation disabled"
                    validation_results["skipped"] = should_skip
                    validation_results["skip_reason"] = skip_reason
                    logger.info(f"KPI validation was skipped: {skip_reason}")

        # If skipping, create a standardized skipped step and return
        if should_skip:
            # Collect KPI config details for all referenced KPIs
            kpi_configs = {kpi_name: get_kpi_config(kpi_name) for kpi_name in current_kpi_refs}
            with allure.step("Validate test results ⚪"):
                allure.attach(
                    json.dumps(
                        {
                            "skipped_reason": skip_reason,
                            "referenced_kpis": current_kpi_refs,
                            "kpi_configs": kpi_configs,
                        },
                        indent=2,
                    ),
                    name="Validation Skipped",
                    attachment_type=allure.attachment_type.JSON,
                )
            return validation_results

        # Pre-validate all KPIs to determine the overall status
        validation_passed = True if mode == "all" else False
        validation_count = 0
        at_least_one_passed = False

        # show results in debug log
        for kpi_name in current_kpi_refs:
            kpi_config = get_kpi_config(kpi_name)
            # Skip validation if KPI config is not defined
            if not kpi_config:
                continue
            # Skip validation if KPI validation is explicitly disabled
            if not kpi_config.get("validation", {}).get("enabled", True):
                continue

            # Handle mapping from generic KPI names to indexed device metrics
            # e.g., "throughput_dgpu" -> ["throughput_dgpu1", "throughput_dgpu2"]
            matched_metrics = []
            if kpi_name in results.metrics:
                # Exact match found
                matched_metrics = [kpi_name]
            else:
                # Check for indexed variants (e.g., throughput_dgpu1, throughput_dgpu2)
                for metric_name in results.metrics.keys():
                    if metric_name.startswith(f"{kpi_name}") and metric_name != kpi_name:
                        # Check if the suffix is a digit (to avoid false matches)
                        suffix = metric_name[len(kpi_name) :]
                        if suffix and suffix[0].isdigit():
                            matched_metrics.append(metric_name)

            # Validate using the first matching metric (highest priority)
            if matched_metrics:
                # Sort to ensure consistent ordering (dgpu1 before dgpu2)
                matched_metrics.sort()
                metric_name = matched_metrics[0]
                kpi_entry = results.metrics[metric_name]
                value = kpi_entry.value
                unit = kpi_entry.unit or ""
                logger.debug(f"Pre-validating KPI: {kpi_name} using metric: {metric_name} with value: {value}")
                from sysagent.utils.core.kpi import validate_kpi

                result = validate_kpi(value, kpi_config)
                result.unit = unit
                if result.passed:
                    at_least_one_passed = True
                else:
                    validation_passed = False
                validation_count += 1
            else:
                logger.warning(f"KPI {kpi_name} not found in results metrics")
                validation_passed = False
                validation_count += 1

        # Decide overall pass/fail based on mode
        if mode == "any":
            overall_passed = at_least_one_passed
        else:  # "all"
            overall_passed = validation_passed
        logger.debug(f"Overall validation status for test {test_name} with mode '{mode}': {overall_passed}")

        validation_results["passed"] = overall_passed

        # Determine the appropriate icon and suffix for the step title
        step_icon = "🟢" if overall_passed else "🔴"
        step_suffix = f"({validation_count} KPIs)"

        # Log the validation summary at INFO level
        if overall_passed:
            logger.info(f"KPI validations passed {step_suffix}")
        else:
            logger.info(f"KPI validation failed {step_suffix} - see test report for details")

        # Now create the step with the appropriate title based on pre-validation
        with allure.step(f"Validate test results {step_icon}"):
            logger.debug(f"Creating detailed validation report for test: {test_name}")

            # Now actually perform the validations with detailed steps for each KPI
            for kpi_name in current_kpi_refs:
                logger.debug(f"Validating KPI: {kpi_name}")
                kpi_config = get_kpi_config(kpi_name)
                # Skip validation if KPI config is not defined
                if not kpi_config:
                    with allure.step(f"Validate KPI: {kpi_name} ⚪"):
                        logger.warning(f"KPI {kpi_name} config not found - skipping validation")
                        result_json = {
                            "kpi_name": kpi_name,
                            "status": "SKIPPED",
                            "reason": "KPI configuration not defined",
                        }
                        allure.attach(
                            json.dumps(result_json, indent=2),
                            name="KPI Validation Result",
                            attachment_type=allure.attachment_type.JSON,
                        )
                        logger.info(f"Skipped validation for KPI {kpi_name} - not defined")
                        validation_results["validations"][kpi_name] = {
                            "skipped": True,
                            "reason": "KPI configuration not defined",
                        }
                    continue

                # Handle mapping from generic KPI names to indexed device metrics
                # e.g., "throughput_dgpu" -> ["throughput_dgpu1", "throughput_dgpu2"]
                matched_metrics = []
                if kpi_name in results.metrics:
                    # Exact match found
                    matched_metrics = [kpi_name]
                else:
                    # Check for indexed variants (e.g., throughput_dgpu1, throughput_dgpu2)
                    for metric_name in results.metrics.keys():
                        if metric_name.startswith(f"{kpi_name}") and metric_name != kpi_name:
                            # Check if the suffix is a digit (to avoid false matches)
                            suffix = metric_name[len(kpi_name) :]
                            if suffix and suffix[0].isdigit():
                                matched_metrics.append(metric_name)

                # Validate using the first matching metric (highest priority)
                if matched_metrics:
                    # Sort to ensure consistent ordering (dgpu1 before dgpu2)
                    matched_metrics.sort()
                    metric_name = matched_metrics[0]
                    kpi_entry = results.metrics[metric_name]
                    value = kpi_entry.value
                    unit = kpi_entry.unit or ""

                    # Log if using an indexed variant instead of exact match
                    if metric_name != kpi_name:
                        logger.info(
                            f"Using metric '{metric_name}' for KPI '{kpi_name}' "
                            f"(first of {len(matched_metrics)} matched metrics)"
                        )

                    from sysagent.utils.core.kpi import validate_kpi

                    result = validate_kpi(value, kpi_config)

                    # Format operator for display
                    operator_display_map = {
                        "eq": "==",
                        "neq": "!=",
                        "gt": ">",
                        "gte": ">=",
                        "lt": "<",
                        "lte": "<=",
                        "between": "between",
                        "contains": "contains",
                        "not_contains": "not contains",
                        "matches": "matches",
                        "in": "in",
                        "not_in": "not in",
                    }
                    operator_str = operator_display_map.get(
                        result.operator if isinstance(result.operator, str) else result.operator.value,
                        str(result.operator),
                    )
                    # Log the validation result
                    if result.passed:
                        logger.info(
                            f"KPI {kpi_name} PASSED: {result.actual_value} {unit} "
                            f"{operator_str} {result.expected_value} {unit}"
                        )
                        actual_str = (
                            f"{result.actual_value:.2f}"
                            if isinstance(result.actual_value, (int, float))
                            else str(result.actual_value)
                        )
                        expected_str = (
                            f"{result.expected_value:.2f}"
                            if isinstance(result.expected_value, (int, float))
                            else str(result.expected_value)
                        )
                        step_title = (
                            f"🟢 Validate KPI: {kpi_name} - Actual: {actual_str} "
                            f"{operator_str} Expected: {expected_str} {unit}"
                        )
                        with allure.step(step_title):
                            result_json = result.to_dict()
                            result_json["status"] = "PASSED"
                            allure.attach(
                                json.dumps(result_json, indent=2),
                                name="KPI Validation Result",
                                attachment_type=allure.attachment_type.JSON,
                            )
                    else:
                        logger.error(
                            f"KPI {kpi_name} FAILED: {result.actual_value} {unit} "
                            f"{operator_str} {result.expected_value} {unit}"
                        )
                        actual_str = (
                            f"{result.actual_value:.2f}"
                            if isinstance(result.actual_value, (int, float))
                            else str(result.actual_value)
                        )
                        expected_str = (
                            f"{result.expected_value:.2f}"
                            if isinstance(result.expected_value, (int, float))
                            else str(result.expected_value)
                        )
                        error_msg = (
                            f"KPI validation failed: {kpi_name}. "
                            f"Expected: {expected_str} {unit}, "
                            f"Actual: {actual_str} {unit}, "
                            f"Operator: {operator_str}"
                        )
                        result_json = result.to_dict()
                        result_json["status"] = "FAILED"
                        failed_step_name = (
                            f"Validate KPI: {kpi_name} - Actual: {actual_str} "
                            f"{operator_str} Expected: {expected_str} {unit}"
                        )
                        mark_step_as_failed(failed_step_name, error_msg, mode, result_json)
                    validation_count += 1
                    validation_results["validations"][kpi_name] = result.to_dict()
                else:
                    result_json = {
                        "kpi_name": kpi_name,
                        "status": "FAILED",
                        "error": "KPI not found in results",
                        "available_metrics": list(results.metrics.keys()),
                    }
                    validation_results["validations"][kpi_name] = {
                        "error": "KPI not found in results",
                        "available_metrics": list(results.metrics.keys()),
                    }
                    error_msg = (
                        f"Required KPI {kpi_name} not found in results. "
                        f"Available metrics: {list(results.metrics.keys())}"
                    )
                    missing_step_name = f"Validate KPI: {kpi_name} - MISSING METRIC"
                    mark_step_as_failed(missing_step_name, error_msg, mode, result_json)

            logger.info(f"Completed KPI validation with result: {overall_passed}")

            # Trigger pytest_check failure if overall_passed is False
            check.equal(overall_passed, True, f"KPI validation failed for test: {test_name}")

        return validation_results

    return _validate_results


@pytest.fixture
def validate_system_requirements_from_configs():
    """
    Fixture that returns a function to validate system requirements from
    test configurations. This fixture extracts system requirements from
    the test configurations, combining profile requirements with test-specific
    overrides, and validates them against the current system.
    The test will be skipped if the system requirements are not met.
    """
    logger = logging.getLogger(__name__)

    def _validate_system_requirements_from_configs(configs: dict) -> None:
        """Validate system requirements based on test configuration."""
        test_name = configs.get("name", "Unknown")
        logger.debug(f"Starting system requirements validation for test: {test_name}")

        # The merged requirements are already present in configs['requirements']
        has_requirements = False
        final_requirements = {"hardware": {}, "software": {}}

        # Categorize merged requirements from configs['requirements']
        if "requirements" in configs and configs["requirements"]:
            req_config = configs["requirements"]
            for key, value in req_config.items():
                if (
                    key.startswith("cpu_")
                    or key.startswith("memory_")
                    or key.startswith("storage_")
                    or key.startswith("network_")
                ):
                    final_requirements["hardware"][key] = value
                    has_requirements = True
                elif (
                    key.startswith("min_")
                    or key.startswith("required_")
                    or "available" in key
                    or "running" in key
                    or key.startswith("os_")
                    or "_min_version" in key
                    or key.startswith("docker_")
                ):
                    final_requirements["software"][key] = value
                    has_requirements = True
                else:
                    final_requirements["hardware"][key] = value
                    has_requirements = True
        else:
            logger.debug("No requirements found in configs")

        logger.info(f"Validating system requirements for test: {test_name}")

        if not has_requirements:
            logger.debug("No requirements found to validate")
            with allure.step("Validate system requirements ⚪"):
                logger.info("✓ No system requirements to validate")
                allure.attach(
                    json.dumps(
                        {
                            "status": "SKIPPED",
                            "message": "No system requirements to validate",
                        },
                        indent=2,
                    ),
                    name="Validation Result",
                    attachment_type=allure.attachment_type.JSON,
                )
            return

        # Setup directories
        data_dir = setup_data_dir()
        cache_dir = os.path.join(data_dir, "cache")
        logger.debug(f"Using cache directory: {cache_dir}")

        # Create system validator
        from sysagent.utils.testing.system_validator import SystemValidator

        validator = SystemValidator(cache_dir)

        # Attach requirements info function for reuse
        def attach_requirements_info():
            allure.attach(
                json.dumps(final_requirements, indent=2),
                name="System Requirements Used for Validation",
                attachment_type=allure.attachment_type.JSON,
            )

        # Validate requirements directly
        validation_results = validator.validate_requirements(final_requirements)

        def format_check_details(check):
            return f"Required {check['required']}, Found {check['actual']}"

        # Determine if we should fail or skip based on profile name
        profile_name = os.environ.get("ACTIVE_PROFILE", "none")
        logger.debug(f"Determined profile name: {profile_name}")
        should_fail = profile_name.startswith("profile.qualification")

        if validation_results["passed"]:
            with allure.step("Validate system requirements 🟢"):
                attach_requirements_info()
                all_checks = validation_results["checks"]
                passed_checks = [check for check in all_checks if check["passed"]]
                details = [f"✓ {format_check_details(check)}" for check in passed_checks]
                passed_details_message = "\n".join(details)
                logger.info("✓ System requirements validation passed")
                for detail in details:
                    logger.debug(detail)
                allure.attach(
                    json.dumps(
                        {
                            "status": "PASSED",
                            "message": "All system requirements met",
                            "passed_checks": passed_checks,
                            "all_checks": all_checks,
                        },
                        indent=2,
                    ),
                    name="Validation Result",
                    attachment_type=allure.attachment_type.JSON,
                )
                if passed_details_message:
                    allure.attach(
                        passed_details_message,
                        name="Passed System Requirement Checks",
                        attachment_type=allure.attachment_type.TEXT,
                    )
        else:
            with allure.step("Validate system requirements 🔴"):
                attach_requirements_info()
                failed_checks = [check for check in validation_results["checks"] if not check["passed"]]
                passed_checks = [check for check in validation_results["checks"] if check["passed"]]
                failed_details = [f"✗ {format_check_details(check)}" for check in failed_checks]
                passed_details = [f"✓ {format_check_details(check)}" for check in passed_checks]
                reason = "; ".join(format_check_details(c) for c in failed_checks)
                skip_message = f"System requirements not met: {reason}"
                failed_details_message = "\n".join(failed_details)
                passed_details_message = "\n".join(passed_details)
                logger.error(f"System requirements validation failed: {skip_message}")
                allure.attach(
                    json.dumps(
                        {
                            "status": "FAILED",
                            "message": skip_message,
                            "failed_checks": failed_checks,
                            "passed_checks": passed_checks,
                            "all_checks": validation_results["checks"],
                        },
                        indent=2,
                    ),
                    name="Validation Result",
                    attachment_type=allure.attachment_type.JSON,
                )
                if failed_details_message:
                    allure.attach(
                        failed_details_message,
                        name="Failed System Requirement Checks",
                        attachment_type=allure.attachment_type.TEXT,
                    )
                if passed_details_message:
                    allure.attach(
                        passed_details_message,
                        name="Passed System Requirement Checks",
                        attachment_type=allure.attachment_type.TEXT,
                    )
            if should_fail:
                pytest.fail(skip_message)
            else:
                pytest.skip(skip_message)

    return _validate_system_requirements_from_configs
