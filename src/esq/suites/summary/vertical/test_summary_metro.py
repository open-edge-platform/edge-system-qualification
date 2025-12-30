# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Summary Test Module for Metro Vertical Workloads.

This module provides benchmark result summarization for metro vertical workloads.
It collects test results from various benchmark categories and generates CSV reports.

Location: suites/summary/vertical/
Purpose: Centralized summarization of vertical workload benchmarks
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import allure
import pytest
from sysagent.utils.core import Metrics, Result
from sysagent.utils.reporting.summary import TestResultsExtractor

logger = logging.getLogger(__name__)


@allure.title("Metro Benchmark Summary")
def test_summary_metro(
    request,
    configs,
    cached_result,
    cache_result,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """
    Summarize metro vertical benchmark test results.

    This summary test collects and aggregates results from specific benchmark categories:
    - System Memory STREAM Benchmark
    - System GPU OpenVINO Benchmark
    - Vision AI OpenVINO Benchmark
    - Media Vision Benchmark
    - Vertical Proxy Pipeline benchmarks (SNVR, Headed VA, VSaaS, LPR)

    The test generates CSV reports for each benchmark category with detailed metrics.

    Test Location: suites/summary/vertical/test_summary_metro.py
    """

    # Request
    test_name = request.node.name.split("[")[0]

    # Parameters
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)
    kpi_validation_mode = configs.get("kpi_validation_mode", "all")
    benchmark_category = configs.get("benchmark_category", "unknown")
    target_test_ids = configs.get("target_test_ids", [])

    logger.info(f"Starting Metro Benchmark Summary: {test_display_name}")
    logger.debug(
        f"Test parameters: test_id={test_id}, category={benchmark_category}, target_tests={len(target_test_ids)}"
    )

    # Step 1: Validate system requirements (minimal for summary test)
    # No specific hardware requirements for summary generation

    # Step 2: Prepare test environment
    def prepare_test_function():
        logger.info(f"Preparing Metro Summary test environment for {benchmark_category}")

        result = Result(
            metadata={
                "status": True,
                "message": f"Metro Summary environment prepared for {benchmark_category}",
                "benchmark_category": benchmark_category,
            }
        )
        return result

    prepare_test(
        test_name=test_name,
        configs=configs,
        prepare_func=prepare_test_function,
        name="Environment",
    )

    # Step 3: Execute test
    def run_test():
        logger.info(f"Running Metro Summary for {benchmark_category}: {len(target_test_ids)} tests")

        # Get data directory
        data_dir = os.environ.get("CORE_DATA_DIR")
        if not data_dir:
            from sysagent.utils.config.config import get_project_name

            project_name = get_project_name()
            data_dir = os.path.join(os.getcwd(), f"{project_name}_data")

        # Extract test results
        extractor = TestResultsExtractor(data_dir)
        test_results = extractor.find_latest_test_results_by_test_id(target_test_ids)

        # Calculate summary metrics
        total_tests = len(target_test_ids)
        tests_found = len(test_results)
        tests_passed = sum(1 for r in test_results.values() if r.get("status") == "passed")
        tests_failed = sum(1 for r in test_results.values() if r.get("status") == "failed")
        tests_skipped = sum(1 for r in test_results.values() if r.get("status") == "skipped")
        tests_missing = total_tests - tests_found

        # Calculate coverage
        coverage_percent = (tests_found / total_tests * 100.0) if total_tests > 0 else 0.0

        # Calculate total execution time
        total_execution_time_seconds = 0.0
        for test_result in test_results.values():
            metadata = test_result.get("metadata", {})
            duration = metadata.get("duration_seconds", 0.0)
            total_execution_time_seconds += duration

        # Convert to minutes
        execution_time_minutes = total_execution_time_seconds / 60.0

        # Define metrics
        metrics = {
            "coverage": Metrics(unit="%", value=round(coverage_percent, 2)),
            "execution_time": Metrics(unit="min", value=round(execution_time_minutes, 2)),
            "total_test_cases": Metrics(unit="tests", value=total_tests),
        }

        # Initialize result template using from_test_config for automatic metadata application
        result = Result.from_test_config(
            configs=configs,
            parameters={
                "test_id": test_id,
                "benchmark_category": benchmark_category,
                "total_tests": total_tests,
                "display_name": test_display_name,
            },
            metrics=metrics,
            metadata={
                "status": True,
                "benchmark Category": benchmark_category,
                "tests_found": tests_found,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "tests_skipped": tests_skipped,
                "tests_missing": tests_missing,
            },
        )

        # Generate CSV attachment for this benchmark category
        try:
            csv_content = _generate_benchmark_csv(test_results, target_test_ids)
            csv_filename = f"metro_{benchmark_category}_{test_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            # Attach CSV to allure report
            allure.attach(
                csv_content,
                name=csv_filename,
                attachment_type=allure.attachment_type.CSV,
            )

            logger.info(f"Generated CSV for {benchmark_category}: {csv_filename}")

        except Exception as e:
            logger.error(f"Failed to generate CSV for {benchmark_category}: {e}")

        logger.info(
            f"Metro Summary for {benchmark_category}: {tests_found}/{total_tests} tests found, "
            f"{tests_passed} passed, {tests_failed} failed, coverage={coverage_percent:.1f}%"
        )

        return result

    # Initialize variables for finally block
    validation_results = {}
    results = None
    test_failed = False
    failure_message = ""

    try:
        # Execute the test with cache
        results = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            test_name=test_name,
            configs=configs,
            run_test_func=run_test,
        )
        logger.debug(f"Test results: {json.dumps(results.to_dict(), indent=2)}")

        # Step 4: Validate test results
        validation_results = validate_test_results(
            results=results,
            configs=configs,
            get_kpi_config=get_kpi_config,
            test_name=test_name,
            mode=kpi_validation_mode,
        )

        # Update KPI validation status in result metadata
        results.update_kpi_validation_status(validation_results, kpi_validation_mode)

        # Explicitly set coverage as the key metric for metro summary FIRST
        # (before auto_set_key_metric which may override it)
        if "coverage" in results.metrics:
            results.metrics["coverage"].key_metric = True
            logger.debug(f"Set 'coverage' as key metric for {benchmark_category} summary")

        # Note: Skipping auto_set_key_metric to preserve coverage as key metric
        # results.auto_set_key_metric(validation_results, kpi_validation_mode)

        # Add KPI configuration and validation results to the Result object
        current_kpi_refs = configs.get("kpi_refs", [])
        if current_kpi_refs:
            kpi_data = {}
            final_mode = results.get_final_validation_mode(validation_results, kpi_validation_mode)
            for kpi_name in current_kpi_refs:
                kpi_config = get_kpi_config(kpi_name)
                if kpi_config is not None:
                    kpi_data[kpi_name] = {
                        "config": kpi_config,
                        "validation": validation_results.get("validations", {}).get(kpi_name, {}),
                        "mode": final_mode,
                    }
            results.kpis = kpi_data

    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during Metro Summary test execution: {str(e)}"
        logger.error(failure_message, exc_info=True)

        if results is None:
            metrics = {
                "coverage": Metrics(unit="%", value=-1.0),
                "execution_time": Metrics(unit="min", value=-1.0),
                "test_total": Metrics(unit="tests", value=-1.0),
                "test_total_skip": Metrics(unit="tests", value=-1.0),
            }
            results = Result(
                parameters={
                    "test_id": test_id,
                    "display_name": test_display_name,
                },
                metrics=metrics,
                metadata={"status": False, "error_message": failure_message},
            )
        else:
            results.metadata["status"] = False
            results.metadata["error_message"] = failure_message

        try:
            validation_results = validate_test_results(
                results=results,
                configs=configs,
                get_kpi_config=get_kpi_config,
                test_name=test_name,
                mode=kpi_validation_mode,
            )
        except Exception as validation_error:
            logger.error(f"Validation also failed: {validation_error}", exc_info=True)
            validation_results = {}

    finally:
        try:
            summarize_test_results(
                results=results,
                configs=configs,
                get_kpi_config=get_kpi_config,
                test_name=test_name,
            )
        except Exception as summary_error:
            logger.error(f"Failed to generate test summary: {summary_error}", exc_info=True)
            allure.attach(
                json.dumps(
                    {"error": str(summary_error), "results": results.to_dict()},
                    indent=2,
                ),
                name="Summary Error",
                attachment_type=allure.attachment_type.JSON,
            )

        if test_failed:
            pytest.fail(failure_message)

    logger.info(f"Metro Summary test '{test_name}' completed successfully")


def _generate_benchmark_csv(test_results: Dict[str, Dict[str, Any]], target_test_ids: List[str]) -> str:
    """
    Generate CSV summary for benchmark category.

    CSV Format:
    Test Name | Reference Metric (unit) | [metadata columns] | [metric_name (unit)]

    Args:
        test_results: Dictionary of test results by test ID
        target_test_ids: List of target test IDs

    Returns:
        CSV content as string
    """
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Collect all available metric names with their units
    all_metrics = {}  # {metric_name: unit}
    key_metric_name = None
    key_metric_unit = None

    # Collect all available metadata field names
    all_metadata_fields = set()

    for test_result in test_results.values():
        core_metrics = test_result.get("core_metrics", {})
        if core_metrics:
            metrics = core_metrics.get("metrics", {})
            for metric_name, metric_data in metrics.items():
                if metric_name not in all_metrics:
                    unit = metric_data.get("unit", "")
                    all_metrics[metric_name] = unit

                # Find the key metric name and unit
                if not key_metric_name:
                    is_key = metric_data.get("is_key_metric", False) or metric_data.get("key_metric", False)
                    if is_key:
                        key_metric_name = metric_name
                        key_metric_unit = metric_data.get("unit", "")

            # Collect metadata fields from individual test result core JSON (excluding internal fields)
            # This extracts metadata from each target test's core_metrics.metadata and adds as CSV columns
            metadata = core_metrics.get("metadata", {})
            if metadata:
                for field_name in metadata.keys():
                    # Skip internal metadata fields that are not useful for CSV output
                    if field_name not in ["status", "error", "error_message", "kpi_validation_status"]:
                        all_metadata_fields.add(field_name)

    # Transform all_metrics keys to Title Case
    all_metrics = {key.replace("_", " ").title(): value for key, value in all_metrics.items()}

    # Transform key_metric_name to Title Case to match all_metrics keys
    if key_metric_name:
        key_metric_name_transformed = key_metric_name.replace("_", " ").title()
    else:
        key_metric_name_transformed = None

    # Write header
    # Use the actual key metric name with unit instead of generic "Reference Metric"
    if key_metric_name_transformed:
        if key_metric_unit:
            key_metric_header = f"{key_metric_name_transformed} ({key_metric_unit})"
        else:
            key_metric_header = key_metric_name_transformed
    else:
        key_metric_header = "Reference Metric"

    header = ["Test Name", key_metric_header]

    # Add metadata columns from individual test results (sorted for consistency)
    # These are extracted from each target test's core_metrics.metadata field
    sorted_metadata_fields = sorted(all_metadata_fields)
    for field_name in sorted_metadata_fields:
        # Convert snake_case to Title Case for header (e.g., "ai_workload" -> "Ai Workload")
        header_name = field_name.replace("_", " ").title()
        header.append(header_name)

    # Add all other metrics as columns with unit in header (excluding the key metric)
    sorted_metrics = sorted(all_metrics.keys())
    for metric_name in sorted_metrics:
        # Skip the key metric since it's already in the second column
        if metric_name == key_metric_name_transformed:
            continue

        unit = all_metrics[metric_name]
        if unit:
            header.append(f"{metric_name} ({unit})")
        else:
            header.append(metric_name)

    writer.writerow(header)
    # Write data rows
    for test_id in target_test_ids:
        if test_id in test_results:
            test_result = test_results[test_id]
            core_metrics = test_result.get("core_metrics") or {}

            # Get test name from Core Metrics JSON (top-level "name" field)
            test_name = core_metrics.get("name", f"{test_id} - Unknown")

            # Get reference metric (key metric) value only (unit is in header)
            ref_metric_value = ""

            if core_metrics:
                # Core metrics JSON has metrics at top level
                metrics_dict = core_metrics.get("metrics", {})
                for metric_name, metric_data in metrics_dict.items():
                    is_key = metric_data.get("is_key_metric", False) or metric_data.get("key_metric", False)
                    if is_key:
                        ref_metric_value = metric_data.get("value", "")
                        break

            row = [
                test_name,
                ref_metric_value,
            ]

            # Add metadata values from individual test result core JSON
            metadata = core_metrics.get("metadata", {})
            for field_name in sorted_metadata_fields:
                if metadata and field_name in metadata:
                    value = metadata[field_name]
                    # Convert value to string (handle various types)
                    if isinstance(value, (int, float, str, bool)):
                        row.append(str(value))
                    else:
                        # For complex types, use empty string
                        row.append("")
                else:
                    row.append("")

            # Add all metric values (excluding the key metric)
            if core_metrics:
                metrics_dict = core_metrics.get("metrics", {})
                metrics_dict = {key.replace("_", " ").title(): value for key, value in metrics_dict.items()}
                for metric_name in sorted_metrics:
                    # Skip the key metric since it's already in the second column
                    if metric_name == key_metric_name_transformed:
                        continue

                    if metric_name in metrics_dict:
                        metric_data = metrics_dict[metric_name]
                        value = metric_data.get("value", "")
                        row.append(value)
                    else:
                        row.append("")
            else:
                # No core metrics found
                for metric_name in sorted_metrics:
                    if metric_name != key_metric_name:
                        row.append("")

            writer.writerow(row)
        else:
            # Test not found - write row with missing data
            row = [f"Test ID: {test_id} - NOT FOUND", ""]

            # Add empty metadata columns
            for field_name in sorted_metadata_fields:
                row.append("")

            # Add empty metric columns
            for metric_name in sorted_metrics:
                if metric_name != key_metric_name_transformed:
                    row.append("")
            writer.writerow(row)

    return output.getvalue()
