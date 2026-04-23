# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test result summarization utilities for core testing framework.

This module provides fixtures and functions for summarizing test results
with standardized charts and reporting.

Now supports Result dataclass structure:
- Renders a table for 'data' (metric, value, unit)
- Renders a table for 'parameters' (key, value)
- Renders a table for 'kpis' (only for KPIs with validation enabled)
"""

import json
import logging
from typing import Any, Dict, Optional

import allure
import pytest

from sysagent.utils.core import Result
from sysagent.utils.plugins.pytest_telemetry import get_telemetry_collector

logger = logging.getLogger(__name__)


@pytest.fixture
def summarize_test_results(request):
    """
    Fixture that returns a function to summarize test results.

    This fixture standardizes test result summarization and Allure reporting
    for tests using the Result structure.
    - Renders a table for 'data' (metric, value, unit)
    - Renders a table for 'parameters' (key, value)
    - Attaches raw result as JSON
    - Optionally visualizes iteration data if provided

    Usage in tests:
        summarize_test_results(
            results=results,  # Result or dict
            test_name=test_name,
            iteration_data=iteration_data,  # Optional
            configs=configs,
            get_kpi_config=get_kpi_config
        )

    Returns:
        A function that performs test result summarization and creates appropriate
        Allure report steps
    """
    logger = logging.getLogger(__name__)

    def _summarize_results(
        results: Result,
        configs: Optional[Dict[str, Any]] = None,
        get_kpi_config: Optional[callable] = None,
        test_name: str = "Unknown",
        iteration_data: Optional[Dict[str, Any]] = None,
        enable_visualizations: bool = False,
    ) -> None:
        logger.info(f"Generating test result summary for test: {test_name}")
        results_dict = results.to_dict()

        # Apply telemetry data collected during this test (if telemetry was active).
        # Skip when the result came from cache: the cached result already carries the
        # full telemetry from the original run, and the current collector only has the
        # short near-empty samples from the cache-hit overhead — overwriting would lose
        # the original data.
        try:
            telemetry_collector = get_telemetry_collector(request.node)
            result_from_cache = getattr(request.node, "_result_from_cache", False)
            if telemetry_collector is not None and not result_from_cache:
                telemetry_collector.apply_to_result(results)
                results_dict = results.to_dict()  # refresh after telemetry merge
            elif result_from_cache:
                logger.debug("Cache hit: preserving cached telemetry data in result (skipping fresh apply).")
                # The outer Result may be a fresh container (empty extended_metadata) whose
                # per-device cached results were only partially merged (metrics + metadata).
                # Restore cached telemetry from the stash set by execute_test_with_cache.
                if not results.extended_metadata.get("telemetry"):
                    _stashed_telemetry = getattr(request.node, "_cached_telemetry", None)
                    if _stashed_telemetry:
                        results.extended_metadata["telemetry"] = _stashed_telemetry
                        logger.debug("Cache hit: restored stashed telemetry into outer Result.")
                results_dict = results.to_dict()  # ensure dict reflects cached extended_metadata
        except Exception as _tel_exc:
            logger.debug("Could not apply telemetry to result: %s", _tel_exc)

        # Update result name
        if configs:
            test_id = configs.get("test_id", "T0000")
            display_name = configs.get("display_name", test_name)
            results_dict["name"] = f"{test_id} - {display_name}"

        visualization_enabled = bool(enable_visualizations)
        plt_module = None
        create_results_table = None
        create_time_series_chart = None

        if visualization_enabled:
            try:
                import matplotlib.pyplot as plt_module  # type: ignore

                from sysagent.utils.reporting.visualization import (
                    create_results_table as _create_results_table,
                )
                from sysagent.utils.reporting.visualization import (
                    create_time_series_chart as _create_time_series_chart,
                )

                create_results_table = _create_results_table
                create_time_series_chart = _create_time_series_chart
            except ImportError as exc:
                logger.error(
                    "Visualization utilities not available for test summarization: %s",
                    exc,
                )
                visualization_enabled = False
                plt_module = None
            except Exception as exc:  # pragma: no cover - unexpected import failure
                logger.error("Failed to initialize visualization utilities: %s", exc)
                visualization_enabled = False
                plt_module = None

        try:
            with allure.step("Summarize test results"):
                logger.debug(f"Generating detailed test result summary for test: {test_name}")

                # Skip summarization if no results available
                if not results_dict:
                    logger.warning("No test results available to summarize")
                    allure.attach(
                        json.dumps(
                            {"skipped_reason": "No test results to summarize"},
                            indent=2,
                        ),
                        name="Summary Skipped",
                        attachment_type=allure.attachment_type.JSON,
                    )
                    logger.info("Skipped test results summary - no data available")
                    return
                else:
                    allure.attach(
                        json.dumps(results_dict, indent=2),
                        name="Core Metrics Test Results",
                        attachment_type=allure.attachment_type.JSON,
                    )

                if iteration_data:
                    allure.attach(
                        json.dumps(iteration_data, indent=2),
                        name="Iteration Data",
                        attachment_type=allure.attachment_type.JSON,
                    )

                if visualization_enabled and create_results_table and plt_module:
                    with allure.step("Generate test results table"):
                        try:
                            metrics = results_dict.get("metrics", {})
                            if metrics:
                                display_name = results_dict.get("parameters", {}).get("Display Name", test_name)
                                image_bytes, fig = create_results_table(
                                    metrics,
                                    title=f"{display_name} - Metrics",
                                    columns=["Metric", "Value", "Unit"],
                                )
                                allure.attach(
                                    image_bytes,
                                    name="Metrics Table",
                                    attachment_type=allure.attachment_type.PNG,
                                )
                                plt_module.close(fig)
                            else:
                                logger.warning("No metrics section found in results for metrics table.")

                        except Exception as exc:
                            logger.error(
                                "Error generating test results tables for summary: %s",
                                exc,
                            )
                            allure.attach(
                                json.dumps(
                                    {"error": str(exc), "results": results_dict},
                                    indent=2,
                                ),
                                name="Results Summary Error",
                                attachment_type=allure.attachment_type.JSON,
                            )
                else:
                    logger.debug("Visualization disabled; skipping metrics table generation")

                if visualization_enabled and create_time_series_chart and iteration_data:
                    with allure.step("Generate iteration trend charts"):
                        try:
                            chart_path = create_time_series_chart(iteration_data, test_name)
                            if chart_path:
                                with open(chart_path, "rb") as chart_file:
                                    allure.attach(
                                        chart_file.read(),
                                        name=f"{test_name} Performance Chart",
                                        attachment_type=allure.attachment_type.PNG,
                                    )
                                logger.debug("Added time series chart from %s", chart_path)
                            else:
                                logger.warning("Failed to create time series chart")
                        except Exception as exc:
                            logger.error(
                                "Error generating iteration data for summary: %s",
                                exc,
                            )
                            allure.attach(
                                json.dumps(
                                    {"error": str(exc), "iteration_data": iteration_data},
                                    indent=2,
                                ),
                                name="Iteration Summary Error",
                                attachment_type=allure.attachment_type.JSON,
                            )
                elif iteration_data:
                    logger.debug("Visualization disabled; skipping iteration chart generation")

                logger.info("Test result summary completed successfully")

        except Exception as exc:  # pragma: no cover - unexpected summarization failure
            logger.error("Unexpected error in test result summarization: %s", exc)

    return _summarize_results
