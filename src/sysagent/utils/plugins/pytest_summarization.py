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

logger = logging.getLogger(__name__)


@pytest.fixture
def summarize_test_results():
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
    ) -> None:
        logger.info(f"Generating test result summary for test: {test_name}")
        results_dict = results.to_dict()

        # Update result name
        if configs:
            test_id = configs.get("test_id", "T0000")
            display_name = configs.get("display_name", test_name)
            results_dict["name"] = f"{test_id} - {display_name}"

        # Import the visualization utilities
        try:
            import matplotlib.pyplot as plt

            from sysagent.utils.reporting.visualization import (
                create_results_table,
                create_time_series_chart,
            )

            with allure.step("Summarize test results"):
                logger.debug(
                    f"Generating detailed test result summary for test: {test_name}"
                )

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

                # Create a results table visualization
                with allure.step("Generate test results table"):
                    try:
                        # Create a results table: result metrics (metric, value, unit)
                        metrics = results_dict.get("metrics", {})
                        if metrics:
                            display_name = results_dict.get("parameters", {}).get(
                                "Display Name", test_name
                            )
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
                            plt.close(fig)
                        else:
                            logger.warning(
                                "No metrics section found in results for metrics table."
                            )

                    except Exception as e:
                        logger.error(
                            f"Error generating test results tables for summary: {e}"
                        )
                        allure.attach(
                            json.dumps(
                                {"error": str(e), "results": results_dict}, indent=2
                            ),
                            name="Results Summary Error",
                            attachment_type=allure.attachment_type.JSON,
                        )

                # Create visualization for iteration data if provided
                if iteration_data:
                    with allure.step("Generate iteration trend charts"):
                        try:
                            allure.attach(
                                json.dumps(iteration_data, indent=2),
                                name="Iteration Data",
                                attachment_type=allure.attachment_type.JSON,
                            )
                            from sysagent.utils.reporting.visualization import (
                                create_time_series_chart,
                            )

                            chart_path = create_time_series_chart(
                                iteration_data, test_name
                            )
                            if chart_path:
                                with open(chart_path, "rb") as f:
                                    allure.attach(
                                        f.read(),
                                        name=f"{test_name} Performance Chart",
                                        attachment_type=allure.attachment_type.PNG,
                                    )
                                logger.debug(
                                    f"Added time series chart from {chart_path}"
                                )
                            else:
                                logger.warning("Failed to create time series chart")
                        except Exception as e:
                            logger.error(
                                f"Error generating iteration data for summary: {e}"
                            )
                            allure.attach(
                                json.dumps(
                                    {"error": str(e), "iteration_data": iteration_data},
                                    indent=2,
                                ),
                                name="Iteration Summary Error",
                                attachment_type=allure.attachment_type.JSON,
                            )

                logger.info("Test result summary completed successfully")

        except ImportError as e:
            logger.error(
                f"Failed to import visualization utilities for test summarization: {e}"
            )

        except Exception as e:
            logger.error(f"Unexpected error in test result summarization: {e}")

    return _summarize_results
