# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Reporting utilities package.

This package provides comprehensive reporting functionality including:
- Allure report generation and configuration
- Test results summary generation
- Report formatting and output utilities
"""

# Import main allure functions
from sysagent.utils.reporting.allure import (
    ALLURE_DIR_NAME,
    ALLURE_VERSION,
    generate_allure_report,
    install_allure_cli_from_repo,
    update_allure_title_with_metrics,
)

# Import summary functions
from sysagent.utils.reporting.summary import (
    AllureResultsParser,
    CoreResultsSummaryGenerator,
    TestSummaryTableGenerator,
)

# Import visualization functions
from sysagent.utils.reporting.visualization import (
    create_comparison_chart,
    create_results_bar_chart,
    create_results_table,
    create_time_series_chart,
    save_chart_as_temp_file,
)

# Make sub-packages available
from . import allure, summary, visualization

__all__ = [
    # Allure functions
    "generate_allure_report",
    "install_allure_cli_from_repo",
    "update_allure_title_with_metrics",
    "ALLURE_DIR_NAME",
    "ALLURE_VERSION",
    # Summary functions
    "AllureResultsParser",
    "CoreResultsSummaryGenerator",
    "TestSummaryTableGenerator",
    # Visualization functions
    "create_results_table",
    "create_results_bar_chart",
    "create_time_series_chart",
    "save_chart_as_temp_file",
    "create_comparison_chart",
    # Sub-packages
    "allure",
    "summary",
    "visualization",
]
