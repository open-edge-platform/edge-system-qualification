# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Report generation command implementation.

Handles generating Allure reports from test results with proper
Node.js and Allure CLI setup and configuration.
"""

import logging
import os

from sysagent.utils.config import get_dist_version, setup_data_dir
from sysagent.utils.infrastructure import NODE_VERSION, setup_node
from sysagent.utils.logging import setup_command_logging
from sysagent.utils.reporting import ALLURE_DIR_NAME, ALLURE_VERSION, generate_allure_report

logger = logging.getLogger(__name__)


def generate_report(
    report_name: str = None, report_version: str = None, force: bool = False, debug: bool = False
) -> int:
    """
    Generate an Allure report from test results.

    Args:
        report_name: Custom name for the Allure report (default: "Edge System Qualification Report")
        report_version: Version string to use for allureVersion in config (if provided)
        force: Force reinstallation of Allure CLI even if it's already installed
        debug: Whether to show debug level logs

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        # Set up logging for this operation
        setup_command_logging("report", verbose=True, debug=debug)

        logger.info("Generating report")

        # Setup data directory
        data_dir = setup_data_dir()

        if not os.path.exists(data_dir):
            logger.error(f"Data directory does not exist: {data_dir}")
            return 1

        # Setup directories
        results_dir = os.path.join(data_dir, "results", "allure")
        report_dir = os.path.join(data_dir, "reports", "allure")
        os.makedirs(report_dir, exist_ok=True)

        if not os.path.exists(results_dir) or not os.listdir(results_dir):
            logger.error(f"No Allure results found in {results_dir}")
            return 1

        # Log the configured versions
        logger.debug(f"Using Node.js version: {NODE_VERSION}")
        logger.debug(f"Using Allure version: {ALLURE_VERSION}")

        # Install Node.js
        try:
            node_dir = setup_node()
            logger.debug(f"Node.js CLI installed at: {node_dir}")
        except Exception as e:
            logger.error(f"Failed to install Node.js: {e}")
            return 1

        # Setup directory for allure installation (using npm)
        project_dir = os.path.join(data_dir, "thirdparty", ALLURE_DIR_NAME)
        os.makedirs(project_dir, exist_ok=True)

        # If no report version provided, use the software version
        if report_version is None:
            report_version = get_dist_version()
            logger.debug(f"Using software version for report: {report_version}")

        # Generate report
        return generate_allure_report(
            node_dir=node_dir,
            results_dir=results_dir,
            report_dir=report_dir,
            report_name=report_name,
            report_version=report_version,
            force_reinstall=force,
            debug=debug,
        )

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return 1
