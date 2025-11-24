# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System Information Test Module for the core framework.

This module contains tests that collect and attach system information
to the Allure report for analysis and debugging purposes.
"""

import json
import logging
import os
from datetime import datetime

import allure

logger = logging.getLogger(__name__)


@allure.title("CLI System")
def test_system(configs):
    """
    Test that collects and attaches system information to the Allure report.

    This test gathers comprehensive system information including hardware
    details (CPU, memory, storage, GPU, NPU, DMI) and software information
    (OS, kernel) and attaches it as a JSON file to the Allure report for analysis.

    This test runs in an isolated 'core' suite context to ensure it appears
    separately from the main test profiles in the Allure report.
    """
    # Get test configuration name for logging
    test_name = configs.get("name", "System Information Attachment")
    display_name = configs.get("display_name", "System Information Collection")

    logger.info(f"Starting {display_name}: {test_name}")

    try:
        with allure.step("Collect and attach system information"):
            # Get data directory for system info cache
            data_dir = os.environ.get("CORE_DATA_DIR")
            if not data_dir:
                logger.warning("CORE_DATA_DIR not set - using default system info collection")
                system_data = {"error": "CORE_DATA_DIR environment variable not set"}
            else:
                try:
                    from sysagent.utils.system import SystemInfoCache

                    # Initialize system info cache and collect data
                    cache_dir = os.path.join(data_dir, "cache")
                    system_info_cache = SystemInfoCache(cache_dir)

                    # Get comprehensive system information
                    hardware_info = system_info_cache.get_hardware_info()
                    software_info = system_info_cache.get_software_info()

                    # Build system data structure similar to test summary format
                    system_data = {
                        "system_info": {
                            "generated_timestamp": datetime.now().isoformat(),
                            "collection_mode": "system_attachment",
                            "hardware": hardware_info,
                            "software": software_info,
                        }
                    }

                    logger.debug("System information collected successfully")

                except Exception as e:
                    logger.error(f"Failed to collect system information: {e}")
                    system_data = {
                        "error": f"Failed to collect system information: {str(e)}",
                        "generated_timestamp": datetime.now().isoformat(),
                    }

            # Convert system data to JSON string
            system_json = json.dumps(system_data, indent=2, default=str)

            # Attach system information as JSON file
            allure.attach(system_json, name="Core CLI System Information", attachment_type=allure.attachment_type.JSON)

            logger.debug(f"Completed system information attachment: {test_name}")

    except Exception as e:
        logger.error(f"Error in system information attachment: {e}")
        # Attach error information
        error_data = {"error": str(e), "test_name": test_name, "generated_timestamp": datetime.now().isoformat()}
        allure.attach(
            json.dumps(error_data, indent=2),
            name="System Information - Error",
            attachment_type=allure.attachment_type.JSON,
        )
