# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Allure results parser.

This module provides functionality to parse Allure test result files
and extract metadata for summary generation.
"""

import glob
import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AllureResultsParser:
    """Parser for Allure test result files."""

    def __init__(self, allure_results_dir: str):
        """
        Initialize parser with Allure results directory.

        Args:
            allure_results_dir: Path to directory containing Allure result files
        """
        self.allure_results_dir = allure_results_dir

    def parse_test_results(self) -> List[Dict[str, Any]]:
        """
        Parse all Allure result JSON files.

        Returns:
            List of parsed test result dictionaries (with added 'file_uuid' field)
        """
        test_results = []
        result_files = glob.glob(os.path.join(self.allure_results_dir, "*-result.json"))

        logger.info(f"Found {len(result_files)} Allure result files to parse")

        for result_file in result_files:
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    result_data = json.load(f)
                    # Extract filename UUID (the actual filename, not the JSON uuid field)
                    filename = os.path.basename(result_file)
                    file_uuid = filename.replace("-result.json", "")
                    result_data["file_uuid"] = file_uuid
                    test_results.append(result_data)
            except Exception as e:
                logger.warning(f"Failed to parse result file {result_file}: {e}")

        return test_results

    def extract_test_metadata(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key metadata from a single test result.

        Args:
            test_result: Parsed Allure test result dictionary

        Returns:
            Dictionary containing extracted metadata
        """
        # Calculate duration in seconds
        start_time = test_result.get("start", 0)
        stop_time = test_result.get("stop", 0)
        duration_ms = stop_time - start_time if stop_time > start_time else 0
        duration_seconds = duration_ms / 1000.0 if duration_ms > 0 else 0.0

        # Extract labels for categorization
        labels = test_result.get("labels", [])
        label_dict = {label.get("name"): label.get("value") for label in labels}

        # Extract status details
        status_details = test_result.get("statusDetails", {})

        # Use historyId for grouping, but only if it's not empty
        # Tests with empty historyId (run via other methods) should be filtered out
        history_id = test_result.get("historyId", "")
        if not history_id or history_id.strip() == "":
            # Return None to indicate this test should be filtered out
            return None

        # Use test_title label if available, otherwise fall back to name
        display_name = label_dict.get("test_title", test_result.get("name", ""))

        metadata = {
            "uuid": test_result.get("uuid", ""),  # Unique execution ID
            "test_case_id": test_result.get("testCaseId", ""),  # Test case ID (base test file)
            "history_id": history_id,
            "test_name": display_name,
            "status": test_result.get("status", "unknown"),
            "duration_seconds": round(duration_seconds, 3),
            "start_timestamp": start_time,
            "stop_timestamp": stop_time,
            "labels": label_dict,
            "status_message": status_details.get("message", ""),
            "status_trace": status_details.get("trace", ""),
            "steps_count": len(test_result.get("steps", [])),
            "attachments_count": len(test_result.get("attachments", [])),
            "suite": label_dict.get("parentSuite", ""),
            "sub_suite": label_dict.get("suite", ""),
            "test_case": label_dict.get("subSuite", ""),
            "host": label_dict.get("host", ""),
            "thread": label_dict.get("thread", ""),
            "framework": label_dict.get("framework", ""),
            "language": label_dict.get("language", ""),
            "package": label_dict.get("package", ""),
        }

        return metadata
