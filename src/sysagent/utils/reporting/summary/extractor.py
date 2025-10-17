# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test results extractor from Allure attachments.

This module provides functionality to extract test results from Allure
attachments, specifically the "Core Metrics Test Results" JSON attachments.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TestResultsExtractor:
    """Extractor for test results from Allure attachments."""

    def __init__(self, data_dir: str):
        """
        Initialize extractor with data directory.

        Args:
            data_dir: Path to core framework data directory
        """
        self.data_dir = data_dir
        self.allure_results_dir = os.path.join(data_dir, "results", "allure")

    def extract_attachment_by_name(self, test_uuid: str, attachment_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract a specific attachment from a test result by name.

        Searches for attachments in both top-level and step-level locations.

        Args:
            test_uuid: UUID of the test result
            attachment_name: Name of the attachment to extract

        Returns:
            Dictionary containing attachment data, or None if not found
        """
        # Find the test result file
        result_file = os.path.join(self.allure_results_dir, f"{test_uuid}-result.json")

        if not os.path.exists(result_file):
            logger.warning(f"Test result file not found: {result_file}")
            return None

        try:
            # Load test result
            with open(result_file, "r", encoding="utf-8") as f:
                result_data = json.load(f)

            # Collect all attachments from multiple locations
            all_attachments = []

            # 1. Top-level attachments
            all_attachments.extend(result_data.get("attachments", []))

            # 2. Step-level attachments (including nested steps)
            def collect_step_attachments(steps):
                """Recursively collect attachments from steps and nested steps."""
                for step in steps:
                    all_attachments.extend(step.get("attachments", []))
                    # Recursively process nested steps
                    nested_steps = step.get("steps", [])
                    if nested_steps:
                        collect_step_attachments(nested_steps)

            steps = result_data.get("steps", [])
            if steps:
                collect_step_attachments(steps)

            # Find attachment with matching name
            for attachment in all_attachments:
                if attachment.get("name") == attachment_name:
                    # Load attachment file
                    attachment_source = attachment.get("source")
                    if attachment_source:
                        attachment_file = os.path.join(self.allure_results_dir, attachment_source)
                        if os.path.exists(attachment_file):
                            with open(attachment_file, "r", encoding="utf-8") as af:
                                attachment_data = json.load(af)
                            logger.debug(f"Found attachment '{attachment_name}' in test {test_uuid}")
                            return attachment_data

            logger.debug(f"Attachment '{attachment_name}' not found in test {test_uuid}")
            return None

        except Exception as e:
            logger.error(f"Failed to extract attachment from {result_file}: {e}")
            return None

    def extract_core_metrics_from_tests(self, test_uuids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Extract Core Metrics Test Results from multiple tests.

        Args:
            test_uuids: List of test UUIDs to extract metrics from

        Returns:
            Dictionary mapping test UUID to core metrics data
        """
        metrics_data = {}

        for test_uuid in test_uuids:
            metrics = self.extract_attachment_by_name(test_uuid, "Core Metrics Test Results")
            if metrics:
                metrics_data[test_uuid] = metrics
            else:
                logger.warning(f"No Core Metrics Test Results found for test {test_uuid}")

        logger.info(f"Extracted core metrics from {len(metrics_data)}/{len(test_uuids)} tests")
        return metrics_data

    def find_latest_test_results_by_test_id(self, test_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Find latest test results for specific test IDs.

        Args:
            test_ids: List of test IDs to find (e.g., "T0001", "MDA-DEC-001")

        Returns:
            Dictionary mapping test ID to latest test result data with core metrics
        """
        from .parser import AllureResultsParser

        parser = AllureResultsParser(self.allure_results_dir)
        all_results = parser.parse_test_results()

        # Group results by test ID
        results_by_test_id = {}
        for result in all_results:
            # Use file_uuid (filename UUID) instead of uuid (JSON internal UUID)
            file_uuid = result.get("file_uuid")
            if not file_uuid:
                logger.warning(f"Result missing file_uuid, skipping: {result.get('name')}")
                continue

            core_metrics = self.extract_attachment_by_name(file_uuid, "Core Metrics Test Results")

            test_id = None
            if core_metrics:
                # Check if Test ID is in Core Metrics parameters
                core_params = core_metrics.get("parameters", {})
                if isinstance(core_params, dict):
                    test_id = core_params.get("Test ID")

            # Fallback: Extract test ID from result parameters if not in Core Metrics
            if not test_id:
                params = result.get("parameters", [])
                for param in params:
                    param_name = param.get("name", "")
                    # Check for various forms of Test ID parameter
                    if param_name in ["Test ID", "Test Id", "test_id"]:
                        test_id = param.get("value")
                        # Remove quotes if present (e.g., "'VSN-THP-001'" -> "VSN-THP-001")
                        if isinstance(test_id, str) and test_id.startswith("'") and test_id.endswith("'"):
                            test_id = test_id[1:-1]
                        break

            if test_id and test_id in test_ids:
                if test_id not in results_by_test_id:
                    results_by_test_id[test_id] = []
                results_by_test_id[test_id].append(result)

        # Find latest result for each test ID
        latest_results = {}
        for test_id, results in results_by_test_id.items():
            # Sort by stop timestamp to get latest
            sorted_results = sorted(results, key=lambda x: x.get("stop", 0), reverse=True)
            if sorted_results:
                latest_result = sorted_results[0]
                # Use file_uuid for extracting attachments
                file_uuid = latest_result.get("file_uuid")

                # Extract core metrics
                core_metrics = self.extract_attachment_by_name(file_uuid, "Core Metrics Test Results")

                # Extract metadata
                metadata = parser.extract_test_metadata(latest_result)

                latest_results[test_id] = {
                    "uuid": file_uuid,  # Use file_uuid as the canonical UUID
                    "metadata": metadata,
                    "core_metrics": core_metrics,
                    "status": latest_result.get("status", "unknown"),
                    "raw_result": latest_result,
                }

        logger.info(f"Found latest results for {len(latest_results)}/{len(test_ids)} test IDs")
        return latest_results

    def find_all_test_results_by_prefix(self, test_id_prefix: str) -> Dict[str, Dict[str, Any]]:
        """
        Find all test results matching a test ID prefix.

        Args:
            test_id_prefix: Prefix to match (e.g., "SYS-MEM", "VSN-THP")

        Returns:
            Dictionary mapping test ID to latest test result data
        """
        from .parser import AllureResultsParser

        parser = AllureResultsParser(self.allure_results_dir)
        all_results = parser.parse_test_results()

        # Find all test IDs matching prefix
        matching_test_ids = set()
        for result in all_results:
            params = result.get("parameters", [])
            for param in params:
                if param.get("name") == "Test ID":
                    test_id = param.get("value")
                    if test_id and test_id.startswith(test_id_prefix):
                        matching_test_ids.add(test_id)

        # Get latest results for all matching test IDs
        return self.find_latest_test_results_by_test_id(list(matching_test_ids))
