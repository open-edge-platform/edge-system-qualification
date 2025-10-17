# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Core results summary generator.

This module provides the main functionality for generating comprehensive
test results summaries from Allure test results.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import SystemInfoCache for system information display
try:
    from sysagent.utils.system.cache import SystemInfoCache

    SYSTEM_INFO_AVAILABLE = True
except ImportError:
    SYSTEM_INFO_AVAILABLE = False
    logger.warning("SystemInfoCache not available - system summary will be skipped")


class CoreResultsSummaryGenerator:
    """Generator for core framework test results summary."""

    # Profiles to exclude from text execution summary
    EXCLUDED_PROFILES = ["core.cli", "core.system", "core.internal"]

    def __init__(self, data_dir: str):
        """
        Initialize summary generator.

        Args:
            data_dir: Path to core framework data directory
        """
        self.data_dir = data_dir
        self.allure_results_dir = os.path.join(data_dir, "results", "allure")
        self.core_results_dir = os.path.join(data_dir, "results", "core")

        # Ensure core results directory exists
        os.makedirs(self.core_results_dir, exist_ok=True)

    def _format_duration(self, duration_seconds: float) -> str:
        """
        Format duration in a human-readable format.

        Args:
            duration_seconds: Duration in seconds

        Returns:
            Formatted duration string
        """
        if duration_seconds < 60:
            return f"{duration_seconds:.3f} seconds"
        elif duration_seconds < 3600:  # Less than 1 hour
            minutes = int(duration_seconds // 60)
            seconds = duration_seconds % 60
            return f"{minutes}m {seconds:.3f}s"
        else:  # 1 hour or more
            hours = int(duration_seconds // 3600)
            remaining_seconds = duration_seconds % 3600
            minutes = int(remaining_seconds // 60)
            seconds = remaining_seconds % 60
            return f"{hours}h {minutes}m {seconds:.3f}s"

    def should_exclude_from_summary(self, profile_name: Optional[str]) -> bool:
        """
        Check if profile should be excluded from text execution summary.

        Args:
            profile_name: Name of the profile to check

        Returns:
            True if profile should be excluded, False otherwise
        """
        if not profile_name:
            return False

        for excluded in self.EXCLUDED_PROFILES:
            if profile_name.startswith(excluded):
                return True

        return False

    def _collect_system_summary_data(self) -> Dict[str, Any]:
        """
        Collect system summary data for inclusion in JSON summary.

        Returns:
            Dictionary containing system summary data
        """
        if not SYSTEM_INFO_AVAILABLE:
            return {}

        try:
            # Initialize system info cache
            data_dir = os.environ.get("CORE_DATA_DIR")
            if not data_dir:
                data_dir = self.data_dir

            cache_dir = os.path.join(data_dir, "cache")
            system_info_cache = SystemInfoCache(cache_dir)

            # Get hardware and software information
            hw_info = system_info_cache.get_hardware_info()
            sw_info = system_info_cache.get_software_info()

            system_data = {}

            # Hardware Summary - format for better display
            if hw_info:
                hardware_summary = {}

                # CPU information with proper formatting
                if "cpu" in hw_info:
                    cpu = hw_info["cpu"]
                    cpu_summary = {}
                    if cpu.get("brand"):
                        cpu_summary["brand"] = cpu["brand"]
                    if cpu.get("logical_cores") or cpu.get("logical_count"):
                        cpu_summary["logical_cores"] = cpu.get("logical_cores") or cpu.get("logical_count", "Unknown")
                    if cpu.get("frequency"):
                        freq = cpu["frequency"]
                        if isinstance(freq, dict) and freq.get("max"):
                            cpu_summary["frequency_mhz"] = freq["max"]
                        elif isinstance(freq, (int, float)):
                            cpu_summary["frequency_mhz"] = freq
                    hardware_summary["cpu"] = cpu_summary

                # Memory information with GB conversion
                if "memory" in hw_info:
                    memory = hw_info["memory"]
                    memory_summary = {}
                    if memory.get("total"):
                        # Convert bytes to GB (1000^3)
                        total_gb = memory["total"] / (1000**3)
                        memory_summary["total_gb"] = round(total_gb)
                    if memory.get("available"):
                        available_gb = memory["available"] / (1000**3)
                        memory_summary["available_gb"] = round(available_gb)
                    if memory.get("percent"):
                        memory_summary["used_percent"] = round(memory["percent"], 1)
                    hardware_summary["memory"] = memory_summary

                # GPU information
                if "gpu" in hw_info:
                    gpu_info = hw_info["gpu"]
                    gpu_summary = {
                        "device_count": gpu_info.get("total_count", 0),
                        "devices": [],
                    }
                    if "devices" in gpu_info:
                        for device in gpu_info["devices"]:
                            device_summary = {}
                            # Try multiple fields for device name
                            device_name = (
                                device.get("openvino", {}).get("full_device_name")
                                or device.get("full_name")
                                or device.get("name")
                                or "Unknown"
                            )
                            device_summary["full_name"] = device_name
                            if device.get("is_discrete") is not None:
                                device_summary["is_discrete"] = device["is_discrete"]
                            gpu_summary["devices"].append(device_summary)
                    hardware_summary["gpu"] = gpu_summary

                # NPU information
                if "npu" in hw_info:
                    npu_info = hw_info["npu"]
                    npu_summary = {
                        "device_count": npu_info.get("count", 0),
                        "devices": [],
                    }
                    if "devices" in npu_info:
                        for device in npu_info["devices"]:
                            device_summary = {}
                            # Try multiple fields for device name
                            device_name = (
                                device.get("openvino", {}).get("full_device_name")
                                or device.get("full_name")
                                or device.get("name")
                                or "Unknown"
                            )
                            device_summary["full_name"] = device_name
                            npu_summary["devices"].append(device_summary)
                    hardware_summary["npu"] = npu_summary

                # Storage information
                if "storage" in hw_info:
                    storage_info = hw_info["storage"]
                    storage_summary = {
                        "device_count": len(storage_info.get("devices", [])),
                        "devices": [],
                    }
                    if "devices" in storage_info:
                        for device in storage_info["devices"]:
                            device_summary = {}
                            if device.get("model"):
                                device_summary["model"] = device["model"]
                            if device.get("interface"):
                                device_summary["interface"] = device["interface"]
                            if device.get("size"):
                                # Convert bytes to GB (1000^3)
                                size_gb = round(device["size"] / (1000**3))
                                device_summary["size_gb"] = size_gb
                            storage_summary["devices"].append(device_summary)
                    hardware_summary["storage"] = storage_summary

                # DMI information (system vendor, product, motherboard)
                if "dmi" in hw_info:
                    dmi_info = hw_info["dmi"]
                    dmi_summary = {}

                    # System information
                    if "system" in dmi_info:
                        system = dmi_info["system"]
                        system_summary = {}
                        if system.get("vendor"):
                            system_summary["vendor"] = system["vendor"]
                        if system.get("product_name"):
                            system_summary["product_name"] = system["product_name"]
                        if system.get("product_version"):
                            system_summary["product_version"] = system["product_version"]
                        if system.get("product_family"):
                            system_summary["product_family"] = system["product_family"]
                        if system_summary:
                            dmi_summary["system"] = system_summary

                    # BIOS information
                    if "bios" in dmi_info:
                        bios = dmi_info["bios"]
                        bios_summary = {}
                        if bios.get("vendor"):
                            bios_summary["vendor"] = bios["vendor"]
                        if bios.get("version"):
                            bios_summary["version"] = bios["version"]
                        if bios.get("date"):
                            bios_summary["date"] = bios["date"]
                        if bios_summary:
                            dmi_summary["bios"] = bios_summary

                    # Motherboard information
                    if "motherboard" in dmi_info:
                        motherboard = dmi_info["motherboard"]
                        motherboard_summary = {}
                        if motherboard.get("vendor"):
                            motherboard_summary["vendor"] = motherboard["vendor"]
                        if motherboard.get("name"):
                            motherboard_summary["name"] = motherboard["name"]
                        if motherboard.get("version"):
                            motherboard_summary["version"] = motherboard["version"]
                        if motherboard_summary:
                            dmi_summary["motherboard"] = motherboard_summary

                    if dmi_summary:
                        hardware_summary["dmi"] = dmi_summary

                system_data["hardware"] = hardware_summary

            # Software Summary with proper OS formatting and package information
            if sw_info:
                software_summary = {}

                if "os" in sw_info:
                    os_info = sw_info["os"]
                    os_summary = {}

                    # Handle distribution information
                    if os_info.get("distribution"):
                        dist = os_info["distribution"]
                        if isinstance(dist, dict):
                            os_summary["name"] = dist.get("name", "Unknown")
                            if dist.get("pretty_name"):
                                os_summary["pretty_name"] = dist["pretty_name"]
                            if dist.get("version_id"):
                                os_summary["version"] = dist["version_id"]
                        else:
                            os_summary["name"] = str(dist)

                    if os_info.get("release"):
                        os_summary["release"] = os_info["release"]

                    software_summary["os"] = os_summary

                # Add Python environment information
                if "python" in sw_info:
                    python_info = sw_info["python"]
                    python_summary = {}

                    if python_info.get("version_info"):
                        version_info = python_info["version_info"]
                        python_summary["version"] = (
                            f"{version_info.get('major', 0)}."
                            f"{version_info.get('minor', 0)}."
                            f"{version_info.get('micro', 0)}"
                        )

                    if python_info.get("in_virtualenv"):
                        python_summary["virtual_environment"] = python_info["in_virtualenv"]

                    if python_info.get("pip_version"):
                        python_summary["pip_version"] = python_info["pip_version"]

                    software_summary["python"] = python_summary

                # Add Python packages information
                if "python_packages" in sw_info:
                    python_packages = sw_info["python_packages"]
                    software_summary["python_packages"] = {
                        "installed": python_packages.get("installed", {}),
                        "missing": python_packages.get("missing", []),
                        "total_packages": python_packages.get("total_installed", 0),
                    }

                # Add system packages information
                if "system_packages" in sw_info:
                    system_packages = sw_info["system_packages"]
                    software_summary["system_packages"] = {
                        "installed": system_packages.get("installed", {}),
                        "missing": system_packages.get("missing", []),
                        "package_manager": system_packages.get("package_manager", "unknown"),
                    }

                system_data["software"] = software_summary

            return system_data

        except Exception as e:
            logger.debug(f"Failed to collect system summary data: {e}")
            return {}

    def generate_summary(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive test results summary.

        Summary Logic:
        - Groups test results by historyId (unique test identifier)
        - Same testCaseId can exist in multiple parentSuite (test suite profile groups)
        - Unique test cases are determined by historyId
        - Total runs counted per historyId
        - Latest run result determined by latest stop_timestamp for same historyId

        Current Run Tracking:
        - Tracks latest_uuid for each test (most recent execution per historyId)
        - Stores all_run_uuids array containing all execution UUIDs for the test
        - Calculates current run duration from latest execution durations only
        - Differentiates new test executions from existing ones by UUID comparison
        - Current run UUIDs are those that don't exist in previous executions
        - Tracking performance of current test execution cycle vs historical data

        Args:
            verbose: Whether to include detailed test information

        Returns:
            Dictionary containing comprehensive test summary with current run tracking
        """
        from .parser import AllureResultsParser

        # Use simple defaults instead of auto-detection
        profile_name = None
        suite_name = None

        if not os.path.exists(self.allure_results_dir):
            logger.warning(f"Allure results directory not found: {self.allure_results_dir}")
            return self._create_empty_summary(profile_name, suite_name)

        # Parse Allure results
        parser = AllureResultsParser(self.allure_results_dir)
        test_results = parser.parse_test_results()

        if not test_results:
            logger.warning("No test results found to summarize")
            return self._create_empty_summary(profile_name, suite_name)

        # Generate summary using simplified logic
        summary = self._build_summary(test_results, profile_name, suite_name, verbose)

        return summary

    def _create_empty_summary(self, profile_name: Optional[str], suite_name: Optional[str]) -> Dict[str, Any]:
        """Create empty summary when no results are available."""
        return {
            "summary": {
                "profile_name": profile_name,
                "suite_name": suite_name,
                "generated_timestamp": datetime.now().isoformat(),
                "total_tests": 0,
                "total_duration_seconds": 0.0,
                "status_counts": {
                    "passed": 0,
                    "failed": 0,
                    "broken": 0,
                    "skipped": 0,
                    "unknown": 0,
                },
                "pass_rate": 0.0,
                "tests_by_profile": {},
                "tests_by_group": {},
                "tests_by_status": {},
                "longest_test": None,
                "shortest_test": None,
                "average_duration": 0.0,
            },
            "tests": [],
        }

    def _build_summary(
        self,
        test_results: List[Dict[str, Any]],
        profile_name: Optional[str],
        suite_name: Optional[str],
        verbose: bool,
    ) -> Dict[str, Any]:
        """
        Summary from test results using simplified historyId-based grouping.

        Logic:
        1. Extract metadata from all test results
        2. Group by historyId (unique test cases)
        3. Count total runs per historyId
        4. Determine latest result by stop_timestamp
        5. Calculate aggregate statistics
        """
        from .parser import AllureResultsParser

        parser = AllureResultsParser(self.allure_results_dir)

        # Extract metadata for all tests
        test_metadata = []
        for result in test_results:
            metadata = parser.extract_test_metadata(result)

            # Skip tests with empty historyId
            if metadata is None or not metadata.get("history_id"):
                continue

            # Skip tests from excluded profiles
            test_suite = metadata.get("suite", "")
            if self.should_exclude_from_summary(test_suite):
                continue

            test_metadata.append(metadata)

        # Group tests by historyId
        test_groups_by_history_id = {}
        for metadata in test_metadata:
            history_id = metadata.get("history_id", "unknown")
            if history_id not in test_groups_by_history_id:
                test_groups_by_history_id[history_id] = []
            test_groups_by_history_id[history_id].append(metadata)

        logger.debug(f"Found {len(test_groups_by_history_id)} unique test cases")
        logger.debug(f"Total test runs: {len(test_metadata)}")

        # Create unique test cases from historyId groups
        unique_test_cases = []
        current_run_test_uuids = []  # Track UUIDs of current run tests (new UUIDs)
        current_run_duration = 0.0  # Total duration of current run

        for history_id, runs in test_groups_by_history_id.items():
            # Sort runs by stop_timestamp to get the latest
            sorted_runs = sorted(runs, key=lambda x: x.get("stop_timestamp", 0), reverse=True)
            latest_run = sorted_runs[0]

            # Collect all UUIDs for this history group
            all_uuids = [run.get("uuid", "") for run in runs if run.get("uuid")]

            # Track latest run UUID
            latest_uuid = latest_run.get("uuid", "")

            # For current run detection: assume latest UUID is from current run
            # In a real scenario, you'd compare against previously stored UUIDs
            if latest_uuid:
                current_run_test_uuids.append(latest_uuid)
                current_run_duration += latest_run.get("duration_seconds", 0)

            # Calculate aggregated data for this test case
            total_runs = len(runs)
            all_durations = [run.get("duration_seconds", 0) for run in runs]
            total_duration = sum(all_durations)
            longest_duration = max(all_durations) if all_durations else 0
            average_duration = total_duration / len(all_durations) if all_durations else 0

            # Create unique test case using latest run data + aggregated stats
            unique_test_case = {
                "history_id": history_id,
                "test_name": latest_run.get("test_name", "Unknown"),
                "status": latest_run.get("status", "unknown"),
                "total_runs": total_runs,
                "duration_seconds": latest_run.get("duration_seconds", 0),  # Latest run duration
                "longest_duration_seconds": longest_duration,
                "total_duration_seconds": total_duration,
                "average_duration_seconds": round(average_duration, 3),
                "start_timestamp": latest_run.get("start_timestamp", 0),
                "stop_timestamp": latest_run.get("stop_timestamp", 0),
                "labels": latest_run.get("labels", {}),
                "status_message": latest_run.get("status_message", ""),
                "status_trace": latest_run.get("status_trace", ""),
                "steps_count": latest_run.get("steps_count", 0),
                "attachments_count": latest_run.get("attachments_count", 0),
                "suite": latest_run.get("suite", ""),
                "sub_suite": latest_run.get("sub_suite", ""),
                "test_case": latest_run.get("test_case", ""),
                "host": latest_run.get("host", ""),
                "thread": latest_run.get("thread", ""),
                "framework": latest_run.get("framework", ""),
                "language": latest_run.get("language", ""),
                "package": latest_run.get("package", ""),
                # Updated tracking fields
                "latest_uuid": latest_uuid,
                "all_run_uuids": all_uuids,  # Store all UUIDs for this test history
            }
            unique_test_cases.append(unique_test_case)

        # Calculate aggregate statistics
        total_tests = len(unique_test_cases)
        total_duration = sum(test["total_duration_seconds"] for test in unique_test_cases)

        # Build tests_by_profile (grouped by parentSuite from labels)
        tests_by_profile = {}
        for test_case in unique_test_cases:
            parent_suite = test_case.get("labels", {}).get("parentSuite", "unknown")
            if parent_suite not in tests_by_profile:
                tests_by_profile[parent_suite] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "broken": 0,
                    "skipped": 0,
                    "unknown": 0,
                    "duration_seconds": 0.0,
                }

            tests_by_profile[parent_suite]["total"] += 1
            tests_by_profile[parent_suite][test_case["status"]] += 1
            tests_by_profile[parent_suite]["duration_seconds"] += test_case["total_duration_seconds"]

        # Build tests_by_group (grouped by group label from test metadata)
        tests_by_group = {}
        for test_case in unique_test_cases:
            group_label = test_case.get("labels", {}).get("group", "unknown")
            if group_label not in tests_by_group:
                tests_by_group[group_label] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "broken": 0,
                    "skipped": 0,
                    "unknown": 0,
                    "duration_seconds": 0.0,
                }

            tests_by_group[group_label]["total"] += 1
            tests_by_group[group_label][test_case["status"]] += 1
            tests_by_group[group_label]["duration_seconds"] += test_case["total_duration_seconds"]

        # Calculate status counts
        status_counts = {
            "passed": sum(1 for tc in unique_test_cases if tc["status"] == "passed"),
            "failed": sum(1 for tc in unique_test_cases if tc["status"] == "failed"),
            "broken": sum(1 for tc in unique_test_cases if tc["status"] == "broken"),
            "skipped": sum(1 for tc in unique_test_cases if tc["status"] == "skipped"),
            "unknown": sum(
                1 for tc in unique_test_cases if tc["status"] not in ["passed", "failed", "broken", "skipped"]
            ),
        }

        # Calculate pass rate
        pass_rate = (status_counts["passed"] / total_tests * 100) if total_tests > 0 else 0.0

        # Collect system information
        system_summary_data = self._collect_system_summary_data()

        # Build final summary with improved current run tracking
        # Load existing test summary to compare UUIDs and detect truly new tests
        existing_uuids = set()
        existing_summary_file = os.path.join(self.core_results_dir, "test_summary.json")

        if os.path.exists(existing_summary_file):
            try:
                with open(existing_summary_file, "r", encoding="utf-8") as f:
                    existing_summary = json.load(f)

                # Collect all existing UUIDs from previous test runs
                for test in existing_summary.get("tests", []):
                    all_uuids = test.get("all_run_uuids", [])
                    existing_uuids.update(all_uuids)

                logger.debug(f"Loaded {len(existing_uuids)} existing UUIDs from previous summary")
            except Exception as e:
                logger.debug(f"Could not load existing summary: {e}")

        # Calculate current run UUIDs by detecting new UUIDs vs existing ones
        actual_current_run_uuids = []
        actual_current_run_duration = 0.0

        # Check each test's latest UUID against existing UUIDs
        for test_case in unique_test_cases:
            latest_uuid = test_case.get("latest_uuid", "")
            if latest_uuid and latest_uuid not in existing_uuids:
                actual_current_run_uuids.append(latest_uuid)
                actual_current_run_duration += test_case.get("duration_seconds", 0)

        logger.debug(f"Found {len(actual_current_run_uuids)} new test UUIDs in current run")
        logger.debug(f"Current run duration: {actual_current_run_duration:.3f} seconds")

        summary = {
            "summary": {
                "profile_name": profile_name,
                "suite_name": suite_name,
                "generated_timestamp": datetime.now().isoformat(),
                "total_tests": total_tests,
                "total_duration_seconds": round(total_duration, 3),
                "current_run_duration_seconds": round(actual_current_run_duration, 3),
                "current_run_test_count": len(actual_current_run_uuids),
                "current_run_test_uuids": actual_current_run_uuids,
                "status_counts": status_counts,
                "pass_rate": round(pass_rate, 2),
                "tests_by_profile": tests_by_profile,
                "tests_by_group": tests_by_group,
                "unique_test_scenarios": len(test_groups_by_history_id),
                "cache_optimization_detected": any(len(runs) > 1 for runs in test_groups_by_history_id.values()),
                "system_summary": system_summary_data,
            },
            "tests": unique_test_cases,
        }

        return summary

    def save_summary_to_file(self, summary: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save summary to JSON file.

        Args:
            summary: Summary dictionary to save
            filename: Optional filename (defaults to timestamp-based name)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = "test_summary.json"

        filepath = os.path.join(self.core_results_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            # Convert to relative path for cleaner CLI output
            relative_filepath = os.path.relpath(filepath, os.getcwd())
            logger.info(f"Test summary saved to: {relative_filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save summary to {filepath}: {e}")
            raise

    def generate_and_save_summary(self, verbose: bool = False, filename: Optional[str] = None) -> str:
        """
        Generate and save test results summary in one call.

        Args:
            verbose: Whether to include detailed test information
            filename: Optional filename for saved summary

        Returns:
            Path to saved summary file
        """
        # Auto-detect execution context from test results
        summary = self.generate_summary(verbose=verbose)
        return self.save_summary_to_file(summary, filename)
