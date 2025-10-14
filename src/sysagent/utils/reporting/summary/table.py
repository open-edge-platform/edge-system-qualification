# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test summary table generator.

This module provides functionality to generate formatted table displays
of test results summaries.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TestSummaryTableGenerator:
    """Generator for test summary tables from core results."""

    # Profiles to exclude from text summary display
    EXCLUDED_PROFILES = ["core.cli", "core.system", "core.internal"]

    def __init__(self, summary_data: Dict[str, Any]):
        """
        Initialize table generator with summary data.

        Args:
            summary_data: Test summary data dictionary
        """
        self.summary_data = summary_data

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

    def _format_system_summary(self, system_data: Dict[str, Any]) -> str:
        """
        Format system summary data for table display.

        Args:
            system_data: System summary data from JSON

        Returns:
            Formatted system summary string
        """
        # Use the consolidated formatter from system/formatter.py
        from ...system.formatter import format_system_summary

        hardware = system_data.get("hardware", {})
        software = system_data.get("software", {})

        return format_system_summary(hardware, software)

    def _calculate_current_run_stats(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate statistics for the current run only.

        Args:
            summary: Summary data dictionary

        Returns:
            Dictionary with current run statistics
        """
        current_run_uuids = summary.get("current_run_test_uuids", [])
        if not current_run_uuids:
            return {"status_counts": {}, "pass_rate": 0.0}

        # Find tests that were part of the current run
        all_tests = self.summary_data.get("tests", [])
        current_run_tests = []

        for test in all_tests:
            test_uuids = test.get("all_run_uuids", [])
            # Check if any of the current run UUIDs match this test's UUIDs
            if any(uuid in current_run_uuids for uuid in test_uuids):
                current_run_tests.append(test)

        # Calculate status counts for current run
        status_counts = {
            "passed": 0,
            "failed": 0,
            "broken": 0,
            "skipped": 0,
            "unknown": 0,
        }

        for test in current_run_tests:
            status = test.get("status", "unknown").lower()
            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts["unknown"] += 1

        # Calculate pass rate
        total_current_tests = len(current_run_tests)
        passed_tests = status_counts["passed"]
        pass_rate = (
            (passed_tests / total_current_tests * 100)
            if total_current_tests > 0
            else 0.0
        )

        return {"status_counts": status_counts, "pass_rate": pass_rate}

    def should_exclude_from_summary(self, profile_name: Optional[str]) -> bool:
        """Check if profile should be excluded from text summary."""
        if not profile_name:
            return False
        return any(
            profile_name.startswith(excluded) for excluded in self.EXCLUDED_PROFILES
        )

    def generate_summary_table(self) -> str:
        """
        Generate a formatted summary table string.

        Returns:
            Formatted summary table as string
        """
        summary = self.summary_data.get("summary", {})

        # Check if this should be excluded
        if self.should_exclude_from_summary(summary.get("profile_name")):
            return ""

        table_lines = []
        table_lines.append("=" * 137)
        table_lines.append("TEST EXECUTION SUMMARY")
        table_lines.append("=" * 137)

        # Basic information
        if summary.get("profile_name"):
            table_lines.append(f"Profile: {summary['profile_name']}")

        if summary.get("suite_name"):
            table_lines.append(f"Suite: {summary['suite_name']}")

        table_lines.append(
            f"Generated: {summary.get('generated_timestamp', 'unknown')}"
        )
        table_lines.append("")

        # Add system summary before overall statistics
        system_summary_data = summary.get("system_summary", {})
        if system_summary_data:
            system_summary = self._format_system_summary(system_summary_data)
        else:
            system_summary = ""

        if system_summary:
            table_lines.append(system_summary)

        # Overall statistics section (historical/cumulative data)
        status_counts = summary.get("status_counts", {})
        table_lines.append("OVERALL STATISTICS")
        table_lines.append("-" * 40)
        table_lines.append(f"Total Tests: {summary.get('total_tests', 0)}")
        table_lines.append(
            f"Total Duration: "
            f" {self._format_duration(summary.get('total_duration_seconds', 0))}"
        )
        table_lines.append(f"Pass Rate: {summary.get('pass_rate', 0):.2f}%")

        # Add status breakdown with symbols to overall statistics
        for status, count in status_counts.items():
            if count > 0:
                status_symbol = self._get_status_symbol(status)
                table_lines.append(f"{status_symbol} {status.title()}: {count}")
        table_lines.append("")

        # Current run statistics section (only show if there was a current run)
        current_run_duration = summary.get("current_run_duration_seconds", 0)
        current_run_count = summary.get("current_run_test_count", 0)
        if current_run_duration > 0 and current_run_count > 0:
            # Calculate current run statistics
            current_run_stats = self._calculate_current_run_stats(summary)

            table_lines.append("CURRENT RUN STATISTICS")
            table_lines.append("-" * 40)
            table_lines.append(f"Total Tests: {current_run_count}")
            table_lines.append(
                f"Total Duration: {self._format_duration(current_run_duration)}"
            )
            table_lines.append(f"Pass Rate: {current_run_stats['pass_rate']:.2f}%")

            # Add current run status breakdown with symbols
            for status, count in current_run_stats["status_counts"].items():
                if count > 0:
                    status_symbol = self._get_status_symbol(status)
                    table_lines.append(f"{status_symbol} {status.title()}: {count}")
            table_lines.append("")

        return "\n".join(table_lines)

    def _get_status_symbol(self, status: str) -> str:
        """
        Get the appropriate symbol for a test status.

        Args:
            status: Test status string

        Returns:
            Unicode symbol for the status
        """
        status_symbols = {
            "passed": "🟢",
            "failed": "🔴",
            "broken": "🟡",
            "skipped": "⚪",
            "unknown": "⚫",
        }
        return status_symbols.get(status.lower(), "?")

    def generate_detailed_test_table(self) -> str:
        """
        Generate detailed test table with unique test cases.

        Returns:
            Formatted detailed test table as string
        """
        # Check if this should be excluded
        summary = self.summary_data.get("summary", {})
        if self.should_exclude_from_summary(summary.get("profile_name")):
            return ""

        unique_tests = self.summary_data.get("tests", [])

        if not unique_tests:
            return "No test data available"

        # Get current run UUIDs for marking current run tests
        current_run_uuids = summary.get("current_run_test_uuids", [])

        table_lines = []
        table_lines.append("-" * 137)

        # Header for unique test cases view
        header = (
            f"{'ID':<9} {'Test Name':<70} {'Total Runs':<12} "
            f"{'Current (s)':<12} {'Longest (s)':<12} {'Status':<12}"
        )
        table_lines.append(header)
        table_lines.append("-" * 137)

        # Sort by test name alphabetically
        sorted_tests = sorted(
            unique_tests, key=lambda x: x.get("test_name", "Unknown").lower()
        )

        for test in sorted_tests:
            history_id = test.get("history_id", "Unknown")[:7]
            test_name = test.get("test_name", "Unknown")[:68]
            total_runs = test.get("total_runs", 1)
            current_duration = test.get("duration_seconds", 0)  # Latest run duration
            longest_duration = test.get("longest_duration_seconds", 0)
            status = test.get("status", "unknown")

            # Check if this test was part of the current run
            test_uuids = test.get("all_run_uuids", [])
            is_current_run = any(uuid in current_run_uuids for uuid in test_uuids)

            # Add current run indicator to test name if it was part of current run
            if is_current_run:
                # Add a "▶" symbol to indicate current run test
                current_run_indicator = " ◼"
                # Adjust the test name length to accommodate the indicator
                available_name_length = 66  # Reduced to fit indicator
                if len(test.get("test_name", "")) > available_name_length:
                    test_name = (
                        test.get("test_name", "Unknown")[:available_name_length] + ".."
                    )
                test_name_display = test_name + current_run_indicator
            else:
                test_name_display = test_name
                # Truncate if too long for non-current run tests
                if len(test.get("test_name", "")) > 68:
                    test_name_display = test_name + ".."

            # Add status symbol prefix
            status_symbol = self._get_status_symbol(status)
            status_display = f"{status_symbol} {status}"

            row = (
                f"{history_id:<9} {test_name_display:<70} {total_runs:<12} "
                f"{current_duration:<12.3f} {longest_duration:<12.3f} "
                f"{status_display:<12}"
            )
            table_lines.append(row)

        table_lines.append("-" * 137)

        return "\n".join(table_lines)
