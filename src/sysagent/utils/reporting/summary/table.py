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
            seconds = int(duration_seconds % 60)
            return f"{minutes}m {seconds}s"
        else:  # 1 hour or more
            hours = int(duration_seconds // 3600)
            remaining_seconds = duration_seconds % 3600
            minutes = int(remaining_seconds // 60)
            seconds = int(remaining_seconds % 60)
            return f"{hours}h {minutes}m {seconds}s"

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
        pass_rate = (passed_tests / total_current_tests * 100) if total_current_tests > 0 else 0.0

        return {"status_counts": status_counts, "pass_rate": pass_rate}

    def should_exclude_from_summary(self, profile_name: Optional[str]) -> bool:
        """Check if profile should be excluded from text summary."""
        if not profile_name:
            return False
        return any(profile_name.startswith(excluded) for excluded in self.EXCLUDED_PROFILES)

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

        table_lines.append(f"Generated: {summary.get('generated_timestamp', 'unknown')}")
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
        table_lines.append(f"Total Duration:  {self._format_duration(summary.get('total_duration_seconds', 0))}")
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
            table_lines.append(f"Total Duration: {self._format_duration(current_run_duration)}")
            table_lines.append(f"Pass Rate: {current_run_stats['pass_rate']:.2f}%")

            # Add current run status breakdown with symbols
            for status, count in current_run_stats["status_counts"].items():
                if count > 0:
                    status_symbol = self._get_status_symbol(status)
                    table_lines.append(f"{status_symbol} {status.title()}: {count}")
            table_lines.append("")

        return "\n".join(table_lines)

    def _get_key_metric_display(self, test: Dict[str, Any], max_width: int = 21) -> str:
        """
        Get the key metric display string for a test.

        Values are rounded to at most 2 decimal places. The result is truncated
        with '..' if it exceeds max_width characters.

        Args:
            test: Test data dictionary containing metrics
            max_width: Maximum display width before truncating with '..'

        Returns:
            Formatted string with key metric value and unit, or 'N/A'
        """
        metrics = test.get("metrics", {})
        for metric_data in metrics.values():
            if metric_data.get("is_key_metric", False):
                value = metric_data.get("value")
                unit = metric_data.get("unit", "") or ""
                if value is not None:
                    if isinstance(value, float):
                        formatted = f"{round(value, 2):.2f}"
                    elif isinstance(value, int):
                        formatted = str(value)
                    else:
                        # For non-numeric, round if it looks numeric
                        try:
                            formatted = f"{round(float(value), 2):.2f}"
                        except (TypeError, ValueError):
                            formatted = str(value)
                    display = f"{formatted} {unit}".strip() if unit else formatted
                    if len(display) > max_width:
                        display = display[: max_width - 2] + ".."
                    return display
        return "N/A"

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
        table_lines.append("-" * 134)

        # Header for unique test cases view
        # Columns: TestName(74) + Metric(22) + Duration(15) + Runs(7) + Status(12) + 4 spaces = 134
        header = f"{'Test Name':<74} {'Metric':<22} {'Duration':<15} {'Runs':<7} {'Status':<12}"
        table_lines.append(header)
        table_lines.append("-" * 134)

        # Sort by test name alphabetically
        sorted_tests = sorted(unique_tests, key=lambda x: x.get("test_name", "Unknown").lower())

        for test in sorted_tests:
            total_runs = test.get("total_runs", 1)
            duration_seconds = test.get("duration_seconds", 0)  # Latest run duration
            status = test.get("status", "unknown")

            # Check if this test was part of the current run
            test_uuids = test.get("all_run_uuids", [])
            is_current_run = any(uuid in current_run_uuids for uuid in test_uuids)

            # Truncate test name to fit within 74-char column, leaving 4 chars of
            # trailing space so the adjacent Metric column is easy to read
            full_test_name = test.get("test_name", "Unknown")
            if len(full_test_name) > 70:
                test_name_display = full_test_name[:68] + ".."
            else:
                test_name_display = full_test_name

            # Get key metric display and formatted duration
            metric_display = self._get_key_metric_display(test)
            duration_display = self._format_duration(duration_seconds)

            # Add status symbol prefix; append ◼ to mark the current run.
            # Placing it in the last column avoids any alignment issues caused by
            # ◼ being rendered as a double-width character in some terminals.
            status_symbol = self._get_status_symbol(status)
            if is_current_run:
                status_display = f"{status_symbol} {status} \u25fc"
            else:
                status_display = f"{status_symbol} {status}"

            runs_display = str(total_runs)

            row = (
                f"{test_name_display:<74} {metric_display:<22} "
                f"{duration_display:<15} {runs_display:<7} "
                f"{status_display:<12}"
            )
            table_lines.append(row)

        table_lines.append("-" * 134)

        return "\n".join(table_lines)
