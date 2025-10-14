# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CLI Summary Attachment Test Module for the core framework.

This module contains tests that run after any 'run' CLI command is completed
to attach summary JSON files to the Allure report for easy access and analysis.
"""
import os
import sys
import logging
import glob
import json
import pytest
import allure
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@allure.title("CLI Summary Files Attachment")
def test_summary(configs):
    """
    Test that attaches CLI summary JSON files to the Allure report.
    
    This test runs after CLI commands to ensure all summary files generated
    during the execution are available in the Allure report for analysis
    and review purposes.
    
    This test runs in an isolated 'core' suite context to ensure it appears
    separately from the main test profiles in the Allure report.
    """
    # Get test configuration name for logging
    test_name = configs.get('name', 'CLI Summary Attachment')
    display_name = configs.get('display_name', 'CLI Summary Files Attachment')
    
    logger.info(f"Starting {display_name}: {test_name}")
    
    # Get data directory where summaries are stored
    data_dir = os.environ.get("CORE_DATA_DIR")
    if not data_dir:
        from sysagent.utils.config.config import get_project_name
        project_name = get_project_name()
        data_dir = os.path.join(os.getcwd(), f"{project_name}_data")
        logger.warning(f"CORE_DATA_DIR not set, using default: {data_dir}")
    
    core_results_dir = os.path.join(data_dir, "results", "core")
    
    # Check if results directory exists
    if not os.path.exists(core_results_dir):
        logger.warning(f"Core results directory not found: {core_results_dir}")
        return
    
    # Step 1: Find all summary files
    with allure.step("Discover summary files"):
        summary_files = _discover_summary_files(core_results_dir)
        logger.info(f"Found {len(summary_files)} summary files in {core_results_dir}")
        
        # Create discovery summary with detailed information
        if summary_files:
            summary_lines = ["Summary Files Discovery Report", "=" * 40, ""]
            
            # Calculate total storage
            total_size_mb = 0.0
            for summary_file in summary_files:
                file_size = _get_file_size_mb(summary_file)
                total_size_mb += file_size
            
            # Add header information
            summary_lines.extend([
                f"Generated on: {_get_current_timestamp()}",
                f"Total summary files: {len(summary_files)}",
                f"Total summary storage: {total_size_mb:.2f} MB",
                ""
            ])
            
            # Add individual file details with modification time and content preview
            for summary_file in summary_files:
                file_size = _get_file_size_mb(summary_file)
                mod_time = _get_file_modification_time(summary_file)
                preview = _get_summary_preview(summary_file)
                summary_lines.append(f"📊 {os.path.basename(summary_file)} ({file_size:.2f} MB) - {mod_time}")
                if preview:
                    summary_lines.append(f"   Preview: {preview}")
        else:
            summary_lines = ["No summary files found in core results directory"]
        
        allure.attach(
            "\n".join(summary_lines),
            name="Summary Files Discovery Report",
            attachment_type=allure.attachment_type.TEXT
        )
    
    # Step 2: Attach each summary file to the report
    with allure.step("Attach summary files"):
        for summary_file in summary_files:
            _attach_summary_file_to_report(summary_file)
    
    logger.info(f"Completed CLI summary attachment: {len(summary_files)} files processed")


def _discover_summary_files(core_results_dir: str) -> List[str]:
    """
    Discover all summary JSON files in the core results directory.
    
    Args:
        core_results_dir: Path to the core results directory
        
    Returns:
        List[str]: List of summary file paths
    """
    summary_files = []
    
    # Define patterns for summary files
    summary_patterns = [
        "test_summary_*.json",
        "*_summary.json"
    ]
    
    logger.debug(f"Searching for summary files in: {core_results_dir}")
    
    for pattern in summary_patterns:
        pattern_path = os.path.join(core_results_dir, pattern)
        found_files = glob.glob(pattern_path)
        for file_path in found_files:
            if os.path.isfile(file_path) and file_path not in summary_files:
                file_name = os.path.basename(file_path)
                summary_files.append(file_path)
                logger.debug(f"Found summary file: {file_name}")
    
    # Sort summary files by modification time (newest first)
    summary_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return summary_files


def _get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        float: File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except (OSError, IOError):
        return 0.0


def _get_summary_preview(file_path: str) -> str:
    """
    Get a preview of the summary file content.
    
    Args:
        file_path: Path to the summary file
        
    Returns:
        str: Preview information about the summary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        summary_section = summary_data.get("summary", {})
        total_tests = summary_section.get("total_tests", 0)
        pass_rate = summary_section.get("pass_rate", 0.0)
        profile_name = summary_section.get("profile_name")
        suite_name = summary_section.get("suite_name")
        
        preview_parts = []
        if profile_name:
            preview_parts.append(f"Profile: {profile_name}")
        if suite_name:
            preview_parts.append(f"Suite: {suite_name}")
        preview_parts.append(f"Tests: {total_tests}")
        preview_parts.append(f"Pass Rate: {pass_rate:.1f}%")
        
        return " | ".join(preview_parts)
        
    except Exception as e:
        logger.debug(f"Failed to get preview for {file_path}: {e}")
        return "Preview unavailable"


def _attach_summary_file_to_report(summary_file_path: str) -> None:
    """
    Attach a single summary JSON file to the Allure report.
    
    Args:
        summary_file_path: Path to the summary file to attach
    """
    try:
        file_name = os.path.basename(summary_file_path)
        
        logger.debug(f"Attaching summary file: {file_name}")
        
        # Read summary file content
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_content = f.read()
        
        # Parse JSON to get summary information for attachment name
        try:
            summary_data = json.loads(summary_content)
            summary_section = summary_data.get("summary", {})
            total_tests = summary_section.get("total_tests", 0)
            pass_rate = summary_section.get("pass_rate", 0.0)
            
            attachment_name = f"📊 {file_name}"
            
            # Add summary info to attachment name
            if total_tests > 0:
                attachment_name += f" (Tests: {total_tests}, Pass: {pass_rate:.1f}%)"
            else:
                attachment_name += " (Empty Summary)"
            
        except (json.JSONDecodeError, KeyError):
            attachment_name = f"📊 {file_name} (Invalid JSON)"
        
        # Attach the summary content as JSON
        allure.attach(
            summary_content,
            name=attachment_name,
            attachment_type=allure.attachment_type.JSON
        )
        
        # Also create a formatted text version for easy reading
        if summary_data:
            formatted_summary = _format_summary_for_display(summary_data)
            allure.attach(
                formatted_summary,
                name=f"📋 {file_name.replace('.json', '')} - Formatted Summary",
                attachment_type=allure.attachment_type.TEXT
            )
        
        logger.info(f"Attached summary file: {file_name}")
        
    except Exception as e:
        logger.error(f"Failed to attach summary file {summary_file_path}: {e}")
        # Attach error information instead
        allure.attach(
            f"Error reading summary file: {summary_file_path}\nError: {str(e)}",
            name=f"Error - {os.path.basename(summary_file_path)}",
            attachment_type=allure.attachment_type.TEXT
        )


def _format_summary_for_display(summary_data: Dict[str, Any]) -> str:
    """
    Format summary data for human-readable display.
    
    Args:
        summary_data: The parsed summary data
        
    Returns:
        str: Formatted summary text
    """
    try:
        summary_section = summary_data.get("summary", {})
        
        lines = []
        lines.append("TEST EXECUTION SUMMARY")
        lines.append("=" * 50)
        lines.append("")
        
        # Basic information
        profile_name = summary_section.get("profile_name")
        if profile_name:
            lines.append(f"Profile: {profile_name}")
        
        suite_name = summary_section.get("suite_name")
        if suite_name:
            lines.append(f"Suite: {suite_name}")
        
        generated_timestamp = summary_section.get("generated_timestamp", "unknown")
        lines.append(f"Generated: {generated_timestamp}")
        lines.append("")
        
        # Statistics
        lines.append("STATISTICS")
        lines.append("-" * 20)
        total_tests = summary_section.get("total_tests", 0)
        total_duration = summary_section.get("total_duration_seconds", 0)
        pass_rate = summary_section.get("pass_rate", 0.0)
        
        lines.append(f"Total Tests: {total_tests}")
        lines.append(f"Total Duration: {_format_duration(total_duration)}")
        lines.append(f"Pass Rate: {pass_rate:.2f}%")
        lines.append("")
        
        # Status breakdown
        status_counts = summary_section.get("status_counts", {})
        if any(count > 0 for count in status_counts.values()):
            lines.append("STATUS BREAKDOWN")
            lines.append("-" * 20)
            for status, count in status_counts.items():
                if count > 0:
                    lines.append(f"{status.title()}: {count}")
            lines.append("")
        
        # Suite breakdown (limited to avoid huge text)
        tests_by_suite = summary_section.get("tests_by_suite", {})
        if tests_by_suite:
            lines.append("SUITE BREAKDOWN")
            lines.append("-" * 20)
            for suite_name, suite_data in list(tests_by_suite.items())[:10]:  # Limit to first 10 suites
                total = suite_data.get("total", 0)
                passed = suite_data.get("passed", 0)
                lines.append(f"{suite_name}: {passed}/{total} passed")
            
            if len(tests_by_suite) > 10:
                lines.append(f"... and {len(tests_by_suite) - 10} more suites")
            lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error formatting summary: {str(e)}"


def _format_duration(duration_seconds: float) -> str:
    """
    Format duration in a human-readable format.
    
    Args:
        duration_seconds: Duration in seconds
        
    Returns:
        str: Formatted duration string
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


def _get_current_timestamp() -> str:
    """Get current timestamp as a formatted string."""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_file_modification_time(file_path: str) -> str:
    """
    Get file modification time as a formatted string.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Formatted modification time
    """
    try:
        import datetime
        mtime = os.path.getmtime(file_path)
        return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, IOError):
        return "Unknown"
