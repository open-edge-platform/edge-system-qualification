# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CLI Log Attachment Test Module for the core framework.

This module contains tests that run after any 'run' CLI command is completed
to attach log files to the Allure report for easy access and debugging.
"""
import os
import sys
import logging
import glob
import pytest
import allure
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@allure.title("CLI Log Files Attachment")
def test_log(configs):
    """
    Test that attaches CLI log files to the Allure report.
    
    This test runs after CLI commands to ensure all log files generated
    during the execution are available in the Allure report for debugging
    and analysis purposes.
    
    This test runs in an isolated 'core' suite context to ensure it appears
    separately from the main test profiles in the Allure report.
    """
    # Get test configuration name for logging
    test_name = configs.get('name', 'CLI Log Attachment')
    display_name = configs.get('display_name', 'CLI Log Files Attachment')
    
    logger.info(f"Starting {display_name}: {test_name}")
    
    # Get data directory where logs are stored
    data_dir = os.environ.get("CORE_DATA_DIR")
    if not data_dir:
        from sysagent.utils.config.config import get_project_name
        project_name = get_project_name()
        data_dir = os.path.join(os.getcwd(), f"{project_name}_data")
        logger.warning(f"CORE_DATA_DIR not set, using default: {data_dir}")
    
    logs_dir = os.path.join(data_dir, "logs")
    
    # Check if logs directory exists
    if not os.path.exists(logs_dir):
        logger.warning(f"Logs directory not found: {logs_dir}")
        return
    
    # Step 1: Find all log files
    with allure.step("Discover log files"):
        log_files = _discover_log_files(logs_dir)
        logger.info(f"Found {len(log_files)} log files in {logs_dir} (excluding attach_logs and other internal files)")
        
        # Create discovery summary with detailed information
        if log_files:
            summary_lines = ["Log Files Discovery Summary", "=" * 40, ""]
            
            # Calculate total storage
            total_size_mb = 0.0
            for log_file in log_files:
                file_size = _get_file_size_mb(log_file)
                total_size_mb += file_size
            
            # Add header information
            summary_lines.extend([
                f"Generated on: {_get_current_timestamp()}",
                f"Total log files: {len(log_files)}",
                f"Total log storage: {total_size_mb:.2f} MB",
                ""
            ])
            
            # Add individual file details with modification time
            for log_file in log_files:
                file_size = _get_file_size_mb(log_file)
                mod_time = _get_file_modification_time(log_file)
                summary_lines.append(f"📄 {os.path.basename(log_file)} ({file_size:.2f} MB) - {mod_time}")
        else:
            summary_lines = ["No log files found in logs directory"]
        
        allure.attach(
            "\n".join(summary_lines),
            name="Log Files Discovery Summary",
            attachment_type=allure.attachment_type.TEXT
        )
    
    # Step 2: Attach each log file to the report
    with allure.step("Attach log files"):
        for log_file in log_files:
            _attach_log_file_to_report(log_file)
    
    logger.info(f"Completed CLI log attachment: {len(log_files)} files processed")


def _discover_log_files(logs_dir: str) -> List[str]:
    """
    Discover all log files in the logs directory, excluding certain patterns.
    
    Args:
        logs_dir: Path to the logs directory
        
    Returns:
        List[str]: List of log file paths (excluding attach_logs and other excluded patterns)
    """
    log_files = []
    
    # Define patterns for different types of log files
    log_patterns = [
        "*.log",
        "*.log.*"  # Include rotated logs
    ]
    
    # Define exclusion patterns (files to skip)
    exclusion_patterns = [
        "*attach_logs*" 
    ]
    
    logger.debug(f"Searching for log files in: {logs_dir}")
    logger.debug(f"Exclusion patterns: {exclusion_patterns}")
    
    for pattern in log_patterns:
        pattern_path = os.path.join(logs_dir, pattern)
        found_files = glob.glob(pattern_path)
        for file_path in found_files:
            if os.path.isfile(file_path) and file_path not in log_files:
                file_name = os.path.basename(file_path)
                
                # Check if file matches any exclusion pattern
                should_exclude = False
                for exclusion_pattern in exclusion_patterns:
                    if _matches_pattern(file_name, exclusion_pattern):
                        logger.debug(f"Excluding log file (matches {exclusion_pattern}): {file_name}")
                        should_exclude = True
                        break
                
                if not should_exclude:
                    log_files.append(file_path)
                    logger.debug(f"Found log file: {file_name}")
    
    # Sort log files by modification time (newest first)
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return log_files


def _matches_pattern(filename: str, pattern: str) -> bool:
    """
    Check if a filename matches a glob-style pattern.
    
    Args:
        filename: The filename to check
        pattern: The glob pattern (e.g., "*attach_logs*")
        
    Returns:
        bool: True if the filename matches the pattern
    """
    import fnmatch
    return fnmatch.fnmatch(filename.lower(), pattern.lower())


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


def _attach_log_file_to_report(log_file_path: str) -> None:
    """
    Attach a single log file to the Allure report.
    
    Args:
        log_file_path: Path to the log file to attach
    """
    try:
        file_name = os.path.basename(log_file_path)
        file_size_mb = _get_file_size_mb(log_file_path)
        
        logger.debug(f"Attaching log file: {file_name} ({file_size_mb:.2f} MB)")
        
        # Read log file content
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
        
        attachment_name = f"📄 {file_name}"
        
        # Add file size to attachment name if significant
        if file_size_mb > 1.0:
            attachment_name += f" ({file_size_mb:.1f} MB)"
        
        # Attach the log content
        allure.attach(
            log_content,
            name=attachment_name,
            attachment_type=allure.attachment_type.TEXT
        )
        
        logger.info(f"Attached log file: {file_name}")
        
    except Exception as e:
        logger.error(f"Failed to attach log file {log_file_path}: {e}")
        # Attach error information instead
        allure.attach(
            f"Error reading log file: {log_file_path}\nError: {str(e)}",
            name=f"Error - {os.path.basename(log_file_path)}",
            attachment_type=allure.attachment_type.TEXT
        )


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
