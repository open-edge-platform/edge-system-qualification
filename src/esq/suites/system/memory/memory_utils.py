# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for memory benchmark tests.

This module provides helper functions for:
- CSV file initialization and formatting
- Memory benchmark result processing
- Result file parsing and conversion
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def init_csv_file(csv_file_path: Path, tc_name: str = "Memory_Benchmark_Test", force: bool = False) -> None:
    """
    Initialize CSV file with header row.

    Args:
        csv_file_path: Path to the CSV file to initialize.
        tc_name: Test case name to use in CSV (default: "Memory_Benchmark_Test").
        force: If True, recreate the file even if it exists (default: False).

    Returns:
        None

    Example:
        >>> init_csv_file(Path("results.csv"), "Memory_BM_Test", force=True)
    """
    if force and csv_file_path.exists():
        csv_file_path.unlink()
        logger.debug(f"Removed existing CSV file: {csv_file_path}")

    if not csv_file_path.exists():
        csv_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_file_path, "w") as f:
            # Write header
            f.write("Memory_BM_Runner,Function,Best Rate,Avg Time,Min Time,Max Time,Result\n")
            # Write initial placeholder row
            f.write(f"{tc_name},NA,NA\n")
        logger.info(f"Initialized CSV file: {csv_file_path}")


def format_benchmark_results(result_data: str, test_flag: str, tc_name: str = "Memory_Benchmark_Test") -> str:
    """
    Format STREAM benchmark results into CSV format.

    Args:
        result_data: Raw result data containing benchmark output with function names and metrics.
        test_flag: Test result flag (e.g., "No Error", "FAIL").
        tc_name: Test case name to prepend to each row (default: "Memory_Benchmark_Test").

    Returns:
        Formatted CSV string with header and data rows.

    Example:
        >>> result = format_benchmark_results(raw_output, "No Error", "Memory_BM_Test")
    """
    lines = [line.strip() for line in result_data.split("\n") if line.strip()]

    if not lines:
        logger.warning("No lines found in result data")
        return ""

    # Find the header line with "Function" in it
    header_idx = -1
    for i, line in enumerate(lines):
        if "Function" in line and "Best Rate" in line:
            header_idx = i
            break
    
    if header_idx == -1:
        logger.error("Could not find benchmark header line with 'Function' and 'Best Rate'")
        return ""

    # Build CSV header
    csv_output = "Memory_BM_Runner,Function,Best Rate MB/s,Avg Time,Min Time,Max Time,Result\n"

    # Process data rows (after header)
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue
        
        # Split by whitespace
        parts = [part.strip() for part in line.split() if part.strip()]

        # Need at least 5 parts: function_name, best_rate, avg_time, min_time, max_time
        if len(parts) < 5:
            logger.debug(f"Skipping line with insufficient data: {line}")
            continue

        # Extract function name (remove trailing colon if present)
        function_name = parts[0].rstrip(":")

        # Extract metrics (next 4 values: best_rate, avg_time, min_time, max_time)
        best_rate = parts[1]
        avg_time = parts[2]
        min_time = parts[3]
        max_time = parts[4]

        # Format CSV row
        csv_row = f"{tc_name},{function_name},{best_rate},{avg_time},{min_time},{max_time},{test_flag}\n"
        csv_output += csv_row

    return csv_output


def update_csv_from_result(result_file_path: Path, csv_file_path: Path, tc_name: str = "Memory_Benchmark_Test") -> bool:
    """
    Update CSV file based on result file content.

    Reads the result file, determines test success/failure, extracts benchmark data,
    and updates the CSV file with formatted results.

    Args:
        result_file_path: Path to the result file containing benchmark output.
        csv_file_path: Path to the CSV file to update.
        tc_name: Test case name to use in CSV (default: "Memory_Benchmark_Test").

    Returns:
        True if test passed (exit code 0), False otherwise.

    Example:
        >>> success = update_csv_from_result(Path("result.txt"), Path("output.csv"))
    """
    if not result_file_path.exists():
        logger.error(f"Result file not found: {result_file_path}")
        # Create CSV with failure status
        init_csv_file(csv_file_path, tc_name, force=True)
        with open(csv_file_path, "a") as f:
            f.write(f"{tc_name},ERROR,Result file not found,,,FAIL\n")
        return False

    try:
        with open(result_file_path, "r") as f:
            result_content = f.read()

        # Check for errors in result content
        test_passed = True
        detail_result = ""

        # Check for success validation message first
        if "Solution Validates" in result_content or "avg error less than" in result_content:
            test_passed = True
            test_flag = "No Error"
        else:
            # Look for real error indicators (excluding success messages)
            error_patterns = [
                "error:", "failed", "exception", "cannot", "unable to", 
                "not found", "invalid", "ERROR", "FAIL"
            ]
            error_lines = []
            for line in result_content.split("\n"):
                line_lower = line.lower()
                # Skip lines that are success messages
                if "avg error less than" in line_lower or "solution validates" in line_lower:
                    continue
                # Check for actual errors
                if any(pattern.lower() in line_lower for pattern in error_patterns):
                    error_lines.append(line)
            
            if error_lines:
                test_passed = False
                test_flag = "FAIL"
                detail_result = "\n".join(error_lines[:10])  # Limit to first 10 error lines
                logger.warning(f"Errors detected in result file: {detail_result}")
            else:
                test_flag = "No Error"
        
        # Extract benchmark results (look for "Function" header)
        if test_passed:
            lines = result_content.split("\n")
            for i, line in enumerate(lines):
                if "Function" in line and "Best Rate" in line:
                    # Extract header and next 4 lines (benchmark data for Copy, Scale, Add, Triad)
                    detail_result = "\n".join(lines[i : i + 5])
                    break

        if not detail_result:
            logger.warning("No benchmark data found in result file")
            detail_result = result_content[:500]  # Take first 500 chars as fallback

        # Format results and recreate CSV file
        csv_content = format_benchmark_results(detail_result, test_flag, tc_name)

        if csv_content:
            # Recreate CSV file with new content
            csv_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_file_path, "w") as f:
                f.write(csv_content)
            logger.info(f"Updated CSV file: {csv_file_path}")
        else:
            logger.error("Failed to format benchmark results")
            init_csv_file(csv_file_path, tc_name, force=True)
            with open(csv_file_path, "a") as f:
                f.write(f"{tc_name},ERROR,Failed to parse results,,,FAIL\n")
            return False

        return test_passed

    except Exception as e:
        logger.error(f"Failed to update CSV from result file: {e}", exc_info=True)
        init_csv_file(csv_file_path, tc_name, force=True)
        with open(csv_file_path, "a") as f:
            f.write(f"{tc_name},ERROR,{str(e)},,,FAIL\n")
        return False
