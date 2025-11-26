# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
General utility functions for ESQ test suite.

This module provides reusable utility functions for:
- Shell script execution
- Archive creation
- CSV data extraction and visualization
- Allure report attachments
"""

import io
import logging
import subprocess  # nosec B404
import tarfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import allure
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Configure matplotlib for headless CI environments
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def execute_shell_script(script_path: Union[str, Path], *args: Any) -> Optional[subprocess.CompletedProcess]:
    """
    Safely execute a shell script using subprocess with validation.

    This function validates the script path, builds a safe command list, and executes
    the script with proper error handling and logging.

    Args:
        script_path: Path to the shell script file to execute.
        *args: Variable length argument list to pass to the script. Each argument
               will be converted to a string.

    Returns:
        CompletedProcess object if execution succeeds, None if execution fails.
        The CompletedProcess contains stdout, stderr, and return code.

    Raises:
        FileNotFoundError: If the script file does not exist at the specified path.
        ValueError: If the script_path is not a regular file.

    Example:
        >>> result = execute_shell_script("/path/to/script.sh", "arg1", "arg2")
        >>> if result:
        ...     print(result.stdout)
    """
    script = Path(script_path)

    # Validate script path before executing
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    if not script.is_file():
        raise ValueError(f"Invalid script type (not a file): {script}")

    # Build safe command list
    cmd = ["bash", str(script)] + [str(arg) for arg in args]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Script executed successfully: {script}")
        logger.debug(f"stdout: {result.stdout}")
        logger.debug(f"stderr: {result.stderr}")
        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"Script execution failed: {script}, rc={e.returncode}")
        logger.debug(f"stderr: {e.stderr}")
        return None


def attach_csv_table_to_allure(csv_path: Union[str, Path], image_path: Optional[Union[str, Path]] = None) -> None:
    """
    Convert CSV file to a table image and attach to Allure report.

    This function reads a CSV file, renders it as a formatted table image,
    and attaches it to the Allure test report. Optionally, an additional
    image can also be attached.

    Args:
        csv_path: Path to the CSV file to render as a table.
        image_path: Optional path to an additional image file to attach.

    Returns:
        None

    Note:
        - If CSV file is not found, an error message is attached to Allure.
        - Any exceptions during processing are caught and attached to Allure.
        - The table is styled with gray header and appropriate sizing.

    Example:
        >>> attach_csv_table_to_allure("results.csv", "graph.png")
    """
    try:
        csv_file = Path(csv_path)

        if not csv_file.exists():
            allure.attach(
                f"CSV not found: {csv_path}",
                name="Error",
                attachment_type=allure.attachment_type.TEXT,
            )
            return

        df = pd.read_csv(csv_file)

        # Create figure and table
        fig, ax = plt.subplots(figsize=(len(df.columns) * 2, len(df) * 0.6 + 1))
        ax.axis("off")

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.2)

        # Style header row
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # header row
                cell.set_facecolor("#d9d9d9")  # light gray
                cell.set_text_props(weight="bold")

        # Save to buffer and attach
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        buf.seek(0)

        allure.attach(
            buf.read(),
            name="CSV Table Summary",
            attachment_type=allure.attachment_type.PNG,
        )

        # Attach additional image if provided
        if image_path:
            image_file = Path(image_path)
            if image_file.exists():
                with open(image_file, "rb") as f:
                    allure.attach(
                        f.read(),
                        name="Additional Image",
                        attachment_type=allure.attachment_type.PNG,
                    )

    except Exception as e:
        logger.error(f"Failed to attach CSV table to Allure: {e}", exc_info=True)
        allure.attach(
            str(e),
            name="Attach Error",
            attachment_type=allure.attachment_type.TEXT,
        )


def extract_csv_values(
    csv_path: Union[str, Path],
    filter_column: str,
    filter_value: Union[str, int, float],
    target_columns: Union[str, List[str]],
) -> Optional[Union[Any, Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Extract values from CSV rows matching a filter condition.

    This function reads a CSV file, filters rows where the specified column
    matches the given value, and extracts data from the target column(s).

    Args:
        csv_path: Path to the CSV file to read.
        filter_column: Name of the column to filter rows on.
        filter_value: Value to match in the filter column. Rows where
                      filter_column == filter_value will be selected.
        target_columns: Column name(s) to extract from matching rows.
                        Can be a single column name (str) or list of column names.

    Returns:
        - If single row matches and single target column: returns the scalar value
        - If single row matches and multiple target columns: returns dict of values
        - If multiple rows match: returns list of dicts
        - None if no matches found or validation fails

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If CSV file is empty or cannot be parsed.

    Warnings:
        Issues warnings (does not raise) for:
        - Missing filter column
        - Missing target columns
        - No matching rows
        - Data extraction failures

    Example:
        >>> # Extract single value
        >>> fps = extract_csv_values("results.csv", "test_id", "T001", "fps")
        >>> # Extract multiple values from single row
        >>> data = extract_csv_values("results.csv", "test_id", "T001", ["fps", "latency"])
        >>> # Extract from multiple matching rows
        >>> all_data = extract_csv_values("results.csv", "status", "passed", ["fps", "latency"])
    """
    csv_file = Path(csv_path)

    # Validate file exists and is not empty
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if csv_file.stat().st_size == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")

    # Parse CSV file
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e}") from e

    # Validate filter column exists
    if filter_column not in df.columns:
        warnings.warn(f"Filter column '{filter_column}' not found in CSV: {csv_path}")
        return None

    # Normalize target_columns to list
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    # Validate target columns exist
    missing_cols = [col for col in target_columns if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Target column(s) missing in CSV: {', '.join(missing_cols)}")
        return None

    # Filter rows
    filtered_df = df[df[filter_column] == filter_value]

    if filtered_df.empty:
        warnings.warn(f"No rows found where '{filter_column}' == '{filter_value}'")
        return None

    # Extract results
    try:
        results = filtered_df[target_columns].to_dict(orient="records")
    except Exception as e:
        warnings.warn(f"Failed to extract data: {e}")
        return None

    # Simplify output based on result count and column count
    if len(results) == 1:
        if len(target_columns) == 1:
            return results[0][target_columns[0]]
        return results[0]
    elif len(results) > 1:
        return results

    return None
