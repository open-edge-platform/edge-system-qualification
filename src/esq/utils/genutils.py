# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
General utility functions for ESQ test suite.

This module provides reusable utility functions for:
- CSV data extraction and visualization
- Allure report attachments
- Directory operations
- File download and extraction
- Generic test utilities (format logging, GPU device ID lookup)
"""

import csv
import io
import logging
import os
import shutil
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import allure
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import requests

# Configure matplotlib for headless CI environments
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

def collect_result_logs(result_dir: Union[str, Path]) -> str:
    """
    Collect all logs from result directory into a single string.

    Note: Docker container logs are now handled by docker_client.run_container() API
    with attach_logs option. This function only collects additional log files from
    the result directory.

    Args:
        result_dir: Directory containing log files.

    Returns:
        Consolidated log content as a string.
    """
    consolidated_logs = []

    # Collect result directory log files
    try:
        result_path = Path(result_dir)

        if result_path.exists() and result_path.is_dir():
            # Find all .log files recursively
            log_files = sorted(result_path.rglob("*.log"))

            if log_files:
                consolidated_logs.append("=" * 80)
                consolidated_logs.append("RESULT DIRECTORY LOG FILES")
                consolidated_logs.append("=" * 80)
                consolidated_logs.append(f"Directory: {result_path}\n")

                for log_file in log_files:
                    try:
                        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                            log_content = f.read()

                        relative_path = log_file.relative_to(result_path)
                        consolidated_logs.append(f"\n{'─' * 80}")
                        consolidated_logs.append(f"Log File: {relative_path}")
                        consolidated_logs.append(f"{'─' * 80}")
                        consolidated_logs.append(log_content)

                    except Exception as e:
                        consolidated_logs.append(f"\n[ERROR] Failed to read {log_file}: {e}")
            else:
                consolidated_logs.append("\n[INFO] No .log files found in result directory")
        else:
            consolidated_logs.append(f"\n[WARNING] Result directory not found: {result_dir}")

    except Exception as e:
        consolidated_logs.append(f"\n[ERROR] Failed to collect result directory logs: {e}")

    return "\n".join(consolidated_logs)

def attach_file_to_allure(
    file_path: Union[str, Path],
    attachment_name: str,
    attachment_type: allure.attachment_type = allure.attachment_type.TEXT,
) -> bool:
    """
    Attach a file to Allure report.

    This is a generic utility to attach any file (logs, images, CSVs, etc.)
    to the Allure test report.

    Args:
        file_path: Path to the file to attach.
        attachment_name: Name for the attachment in Allure report.
        attachment_type: Type of attachment (TEXT, PNG, CSV, etc.). Default: TEXT

    Returns:
        True if attachment succeeded, False otherwise.

    Example:
        >>> attach_file_to_allure("output.log", "Test Output Log")
        >>> attach_file_to_allure("graph.png", "Performance Graph", allure.attachment_type.PNG)
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"File not found for Allure attachment: {file_path}")
            return False

        # Determine read mode based on attachment type
        mode = "rb" if attachment_type in [allure.attachment_type.PNG, allure.attachment_type.JPG] else "r"

        with open(file_path, mode) as f:
            content = f.read()

        allure.attach(content, name=attachment_name, attachment_type=attachment_type)
        logger.debug(f"Attached file to Allure: {attachment_name}")
        return True

    except ImportError:
        logger.warning("Allure not available, skipping attachment")
        return False
    except Exception as e:
        logger.error(f"Failed to attach file to Allure: {e}")
        return False

def copy_directory_contents(src_dir: Union[str, Path], dst_dir: Union[str, Path]) -> None:
    """
    Recursively copy all contents from source directory to destination directory.

    This function copies all files and subdirectories from src_dir to dst_dir,
    preserving file metadata. If the destination directory doesn't exist, it will
    be created. Existing files in the destination will be overwritten.

    Args:
        src_dir: Source directory path containing files and directories to copy.
        dst_dir: Destination directory path where contents will be copied.

    Returns:
        None

    Raises:
        FileNotFoundError: If source directory does not exist.
        PermissionError: If insufficient permissions to read source or write to destination.
        OSError: If other I/O errors occur during copy operation.

    Example:
        >>> copy_directory_contents("/path/to/source", "/path/to/destination")
        # All contents from source are now in destination
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    # Validate source directory exists
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    if not src_path.is_dir():
        raise ValueError(f"Source path is not a directory: {src_dir}")

    # Ensure destination exists
    os.makedirs(dst_path, exist_ok=True)

    # Copy all items from source to destination
    for item in os.listdir(src_path):
        src_item = src_path / item
        dst_item = dst_path / item

        if src_item.is_dir():
            # Recursively copy directories
            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
            logger.debug(f"Copied directory: {src_item} -> {dst_item}")
        else:
            # Copy individual files with metadata preservation
            shutil.copy2(src_item, dst_item)
            logger.debug(f"Copied file: {src_item} -> {dst_item}")

    logger.info(f"Successfully copied all contents from {src_dir} to {dst_dir}")


def collect_all_logs(result_dir: Union[str, Path], container_name: Optional[str] = None) -> str:
    """
    Collect all logs from result directory into a single string.

    Note: Docker container logs are now handled by docker_client.run_container() API
    with attach_logs option. This function only collects additional log files from
    the result directory.

    Args:
        result_dir: Directory containing log files.
        container_name: Optional Docker container name (deprecated - kept for compatibility).

    Returns:
        Consolidated log content as a string.
    """
    consolidated_logs = []

    # Collect result directory log files
    try:
        result_path = Path(result_dir)

        if result_path.exists() and result_path.is_dir():
            # Find all .log files recursively
            log_files = sorted(result_path.rglob("*.log"))

            if log_files:
                consolidated_logs.append("=" * 80)
                consolidated_logs.append("RESULT DIRECTORY LOG FILES")
                consolidated_logs.append("=" * 80)
                consolidated_logs.append(f"Directory: {result_path}\n")

                for log_file in log_files:
                    try:
                        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                            log_content = f.read()

                        relative_path = log_file.relative_to(result_path)
                        consolidated_logs.append(f"\n{'─' * 80}")
                        consolidated_logs.append(f"Log File: {relative_path}")
                        consolidated_logs.append(f"{'─' * 80}")
                        consolidated_logs.append(log_content)

                    except Exception as e:
                        consolidated_logs.append(f"\n[ERROR] Failed to read {log_file}: {e}")
            else:
                consolidated_logs.append("\n[INFO] No .log files found in result directory")
        else:
            consolidated_logs.append(f"\n[WARNING] Result directory not found: {result_dir}")

    except Exception as e:
        consolidated_logs.append(f"\n[ERROR] Failed to collect result directory logs: {e}")

    # Note: Docker container log collection is now handled by docker_client.run_container()
    # with attach_logs=True option. The container_name parameter is kept for compatibility
    # but no longer used for log collection.
    if container_name:
        consolidated_logs.append(
            f"\n[INFO] Docker container logs for '{container_name}' are automatically "
            "attached by DockerClient.run_container() with attach_logs=True"
        )

    return "\n".join(consolidated_logs)

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

def plot_grouped_bar_chart(
    csv_path: Union[str, Path],
    output_path: Union[str, Path],
    x_column: str,
    y_column: str,
    group_column: Optional[str] = None,
    title: str = "Performance Comparison",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    reference_value: Optional[Union[float, Dict[str, float]]] = None,
    reference_label: Optional[str] = "Reference",
    figsize: tuple = (12, 6),
    rotation: int = 0,
    color_map: Optional[Dict[str, str]] = None,
    use_secondary_axis: bool = True,
    secondary_axis_threshold: float = 10.0,
) -> Path:
    """
    Create a grouped bar chart from CSV data with optional reference line(s).

    This function generates publication-quality bar charts suitable for test result
    visualization. It supports grouping bars by a category (e.g., precision, resolution),
    displaying them across different x-axis values (e.g., devices, operations), and
    adding horizontal reference line(s) for comparison.

    Reference values can be either:
    - A single float: draws one horizontal line across all x-axis categories
    - A dict mapping x-axis values to floats: draws individual reference values per category

    Reference Use Cases:
        1. OpenVINO benchmarks: Throughput (y) vs Devices (x), grouped by Precision
        2. Memory tests: Best Rate (y) vs Operations (x), no grouping
        3. Media tests: Max Channels (y) vs Devices (x), grouped by Resolution
        4. Frequency tests: Frequency (y) vs Time/Load (x), grouped by Metric type

    Args:
        csv_path: Path to CSV file containing the data to plot.
        output_path: Path where the PNG image will be saved.
        x_column: Column name for x-axis categories (e.g., "Device", "Operation").
        y_column: Column name for y-axis values (e.g., "Throughput", "Best Rate").
        group_column: Column name for grouping bars (e.g., "Precision", "Resolution").
                      Use same as x_column if no grouping is needed.
        title: Chart title.
        xlabel: X-axis label (default: uses x_column name).
        ylabel: Y-axis label (default: uses y_column name).
        reference_value: Optional reference value(s) to display as horizontal line(s).
                         Can be a single float (same for all categories) or a dict
                         mapping x-axis values to floats (different per category).
        reference_label: Label for the reference line (default: "Reference").
        figsize: Figure size as (width, height) in inches (default: (12, 6)).
        rotation: Rotation angle for x-axis labels (default: 0).
        color_map: Optional dict mapping group values to colors.
                   If None, uses default color palette.
        use_secondary_axis: Enable secondary y-axis when value ranges differ significantly
                            (default: True).
        secondary_axis_threshold: Ratio threshold to trigger secondary axis. If max/min ratio
                                  exceeds this value, creates secondary axis (default: 10.0).

    Returns:
        Path object pointing to the saved PNG file.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If required columns are missing from CSV.
        IOError: If output path cannot be written.

    Example:
        >>> # OpenVINO benchmark: Throughput vs Device, grouped by Precision
        >>> plot_grouped_bar_chart(
        ...     csv_path="results.csv",
        ...     output_path="ov_benchmark.png",
        ...     x_column="Device",
        ...     y_column="Throughput",
        ...     group_column="Precision",
        ...     title="ResNet-50 Performance",
        ...     ylabel="Throughput (FPS)",
        ...     reference_value=485.84,
        ...     reference_label="Reference (MTL 165H)"
        ... )

        >>> # Memory test: Best Rate vs Operation (no grouping)
        >>> plot_grouped_bar_chart(
        ...     csv_path="memory_results.csv",
        ...     output_path="memory_test.png",
        ...     x_column="Operation",
        ...     y_column="Best Rate",
        ...     group_column="Operation",  # Same as x_column for no grouping
        ...     title="Memory Bandwidth",
        ...     ylabel="Best Rate (MB/s)"
        ... )

        >>> # Media test: Max Channels vs Device, grouped by Resolution
        >>> plot_grouped_bar_chart(
        ...     csv_path="media_decode.csv",
        ...     output_path="h264_decode.png",
        ...     x_column="Device",
        ...     y_column="Max Channels",
        ...     group_column="Resolution",
        ...     title="H.264 Decode Performance",
        ...     ylabel="Max Channels (Stream Count)",
        ...     reference_value=8,
        ...     color_map={"1080p": "#1f77b4", "4K": "#ff7f0e"}
        ... )
    """
    csv_file = Path(csv_path)
    output_file = Path(output_path)

    # Validate input file
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV data
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}") from e

    # Normalize column names (strip leading/trailing spaces)
    df.columns = [col.strip() for col in df.columns]

    # Normalize string data in columns (strip spaces from values)
    for col in df.columns:
        if df[col].dtype == "object":  # String columns
            df[col] = df[col].astype(str).str.strip()

    # Validate required columns
    required_cols = [x_column, y_column, group_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing from CSV: {', '.join(missing_cols)}")

    # Replace NaN values with 0.0 for plotting
    df = df.fillna(0.0)

    # Set default labels
    if xlabel is None:
        xlabel = x_column
    if ylabel is None:
        ylabel = y_column

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique values for grouping
    x_categories = df[x_column].unique()
    groups = df[group_column].unique()

    # Set up bar positions
    x_pos = range(len(x_categories))
    bar_width = 0.8 / len(groups) if len(groups) > 1 else 0.6
    offset = -(len(groups) - 1) * bar_width / 2

    # Default color palette
    default_colors = plt.cm.tab10.colors if color_map is None else None

    # Analyze data to determine if secondary axis is needed
    all_values = []
    group_max_values = {}
    for group in groups:
        group_data = df[df[group_column] == group]
        max_val = group_data[y_column].max()
        group_max_values[group] = max_val
        all_values.extend(group_data[y_column].values)

    # Check if we need secondary axis (large value disparity between groups)
    non_zero_values = [v for v in all_values if v > 0]
    needs_secondary_axis = False
    if use_secondary_axis and len(non_zero_values) > 1 and len(groups) > 1:
        max_val = max(non_zero_values)
        min_val = min(non_zero_values)
        if min_val > 0 and (max_val / min_val) >= secondary_axis_threshold:
            needs_secondary_axis = True
            # Split groups into high value groups (above median) for secondary axis
            median_max = sorted(group_max_values.values())[len(group_max_values) // 2]
            high_value_groups = [g for g, v in group_max_values.items() if v > median_max]

    # Create secondary axis if needed
    ax2 = None
    if needs_secondary_axis:
        ax2 = ax.twinx()
        ax2.set_ylabel(f"{ylabel} (High Range)", fontsize=12, fontweight="bold", color="darkblue")
        ax2.tick_params(axis="y", labelcolor="darkblue")

    # Plot grouped bars
    for i, group in enumerate(groups):
        group_data = df[df[group_column] == group]

        # Get y values for each x category
        y_values = []
        for x_cat in x_categories:
            matching = group_data[group_data[x_column] == x_cat]
            if not matching.empty:
                y_values.append(matching[y_column].values[0])
            else:
                y_values.append(0.0)

        # Determine bar color
        if color_map and group in color_map:
            color = color_map[group]
        elif default_colors:
            color = default_colors[i % len(default_colors)]
        else:
            color = None

        # Determine which axis to use
        target_ax = ax
        if needs_secondary_axis and group in high_value_groups:
            target_ax = ax2
            # Make high value bars slightly transparent to distinguish
            alpha = 0.7
        else:
            alpha = 0.8

        # Plot bars for this group
        bar_positions = [x + offset + i * bar_width for x in x_pos]
        target_ax.bar(
            bar_positions,
            y_values,
            bar_width,
            label=str(group),
            color=color,
            alpha=alpha,
            edgecolor="black",
            linewidth=0.5,
        )

    # Add reference line(s) with marker points if provided
    if reference_value is not None and reference_label is not None:
        # Check if reference_value is a dict (per-category references) or single value
        if isinstance(reference_value, dict):
            # Determine if dict maps x-axis categories or group names
            dict_keys_set = set(reference_value.keys())
            x_categories_set = set(x_categories)

            # Check if dict keys match x-axis categories (mode-specific for LPR)
            if dict_keys_set == x_categories_set or all(k in x_categories_set for k in dict_keys_set):
                # Dict maps x-axis categories - one reference per x position
                ref_y_values = []
                for x_cat in x_categories:
                    ref_y_values.append(reference_value.get(x_cat, 0))

                # Plot stepped/connected reference line through all points
                ax.plot(
                    x_pos,
                    ref_y_values,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    marker="o",
                    markersize=8,
                    label=f"{reference_label}",
                    alpha=0.7,
                    zorder=10,
                )

                # Add value annotations at each position
                for i, (x, y) in enumerate(zip(x_pos, ref_y_values)):
                    if y > 0:  # Only annotate non-zero values
                        ax.annotate(
                            f"{y:.1f}",
                            xy=(x, y),
                            xytext=(0, 10),
                            textcoords="offset points",
                            fontsize=9,
                            fontweight="bold",
                            color="green",
                            ha="center",
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                facecolor="white",
                                edgecolor="green",
                                alpha=0.8,
                            ),
                            zorder=12,
                        )
            else:
                # Dict maps group names to reference values (model-specific)
                # Draw horizontal reference lines for each group across all x positions
                if group_column and len(groups) > 0:
                    n_groups_ref = len(groups)
                    logger.info(f"Drawing group-based reference lines for groups: {groups}")
                    logger.info(f"Reference values: {reference_value}")

                    for group_idx, group in enumerate(groups):
                        ref_val = reference_value.get(group)
                        if ref_val is not None and ref_val > 0:
                            # Calculate bar positions for this group across all x positions
                            group_offset = (group_idx - n_groups_ref / 2 + 0.5) * bar_width
                            group_x_pos = [x + group_offset for x in x_pos]

                            # Draw horizontal line from marker point to right side (for annotation)
                            if len(group_x_pos) > 0:
                                # Get the rightmost marker position for this group
                                rightmost_marker = max(group_x_pos)

                                # Calculate extended position for annotation (beyond plot)
                                # We'll draw line and let annotation be placed to the right
                                ax.plot(
                                    [rightmost_marker, rightmost_marker + 0.5],
                                    [ref_val, ref_val],
                                    color="green",
                                    linestyle="--",
                                    linewidth=1.5,
                                    alpha=0.7,
                                    zorder=10,
                                )

                            # Add marker points at bar positions
                            ax.plot(
                                group_x_pos,
                                [ref_val] * len(group_x_pos),
                                color="green",
                                linestyle="",
                                marker="o",
                                markersize=6,
                                label=f"{reference_label} ({group})" if group_idx == 0 else "",
                                zorder=11,
                            )

                            # Add annotation at the last x position for this group
                            if len(group_x_pos) > 0:
                                ax.annotate(
                                    f"{ref_val:.0f}",
                                    xy=(group_x_pos[-1], ref_val),
                                    xytext=(5, 0),
                                    textcoords="offset points",
                                    fontsize=8,
                                    fontweight="bold",
                                    color="green",
                                    ha="left",
                                    bbox=dict(
                                        boxstyle="round,pad=0.2",
                                        facecolor="white",
                                        edgecolor="green",
                                        alpha=0.8,
                                    ),
                                    zorder=12,
                                )
        else:
            # Single reference value - horizontal line across all categories
            ax.axhline(
                y=reference_value,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"{reference_label}: {reference_value:.2f}",
                alpha=0.7,
                zorder=10,
            )

            # Add marker points along the reference line at each x position
            ref_y = [reference_value] * len(x_pos)
            ax.plot(
                x_pos,
                ref_y,
                marker="o",
                markersize=8,
                color="green",
                linestyle="None",
                alpha=0.9,
                zorder=11,
            )

            # Add value annotation at the rightmost position
            ax.annotate(
                f"{reference_value:.2f}",
                xy=(x_pos[-1], reference_value),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="green",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="green", alpha=0.8),
                zorder=12,
            )

    # Customize chart
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    if needs_secondary_axis:
        ax.set_ylabel(f"{ylabel} (Low Range)", fontsize=12, fontweight="bold", color="darkgreen")
        ax.tick_params(axis="y", labelcolor="darkgreen")
    else:
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_categories, rotation=rotation)

    # Combine legends from both axes if secondary axis is used
    # Place legend outside plot area (upper right, outside bbox)
    if needs_secondary_axis and ax2 is not None:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            framealpha=0.9,
            fontsize=10,
            borderaxespad=0,
        )
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.9, fontsize=10, borderaxespad=0)

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved chart to: {output_file}")
    except Exception as e:
        raise IOError(f"Failed to save chart: {e}") from e
    finally:
        plt.close(fig)

    return output_file


def plot_multi_grouped_bar_chart(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    split_column: str,
    x_column: str,
    y_column: str,
    group_column: str,
    title_template: str,
    xlabel: str = None,
    ylabel: str = None,
    reference_column: Optional[str] = None,
    reference_label: str = "Reference",
    figsize: tuple = (12, 6),
    rotation: int = 0,
    color_map: Optional[Dict[str, str]] = None,
    use_secondary_axis: bool = True,
    secondary_axis_threshold: float = 10.0,
) -> List[Path]:
    """
    Create multiple grouped bar charts, one for each unique value in split_column.

    This function is useful when you need separate charts for different models,
    codecs, or test scenarios from a single CSV file.

    Args:
        csv_path: Path to CSV file containing the data.
        output_dir: Directory where PNG images will be saved.
        split_column: Column to split data by (e.g., "Model", "Codec").
                      One chart will be created for each unique value.
        x_column: Column name for x-axis (e.g., "Device").
        y_column: Column name for y-axis (e.g., "Throughput").
        group_column: Column name for grouping bars (e.g., "Precision").
        title_template: Title template with {value} placeholder for split value.
                        Example: "{value} Performance"
        xlabel: X-axis label (default: uses x_column name).
        ylabel: Y-axis label (default: uses y_column name).
        reference_column: Optional column containing reference values.
                          Each split will use its corresponding reference value.
        reference_label: Label for reference lines (default: "Reference").
        figsize: Figure size as (width, height) in inches.
        rotation: Rotation angle for x-axis labels.
        color_map: Optional dict mapping group values to colors.

    Returns:
        List of Path objects pointing to the saved PNG files.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If required columns are missing.

    Example:
        >>> # Create separate charts for each model in OpenVINO benchmark
        >>> plot_multi_grouped_bar_chart(
        ...     csv_path="all_models_results.csv",
        ...     output_dir="charts/",
        ...     split_column="Model",
        ...     x_column="Device",
        ...     y_column="Throughput",
        ...     group_column="Precision",
        ...     title_template="{value} Performance",
        ...     ylabel="Throughput (FPS)",
        ...     reference_column="Reference Throughput"
        ... )
    """
    csv_file = Path(csv_path)
    output_directory = Path(output_dir)

    # Validate input
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}") from e

    # Validate required columns
    required_cols = [split_column, x_column, y_column, group_column]
    if reference_column:
        required_cols.append(reference_column)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {', '.join(missing_cols)}")

    # Create output directory
    output_directory.mkdir(parents=True, exist_ok=True)

    # Generate one chart per split value
    saved_files = []
    for split_value in df[split_column].unique():
        # Filter data for this split
        split_df = df[df[split_column] == split_value]

        # Create temporary CSV for this split
        temp_csv = output_directory / f"temp_{split_value}.csv"
        split_df.to_csv(temp_csv, index=False)

        # Determine reference value if column provided
        ref_value = None
        if reference_column:
            ref_values = split_df[reference_column].dropna().unique()
            if len(ref_values) > 0:
                ref_value = float(ref_values[0])

        # Generate output filename
        safe_name = str(split_value).replace(" ", "_").replace("/", "_")
        output_file = output_directory / f"{safe_name}.png"

        # Generate title
        title = title_template.format(value=split_value)

        # Create chart
        try:
            plot_grouped_bar_chart(
                csv_path=temp_csv,
                output_path=output_file,
                x_column=x_column,
                y_column=y_column,
                group_column=group_column,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                reference_value=ref_value,
                reference_label=reference_label,
                figsize=figsize,
                rotation=rotation,
                color_map=color_map,
                use_secondary_axis=use_secondary_axis,
                secondary_axis_threshold=secondary_axis_threshold,
            )
            saved_files.append(output_file)
        finally:
            # Clean up temporary CSV
            if temp_csv.exists():
                temp_csv.unlink()

    logger.info(f"Created {len(saved_files)} charts in {output_directory}")
    return saved_files

# Device name mapping for CSV normalization
# Maps OpenVINO device IDs to common CSV device format used in media/AI benchmarks
DEVICE_NAME_MAP = {
    "gpu.0": "igpu",  # GPU.0 integrated -> iGPU
    "gpu.1": "dgpu",  # GPU.1 discrete -> dGPU (generic)
    "gpu.2": "dgpu",  # GPU.2 discrete -> dGPU (generic)
    "igpu": "igpu",  # Already in correct format
    "dgpu": "dgpu",  # Already in correct format (generic dGPU)
    "dgpu.0": "dgpu",  # dGPU.0 -> generic dGPU
    "dgpu.1": "dgpu",  # dGPU.1 -> generic dGPU
}


def normalize_device_and_codec(val: Any) -> Any:
    """
    Normalize device names and codec formats for CSV comparison in media/AI benchmarks.

    This function is specifically designed for tests that involve computing devices
    (CPU, GPU, NPU) and video codecs. It handles:
    - Device name mapping: OpenVINO device IDs (GPU.0, GPU.1) to benchmark format (iGPU, dGPU.0)
    - Codec format normalization: H.264 -> h264, H.265 -> h265
    - Case-insensitive string comparison
    - Whitespace trimming

    Args:
        val: Value to normalize. Can be string, number, or other type.

    Returns:
        Normalized value. Strings are lowercased and trimmed.
        Device names are mapped using DEVICE_NAME_MAP.
        Codec formats (H.264/H.265) are normalized to h264/h265.
        Non-string values are returned unchanged.

    Example:
        >>> normalize_device_and_codec("GPU.0")
        'igpu'
        >>> normalize_device_and_codec("H.264")
        'h264'
        >>> normalize_device_and_codec("  iGPU  ")
        'igpu'
        >>> normalize_device_and_codec(123)
        123
    """
    if isinstance(val, str):
        val = val.strip().lower()

        # Check device mapping first
        if val in DEVICE_NAME_MAP:
            return DEVICE_NAME_MAP[val]

        # Normalize codec formats: "h.264" -> "h264", "h.265" -> "h265"
        if val.startswith("h."):
            val = val.replace(".", "")

        return val
    return val


def filter_csv_by_columns(
    csv_path: Union[str, Path],
    filter_dict: Dict[str, Any],
    target_columns: List[str],
    normalize: bool = True,
) -> tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Filter CSV file by multiple column-value pairs with normalized matching.

    This function filters a CSV file based on multiple column conditions (AND logic)
    and returns the values of specified target columns from the first matching row.
    Supports normalized matching for case-insensitive and device name comparisons.

    Args:
        csv_path: Path to the CSV file to read.
        filter_dict: Dictionary of {column_name: filter_value} pairs.
                     All conditions must match (AND logic).
                     Example: {"Device Used": "iGPU", "Input Codec": "h264"}
        target_columns: List of column names to retrieve values from.
                        Example: ["Max Channels", "AVG CPU Util(%)"]
        normalize: If True, use normalized matching (case-insensitive, device mapping).
                   If False, use exact string matching. Default: True.

    Returns:
        tuple[dict, DataFrame] | tuple[None, None]:
            - Match found: ({"column1": value1, ...}, filtered_row_df)
              where filtered_row_df is a single-row DataFrame with all columns
            - No match or error: (None, None)

    Example:
        >>> # Filter with normalized matching (case-insensitive, device mapping)
        >>> filters = {"Device Used": "GPU.0", "Input Codec": "H.264"}
        >>> targets = ["Max Channels", "AVG CPU Util(%)"]
        >>> values, row_df = filter_csv_by_columns("results.csv", filters, targets)
        >>> # values: {"Max Channels": 69, "AVG CPU Util(%)": 245.96}
        >>> # row_df: DataFrame with all columns for the matched row
        >>> # Save filtered result: row_df.to_csv("filtered.csv", index=False)

    Notes:
        - When normalize=True, "GPU.0" matches "iGPU" in CSV, "H.264" matches "h264"
        - Returns data from first matching row only
        - Logs warnings for missing columns or no matches
    """
    # Validate CSV file exists
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return None, None

    # Read CSV with error handling
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return None, None

    # Validate filter columns exist
    for col in filter_dict.keys():
        if col not in df.columns:
            logger.error(f"Filter column '{col}' not found in CSV. Available columns: {list(df.columns)}")
            return None, None

    # Validate target columns exist
    for col in target_columns:
        if col not in df.columns:
            logger.error(f"Target column '{col}' not found in CSV. Available columns: {list(df.columns)}")
            return None, None

    if normalize:
        # Normalize entire DataFrame for matching (preserves original df)
        norm_df = df.copy()
        for col in df.columns:
            norm_df[col] = df[col].apply(normalize_device_and_codec)

        # Normalize filter keys + values
        norm_filters = {k: normalize_device_and_codec(v) for k, v in filter_dict.items()}

        # Apply ALL filters using boolean mask (AND logic)
        mask = pd.Series([True] * len(norm_df))
        for col, val in norm_filters.items():
            col_mask = norm_df[col] == val
            rows_matching = col_mask.sum()
            logger.debug(f"Filter '{col}' == '{val}': {rows_matching} rows match")

            # Debug: Show unique values in this column if no match
            if rows_matching == 0:
                unique_vals = norm_df[col].unique()[:10]  # Show first 10 unique values
                logger.warning(f"No rows match '{col}' == '{val}'. Unique values in column: {unique_vals}")

            mask = mask & col_mask

        # Use mask on original DataFrame to preserve original values
        filtered_df = df[mask]
    else:
        # Exact matching without normalization
        mask = pd.Series([True] * len(df))
        for col, val in filter_dict.items():
            col_mask = df[col] == val
            rows_matching = col_mask.sum()
            logger.debug(f"Filter '{col}' == '{val}': {rows_matching} rows match")

            if rows_matching == 0:
                unique_vals = df[col].unique()[:10]
                logger.warning(f"No rows match '{col}' == '{val}'. Unique values: {unique_vals}")

            mask = mask & col_mask

        filtered_df = df[mask]

    logger.debug(f"Total rows matching ALL filters: {len(filtered_df)}")

    if filtered_df.empty:
        logger.error(f"No matching rows found for filters: {filter_dict}")
        if normalize:
            logger.error(f"Normalized filters were: {norm_filters}")
        return None, None

    # Take the first matching row as DataFrame (preserves all columns)
    filtered_row_df = filtered_df.iloc[[0]].copy()  # Use [[0]] to keep as DataFrame
    row = filtered_df.iloc[0]  # Series for value extraction

    # Extract target column values and convert numpy types to Python native types
    results = {}
    for col in target_columns:
        value = row[col]
        # Convert numpy types to Python native types for JSON serialization
        if hasattr(value, "item"):  # numpy scalar
            results[col] = value.item()
        else:
            results[col] = value

    logger.info(f"Successfully retrieved {len(results)} values from CSV")
    return results


def download_file_from_url(url: str, output_path: Path, timeout: int = 600, max_retries: int = 3) -> bool:
    """
    Download a file from URL with progress tracking and retry mechanism.

    Implements automatic retry with exponential backoff to handle transient
    network failures (connection errors, timeouts, etc.).

    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        timeout: Download timeout in seconds
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        bool: True if download successful
    """
    from esq.utils.downloads.retry_utils import RETRYABLE_EXCEPTIONS

    delay = 2.0  # Initial retry delay in seconds
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading from: {url} (attempt {attempt}/{max_retries})")
            response = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    chunk_size = 8192
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end="", flush=True)
                    print()  # New line after progress

            logger.info(f"Downloaded successfully: {output_path} ({output_path.stat().st_size} bytes)")
            return True

        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e

            # Don't retry non-transient HTTP errors (4xx except 429 Rate Limit)
            if isinstance(e, requests.exceptions.HTTPError):
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    if 400 <= status_code < 500 and status_code != 429:
                        logger.error(f"Failed to download {url} - HTTP {status_code} (non-retryable): {e}")
                        if output_path.exists():
                            output_path.unlink()
                        return False

            # Clean up partial download
            if output_path.exists():
                output_path.unlink()

            if attempt < max_retries:
                logger.warning(f"Download failed: {e}. Retrying in {delay:.1f}s...")
                import time
                time.sleep(delay)
                delay *= 2.0  # Exponential backoff
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts: {e}")
                return False

        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False

    # Should not reach here, but for safety
    return False

def extract_zip_archive(zip_path: Path, extract_to: Path, search_dir: Optional[str] = None) -> bool:
    """
    Extract zip archive with optional subdirectory search.

    Args:
        zip_path: Path to zip file
        extract_to: Destination directory
        search_dir: Optional subdirectory name to search for and extract (e.g., "models")

    Returns:
        bool: True if extraction successful
    """
    try:
        logger.info(f"Extracting {zip_path}")
        temp_dir = extract_to / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # If search_dir specified, find and extract that directory
        if search_dir:
            target_dirs = list(temp_dir.rglob(search_dir))
            if target_dirs:
                target_dir = target_dirs[0]
                logger.info(f"Found {search_dir} directory: {target_dir}")

                # Copy contents to destination
                extract_to.mkdir(parents=True, exist_ok=True)
                for item in target_dir.iterdir():
                    dest = extract_to / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)

                logger.info(f"Extracted {search_dir} to: {extract_to}")
            else:
                logger.warning(f"No {search_dir} directory found in zip, extracting first directory")
                # Fallback: copy first directory
                subdirs = [d for d in temp_dir.iterdir() if d.is_dir()]
                if subdirs:
                    for item in subdirs[0].iterdir():
                        dest = extract_to / item.name
                        if item.is_dir():
                            shutil.copytree(item, dest, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, dest)
        else:
            # No search_dir: extract contents of first directory (skip top-level wrapper)
            extract_to.mkdir(parents=True, exist_ok=True)
            subdirs = [d for d in temp_dir.iterdir() if d.is_dir()]

            if subdirs and len(subdirs) == 1:
                # Single top-level directory: extract its contents
                top_level_dir = subdirs[0]
                logger.info(f"Extracting contents of {top_level_dir.name}")
                for item in top_level_dir.iterdir():
                    dest = extract_to / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)
            else:
                # Multiple items or no directories: extract all directly
                for item in temp_dir.iterdir():
                    dest = extract_to / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)

        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        return True

    except zipfile.BadZipFile as e:
        logger.error(f"Invalid zip file {zip_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False
