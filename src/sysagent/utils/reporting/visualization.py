# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Visualization utilities for test results.

This module provides standardized functions for creating visualizations
of test results, including tables and charts.
"""

import io
import logging
import tempfile

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
# Suppress matplotlib debug logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def create_results_table(
    data: Dict[str, Any],
    title: str = "Results Summary",
    columns: Optional[List[str]] = None,
    row_transform: Optional[callable] = None,
) -> Tuple[bytes, Any]:
    """
    Create a standardized, generic table visualization of test results.

    Args:
        data: Dictionary containing result metrics or parameters.
        title: Title for the table.
        columns: List of column names. If None, will infer from data.
        row_transform: Optional function to transform each (key, value) into a row list.

    Returns:
        Tuple containing:
            - The PNG image as bytes
            - The matplotlib figure object (caller should close this)
    """
    logger.debug(f"Creating results table with {len(data)} rows, columns={columns}")

    # Infer columns if not provided
    if columns is None:
        # Try to infer from first row
        if data:
            first_val = next(iter(data.values()))
            if isinstance(first_val, dict):
                columns = ["Key"] + list(first_val.keys())
            else:
                columns = ["Key", "Value"]
        else:
            columns = ["Key", "Value"]

    # Create table data
    rows = []
    for key, value in data.items():
        if row_transform:
            row = row_transform(key, value)
        elif isinstance(value, dict):
            # Handle special case for metrics with Value/Unit columns
            if columns == ["Metric", "Value", "Unit"]:
                metric_value = value.get("value", "")
                metric_unit = value.get("unit", "")
                row = [key, str(metric_value), str(metric_unit)]
            else:
                # Generic handling for other dictionary structures
                row = [key] + [
                    str(value.get(col.lower(), value.get(col, "")))
                    for col in columns[1:]
                ]
        else:
            row = [key, str(value)]
        rows.append(row)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(rows) * 0.4)))
    ax.axis("tight")
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        loc="center",
        colWidths=[0.3] + [0.7 / (len(columns) - 1)] * (len(columns) - 1),
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    # Add title
    plt.title(title, fontsize=14, fontweight="bold", pad=20)

    # Convert to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
    img_buffer.seek(0)

    return img_buffer.getvalue(), fig


def create_results_bar_chart(
    data: Dict[str, Union[int, float]],
    title: str = "Results Bar Chart",
    ylabel: str = "Value",
    color: str = "#4CAF50",
) -> Tuple[bytes, Any]:
    """
    Create a bar chart visualization of numeric results.

    Args:
        data: Dictionary mapping labels to numeric values.
        title: Title for the chart.
        ylabel: Label for the y-axis.
        color: Color for the bars.

    Returns:
        Tuple containing:
            - The PNG image as bytes
            - The matplotlib figure object (caller should close this)
    """
    logger.debug(f"Creating bar chart with {len(data)} bars")

    # Extract data
    labels = list(data.keys())
    values = list(data.values())

    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 6))

    # Create bars
    bars = ax.bar(labels, values, color=color)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.2f}" if isinstance(value, float) else str(value),
            ha="center",
            va="bottom",
        )

    # Customize chart
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)

    # Rotate x-axis labels if needed
    if len(max(labels, key=len)) > 10:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Convert to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
    img_buffer.seek(0)

    return img_buffer.getvalue(), fig


def create_time_series_chart(
    data: Dict[str, List[Tuple[float, float]]],
    title: str = "Time Series Chart",
    xlabel: str = "Time",
    ylabel: str = "Value",
) -> Tuple[bytes, Any]:
    """
    Create a time series line chart.

    Args:
        data: Dictionary mapping series names to lists of (timestamp, value) tuples.
        title: Title for the chart.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.

    Returns:
        Tuple containing:
            - The PNG image as bytes
            - The matplotlib figure object (caller should close this)
    """
    logger.debug(f"Creating time series chart with {len(data)} series")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each series
    colors = plt.cm.Set1(np.linspace(0, 1, len(data)))
    for (series_name, series_data), color in zip(data.items(), colors):
        if series_data:
            times, values = zip(*series_data)
            ax.plot(
                times, values, label=series_name, color=color, marker="o", markersize=3
            )

    # Customize chart
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if len(data) > 1:
        ax.legend()

    plt.tight_layout()

    # Convert to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
    img_buffer.seek(0)

    return img_buffer.getvalue(), fig


def save_chart_as_temp_file(image_bytes: bytes, prefix: str = "chart") -> str:
    """
    Save chart image bytes to a temporary file.

    Args:
        image_bytes: The PNG image data as bytes.
        prefix: Prefix for the temporary file name.

    Returns:
        Path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".png", prefix=f"{prefix}_"
    ) as tmp_file:
        tmp_file.write(image_bytes)
        return tmp_file.name


def create_comparison_chart(
    data: Dict[str, Dict[str, Union[int, float]]],
    title: str = "Comparison Chart",
    ylabel: str = "Value",
) -> Tuple[bytes, Any]:
    """
    Create a grouped bar chart for comparing multiple metrics across categories.

    Args:
        data: Nested dictionary where outer keys are categories & inner keys are metrics
        title: Title for the chart.
        ylabel: Label for the y-axis.

    Returns:
        Tuple containing:
            - The PNG image as bytes
            - The matplotlib figure object (caller should close this)
    """
    logger.debug(f"Creating comparison chart with {len(data)} categories")

    if not data:
        # Return empty chart
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No data to display",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
        img_buffer.seek(0)
        return img_buffer.getvalue(), fig

    # Extract categories and metrics
    categories = list(data.keys())
    all_metrics = set()
    for category_data in data.values():
        all_metrics.update(category_data.keys())
    metrics = sorted(all_metrics)

    # Prepare data for plotting
    x = np.arange(len(categories))
    width = 0.8 / len(metrics) if metrics else 0.8

    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.2), 6))

    # Create bars for each metric
    colors = plt.cm.Set1(np.linspace(0, 1, len(metrics)))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [data[cat].get(metric, 0) for cat in categories]
        bars = ax.bar(x + i * width, values, width, label=metric, color=color)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value != 0:  # Only show non-zero values
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.2f}" if isinstance(value, float) else str(value),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Customize chart
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Categories")
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(categories)

    if len(metrics) > 1:
        ax.legend()

    # Rotate x-axis labels if needed
    if any(len(cat) > 10 for cat in categories):
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Convert to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
    img_buffer.seek(0)

    return img_buffer.getvalue(), fig
