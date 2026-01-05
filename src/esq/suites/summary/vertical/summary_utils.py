# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import os
import subprocess  # nosec B404 # Subprocess needed for secure script execution and archiving
import tarfile
import warnings
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # important for headless CI
import logging

import allure
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def run_shell_script(script_path, *args):
    """
    Safely runs a shell script using subprocess.
    Args:
        script_path (str | Path): Path to the script.
        *args: Arguments to pass to the script.
    Returns:
        CompletedProcess: subprocess result object.
    Raises:
        subprocess.CalledProcessError: if command fails.
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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Script executed successfully: {script}")
        logger.debug(f"stdout: {result.stdout}")
        logger.debug(f"stderr: {result.stderr}")
        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"Script execution failed: {script}, rc={e.returncode}")
        logger.debug(f"stderr: {e.stderr}")
        return None


def create_tar(source_path, output_tar, compress=True):
    """
    Create a tar or tar.gz archive from a given folder or file path.

    Args:
        source_path (str): Path to the directory or file to archive.
        output_tar (str): Output tar file path (e.g., 'output.tar' or 'output.tar.gz').
        compress (bool): Whether to use gzip compression.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    mode = "w:gz" if compress else "w"
    with tarfile.open(output_tar, mode) as tar:
        tar.add(source_path, arcname=os.path.basename(source_path))
    logger.debug(f"Archive created successfully: {output_tar}")


def attach_csv_and_image(csv_path, image_path=None):
    """
    Convert CSV file content into a table PNG and attach both the table
    and an optional additional image to Allure.
    """
    try:
        if not os.path.exists(csv_path):
            allure.attach(f"CSV not found: {csv_path}", name="Error", attachment_type=allure.attachment_type.TEXT)
            return

        df = pd.read_csv(csv_path)

        fig, ax = plt.subplots(figsize=(len(df.columns) * 2, len(df) * 0.6 + 1))
        ax.axis("off")

        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.2)

        for (row, col), cell in table.get_celld().items():
            if row == 0:  # header row
                cell.set_facecolor("#d9d9d9")  # light gray
                cell.set_text_props(weight="bold")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        buf.seek(0)

        allure.attach(buf.read(), name="CSV Table Summary", attachment_type=allure.attachment_type.PNG)

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                allure.attach(f.read(), name="Additional Image", attachment_type=allure.attachment_type.PNG)
    except Exception as e:
        allure.attach(str(e), name="Attach Error", attachment_type=allure.attachment_type.TEXT)


def get_values_from_csv(csv_path, filter_column, filter_value, target_columns):
    """
    Retrieve value(s) from specific row(s) in a CSV file where filter_column == filter_value.

    Args:
        csv_path (str): Path to the CSV file.
        filter_column (str): Column name to filter rows on.
        filter_value (str|int|float): Value to match in filter_column.
        target_columns (str|list): Column(s) whose value(s) to retrieve from the matching row(s).

    Returns:
        list[dict] | dict | any | None:
            - Matching values as dict(s) or list
            - None if warnings encountered or no data

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If file is empty or unreadable.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Resultant CSV file not found: {csv_path}")

    if os.path.getsize(csv_path) == 0:
        raise ValueError(f"Resultant CSV file is empty: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to parse given CSV file: {e}")

    if filter_column not in df.columns:
        warnings.warn(f"Filter column '{filter_column}' not found in CSV: {csv_path}")
        return None

    if isinstance(target_columns, str):
        target_columns = [target_columns]

    missing_cols = [col for col in target_columns if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Target column(s) missing in resultant CSV: {', '.join(missing_cols)}")
        return None

    filtered_df = df[df[filter_column] == filter_value]

    if filtered_df.empty:
        warnings.warn(f"No rows found where '{filter_column}' == '{filter_value}'")
        return None

    try:
        results = filtered_df[target_columns].to_dict(orient="records")
    except Exception as e:
        warnings.warn(f"Failed to extract data: {e}")
        return None

    # Simplify output
    if len(results) == 1:
        if len(target_columns) == 1:
            return results[0][target_columns[0]]
        return results[0]
    elif len(results) > 1:
        return results

    return None
