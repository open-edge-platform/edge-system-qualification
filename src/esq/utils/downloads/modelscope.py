# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
ModelScope-specific download utilities.

This module provides download functions for models and datasets from ModelScope
using the 'modelscope' CLI tool with proper timeout enforcement.

Dependencies:
    - modelscope (provides 'modelscope' CLI)
    - datasets (required by ModelScope CLI for dataset operations)
"""

import logging
import os
import shutil
import sys
import time
from typing import Optional

from sysagent.utils.core import run_command

logger = logging.getLogger(__name__)


def get_modelscope_cli_path() -> str:
    """Get the modelscope executable path in the same environment as Python."""
    python_path = sys.executable
    python_dir = os.path.dirname(python_path)

    # First, try to find modelscope in the same directory as python
    cli_path = os.path.join(python_dir, "modelscope")
    if os.path.exists(cli_path):
        logger.debug(f"Found modelscope at: {cli_path}")
        return cli_path

    # Second, try to find modelscope in PATH
    cli_in_path = shutil.which("modelscope")
    if cli_in_path:
        logger.debug(f"Found modelscope in PATH at: {cli_in_path}")
        return cli_in_path

    # If all else fails, raise an error
    raise ValueError("modelscope CLI not found in the current environment.")


def download_model(
    model_id: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    revision: Optional[str] = None,
    timeout: float = 1800,
    max_workers: int = 4,
) -> str:
    """
    Download model from ModelScope using CLI command.

    Uses CLI command via subprocess to ensure proper timeout enforcement.

    Args:
        model_id: Model identifier
        cache_dir: Cache directory for downloaded files
        local_dir: Local directory to save files (overrides cache_dir)
        revision: Git revision (branch, tag, or commit hash)
        timeout: Download timeout in seconds
        max_workers: Maximum number of download workers (default: 4)

    Returns:
        Path to the downloaded model directory

    Raises:
        TimeoutError: If download times out
        RuntimeError: If download fails
    """
    logger.debug(f"Downloading model '{model_id}' from ModelScope...")

    download_start = time.time()

    # Get the full path to modelscope CLI
    cli_path = get_modelscope_cli_path()

    # Build the command
    cmd = [cli_path, "download", "--model", model_id]

    # Add optional arguments
    if revision:
        cmd.extend(["--revision", revision])
    if cache_dir:
        cmd.extend(["--cache_dir", cache_dir])
    if local_dir:
        cmd.extend(["--local_dir", local_dir])

    # Add max-workers option for parallel downloads
    cmd.extend(["--max-workers", str(max_workers)])

    logger.debug(f"Running command: {' '.join(cmd)}")

    # Run the command with timeout
    result = run_command(
        command=cmd,
        timeout=timeout,
        stream_output=True,
    )

    # Check if command timed out
    if result.timed_out:
        raise TimeoutError(f"Model download timed out after {timeout}s")

    # Check return code
    if result.returncode != 0:
        error_msg = f"ModelScope download failed with return code {result.returncode}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Determine model path using known cache structure
    if local_dir:
        # Model downloaded to specified local directory
        model_path = local_dir
        logger.debug(f"Model downloaded to local_dir: {model_path}")
    else:
        # Model downloaded to cache - construct path based on ModelScope cache structure
        # ModelScope cache structure: {cache_dir}/hub/models/{model_id}/
        cache_location = cache_dir if cache_dir else os.path.expanduser("~/.cache/modelscope/hub/models")
        # ModelScope uses model_id as-is in the path
        model_path = os.path.join(cache_location, model_id)
        logger.debug(f"Model located in cache: {model_path}")

    # Verify the path exists
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model path does not exist: {model_path}")

    download_duration = time.time() - download_start
    logger.debug(f"Model downloaded in {download_duration:.2f}s: {model_path}")

    return model_path


def download_dataset(
    dataset_id: str,
    local_dir: str,
    filename: str = None,
    timeout: float = 300,
    max_workers: int = 4,
) -> str:
    """
    Download dataset from ModelScope using CLI command.

    Args:
        dataset_id: Dataset ID on ModelScope
        local_dir: Local directory to save dataset
        filename: Specific file to download (optional - downloads all if not specified)
        timeout: Download timeout in seconds
        max_workers: Maximum number of download workers (default: 4)

    Returns:
        Path to downloaded dataset directory

    Raises:
        TimeoutError: If download times out
        RuntimeError: If download fails
    """
    logger.debug(f"Downloading dataset '{dataset_id}' from ModelScope...")

    # Get the full path to modelscope CLI
    cli_path = get_modelscope_cli_path()

    # Build command: modelscope download --dataset dataset_id --local_dir local_dir --max-workers N [filename]
    cmd = [cli_path, "download", "--dataset", dataset_id, "--local_dir", local_dir]

    # Add max-workers option for parallel downloads
    cmd.extend(["--max-workers", str(max_workers)])

    # Add specific filename if provided (as positional argument)
    if filename:
        cmd.append(filename)

    result = run_command(
        command=cmd,
        timeout=timeout,
        stream_output=True,
    )

    if result.returncode != 0:
        # Check if the failure was due to timeout
        if result.timed_out:
            error_msg = f"ModelScope dataset download timed out after {timeout}s"
            logger.error(error_msg)
            raise TimeoutError(error_msg)
        else:
            error_msg = f"ModelScope dataset download failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    logger.debug(f"Dataset downloaded to: {local_dir}")
    return local_dir
