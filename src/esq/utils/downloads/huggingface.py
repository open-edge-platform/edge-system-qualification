# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
HuggingFace-specific download utilities.

This module provides download functions for models and datasets from HuggingFace
using the 'hf' CLI tool with proper timeout enforcement.

Dependencies:
    - huggingface-hub (provides 'hf' CLI)
"""

import logging
import os
import shutil
import sys
import time
from typing import Optional

from sysagent.utils.core import run_command

logger = logging.getLogger(__name__)


def get_huggingface_cli_path() -> str:
    """Get the hf CLI executable path in the same environment as Python."""
    python_path = sys.executable
    python_dir = os.path.dirname(python_path)

    # First, try to find hf in the same directory as python
    cli_path = os.path.join(python_dir, "hf")
    if os.path.exists(cli_path):
        logger.debug(f"Found hf CLI at: {cli_path}")
        return cli_path

    # Second, try to find hf in PATH
    cli_in_path = shutil.which("hf")
    if cli_in_path:
        logger.debug(f"Found hf CLI in PATH at: {cli_in_path}")
        return cli_in_path

    # If all else fails, raise an error
    raise ValueError("hf CLI not found in the current environment.")


def download_model(
    model_id: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    revision: Optional[str] = None,
    timeout: float = 1800,
) -> str:
    """
    Download model from HuggingFace using CLI command.

    Uses CLI command via subprocess to ensure proper timeout enforcement.

    Args:
        model_id: Model identifier (e.g., "microsoft/Phi-4-mini-instruct")
        cache_dir: Cache directory for downloaded files
        local_dir: Local directory to save files (overrides cache_dir)
        revision: Git revision (branch, tag, or commit hash)
        timeout: Download timeout in seconds

    Returns:
        Path to the downloaded model directory

    Raises:
        TimeoutError: If download times out
        RuntimeError: If download fails
    """
    logger.debug(f"Downloading model '{model_id}' from HuggingFace...")

    download_start = time.time()

    # Get the full path to huggingface-cli
    cli_path = get_huggingface_cli_path()

    # Build the command
    cmd = [cli_path, "download", model_id]

    # Add optional arguments
    if revision:
        cmd.extend(["--revision", revision])
    if cache_dir:
        cmd.extend(["--cache-dir", cache_dir])
    if local_dir:
        cmd.extend(["--local-dir", local_dir])

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
        error_msg = f"HuggingFace download failed with return code {result.returncode}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Determine model path using known cache structure
    if local_dir:
        # Model downloaded to specified local directory
        model_path = local_dir
        logger.debug(f"Model downloaded to local_dir: {model_path}")
    else:
        # Model downloaded to cache - construct path based on HuggingFace cache structure
        # HuggingFace cache structure: {cache_dir}/models--{org}--{model}/snapshots/{revision}/
        cache_location = cache_dir if cache_dir else os.path.expanduser("~/.cache/huggingface/hub")

        # Convert model_id to cache directory name (replace / with --)
        model_cache_name = model_id.replace("/", "--")
        model_cache_path = os.path.join(cache_location, f"models--{model_cache_name}")

        # Find the snapshot directory (could be revision-specific or latest)
        snapshots_dir = os.path.join(model_cache_path, "snapshots")
        if os.path.exists(snapshots_dir):
            # Get the most recent snapshot (or specific revision if provided)
            snapshots = os.listdir(snapshots_dir)
            if revision:
                # Try to find matching revision
                matching = [s for s in snapshots if s.startswith(revision)]
                if matching:
                    model_path = os.path.join(snapshots_dir, matching[0])
                else:
                    raise RuntimeError(f"Revision {revision} not found for model {model_id}")
            elif snapshots:
                # Use the first (typically only one) snapshot
                model_path = os.path.join(snapshots_dir, snapshots[0])
            else:
                raise RuntimeError(f"No snapshots found for model {model_id}")
        else:
            raise RuntimeError(f"Model cache directory not found: {snapshots_dir}")

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
) -> str:
    """
    Download dataset from HuggingFace using CLI command.

    Args:
        dataset_id: Dataset ID (e.g., "anon8231489123/ShareGPT_Vicuna_unfiltered")
        local_dir: Local directory to save dataset
        filename: Specific file to download (optional - downloads all if not specified)
        timeout: Download timeout in seconds

    Returns:
        Path to downloaded dataset directory

    Raises:
        TimeoutError: If download times out
        RuntimeError: If download fails
    """
    logger.debug(f"Downloading dataset '{dataset_id}' from HuggingFace...")

    # Get the full path to hf CLI
    cli_path = get_huggingface_cli_path()

    # Build command: hf download dataset_id [filename] --repo-type dataset --local-dir local_dir
    cmd = [cli_path, "download", dataset_id]

    # Add specific filename if provided (downloads only that file)
    if filename:
        cmd.append(filename)

    cmd.extend(["--repo-type", "dataset", "--local-dir", local_dir])

    result = run_command(
        command=cmd,
        timeout=timeout,
        stream_output=True,
    )

    if result.returncode != 0:
        if result.timed_out:
            error_msg = f"HuggingFace dataset download timed out after {timeout}s"
            raise TimeoutError(error_msg)
        else:
            error_msg = f"HuggingFace dataset download failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr}"
            raise RuntimeError(error_msg)

    logger.debug(f"Dataset downloaded to: {local_dir}")
    return local_dir
