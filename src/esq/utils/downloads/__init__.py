# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Unified download interface for models and datasets.

This module provides a unified interface for downloading models and datasets
from multiple sources (HuggingFace, ModelScope, etc.), with configurable source
selection via PREFER_* environment variable flags.

Also provides retry utilities for robust download operations with automatic
retry and exponential backoff for transient network failures.

Usage:
    from esq.utils.downloads import download_model, download_dataset_file, with_retry

    # Download model (defaults to HuggingFace unless PREFER_* flag set)
    model_path = download_model("microsoft/Phi-4-mini-instruct")

    # Download dataset file
    download_dataset_file("anon8231489123/ShareGPT_Vicuna_unfiltered",
                         "file.json", "/path/to/target.json")

    # Use retry decorator for custom download functions
    @with_retry(max_attempts=3, initial_delay=2.0)
    def my_download_function():
        # Download logic here
        pass

    # Configure preferred sources:
    # export PREFER_MODELSCOPE=1              # Global preference
    # export PREFER_HUGGINGFACE_MODEL=1       # Model-specific preference
"""

import logging
import os
import time
from typing import Optional

# Export retry utilities for use in other modules
from .retry_utils import RETRYABLE_EXCEPTIONS, retry_download, with_retry

logger = logging.getLogger(__name__)


def download_model(
    model_id: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    revision: Optional[str] = None,
    timeout: float = 1800,
    source: Optional[str] = None,
) -> str:
    """
    Download a model from HuggingFace or ModelScope using CLI commands.

    Source selection priority:
    1. Explicit source parameter
    2. PREFER_<SOURCE>_MODEL environment variable flags (e.g., PREFER_HUGGINGFACE_MODEL=1)
    3. PREFER_<SOURCE> environment variable flags (e.g., PREFER_MODELSCOPE=1)
    4. Default: huggingface

    Uses CLI commands via subprocess to ensure proper timeout enforcement.
    This is more reliable than Python library calls which cannot be properly interrupted.

    Args:
        model_id: Model identifier (e.g., "microsoft/Phi-4-mini-instruct")
        cache_dir: Cache directory for downloaded files
        local_dir: Local directory to save files (overrides cache_dir)
        revision: Git revision (branch, tag, or commit hash)
        timeout: Download timeout in seconds
        source: Specific source ("huggingface" or "modelscope", None for auto-detect)

    Returns:
        Path to the downloaded model directory

    Raises:
        RuntimeError: If download fails from all available sources
    """
    from sysagent.utils.infrastructure.network import get_preferred_download_source

    # Determine which source to use
    if source is None:
        source = get_preferred_download_source(
            source_type="model",
            default="huggingface",
            valid_sources=["huggingface", "modelscope"],
        )

    use_modelscope = source == "modelscope"

    download_start = time.time()

    try:
        if use_modelscope:
            from .modelscope import download_model as ms_download_model

            logger.info(f"Downloading model '{model_id}' from ModelScope with timeout of {timeout} seconds...")
            model_path = ms_download_model(
                model_id=model_id,
                cache_dir=cache_dir,
                local_dir=local_dir,
                revision=revision,
                timeout=timeout,
            )
        else:
            from .huggingface import download_model as hf_download_model

            logger.info(f"Downloading model '{model_id}' from HuggingFace with timeout of {timeout} seconds...")
            model_path = hf_download_model(
                model_id=model_id,
                cache_dir=cache_dir,
                local_dir=local_dir,
                revision=revision,
                timeout=timeout,
            )

        download_duration = time.time() - download_start
        logger.info(f"Model download completed in {download_duration:.2f}s: {model_path}")
        return model_path

    except Exception as e:
        download_duration = time.time() - download_start

        # Check if it was a timeout
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            error_msg = f"Model download timeout after {download_duration:.2f} seconds (limit: {timeout}s)"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # No fallback - fail with clear error message
        source_name = "ModelScope" if use_modelscope else "HuggingFace"
        error_msg = f"Failed to download model '{model_id}' from {source_name}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def download_dataset_file(
    dataset_id: str,
    filename: str,
    target_path: str,
    timeout: float = 300,
    source: Optional[str] = None,
) -> None:
    """
    Download a dataset file using HuggingFace or ModelScope CLI commands.

    Automatically selects the appropriate source based on network restrictions.

    Args:
        dataset_id: Dataset identifier (e.g., "anon8231489123/ShareGPT_Vicuna_unfiltered")
        filename: File within the dataset to download
        target_path: Local path to save the file
        timeout: Download timeout in seconds
        source: Specific source ("huggingface" or "modelscope", None for auto-detect)

    Raises:
        TimeoutError: If download times out
        RuntimeError: If download fails for other reasons
        FileNotFoundError: If file not found after download
    """
    from sysagent.utils.infrastructure.network import get_preferred_download_source

    # Determine which source to use
    if source is None:
        source = get_preferred_download_source(
            source_type="dataset",
            default="huggingface",
            valid_sources=["huggingface", "modelscope"],
        )

    use_modelscope = source == "modelscope"

    try:
        logger.info(f"Downloading dataset file '{filename}' from dataset '{dataset_id}'")

        target_dir = os.path.dirname(target_path)
        os.makedirs(target_dir, exist_ok=True)

        if use_modelscope:
            from .modelscope import download_dataset as ms_download_dataset

            ms_download_dataset(
                dataset_id=dataset_id,
                local_dir=target_dir,
                filename=filename,
                timeout=timeout,
            )
        else:
            from .huggingface import download_dataset as hf_download_dataset

            hf_download_dataset(
                dataset_id=dataset_id,
                local_dir=target_dir,
                filename=filename,
                timeout=timeout,
            )

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"File '{filename}' not found at expected path: {target_path}")

        logger.info(f"Dataset file downloaded successfully: {target_path}")

    except Exception as e:
        # Clean up partial file if it exists
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except Exception:
                pass

        # Re-raise the exception to preserve timeout and other error details
        logger.error(f"Failed to download dataset file: {e}")
        raise


# Expose main functions at package level
__all__ = ["download_model", "download_dataset_file"]
