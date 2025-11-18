# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for model setup and management.

This module provides shared functions and constants used across different
model source handlers.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def download_model(model_id_or_path: str, revision: str = "main", ignore_patterns: Optional[list] = None) -> str:
    """
    Download a model from Hugging Face Hub using snapshot_download.

    Args:
        model_id_or_path: Model ID or local path
        revision: Model revision to download
        ignore_patterns: Patterns to ignore during download

    Returns:
        str: Local path to the downloaded model

    Raises:
        RuntimeError: If download fails or is interrupted
    """
    logger.info(f"Downloading model: {model_id_or_path}")
    try:
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(repo_id=model_id_or_path, revision=revision, ignore_patterns=ignore_patterns)
        logger.info(f"Model downloaded to: {local_path}")
        return local_path
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user")
        raise RuntimeError("Download interrupted") from None
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise RuntimeError(f"Model download failed: {e}") from e


def ensure_model_directory(models_dir: str, model_id: str) -> Path:
    """
    Ensure model directory exists and return Path object.

    Args:
        models_dir: Base models directory
        model_id: Model identifier

    Returns:
        Path: Path object for model directory
    """
    model_path = Path(models_dir) / model_id
    model_path.mkdir(parents=True, exist_ok=True)
    return model_path


def check_model_exists(model_path: Path, model_id: str) -> bool:
    """
    Check if model already exists at the given path.

    Args:
        model_path: Path to check
        model_id: Model identifier for logging

    Returns:
        bool: True if model exists
    """
    if model_path.exists():
        logger.info(f" ✓ Model {model_id} already exists: {model_path}")
        return True
    return False


def construct_export_path_components(export_args: Optional[dict] = None) -> list:
    """
    Construct path components based on export arguments that differ from defaults.

    This function generates subdirectory components for model export paths based on
    export arguments that differ from their default values. The components are used
    to organize exported models by their configuration.

    Components are sorted alphabetically to ensure consistent path construction.

    Args:
        export_args: Optional dictionary of export arguments. If None or empty,
                    returns empty list (uses default path without subdirectory).

    Returns:
        list: Path components sorted alphabetically and joined with underscore separator.
              Empty list if no args differ from defaults.
    """
    path_components = []

    if not export_args:
        return path_components

    # Collect all components that differ from defaults
    # Components will be sorted alphabetically for consistent paths

    batch_size = export_args.get("batch", 1)
    if batch_size != 1:
        path_components.append(f"batch{batch_size}")

    if export_args.get("dynamic", False):
        path_components.append("dynamic")

    if export_args.get("half", False):
        path_components.append("half")

    if export_args.get("int8", False):
        path_components.append("int8")

    if export_args.get("nms", False):
        path_components.append("nms")

    # Sort alphabetically for consistent path construction
    path_components.sort()

    return path_components


def quantize_model_with_nncf(model, calibration_dataset, subset_size: int = 300):
    """
    Quantize OpenVINO model to INT8 using NNCF.

    Args:
        model: OpenVINO model to quantize
        calibration_dataset: NNCF calibration dataset
        subset_size: Number of samples for calibration

    Returns:
        Quantized OpenVINO model
    """
    try:
        import nncf

        logger.info(f"Quantizing model to INT8 using NNCF with {subset_size} calibration samples")
        quantized_model = nncf.quantize(
            model=model,
            calibration_dataset=calibration_dataset,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            subset_size=subset_size,
            fast_bias_correction=True,
        )
        logger.info("Model quantization completed successfully")
        return quantized_model

    except ImportError:
        logger.error("NNCF library not installed. Install with: pip install nncf")
        raise
    except Exception as e:
        logger.error(f"Failed to quantize model with NNCF: {e}")
        raise


def save_openvino_model(model, output_path: Path, compress_fp16: bool = False) -> bool:
    """
    Save OpenVINO model to specified path.

    Args:
        model: OpenVINO model to save
        output_path: Path to save the model (.xml file)
        compress_fp16: Whether to compress FP32 weights to FP16

    Returns:
        bool: True if save was successful
    """
    try:
        import openvino as ov

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ov.save_model(model, str(output_path), compress_to_fp16=compress_fp16)
        logger.info(f"Model saved successfully to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save OpenVINO model: {e}")
        return False
