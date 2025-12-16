# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Unified Model Downloader for OpenVINO Benchmark Tests.

This module provides a unified interface for downloading all models required
for the OpenVINO benchmark test suite, eliminating the need for containerized
model fetching.

IMPORTANT: This implementation does NOT use openvino-dev or OMZ tools
(omz_downloader, omz_converter). It uses:
- Direct HTTP downloads for OpenVINO Model Zoo models
- Ultralytics Python API for YOLO models
- Hugging Face Transformers API for CLIP models

This avoids compatibility issues with OMZ tools and reduces dependencies.

Functions:
    - download_model_by_id: Download any model by ID
    - download_all_benchmark_models: Download all models for benchmark suite
    - get_model_path: Get path to a downloaded model
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Import model-specific utilities
try:
    from .clip_model_utils import CLIP_MODELS, download_and_export_clip_model
    from .openvino_model_utils import AVAILABLE_MODELS as OMZ_MODELS
    from .openvino_model_utils import download_openvino_model
    from .yolo_model_utils import YOLO_MODELS, download_yolo_model, export_yolo_model
except ImportError as e:
    logger.warning(f"Failed to import model utilities: {e}")
    OMZ_MODELS = {}
    YOLO_MODELS = {}
    CLIP_MODELS = {}


# Model registry mapping profile model names to download functions
MODEL_REGISTRY = {
    # OpenVINO Model Zoo models (pre-quantized INT8 available)
    "resnet-50-tf": "omz",
    "efficientnet-b0": "omz",
    "mobilenet-v2-pytorch": "omz",
    "ssdlite_mobilenet_v2": "omz",
    # YOLO models (requires Ultralytics export)
    "yolo-v5s": "yolo",
    "yolov5s": "yolo",
    "yolo-v8s": "yolo",
    "yolov8s": "yolo",
    # CLIP models (requires Hugging Face + conversion)
    "clip-vit-base-patch16": "clip",
}


def download_model_by_id(
    model_id: str,
    precision: str = "INT8",
    models_dir: Optional[str] = None,
    force_download: bool = False,
) -> Optional[Path]:
    """
    Download any model by its ID using the appropriate downloader.

    Args:
        model_id: Model identifier (e.g., 'resnet-50-tf', 'yolo-v5s', 'clip-vit-base-patch16')
        precision: Model precision ('FP16', 'FP32', 'INT8')
        models_dir: Directory to save models (default: ~/share/public/models)
        force_download: Force re-download even if model exists

    Returns:
        Path to the downloaded/exported model XML file, or None if failed

    Example:
        >>> model_path = download_model_by_id("resnet-50-tf", "INT8")
        >>> print(model_path)
        PosixPath('/home/user/share/public/models/resnet-50-tf/INT8/resnet-50-tf.xml')
    """
    if model_id not in MODEL_REGISTRY:
        logger.error(f"Unknown model ID: {model_id}")
        logger.info(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
        return None

    model_type = MODEL_REGISTRY[model_id]

    # Use CORE_DATA_DIR structure: esq_data/data/vertical/metro/models
    if models_dir is None:
        core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
        core_data_dir = "".join(c for c in core_data_dir_tainted)
        models_dir = Path(core_data_dir) / "data" / "vertical" / "metro" / "models"
    else:
        models_dir = Path(models_dir)

    logger.info(f"Downloading model: {model_id} ({precision})")

    try:
        if model_type == "omz":
            # OpenVINO Model Zoo models - pre-quantized available
            return download_openvino_model(
                model_id=model_id,
                precision=precision,
                models_dir=str(models_dir),
                force_download=force_download,
            )

        elif model_type == "yolo":
            # YOLO models - download weights and export to OpenVINO IR
            # Normalize model ID (remove dash if present)
            normalized_id = model_id.replace("-", "")

            # Check if already exported
            model_path = models_dir / model_id / precision.upper() / f"{model_id}.xml"
            if not force_download and model_path.exists():
                logger.info(f"✓ Model {model_id} ({precision}) already exists: {model_path}")
                return model_path

            # Download weights
            weights_path = download_yolo_model(normalized_id, str(models_dir))
            if not weights_path:
                logger.error(f"Failed to download YOLO weights for {model_id}")
                return None

            # Export to OpenVINO IR
            export_args = {}
            if precision.upper() == "INT8":
                export_args["int8"] = True
                export_args["data"] = "coco8.yaml"

            exported_path = export_yolo_model(
                model_id=normalized_id,
                models_dir=str(models_dir),
                model_precision=precision.lower(),
                weights_path=weights_path,
                export_args=export_args if export_args else None,
            )

            return exported_path

        elif model_type == "clip":
            # CLIP models - download from Hugging Face and export
            return download_and_export_clip_model(
                model_id=model_id,
                models_dir=str(models_dir),
                model_precision=precision.lower(),
            )

        else:
            logger.error(f"Unknown model type: {model_type}")
            return None

    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}", exc_info=True)
        return None


def download_all_benchmark_models(
    model_list: Optional[List[str]] = None,
    precision: str = "INT8",
    models_dir: Optional[str] = None,
) -> Dict[str, Optional[Path]]:
    """
    Download all models required for benchmark tests.

    Args:
        model_list: List of model IDs to download (default: all models in registry)
        precision: Model precision (default: INT8)
        models_dir: Directory to save models

    Returns:
        Dictionary mapping model_id -> model_path (or None if failed)

    Example:
        >>> results = download_all_benchmark_models()
        >>> print(f"Downloaded {sum(1 for p in results.values() if p)} models")
        Downloaded 7 models
    """
    if model_list is None:
        model_list = list(MODEL_REGISTRY.keys())

    logger.info(f"Downloading {len(model_list)} models for benchmark tests...")

    results = {}
    success_count = 0
    fail_count = 0

    for model_id in model_list:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing model: {model_id}")
        logger.info(f"{'=' * 60}")

        model_path = download_model_by_id(model_id, precision, models_dir)
        results[model_id] = model_path

        if model_path:
            success_count += 1
            logger.info(f"✓ Success: {model_id}")
        else:
            fail_count += 1
            logger.error(f"✗ Failed: {model_id}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Download Summary: {success_count} succeeded, {fail_count} failed")
    logger.info(f"{'=' * 60}")

    return results


def get_model_path(
    model_id: str,
    precision: str = "INT8",
    models_dir: Optional[str] = None,
) -> Optional[Path]:
    """
    Get path to a downloaded model without downloading.

    Args:
        model_id: Model identifier
        precision: Model precision
        models_dir: Models directory

    Returns:
        Path to model XML file if it exists, None otherwise

    Example:
        >>> model_path = get_model_path("resnet-50-tf", "INT8")
        >>> if model_path:
        ...     print(f"Model exists: {model_path}")
        ... else:
        ...     print("Model not found")
    """
    # Use CORE_DATA_DIR structure: esq_data/data/vertical/metro/models
    if models_dir is None:
        core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
        core_data_dir = "".join(c for c in core_data_dir_tainted)
        models_dir = Path(core_data_dir) / "data" / "vertical" / "metro" / "models"
    else:
        models_dir = Path(models_dir)

    model_path = models_dir / model_id / precision.upper() / f"{model_id}.xml"

    if model_path.exists():
        return model_path
    return None


def list_available_models() -> Dict[str, str]:
    """
    List all available models and their types.

    Returns:
        Dictionary mapping model_id -> model_type

    Example:
        >>> models = list_available_models()
        >>> for model_id, model_type in models.items():
        ...     print(f"{model_id}: {model_type}")
    """
    return MODEL_REGISTRY.copy()


def get_models_for_profile(profile_name: str) -> List[str]:
    """
    Get list of models required for a specific test profile.

    Args:
        profile_name: Name of the test profile

    Returns:
        List of model IDs

    Example:
        >>> models = get_models_for_profile("ai_vision_ov")
        >>> print(models)
        ['resnet-50-tf', 'efficientnet-b0', ...]
    """
    # Map profile names to model lists
    profile_models = {
        "ai_vision_ov": [
            "resnet-50-tf",
            "efficientnet-b0",
            "ssdlite_mobilenet_v2",
            "mobilenet-v2-pytorch",
            "yolo-v5s",
            "yolo-v8s",
            "clip-vit-base-patch16",
        ],
    }

    return profile_models.get(profile_name, [])
