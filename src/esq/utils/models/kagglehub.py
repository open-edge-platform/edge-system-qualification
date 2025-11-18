# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
KaggleHub model utilities for models hosted on Kaggle.

This module handles downloading models from Kaggle Hub, converting to OpenVINO format,
and applying INT8 quantization using NNCF.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def download_kaggle_model(
    model_id: str,
    kaggle_handle: str,
    models_dir: Optional[str] = None,
) -> Optional[Path]:
    """
    Download model from Kaggle Hub.

    Args:
        model_id: Kaggle model identifier for local reference
        kaggle_handle: Kaggle model handle (e.g., 'google/resnet-v1/tensorFlow2/50-classification')
        models_dir: Directory to track model downloads

    Returns:
        Path to the downloaded model directory or None if failed
    """
    if not kaggle_handle:
        logger.error(f"Missing 'kaggle_handle' parameter for model '{model_id}'")
        return None

    logger.info(f"Downloading model '{model_id}' from Kaggle Hub: {kaggle_handle}")

    try:
        import kagglehub

        # Download model from Kaggle Hub (kagglehub manages its own cache)
        model_path = kagglehub.model_download(kaggle_handle)
        logger.info(f"Model downloaded to: {model_path}")

        # Create a marker file in models_dir to track that we've downloaded this model
        if models_dir:
            marker_dir = Path(models_dir) / model_id
            marker_dir.mkdir(parents=True, exist_ok=True)
            marker_file = marker_dir / ".kagglehub_downloaded"
            marker_file.write_text(str(model_path))
            logger.debug(f"Created download marker: {marker_file}")

        return Path(model_path)

    except ImportError:
        logger.error("kagglehub library not installed. Install with: pip install kagglehub")
        return None
    except Exception as e:
        logger.error(f"Failed to download Kaggle model {model_id}: {e}")
        return None


def export_kaggle_model(
    model_id: str,
    kaggle_handle: str,
    models_dir: str = "models",
    model_precision: str = "fp16",
    convert_args: Optional[dict] = None,
    quantize_args: Optional[dict] = None,
) -> bool:
    """
    Convert Kaggle model to OpenVINO format and optionally quantize to INT8.

    Args:
        model_id: ID of the Kaggle model to export
        kaggle_handle: Kaggle model handle (e.g., 'google/resnet-v1/tensorFlow2/50-classification')
        models_dir: Directory to save the exported model
        model_precision: Precision of the exported model (fp32, fp16, int8)
        convert_args: Optional dictionary of conversion arguments:
            - input_shape: list - Model input shape (default: [1, 224, 224, 3])
            - input_layout: str - Input tensor layout (e.g., 'NHWC')
            - tensor_layout: str - Target tensor layout (e.g., 'NCHW')
        quantize_args: Optional dictionary of quantization arguments:
            - calibration_samples: int - Number of samples for INT8 calibration (default: 300)
            - calibration_dataset: str - Dataset for calibration ('cifar100', default: 'cifar100')

    Returns:
        bool: True if export was successful, False otherwise
    """
    if not kaggle_handle:
        logger.error(f"Missing 'kaggle_handle' parameter for model '{model_id}'")
        return False

    # Get conversion parameters with defaults
    convert_args = convert_args or {}
    input_shape = convert_args.get("input_shape", [1, 224, 224, 3])
    input_layout = convert_args.get("input_layout")
    tensor_layout = convert_args.get("tensor_layout")
    base_dir = (Path(models_dir) / model_id).resolve()
    model_dir = base_dir / model_precision
    model_path = model_dir / f"{model_id}.xml"

    # Check if model is already exported
    if model_path.exists():
        logger.info(f" ✓ Model {model_id} already exported to OpenVINO {model_precision} format: {model_path}")
        return True

    logger.info(f"Exporting {model_id} to OpenVINO {model_precision} format...")

    # Create directory for the model
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        import openvino as ov

        from .common import quantize_model_with_nncf, save_openvino_model

        # Download model from Kaggle Hub
        source_path = download_kaggle_model(model_id, kaggle_handle, models_dir)
        if not source_path:
            logger.error(f"Failed to download model: {model_id}")
            return False

        logger.info("Converting model to OpenVINO IR format...")
        logger.debug(f"Using input_shape: {input_shape}")

        # Convert model to OpenVINO IR
        fp32_model = ov.convert_model(str(source_path), input=input_shape)

        # Apply preprocessing (layout conversion if needed)

        if input_layout and tensor_layout:
            logger.debug(f"Applying layout conversion: {tensor_layout} -> {input_layout}")
            ppp = ov.preprocess.PrePostProcessor(fp32_model)
            ppp.input().tensor().set_layout(ov.Layout(tensor_layout))
            ppp.input().model().set_layout(ov.Layout(input_layout))
            fp32_model = ppp.build()

        # Handle different precision targets
        if model_precision == "fp32":
            # Save FP32 model without compression
            success = save_openvino_model(fp32_model, model_path, compress_fp16=False)
            return success

        elif model_precision == "fp16":
            # Save FP16 model with compression
            success = save_openvino_model(fp32_model, model_path, compress_fp16=True)
            return success

        elif model_precision == "int8":
            # Perform INT8 quantization using NNCF
            logger.info("Preparing INT8 quantization with NNCF...")

            # Get quantization parameters
            quantize_args = quantize_args or {}
            calibration_samples = quantize_args.get("calibration_samples", 300)
            dataset_type = quantize_args.get("calibration_dataset", "cifar100")

            # Create datasets directory within the model's data directory
            datasets_dir = base_dir / "datasets"
            datasets_dir.mkdir(parents=True, exist_ok=True)

            # Prepare calibration dataset
            calibration_dataset = _prepare_calibration_dataset(
                model_id=model_id,
                dataset_type=dataset_type,
                samples=calibration_samples,
                input_name=fp32_model.input(0).get_any_name(),
                datasets_dir=str(datasets_dir),
            )

            if not calibration_dataset:
                logger.error("Failed to prepare calibration dataset for INT8 quantization")
                return False

            # Quantize model
            int8_model = quantize_model_with_nncf(
                model=fp32_model,
                calibration_dataset=calibration_dataset,
                subset_size=calibration_samples,
            )

            # Save INT8 model
            success = save_openvino_model(int8_model, model_path, compress_fp16=False)
            return success

        else:
            logger.error(f"Unsupported precision: {model_precision}")
            return False

    except Exception as e:
        logger.error(f"Failed to export Kaggle model {model_id}: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return False


def _prepare_calibration_dataset(model_id: str, dataset_type: str, samples: int, input_name: str, datasets_dir: str):
    """
    Prepare calibration dataset for NNCF quantization.

    Args:
        model_id: Model identifier
        dataset_type: Type of dataset (only 'cifar100' is supported)
        samples: Number of samples to use
        input_name: Name of the model input tensor
        datasets_dir: Directory to download and store datasets

    Returns:
        NNCF Dataset object or None if failed
    """
    try:
        import random

        import nncf
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms

        logger.info(f"Preparing CIFAR-100 calibration dataset with {samples} samples")
        logger.debug(f"Datasets directory: {datasets_dir}")

        # Define transforms for ResNet models (224x224 input)
        transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=Image.BILINEAR),
                transforms.CenterCrop(224),
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: x.to(torch.float32) / 255.0),
            ]
        )

        # Use CIFAR-100 as calibration dataset (download to proper location)
        dataset = datasets.CIFAR100(root=datasets_dir, train=False, transform=transform, download=True)

        # Subsample dataset
        if samples > 0 and samples < len(dataset):
            rng = random.Random(0)  # nosec B311
            idx = list(range(len(dataset)))
            rng.shuffle(idx)
            dataset = Subset(dataset, idx[:samples])

        # Create data loader
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # Create NNCF transform function
        def transform_fn(data_item):
            images, _ = data_item
            return {input_name: images.numpy()}

        # Create NNCF calibration dataset
        calibration_dataset = nncf.Dataset(data_loader, transform_fn)
        logger.info(f"Calibration dataset prepared with {len(dataset)} samples")

        # Clean up intermediate tar.gz file to save storage
        cifar_archive = Path(datasets_dir) / "cifar-100-python.tar.gz"
        if cifar_archive.exists():
            try:
                cifar_archive.unlink()
                logger.debug(f"Cleaned up intermediate archive: {cifar_archive}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up archive {cifar_archive}: {cleanup_error}")

        return calibration_dataset

    except Exception as e:
        logger.error(f"Failed to prepare calibration dataset: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None
