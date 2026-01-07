# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CLIP Model Utilities.

This module provides utilities for downloading and exporting CLIP (Contrastive Language-Image Pre-training)
models to OpenVINO IR format for vision-language tasks.

Functions:
    - download_clip_model: Download CLIP model from Hugging Face
    - export_clip_model: Export CLIP model to OpenVINO IR format
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import openvino as ov

from .common import save_openvino_model

logger = logging.getLogger(__name__)

# Supported CLIP models
CLIP_MODELS = {
    "clip-vit-base-patch16": {
        "hf_id": "openai/clip-vit-base-patch16",
        "input_shape": [1, 3, 224, 224],
        "task": "vision-language",
    },
    "clip-vit-base-patch32": {
        "hf_id": "openai/clip-vit-base-patch32",
        "input_shape": [1, 3, 224, 224],
        "task": "vision-language",
    },
    "clip-vit-large-patch14": {
        "hf_id": "openai/clip-vit-large-patch14",
        "input_shape": [1, 3, 224, 224],
        "task": "vision-language",
    },
}


def download_clip_model(
    model_id: str,
    models_dir: Optional[str] = None,
) -> Optional[Path]:
    """
    Download CLIP model from Hugging Face.

    Args:
        model_id: CLIP model identifier (e.g., 'clip-vit-base-patch16')
        models_dir: Directory to save models (default: ~/share/public/models)

    Returns:
        Path to the downloaded model directory, or None if download failed

    Example:
        >>> model_path = download_clip_model("clip-vit-base-patch16")
        >>> print(model_path)
        PosixPath('/home/user/share/public/models/clip-vit-base-patch16/original')
    """
    if model_id not in CLIP_MODELS:
        logger.error(f"Invalid CLIP model ID '{model_id}'")
        logger.info(f"Available models: {', '.join(CLIP_MODELS.keys())}")
        return None

    model_info = CLIP_MODELS[model_id]
    hf_model_id = model_info["hf_id"]

    # Use CORE_DATA_DIR structure: esq_data/data/vertical/metro/models
    if models_dir is None:
        core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
        core_data_dir = "".join(c for c in core_data_dir_tainted)
        models_dir = Path(core_data_dir) / "data" / "vertical" / "metro" / "models"
    else:
        models_dir = Path(models_dir)

    # Create model directory
    model_dir = models_dir / model_id / "original"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check if model already exists
    config_file = model_dir / "config.json"
    if config_file.exists():
        logger.info(f"✓ CLIP model {model_id} already downloaded: {model_dir}")
        return model_dir

    logger.info(f"Downloading CLIP model {model_id} from Hugging Face...")

    try:
        from transformers import CLIPModel, CLIPProcessor

        # Download model and processor
        logger.debug(f"Downloading from {hf_model_id}")
        model = CLIPModel.from_pretrained(hf_model_id)
        processor = CLIPProcessor.from_pretrained(hf_model_id)

        # Save to local directory
        model.save_pretrained(str(model_dir))
        processor.save_pretrained(str(model_dir))

        logger.info(f"CLIP model downloaded: {model_dir}")
        return model_dir

    except ImportError:
        logger.error("Required library not available.")
        return None
    except Exception as e:
        logger.error(f"Failed to download CLIP model {model_id}: {e}", exc_info=True)
        return None


def export_clip_model(
    model_id: str,
    models_dir: Optional[Union[str, Path]] = None,
    model_precision: str = "fp16",
    original_model_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Export CLIP model to OpenVINO IR format.

    Exports CLIP vision encoder to OpenVINO Intermediate Representation (IR)
    format (.xml + .bin files) for use with OpenVINO inference engine.

    Args:
        model_id: CLIP model identifier
        models_dir: Directory to save exported model (example: ~/share/public/models)
        model_precision: Precision of the exported model (fp32, fp16, int8)
        original_model_path: Optional path to downloaded model directory

    Returns:
        Path to the exported model XML file, or None if export failed

    Example:
        >>> xml_path = export_clip_model("clip-vit-base-patch16", model_precision="fp16")
        >>> print(xml_path)
        PosixPath('/home/user/share/public/models/clip-vit-base-patch16/FP16/clip-vit-base-patch16.xml')
    """
    if model_id not in CLIP_MODELS:
        logger.error(f"Invalid CLIP model ID '{model_id}'")
        logger.info(f"Available models: {', '.join(CLIP_MODELS.keys())}")
        return None

    model_info = CLIP_MODELS[model_id]

    # Normalize precision to lowercase
    model_precision = model_precision.lower()

    # Use CORE_DATA_DIR structure: esq_data/data/vertical/metro/models
    if models_dir is None:
        core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
        core_data_dir = "".join(c for c in core_data_dir_tainted)
        models_dir = Path(core_data_dir) / "data" / "vertical" / "metro" / "models"
    else:
        models_dir = Path(models_dir)

    # Construct export path
    model_dir = models_dir / model_id / model_precision.upper()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_id}.xml"

    # Check if model is already exported
    if model_path.exists():
        logger.info(f"Model {model_id} already exported: {model_path}")
        return model_path

    # Determine original model path
    if original_model_path is None:
        original_model_path = models_dir / model_id / "original"

    # Verify original model exists
    if not original_model_path.exists():
        logger.error(f"Original CLIP model not found: {original_model_path}")
        logger.info("Download the model first using download_clip_model()")
        return None

    logger.info(f"Exporting {model_id} to OpenVINO {model_precision.upper()} format...")

    try:
        import torch
        from transformers import CLIPModel

        # Load model
        logger.debug(f"Loading model from {original_model_path}")
        model = CLIPModel.from_pretrained(str(original_model_path))
        model.eval()

        # Get vision model (encoder)
        vision_model = model.vision_model

        # Create dummy input for tracing
        input_shape = model_info["input_shape"]
        dummy_input = torch.randn(*input_shape)

        logger.debug("Tracing model with torch.jit.trace")
        traced_model = torch.jit.trace(vision_model, dummy_input)

        # Convert to OpenVINO
        logger.debug("Converting to OpenVINO IR")
        ov_model = ov.convert_model(traced_model, example_input=dummy_input)

        # Add model metadata
        ov_model.set_rt_info("CLIP", ["model_info", "model_type"])
        ov_model.set_rt_info(model_id, ["model_info", "model_name"])

        # Save model in requested precision
        if model_precision == "fp32":
            save_openvino_model(ov_model, model_path, compress_fp16=False)
        elif model_precision == "fp16":
            save_openvino_model(ov_model, model_path, compress_fp16=True)
        elif model_precision == "int8":
            # For INT8, quantize the model
            try:
                from nncf import AdvancedQuantizationParameters, Dataset, ModelType, quantize

                # Create simple calibration dataset
                def data_gen():
                    for _ in range(10):
                        dummy = np.random.rand(*input_shape).astype(np.float32)
                        yield {ov_model.inputs[0].get_any_name(): dummy}

                dataset = Dataset(data_gen())
                quantized_model = quantize(
                    model=ov_model,
                    calibration_dataset=dataset,
                    model_type=ModelType.TRANSFORMER,
                    advanced_parameters=AdvancedQuantizationParameters(smooth_quant_alpha=0.6),
                    subset_size=10,
                )
                quantized_model.set_rt_info("CLIP", ["model_info", "model_type"])
                quantized_model.set_rt_info(model_id, ["model_info", "model_name"])
                save_openvino_model(quantized_model, model_path, compress_fp16=False)
            except ImportError:
                logger.warning("NNCF not available, saving as FP16 instead of INT8")
                save_openvino_model(ov_model, model_path, compress_fp16=True)
        else:
            logger.error(f"Unsupported model_precision: {model_precision}")
            return None

        logger.info(f"Exported {model_id} to {model_path}")
        return model_path

    except ImportError as e:
        logger.error(f"Required library not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to export CLIP model {model_id}: {e}", exc_info=True)
        return None


def download_and_export_clip_model(
    model_id: str,
    models_dir: Optional[str] = None,
    model_precision: str = "fp16",
) -> Optional[Path]:
    """
    Download and export CLIP model in one step.

    Args:
        model_id: CLIP model identifier
        models_dir: Directory to save models
        model_precision: Target precision (fp32, fp16, int8)

    Returns:
        Path to exported model XML file, or None if failed

    Example:
        >>> model_path = download_and_export_clip_model("clip-vit-base-patch16", model_precision="int8")
        >>> print(model_path)
        PosixPath('/home/user/share/public/models/clip-vit-base-patch16/INT8/clip-vit-base-patch16.xml')
    """
    # Download original model
    original_path = download_clip_model(model_id, models_dir)
    if not original_path:
        return None

    # Export to OpenVINO IR
    return export_clip_model(model_id, models_dir, model_precision, original_path)
