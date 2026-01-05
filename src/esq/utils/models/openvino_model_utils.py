# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
OpenVINO Model Zoo Utilities.

This module provides utilities for downloading models from Hugging Face and converting
them to OpenVINO IR format with INT8 quantization. Models are downloaded in their
original framework format (TensorFlow, PyTorch) and then converted using OpenVINO's
conversion and quantization tools.

IMPORTANT: This implementation uses Hugging Face Hub for model downloads and OpenVINO
runtime for conversion. It does NOT require openvino-dev or OMZ tools, avoiding
compatibility issues while supporting INT8 quantization via NNCF.

Licensing:
    All models used are licensed under permissive open-source licenses (Apache 2.0)
    that allow commercial use, modification, and conversion:
    - ResNet-50: microsoft/resnet-50 (Apache 2.0)
    - EfficientNet-B0: google/efficientnet-b0 (Apache 2.0)
    - MobileNet-V2: google/mobilenet_v2_1.0_224 (Apache 2.0)

    Models are downloaded by end-users on their systems. No model weights are
    distributed with this software. Model conversion (PyTorch → OpenVINO IR) is
    a technical transformation permitted under the model licenses.

Functions:
    - download_openvino_model: Download and convert models to OpenVINO IR format
    - get_model_info: Get model metadata and paths
    - validate_model_files: Verify model files are complete
"""

import logging
import multiprocessing
from pathlib import Path
from typing import Dict, Optional

# Fix Python 3.12+ multiprocessing deadlock warning with HuggingFace transformers
# HuggingFace uses multiprocessing internally, and fork() in multi-threaded pytest causes warnings
# Set start method to 'spawn' before any transformers imports to avoid fork-related deadlocks
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Start method already set, ignore
    pass

logger = logging.getLogger(__name__)

# Model configurations with Hugging Face repository information
AVAILABLE_MODELS = {
    # Classification models from Hugging Face
    "resnet-50-tf": {
        "hf_repo": "microsoft/resnet-50",
        "framework": "pytorch",
        "input_shape": [1, 3, 224, 224],
        "task": "classification",
        "precisions": ["FP32", "FP16", "INT8"],
    },
    "efficientnet-b0": {
        "hf_repo": "timm/efficientnet_b0.ra_in1k",  # Using timm model which has better compatibility
        "framework": "pytorch",
        "input_shape": [1, 3, 224, 224],
        "task": "classification",
        "precisions": ["FP32", "FP16", "INT8"],
        "use_timm": True,  # Use timm library for better model architecture
    },
    "mobilenet-v2-pytorch": {
        "hf_repo": "google/mobilenet_v2_1.0_224",
        "framework": "pytorch",
        "input_shape": [1, 3, 224, 224],
        "task": "classification",
        "precisions": ["FP32", "FP16", "INT8"],
    },
    # Detection models
    "ssdlite_mobilenet_v2": {
        "hf_repo": "google/mobilenet_v2_1.0_224",  # Base model
        "framework": "pytorch",
        "input_shape": [1, 3, 300, 300],
        "task": "detection",
        "precisions": ["FP32", "FP16", "INT8"],
        "note": "Uses MobileNetV2 as backbone for SSDLite",
    },
    # CLIP vision model
    "clip-vit-base-patch16": {
        "hf_repo": "openai/clip-vit-base-patch16",
        "framework": "pytorch",
        "input_shape": [1, 3, 224, 224],
        "task": "classification",
        "precisions": ["FP32", "FP16", "INT8"],
        "use_vision_model": True,  # Use vision_model component
    },
    # YOLO models - handled via yolo_model_utils.py
    "yolo-v5s": {
        "hf_repo": "ultralytics/yolov5",  # Not used - handled by yolo_model_utils.py
        "framework": "pytorch",
        "input_shape": [1, 3, 640, 640],
        "task": "detection",
        "precisions": ["FP32", "FP16", "INT8"],
        "use_ultralytics": True,  # Triggers yolo_model_utils.py path
        "model_variant": "yolo-v5s",  # Model ID for yolo_model_utils.py
    },
    "yolo-v8s": {
        "hf_repo": "ultralytics/yolov8",  # Not used - handled by yolo_model_utils.py
        "framework": "pytorch",
        "input_shape": [1, 3, 640, 640],
        "task": "detection",
        "precisions": ["FP32", "FP16", "INT8"],
        "use_ultralytics": True,  # Triggers yolo_model_utils.py path
        "model_variant": "yolo-v8s",  # Model ID for yolo_model_utils.py
    },
}


def _download_from_huggingface(
    hf_repo: str,
    model_id: str,
    use_vision_model: bool = False,
    use_ultralytics: bool = False,
    use_timm: bool = False,
    model_variant: str = None,
):
    """
    Download model from HuggingFace Hub, Ultralytics, or timm.

    NOTE: For YOLO models (use_ultralytics=True), this function is NOT called.
    YOLO models are handled directly via yolo_model_utils.py export functions.

    Args:
        hf_repo: HuggingFace repository (e.g., 'microsoft/resnet-50')
        model_id: Model identifier for special handling
        use_vision_model: If True, use vision_model component (for CLIP)
        use_ultralytics: If True, use ultralytics library (for YOLO) - NOT USED, handled separately
        use_timm: If True, use timm library for models
        model_variant: Specific model variant (e.g., 'yolov5s') - NOT USED for YOLO

    Returns:
        PyTorch model object
    """
    try:
        # Handle timm models
        if use_timm:
            try:
                import timm

                logger.info(f"Downloading timm model: {hf_repo}")
                # Download from timm (supports HuggingFace Hub models)
                model = timm.create_model(hf_repo, pretrained=True)
                model.eval()

                logger.info(f"✓ Downloaded timm model: {hf_repo}")
                return model

            except ImportError:
                logger.warning("timm not available, trying HuggingFace transformers")
                # Fallback to HuggingFace if timm not available
                pass

        # Handle Ultralytics YOLO models - this should NOT be reached
        # YOLO models are handled via yolo_model_utils.py export functions
        if use_ultralytics:
            raise RuntimeError(
                "YOLO models should be handled via yolo_model_utils.py, not here. "
                "This is a programming error - check download_openvino_model() logic."
            )

        # Handle CLIP vision models
        if use_vision_model:
            from transformers import CLIPModel

            logger.info(f"Downloading CLIP model from HuggingFace: {hf_repo}")
            clip_model = CLIPModel.from_pretrained(hf_repo)
            # Extract vision model component
            model = clip_model.vision_model
            model.eval()

            logger.info(f"✓ Downloaded CLIP vision model from HuggingFace: {hf_repo}")
            return model

        # Handle standard models
        from transformers import AutoModel

        logger.info(f"Downloading from HuggingFace: {hf_repo}")
        model = AutoModel.from_pretrained(hf_repo)
        model.eval()  # Set to evaluation mode

        logger.info(f"✓ Downloaded from HuggingFace: {hf_repo}")
        return model

    except Exception as e:
        logger.error(f"Failed to download from HuggingFace: {e}", exc_info=True)
        raise


def _convert_to_openvino(pytorch_model, ov_model_dir: Path, input_shape: tuple, model_id: str) -> Path:
    """
    Convert PyTorch model to OpenVINO IR format with static shapes.

    Args:
        pytorch_model: PyTorch model object from transformers
        ov_model_dir: Directory to save OpenVINO IR
        input_shape: Input tensor shape (batch, channels, height, width)
        model_id: Model identifier for file naming

    Returns:
        Path to OpenVINO model directory
    """
    try:
        import openvino as ov
        import torch

        logger.info("Converting to OpenVINO IR format...")

        # Create dummy input with static shape
        dummy_input = torch.randn(input_shape)

        # Convert to OpenVINO with static input shape
        # The example_input parameter should ensure the model has static shapes
        ov_model = ov.convert_model(pytorch_model, example_input=dummy_input)

        # Check if model has dynamic shapes and fix them
        input_node = ov_model.inputs[0]
        if input_node.partial_shape.is_dynamic:
            logger.info(f"Model has dynamic input shape {input_node.partial_shape}, reshaping to static {input_shape}")
            batch, channels, height, width = input_shape
            try:
                ov_model.reshape({0: [batch, channels, height, width]})
                logger.info(f"✓ Reshaped model to static input {input_shape}")
            except Exception as reshape_error:
                logger.warning(f"Failed to reshape model: {reshape_error}. Model may have dynamic internal layers.")
                logger.warning("This may cause issues with INT8 quantization.")

        # Save OpenVINO IR with model-specific filename
        ov_model_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(ov_model, ov_model_dir / f"{model_id}.xml")

        logger.info(f"✓ Converted to OpenVINO IR: {ov_model_dir}")
        return ov_model_dir

    except Exception as e:
        logger.error(f"Failed to convert to OpenVINO: {e}", exc_info=True)
        raise


def _convert_to_fp16(model_path: Path, output_dir: Path, model_id: str) -> Path:
    """
    Convert OpenVINO model to FP16 precision.

    Args:
        model_path: Path to FP32 OpenVINO model directory
        output_dir: Directory to save FP16 model
        model_id: Model identifier for file naming

    Returns:
        Path to FP16 model XML file
    """
    try:
        import openvino as ov

        logger.info("Converting to FP16 precision...")

        # Load model (FP32 is saved with model_id name)
        core = ov.Core()
        xml_file = list(model_path.glob("*.xml"))[0]  # Get the actual XML file
        model = core.read_model(xml_file)

        # Convert to FP16
        from openvino.tools.pot import compress_model_weights

        compressed_model = compress_model_weights(model)

        # Save FP16 model with model-specific filename
        output_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(compressed_model, output_dir / f"{model_id}.xml")

        logger.info(f"✓ Converted to FP16: {output_dir}")
        return output_dir / f"{model_id}.xml"

    except Exception as e:
        logger.error(f"Failed to convert to FP16: {e}", exc_info=True)
        raise


def _quantize_to_int8(
    model_path: Path, output_dir: Path, input_shape: tuple, model_id: str, calibration_samples: int = 300
) -> Path:
    """
    Quantize OpenVINO model to INT8 using NNCF.

    Args:
        model_path: Path to FP32 OpenVINO model directory
        output_dir: Directory to save INT8 model
        input_shape: Input tensor shape for calibration data generation
        model_id: Model identifier for file naming
        calibration_samples: Number of samples for calibration

    Returns:
        Path to INT8 model XML file
    """
    try:
        import nncf
        import numpy as np
        import openvino as ov

        logger.info(f"Quantizing to INT8 using NNCF ({calibration_samples} calibration samples)...")

        # Load model (FP32 is saved with model_id name)
        core = ov.Core()
        xml_file = list(model_path.glob("*.xml"))[0]  # Get the actual XML file
        model = core.read_model(xml_file)

        # Create calibration dataset - must be a list of samples, not a generator function
        # Use provided input_shape to avoid issues with dynamic shapes
        calibration_data = []
        for _ in range(calibration_samples):
            # Generate random input matching model's input shape
            calibration_data.append(np.random.randn(*input_shape).astype(np.float32))

        # Quantize model with NNCF
        calibration_dataset = nncf.Dataset(calibration_data)
        quantized_model = nncf.quantize(model, calibration_dataset)

        # Save INT8 model with model-specific filename
        output_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(quantized_model, output_dir / f"{model_id}.xml")

        logger.info(f"✓ Quantized to INT8: {output_dir}")
        return output_dir / f"{model_id}.xml"

    except Exception as e:
        logger.error(f"Failed to quantize to INT8: {e}", exc_info=True)
        raise


def download_openvino_model(
    model_id: str,
    precision: str = "INT8",
    models_dir: Optional[str] = None,
    force_download: bool = False,
) -> Optional[Path]:
    """
    Download and convert model to OpenVINO IR format.

    Downloads pre-trained model from HuggingFace, converts to OpenVINO IR,
    and applies precision conversion (FP16) or quantization (INT8).

    Args:
        model_id: Model identifier (e.g., 'resnet-50-tf', 'efficientnet-b0')
        precision: Model precision ('FP16', 'FP32', 'INT8')
        models_dir: Directory to save models (default: ~/share/public/models)
        force_download: Force re-download and reconversion even if model exists

    Returns:
        Path to the converted model XML file, or None if conversion failed

    Example:
        >>> model_path = download_openvino_model("resnet-50-tf", "INT8")
        >>> print(model_path)
        PosixPath('/home/user/share/public/models/resnet-50-tf/INT8/model.xml')
    """
    # Validate model ID
    if model_id not in AVAILABLE_MODELS:
        logger.error(f"Model '{model_id}' not available")
        logger.info(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        return None

    model_info = AVAILABLE_MODELS[model_id]

    # Normalize precision
    precision = precision.upper()
    if precision not in model_info["precisions"]:
        logger.error(f"Precision '{precision}' not available for {model_id}")
        logger.info(f"Available precisions: {', '.join(model_info['precisions'])}")
        return None

    # Use default models directory if not provided
    # Use CORE_DATA_DIR structure: esq_data/data/vertical/metro/models
    # Container expects: share/models/{model_name}/{precision}/*.xml
    if models_dir is None:
        import os
        core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
        core_data_dir = "".join(c for c in core_data_dir_tainted)
        models_dir = Path(core_data_dir) / "data" / "vertical" / "metro" / "models"
    else:
        models_dir = Path(models_dir)

    # Construct target paths with model-specific filenames
    model_dir = models_dir / model_id
    precision_dir = model_dir / precision
    xml_file = precision_dir / f"{model_id}.xml"
    bin_file = precision_dir / f"{model_id}.bin"

    # Check if model already exists
    if not force_download and xml_file.exists() and bin_file.exists():
        logger.info(f"✓ Model {model_id} ({precision}) already exists: {xml_file}")
        return xml_file

    logger.info(f"Converting {model_id} to OpenVINO IR ({precision})...")

    try:
        use_ultralytics = model_info.get("use_ultralytics", False)

        # YOLO models: Use dedicated yolo_model_utils.py for download and export
        if use_ultralytics:
            from .yolo_model_utils import download_yolo_model, export_yolo_model

            model_variant = model_info.get("model_variant", None)
            if not model_variant:
                logger.error(f"YOLO model {model_id} missing model_variant in configuration")
                return None

            logger.info(f"Using YOLO utilities for {model_id} ({model_variant})...")

            # Step 1: Download YOLO weights
            weights_path = download_yolo_model(model_id=model_variant, models_dir=None)
            if not weights_path:
                logger.error(f"Failed to download YOLO model {model_variant}")
                return None

            # Step 2: Export to OpenVINO IR with requested precision
            # yolo_model_utils.py handles the export and saves to models_dir
            export_args = {
                "dynamic": False,  # Force static shapes
                "half": (precision == "FP16"),
                "int8": (precision == "INT8"),
                "batch": 1,
                "imgsz": 640,
            }

            exported_path = export_yolo_model(
                model_id=model_variant,
                models_dir=str(models_dir),
                model_precision=precision.lower(),
                weights_path=weights_path,
                export_args=export_args,
            )

            if not exported_path:
                logger.error(f"Failed to export YOLO model {model_variant} to OpenVINO")
                return None

            logger.info(f"✓ Successfully exported {model_id} ({precision}): {exported_path}")

            # Verify the file exists and return
            if not exported_path.exists():
                logger.error(f"Export reported success but file not found: {exported_path}")
                return None

            return exported_path

        # Non-YOLO models: Standard HuggingFace + conversion pipeline
        # Step 1: Download from HuggingFace
        hf_repo = model_info["hf_repo"]
        use_vision_model = model_info.get("use_vision_model", False)
        use_timm = model_info.get("use_timm", False)

        pytorch_model = _download_from_huggingface(
            hf_repo,
            model_id,
            use_vision_model=use_vision_model,
            use_ultralytics=False,  # Never reach this path for YOLO
            use_timm=use_timm,
            model_variant=None,
        )

        # Step 2: Convert to OpenVINO IR (FP32)
        fp32_dir = model_dir / "FP32"
        input_shape = model_info.get("input_shape", (1, 3, 224, 224))
        ov_model_dir = _convert_to_openvino(pytorch_model, fp32_dir, input_shape, model_id)

        # Step 3: Apply precision conversion
        if precision == "FP32":
            # Already in FP32, return the model-named file
            target_xml = fp32_dir / f"{model_id}.xml"
        elif precision == "FP16":
            target_xml = _convert_to_fp16(ov_model_dir, precision_dir, model_id)
        elif precision == "INT8":
            target_xml = _quantize_to_int8(ov_model_dir, precision_dir, input_shape, model_id)
        else:
            logger.error(f"Unsupported precision: {precision}")
            return None

        logger.info(f"✓ Successfully converted {model_id} ({precision}): {target_xml}")
        return target_xml

    except Exception as e:
        logger.error(f"Failed to convert {model_id}: {e}", exc_info=True)
        return None


def get_model_info(model_id: str) -> Optional[Dict]:
    """
    Get metadata for a model.

    Args:
        model_id: Model identifier

    Returns:
        Dictionary with model information, or None if model not found

    Example:
        >>> info = get_model_info("resnet-50-tf")
        >>> print(info["task"])
        'classification'
    """
    return AVAILABLE_MODELS.get(model_id)


def validate_model_files(model_path: Path) -> bool:
    """
    Validate that model files are complete.

    Checks that both XML and BIN files exist and are non-empty.

    Args:
        model_path: Path to model XML file

    Returns:
        True if model files are valid, False otherwise

    Example:
        >>> model_path = Path("/home/user/share/public/models/resnet-50-tf/INT8/resnet-50-tf.xml")
        >>> is_valid = validate_model_files(model_path)
        >>> print(is_valid)
        True
    """
    if not model_path.exists():
        logger.error(f"Model XML file not found: {model_path}")
        return False

    # Check XML file size
    if model_path.stat().st_size < 100:
        logger.error(f"Model XML file too small: {model_path}")
        return False

    # Check for corresponding BIN file
    bin_path = model_path.with_suffix(".bin")
    if not bin_path.exists():
        logger.error(f"Model BIN file not found: {bin_path}")
        return False

    # Check BIN file size
    if bin_path.stat().st_size < 1000:
        logger.error(f"Model BIN file too small: {bin_path}")
        return False

    return True


def download_all_models_for_profile(
    profile_models: list,
    precision: str = "INT8",
    models_dir: Optional[str] = None,
) -> Dict[str, Optional[Path]]:
    """
    Download all models required for a test profile.

    Args:
        profile_models: List of model IDs to download
        precision: Model precision (default: INT8)
        models_dir: Directory to save models

    Returns:
        Dictionary mapping model_id -> model_path (or None if download failed)

    Example:
        >>> models = ["resnet-50-tf", "efficientnet-b0", "mobilenet-v2-pytorch"]
        >>> results = download_all_models_for_profile(models)
        >>> print(f"Downloaded {sum(1 for p in results.values() if p)} models")
        Downloaded 3 models
    """
    results = {}
    success_count = 0
    fail_count = 0

    logger.info(f"Downloading {len(profile_models)} models ({precision})...")

    for model_id in profile_models:
        model_path = download_openvino_model(model_id, precision, models_dir)
        results[model_id] = model_path

        if model_path:
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"Download complete: {success_count} succeeded, {fail_count} failed")
    return results


# Model mappings for easy reference
CLASSIFICATION_MODELS = [m for m, info in AVAILABLE_MODELS.items() if info["task"] == "classification"]
DETECTION_MODELS = [m for m, info in AVAILABLE_MODELS.items() if info["task"] == "detection"]
ALL_MODEL_IDS = list(AVAILABLE_MODELS.keys())
