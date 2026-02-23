# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
YOLO Model Utilities for GPU Testing.

This module provides GPU test-specific functions for downloading and exporting
YOLO models to OpenVINO IR format. Extracted from ultralytics.py to avoid
impacting the shared Ultralytics module used by other teams.

Functions:
    - download_yolo_model: Download YOLO weights using Ultralytics
    - export_yolo_model: Export YOLO weights to OpenVINO IR format
    - download_test_image: Download test images for inference
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np
import openvino as ov
import requests

from .common import save_openvino_model

logger = logging.getLogger(__name__)

# OpenVINO storage for test assets
OPENVINO_STORAGE_BASE = "https://storage.openvinotoolkit.org"

# Supported YOLO models
YOLO_MODELS = {
    # YOLOv5 models (recommended for GPU testing)
    "yolov5n": "YOLOv5",  # Nano - smallest, fastest
    "yolov5s": "YOLOv5",  # Small - recommended for benchmarking
    "yolo-v5s": "YOLOv5",  # Alias with dash
    "yolov5m": "YOLOv5",  # Medium
    "yolov5l": "YOLOv5",  # Large
    "yolov5x": "YOLOv5",  # Extra large
    # YOLOv8 models
    "yolov8n": "YOLOv8",
    "yolov8s": "YOLOv8",
    "yolo-v8s": "YOLOv8",  # Alias with dash
    "yolov8m": "YOLOv8",
    "yolov8l": "YOLOv8",
    "yolov8x": "YOLOv8",
    # YOLOv11 models (latest)
    "yolov11n": "YOLOv11",  # Nano - smallest, fastest
    "yolo11n": "YOLOv11",  # Alias without 'v'
    "yolov11s": "YOLOv11",  # Small
    "yolo11s": "YOLOv11",  # Alias without 'v'
    "yolov11m": "YOLOv11",  # Medium
    "yolo11m": "YOLOv11",  # Alias without 'v'
    "yolov11l": "YOLOv11",  # Large
    "yolo11l": "YOLOv11",  # Alias without 'v'
    "yolov11x": "YOLOv11",  # Extra large
    "yolo11x": "YOLOv11",  # Alias without 'v'
}


def download_yolo_model(model_id: str, models_dir: Optional[str] = None) -> Optional[Path]:
    """
    Download YOLO model weights using Ultralytics.

    Downloads to Ultralytics standard cache location (~/.cache/ultralytics/)
    to avoid duplication across projects.

    Args:
        model_id: YOLO model identifier (e.g., 'yolov5s', 'yolov5n')
        models_dir: Directory hint for export location (weights stay in Ultralytics cache)

    Returns:
        Path to the downloaded weights in Ultralytics cache, or None if failed

    Example:
        >>> weights_path = download_yolo_model("yolov5s")
        >>> print(weights_path)
        PosixPath('/home/user/.cache/ultralytics/yolov5s.pt')
    """
    if model_id not in YOLO_MODELS:
        logger.error(f"Invalid model name '{model_id}'.")
        logger.debug(f"Available models: {', '.join(YOLO_MODELS.keys())}")
        return None

    logger.info(f"Downloading YOLO weights for: {model_id}")

    # Normalize model ID - Ultralytics doesn't recognize dash variants
    # yolo-v5s → yolov5s, yolo-v8s → yolov8s
    ultralytics_model_id = model_id.replace("-", "")

    # YOLOv11 models use 'yolo11n' format (without 'v') in Ultralytics
    # yolov11n → yolo11n, yolov11s → yolo11s, etc.
    if ultralytics_model_id.startswith("yolov11"):
        ultralytics_model_id = ultralytics_model_id.replace("yolov11", "yolo11")

    logger.debug(f"Normalized model ID for Ultralytics: {model_id} → {ultralytics_model_id}")

    # Use CORE_DATA_DIR structure: esq_data/data/vertical/metro/temp for Ultralytics downloads
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    core_data_dir = "".join(c for c in core_data_dir_tainted)
    temp_dir = Path(core_data_dir) / "data" / "vertical" / "metro" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()

    try:
        from ultralytics import YOLO

        # Change to temp directory to avoid polluting workspace
        os.chdir(temp_dir)
        logger.debug(f"Using temp directory for downloads: {temp_dir}")

        # Download model (Ultralytics manages cache automatically)
        # Use normalized name (without dashes) for Ultralytics
        model_name = f"{ultralytics_model_id}.pt"
        logger.debug(f"Initializing YOLO model: {model_name}")
        model = YOLO(model_name)
        model.info()

        weights_path = None
        # Check temp directory first using normalized name
        temp_weights = temp_dir / f"{ultralytics_model_id}.pt"
        if temp_weights.exists():
            weights_path = temp_weights
            logger.debug(f"Found weights in temp directory: {temp_weights}")
        else:
            # Try with suffix in temp dir
            matching_temp = list(temp_dir.glob(f"{ultralytics_model_id}*.pt"))
            if matching_temp:
                weights_path = matching_temp[0]
                logger.debug(f"Found weights with suffix in temp: {weights_path.name}")

        # If not in temp, check cache directory
        if not weights_path:
            cache_dir = Path.home() / ".cache" / "ultralytics"
            if cache_dir.exists():
                cache_weights = cache_dir / f"{ultralytics_model_id}.pt"
                if cache_weights.exists():
                    weights_path = cache_weights
                    logger.debug(f"Found weights in cache: {cache_weights}")
                else:
                    # Try with suffix in cache
                    matching_cache = list(cache_dir.glob(f"{ultralytics_model_id}*.pt"))
                    if matching_cache:
                        weights_path = matching_cache[0]
                        logger.debug(f"Found weights with suffix in cache: {weights_path.name}")

        # If still not found, report error
        if not weights_path:
            logger.error("Weights file not found after download")
            logger.error(f"Searched in temp: {temp_dir}")
            logger.error(f"Searched in cache: {cache_dir if cache_dir.exists() else 'cache dir does not exist'}")
            logger.error(f"Pattern: {model_id}*.pt")
            # List what's in both locations
            temp_files = list(temp_dir.glob("*.pt"))
            if temp_files:
                logger.debug(f"Files in temp: {[f.name for f in temp_files]}")
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.pt"))
                if cache_files:
                    logger.debug(f"Files in cache: {[f.name for f in cache_files]}")
            return None

        logger.info(f"✓ YOLO model weights ready at: {weights_path}")
        return weights_path

    except Exception as e:
        logger.error(f"Failed to download YOLO model {model_id}: {e}", exc_info=True)
        return None
    finally:
        # Always restore original directory
        os.chdir(original_cwd)

        # NOTE: Do NOT clean up .pt files here - they are needed by export_yolo_model()
        # The export function will handle cleanup after it's done using the weights


def export_yolo_model(
    model_id: str,
    models_dir: Optional[Union[str, Path]] = None,
    model_precision: str = "fp16",
    weights_path: Optional[Path] = None,
    export_args: Optional[dict] = None,
) -> Optional[Path]:
    """
    Convert YOLO model weights to OpenVINO IR format.

    Exports PyTorch weights to OpenVINO Intermediate Representation (IR)
    format (.xml + .bin files) for use with OpenVINO inference engine.

    Args:
        model_id: ID of the YOLO model to export (e.g., 'yolov5s')
        models_dir: Directory to save the exported model (default: esq_data/data/vertical/metro/models)
        model_precision: Precision of the exported model (fp32, fp16, int8)
        weights_path: Optional path to .pt weights file (if None, uses Ultralytics cache)
        export_args: Optional dictionary of export arguments for OpenVINO:
            - dynamic: bool - Allow dynamic input sizes (default: False)
            - half: bool - Enable FP16 quantization (default: False)
            - int8: bool - Enable INT8 quantization (default: False)
            - nms: bool - Add Non-Maximum Suppression (default: False)
            - batch: int - Export model batch size (default: 1)
            - imgsz: int or tuple - Input image size (default: 640)
            - data: str - Dataset config for INT8 calibration (default: 'coco8.yaml')
            - fraction: float - Fraction of dataset for INT8 calibration (default: 1.0)

    Returns:
        Path to the exported model XML file, or None if export failed

    Example:
        >>> xml_path = export_yolo_model(
        ...     model_id="yolov5s",
        ...     models_dir="esq_data/data/vertical/metro/models",
        ...     model_precision="fp16"
        ... )
        >>> print(xml_path)
        PosixPath('esq_data/data/vertical/metro/models/yolov5s/FP16/yolov5s.xml')
    """
    if model_id not in YOLO_MODELS:
        logger.debug(f"Error: Invalid model name '{model_id}'.")
        logger.debug(f"Available models: {', '.join(YOLO_MODELS.keys())}")
        return None

    model_type = YOLO_MODELS[model_id]

    # Normalize precision to lowercase for consistency
    model_precision = model_precision.lower()

    # Use default models directory if not provided
    # Use CORE_DATA_DIR structure: esq_data/data/vertical/metro/models
    if models_dir is None:
        core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
        core_data_dir = "".join(c for c in core_data_dir_tainted)
        models_dir = Path(core_data_dir) / "data" / "vertical" / "metro" / "models"
        logger.debug(f"Using default model directory: {models_dir}")

    # Construct export path
    # NOTE: For YOLO models, do NOT use subdirectory components since precision is already in path
    # The benchmark script expects models at: share/models/{model_id}/{PRECISION}/
    # NOT: share/models/{model_id}/{PRECISION}/int8/ (which would duplicate precision info)
    base_dir = (Path(models_dir) / model_id).resolve()
    model_dir = base_dir / model_precision.upper()
    model_path = model_dir / f"{model_id}.xml"

    # Check if model is already exported
    if model_path.exists():
        logger.info(f"✓ Model {model_id} already exported: {model_path}")
        return model_path

    # Determine weights path (use provided path or fallback to Ultralytics cache)
    if weights_path is None:
        # Normalize model ID for Ultralytics cache lookup
        # YOLOv11 models use 'yolo11n' format (without 'v') in Ultralytics
        cache_model_id = model_id
        if cache_model_id.startswith("yolov11"):
            cache_model_id = cache_model_id.replace("yolov11", "yolo11")

        # Check standard Ultralytics cache location
        cache_dir = Path.home() / ".cache" / "ultralytics"
        weights_path = cache_dir / f"{cache_model_id}.pt"
        logger.debug(f"Using Ultralytics cache location: {weights_path}")

    # Verify model weights exist
    if not weights_path.exists():
        logger.error(f"YOLO model weights not found: {weights_path}. Please download first.")
        return None

    logger.debug(f"Exporting {model_id} to OpenVINO {model_precision.upper()} format...")

    # Create directory for the model
    model_dir.mkdir(parents=True, exist_ok=True)

    # Prepare export arguments
    export_kwargs = {"format": "openvino"}

    if export_args:
        # Apply custom export arguments from config
        if "dynamic" in export_args:
            export_kwargs["dynamic"] = export_args["dynamic"]
        if "half" in export_args:
            export_kwargs["half"] = export_args["half"]
        if "int8" in export_args:
            export_kwargs["int8"] = export_args["int8"]
        if "nms" in export_args:
            export_kwargs["nms"] = export_args["nms"]
        if "batch" in export_args:
            export_kwargs["batch"] = export_args["batch"]
        if "imgsz" in export_args:
            export_kwargs["imgsz"] = export_args["imgsz"]
        if "data" in export_args:
            export_kwargs["data"] = export_args["data"]
        if "fraction" in export_args:
            export_kwargs["fraction"] = export_args["fraction"]

    logger.debug(f"Export arguments: {export_kwargs}")

    # Use CORE_DATA_DIR structure: esq_data/data/vertical/metro/temp for Ultralytics export
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    core_data_dir = "".join(c for c in core_data_dir_tainted)
    temp_dir = Path(core_data_dir) / "data" / "vertical" / "metro" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()

    try:
        from ultralytics import YOLO

        # Change to temp directory to avoid polluting workspace
        os.chdir(temp_dir)
        logger.debug(f"Using temp directory for export: {temp_dir}")

        # Convert to OpenVINO IR with custom export arguments
        model = YOLO(str(weights_path), task="detect")
        converted_path = Path(model.export(**export_kwargs)).resolve()

        # Find the actual exported .xml file (Ultralytics may add suffix like 'u')
        # Note: Ultralytics may create nested subdirectories (e.g., /INT8/int8/)
        xml_files = list(converted_path.rglob("*.xml"))  # Use rglob to search recursively
        if not xml_files:
            logger.error(f"No XML file found in exported directory: {converted_path}")
            return None

        exported_xml = xml_files[0]
        logger.debug(f"Found exported model: {exported_xml} (Ultralytics naming)")
        if exported_xml.stem != model_id:
            logger.debug(f"Note: Ultralytics used '{exported_xml.stem}' instead of '{model_id}'")

        core = ov.Core()
        ov_model = core.read_model(model=str(exported_xml))

        # Reshape model if dynamic shape was used in export
        if export_args and export_args.get("dynamic", False):
            batch_size = export_args.get("batch", 1)
            imgsz = export_args.get("imgsz", 640)

            # Handle both int and tuple formats for imgsz
            if isinstance(imgsz, int):
                height, width = imgsz, imgsz
            else:
                height, width = imgsz[0], imgsz[1]

            target_shape = [batch_size, 3, height, width]
            logger.debug(f"Reshaping dynamic model to static shape: {target_shape}")
            ov_model.reshape(target_shape)

        ov_model.set_rt_info(model_type, ["model_info", "model_type"])

        # Save model with standardized name (model_id) regardless of Ultralytics naming
        logger.debug(f"Saving model as: {model_path}")

        # Save model in requested precision
        if model_precision == "fp32":
            save_openvino_model(ov_model, model_path, compress_fp16=False)
        elif model_precision == "fp16":
            save_openvino_model(ov_model, model_path, compress_fp16=True)
        elif model_precision == "int8":

            def data_gen():
                for _ in range(10):
                    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
                    yield {input_key: dummy}

            # For INT8 quantization, re-read and reshape the model
            fp16_model = core.read_model(model=str(exported_xml))

            if export_args and export_args.get("dynamic", False):
                batch_size = export_args.get("batch", 1)
                imgsz = export_args.get("imgsz", 640)

                if isinstance(imgsz, int):
                    height, width = imgsz, imgsz
                else:
                    height, width = imgsz[0], imgsz[1]

                target_shape = [batch_size, 3, height, width]
                logger.debug(f"Reshaping model for INT8 quantization: {target_shape}")
                fp16_model.reshape(target_shape)

            input_key = fp16_model.inputs[0].get_any_name()

            try:
                from nncf import Dataset, quantize

                dataset = Dataset(data_gen())
                quantized_model = quantize(model=fp16_model, calibration_dataset=dataset, subset_size=10)
                quantized_model.set_rt_info(model_type, ["model_info", "model_type"])
                save_openvino_model(quantized_model, model_path, compress_fp16=False)
            except ImportError:
                logger.warning("NNCF not available, saving as FP16 instead of INT8")
                save_openvino_model(ov_model, model_path, compress_fp16=True)
        else:
            logger.debug(f"Unsupported model_precision: {model_precision}")
            return None

        # Clean up temporary files
        shutil.rmtree(str(converted_path))

        logger.info(f"✓ Exported {model_id} to {model_path}")
        return model_path

    except Exception as e:
        logger.error(f"Failed to export YOLO model {model_id}: {e}", exc_info=True)
        return None
    finally:
        # Always restore original directory
        os.chdir(original_cwd)

        # Clean up temporary export artifacts in temp directory
        try:
            for item in temp_dir.iterdir():
                if item.is_dir() and item.name.endswith("_openvino_model"):
                    shutil.rmtree(item)
                    logger.debug(f"Cleaned up temp export dir: {item.name}")
                elif item.is_file() and item.suffix in [".pt", ".onnx"]:
                    item.unlink()
                    logger.debug(f"Cleaned up temp file: {item.name}")
        except Exception as cleanup_err:
            logger.warning(f"Failed to clean up temp export files: {cleanup_err}")


def download_test_image(image_name: str = "car.png", output_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Download test image from OpenVINO storage.

    Args:
        image_name: Name of the image file to download (default: 'car.png')
        output_dir: Directory to save the image (default: esq_data/data/vertical/metro/images)

    Returns:
        Path to the downloaded image, or None if failed

    Example:
        >>> image_path = download_test_image("car.png", "esq_data/data/vertical/metro/images")
        >>> print(image_path)
        PosixPath('esq_data/data/vertical/metro/images/car.png')
    """
    image_url = f"{OPENVINO_STORAGE_BASE}/data/test_data/images/{image_name}"

    # Use CORE_DATA_DIR structure: esq_data/data/vertical/metro/images
    if output_dir is None:
        core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
        core_data_dir = "".join(c for c in core_data_dir_tainted)
        output_dir = Path(core_data_dir) / "data" / "vertical" / "metro" / "images"

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    image_path = target_dir / image_name

    if image_path.exists():
        logger.info(f"Test image {image_name} already exists at: {image_path}")
        return image_path

    logger.info(f"Downloading test image: {image_name}")
    try:
        response = requests.get(image_url, timeout=300)
        response.raise_for_status()
        content = response.content

        # Validate we got image data, not HTML error page
        if len(content) < 1000 and b"<!DOCTYPE html>" in content[:500]:
            raise RuntimeError(f"Received HTML page instead of image from {image_url}")

        with open(image_path, "wb") as out_file:
            out_file.write(content)

        logger.info(f"Successfully downloaded {image_name} to: {image_path}")
        return image_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download test image {image_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading {image_name}: {e}", exc_info=True)
        return None
