# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Ultralytics model utilities for YOLO models.

This module handles downloading, exporting, and preparing YOLO models
from Ultralytics for OpenVINO format.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import openvino as ov

from .common import construct_export_path_components, save_openvino_model

logger = logging.getLogger(__name__)

YOLO_MODELS = {
    "yolov8n": "YOLOv8",
    "yolov8s": "YOLOv8",
    "yolov8m": "YOLOv8",
    "yolov8l": "YOLOv8",
    "yolov8x": "YOLOv8",
    "yolov8n-obb": "YOLOv8-OBB",
    "yolov8s-obb": "YOLOv8-OBB",
    "yolov8m-obb": "YOLOv8-OBB",
    "yolov8l-obb": "YOLOv8-OBB",
    "yolov8x-obb": "YOLOv8-OBB",
    "yolov9t": "YOLOv8",
    "yolov9s": "YOLOv8",
    "yolov9m": "YOLOv8",
    "yolov9c": "YOLOv8",
    "yolov9e": "YOLOv8",
    "yolov10n": "yolo_v10",
    "yolov10s": "yolo_v10",
    "yolov10m": "yolo_v10",
    "yolov10b": "yolo_v10",
    "yolov10l": "yolo_v10",
    "yolov10x": "yolo_v10",
    "yolo11n": "yolo_v11",
    "yolo11s": "yolo_v11",
    "yolo11m": "yolo_v11",
    "yolo11l": "yolo_v11",
    "yolo11x": "yolo_v11",
    "yolo11n-obb": "yolo_v11_obb",
    "yolo11s-obb": "yolo_v11_obb",
    "yolo11ms-obb": "yolo_v11_obb",
    "yolo11l-obb": "yolo_v11_obb",
    "yolo11x-obb": "yolo_v11_obb",
    "yolo11n-pose": "yolo_v11_pose",
    "yolo11s-pose": "yolo_v11_pose",
    "yolo11m-pose": "yolo_v11_pose",
    "yolo11l-pose": "yolo_v11_pose",
    "yolo11x-pose": "yolo_v11_pose",
}


def download_yolo_model(model_id: str, models_dir: Optional[str] = None) -> Optional[Path]:
    """
    Download YOLO model weights using Ultralytics.

    Args:
        model_id: YOLO model identifier
        models_dir: Directory to save models

    Returns:
        Path to the downloaded weights (.pt file) or None if failed
    """
    if model_id not in YOLO_MODELS:
        logger.error(f"Invalid model name '{model_id}'.")
        logger.debug(f"Available models: {', '.join(YOLO_MODELS.keys())}")
        return None

    if models_dir:
        pt_path = (Path(models_dir) / model_id / f"{model_id}.pt").resolve()
    else:
        pt_path = Path(f"{model_id}.pt").resolve()

    if pt_path.exists():
        logger.debug(f"YOLO model weights already downloaded: {pt_path}")
        return pt_path

    logger.debug(f"Downloading YOLO weights for: {model_id}")
    try:
        from ultralytics import YOLO

        model = YOLO(model_id)
        model.info()

        # Find where Ultralytics saved the weights
        downloaded_pt_path = Path(model.model.pt_path if hasattr(model.model, "pt_path") else pt_path)

        # Move to target directory if needed
        if models_dir and downloaded_pt_path != pt_path:
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded_pt_path), str(pt_path))

        logger.info(f"YOLO model weights downloaded: {pt_path}")
        return pt_path

    except Exception as e:
        logger.error(f"Failed to download YOLO model {model_id}: {e}")
        return None


def export_yolo_model(
    model_id: str,
    models_dir: str = "models",
    model_precision: str = "fp16",
    export_args: Optional[dict] = None,
) -> bool:
    """
    Convert YOLO model weights to OpenVINO format.

    Args:
        model_id: ID of the YOLO model to export
        models_dir: Directory to save the exported model
        model_precision: Precision of the exported model (fp32, fp16, int8)
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
        bool: True if export was successful, False otherwise
    """
    if model_id not in YOLO_MODELS:
        logger.debug(f"Error: Invalid model name '{model_id}'.")
        logger.debug(f"Available models: {', '.join(YOLO_MODELS.keys())}")
        return False

    model_type = YOLO_MODELS[model_id]
    base_dir = (Path(models_dir) / model_id).resolve()

    path_components = construct_export_path_components(export_args)
    if path_components:
        subdir = "_".join(path_components)
        model_dir = base_dir / model_precision / subdir
    else:
        model_dir = base_dir / model_precision

    model_path = model_dir / f"{model_id}.xml"
    pt_path = base_dir / f"{model_id}.pt"

    # Check if model is already exported
    if model_path.exists():
        logger.info(f" ✓ Model {model_id} already exported to OpenVINO {model_precision} format: {model_path}")
        return True

    # Verify model weights exist
    if not pt_path.exists():
        logger.error(f"YOLO model weights not found: {pt_path}. Please download first.")
        return False

    logger.debug(f"Exporting {model_id} to OpenVINO {model_precision} format...")

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

    try:
        from ultralytics import YOLO

        # Convert to OpenVINO IR with custom export arguments
        model = YOLO(str(pt_path), task="detect")
        converted_path = Path(model.export(**export_kwargs)).resolve()

        core = ov.Core()
        ov_model = core.read_model(model=os.path.join(converted_path, f"{model_id}.xml"))

        # Reshape model if dynamic shape was used in export
        # This is critical for proper quantization and inference optimization
        # Dynamic shapes are exported as [?, 3, 640, 640] but need to be fixed to [1, 3, 640, 640]
        # for quantization and optimal performance
        if export_args and export_args.get("dynamic", False):
            # Get the input shape dimensions from export args or use defaults
            batch_size = export_args.get("batch", 1)
            imgsz = export_args.get("imgsz", 640)

            # Handle both int and tuple formats for imgsz
            if isinstance(imgsz, int):
                height, width = imgsz, imgsz
            else:
                height, width = imgsz[0], imgsz[1]

            # Reshape to concrete static shape for quantization and inference
            target_shape = [batch_size, 3, height, width]
            logger.debug(f"Reshaping dynamic model to static shape: {target_shape}")
            ov_model.reshape(target_shape)

        ov_model.set_rt_info(model_type, ["model_info", "model_type"])

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

            # For INT8 quantization, we need to work with the reshaped model
            # Re-read the model to ensure reshape is applied
            fp16_model = core.read_model(model=os.path.join(converted_path, f"{model_id}.xml"))

            # Apply reshape to the model used for quantization as well
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
            return False

        # Clean up temporary files
        shutil.rmtree(str(converted_path))

        logger.debug(f"Exported {model_id} to {model_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export YOLO model {model_id}: {e}")
        return False
