# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Model setup utilities for downloading and preparing models.

This module provides utilities for downloading models from Hugging Face Hub,
setting up YOLO models, and preparing models for inference.
"""

import json
import logging
import multiprocessing
import os
import queue
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import openvino as ov

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


def _run_export_model_worker(
    result_queue: multiprocessing.Queue,
    source_model: str,
    model_name: str,
    model_repository_path: str,
    precision: str,
    task_parameters: Dict[str, Any],
    config_file_path: str,
) -> None:
    """Worker process that runs model export and reports status via queue."""
    try:
        from esq.utils.models.export_model import export_text_generation_model

        export_text_generation_model(
            source_model=source_model,
            model_name=model_name,
            model_repository_path=model_repository_path,
            precision=precision,
            task_parameters=task_parameters,
            config_file_path=config_file_path,
            overwrite_models=False,
        )

        result_queue.put({"success": True})
    except Exception as exc:  # pragma: no cover - error path
        result_queue.put(
            {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def download_model(model_id_or_path: str, revision: str = "main", ignore_patterns: Optional[List[str]] = None) -> str:
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

        local_path = snapshot_download(
            model_id_or_path, revision=revision, ignore_patterns=ignore_patterns or ["*.pth"]
        )
        logger.info(f"Model downloaded to: {local_path}")
        return local_path
    except KeyboardInterrupt:
        logger.error("Model download interrupted by user (KeyboardInterrupt).")
        raise RuntimeError("Model download interrupted by user")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise RuntimeError(f"Failed to download model: {e}")


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


def convert_yolo_to_openvino(
    pt_path: Path, output_dir: Optional[Path] = None, precision: str = "fp16"
) -> Optional[Path]:
    """
    Convert YOLO PyTorch model to OpenVINO format.

    Args:
        pt_path: Path to YOLO .pt file
        output_dir: Output directory for converted model
        precision: Model precision (fp16, fp32, int8)

    Returns:
        Path to converted OpenVINO model or None if failed
    """
    if not pt_path.exists():
        logger.error(f"YOLO weights file not found: {pt_path}")
        return None

    if output_dir is None:
        output_dir = pt_path.parent / "openvino"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO

        model_id = pt_path.stem
        final_xml_path = output_dir / f"{model_id}.xml"

        # Check if model is already converted
        if final_xml_path.exists():
            logger.info(f"YOLO model already converted: {final_xml_path}")
            return final_xml_path

        model = YOLO(str(pt_path))

        # Export to OpenVINO - Ultralytics creates a directory with model files
        export_path = model.export(format="openvino", imgsz=640)

        # export_path is usually the path to the exported directory or XML file
        # Let's find the actual XML file
        if isinstance(export_path, str):
            export_path = Path(export_path)

        # If export_path is a directory, find the XML file in it
        if export_path.is_dir():
            xml_files = list(export_path.glob("*.xml"))
            if not xml_files:
                logger.error(f"No XML files found in export directory: {export_path}")
                return None
            source_xml_path = xml_files[0]
        elif export_path.suffix == ".xml":
            source_xml_path = export_path
        else:
            # Try to find XML file based on model name
            possible_xml = export_path.parent / f"{model_id}.xml"
            if possible_xml.exists():
                source_xml_path = possible_xml
            else:
                logger.error(f"Cannot find XML file for exported model: {export_path}")
                return None

        # Read the model and apply precision settings
        core = ov.Core()
        ov_model = core.read_model(str(source_xml_path))

        # Set model type metadata
        if model_id in YOLO_MODELS:
            model_type = YOLO_MODELS[model_id]
            ov_model.set_rt_info(model_type, ["model_info", "model_type"])

        # Save model with specified precision
        if precision == "fp32":
            ov.save_model(ov_model, str(final_xml_path), compress_to_fp16=False)
        elif precision == "fp16":
            ov.save_model(ov_model, str(final_xml_path), compress_to_fp16=True)
        elif precision == "int8":
            # Implement proper int8 quantization with calibration data
            def data_gen():
                for _ in range(10):
                    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
                    yield {input_key: dummy}

            input_key = ov_model.inputs[0].get_any_name()

            try:
                from nncf import Dataset, quantize

                dataset = Dataset(data_gen())
                quantized_model = quantize(model=ov_model, calibration_dataset=dataset, subset_size=10)
                quantized_model.set_rt_info(
                    model_type if model_id in YOLO_MODELS else "YOLO", ["model_info", "model_type"]
                )
                ov.save_model(quantized_model, str(final_xml_path))
            except ImportError:
                logger.warning("NNCF not available, saving as FP16 instead of INT8")
                ov.save_model(ov_model, str(final_xml_path), compress_to_fp16=True)
        else:
            logger.error(f"Unsupported precision: {precision}")
            return None

        # Clean up the original export directory if it's different from our target
        if source_xml_path.parent != output_dir:
            try:
                shutil.rmtree(str(source_xml_path.parent))
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary export directory: {e}")

        if final_xml_path.exists():
            logger.info(f"YOLO model converted to OpenVINO {precision}: {final_xml_path}")
            return final_xml_path
        else:
            logger.error("OpenVINO conversion completed but final XML file not found")
            return None

    except Exception as e:
        logger.error(f"Failed to convert YOLO model to OpenVINO: {e}")
        return None


def export_yolo_model(model_id: str, models_dir: str = "models", model_precision: str = "fp16") -> bool:
    """
    Convert YOLO model weights to OpenVINO format.
    Args:
        model_id (str): ID of the YOLO model to export.
        models_dir (str): Directory to save the exported model.
        model_precision (str): Precision of the exported model (fp32, fp16, int8).
    Returns:
        bool: True if export was successful, False otherwise.
    """
    if model_id not in YOLO_MODELS:
        logger.debug(f"Error: Invalid model name '{model_id}'.")
        logger.debug(f"Available models: {', '.join(YOLO_MODELS.keys())}")
        return False

    model_type = YOLO_MODELS[model_id]
    base_dir = (Path(models_dir) / model_id).resolve()
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

    try:
        from ultralytics import YOLO

        # Convert to OpenVINO IR
        model = YOLO(str(pt_path), task="detect")
        converted_path = Path(model.export(format="openvino")).resolve()

        core = ov.Core()
        ov_model = core.read_model(model=os.path.join(converted_path, f"{model_id}.xml"))
        ov_model.set_rt_info(model_type, ["model_info", "model_type"])

        # Save model in requested precision
        if model_precision == "fp32":
            ov.save_model(ov_model, str(model_path), compress_to_fp16=False)
        elif model_precision == "fp16":
            ov.save_model(ov_model, str(model_path), compress_to_fp16=True)
        elif model_precision == "int8":

            def data_gen():
                for _ in range(10):
                    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
                    yield {input_key: dummy}

            fp16_model = core.read_model(model=os.path.join(converted_path, f"{model_id}.xml"))
            input_key = fp16_model.inputs[0].get_any_name()

            try:
                from nncf import Dataset, quantize

                dataset = Dataset(data_gen())
                quantized_model = quantize(model=fp16_model, calibration_dataset=dataset, subset_size=10)
                quantized_model.set_rt_info(model_type, ["model_info", "model_type"])
                ov.save_model(quantized_model, str(model_path))
            except ImportError:
                logger.warning("NNCF not available, saving as FP16 instead of INT8")
                ov.save_model(ov_model, str(model_path), compress_to_fp16=True)
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


def quantize_model(
    model_path: Path, calibration_data: Optional[Any] = None, output_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Quantize an OpenVINO model using NNCF.

    Args:
        model_path: Path to OpenVINO model
        calibration_data: Calibration dataset
        output_path: Output path for quantized model

    Returns:
        Path to quantized model or None if failed
    """
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None

    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}_quantized.xml"
    else:
        output_path = Path(output_path)

    try:
        from nncf import Dataset, quantize

        # Load the model
        core = ov.Core()
        model = core.read_model(str(model_path))

        if calibration_data is not None:
            # Create NNCF dataset
            dataset = Dataset(calibration_data)

            # Quantize the model
            quantized_model = quantize(model, dataset)
        else:
            # Use basic quantization without calibration
            quantized_model = quantize(model)

        # Save quantized model
        ov.save_model(quantized_model, str(output_path))

        logger.info(f"Model quantized and saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to quantize model: {e}")
        return None


def verify_model(model_path: Path) -> Dict[str, Any]:
    """
    Verify an OpenVINO model by loading it and checking basic properties.

    Args:
        model_path: Path to OpenVINO model

    Returns:
        Dict containing verification results
    """
    result = {"valid": False, "path": str(model_path), "inputs": [], "outputs": [], "error": None}

    try:
        core = ov.Core()
        model = core.read_model(str(model_path))

        # Get input information
        for input_layer in model.inputs:
            result["inputs"].append(
                {
                    "name": input_layer.get_any_name(),
                    "shape": list(input_layer.shape),
                    "type": str(input_layer.element_type),
                }
            )

        # Get output information
        for output_layer in model.outputs:
            result["outputs"].append(
                {
                    "name": output_layer.get_any_name(),
                    "shape": list(output_layer.shape),
                    "type": str(output_layer.element_type),
                }
            )

        result["valid"] = True
        logger.info(f"Model verification successful: {model_path}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Model verification failed: {e}")

    return result


def get_model_info(model_path: Path) -> Dict[str, Any]:
    """
    Get detailed information about a model.

    Args:
        model_path: Path to model

    Returns:
        Dict containing model information
    """
    info = {"path": str(model_path), "exists": model_path.exists(), "type": "unknown", "size_mb": 0, "files": []}

    if not info["exists"]:
        return info

    # Determine model type
    if model_path.suffix == ".xml":
        info["type"] = "openvino"
    elif model_path.suffix == ".pt":
        info["type"] = "pytorch"
    elif model_path.suffix == ".onnx":
        info["type"] = "onnx"

    # Get file information
    if model_path.is_file():
        info["size_mb"] = model_path.stat().st_size / (1024 * 1024)
        info["files"] = [model_path.name]

        # For OpenVINO models, include .bin file if it exists
        if model_path.suffix == ".xml":
            bin_path = model_path.with_suffix(".bin")
            if bin_path.exists():
                info["files"].append(bin_path.name)
                info["size_mb"] += bin_path.stat().st_size / (1024 * 1024)
    elif model_path.is_dir():
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                info["files"].append(str(file_path.relative_to(model_path)))
                info["size_mb"] += file_path.stat().st_size / (1024 * 1024)

    info["size_mb"] = round(info["size_mb"], 2)
    return info


def setup_model(
    model_id: str, model_type: str = "auto", models_dir: Optional[str] = None, convert_to_openvino: bool = True
) -> Optional[Path]:
    """
    Set up a model by downloading and optionally converting it.

    Args:
        model_id: Model identifier
        model_type: Type of model (auto, yolo, huggingface)
        models_dir: Directory to save models
        convert_to_openvino: Whether to convert to OpenVINO format

    Returns:
        Path to the set up model or None if failed
    """
    if models_dir:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Auto-detect model type if not specified
        if model_type == "auto":
            if model_id in YOLO_MODELS:
                model_type = "yolo"
            else:
                model_type = "huggingface"

        if model_type == "yolo":
            # Download YOLO model
            pt_path = download_yolo_model(model_id, str(models_dir) if models_dir else None)
            if not pt_path:
                return None

            if convert_to_openvino:
                # Convert to OpenVINO
                output_dir = models_dir / model_id / "openvino" if models_dir else Path(model_id) / "openvino"
                xml_path = convert_yolo_to_openvino(pt_path, output_dir)
                return xml_path if xml_path else pt_path
            else:
                return pt_path

        elif model_type == "huggingface":
            # Download from Hugging Face
            local_path = download_model(model_id)
            return Path(local_path)

        else:
            logger.error(f"Unknown model type: {model_type}")
            return None

    except Exception as e:
        logger.error(f"Failed to set up model {model_id}: {e}")
        return None


def cleanup_model(model_path: Path, include_source: bool = False) -> bool:
    """
    Clean up model files and directories.

    Args:
        model_path: Path to model to clean up
        include_source: Whether to also remove source files (e.g., .pt files)

    Returns:
        bool: True if cleanup was successful
    """
    try:
        if model_path.is_file():
            model_path.unlink()
            logger.info(f"Removed model file: {model_path}")

            # Remove associated files for OpenVINO models
            if model_path.suffix == ".xml":
                bin_path = model_path.with_suffix(".bin")
                if bin_path.exists():
                    bin_path.unlink()
                    logger.info(f"Removed model binary: {bin_path}")

        elif model_path.is_dir():
            shutil.rmtree(model_path)
            logger.info(f"Removed model directory: {model_path}")

        if include_source:
            # Remove source files in parent directory
            parent_dir = model_path.parent
            for pattern in ["*.pt", "*.pth", "*.onnx"]:
                for file_path in parent_dir.glob(pattern):
                    file_path.unlink()
                    logger.info(f"Removed source file: {file_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to cleanup model {model_path}: {e}")
        return False


def list_available_models(models_dir: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    List all available models in a directory.

    Args:
        models_dir: Directory to scan for models

    Returns:
        Dict mapping model types to lists of model information
    """
    if models_dir:
        models_dir = Path(models_dir)
    else:
        models_dir = Path.cwd()

    if not models_dir.exists():
        return {}

    models = {"openvino": [], "pytorch": [], "onnx": [], "directories": []}

    for item in models_dir.rglob("*"):
        if item.is_file():
            if item.suffix == ".xml":
                models["openvino"].append(get_model_info(item))
            elif item.suffix in [".pt", ".pth"]:
                models["pytorch"].append(get_model_info(item))
            elif item.suffix == ".onnx":
                models["onnx"].append(get_model_info(item))
        elif item.is_dir() and any(item.glob("*.xml")):
            models["directories"].append(get_model_info(item))

    return models


def export_ovms_model(
    model_id_or_path,
    models_dir,
    model_precision,
    device_id,
    configs=None,
    export_timeout=1800,
) -> tuple[bool, float, dict, str]:
    """
    Export model to OpenVINO Model Server format.

    For consistency with pre-quantized models, this function uses safe model names
    (replacing / with _) for the MediaPipe servable configuration. This ensures
    both pre-quantized and on-demand quantized models are configured identically.

    Models with different quantization parameters are exported to separate directories
    to avoid conflicts and ensure correct configuration.

    Args:
        model_id_or_path: Model identifier or path
        models_dir: Directory for models
        model_precision: Model precision (int4, int8, fp16, etc.)
        device_id: Target device
        configs: Optional configuration dict with quantization parameters
        export_timeout: Maximum time in seconds for model export (default: 1800)

    Returns:
        tuple[bool, float, dict, str]: (success_status, export_duration_seconds, quantization_config, actual_model_name)
            - success_status: True if export succeeded
            - export_duration_seconds: Time taken for export in seconds
            - quantization_config: Dict with quantization parameters used (empty if no quantization)
            - actual_model_name: Actual model name used in OVMS config (with quantization suffix if applicable)
    """
    import hashlib
    import time

    export_start_time = time.time()
    quantization_config = {}  # Track quantization parameters used

    config_path = os.path.join(models_dir, "config_all.json")
    if os.path.exists(config_path):
        os.remove(config_path)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)

    try:
        # Use safe model name (replace / with _) for directory structure and MediaPipe config
        # This matches the behavior of pre-quantized models
        model_safe_name = model_id_or_path.replace("/", "_")

        logger.info(f"Exporting model {model_id_or_path} to OpenVINO Model Server format at {models_dir}")
        logger.info(f"Using safe model name: {model_safe_name}")

        # Determine default quantization parameters
        def get_default_quantization_params(model_id):
            """
            Get default quantization parameters for all models.

            Returns dict with individual quantization parameters.
            Uses simple defaults for all models - can be overridden via profile config.

            Default configuration:
            - Symmetric quantization (sym=True)
            - Per-column quantization (group_size=-1)
            - 100% INT4 (ratio=1.0)
            - No AWQ or scale estimation (simpler/faster)

            Note: Dataset parameter should be specified in profile config when needed.

            References:
            - Hugging Face Optimum Intel: https://huggingface.co/docs/optimum/intel/openvino/optimization
            - Per-column quantization (group_size=-1) provides good baseline quality
            - Full INT4 (ratio=1.0) maximizes compression and speed
            """
            # Simple default parameters for all models
            # Can be overridden at profile config level for model-specific tuning
            return {"sym": True, "group_size": -1, "ratio": 1.0}

        # Get default parameters
        default_quant_params = get_default_quantization_params(model_id_or_path)

        # Allow override from configs (individual parameters or consolidated string)
        # Priority: configs > defaults
        quant_config = {}

        # Check if consolidated parameter string is provided
        if configs and "quantization_params" in configs:
            # Use provided string directly (backwards compatibility)
            quant_config["extra_quantization_params"] = configs["quantization_params"]
            logger.info(f"Using configured quantization parameters: {configs['quantization_params']}")

            # Store consolidated string in quantization_config for metadata
            quantization_config["quantization_params"] = configs["quantization_params"]
        else:
            # Build from individual parameters (allows fine-grained control)
            quant_config["sym"] = configs.get("quant_sym", default_quant_params.get("sym", False))
            quant_config["group_size"] = configs.get("quant_group_size", default_quant_params.get("group_size"))
            quant_config["ratio"] = configs.get("quant_ratio", default_quant_params.get("ratio"))
            quant_config["awq"] = configs.get("quant_awq", default_quant_params.get("awq", False))
            quant_config["scale_estimation"] = configs.get(
                "quant_scale_estimation", default_quant_params.get("scale_estimation", False)
            )

            # Only include dataset if explicitly specified in configs (not in defaults)
            if "quant_dataset" in configs:
                quant_config["dataset"] = configs["quant_dataset"]

            # Store individual parameters in quantization_config for metadata
            quantization_config["sym"] = quant_config["sym"]
            quantization_config["group_size"] = quant_config["group_size"]
            quantization_config["ratio"] = quant_config["ratio"]
            quantization_config["awq"] = quant_config["awq"]
            quantization_config["scale_estimation"] = quant_config["scale_estimation"]

            # Only store dataset if it was specified
            if "dataset" in quant_config:
                quantization_config["dataset"] = quant_config["dataset"]

            # Build parameter string for export
            param_parts = []
            if quant_config["sym"]:
                param_parts.append("--sym")
            if quant_config["group_size"] is not None:
                param_parts.append(f"--group-size {quant_config['group_size']}")
            if quant_config["ratio"] is not None:
                param_parts.append(f"--ratio {quant_config['ratio']}")
            if quant_config["awq"]:
                param_parts.append("--awq")
            if quant_config["scale_estimation"]:
                param_parts.append("--scale-estimation")
            # Only add dataset if it exists in config
            if "dataset" in quant_config and quant_config["dataset"]:
                param_parts.append(f"--dataset {quant_config['dataset']}")

            quant_config["extra_quantization_params"] = " ".join(param_parts) if param_parts else ""

            logger.info("=" * 80)
            logger.info("QUANTIZATION CONFIGURATION:")
            logger.info(f"  Model: {model_id_or_path}")
            logger.info(f"  Symmetric: {quant_config['sym']}")
            logger.info(f"  Group Size: {quant_config['group_size']}")
            logger.info(f"  Ratio (int4/int8): {quant_config['ratio']}")
            logger.info(f"  AWQ: {quant_config['awq']}")
            logger.info(f"  Scale Estimation: {quant_config['scale_estimation']}")
            logger.info(f"  Dataset: {quant_config.get('dataset', 'Not specified')}")
            logger.info(f"  Full Parameters: {quant_config['extra_quantization_params']}")
            logger.info("=" * 80)

        # Generate unique suffix for directory based on quantization parameters
        # This ensures models with different quantization configs are exported to separate folders
        quant_suffix = ""
        if quant_config.get("extra_quantization_params"):
            # Create a hash of the quantization parameters for unique identification
            quant_hash = hashlib.sha256(quant_config["extra_quantization_params"].encode()).hexdigest()[:8]
            quant_suffix = f"_q{quant_hash}"
            logger.info(f"Quantization suffix: {quant_suffix} (based on parameters hash)")

        # Update model safe name to include quantization suffix
        # This ensures different quantization configs export to different directories
        model_safe_name_with_quant = f"{model_safe_name}{quant_suffix}"
        logger.info(f"Full model directory name: {model_safe_name_with_quant}")

        # Note: The target device is handled by the export_text_generation_model function
        # It updates the servable configuration dynamically without re-exporting the model
        # So we don't need separate directories per device

        # Determine pipeline_type for HETERO devices
        # HETERO requires pipeline_type to be set for MODEL_DISTRIBUTION_POLICY
        pipeline_type = None
        if device_id.upper().startswith("HETERO:"):
            # For LLM text generation with HETERO, use "LM_CB" (Continuous Batching) pipeline type
            pipeline_type = "LM_CB"
            logger.info(f"HETERO device detected, setting pipeline_type to: {pipeline_type}")

        task_parameters = {
            "target_device": device_id,
            "pipeline_type": pipeline_type,  # Required for HETERO devices
            "kv_cache_precision": None,
            "extra_quantization_params": quant_config["extra_quantization_params"],
            "enable_prefix_caching": True,
            "dynamic_split_fuse": True,
            "max_num_batched_tokens": None,
            "max_num_seqs": "2048",  # Set 256 if want to align with pre-quantized models
            "cache_size": 2,  # Reduced from 2GB to 1GB for HETERO stability if needed
            "draft_source_model": None,
            "draft_model_name": None,
            "max_prompt_len": None,
            "prompt_lookup_decoding": False,  # Add missing parameter
        }

        logger.info(f"Running model export with timeout of {export_timeout} seconds...")
        logger.info("Export progress will be shown below:")
        logger.info("=" * 80)

        export_queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_run_export_model_worker,
            args=(
                export_queue,
                model_id_or_path,
                model_safe_name_with_quant,
                models_dir,
                model_precision,
                task_parameters,
                config_path,
            ),
        )

        try:
            process.start()
            process.join(timeout=export_timeout)

            if process.is_alive():
                logger.error(f"Model export timed out after {export_timeout} seconds")
                process.terminate()
                process.join()
                raise RuntimeError(f"Model export exceeded timeout of {export_timeout} seconds.")

            export_result = None
            try:
                export_result = export_queue.get_nowait()
            except queue.Empty:
                export_result = None

            if export_result and not export_result.get("success", False):
                error_message = export_result.get("error", "Unknown error")
                logger.error(f"Model export reported error: {error_message}")
                if export_result.get("traceback"):
                    logger.error(export_result["traceback"])  # Detailed traceback for debugging
                raise RuntimeError(f"Model export failed: {error_message}")

            if process.exitcode not in (0, None):
                logger.error(f"Model export process exited with code {process.exitcode}")
                raise RuntimeError(f"Model export failed with exit code {process.exitcode}")

            logger.info("=" * 80)
            logger.info(f"Model exported successfully to {models_dir}")
            logger.info(f"MediaPipe servable registered as: {model_safe_name_with_quant}")

        finally:
            export_queue.close()
            export_queue.join_thread()

        export_duration = time.time() - export_start_time
        logger.debug(f"Model export completed in {export_duration:.2f} seconds")

        return True, export_duration, quantization_config, model_safe_name_with_quant
    except Exception as e:
        logger.error(f"OVMS model export failed with error: {e}")
        export_duration = time.time() - export_start_time
        # Include detailed error message in the exception
        raise RuntimeError(f"OVMS model export failed: {str(e)}") from e


def download_and_setup_prequantized_ovms_model(
    model_id: str,
    models_dir: str,
    device_id: str,
) -> bool:
    """
    Download and setup a pre-quantized OpenVINO model from HuggingFace for OVMS.

    This function handles models that are already in OpenVINO IR format with quantization
    applied (e.g., OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov).

    OVMS expects models to be organized in versioned directories:
    models/
      └── model_name/
          └── 1/  (version directory)
              ├── openvino_model.xml
              └── openvino_model.bin

    Args:
        model_id: HuggingFace model ID (e.g., "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov")
        models_dir: Base directory for models
        device_id: Target device ID (CPU, GPU, etc.)

    Returns:
        bool: True if successful, raises RuntimeError otherwise
    """

    from huggingface_hub import snapshot_download

    config_path = os.path.join(models_dir, "config_all.json")

    try:
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Download model from HuggingFace Hub
        # The model will be downloaded to a temporary location first
        logger.info(f"Downloading pre-quantized model {model_id} from HuggingFace Hub to {models_dir}")

        # Use a safe model name for the directory (replace / with _)
        model_safe_name = model_id.replace("/", "_")
        model_base_path = os.path.join(models_dir, model_safe_name)
        model_version_path = os.path.join(model_base_path, "1")  # OVMS expects version subdirectories

        # Check if model is already downloaded and organized
        model_xml_path = os.path.join(model_version_path, "openvino_model.xml")
        if os.path.exists(model_version_path) and os.path.exists(model_xml_path):
            logger.info(f"Model {model_id} already exists at {model_version_path}, skipping download")
        else:
            # Download the model to a temporary directory first
            temp_download_dir = os.path.join(models_dir, f".tmp_{model_safe_name}")

            logger.info(f"Downloading model {model_id} to temporary location: {temp_download_dir}")
            snapshot_download(
                repo_id=model_id,
                local_dir=temp_download_dir,
                local_dir_use_symlinks=False,
            )
            logger.info(f"Model {model_id} downloaded successfully")

            # Create the versioned directory structure for OVMS
            os.makedirs(model_version_path, exist_ok=True)

            # Move model files to the version directory
            logger.info(f"Organizing model files into OVMS structure at {model_version_path}")
            for file in os.listdir(temp_download_dir):
                src_file = os.path.join(temp_download_dir, file)
                dst_file = os.path.join(model_version_path, file)

                # Skip .cache and .git directories
                if file.startswith("."):
                    if os.path.isdir(src_file):
                        shutil.rmtree(src_file)
                    else:
                        os.remove(src_file)
                    continue

                # Move the file or directory
                if os.path.exists(dst_file):
                    if os.path.isdir(dst_file):
                        shutil.rmtree(dst_file)
                    else:
                        os.remove(dst_file)
                shutil.move(src_file, dst_file)

            # Remove temporary download directory
            if os.path.exists(temp_download_dir):
                shutil.rmtree(temp_download_dir)

            logger.info(f"Model files organized successfully in {model_version_path}")

        # Verify that the model files exist
        required_files = ["openvino_model.xml", "openvino_model.bin"]
        for file in required_files:
            file_path = os.path.join(model_version_path, file)
            if not os.path.exists(file_path):
                raise RuntimeError(f"Required model file {file} not found at {file_path}")

        logger.info(f"All required model files found for {model_id}")

        # Create MediaPipe graph for LLM serving with OpenAI-compatible API
        # Pre-quantized models already have openvino_model.xml in version directory
        # We need to create graph.pbtxt that points to the version subdirectory
        import jinja2

        # Load the text generation graph template
        from .export_model import text_generation_graph_template

        # Plugin configuration for performance
        plugin_config = {
            "PERFORMANCE_HINT": "LATENCY",
        }
        plugin_config_str = json.dumps(plugin_config)

        # Task parameters for graph generation
        task_parameters = {
            "target_device": device_id.upper(),
            "plugin_config": plugin_config_str,
            "enable_prefix_caching": True,
            "cache_size": 2,
            "max_num_seqs": "256",
            "dynamic_split_fuse": True,
            "pipeline_type": None,  # Let OVMS auto-detect
        }

        # Create graph.pbtxt in the model base directory
        graph_dir = model_base_path
        os.makedirs(graph_dir, exist_ok=True)

        # Render the graph template
        # models_path should point to version directory (1/)
        gtemplate = jinja2.Environment(loader=jinja2.BaseLoader, autoescape=True).from_string(
            text_generation_graph_template
        )

        graph_content = gtemplate.render(
            model_path="./1",  # Relative path to version directory
            draft_model_dir_name=None,
            **task_parameters,
        )

        graph_path = os.path.join(graph_dir, "graph.pbtxt")
        with open(graph_path, "w") as f:
            f.write(graph_content)

        logger.info(f"Created MediaPipe graph at {graph_path}")

        # Create OVMS config file with MediaPipe servable
        # For LLM serving, we use mediapipe_config_list instead of model_config_list
        from .export_model import add_servable_to_config

        add_servable_to_config(
            config_path,
            model_safe_name,  # Use safe name as servable name
            model_safe_name,  # Base path relative to config file
        )

        logger.info(f"OVMS MediaPipe config created at {config_path}")
        logger.info(f"Pre-quantized model {model_id} setup completed successfully")

        return True

    except Exception as e:
        logger.error(f"Failed to download and setup pre-quantized model {model_id}: {e}")
        raise RuntimeError(f"Pre-quantized model setup failed: {e}") from e


def download_and_prepare_model(model_config: dict, models_dir: str) -> dict:
    """
    Download and prepare a model based on its configuration.

    Args:
        model_config (dict): Model configuration containing:
            - id: Model ID
            - source: Source type ('zip', 'files', 'ultralytics')
            - format: Model format ('openvino', 'pt')
            - precision: Model precision ('fp16', 'fp32', 'int8')
            - url: URL(s) for download
            - sha256: SHA256 hash(es) for verification
        models_dir (str): Base directory for models

    Returns:
        dict: Result containing:
            - success: bool indicating if operation was successful
            - model_path: str path to the main model file
            - error: str error message if failed
            - files: list of downloaded/created files
    """
    result = {"success": False, "model_path": None, "error": None, "files": []}

    try:
        model_id = model_config.get("id", "")
        source = model_config.get("source", "")
        model_format = model_config.get("format", "")
        precision = model_config.get("precision", "fp16")
        url = model_config.get("url", "")
        sha256 = model_config.get("sha256", "")

        if not model_id:
            result["error"] = "Model ID is required"
            return result

        model_id_dir = os.path.join(models_dir, model_id)
        os.makedirs(model_id_dir, exist_ok=True)

        # Handle zip source with openvino format
        if source == "zip" and model_format == "openvino" and isinstance(url, str) and url.endswith(".zip"):
            from sysagent.utils.infrastructure import download_file, extract_zip_file

            zip_path = os.path.join(model_id_dir, f"{model_id}.zip")
            extracted_flag = os.path.join(model_id_dir, ".extracted")

            # Check if model is already extracted and ready
            if os.path.exists(extracted_flag):
                # Find existing model.xml file
                xml_path = None
                for root, dirs, files in os.walk(model_id_dir):
                    for file in files:
                        if file.endswith(".xml"):
                            xml_path = os.path.join(root, file)
                            break
                    if xml_path:
                        break

                if xml_path:
                    result["model_path"] = xml_path
                    result["success"] = True
                    logger.info(f" ✓ Model {model_id} already extracted and ready: {xml_path}")
                    return result

            # Download zip if not present
            if not os.path.exists(zip_path):
                logger.info(f"Downloading model zip from {url} for {model_id}")
                download_result = download_file(
                    url=url, target_path=zip_path, sha256sum=sha256 if isinstance(sha256, str) else None
                )
                result["files"].append(download_result["path"])

            # Extract zip if not already extracted
            if not os.path.exists(extracted_flag):
                logger.info(f"Extracting model zip for {model_id}")
                extract_zip_file(zip_path, model_id_dir)
                Path(extracted_flag).touch()
            else:
                logger.info(f" ✓ Model {model_id} already extracted")

            # Find model.xml file
            xml_path = None
            for root, dirs, files in os.walk(model_id_dir):
                for file in files:
                    if file.endswith(".xml"):
                        xml_path = os.path.join(root, file)
                        break
                if xml_path:
                    break

            if xml_path:
                result["model_path"] = xml_path
                result["success"] = True
                logger.info(f" ✓ Model {model_id} OpenVINO XML found: {xml_path}")
            else:
                result["error"] = f"No OpenVINO XML file found for {model_id} after extraction"

        # Handle files source with openvino format (separate xml/bin files)
        elif source == "files" and model_format == "openvino":
            from sysagent.utils.infrastructure import download_file

            precision_dir = os.path.join(model_id_dir, precision)
            os.makedirs(precision_dir, exist_ok=True)

            urls = url if isinstance(url, list) else []
            sha256_hashes = sha256 if isinstance(sha256, list) else []

            # First pass: determine original filenames from URLs
            expected_files = []
            for i, url_info in enumerate(urls):
                if isinstance(url_info, dict):
                    # Format: - xml: "url" or - bin: "url"
                    file_type = list(url_info.keys())[0]  # xml or bin
                    file_url = url_info[file_type]
                else:
                    # Fallback: use URL directly
                    file_url = url_info

                # Extract original filename from URL
                original_filename = os.path.basename(file_url.split("?")[0])  # Remove query params
                if original_filename:
                    expected_files.append(original_filename)

            # Check if all expected files already exist
            all_files_exist = True
            existing_files = []
            for filename in expected_files:
                file_path = os.path.join(precision_dir, filename)
                if os.path.exists(file_path):
                    existing_files.append(file_path)
                else:
                    all_files_exist = False
                    break

            if all_files_exist and existing_files:
                # Find the XML file to use as model_path
                xml_file = next((f for f in existing_files if f.endswith(".xml")), None)
                if xml_file:
                    result["model_path"] = xml_file
                    result["files"] = existing_files
                    result["success"] = True
                    logger.info(f" ✓ Model {model_id} files already exist: {', '.join(existing_files)}")
                    return result

            logger.info(f"Preparing model {model_id} with files source and precision {precision}")

            downloaded_files = []
            # Download each file (xml and bin)
            for i, url_info in enumerate(urls):
                if isinstance(url_info, dict):
                    # Format: - xml: "url" or - bin: "url"
                    file_type = list(url_info.keys())[0]  # xml or bin
                    file_url = url_info[file_type]
                    file_extension = file_type
                else:
                    # Fallback: determine file type from URL
                    file_url = url_info
                    if file_url.endswith(".xml"):
                        file_extension = "xml"
                    elif file_url.endswith(".bin"):
                        file_extension = "bin"
                    else:
                        logger.warning(f"Cannot determine file type for {file_url}, skipping")
                        continue

                # Use original filename from URL instead of model_id
                original_filename = os.path.basename(file_url.split("?")[0])  # Remove query params
                if not original_filename:
                    # Fallback to model_id if we can't extract filename
                    original_filename = f"{model_id}.{file_extension}"

                file_path = os.path.join(precision_dir, original_filename)

                # Get corresponding SHA256 if available
                file_sha256 = None
                if isinstance(sha256_hashes, list) and i < len(sha256_hashes):
                    sha256_info = sha256_hashes[i]
                    if isinstance(sha256_info, dict) and file_extension in sha256_info:
                        file_sha256 = sha256_info[file_extension]

                # Download file if not present
                if not os.path.exists(file_path):
                    logger.info(f"Downloading {file_extension.upper()} file from {file_url} for {model_id}")
                    download_result = download_file(url=file_url, target_path=file_path, sha256sum=file_sha256)
                    downloaded_files.append(download_result["path"])
                else:
                    logger.info(f" ✓ Model {model_id} {file_extension.upper()} file already exists")
                    downloaded_files.append(file_path)

            # Verify at least one XML file exists (for OpenVINO models)
            xml_files = [f for f in downloaded_files if f.endswith(".xml")]
            if xml_files:
                result["model_path"] = xml_files[0]  # Use first XML file found
                result["files"] = downloaded_files
                result["success"] = True
                logger.info(f" ✓ Model {model_id} OpenVINO files ready: {', '.join(downloaded_files)}")
            else:
                result["error"] = f"No XML file found for OpenVINO model {model_id}"

        # Handle ultralytics source (requires download and export)
        elif source == "ultralytics":
            model_path = os.path.join(model_id_dir, f"{model_id}.pt")
            precision_dir = os.path.join(model_id_dir, precision)
            xml_path = os.path.join(precision_dir, f"{model_id}.xml")

            # Check if model is already exported
            if os.path.exists(xml_path):
                result["model_path"] = xml_path
                result["success"] = True
                result["files"].append(xml_path)
                logger.info(f" ✓ Model {model_id} already exported and ready: {xml_path}")
                return result

            # Download model if not present
            if not os.path.exists(model_path):
                # Download model
                downloaded_path = download_yolo_model(model_id=model_id, models_dir=models_dir)
                if downloaded_path:
                    result["files"].append(str(downloaded_path))

                if not downloaded_path and not os.path.exists(model_path):
                    result["error"] = f"Failed to download YOLO model: {model_id}"
                    return result

            # Export model to OpenVINO format
            export_success = export_yolo_model(
                model_id=model_id,
                models_dir=models_dir,
                model_precision=precision,
            )

            if export_success:
                # Verify the exported model path exists
                if os.path.exists(xml_path):
                    result["model_path"] = xml_path
                    result["success"] = True
                    result["files"].append(xml_path)
                    logger.info(f" ✓ Model {model_id} exported and ready")
                else:
                    result["error"] = f"Exported model not found at expected path: {xml_path}"
            else:
                result["error"] = f"Failed to export YOLO model: {model_id}"

        else:
            result["error"] = f"Unsupported model source '{source}' with format '{model_format}'"

    except Exception as e:
        result["error"] = f"Error processing model {model_config.get('id', 'unknown')}: {str(e)}"
        logger.error(result["error"])

    return result


def prepare_models_batch(models_config: list, models_dir: str) -> dict:
    """
    Prepare multiple models in batch.

    Args:
        models_config (list): List of model configurations
        models_dir (str): Base directory for models

    Returns:
        dict: Results containing:
            - success: bool indicating if all operations were successful
            - results: list of individual model results
            - failed_models: list of model IDs that failed
            - prepared_models: dict of model_id -> model_path for successful models
    """
    batch_result = {"success": True, "results": [], "failed_models": [], "prepared_models": {}}

    for i, model_config in enumerate(models_config, 1):
        model_id = model_config.get("id", f"model_{i}")
        logger.info(f"Preparing model {i}/{len(models_config)}: {model_id}")

        result = download_and_prepare_model(model_config, models_dir)
        batch_result["results"].append(result)

        if result["success"]:
            batch_result["prepared_models"][model_id] = result["model_path"]
            logger.info(f" ✓ Model {model_id} prepared successfully")
        else:
            batch_result["failed_models"].append(model_id)
            batch_result["success"] = False
            logger.error(f" ✗ Model {model_id} preparation failed: {result['error']}")

    return batch_result
