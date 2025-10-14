# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Model setup utilities for downloading and preparing models.

This module provides utilities for downloading models from Hugging Face Hub,
setting up YOLO models, and preparing models for inference.
"""
import os
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
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
            model_id_or_path,
            revision=revision,
            ignore_patterns=ignore_patterns or ["*.pth"]
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


def convert_yolo_to_openvino(pt_path: Path, output_dir: Optional[Path] = None, precision: str = "fp16") -> Optional[Path]:
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
                quantized_model = quantize(
                    model=ov_model,
                    calibration_dataset=dataset,
                    subset_size=10
                )
                quantized_model.set_rt_info(model_type if model_id in YOLO_MODELS else "YOLO", ["model_info", "model_type"])
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


def export_yolo_model(
    model_id: str,
    models_dir: str = "models",
    model_precision: str = "fp16"
) -> bool:
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
        converted_path = Path(
            model.export(
                format="openvino"
            )
        ).resolve()

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
                quantized_model = quantize(
                    model=fp16_model,
                    calibration_dataset=dataset,
                    subset_size=10
                )
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


def quantize_model(model_path: Path, calibration_data: Optional[Any] = None, output_path: Optional[Path] = None) -> Optional[Path]:
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
    result = {
        "valid": False,
        "path": str(model_path),
        "inputs": [],
        "outputs": [],
        "error": None
    }
    
    try:
        core = ov.Core()
        model = core.read_model(str(model_path))
        
        # Get input information
        for input_layer in model.inputs:
            result["inputs"].append({
                "name": input_layer.get_any_name(),
                "shape": list(input_layer.shape),
                "type": str(input_layer.element_type)
            })
        
        # Get output information
        for output_layer in model.outputs:
            result["outputs"].append({
                "name": output_layer.get_any_name(),
                "shape": list(output_layer.shape),
                "type": str(output_layer.element_type)
            })
        
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
    info = {
        "path": str(model_path),
        "exists": model_path.exists(),
        "type": "unknown",
        "size_mb": 0,
        "files": []
    }
    
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


def setup_model(model_id: str, model_type: str = "auto", models_dir: Optional[str] = None, convert_to_openvino: bool = True) -> Optional[Path]:
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
    
    models = {
        "openvino": [],
        "pytorch": [],
        "onnx": [],
        "directories": []
    }
    
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
) -> bool:
    """
    Export model to OpenVINO Model Server format.
    """
    config_path = os.path.join(models_dir, "config_all.json")
    if os.path.exists(config_path):
        os.remove(config_path)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)

    try:
        logger.info(f"Exporting model {model_id_or_path} to OpenVINO Model Server format at {models_dir}")
        task_parameters = {
            "target_device": device_id,
            "pipeline_type": "LM",
            "kv_cache_precision": None,
            "extra_quantization_params": "--sym --group-size -1 --ratio 1.0",
            "enable_prefix_caching": True,
            "dynamic_split_fuse": True,
            "max_num_batched_tokens": None,
            "max_num_seqs": "2048",
            "cache_size": 2,
            "draft_source_model": None,
            "draft_model_name": None,
            "max_prompt_len": None,
        }
        
        # Import the export function
        from .export_model import export_text_generation_model
        
        export_text_generation_model(
            source_model=model_id_or_path,
            model_name=model_id_or_path,
            model_repository_path=models_dir,
            precision=model_precision,
            task_parameters=task_parameters,
            config_file_path=config_path,
        )
        logger.info(f"Model exported successfully to {models_dir}")
        return True
    except Exception as e:
        logger.error(f"OVMS model export failed with error: {e}")
        raise RuntimeError("OVMS model export failed") from e


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
    result = {
        "success": False,
        "model_path": None,
        "error": None,
        "files": []
    }
    
    try:
        model_id = model_config.get('id', '')
        source = model_config.get('source', '')
        model_format = model_config.get('format', '')
        precision = model_config.get('precision', 'fp16')
        url = model_config.get('url', '')
        sha256 = model_config.get('sha256', '')
        
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
                    url=url,
                    target_path=zip_path,
                    sha256sum=sha256 if isinstance(sha256, str) else None
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
                original_filename = os.path.basename(file_url.split('?')[0])  # Remove query params
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
                xml_file = next((f for f in existing_files if f.endswith('.xml')), None)
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
                original_filename = os.path.basename(file_url.split('?')[0])  # Remove query params
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
                    download_result = download_file(
                        url=file_url,
                        target_path=file_path,
                        sha256sum=file_sha256
                    )
                    downloaded_files.append(download_result["path"])
                else:
                    logger.info(f" ✓ Model {model_id} {file_extension.upper()} file already exists")
                    downloaded_files.append(file_path)
            
            # Verify at least one XML file exists (for OpenVINO models)
            xml_files = [f for f in downloaded_files if f.endswith('.xml')]
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
                downloaded_path = download_yolo_model(
                    model_id=model_id,
                    models_dir=models_dir
                )
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
    batch_result = {
        "success": True,
        "results": [],
        "failed_models": [],
        "prepared_models": {}
    }
    
    for i, model_config in enumerate(models_config, 1):
        model_id = model_config.get('id', f'model_{i}')
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
