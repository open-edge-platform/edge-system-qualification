# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Batch processing utilities for model preparation.

This module handles batch downloading and preparation of models from various sources
including ultralytics, zip files, and direct file downloads.
"""

import logging
import os
from pathlib import Path

from .common import construct_export_path_components
from .kagglehub import export_kaggle_model
from .ultralytics import download_yolo_model, export_yolo_model

logger = logging.getLogger(__name__)


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

            # Get export_args from model config if available
            export_args = model_config.get("export_args", None)

            path_components = construct_export_path_components(export_args)
            if path_components:
                subdir = "_".join(path_components)
                precision_dir = os.path.join(model_id_dir, precision, subdir)
            else:
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
                export_args=export_args,
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

        # Handle kagglehub source (download from Kaggle and export to OpenVINO)
        elif source == "kagglehub" and model_format == "openvino":
            precision_dir = os.path.join(model_id_dir, precision)
            xml_path = os.path.join(precision_dir, f"{model_id}.xml")

            # Check if model is already exported
            if os.path.exists(xml_path):
                result["model_path"] = xml_path
                result["success"] = True
                result["files"].append(xml_path)
                logger.info(f" ✓ Model {model_id} already exported and ready: {xml_path}")
                return result

            # Get kagglehub-specific parameters
            kaggle_handle = model_config.get("kaggle_handle")
            if not kaggle_handle:
                result["error"] = f"Missing 'kaggle_handle' for kagglehub model: {model_id}"
                logger.error(result["error"])
                return result

            # Get conversion and quantization args from model config
            convert_args = model_config.get("convert_args", None)
            quantize_args = model_config.get("quantize_args", None)

            # Export model to OpenVINO format (includes download, conversion, and optional quantization)
            export_success = export_kaggle_model(
                model_id=model_id,
                kaggle_handle=kaggle_handle,
                models_dir=models_dir,
                model_precision=precision,
                convert_args=convert_args,
                quantize_args=quantize_args,
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
                result["error"] = f"Failed to export Kaggle model: {model_id}"

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
