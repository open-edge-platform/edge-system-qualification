# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
HuggingFace model utilities for text generation models.

This module handles downloading, exporting, and setting up HuggingFace models
for OpenVINO Model Server (OVMS) format, including quantization support.
"""

import json
import logging
import os
import shutil

logger = logging.getLogger(__name__)


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

        try:
            from esq.utils.models.export_model import export_text_generation_model

            export_text_generation_model(
                source_model=model_id_or_path,
                model_name=model_safe_name_with_quant,
                model_repository_path=models_dir,
                precision=model_precision,
                task_parameters=task_parameters,
                config_file_path=config_path,
                overwrite_models=False,
                export_timeout=export_timeout,
            )

            logger.info("=" * 80)
            logger.info(f"Model exported successfully to {models_dir}")
            logger.info(f"MediaPipe servable registered as: {model_safe_name_with_quant}")

            # Final validation check to ensure model is properly exported
            model_export_path = os.path.join(models_dir, model_safe_name_with_quant)
            from esq.utils.models.export_model import cleanup_incomplete_model_export, validate_openvino_model_export

            if not validate_openvino_model_export(model_export_path, model_type="text_generation"):
                logger.error(f"Post-export validation failed for: {model_export_path}")
                cleanup_incomplete_model_export(model_export_path)
                raise ValueError(
                    f"Model export validation failed: required OpenVINO model files missing in {model_export_path}"
                )

        except TimeoutError:
            raise
        except ValueError:
            raise
        except Exception as e:
            # Ensure cleanup on unexpected errors
            model_export_path = os.path.join(models_dir, model_safe_name_with_quant)
            from esq.utils.models.export_model import cleanup_incomplete_model_export

            cleanup_incomplete_model_export(model_export_path)
            raise RuntimeError(f"Export failed: {str(e)}") from e

        export_duration = time.time() - export_start_time
        logger.debug(f"Model export completed in {export_duration:.2f} seconds")

        return True, export_duration, quantization_config, model_safe_name_with_quant
    except Exception as e:
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
