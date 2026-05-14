# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Local OVMS export utilities (validation, config management, graph templates).

The actual model-export work (optimum-cli / optimum-intel) is no longer performed
here.  Instead it is delegated to the upstream OVMS script that is downloaded at
runtime and executed inside an isolated virtual environment by
:mod:`esq.utils.servers.ovms.export_runner`.

Upstream script reference:
  https://github.com/openvinotoolkit/model_server/tree/main/demos/common/export_models
  commit: 5af61fb (Feb 5, 2026)

Functions kept in this module:
  - validate_openvino_model_export  – verify model directory integrity post-export
  - cleanup_incomplete_model_export – remove a partially-written model directory
  - add_servable_to_config          – register a model in the OVMS config JSON
  - text_generation_graph_template  – Jinja2 template for the MediaPipe graph
"""

import json
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Jinja2 graph template used by setup.py for pre-quantised model deployments
# (kept in sync with the upstream export_model.py at OVMS_COMMIT in export_runner.py)
# ---------------------------------------------------------------------------

text_generation_graph_template = """input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {
    tag_index: 'LOOPBACK:0',
    back_edge: true
  }
  node_options: {
      [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
          {%- if pipeline_type %}
          pipeline_type: {{pipeline_type}},{% endif %}
          models_path: "{{model_path}}",
          plugin_config: '{{plugin_config|safe}}',
          enable_prefix_caching: {% if not enable_prefix_caching %}false{% else %} true{% endif%},
          cache_size: {{cache_size|default("0", true)}},
          {%- if max_num_batched_tokens %}
          max_num_batched_tokens: {{max_num_batched_tokens}},{% endif %}
          {%- if not dynamic_split_fuse %}
          dynamic_split_fuse: false, {% endif %}
          max_num_seqs: {% if draft_eagle3_mode %}1{% else %}{{max_num_seqs|default("256", true)}}{% endif %},
          device: "{{target_device|default("CPU", true)}}",
          {%- if draft_model_dir_name %}
          # Speculative decoding configuration
          draft_models_path: "./{{draft_model_dir_name}}",
          draft_device: "{{target_device|default("CPU", true)}}",
          draft_eagle3_mode: {{draft_eagle3_mode|default(false)}},{% endif %}
          {%- if reasoning_parser %}
          reasoning_parser: "{{reasoning_parser}}",{% endif %}
          {%- if tool_parser %}
          tool_parser: "{{tool_parser}}",{% endif %}
          {%- if enable_tool_guided_generation %}
          enable_tool_guided_generation: {% if not enable_tool_guided_generation %}false{% else %} true{% endif%},{% endif %}
      }
  }
  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler",
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "LOOPBACK:0"
        }
      }
    }
  }
}"""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_openvino_model_export(model_path: str, model_type: str = "text_generation") -> bool:
    """
    Validate that an OpenVINO model directory contains all required files.

    This function checks if a model has been properly exported by verifying the presence
    of critical model files. This is essential to detect incomplete or failed exports.

    Args:
        model_path: Path to the model directory
        model_type: Type of model (text_generation, embeddings, etc.)

    Returns:
        bool: True if model is valid, False otherwise

    Required files for text_generation models:
        - openvino_language_model.xml (primary model file) OR openvino_model.xml
        - openvino_language_model.bin (weights) OR openvino_model.bin
        - openvino_tokenizer.xml (tokenizer)
        - openvino_detokenizer.xml (detokenizer)

    Reference: https://huggingface.co/docs/optimum/intel/openvino/export
    """
    if not os.path.isdir(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return False

    # For text generation models, check for essential files
    if model_type == "text_generation":
        # Check for model files (either openvino_language_model.* or openvino_model.*)
        has_language_model = os.path.isfile(
            os.path.join(model_path, "openvino_language_model.xml")
        ) and os.path.isfile(os.path.join(model_path, "openvino_language_model.bin"))
        has_model = os.path.isfile(os.path.join(model_path, "openvino_model.xml")) and os.path.isfile(
            os.path.join(model_path, "openvino_model.bin")
        )

        # Must have at least one set of model files
        if not (has_language_model or has_model):
            logger.error(
                f"Model directory missing required XML/BIN files: {model_path}\n"
                f"Expected: openvino_language_model.xml/.bin OR openvino_model.xml/.bin"
            )
            return False

        # Check for tokenizer/detokenizer files
        has_tokenizer = os.path.isfile(os.path.join(model_path, "openvino_tokenizer.xml"))
        has_detokenizer = os.path.isfile(os.path.join(model_path, "openvino_detokenizer.xml"))

        if not (has_tokenizer and has_detokenizer):
            logger.warning(
                f"Model directory missing tokenizer/detokenizer files: {model_path}\n"
                f"This may cause inference issues. Expected: openvino_tokenizer.xml and openvino_detokenizer.xml"
            )
            # Don't fail validation for missing tokenizer - it may be added later

    logger.debug(f"Model validation passed for: {model_path}")
    return True


def cleanup_incomplete_model_export(model_path: str) -> None:
    """
    Clean up an incomplete or failed model export directory.

    This removes directories that were created but not properly populated with model
    files.  This is critical to prevent subsequent runs from detecting a directory
    and assuming the model was exported successfully when it was not.

    Args:
        model_path: Path to the model directory to clean up
    """
    if os.path.isdir(model_path):
        logger.info(f"Cleaning up incomplete model export directory: {model_path}")
        try:
            shutil.rmtree(model_path)
            logger.info(f"Successfully removed incomplete export directory: {model_path}")
        except Exception as e:
            logger.error(f"Failed to remove incomplete export directory {model_path}: {e}")


# ---------------------------------------------------------------------------
# OVMS config helpers
# ---------------------------------------------------------------------------

def add_servable_to_config(config_path: str, model_name: str, base_path: str) -> None:
    """
    Register (or update) a model servable entry in the OVMS config JSON file.

    Creates the config file if it does not yet exist.  Migrates legacy
    ``mediapipe_config_list`` entries to ``model_config_list`` automatically.

    Args:
        config_path: Path to the OVMS ``config_all.json`` file.
        model_name: Name of the model servable to register.
        base_path: Path to the model directory (may be relative to the config file).
    """
    base_path = Path(base_path).as_posix()
    logger.debug(f"add_servable_to_config: {config_path}  model={model_name}  base={base_path}")

    if not os.path.isfile(config_path):
        logger.debug("Creating new OVMS config file")
        with open(config_path, "w") as config_file:
            json.dump({"mediapipe_config_list": [], "model_config_list": []}, config_file, indent=4)

    with open(config_path, "r") as config_file:
        config_data = json.load(config_file)

    if "model_config_list" not in config_data:
        config_data["model_config_list"] = []

    # Migrate legacy mediapipe_config_list → model_config_list
    if "mediapipe_config_list" in config_data:
        for mp_config in config_data["mediapipe_config_list"]:
            if "name" in mp_config and "base_path" in mp_config:
                legacy_name = mp_config["name"] + "_model"
                if not any(d["config"]["name"] == legacy_name for d in config_data["model_config_list"]):
                    config_data["model_config_list"].append(
                        {"config": {"name": legacy_name, "base_path": mp_config["base_path"]}}
                    )
        del config_data["mediapipe_config_list"]

    model_list = config_data["model_config_list"]
    updated = False
    for model_config in model_list:
        if model_config["config"]["name"] == model_name:
            model_config["config"]["base_path"] = base_path
            updated = True
    if not updated:
        model_list.append({"config": {"name": model_name, "base_path": base_path}})

    with open(config_path, "w") as config_file:
        json.dump(config_data, config_file, indent=4)

    logger.debug(f"Registered servable '{model_name}' in config: {config_path}")
