# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Models utilities package.

Provides utilities for model management including export, setup, and conversion
for various model formats including OpenVINO, PyTorch, and ONNX.
"""

# Import from export_model
from .export_model import (
    add_common_arguments,
    add_servable_to_config,
    export_embeddings_model,
    export_embeddings_model_ov,
    export_image_generation_model,
    export_rerank_model,
    export_rerank_model_ov,
    export_rerank_tokenizer,
    export_text_generation_model,
    get_models_max_context,
    set_rt_info,
)
from .setup_model import (
    YOLO_MODELS,
    cleanup_model,
    convert_yolo_to_openvino,
    download_and_setup_prequantized_ovms_model,
    download_model,
    download_yolo_model,
    export_ovms_model,
    list_available_models,
    quantize_model,
    setup_model,
    verify_model,
)

# Re-export all functions and classes
__all__ = [
    # From export_model
    "export_rerank_tokenizer",
    "set_rt_info",
    "get_models_max_context",
    "add_servable_to_config",
    "export_text_generation_model",
    "export_embeddings_model",
    "export_embeddings_model_ov",
    "export_rerank_model",
    "export_rerank_model_ov",
    "export_image_generation_model",
    "add_common_arguments",
    # From setup_model
    "download_model",
    "download_yolo_model",
    "convert_yolo_to_openvino",
    "quantize_model",
    "verify_model",
    "setup_model",
    "cleanup_model",
    "list_available_models",
    "export_ovms_model",
    "download_and_setup_prequantized_ovms_model",
    "YOLO_MODELS",
]
