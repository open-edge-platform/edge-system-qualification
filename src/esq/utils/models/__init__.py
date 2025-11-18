# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Models utilities package.

Provides utilities for model management including export, setup, and conversion
for various model formats including OpenVINO, PyTorch, and ONNX.

Package structure:
- common: Shared utilities for model management
- ultralytics: YOLO model handling (detection models)
- huggingface: HuggingFace LLM model handling (text generation)
- batch: Batch processing for multiple models
- export_model: Model export utilities for OVMS format
"""

from .batch import download_and_prepare_model, prepare_models_batch
from .common import download_model
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
from .huggingface import download_and_setup_prequantized_ovms_model, export_ovms_model
from .ultralytics import YOLO_MODELS, download_yolo_model, export_yolo_model

# Re-export all functions and classes
__all__ = [
    # From common
    "download_model",
    # From ultralytics
    "YOLO_MODELS",
    "download_yolo_model",
    "export_yolo_model",
    # From huggingface
    "export_ovms_model",
    "download_and_setup_prequantized_ovms_model",
    # From batch
    "download_and_prepare_model",
    "prepare_models_batch",
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
]
