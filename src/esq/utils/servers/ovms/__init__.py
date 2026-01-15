# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
OpenVINO Model Server (OVMS) utilities.

This module provides utilities for exporting and setting up models for
OpenVINO Model Server deployment with text generation support.

Usage:
    from esq.utils.servers.ovms import export_ovms_model, download_and_setup_prequantized_ovms_model
    
    # Export model to OVMS format
    success, export_duration, download_duration, quant_config, model_name = export_ovms_model(
        model_id_or_path="microsoft/Phi-4-mini-instruct",
        models_dir="/path/to/models",
        model_precision="int4",
        device_id="CPU"
    )
    
    # Setup pre-quantized model
    success = download_and_setup_prequantized_ovms_model(
        model_id="OpenVINO/model-int4-ov",
        models_dir="/path/to/models",
        device_id="CPU"
    )
"""

from .setup import download_and_setup_prequantized_ovms_model, export_ovms_model

# Expose main functions at package level
__all__ = [
    "export_ovms_model",
    "download_and_setup_prequantized_ovms_model",
]
