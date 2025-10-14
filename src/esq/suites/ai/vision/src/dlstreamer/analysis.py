# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer analysis and result processing functions - simplified."""

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def parse_device_result_file(device_id: str, results_dir: str) -> Dict[str, Any]:
    """
    Parse result file for a specific device.

    Args:
        device_id: Device ID to parse results for
        results_dir: Directory containing result files

    Returns:
        Dictionary with parsed device results
    """
    result_file = os.path.join(results_dir, f"total_streams_result_0_{device_id}.json")

    default_result = {
        "device_id": device_id,
        "num_streams": 0,
        "per_stream_fps": 0.0,
        "pass": False,
        "status": False,
        "error": "No result file found",
    }

    if not os.path.exists(result_file):
        logger.warning(f"No result file found for {device_id} at {result_file}")
        return default_result

    try:
        with open(result_file, "r") as f:
            data = json.load(f)

        if device_id in data:
            device_data = data[device_id]
            return {
                "device_id": device_id,
                "num_streams": device_data.get("num_streams", 0),
                "per_stream_fps": device_data.get("per_stream_fps", 0.0),
                "pass": device_data.get("pass", False),
                "status": True,
            }
        else:
            return {**default_result, "error": f"Device {device_id} not found in result file"}

    except Exception as e:
        logger.error(f"Failed to parse results for {device_id}: {e}")
        return {**default_result, "error": f"Failed to parse results: {str(e)}"}
