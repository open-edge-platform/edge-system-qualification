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
        "pass": False,
        "error": "No result file found",
        "metadata": {
            "num_streams": -1,
            "per_stream_fps": 0.0,
        },
        # Note: qualification_state is excluded from parsed results (internal only)
    }

    if not os.path.exists(result_file):
        logger.warning(f"No result file found for {device_id} at {result_file}")
        return default_result

    try:
        with open(result_file, "r") as f:
            data = json.load(f)

        if device_id in data:
            device_data = data[device_id]

            # Return complete device data including all metadata from FPS counter
            # This includes: parse_source, fps_counter_duration, per_stream_fps_list,
            # num_streams, per_stream_fps, total_fps, etc.
            result = {
                "pass": device_data.get("pass", False),
            }

            # Include all fields from device_data EXCEPT internal/redundant ones
            # Exclude: device_id (already the key), status (redundant with pass),
            #          qualification_state (internal binary search tracking)
            for key, value in device_data.items():
                if key not in ["device_id", "status", "qualification_state"]:
                    result[key] = value

            return result
        else:
            return {**default_result, "error": f"Device {device_id} not found in result file"}

    except Exception as e:
        logger.error(f"Failed to parse results for {device_id}: {e}")
        return {**default_result, "error": f"Failed to parse results: {str(e)}"}
