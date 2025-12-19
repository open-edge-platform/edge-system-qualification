# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
GPU topology detection utilities for media test suites.

Provides functions to detect Intel GPU hardware topology information,
including Video Decode/Encode box (VD box) counts for GPU capability assessment.
Used by media and AI test suites for platform-aware reference matching.
"""

import logging
import re
from typing import Optional

from sysagent.utils.core.process import run_command

logger = logging.getLogger(__name__)


def get_vdbox_count_for_device(device: str) -> Optional[int]:
    """
    Get VD box count for GPU devices (optional secondary validation).

    This function counts Video Decode/Encode boxes (VD boxes) on Intel GPUs
    using intel_gpu_top. VD box count indicates GPU media acceleration capabilities
    and helps validate platform matches for GPU-based tests.

    Args:
        device: Device identifier (e.g., "CPU", "GPU.0", "GPU.1", "NPU")

    Returns:
        VD box count (1, 2, etc.) or None if:
        - Device is not GPU (CPU/NPU tests don't need VD validation)
        - Detection fails (graceful degradation)
        - intel_gpu_top not available

    Examples:
        - N97 systems: 1 VD box
        - i7-1360P systems: 2 VD boxes
        - dGPU (Arc A380): 2 VD boxes
    """
    # Only detect VD boxes for GPU devices
    if device not in ["GPU.0", "GPU.1"] and "GPU" not in device:
        return None

    try:
        # Get GPU device path from lsgpu
        lsgpu_result = run_command(["lsgpu"], timeout=5, capture_output=True)

        if not lsgpu_result.success:
            logger.debug("lsgpu command failed, skipping VD box detection")
            return None

        # Parse lsgpu output to find DRM device path
        # Example line: "card0                8086:7d55           drm:/dev/dri/renderD128"
        drm_path = None
        for line in lsgpu_result.stdout.splitlines():
            if "drm:/dev/dri/" in line:
                match = re.search(r"drm:(/dev/dri/[^\s]+)", line)
                if match:
                    drm_path = match.group(1)
                    break

        if not drm_path:
            logger.debug("No DRM device path found in lsgpu output")
            return None

        # Use intel_gpu_top to count VCS (Video Codec Stream) units = VD boxes
        # Run in background mode (-l for list mode, duration controlled by timeout)
        # NOTE: Running without sudo - user must have render group permissions
        # If permissions are missing, this will fail gracefully and return None
        command = ["intel_gpu_top", "-d", drm_path, "-l"]

        # Execute command with timeout (3 seconds to capture header)
        try:
            gpu_top_result = run_command(command, timeout=3, capture_output=True)
        except Exception as e:
            logger.debug(f"Failed to run intel_gpu_top (missing permissions?): {e}")
            return None

        # Even if timed out, we should have captured the header
        output = gpu_top_result.stdout
        vdbox_count = 0

        # Count VCS units in the header line
        # Example header: "     TIME      GPU       RENDER     COPY       VCS0       VCS1     VECS"
        for line in output.split("\n"):
            if "Freq MHz" in line or "TIME" in line:
                vdbox_count = line.count("VCS")
                if vdbox_count > 0:
                    break

        if vdbox_count > 0:
            logger.debug(f"Detected {vdbox_count} VD box(es) on {device}")
            return vdbox_count
        else:
            logger.debug("Could not parse VD box count from intel_gpu_top output")
            return None

    except Exception as e:
        logger.debug(f"Failed to detect VD box count: {e}")
        return None


def get_gpu_drm_paths() -> dict:
    """
    Get mapping of GPU PCI IDs to DRM device paths.

    Returns:
        Dictionary mapping PCI IDs to DRM paths (e.g., {"8086:7d55": "/dev/dri/renderD128"})
    """
    try:
        lsgpu_result = run_command(["lsgpu"], timeout=5, capture_output=True)

        if not lsgpu_result.success:
            logger.debug("lsgpu command failed")
            return {}

        # Parse output to extract PCI ID and DRM path mappings
        # Example: "card0                8086:7d55           drm:/dev/dri/renderD128"
        pattern = re.compile(r"\b([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\b.*?(drm:/dev/dri/[^\s]+)")
        matches_info = pattern.findall(lsgpu_result.stdout)

        result = {}
        for device_id, drm_info in matches_info:
            # Remove 'drm:' prefix
            drm_path = drm_info.replace("drm:", "")
            result[device_id] = drm_path
            logger.debug(f"Found GPU: Device ID={device_id}, DRM Path={drm_path}")

        return result

    except Exception as e:
        logger.warning(f"Failed to get GPU DRM paths: {e}")
        return {}
