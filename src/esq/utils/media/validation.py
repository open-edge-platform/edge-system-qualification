# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Validation and pre-check utilities for media benchmarks.

Consolidates shell script functionality from:
- opt.sh: Device detection and validation
- common.sh: Logging and formatting utilities
- check_va_plugins.sh: VA-API plugin verification
- gst_loop_mp4.sh: Video looping for test preparation
"""

import logging
from typing import Dict, List, Tuple

# Support both installed package and Docker container usage
try:
    from sysagent.utils.core.process import run_command
except ModuleNotFoundError:
    # Inside Docker container, use the lightweight container utilities
    from .container_utils import run_command

# Known device IDs for iGPU and dGPU
IGPU_DEV_IDS = [
    "A7A9",
    "A7A8",
    "A7A1",
    "A7A0",
    "A721",
    "A720",
    "A78B",
    "A78A",
    "A789",
    "A788",
    "A783",
    "A782",
    "A781",
    "A780",
    "4907",
    "4905",
    "4680",
    "4682",
    "4688",
    "468A",
    "468B",
    "4690",
    "4692",
    "4693",
    "46D0",
    "46D1",
    "46D2",
    "4626",
    "4628",
    "462A",
    "46A0",
    "46A1",
    "46A2",
    "46A3",
    "46A6",
    "46A8",
    "46AA",
    "46B0",
    "46B1",
    "46B2",
    "46B3",
    "46C0",
    "46C1",
    "46C2",
    "46C3",
    "4C8A",
    "4C8B",
    "4C90",
    "4C9A",
    "4C8C",
    "4C80",
    "4E71",
    "4E61",
    "4E57",
    "4E55",
    "4E51",
    "4571",
    "4557",
    "4555",
    "4551",
    "4541",
    "9A59",
    "9A60",
    "9A68",
    "9A70",
    "9A40",
    "9A49",
    "9A78",
    "9AC0",
    "9AC9",
    "9AD9",
    "9AF8",
    # Recent Platforms - Arrow Lake, Meteor Lake, Lunar Lake
    "7DD1",
    "7DD5",
    "7D55",
    "7D60",  # MTL-S (ADDED)
    "7D45",
    "7D40",
    "6420",
    "6422",  # LNL variant (ADDED)
    "64B0",
    "7D51",
    "7D67",
    "7D41",
    # Raptor Lake
    "A720",  # RPL-P (ADDED)
    "A721",  # RPL-P (ADDED)
    "A7A0",  # RPL-P (ADDED)
    "A7A1",  # RPL-P (ADDED)
    "A78A",  # RPL-S (ADDED)
    "A78B",  # RPL-S (ADDED)
    "A7A9",  # RPL-S Refresh (ADDED)
    "A7AA",
    "A7AB",
    "A7AC",
    "A7AD",
    # Alder Lake
    "4636",  # ADL-P (ADDED)
    "4638",  # ADL-P (ADDED)
    "463A",  # ADL-P (ADDED)
    # Tiger Lake-H
    "9A01",  # TGL-H (ADDED)
    "9A02",  # TGL-H (ADDED)
    "9A09",  # TGL-H (ADDED)
    "9A0A",  # TGL-H (ADDED)
    "9A0B",  # TGL-H (ADDED)
    "9A0C",  # TGL-H (ADDED)
    # Ice Lake
    "8A50",  # ICL-LP (ADDED)
    "8A51",  # ICL-LP (ADDED)
    "8A52",  # ICL-LP (ADDED)
    "8A53",  # ICL-LP (ADDED)
    "8A56",  # ICL-LP (ADDED)
    "8A57",  # ICL-LP (ADDED)
    "8A58",  # ICL-LP (ADDED)
    "8A59",  # ICL-LP (ADDED)
    "8A5A",  # ICL-LP (ADDED)
    "8A5B",  # ICL-LP (ADDED)
    "8A5C",  # ICL-LP (ADDED)
    "8A5D",  # ICL-LP (ADDED)
    # Legacy (DG1 - technically discrete but included for compatibility)
    "0BD5",
    "0BDA",
    "56C0",
    "56C1",
    "4908",
    "4909",
    "46D3",
    "46D4",
]

DGPU_DEV_IDS = [
    "56B3",
    "56B2",
    "56A4",
    "56A3",
    "56BA",
    "5697",
    "5696",
    "5695",
    "56B1",
    "56B0",
    "56A6",
    "56A5",
    "56A1",
    "56A0",
    "5694",
    "5693",
    "5692",
    "5691",
    "5690",
    "56A2",
    "E20B",
    "E20C",
    "64A0",
    "7D55",
    "56BC",
    "56BD",
    "56BB",
    "E212",
    "E211",
]

# NPU Platform Device IDs (Meteor Lake, Arrow Lake, Lunar Lake)
# These platforms have NPU co-processor support when paired with iGPU
NPU_PLATFORM_DEV_IDS = [
    # Meteor Lake iGPU IDs
    "7D55",  # MTL-H
    "7D45",  # MTL-U
    "7D40",  # MTL-P
    "7DD1",  # MTL variant
    "7DD5",  # MTL-G
    # Arrow Lake iGPU IDs
    "7D51",  # ARL-H
    "7D67",  # ARL-S
    "7D41",  # ARL-U
    # Lunar Lake iGPU IDs
    "6420",  # LNL
    "6422",  # LNL variant
    "64B0",  # LNL variant
]

logger = logging.getLogger(__name__)


def normalize_device_name(device: str) -> str:
    """
    Normalize OpenVINO device format to canonical benchmark format.

    Converts OpenVINO device names (GPU, GPU.0, GPU.1, CPU, NPU, HETERO:...)
    to canonical format used by benchmark scripts (iGPU, dGPU.0, dGPU.1, CPU, NPU).

    Args:
        device: OpenVINO device name (e.g., "GPU", "GPU.0", "GPU.1", "CPU", "NPU", "HETERO:GPU.0,GPU.1")

    Returns:
        Canonical device name:
        - "GPU" or "GPU.0" → "iGPU" (integrated GPU)
        - "GPU.1" → "dGPU.0" (first discrete GPU)
        - "GPU.2" → "dGPU.1" (second discrete GPU)
        - "CPU" → "CPU"
        - "NPU" → "NPU"
        - "HETERO:..." → "HETERO:..." (pass through, handled separately)

    Note: Relies on OpenVINO's device numbering where:
        - GPU or GPU.0 is always the integrated GPU (if present)
        - GPU.1+ are discrete GPUs
    """
    device_upper = device.upper()

    # Handle HETERO devices - pass through as-is
    if device_upper.startswith("HETERO:"):
        return device

    # Handle CPU
    if device_upper == "CPU":
        return "CPU"

    # Handle NPU
    if device_upper == "NPU":
        return "NPU"

    # Handle GPU devices
    if device_upper == "GPU" or device_upper == "GPU.0":
        # GPU or GPU.0 is always integrated GPU in OpenVINO
        return "iGPU"

    if device_upper.startswith("GPU."):
        try:
            # Extract GPU number (e.g., "GPU.1" → 1)
            gpu_num = int(device_upper.split(".")[1])
            if gpu_num > 0:
                # Convert to dGPU numbering: GPU.1 → dGPU.0, GPU.2 → dGPU.1
                return f"dGPU.{gpu_num - 1}"
        except (IndexError, ValueError):
            pass

    # Unknown format - log warning and return as-is
    logger.warning(f"Unknown device format: '{device}'. Returning as-is.")
    return device


def detect_platform_type() -> Dict[str, any]:
    """
    Detect available devices and platform characteristics.

    Returns:
        dict: Platform information with keys:
            - available_devices: List of device strings (CPU, iGPU, dGPU.0, dGPU.1)
            - is_mtl: Boolean indicating NPU platform (Meteor Lake, Arrow Lake, or Lunar Lake)
            - has_igpu: Boolean indicating iGPU presence
            - dgpu_count: Number of discrete GPUs
            - vendor_ids: List of vendor IDs
            - device_ids: List of device IDs
    """
    platform_info = {
        "available_devices": [],
        "is_mtl": False,
        "has_igpu": False,
        "dgpu_count": 0,
        "vendor_ids": [],
        "device_ids": [],
    }

    # Check CPU type for Xeon
    try:
        lscpu_output = run_command(["lscpu"], capture_output=True)

        for line in lscpu_output.stdout.split("\n"):
            if "Model name" in line:
                cpu_model = line.split(":", 1)[1].strip()
                if "Xeon" in cpu_model:
                    platform_info["available_devices"].append("CPU")
                break
    except Exception:
        logger.warning("Failed to run lscpu")

    # Detect GPUs using lspci
    try:
        lspci_output = run_command(["lspci", "-nn"], capture_output=True)

        for line in lspci_output.stdout.split("\n"):
            if "DISPLAY" in line.upper() or "VGA" in line.upper():
                # Extract device ID from format: [8086:56a0]
                import re

                match = re.search(r"\[([0-9a-fA-F]+):([0-9a-fA-F]+)\]", line)
                if match:
                    vendor_id = match.group(1)
                    device_id = match.group(2).upper()

                    platform_info["vendor_ids"].append(vendor_id)
                    platform_info["device_ids"].append(device_id)

                    # Check if NPU platform (Meteor Lake, Arrow Lake, or Lunar Lake)
                    if device_id in NPU_PLATFORM_DEV_IDS:
                        platform_info["is_mtl"] = True

                    # Check device type
                    if device_id in IGPU_DEV_IDS:
                        platform_info["available_devices"].append("iGPU")
                        platform_info["has_igpu"] = True
                    elif device_id in DGPU_DEV_IDS:
                        dgpu_idx = platform_info["dgpu_count"]
                        platform_info["available_devices"].append(f"dGPU.{dgpu_idx}")
                        platform_info["dgpu_count"] += 1

    except Exception:
        logger.warning("Failed to run lspci")

    return platform_info


def get_render_device(device: str, has_igpu: bool) -> int:
    """
    Calculate render device number for GPU.

    Args:
        device: Device string (iGPU, dGPU.0, dGPU.1, etc.)
        has_igpu: Whether system has iGPU

    Returns:
        int: Render device number (128 for iGPU, 129+ for dGPU)
    """
    if device == "iGPU":
        return 128

    if device.startswith("dGPU."):
        dgpu_idx = int(device.split(".")[1])
        if has_igpu:
            return 129 + dgpu_idx
        else:
            return 128 + dgpu_idx

    return 128


def validate_options(
    devices: List[str] = None, codecs: List[str] = None, allowed_codecs: List[str] = None
) -> Tuple[List[str], List[str], Dict[str, any]]:
    """
    Validate and normalize device and codec options.

    Replicates opt.sh functionality for option validation.

    Args:
        devices: List of requested devices (None = all available)
        codecs: List of requested codecs (None = all allowed)
        allowed_codecs: List of allowed codec values

    Returns:
        tuple: (validated_devices, validated_codecs, platform_info)
    """
    platform_info = detect_platform_type()
    available_devices = platform_info["available_devices"]

    # Validate devices
    if devices is None:
        validated_devices = available_devices
    else:
        validated_devices = []
        for device in devices:
            if device in available_devices:
                validated_devices.append(device)
            else:
                logger.warning(f"Invalid device: {device}. Available: {', '.join(available_devices)}")

    # Validate codecs
    if allowed_codecs is None:
        allowed_codecs = ["h264", "h265", "av1", "vp9"]

    if codecs is None:
        validated_codecs = allowed_codecs
    else:
        validated_codecs = []
        for codec in codecs:
            if codec in allowed_codecs:
                validated_codecs.append(codec)
            else:
                logger.warning(f"Invalid codec: {codec}. Allowed: {', '.join(allowed_codecs)}")

    return validated_devices, validated_codecs, platform_info


def check_va_plugins() -> str:
    """
    Check available VA-API plugins by inspecting GStreamer.

    Replicates check_va_plugins.sh functionality.

    Returns:
        str: Output from gst-inspect-1.0 showing available VA plugins
    """
    try:
        # Source OpenVINO and DLStreamer environment
        # Note: In production, these should be sourced before Python execution
        # or environment variables passed to subprocess

        # Check gvafpscounter plugin
        gva_result = run_command(
            ["gst-inspect-1.0", "gvafpscounter"], capture_output=True, env={"GST_DEBUG": "3"}
        )

        logger.debug(f"gvafpscounter check: {gva_result.stdout}")

        # Check VA-API plugins
        va_result = run_command(["gst-inspect-1.0", "va"], capture_output=True, env={"GST_DEBUG": "3"})

        logger.debug(f"VA plugins: {va_result.stdout}")

        return va_result.stdout

    except Exception as e:
        logger.error(f"Failed to check VA plugins: {e}")
        return ""


def loop_video(
    loops: int, video_resolution: str, video_type: str, video_in_path: str = None, video_out_path: str = None
) -> bool:
    """
    Loop video file to create longer test video.

    Replicates gst_loop_mp4.sh functionality using GStreamer concat.

    Args:
        loops: Number of times to loop input video
        video_resolution: Resolution string (1080p, 720p, etc.)
        video_type: Video codec (h264, h265)
        video_in_path: Path to input video (default: auto-generated)
        video_out_path: Path to output video (default: auto-generated)

    Returns:
        bool: True if successful, False otherwise
    """
    if video_in_path is None:
        video_in_path = f"/home/dlstreamer/sample_video/car_{video_resolution}30_10s_{video_type}.mp4"

    if video_out_path is None:
        video_out_path = f"/home/dlstreamer/sample_video/car_{video_resolution}30_120s_{video_type}.mp4"

    # Remove existing output file
    import os

    if os.path.exists(video_out_path):
        os.remove(video_out_path)

    # Build GStreamer pipeline
    if "h265" in video_type:
        pipeline = f"gst-launch-1.0 -e concat name=c ! h265parse ! mp4mux ! filesink location={video_out_path}"
    else:
        pipeline = f"gst-launch-1.0 -e concat name=c ! h264parse ! mp4mux ! filesink location={video_out_path}"

    # Add source elements for each loop
    for i in range(1, loops + 1):
        pipeline += f" filesrc location={video_in_path} ! qtdemux ! queue ! c. "

    try:
        # Run GStreamer pipeline
        result = run_command(pipeline, capture_output=True, timeout=300.0)

        if not result.success:
            logger.error(f"Failed to loop video: {result.stderr}")
            return False

        logger.info(f"Video looped successfully: {video_out_path}")

        # Verify output with gst-discoverer
        discover_result = run_command(["gst-discoverer-1.0", video_out_path], capture_output=True)

        logger.debug(f"Video info: {discover_result.stdout}")

        return result.success

    except Exception as e:
        logger.error(f"Failed to loop video: {e}")
        return False


def format_log_section(title: str, details_log_file: str = None) -> str:
    """
    Format log section with separator lines.

    Replicates format_log_section from common.sh.

    Args:
        title: Section title
        details_log_file: Optional log file to write to

    Returns:
        str: Formatted log string
    """
    import time

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Calculate separator line length
    len_title = len(title)
    separator_len = max(120 - len_title, 40)
    separator = "-" * separator_len

    # Format log section
    log_output = f"\n[{timestamp}] {separator}\n[{timestamp}] {title}\n[{timestamp}] {separator}\n"

    # Write to file if specified
    if details_log_file:
        with open(details_log_file, "a") as f:
            f.write(log_output)

    return log_output
