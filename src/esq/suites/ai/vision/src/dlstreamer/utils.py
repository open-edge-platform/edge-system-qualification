# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer utility functions for cleanup, device management, and system operations."""

import concurrent.futures
import json
import logging
import os
from typing import Any, Dict, List, Optional

import docker
from sysagent.utils.core import Metrics, Result, get_metric_name_for_device
from sysagent.utils.infrastructure import DockerClient

logger = logging.getLogger(__name__)

# Global thread pool for concurrent operations
THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2)
LOG_FUTURES = []


def create_device_metrics(
    device_list: List[str], prefix: str = "streams_max", unit: str = "streams", value: int = -1
) -> Dict[str, Metrics]:
    """
    Create metrics dictionary for a list of devices.

    Args:
        device_list: List of device IDs
        prefix: Metric name prefix (default: "streams_max")
        unit: Metric unit (default: "streams")
        value: Metric value (default: -1 for error state)

    Returns:
        Dictionary mapping metric names to Metrics objects
    """
    return {get_metric_name_for_device(dev, prefix=prefix): Metrics(unit=unit, value=value) for dev in device_list}


def create_error_result(
    device_id: str,
    error_message: str,
    num_streams: int = -1,
    per_stream_fps: float = 0.0,
    include_device_metadata: bool = True,
) -> Result:
    """
    Create a standardized error Result object.

    Args:
        device_id: Device identifier
        error_message: Error description
        num_streams: Number of streams (-1 for error, 0 for ran but failed)
        per_stream_fps: FPS achieved
        include_device_metadata: Whether to include device_id in metadata (default: True)

    Returns:
        Result object with error metadata
    """
    metadata = {
        "status": False,
        "per_stream_fps": per_stream_fps,
        "num_streams": num_streams,
        "pass": False,
        "error": error_message,
    }

    if include_device_metadata:
        metadata["device_id"] = device_id

    return Result(metadata=metadata)


def update_metrics_to_error_state(
    metrics: Dict[str, Metrics], value: int = -1, filter_prefix: Optional[str] = None
) -> None:
    """
    Update metrics dictionary to error state (modifies in-place).

    Args:
        metrics: Dictionary of Metrics objects to update
        value: Error value to set (default: -1)
        filter_prefix: Only update metrics starting with this prefix (default: None, updates all)
    """
    for metric_name in metrics:
        if filter_prefix is None or metric_name.startswith(filter_prefix):
            metrics[metric_name].value = value


def cleanup_stale_containers(docker_client, docker_container_prefix: str) -> None:
    """Removes any stale benchmark containers from previous runs."""
    logger.info("Removing stale containers...")
    stale_containers = docker_client.client.containers.list(all=True, filters={"name": f"{docker_container_prefix}*"})
    logger.debug(f"Stale containers found: {[container.name for container in stale_containers]}")
    for container in stale_containers:
        docker_client.cleanup_container(container.name)


def cleanup_thread_pool(thread_pool=None) -> None:
    """Clean up the thread pool."""
    global THREAD_POOL

    # Use global THREAD_POOL if no parameter provided
    if thread_pool is None:
        thread_pool = THREAD_POOL

    if thread_pool is None:
        logger.warning("Thread pool is already cleaned up or not initialized.")
        return

    logger.info("Cleaning up thread pool ...")
    for thread in thread_pool._threads:
        if thread.is_alive():
            logger.debug(f"Waiting for thread {thread.name} to finish ...")
            thread.join(timeout=30)

    if hasattr(thread_pool, "shutdown"):
        thread_pool.shutdown(wait=True)
    else:
        logger.warning("Thread pool is not initialized or already cleaned up.")


def cleanup_all(docker_client: DockerClient, container_prefix: str = "test-dlstreamer") -> None:
    """Comprehensive cleanup function."""
    cleanup_stale_containers(docker_client, container_prefix)
    cleanup_thread_pool()


def sort_devices_by_priority(baseline_num_streams, device_dict) -> Dict[str, Any]:
    """
    Sort devices by priority: discrete GPU -> integrated GPU -> NPU -> CPU
    Within each category, sort by device number for GPUs, else maintain original order.

    Special handling for multi-stage pipelines: returns baseline_num_streams as-is
    since "multi_stage" is a synthetic device ID not in device_dict.

    Args:
        baseline_num_streams: Dictionary with device info
        device_dict: Dictionary containing device information

    Returns:
        dict: Sorted dictionary
    """
    # Multi-stage mode: skip sorting since "multi_stage" is not a real device
    if "multi_stage" in baseline_num_streams:
        logger.debug("Multi-stage mode detected, skipping device priority sorting")
        return baseline_num_streams

    def get_device_priority(item):
        device_id, _ = item
        device_type = device_dict[device_id]["device_type"]

        device_type = str(device_type).lower() if device_type else ""

        # Extract device category and number
        if device_id.startswith("GPU"):
            category = "GPU"
            # Extract device number (e.g., GPU.1 -> 1, GPU.0 -> 0)
            try:
                dev_num = int(device_id.split(".")[1])
            except Exception:
                dev_num = -1
        elif device_id.startswith("NPU"):
            category = "NPU"
            dev_num = 0
        elif device_id.startswith("CPU"):
            category = "CPU"
            dev_num = 0
        else:
            category = "OTHER"
            dev_num = 0

        # Priority mapping: lower number = higher priority
        if category == "GPU" and "discrete" in device_type:
            priority = 0
        elif category == "GPU" and "integrated" in device_type:
            priority = 1
        elif category == "NPU":
            priority = 2
        elif category == "CPU":
            priority = 3
        else:
            priority = 4

        # For GPUs, sort by device number within category (higher device number first for dGPU)
        return (priority, -dev_num if category == "GPU" else dev_num)

    # Sort the items by priority (higher priority first)
    sorted_items = sorted(baseline_num_streams.items(), key=get_device_priority)
    sorted_dict = dict(sorted_items)

    logger.debug(f"Sorted devices by priority: {list(sorted_dict.keys())}")
    return sorted_dict


def update_device_metrics(
    active_devices: Dict[str, Any], device_id: str, results_dir: str, num_sockets: int, target_fps: float
):
    """
    Update metrics for all active devices from the latest result files and write back
    the updated metrics to ensure consistency across devices.

    Args:
        active_devices: Dict of active devices and their current data
        device_id: Current device being qualified
        results_dir: Directory containing result files
        num_sockets: Number of CPU sockets for multi-socket handling
        target_fps: Target FPS for pass/fail determination
    """
    if not active_devices:
        logger.debug("No active devices to update")
        return

    logger.debug("Updating device metrics for all active devices based on latest concurrent run results")

    for dev_id in active_devices.keys():
        result_file = os.path.join(results_dir, f"streams_analysis_{dev_id}.json")

        if os.path.exists(result_file):
            try:
                with open(result_file, "r") as f:
                    latest_data = json.load(f)

                # Update the active device data with latest results
                if latest_data:
                    latest_num_streams = latest_data.get("num_streams", 0)
                    latest_fps = latest_data.get("per_stream_fps", 0)

                    # Determine pass/fail based on FPS for this device
                    pass_qualification = latest_fps >= target_fps if latest_fps > 0 else False

                    # Update active devices dict
                    active_devices[dev_id].update(
                        {
                            "num_streams": latest_num_streams,
                            "per_stream_fps": latest_fps,
                            "pass": pass_qualification,
                            "updated": True,
                        }
                    )

                    logger.debug(
                        f"Updated {dev_id}: {latest_num_streams} streams, "
                        f"{latest_fps:.2f} FPS, pass={pass_qualification}"
                    )

            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                logger.warning(f"Failed to update metrics for {dev_id}: {e}")
        else:
            logger.debug(f"No result file found for {dev_id} at {result_file}")


def read_device_result(
    device_id: str, results_dir: str, num_sockets: int = 1, aggregate_multi_socket: bool = True
) -> Dict[str, Any]:
    """
    Read device result from standardized result files.

    Handles both single-device and multi-socket CPU scenarios with consistent run_id scheme:
    - run_id=1: Socket 0 or single-device
    - run_id=2+: Additional sockets (multi-socket CPU only)

    Args:
        device_id: Device identifier
        results_dir: Directory containing result files
        num_sockets: Number of CPU sockets (default: 1)
        aggregate_multi_socket: If True, aggregate multi-socket results (default: True)

    Returns:
        Dictionary with device results including metadata, or error state if reading fails
    """
    is_multisocket = device_id == "CPU" and num_sockets > 1

    if is_multisocket and aggregate_multi_socket:
        # Multi-socket aggregation: read from run_id=1, 2, ...
        all_socket_fps = []
        total_streams = 0
        aggregated_fps_list = []
        max_duration = 0.0
        max_fps_counter_duration = 0.0
        all_successful = True

        logger.debug(f"[{device_id}] Reading multi-socket results from {num_sockets} sockets")

        for socket_idx in range(num_sockets):
            socket_run_id = 1 + socket_idx
            socket_path = os.path.join(results_dir, f"total_streams_result_{socket_run_id}_{device_id}.json")

            if not os.path.exists(socket_path):
                logger.warning(f"[{device_id}] Missing socket {socket_idx} result at {socket_path}")
                all_successful = False
                all_socket_fps.append(0)
                continue

            try:
                with open(socket_path, "r") as f:
                    socket_data = json.load(f)

                if device_id not in socket_data:
                    logger.warning(f"[{device_id}] No {device_id} data in socket {socket_idx} result")
                    all_successful = False
                    all_socket_fps.append(0)
                    continue

                socket_device_data = socket_data[device_id]
                socket_meta = socket_device_data.get("metadata", {})
                socket_fps_list = socket_meta.get("per_stream_fps_list", [])

                socket_per_stream_fps = socket_meta.get("per_stream_fps", 0)
                socket_num_streams = socket_meta.get("num_streams", 0)

                # Handle case where per_stream_fps_list is empty but we have valid per_stream_fps
                if not socket_fps_list and socket_num_streams > 0 and socket_per_stream_fps > 0:
                    # Reconstruct FPS list using per_stream_fps value
                    socket_fps_list = [socket_per_stream_fps] * socket_num_streams
                    logger.debug(
                        f"[{device_id}] Socket {socket_idx}: Reconstructed FPS list from per_stream_fps "
                        f"({socket_num_streams} x {socket_per_stream_fps})"
                    )

                logger.debug(
                    f"[{device_id}] Socket {socket_idx}: {socket_num_streams} streams, "
                    f"{socket_per_stream_fps} FPS, fps_list={socket_fps_list}"
                )

                all_socket_fps.append(socket_per_stream_fps)
                aggregated_fps_list.extend(socket_fps_list)
                total_streams += socket_num_streams
                max_duration = max(max_duration, socket_device_data.get("analysis_duration", 0.0))
                max_fps_counter_duration = max(max_fps_counter_duration, socket_meta.get("fps_counter_duration", 0.0))

            except Exception as e:
                logger.warning(f"[{device_id}] Failed to read socket {socket_idx} result: {e}")
                all_successful = False
                all_socket_fps.append(0)

        logger.debug(
            f"[{device_id}] Aggregation: all_successful={all_successful}, "
            f"total_streams={total_streams}, aggregated_fps_list_len={len(aggregated_fps_list)}"
        )

        if not all_successful or not aggregated_fps_list:
            return {
                "per_stream_fps": 0,
                "num_streams": 0,
                "error": "Failed to read multi-socket results",
            }

        # Return aggregated result
        total_fps = sum(aggregated_fps_list)
        avg_per_stream_fps = total_fps / len(aggregated_fps_list) if aggregated_fps_list else 0.0

        return {
            "per_stream_fps": min(all_socket_fps),  # Use minimum for conservative estimate
            "num_streams": total_streams,
            "metadata": {
                "parse_source": "average",
                "fps_counter_duration": max_fps_counter_duration,
                "per_stream_fps_list": aggregated_fps_list,
                "num_streams": len(aggregated_fps_list),
                "per_stream_fps": avg_per_stream_fps,
                "total_fps": total_fps,
            },
            "analysis_duration": max_duration,
            "analysis_status": "success",
        }
    else:
        # Single-device or non-aggregated read: use run_id=1
        result_path = os.path.join(results_dir, f"total_streams_result_1_{device_id}.json")

        if not os.path.exists(result_path):
            logger.warning(f"Result file not found: {result_path}")
            return {
                "per_stream_fps": 0,
                "num_streams": 0,
                "error": "Result file not found",
            }

        try:
            with open(result_path, "r") as f:
                result_data = json.load(f)

            if device_id not in result_data:
                logger.warning(f"No {device_id} data in result file")
                return {
                    "per_stream_fps": 0,
                    "num_streams": 0,
                    "error": "Device not found in result",
                }

            device_data = result_data[device_id]
            metadata = device_data.get("metadata", {})

            return {
                "per_stream_fps": metadata.get("per_stream_fps", 0),
                "num_streams": metadata.get("num_streams", 0),
                "metadata": metadata,
                "analysis_status": device_data.get("analysis_status", "unknown"),
                "analysis_duration": device_data.get("analysis_duration", 0.0),
                "main_process_exit_code": device_data.get("main_process_exit_code", 0),
                "result_process_exit_code": device_data.get("result_process_exit_code", 0),
            }
        except Exception as e:
            logger.error(f"Failed to read result file {result_path}: {e}")
            return {
                "per_stream_fps": 0,
                "num_streams": 0,
                "error": f"Failed to read result: {str(e)}",
            }


def wait_for_containers(containers: List[docker.models.containers.Container]) -> None:
    """
    Wait for all containers to complete execution.

    Args:
        containers: List of container objects to wait for
    """
    logger.info(f"Waiting for {len(containers)} containers to complete...")

    for container in containers:
        try:
            logger.debug(f"Waiting for container {container.name} to complete...")
            result = container.wait(timeout=300)  # 5 minute timeout
            logger.debug(f"Container {container.name} completed with exit code: {result.get('StatusCode', 'unknown')}")
        except Exception as e:
            logger.warning(f"Error waiting for container {container.name}: {e}")

    logger.info("All containers have completed")
