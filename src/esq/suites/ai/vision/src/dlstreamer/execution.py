# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer test execution and result processing functions."""

import json
import logging
import time
from typing import Any, Dict

from sysagent.utils.core import Metrics, Result, get_metric_name_for_device
from sysagent.utils.system.ov_helper import get_openvino_device_type

from .preparation import get_device_specific_docker_image
from .qualification import qualify_device

logger = logging.getLogger(__name__)


def run_device_test(
    docker_client,
    device_dict: Dict[str, Any],
    device_id: str,
    pipeline: str,
    pipeline_params: Dict[str, Dict[str, str]],
    docker_image_tag_analyzer: str,
    docker_container_prefix: str,
    data_dir: str,
    container_mnt_dir: str,
    pipeline_timeout: int,
    results_dir: str,
    target_fps: float,
    num_sockets: int,
    max_plateau_iterations: int,
    qualified_devices: Dict[str, Any],
    metrics=None,
    baseline_streams=None,
    visualize_stream: bool = False,
    container_config: Dict[str, Any] = None,
) -> Result:
    """
    Execute DL Streamer test for a single device.

    Args:
        device_id: Device ID to test
        pipeline: Pipeline string to use
        pipeline_params: Pipeline parameters
        metrics: Default metrics for the device
        baseline_streams: Baseline stream information
        visualize_stream: Whether to visualize the stream
        container_config: Container configuration with available images (optional)

    Returns:
        Result object with test outcomes
    """
    logger.info(f"Executing DL Streamer test for device: {device_id}")

    # Select device-specific Docker image
    if container_config:
        docker_image_tag_analyzer = get_device_specific_docker_image(
            device_id, container_config, docker_image_tag_analyzer, device_dict
        )

    result = Result(
        parameters={
            "Device ID": device_id,
        },
        metrics=metrics,
        metadata={
            "status": False,
        },
    )
    logger.debug(f"Initial Result template: {json.dumps(result.to_dict(), indent=2)}")

    logger.debug(f"Baseline streams for device {device_id}: {json.dumps(baseline_streams, indent=2)}")

    # Initial guess for stream count based on baseline
    device_type = get_openvino_device_type(device_id)
    estimated_streams = max(1, int(baseline_streams.get("num_streams", 1)))
    device_result = {
        "device_type": device_type,
        "num_streams": estimated_streams,
        "per_stream_fps": 0,
        "pass": False,
    }

    # Store original qualified devices to restore on error
    final_results = {}

    try:
        is_qualified = qualify_device(
            docker_client=docker_client,
            device_dict=device_dict,
            device_id=device_id,
            device_data=device_result,
            active_devices=qualified_devices,
            target_fps=target_fps,
            pipeline=pipeline,
            pipeline_params=pipeline_params,
            docker_image_tag_analyzer=docker_image_tag_analyzer,
            docker_container_prefix=docker_container_prefix,
            data_dir=data_dir,
            container_mnt_dir=container_mnt_dir,
            pipeline_timeout=pipeline_timeout,
            results_dir=results_dir,
            num_sockets=num_sockets,
            max_plateau_iterations=max_plateau_iterations,
            visualize_stream=visualize_stream,
            container_config=container_config,
        )
        final_results[device_id] = device_result

        if is_qualified:
            logger.info(f"[{device_id}] ✓ PASSED qualification. Adding to active devices.")
            qualified_devices[device_id] = device_result
        else:
            logger.warning(f"[{device_id}] ✗ FAILED qualification. Continuing with next device.")

        logger.debug(f"Updating final Result after benchmark: {json.dumps(final_results, indent=2)}")
        # Update metrics and metadata if pass and per stream FPS is available
        if is_qualified and device_result.get("per_stream_fps", 0) > 0:
            status = True
            num_streams = device_result.get("num_streams", -1)
            per_stream_fps = device_result.get("per_stream_fps", -1)
        else:
            status = False
            num_streams = -1
            per_stream_fps = -1

        result.metrics.update(
            {get_metric_name_for_device(device_id, prefix="streams_max"): Metrics(unit="streams", value=num_streams)}
        )
        result.metadata.update(
            {
                "status": status,
                "Number of Streams": num_streams,
                "Per Stream FPS": per_stream_fps,
            }
        )
        logger.debug(f"Final Result after benchmark: {json.dumps(result.to_dict(), indent=2)}")

        time.sleep(5)

        from .utils import cleanup_stale_containers

        cleanup_stale_containers(docker_client, docker_container_prefix)
    except KeyboardInterrupt:
        logger.warning("Test interrupted by user. Cleaning up containers.")
        from .utils import cleanup_stale_containers, cleanup_thread_pool

        cleanup_stale_containers(docker_client, docker_container_prefix)
        cleanup_thread_pool()
        raise
    except Exception as e:
        error_message = f"An error occurred during the test execution for device {device_id}: {str(e)}"
        logger.error(error_message)
        from .utils import cleanup_stale_containers, cleanup_thread_pool

        cleanup_stale_containers(docker_client, docker_container_prefix)
        cleanup_thread_pool()
        import pytest

        pytest.fail(error_message)

    return result


def process_device_results(device_list: list, qualified_devices: Dict[str, Any], results: Result) -> None:
    """
    Process and update results metadata for all qualified devices.
    Individual device metrics and aggregate metrics are handled separately in the main test.

    Args:
        device_list: List of device IDs
        qualified_devices: Dictionary of qualified device results
        results: Main results object to update
    """
    logger.info("Updating device metadata with latest concurrent run metrics that passed qualification")

    for dev_id, dev_data in qualified_devices.items():
        # Only update with devices that have passed qualification
        if dev_data.get("pass", False):
            # Update the results metadata with the latest qualified device metrics
            latest_streams = dev_data.get("num_streams", -1)
            latest_fps = dev_data.get("per_stream_fps", -1)

            # Update metadata
            results.metadata[f"Maximum Streams {dev_id}"] = latest_streams
            results.metadata[f"Stream FPS {dev_id}"] = latest_fps

            logger.info(f"Updated device {dev_id}: {latest_streams} streams at {latest_fps:.2f} FPS")

    logger.info(f"Final qualified devices: {len(qualified_devices)} of {len(device_list)} devices passed")


def update_device_pipeline_info(
    results: Result,
    qualified_devices: Dict[str, Any],
    pipeline: str,
    pipeline_params: Dict[str, Any],
    device_dict: Dict[str, Any],
) -> None:
    """
    Update results with pipeline information for all qualified devices.

    Args:
        results: Main results object to update
        qualified_devices: Dictionary of qualified device results
        pipeline: Pipeline configuration string
        pipeline_params: Pipeline parameters dictionary
        device_dict: Dictionary containing device information
    """
    from .pipeline import get_pipeline_info

    logger.info("Updating pipeline information for qualified devices")

    for dev_id, dev_data in qualified_devices.items():
        if not dev_data.get("pass", False) or dev_data.get("num_streams", 0) <= 0:
            logger.warning(f"Device {dev_id} did not qualify with any streams. Skipping pipeline info.")
            continue

        num_streams = dev_data.get("num_streams", 0)

        try:
            # Get baseline pipeline information
            base_pipeline_info = get_pipeline_info(
                device_id=dev_id,
                pipeline=pipeline,
                pipeline_params=pipeline_params,
                device_dict=device_dict,
                num_streams=num_streams,
                is_baseline=True,
            )

            # Get pipeline information for this device
            multi_pipeline_info = get_pipeline_info(
                device_id=dev_id,
                pipeline=pipeline,
                pipeline_params=pipeline_params,
                device_dict=device_dict,
                num_streams=num_streams,
                is_baseline=False,
            )

            # Update results parameters with pipeline information
            results.parameters[f"Base Pipeline {dev_id}"] = base_pipeline_info.get("baseline_pipeline", "")
            results.parameters[f"Base Result Pipeline {dev_id}"] = base_pipeline_info.get("result_pipeline", "")
            results.parameters[f"Multi Pipeline {dev_id}"] = multi_pipeline_info.get("multi_pipeline", "")
            results.parameters[f"Multi Result Pipeline {dev_id}"] = multi_pipeline_info.get("result_pipeline", "")

            logger.debug(f"Updated pipeline info for device {dev_id}")

        except Exception as e:
            logger.warning(f"Failed to get pipeline info for device {dev_id}: {e}")
            continue


def validate_final_streams_results(results: Result, qualified_devices: Dict[str, Any], device_list: list) -> bool:
    """
    Validate final streams results and update error status if needed.

    Args:
        results: Main results object to update
        qualified_devices: Dictionary of qualified device results
        device_list: List of all device IDs tested

    Returns:
        True if validation passed, False if failed
    """
    # Check if total streams is 0 or less for both single and multiple device scenarios
    final_total_streams = 0

    if len(device_list) > 1 and "streams_max" in results.metrics:
        final_total_streams = results.metrics["streams_max"].value
    else:
        # For single device, check the individual device metric
        for metric_name, metric_obj in results.metrics.items():
            if metric_name.startswith("streams_max") and metric_obj.value > 0:
                final_total_streams += metric_obj.value

    if final_total_streams <= 0:
        qualified_device_names = [
            dev_id for dev_id, dev_data in qualified_devices.items() if dev_data.get("pass", False)
        ]
        failed_device_names = [dev_id for dev_id in device_list if dev_id not in qualified_device_names]

        if len(device_list) > 1:
            error_message = (
                f"No devices qualified for multi-device testing. "
                f"All devices failed qualification: {failed_device_names}"
            )
        else:
            error_message = f"Device failed qualification. No streams achieved: {failed_device_names}"

        logger.error(error_message)
        results.metadata["status"] = False
        results.metadata["error"] = error_message
        return False

    logger.info(f"Final streams validation passed: {final_total_streams} total streams")
    return True


def finalize_device_metrics(results: Result, qualified_devices: Dict[str, Any], device_list: list) -> None:
    """
    Finalize device metrics by ensuring individual device metrics are preserved
    and updating aggregate metrics for multi-device scenarios.

    Args:
        results: Main results object to update
        qualified_devices: Dictionary of qualified device results
        device_list: List of all device IDs tested
    """
    logger.info("Finalizing device metrics")

    # Always ensure individual device metrics are updated for all qualified devices
    for dev_id, dev_data in qualified_devices.items():
        if dev_data.get("pass", False):
            device_metric_name = get_metric_name_for_device(dev_id, prefix="streams_max")
            device_streams = dev_data.get("num_streams", 0)

            # Ensure individual device metric exists and is updated
            if device_metric_name not in results.metrics:
                results.metrics[device_metric_name] = Metrics(unit="streams", value=device_streams)
                logger.debug(f"Added missing individual device metric {device_metric_name} = {device_streams}")
            else:
                results.metrics[device_metric_name].value = device_streams
                logger.debug(f"Updated individual device metric {device_metric_name} = {device_streams}")

    # For multiple devices, also update aggregate streams_max if it exists or create it
    if len(device_list) > 1:
        total_streams = sum(
            dev_data.get("num_streams", 0)
            for dev_id, dev_data in qualified_devices.items()
            if dev_data.get("pass", False)
        )

        if "streams_max" in results.metrics:
            results.metrics["streams_max"].value = total_streams
            logger.info(f"Updated aggregate streams_max for multi-device scenario: {total_streams}")
        else:
            # Create aggregate streams_max metric if it doesn't exist but we have multiple devices
            results.metrics["streams_max"] = Metrics(unit="streams", value=total_streams)
            logger.info(f"Created aggregate streams_max for multi-device scenario: {total_streams}")

    logger.debug(f"Finalized metrics for {len(qualified_devices)} qualified devices")


def update_final_results_metadata(
    results: Result,
    qualified_devices: Dict[str, Any],
    device_list: list,
    baseline_streams_results: list = None,
) -> None:
    """
    Update final results with device metadata and summary information.
    Uses the enhanced auto_set_key_metric function instead of hardcoded logic.

    Args:
        results: Main results object to update
        qualified_devices: Dictionary of qualified device results
        device_list: List of all device IDs tested
        baseline_streams_results: List of baseline preparation results (optional)
    """
    if baseline_streams_results:
        baseline_fps_count = 0
        for baseline_result in baseline_streams_results:
            device_id = baseline_result.metadata.get("device_id")
            per_stream_fps = baseline_result.metadata.get("per_stream_fps", 0.0)
            if device_id:
                results.metadata[f"Baseline Pipeline Throughput (FPS) - {device_id}"] = f"{per_stream_fps:.2f}"
                baseline_fps_count += 1
                logger.debug(f"Adding baseline pipeline throughput FPS for {device_id}: {per_stream_fps:.2f}")

        if baseline_fps_count > 0:
            logger.info(f"Baseline per-stream FPS captured for {baseline_fps_count} device(s)")

    # Update overall test status
    if qualified_devices:
        results.metadata["status"] = True

        # Add summary information
        results.metadata["qualified_devices_count"] = len(qualified_devices)
        results.metadata["total_devices_count"] = len(device_list)
        results.metadata["qualification_rate"] = len(qualified_devices) / len(device_list) if device_list else 0

        # Calculate total streams
        total_streams = 0
        for device_id, device_data in qualified_devices.items():
            total_streams += device_data.get("num_streams", 0)

        results.metadata["total_qualified_streams"] = total_streams

        # Use the enhanced auto_set_key_metric function instead of hardcoded logic
        results.auto_set_key_metric(device_count=len(device_list))

        logger.info(f"Test completed successfully: {len(qualified_devices)}/{len(device_list)} devices qualified")
        logger.info(f"Total qualified streams: {total_streams}")
    else:
        results.metadata["status"] = False
        results.metadata["error"] = "No devices passed qualification"
        logger.warning("Test failed: No devices passed qualification")
