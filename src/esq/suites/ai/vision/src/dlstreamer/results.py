# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer result processing and validation functions."""

import logging
from typing import Any, Dict

from sysagent.utils.core import Metrics, Result, get_metric_name_for_device

logger = logging.getLogger(__name__)


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
            # Extract metrics from metadata (already populated from qualification_state by analysis.py)
            device_metadata = dev_data.get("metadata", {})
            latest_streams = device_metadata.get("num_streams", 0)
            latest_fps = device_metadata.get("per_stream_fps", 0)

            # Update metadata (Maximum Streams removed as it's redundant with metrics)
            results.metadata[f"Average Per-Stream FPS - {dev_id}"] = f"{latest_fps:.2f}"

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
        device_metadata = dev_data.get("metadata", {})
        num_streams = device_metadata.get("num_streams", 0)

        if not dev_data.get("pass", False) or num_streams <= 0:
            logger.warning(f"Device {dev_id} did not qualify with any streams. Skipping pipeline info.")
            continue

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


def validate_final_streams_results(
    results: Result,
    qualified_devices: Dict[str, Any],
    device_list: list,
    failed_devices: Dict[str, Any] = None,
) -> bool:
    """
    Validate final streams results and update error status if needed.

    Args:
        results: Main results object to update
        qualified_devices: Dictionary of qualified device results
        device_list: List of all device IDs tested
        failed_devices: Dictionary of failed device results with error_reason (optional)

    Returns:
        True if validation passed, False if failed
    """
    # Check if total streams is 0 or less (including -1 for errors)
    final_total_streams = 0

    if len(device_list) > 1 and "streams_max" in results.metrics:
        final_total_streams = results.metrics["streams_max"].value
    else:
        # For single device, check the individual device metric
        for metric_name, metric_obj in results.metrics.items():
            if metric_name.startswith("streams_max") and metric_obj.value > 0:
                final_total_streams += metric_obj.value

    # Consider it a failure if total streams is 0 or negative (including -1 for errors)
    if final_total_streams <= 0:
        qualified_device_names = [
            dev_id for dev_id, dev_data in qualified_devices.items() if dev_data.get("pass", False)
        ]
        failed_device_names = [dev_id for dev_id in device_list if dev_id not in qualified_device_names]

        # Build detailed error message with device-specific error reasons
        error_details = []
        if failed_devices:
            for dev_id in failed_device_names:
                if dev_id in failed_devices and "error_reason" in failed_devices[dev_id]:
                    error_details.append(f"{dev_id}: {failed_devices[dev_id]['error_reason']}")
                else:
                    error_details.append(f"{dev_id}: Unknown error")

        if len(device_list) > 1:
            base_message = (
                f"No devices qualified for multi-device testing. "
                f"All devices failed qualification: {failed_device_names}"
            )
        else:
            base_message = f"Device failed qualification. No streams achieved: {failed_device_names}"

        # Append detailed error reasons if available
        if error_details:
            error_message = base_message + ". Details: " + "; ".join(error_details)
        else:
            error_message = base_message

        logger.error(error_message)
        results.metadata["status"] = False
        results.metadata["error"] = error_message
        return False

    logger.info(f"Final streams validation passed: {final_total_streams} total streams")
    return True


def finalize_device_metrics(
    results: Result, qualified_devices: Dict[str, Any], device_list: list, failed_devices: Dict[str, Any] = None
) -> None:
    """
    Finalize device metrics by ensuring individual device metrics are preserved
    and updating aggregate metrics for multi-device scenarios.

    Args:
        results: Main results object to update
        qualified_devices: Dictionary of qualified device results
        device_list: List of all device IDs tested
        failed_devices: Dictionary of failed device results with num_streams (optional)
    """
    logger.info("Finalizing device metrics")

    # Always ensure individual device metrics are updated for all qualified devices
    for dev_id, dev_data in qualified_devices.items():
        if dev_data.get("pass", False):
            device_metric_name = get_metric_name_for_device(dev_id, prefix="streams_max")
            device_metadata = dev_data.get("metadata", {})
            device_streams = device_metadata.get("num_streams", -1)

            # Ensure individual device metric exists and is updated
            if device_metric_name not in results.metrics:
                results.metrics[device_metric_name] = Metrics(unit="streams", value=device_streams)
                logger.debug(f"Added missing individual device metric {device_metric_name} = {device_streams}")
            else:
                results.metrics[device_metric_name].value = device_streams
                logger.debug(f"Updated individual device metric {device_metric_name} = {device_streams}")

    # Update metrics for failed devices using their actual num_streams value
    # This distinguishes between errors (-1) and not meeting target FPS (0)
    if failed_devices:
        for dev_id, failed_data in failed_devices.items():
            device_metric_name = get_metric_name_for_device(dev_id, prefix="streams_max")
            # Get num_streams from failed device data (0 = ran but failed, -1 = error)
            device_streams = failed_data.get("num_streams", -1)

            if device_metric_name not in results.metrics:
                results.metrics[device_metric_name] = Metrics(unit="streams", value=device_streams)
                logger.debug(
                    f"Added missing failed device metric {device_metric_name} = {device_streams} "
                    f"({'error' if device_streams == -1 else 'ran but did not meet target'})"
                )
            else:
                results.metrics[device_metric_name].value = device_streams
                logger.debug(
                    f"Updated failed device metric {device_metric_name} = {device_streams} "
                    f"({'error' if device_streams == -1 else 'ran but did not meet target'})"
                )

    # Update aggregate streams_max if it exists (for both single and multi-device scenarios)
    if "streams_max" in results.metrics:
        # Only sum positive values (skip -1 errors and 0 failures)
        total_streams = 0
        for dev_id, dev_data in qualified_devices.items():
            if dev_data.get("pass", False):
                device_metadata = dev_data.get("metadata", {})
                device_streams = device_metadata.get("num_streams", 0)

                if device_streams > 0:
                    total_streams += device_streams

        results.metrics["streams_max"].value = total_streams
        if len(device_list) > 1:
            logger.info(f"Updated aggregate streams_max for multi-device scenario: {total_streams}")
        else:
            logger.info(f"Updated aggregate streams_max for single-device scenario: {total_streams}")

    logger.debug(f"Finalized metrics for {len(qualified_devices)} qualified devices")


def update_final_results_metadata(
    results: Result,
    qualified_devices: Dict[str, Any],
    device_list: list,
    baseline_streams_results: list = None,
    requested_device_categories: list = None,
    target_fps: float = None,
) -> None:
    """
    Update final results with device metadata and summary information.
    Uses the enhanced auto_set_key_metric function instead of hardcoded logic.

    Args:
        results: Main results object to update
        qualified_devices: Dictionary of qualified device results
        device_list: List of all device IDs tested
        baseline_streams_results: List of baseline preparation results (optional)
        requested_device_categories: List of device categories requested in config (optional)
        target_fps: Target FPS for stream qualification (optional)
    """
    # Add target FPS to metadata
    if target_fps is not None:
        results.metadata["Target FPS"] = target_fps
        logger.debug(f"Adding target FPS to metadata: {target_fps}")

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

    # Determine effective device count for key metric selection before status check
    # Treat as multi-device if:
    # 1. Multiple devices detected, OR
    # 2. Multiple device categories requested in config, OR
    # 3. Multiple dGPU devices detected (GPU.0, GPU.1, etc.)
    dgpu_count = sum(1 for dev_id in device_list if dev_id.upper().startswith("GPU."))
    effective_device_count = max(
        len(device_list),
        len(requested_device_categories) if requested_device_categories else 0,
        dgpu_count if dgpu_count > 1 else 0,
    )

    # Update overall test status
    if qualified_devices:
        results.metadata["status"] = True

        # Use the enhanced auto_set_key_metric function instead of hardcoded logic
        results.auto_set_key_metric(device_count=effective_device_count)

        # Calculate total streams for logging
        total_streams = sum(
            device_data.get("metadata", {}).get("num_streams", 0) for device_data in qualified_devices.values()
        )

        logger.info(f"Test completed successfully: {len(qualified_devices)}/{len(device_list)} devices qualified")
        logger.info(f"Total qualified streams: {total_streams}")
    else:
        results.metadata["status"] = False
        # Only set generic error if no specific error already exists
        if "error" not in results.metadata or not results.metadata["error"]:
            results.metadata["error"] = "No devices passed qualification"
        logger.warning("Test failed: No devices passed qualification")

        # Set key metric even when test fails (e.g., for streams_max = 0)
        # Use same effective_device_count calculated above
        results.auto_set_key_metric(device_count=effective_device_count)
