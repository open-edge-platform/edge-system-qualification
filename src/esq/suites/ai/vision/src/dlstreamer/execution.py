# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer test execution and result processing functions."""

import json
import logging
import time
from typing import Any, Dict

from sysagent.utils.core import Metrics, Result, get_metric_name_for_device
from sysagent.utils.system.ov_helper import get_openvino_device_type

from .concurrent import confirm_steady_state_concurrent_analysis
from .preparation import get_device_specific_docker_image
from .qualification import qualify_device
from .results import (
    finalize_device_metrics,
    process_device_results,
    update_device_pipeline_info,
    update_final_results_metadata,
    validate_final_streams_results,
)
from .utils import update_metrics_to_error_state

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
    consecutive_success_threshold: int,
    consecutive_failure_threshold: int,
    consecutive_timeout_threshold: int,
    steady_state_confirmation_threshold: int,
    max_streams_above_baseline: int,
    qualified_devices: Dict[str, Any],
    metrics=None,
    baseline_streams=None,
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
    # For multi-stage pipelines, device_type is "multi_stage" (not a real OpenVINO device)
    if device_id == "multi_stage":
        device_type = "multi_stage"
    else:
        device_type = get_openvino_device_type(device_id)

    estimated_streams = max(1, int(baseline_streams.get("num_streams", 1)))
    device_result = {
        "device_type": device_type,
        "pass": False,
        # Binary search state - used internally during qualification iterations
        "qualification_state": {
            "num_streams": estimated_streams,
            "per_stream_fps": 0,
            "last_successful_fps": 0,
        },
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
            consecutive_success_threshold=consecutive_success_threshold,
            consecutive_failure_threshold=consecutive_failure_threshold,
            consecutive_timeout_threshold=consecutive_timeout_threshold,
            max_streams_above_baseline=max_streams_above_baseline,
            container_config=container_config,
        )
        final_results[device_id] = device_result

        if is_qualified:
            logger.info(f"[{device_id}] ✓ PASSED qualification. Adding to active devices.")
            # Keep qualification_state during qualification for multi-device concurrent analysis
            # It will be removed later when storing in extended_metadata
            qualified_devices[device_id] = device_result
        else:
            logger.warning(f"[{device_id}] ✗ FAILED qualification. Continuing with next device.")

        logger.debug(f"Updating final Result after benchmark: {json.dumps(final_results, indent=2)}")
        # Update metrics and metadata if pass and per stream FPS is available
        # Read from metadata structure for measurements, qualification_state for qualification results
        metadata = device_result.get("metadata", {})
        qual_state = device_result.get("qualification_state", {})
        if is_qualified and metadata.get("per_stream_fps", 0) > 0:
            status = True
            num_streams = metadata.get("num_streams", -1)
            per_stream_fps = metadata.get("per_stream_fps", 0)
        else:
            status = False
            # Read num_streams from qualification_state to distinguish:
            # -1 = pipeline errors/timeouts, 0 = ran but didn't meet target FPS
            num_streams = qual_state.get("num_streams", -1)
            per_stream_fps = 0
            # Propagate error reason if available
            if "error_reason" in device_result:
                result.metadata["error"] = device_result["error_reason"]
            logger.debug(
                f"Device {device_id} failed qualification: num_streams from qualification_state = {num_streams}"
            )

        result.metrics.update(
            {get_metric_name_for_device(device_id, prefix="streams_max"): Metrics(unit="streams", value=num_streams)}
        )
        result.metadata.update(
            {
                "status": status,
                f"Average Per-Stream FPS - {device_id}": f"{per_stream_fps:.2f}",
            }
        )
        logger.debug(f"Final Result after benchmark: {json.dumps(result.to_dict(), indent=2)}")

        time.sleep(5)

        from .utils import cleanup_stale_containers

        cleanup_stale_containers(docker_client, docker_container_prefix)
    except KeyboardInterrupt:
        logger.warning("Test interrupted by user. Cleaning up containers.")

        # Set metrics to -1 to indicate interruption
        if metrics:
            update_metrics_to_error_state(metrics, value=-1, filter_prefix="streams_max")

        # Update result to reflect interruption
        result.metrics = metrics if metrics else result.metrics
        result.metadata["status"] = False
        result.metadata["error"] = "Test interrupted by user"

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


def run_multi_device_test(
    docker_client,
    device_dict: Dict[str, Any],
    sorted_devices: list,
    device_list: list,
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
    consecutive_success_threshold: int,
    consecutive_failure_threshold: int,
    consecutive_timeout_threshold: int,
    steady_state_confirmation_threshold: int,
    max_streams_above_baseline: int,
    baseline_streams: Dict[str, Any],
    container_config: Dict[str, Any],
    configs: Dict[str, Any],
    test_name: str,
    devices: list,
    timeout: int,
    test_display_name: str,
    metrics: Dict[str, Any],
    baseline_streams_results: list,
) -> Result:
    """
    Execute total streams analysis for all devices and return final result.

    This function orchestrates the multi-device test workflow:
    1. Run qualification for each device sequentially
    2. Confirm steady-state with concurrent analysis
    3. Build and return final consolidated result

    Args:
        docker_client: Docker client instance
        device_dict: Dictionary containing device information
        sorted_devices: List of device IDs in priority order
        device_list: List of all device IDs
        pipeline: Pipeline configuration string
        pipeline_params: Pipeline parameters dictionary
        docker_image_tag_analyzer: Docker image tag for analyzer
        docker_container_prefix: Prefix for container names
        data_dir: Data directory path
        container_mnt_dir: Container mount directory
        pipeline_timeout: Timeout for pipeline execution
        results_dir: Directory for result files
        target_fps: Target FPS threshold
        num_sockets: Number of CPU sockets
        consecutive_success_threshold: Success threshold for binary search
        consecutive_failure_threshold: Failure threshold for binary search
        consecutive_timeout_threshold: Timeout threshold for binary search
        steady_state_confirmation_threshold: Confirmation threshold for steady-state
        max_streams_above_baseline: Maximum streams to explore above baseline
        baseline_streams: Dictionary of baseline stream information per device
        container_config: Container configuration with available images
        configs: Test configuration dictionary
        test_name: Test name
        devices: List of device categories
        timeout: Overall test timeout
        test_display_name: Display name for the test
        metrics: Default metrics dictionary
        baseline_streams_results: List of baseline preparation results

    Returns:
        Result object with consolidated test outcomes
    """
    import json
    import os

    from sysagent.utils.core import Metrics, Result, get_metric_name_for_device

    local_qualified_devices = {}
    local_failed_devices = {}

    # Run total streams analysis for all devices in device_list
    for device_id in sorted_devices:
        logger.info(f"\n{'=' * 60}\nProcessing device: {device_id}\n{'=' * 60}")

        # Prepare device-specific configurations
        metric_name = get_metric_name_for_device(device_id, prefix="streams_max")
        default_metrics = {metric_name: Metrics(unit="streams", value=-1)}

        # Execute device test WITHOUT caching at device level
        # The overall multi-device result will be cached instead
        result = run_device_test(
            docker_client=docker_client,
            device_dict=device_dict,
            device_id=device_id,
            pipeline=pipeline,
            pipeline_params=pipeline_params,
            docker_image_tag_analyzer=docker_image_tag_analyzer,
            docker_container_prefix=docker_container_prefix,
            data_dir=data_dir,
            container_mnt_dir=container_mnt_dir,
            pipeline_timeout=pipeline_timeout,
            results_dir=results_dir,
            target_fps=target_fps,
            num_sockets=num_sockets,
            consecutive_success_threshold=consecutive_success_threshold,
            consecutive_failure_threshold=consecutive_failure_threshold,
            consecutive_timeout_threshold=consecutive_timeout_threshold,
            steady_state_confirmation_threshold=steady_state_confirmation_threshold,
            max_streams_above_baseline=max_streams_above_baseline,
            qualified_devices=local_qualified_devices,
            metrics=default_metrics,
            baseline_streams=baseline_streams.get(device_id, {}),
            container_config=container_config,
        )

        if result.metadata.get("status", False):
            # Device test succeeded - read complete result file directly
            # (including qualification_state for concurrent analysis)
            result_file = os.path.join(results_dir, f"total_streams_result_0_{device_id}.json")

            if os.path.exists(result_file):
                try:
                    with open(result_file, "r") as f:
                        result_data = json.load(f)

                    if device_id in result_data:
                        device_result = result_data[device_id]

                        # Check 'pass' field to determine if device qualified
                        if device_result.get("pass", False):
                            # Extract num_streams from result metadata for logging
                            device_metadata = device_result.get("metadata", {})
                            device_streams = device_metadata.get("num_streams", -1)
                            stream_fps = device_metadata.get("per_stream_fps", 0.0)

                            # Keep full result with qualification_state for multi-device concurrent analysis
                            local_qualified_devices[device_id] = device_result
                            logger.info(
                                f"Device {device_id} qualified with {device_streams} streams at {stream_fps:.2f} FPS"
                            )
                        else:
                            error_reason = device_result.get("error_reason", "Device failed qualification")
                            logger.error(f"Device {device_id} failed qualification: {error_reason}")

                            # Read num_streams from qualification_state to distinguish between:
                            # -1 = pipeline errors/timeouts, 0 = ran but didn't meet target FPS
                            qual_state = device_result.get("qualification_state", {})
                            num_streams = qual_state.get("num_streams", -1)
                            logger.debug(
                                f"Device {device_id} failed: reading num_streams = {num_streams} from qualification_state"
                            )

                            local_failed_devices[device_id] = {
                                "error_reason": error_reason,
                                "num_streams": num_streams,
                                "pass": False,
                            }
                            continue
                    else:
                        error_reason = f"Device {device_id} not found in result file"
                        logger.error(error_reason)
                        local_failed_devices[device_id] = {
                            "error_reason": error_reason,
                            "num_streams": -1,  # True error - device not in result
                            "pass": False,
                        }
                        continue
                except Exception as e:
                    error_reason = f"Failed to read result file: {str(e)}"
                    logger.error(f"Error reading results for device {device_id}: {error_reason}")
                    local_failed_devices[device_id] = {
                        "error_reason": error_reason,
                        "num_streams": -1,  # True error - couldn't read file
                        "pass": False,
                    }
                    continue
            else:
                error_reason = result.metadata.get("error", "Failed to parse device result")
                logger.error(f"Failed to parse results for device {device_id}: {error_reason}")
                local_failed_devices[device_id] = {
                    "error_reason": error_reason,
                    "num_streams": -1,  # True error - no result file
                    "pass": False,
                }
                continue
        else:
            error_reason = result.metadata.get("error", "Unknown error")
            logger.error(f"Test failed for device {device_id}: {error_reason}")

            # Even if status is False, try to read result file to get qualification_state
            # This allows us to distinguish between errors (-1) and performance failures (0)
            result_file = os.path.join(results_dir, f"total_streams_result_0_{device_id}.json")
            num_streams = -1  # Default to error

            if os.path.exists(result_file):
                try:
                    with open(result_file, "r") as f:
                        result_data = json.load(f)
                    if device_id in result_data:
                        device_result = result_data[device_id]
                        qual_state = device_result.get("qualification_state", {})
                        num_streams = qual_state.get("num_streams", -1)
                        logger.debug(
                            f"Device {device_id} test failed but qualification ran: "
                            f"num_streams from qualification_state = {num_streams}"
                        )
                except Exception as e:
                    logger.debug(f"Could not read qualification_state for {device_id}: {e}")

            # Store failed device information for error reporting
            local_failed_devices[device_id] = {
                "error_reason": error_reason,
                "num_streams": num_streams,
                "pass": False,
            }
            continue

    # After all devices qualified, confirm steady-state with concurrent analysis
    steady_state_confirmed = confirm_steady_state_concurrent_analysis(
        docker_client=docker_client,
        device_dict=device_dict,
        qualified_devices=local_qualified_devices,
        pipeline=pipeline,
        pipeline_params=pipeline_params,
        docker_image_tag_analyzer=docker_image_tag_analyzer,
        docker_container_prefix=docker_container_prefix,
        data_dir=data_dir,
        container_mnt_dir=container_mnt_dir,
        pipeline_timeout=pipeline_timeout,
        results_dir=results_dir,
        target_fps=target_fps,
        num_sockets=num_sockets,
        confirmation_threshold=steady_state_confirmation_threshold,
        container_config=container_config,
    )

    if not steady_state_confirmed:
        logger.debug("Steady-state not fully confirmed but proceeding with last known stable configuration")

    # Build final result with all device metrics
    final_result = Result.from_test_config(
        configs=configs,
        name=test_name,
        parameters={
            "Devices": devices,
            "Device List": device_list,
            "Target FPS": target_fps,
            "Consecutive Success Threshold": consecutive_success_threshold,
            "Consecutive Failure Threshold": consecutive_failure_threshold,
            "Max Streams Above Baseline": max_streams_above_baseline,
            "Timeout (s)": timeout,
            "Display Name": test_display_name,
            "Pipeline": pipeline,
            "Pipeline Params": pipeline_params,
        },
        metrics=metrics,
        metadata={
            "status": True,
        },
    )

    # Store structured device data in extended_metadata for analysis and visualization
    # Remove internal qualification_state from qualified devices for clean final results
    clean_qualified_devices = {
        dev_id: {k: v for k, v in dev_data.items() if k != "qualification_state"}
        for dev_id, dev_data in local_qualified_devices.items()
    }
    final_result.extended_metadata = {
        "qualified_devices": clean_qualified_devices,
        "failed_devices": local_failed_devices,
    }

    # Process device results using modular function
    process_device_results(device_list, local_qualified_devices, final_result)

    # Finalize device metrics using modular function (pass failed_devices for proper metric updates)
    finalize_device_metrics(final_result, local_qualified_devices, device_list, local_failed_devices)

    # Validate final streams results using modular function
    validate_final_streams_results(final_result, local_qualified_devices, device_list, local_failed_devices)

    # Update pipeline information for qualified devices using modular function
    update_device_pipeline_info(final_result, local_qualified_devices, pipeline, pipeline_params, device_dict)

    # Update final results metadata using modular function
    update_final_results_metadata(
        final_result,
        local_qualified_devices,
        device_list,
        baseline_streams_results,
        requested_device_categories=devices,
        target_fps=target_fps,
    )

    logger.debug(f"DL Streamer Test results: {json.dumps(final_result.to_dict(), indent=2)}")
    return final_result
