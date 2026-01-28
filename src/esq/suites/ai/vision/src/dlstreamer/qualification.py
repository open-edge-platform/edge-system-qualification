# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer device qualification orchestration."""

import json
import logging
import os
import time
from typing import Any, Dict, Tuple

from .concurrent import run_concurrent_analysis
from .utils import read_device_result, update_device_metrics

logger = logging.getLogger(__name__)


def _read_essential_result_with_retry(
    device_id: str,
    results_dir: str,
    num_sockets: int = 1,
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> Tuple[float, str, int]:
    """
    Lightweight result reader for qualification loop with retry logic.

    Reads only essential data needed for binary search decisions:
    - per_stream_fps: For target FPS comparison
    - analysis_status: For timeout detection
    - num_streams: For actual stream count (multi-socket)

    This avoids reading large metadata objects (per_stream_fps_list, etc.) during
    each qualification iteration, reducing file I/O overhead and contention.

    Args:
        device_id: Device identifier
        results_dir: Directory containing result files
        num_sockets: Number of CPU sockets
        max_retries: Maximum number of retry attempts for file reads
        retry_delay: Initial delay between retries (exponential backoff)

    Returns:
        Tuple of (per_stream_fps, analysis_status, actual_streams_run)
    """
    is_multisocket = device_id == "CPU" and num_sockets > 1

    for attempt in range(max_retries):
        try:
            if is_multisocket:
                # Multi-socket: read only FPS from each socket, skip heavy metadata
                all_socket_fps = []
                total_actual_streams = 0
                all_status_success = True

                for socket_idx in range(num_sockets):
                    socket_run_id = 1 + socket_idx
                    socket_path = os.path.join(results_dir, f"total_streams_result_{socket_run_id}_{device_id}.json")

                    if not os.path.exists(socket_path):
                        logger.debug(f"Socket {socket_idx} result not yet available: {socket_path}")
                        all_socket_fps.append(0)
                        all_status_success = False
                        continue

                    with open(socket_path, "r") as f:
                        socket_data = json.load(f)

                    if device_id not in socket_data:
                        all_socket_fps.append(0)
                        all_status_success = False
                        continue

                    device_data = socket_data[device_id]
                    # Read only essential fields - skip per_stream_fps_list to reduce I/O
                    metadata = device_data.get("metadata", {})
                    all_socket_fps.append(metadata.get("per_stream_fps", 0))
                    total_actual_streams += metadata.get("num_streams", 0)

                    socket_status = device_data.get("analysis_status", "unknown")
                    if socket_status != "success":
                        all_status_success = False

                # Return minimum FPS across sockets for conservative estimate
                min_fps = min(all_socket_fps) if all_socket_fps else 0
                overall_status = "success" if all_status_success else "timeout"

                return (min_fps, overall_status, total_actual_streams)
            else:
                # Single-device: read from run_id=1
                result_path = os.path.join(results_dir, f"total_streams_result_1_{device_id}.json")

                if not os.path.exists(result_path):
                    logger.debug(f"Result file not yet available: {result_path}")
                    return (0, "unknown", 0)

                with open(result_path, "r") as f:
                    result_data = json.load(f)

                if device_id not in result_data:
                    return (0, "unknown", 0)

                device_data = result_data[device_id]
                metadata = device_data.get("metadata", {})

                return (
                    metadata.get("per_stream_fps", 0),
                    device_data.get("analysis_status", "unknown"),
                    metadata.get("num_streams", 0),
                )

        except (json.JSONDecodeError, IOError) as e:
            # File might still be being written - retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                logger.debug(
                    f"[{device_id}] Failed to read result (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time:.2f}s..."
                )
                time.sleep(wait_time)
                continue
            else:
                logger.warning(f"[{device_id}] Failed to read result after {max_retries} attempts: {e}")
                return (0, "timeout", 0)
        except Exception as e:
            logger.error(f"[{device_id}] Unexpected error reading result: {e}")
            return (0, "timeout", 0)

    return (0, "timeout", 0)


def qualify_device(
    docker_client,
    device_dict: Dict[str, Any],
    device_id: str,
    device_data: Dict[str, Any],
    active_devices: Dict[str, Any],
    target_fps: float,
    pipeline: str,
    pipeline_params: Dict[str, Dict[str, str]],
    docker_image_tag_analyzer: str,
    docker_container_prefix: str,
    data_dir: str,
    container_mnt_dir: str,
    pipeline_timeout: int,
    results_dir: str,
    num_sockets: int = 1,
    consecutive_timeout_threshold: int = 2,
    max_streams_above_baseline: int = 10,
    container_config: Dict[str, Any] = None,
) -> bool:
    """
    Iteratively find the max streams for a single device while others are active.
    Uses intelligent binary search strategy to quickly find optimal stream count.
    Handles multi-socket CPUs by running them concurrently and aggregating results.

    Args:
        consecutive_timeout_threshold: Number of consecutive timeout failures before converging
        max_streams_above_baseline: Maximum streams to explore above baseline (limits search range)
    """
    logger.debug(f"Active devices for qualification: {active_devices}")
    device_result_path = os.path.join(results_dir, f"total_streams_result_0_{device_id}.json")

    is_multisocket = device_id == "CPU" and num_sockets > 1

    # Initialize binary search bounds
    qual_state = device_data.get("qualification_state", {})
    initial_streams = qual_state.get("num_streams", 1)
    min_streams = 1

    # For multi-socket, apply max_streams_above_baseline per socket
    if is_multisocket:
        max_streams = initial_streams + (max_streams_above_baseline * num_sockets)
    else:
        max_streams = initial_streams + max_streams_above_baseline

    current_num_streams = initial_streams

    # Track search state
    last_successful_streams = 0
    last_successful_fps = 0.0
    last_successful_metadata = None  # Store complete metadata for full restoration
    current_fps = 0.0
    current_analysis_status = "unknown"
    consecutive_timeouts = 0
    had_valid_analysis = False  # Track if we ever got valid FPS values (even if below target)

    logger.debug(f"[{device_id}] Current FPS: {current_fps}")
    logger.info(
        f"[{device_id}] Starting intelligent qualification with {current_num_streams} streams "
        f"(search range: {min_streams}-{max_streams}), target FPS: {target_fps}"
    )
    logger.info(
        f"[{device_id}] Timeout threshold: {consecutive_timeout_threshold} consecutive timeouts before converging"
    )

    if is_multisocket:
        logger.debug(
            f"[{device_id}] Conservative limit: max {max_streams_above_baseline} streams above baseline per socket "
            f"({initial_streams} baseline + {max_streams_above_baseline * num_sockets} = {max_streams} max total)"
        )
    else:
        logger.debug(
            f"[{device_id}] Conservative limit: max {max_streams_above_baseline} streams above baseline "
            f"({initial_streams + max_streams_above_baseline} total)"
        )

    logger.info(f"[{device_id}] Active devices: {list(active_devices.keys())}")

    # Maximum iterations to prevent infinite loops
    max_iterations = 20
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        previous_fps = device_data.get("qualification_state", {}).get("per_stream_fps", 0)
        # Update qualification state for current iteration
        device_data["qualification_state"]["num_streams"] = current_num_streams
        combined_analysis = {**active_devices, device_id: device_data}

        logger.info(
            f"[{device_id}] Iteration {iteration}/{max_iterations}: Testing with {current_num_streams} streams "
            f"(range: {min_streams}-{max_streams}), alongside {len(active_devices)} other devices."
        )

        run_concurrent_analysis(
            docker_client=docker_client,
            device_dict=device_dict,
            analysis_tasks=combined_analysis,
            pipeline=pipeline,
            pipeline_params=pipeline_params,
            docker_image_tag_analyzer=docker_image_tag_analyzer,
            docker_container_prefix=docker_container_prefix,
            data_dir=data_dir,
            container_mnt_dir=container_mnt_dir,
            pipeline_timeout=pipeline_timeout,
            target_fps=target_fps,
            num_sockets=num_sockets,
            container_config=container_config,
            base_run_id=1,  # Use run_id=1+ for socket results, keeping run_id=0 for qualification state
        )

        # --- Read essential results with retry logic (lightweight, no heavy metadata) ---
        # This reads only FPS, status, and actual stream count - defers metadata reading until convergence
        current_fps, current_analysis_status, actual_streams_run = _read_essential_result_with_retry(
            device_id=device_id,
            results_dir=results_dir,
            num_sockets=num_sockets,
            max_retries=3,
            retry_delay=0.5,
        )

        # Track the requested stream count for binary search (don't let error values overwrite it)
        requested_num_streams = current_num_streams

        # For single-device scenarios, update current_num_streams if we got valid data
        if not is_multisocket and actual_streams_run >= 0:
            current_num_streams = actual_streams_run
        # For multi-socket, keep current_num_streams as requested (binary search uses total requested)

        logger.debug(
            f"[{device_id}] After run: requested={requested_num_streams}, "
            f"actual_run={actual_streams_run}, current_fps={current_fps:.2f}, status={current_analysis_status}"
        )

        # Update qualification state
        device_data["qualification_state"]["per_stream_fps"] = current_fps
        device_data["qualification_state"]["num_streams"] = current_num_streams
        # For multi-socket, also track the actual streams that ran (after division)
        if is_multisocket and actual_streams_run > 0:
            device_data["qualification_state"]["actual_streams_run"] = actual_streams_run

        # Only "success" status is valid for qualification - timeout means incomplete analysis
        if current_analysis_status != "success":
            # Check if this is a timeout while INCREASING streams above last successful
            # If we're trying higher than last successful and timeout → we found the natural boundary
            if last_successful_streams > 0 and current_num_streams > last_successful_streams:
                # Timeout while increasing streams - natural boundary found, converge to last successful
                logger.warning(
                    f"[{device_id}] Timeout/failure at {current_num_streams} streams while increasing "
                    f"(last successful: {last_successful_streams}, baseline: {initial_streams}). "
                    f"Using last successful: {last_successful_streams} streams."
                )
                # Treat as convergence - use last successful configuration
                device_data["pass"] = True
                # Update qualification_state with last successful values
                device_data["qualification_state"]["per_stream_fps"] = last_successful_fps
                device_data["qualification_state"]["num_streams"] = last_successful_streams
                device_data["qualification_state"]["last_successful_fps"] = last_successful_fps

                # Restore last successful FPS to all active devices
                for other_dev_id in active_devices:
                    other_qual_state = active_devices[other_dev_id].get("qualification_state", {})
                    if other_dev_id != device_id and "last_successful_fps" in other_qual_state:
                        other_qual_state["per_stream_fps"] = other_qual_state["last_successful_fps"]

                # Load complete metadata now that qualification has converged

                result = read_device_result(
                    device_id=device_id,
                    results_dir=results_dir,
                    num_sockets=num_sockets,
                    aggregate_multi_socket=True,
                )
                if "metadata" in result:
                    device_data["metadata"] = result["metadata"]
                    logger.debug(f"[{device_id}] Loaded complete metadata after convergence")

                _save_device_result(device_result_path, device_id, device_data)
                update_device_metrics(
                    active_devices=active_devices,
                    device_id=device_id,
                    results_dir=results_dir,
                    num_sockets=num_sockets,
                    target_fps=target_fps,
                )
                return True

            # Failure at or below last successful - check if it's a timeout or other failure
            if current_analysis_status == "timeout":
                # Timeout at or below last successful - increment counter and binary search lower
                consecutive_timeouts += 1

                logger.warning(
                    f"[{device_id}] Timeout at {requested_num_streams} streams "
                    f"(last successful: {last_successful_streams}, baseline: {initial_streams}), "
                    f"consecutive timeouts: {consecutive_timeouts}/{consecutive_timeout_threshold}"
                )

                # Calculate next stream count via binary search
                max_streams = requested_num_streams
                next_streams = (min_streams + requested_num_streams) // 2

                # Ensure we're making progress in the search
                if next_streams == requested_num_streams:
                    next_streams = requested_num_streams - 1
                if next_streams < 1:
                    next_streams = 1

                # Check if binary search range is exhausted (can't go lower)
                search_exhausted = next_streams < 1 or (min_streams >= max_streams and max_streams == 1)

                # If search exhausted, use last successful if available, else fail
                if search_exhausted:
                    if last_successful_streams > 0:
                        logger.warning(
                            f"[{device_id}] Binary search exhausted, using last successful configuration: "
                            f"{last_successful_streams} streams at {last_successful_fps:.2f} FPS"
                        )
                        device_data["pass"] = True
                        device_data["qualification_state"]["per_stream_fps"] = last_successful_fps
                        device_data["qualification_state"]["num_streams"] = last_successful_streams
                        device_data["qualification_state"]["last_successful_fps"] = last_successful_fps

                        # Restore last successful FPS to all active devices
                        for other_dev_id in active_devices:
                            other_qual_state = active_devices[other_dev_id].get("qualification_state", {})
                            if other_dev_id != device_id and "last_successful_fps" in other_qual_state:
                                other_qual_state["per_stream_fps"] = other_qual_state["last_successful_fps"]

                        # Load complete metadata now that qualification has converged

                        result = read_device_result(
                            device_id=device_id,
                            results_dir=results_dir,
                            num_sockets=num_sockets,
                            aggregate_multi_socket=True,
                        )
                        if "metadata" in result:
                            device_data["metadata"] = result["metadata"]
                            logger.debug(f"[{device_id}] Loaded complete metadata after convergence")

                        _save_device_result(device_result_path, device_id, device_data)
                        update_device_metrics(
                            active_devices=active_devices,
                            device_id=device_id,
                            results_dir=results_dir,
                            num_sockets=num_sockets,
                            target_fps=target_fps,
                        )
                        return True
                    else:
                        # Search exhausted with no successful config - fail device
                        error_reason = (
                            f"Pipeline timeout at {requested_num_streams} streams "
                            f"(baseline: {initial_streams}) after {consecutive_timeouts} consecutive timeouts. "
                            f"Binary search exhausted with no successful configuration."
                        )
                        logger.error(f"[{device_id}] {error_reason}. Device disqualified.")
                        device_data["pass"] = False
                        device_data["qualification_state"]["num_streams"] = -1
                        device_data["error_reason"] = error_reason
                        _save_device_result(device_result_path, device_id, device_data)
                        return False

                # If consecutive timeout threshold reached and we have a last successful config, converge
                if consecutive_timeouts >= consecutive_timeout_threshold and last_successful_streams > 0:
                    logger.warning(
                        f"[{device_id}] Consecutive timeout threshold reached ({consecutive_timeouts}/"
                        f"{consecutive_timeout_threshold}), converging to last successful configuration: "
                        f"{last_successful_streams} streams at {last_successful_fps:.2f} FPS"
                    )
                    device_data["pass"] = True
                    device_data["qualification_state"]["per_stream_fps"] = last_successful_fps
                    device_data["qualification_state"]["num_streams"] = last_successful_streams
                    device_data["qualification_state"]["last_successful_fps"] = last_successful_fps

                    # Restore last successful FPS to all active devices
                    for other_dev_id in active_devices:
                        other_qual_state = active_devices[other_dev_id].get("qualification_state", {})
                        if other_dev_id != device_id and "last_successful_fps" in other_qual_state:
                            other_qual_state["per_stream_fps"] = other_qual_state["last_successful_fps"]

                    # Load complete metadata now that qualification has converged

                    result = read_device_result(
                        device_id=device_id,
                        results_dir=results_dir,
                        num_sockets=num_sockets,
                        aggregate_multi_socket=True,
                    )
                    if "metadata" in result:
                        device_data["metadata"] = result["metadata"]
                        logger.debug(f"[{device_id}] Loaded complete metadata after convergence")

                    _save_device_result(device_result_path, device_id, device_data)
                    update_device_metrics(
                        active_devices=active_devices,
                        device_id=device_id,
                        results_dir=results_dir,
                        num_sockets=num_sockets,
                        target_fps=target_fps,
                    )
                    return True

                # Continue binary search to lower stream count
                current_num_streams = next_streams
                logger.info(
                    f"[{device_id}] Continuing binary search to lower streams: {current_num_streams} "
                    f"(range: {min_streams}-{max_streams}), "
                    f"consecutive timeouts: {consecutive_timeouts}/{consecutive_timeout_threshold}"
                )
                # Skip to next iteration to test with lower streams
                time.sleep(2)
                continue
            else:
                # Non-timeout failure (parse_failed, main_failed, result_failed, error, etc.)
                error_reason = (
                    f"Analysis failed at {current_num_streams} streams (baseline: {initial_streams}) "
                    f"with {current_fps:.2f} FPS due to '{current_analysis_status}'"
                )
                logger.error(f"[{device_id}] {error_reason}. Device disqualified.")
                device_data["pass"] = False
                device_data["qualification_state"]["num_streams"] = -1
                device_data["error_reason"] = error_reason
                _save_device_result(device_result_path, device_id, device_data)
                return False

        # Mark that we got valid analysis results (FPS extracted successfully)
        if current_fps > 0:
            had_valid_analysis = True

        # Check all devices for pass/fail
        all_devices_pass = True
        failed_devices = []

        # Update all device metrics from latest results using centralized reader
        for dev_id in combined_analysis.keys():
            try:
                dev_result = read_device_result(
                    device_id=dev_id, results_dir=results_dir, num_sockets=num_sockets, aggregate_multi_socket=True
                )

                if "error" not in dev_result:
                    dev_fps = dev_result["per_stream_fps"]
                    dev_status = dev_result.get("analysis_status", "unknown")
                    # Update the device metrics with latest data
                    combined_analysis[dev_id]["qualification_state"]["per_stream_fps"] = dev_fps

                    # Check if this device meets the target AND has successful status
                    # Timeout or error status means insufficient resources even if FPS looks good
                    if dev_fps < target_fps:
                        all_devices_pass = False
                        failed_devices.append(dev_id)
                        logger.debug(f"[{dev_id}] Below target FPS: {dev_fps:.2f} < {target_fps}")
                    elif dev_status != "success":
                        all_devices_pass = False
                        failed_devices.append(dev_id)
                        logger.warning(
                            f"[{dev_id}] Analysis failed with status '{dev_status}' - "
                            f"insufficient resources for concurrent workload"
                        )
                else:
                    dev_fps = 0
                    logger.warning(f"Failed to read result for {dev_id}: {dev_result.get('error', 'unknown')}")
                    all_devices_pass = False
                    failed_devices.append(dev_id)

            except Exception as e:
                logger.error(f"Error reading result for device {dev_id}: {e}")
                all_devices_pass = False
                failed_devices.append(dev_id)

        logger.info(
            f"[{device_id}] Aggregated/Current FPS: {current_fps:.2f} "
            f"(Previous: {previous_fps:.2f}, Target: {target_fps})"
        )

        # Log concurrent device status if any failed
        if not all_devices_pass:
            logger.info(
                f"[{device_id}] Concurrent device(s) failed: {', '.join(failed_devices)}. "
                f"Adjusting search to find configuration where all devices succeed."
            )

        # Simplified binary search strategy - no consecutive thresholds needed
        if all_devices_pass and current_fps >= target_fps:
            # Success - update lower bound and track this configuration
            last_successful_streams = current_num_streams
            last_successful_fps = current_fps
            # Store complete metadata for full restoration (includes per_stream_fps_list)
            last_successful_metadata = device_data.get("metadata", {}).copy() if "metadata" in device_data else None
            device_data["qualification_state"]["last_successful_fps"] = last_successful_fps
            # Save complete analysis metadata to qualification_state for direct use
            if last_successful_metadata:
                device_data["qualification_state"]["metadata"] = last_successful_metadata
            consecutive_timeouts = 0  # Reset timeout counter on success

            # Track last successful FPS for all other active devices
            for other_dev_id, other_dev_data in combined_analysis.items():
                if other_dev_id != device_id and other_dev_id in active_devices:
                    other_fps = other_dev_data.get("qualification_state", {}).get("per_stream_fps", 0)
                    if other_fps >= target_fps:
                        logger.debug(f"Saving last successful FPS for {other_dev_id}: {other_fps:.2f}")
                        active_devices[other_dev_id]["qualification_state"]["last_successful_fps"] = other_fps

            # Update lower bound for binary search
            min_streams = current_num_streams

            # Check if we can continue searching or should stop
            if max_streams - min_streams <= 1:
                # Converged - we found the maximum
                # For multi-socket, show both requested and actual streams for clarity
                if is_multisocket:
                    logger.info(
                        f"[{device_id}] Binary search converged. Qualified with {current_num_streams} "
                        f"total streams ({actual_streams_run} actual: "
                        f"{actual_streams_run // num_sockets}+{actual_streams_run - actual_streams_run // num_sockets} "
                        f"per socket) at {current_fps:.2f} FPS."
                    )
                else:
                    logger.info(
                        f"[{device_id}] Binary search converged. Qualified with {current_num_streams} "
                        f"streams at {current_fps:.2f} FPS."
                    )
                device_data["pass"] = True
                device_data["qualification_state"]["per_stream_fps"] = current_fps
                # For multi-socket, save the actual streams that ran
                if is_multisocket:
                    device_data["qualification_state"]["actual_streams_run"] = actual_streams_run
                # Save complete metadata to qualification_state
                if last_successful_metadata:
                    device_data["qualification_state"]["metadata"] = last_successful_metadata

                # Load complete metadata now that qualification has converged

                result = read_device_result(
                    device_id=device_id,
                    results_dir=results_dir,
                    num_sockets=num_sockets,
                    aggregate_multi_socket=True,
                )
                if "metadata" in result:
                    device_data["metadata"] = result["metadata"]
                    logger.debug(f"[{device_id}] Loaded complete metadata after convergence")

                # Save the current device's successful result
                _save_device_result(device_result_path, device_id, device_data)

                # Update metrics for all active devices
                update_device_metrics(
                    active_devices=active_devices,
                    device_id=device_id,
                    results_dir=results_dir,
                    num_sockets=num_sockets,
                    target_fps=target_fps,
                )

                return True

            # Success - try higher stream count using binary search
            # Determine increment strategy based on position relative to initial baseline
            logger.debug(
                f"[{device_id}] DEBUG: current_num_streams={current_num_streams}, "
                f"initial_streams={initial_streams}, is_multisocket={is_multisocket}, num_sockets={num_sockets}"
            )

            if current_num_streams >= initial_streams:
                # At or above baseline: increment by 1 to avoid overcommitment/timeout
                # But check if we've reached the maximum exploration limit
                max_allowed_streams = initial_streams + max_streams_above_baseline
                if current_num_streams >= max_allowed_streams:
                    # Reached exploration limit - converge here
                    logger.info(
                        f"[{device_id}] Reached exploration limit ({max_allowed_streams} streams, "
                        f"{max_streams_above_baseline} above baseline). Converging."
                    )
                    device_data["pass"] = True
                    device_data["qualification_state"]["per_stream_fps"] = current_fps
                    _save_device_result(device_result_path, device_id, device_data)
                    update_device_metrics(
                        active_devices=active_devices,
                        device_id=device_id,
                        results_dir=results_dir,
                        num_sockets=num_sockets,
                        target_fps=target_fps,
                    )
                    return True

                # For multi-socket CPU, increment by num_sockets to avoid redundant analysis
                # (e.g., 12→13 both result in 6 per socket with 2 sockets)
                increment = num_sockets if is_multisocket else 1
                next_streams = current_num_streams + increment
                logger.debug(
                    f"[{device_id}] At/above baseline ({initial_streams}), calculated increment={increment}, "
                    f"next_streams={next_streams} (limit: {max_allowed_streams})"
                    f"{' [multi-socket]' if is_multisocket else ''}"
                )
            else:
                # Below baseline: use binary search within safe range
                next_streams = (current_num_streams + max_streams) // 2
                if next_streams == current_num_streams:
                    next_streams = current_num_streams + 1
                # Cap at initial baseline to avoid jumping beyond safe estimate
                if next_streams > initial_streams:
                    next_streams = initial_streams
                logger.debug(
                    f"[{device_id}] Below baseline ({initial_streams}), binary search: "
                    f"({current_num_streams} + {max_streams}) // 2 = {next_streams}"
                )

            current_num_streams = next_streams
            logger.info(
                f"[{device_id}] Target met at {current_fps:.2f} FPS, trying higher: {current_num_streams} streams "
                f"(range: {min_streams}-{max_streams})."
            )
        else:
            # Failed to meet target (non-timeout failure - FPS > 0 but below target)
            # Note: Timeout failures are handled earlier in the code when analysis_status != "success"
            consecutive_timeouts = 0  # Reset timeout counter on FPS-based failure (pipeline ran but didn't meet target)

            # Update upper bound for binary search
            max_streams = current_num_streams

            # Validate range - if min >= max, we've converged
            if min_streams >= max_streams:
                # Range exhausted - use last successful configuration if available
                if last_successful_streams is not None and last_successful_streams > 0:
                    logger.info(
                        f"[{device_id}] Binary search exhausted. Using last successful: "
                        f"{last_successful_streams} streams with {last_successful_fps:.2f} FPS."
                    )
                    device_data["pass"] = True
                    device_data["qualification_state"]["per_stream_fps"] = last_successful_fps
                    device_data["qualification_state"]["num_streams"] = last_successful_streams
                    device_data["qualification_state"]["last_successful_fps"] = last_successful_fps

                    # Restore complete metadata from last successful iteration
                    if last_successful_metadata is not None and "metadata" in device_data:
                        device_data["metadata"] = last_successful_metadata.copy()
                        logger.debug(
                            f"[{device_id}] Restored complete metadata from last successful iteration "
                            f"(including per_stream_fps_list)"
                        )

                    # Restore last successful FPS to all active devices
                    for other_dev_id in active_devices:
                        other_qual_state = active_devices[other_dev_id].get("qualification_state", {})
                        if other_dev_id != device_id and "last_successful_fps" in other_qual_state:
                            other_qual_state["per_stream_fps"] = other_qual_state["last_successful_fps"]

                    # Load complete metadata now that qualification has converged

                    result = read_device_result(
                        device_id=device_id,
                        results_dir=results_dir,
                        num_sockets=num_sockets,
                        aggregate_multi_socket=True,
                    )
                    if "metadata" in result and not last_successful_metadata:
                        # Only override if we don't have saved metadata
                        device_data["metadata"] = result["metadata"]
                        logger.debug(f"[{device_id}] Loaded complete metadata after convergence")

                    _save_device_result(device_result_path, device_id, device_data)
                    update_device_metrics(
                        active_devices=active_devices,
                        device_id=device_id,
                        results_dir=results_dir,
                        num_sockets=num_sockets,
                        target_fps=target_fps,
                    )
                    return True

                # No successful configuration found - build detailed error message
                concurrent_info = ""
                if active_devices:
                    concurrent_device_names = ", ".join(active_devices.keys())
                    concurrent_info = f" Concurrent devices running: {concurrent_device_names}."

                error_reason = (
                    f"Cannot meet target FPS of {target_fps:.2f} under concurrent workload. "
                    f"Last tested: {current_fps:.2f} FPS at {current_num_streams} streams. "
                    f"Binary search exhausted (min={min_streams}, max={max_streams}).{concurrent_info}"
                )
                logger.error(f"[{device_id}] {error_reason}")
                device_data["pass"] = False
                device_data["qualification_state"]["per_stream_fps"] = current_fps
                device_data["qualification_state"]["num_streams"] = 0
                device_data["error_reason"] = error_reason
                _save_device_result(device_result_path, device_id, device_data)
                update_device_metrics(
                    active_devices=active_devices,
                    device_id=device_id,
                    results_dir=results_dir,
                    num_sockets=num_sockets,
                    target_fps=target_fps,
                )
                return False

            # Determine if failure is due to concurrent device (not current device FPS)
            concurrent_device_failed = not all_devices_pass and current_fps >= target_fps

            # Calculate next stream count to try (binary search lower half)
            next_streams = (min_streams + current_num_streams) // 2
            if next_streams == current_num_streams:
                next_streams = current_num_streams - 1

            # Make sure we don't go below minimum
            if next_streams < 1:
                next_streams = 1

            # Check if we should converge AFTER calculating next stream
            # Only converge if next_streams equals current (nowhere left to search)
            if next_streams == current_num_streams:
                # Converged - use the last successful configuration
                if last_successful_streams is not None and last_successful_streams > 0:
                    logger.info(
                        f"[{device_id}] Binary search converged. Using last successful: "
                        f"{last_successful_streams} streams with {last_successful_fps:.2f} FPS."
                    )
                    device_data["pass"] = True
                    device_data["qualification_state"]["per_stream_fps"] = last_successful_fps
                    device_data["qualification_state"]["num_streams"] = last_successful_streams
                    device_data["qualification_state"]["last_successful_fps"] = last_successful_fps

                    # Restore complete metadata from last successful iteration
                    if last_successful_metadata is not None and "metadata" in device_data:
                        device_data["metadata"] = last_successful_metadata.copy()
                        logger.debug(
                            f"[{device_id}] Restored complete metadata from last successful iteration "
                            f"(including per_stream_fps_list)"
                        )

                    # Restore last successful FPS to all active devices
                    for other_dev_id in active_devices:
                        other_qual_state = active_devices[other_dev_id].get("qualification_state", {})
                        if other_dev_id != device_id and "last_successful_fps" in other_qual_state:
                            other_qual_state["per_stream_fps"] = other_qual_state["last_successful_fps"]

                    # Load complete metadata now that qualification has converged

                    result = read_device_result(
                        device_id=device_id,
                        results_dir=results_dir,
                        num_sockets=num_sockets,
                        aggregate_multi_socket=True,
                    )
                    if "metadata" in result and not last_successful_metadata:
                        # Only override if we don't have saved metadata
                        device_data["metadata"] = result["metadata"]
                        logger.debug(f"[{device_id}] Loaded complete metadata after convergence")

                    # Save the device result
                    _save_device_result(device_result_path, device_id, device_data)

                    # Update metrics for all active devices
                    update_device_metrics(
                        active_devices=active_devices,
                        device_id=device_id,
                        results_dir=results_dir,
                        num_sockets=num_sockets,
                        target_fps=target_fps,
                    )

                    return True
                else:
                    # No successful configuration found
                    error_reason = (
                        f"Cannot meet target FPS of {target_fps:.2f}. "
                        f"Current FPS: {current_fps:.2f} at {current_num_streams} streams "
                        f"after {iteration} iterations"
                    )
                    logger.warning(f"[{device_id}] {error_reason}")
                    for dev_id, dev_data in combined_analysis.items():
                        logger.warning(f"  - {dev_id}: {dev_data.get('per_stream_fps', 0):.2f} FPS")

                    device_data["pass"] = False
                    device_data["qualification_state"]["per_stream_fps"] = current_fps
                    device_data["qualification_state"]["num_streams"] = 0
                    device_data["error_reason"] = error_reason
                    logger.error(
                        f"[{device_id}] Disqualified after {iteration} iterations "
                        f"(max: {max_iterations}). {error_reason}"
                    )

                    # Save the failed device result
                    _save_device_result(device_result_path, device_id, device_data)

                    # Update metrics for all active devices
                    update_device_metrics(
                        active_devices=active_devices,
                        device_id=device_id,
                        results_dir=results_dir,
                        num_sockets=num_sockets,
                        target_fps=target_fps,
                    )

                    return False

            # Continue searching with next_streams
            current_num_streams = next_streams

            # Log reason for trying lower stream count
            if concurrent_device_failed:
                logger.info(
                    f"[{device_id}] Reducing streams due to concurrent device failure(s): "
                    f"{current_num_streams} streams (range: {min_streams}-{max_streams})."
                )
            else:
                logger.info(
                    f"[{device_id}] Below target at {current_fps:.2f} FPS, trying lower: "
                    f"{current_num_streams} streams (range: {min_streams}-{max_streams})."
                )

        time.sleep(2)

    # If we've reached max iterations without converging
    logger.warning(f"[{device_id}] Reached maximum iterations ({max_iterations}) without full convergence.")

    # If we have a last successful configuration, use it
    if last_successful_streams is not None and last_successful_streams > 0:
        logger.info(
            f"[{device_id}] Using last successful configuration: "
            f"{last_successful_streams} streams at {last_successful_fps:.2f} FPS"
        )
        device_data["qualification_state"]["num_streams"] = last_successful_streams
        device_data["qualification_state"]["per_stream_fps"] = last_successful_fps
        device_data["qualification_state"]["last_successful_fps"] = last_successful_fps
        device_data["pass"] = True

        # Restore complete metadata from last successful iteration (includes per_stream_fps_list)
        if last_successful_metadata is not None and "metadata" in device_data:
            device_data["metadata"] = last_successful_metadata.copy()
            logger.debug(
                f"[{device_id}] Restored complete metadata from last successful iteration: "
                f"{last_successful_streams} streams at {last_successful_fps:.2f} FPS "
                f"(including per_stream_fps_list)"
            )

        # Revert all active devices to their last successful FPS values
        logger.info("Reverting other active devices to their last successful FPS values")
        for other_dev_id, other_dev_data in active_devices.items():
            other_qual_state = other_dev_data.get("qualification_state", {})
            if other_dev_id != device_id and "last_successful_fps" in other_qual_state:
                other_last_fps = other_qual_state["last_successful_fps"]
                logger.info(f"  - Reverting {other_dev_id} to last successful FPS: {other_last_fps:.2f}")
                other_qual_state["per_stream_fps"] = other_last_fps

    # If there's absolutely no successful configuration, device failed
    else:
        # Determine if we should use 0 (got FPS but didn't meet target) or -1 (pipeline errors)
        # Use 0 if we ever got valid analysis results, -1 if we only had errors/timeouts
        num_streams_value = 0 if had_valid_analysis else -1

        error_reason = (
            f"No successful configuration found after {max_iterations} iterations. "
            f"Unable to meet target FPS of {target_fps:.2f}"
        )
        logger.error(f"[{device_id}] {error_reason}. Qualification failed.")
        device_data["qualification_state"]["num_streams"] = num_streams_value
        device_data["pass"] = False
        device_data["error_reason"] = error_reason

    # Save final result
    _save_device_result(device_result_path, device_id, device_data)
    return device_data["pass"]


def _save_device_result(device_result_path: str, device_id: str, device_data: Dict[str, Any]) -> None:
    """Save device result to file with proper permissions."""
    try:
        with open(device_result_path, "w") as wfile:
            data = {device_id: device_data}
            json.dump(data, wfile, indent=4)
        os.chmod(str(device_result_path), 0o770)
    except Exception as e:
        logger.error(f"Failed to write to {device_result_path}: {e}")
        raise
