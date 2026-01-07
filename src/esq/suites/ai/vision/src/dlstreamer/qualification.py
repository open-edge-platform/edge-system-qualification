# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer device qualification orchestration."""

import json
import logging
import os
import time
from typing import Any, Dict

from .concurrent import run_concurrent_analysis
from .utils import update_device_metrics

logger = logging.getLogger(__name__)


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
    consecutive_success_threshold: int = 1,
    consecutive_failure_threshold: int = 1,
    consecutive_timeout_threshold: int = 2,
    max_streams_above_baseline: int = 10,
    container_config: Dict[str, Any] = None,
) -> bool:
    """
    Iteratively find the max streams for a single device while others are active.
    Uses intelligent binary search strategy to quickly find optimal stream count.
    Handles multi-socket CPUs by running them concurrently and aggregating results.

    Args:
        consecutive_success_threshold: Number of consecutive successes before trying higher streams
        consecutive_failure_threshold: Number of consecutive failures (non-timeout) before trying lower streams
        consecutive_timeout_threshold: Number of consecutive timeout failures before trying lower streams
                                       (only below baseline)
        max_streams_above_baseline: Maximum streams to explore above baseline (limits +1 increment)
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
    consecutive_successes = 0
    consecutive_failures = 0
    consecutive_timeouts = 0
    had_valid_analysis = False  # Track if we ever got valid FPS values (even if below target)

    logger.debug(f"[{device_id}] Current FPS: {current_fps}")
    logger.info(
        f"[{device_id}] Starting intelligent qualification with {current_num_streams} streams "
        f"(search range: {min_streams}-{max_streams}), target FPS: {target_fps}"
    )
    logger.info(
        f"[{device_id}] Thresholds: {consecutive_success_threshold} consecutive successes to go higher, "
        f"{consecutive_failure_threshold} consecutive failures to go lower, "
        f"{consecutive_timeout_threshold} consecutive timeouts (below baseline) to go lower"
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
        )

        # --- Read and aggregate results ---
        current_fps = 0
        if is_multisocket:
            all_socket_fps = []
            total_achieved_streams = 0
            for i in range(num_sockets):
                socket_path = os.path.join(results_dir, f"total_streams_result_{i}_{device_id}.json")
                with open(socket_path, "r") as file:
                    socket_result = json.load(file)
                if not socket_result or device_id not in socket_result:
                    if "CPU" not in device_id:
                        continue
                    logger.warning(f"[{device_id}] No valid results for socket {i} in {socket_path}. Assuming 0 FPS.")
                    all_socket_fps.append(0)
                    continue
                result_data = socket_result[device_id]
                # Extract from metadata
                metadata = result_data.get("metadata", {})
                all_socket_fps.append(metadata.get("per_stream_fps", 0))
                total_achieved_streams += metadata.get("num_streams", 0)

            current_fps = min(all_socket_fps) if all_socket_fps else 0
            # Update qualification state with aggregated results
            device_data["qualification_state"]["per_stream_fps"] = current_fps
            device_data["qualification_state"]["num_streams"] = total_achieved_streams
        else:
            with open(device_result_path, "r") as file:
                latest_result = json.load(file)
            if device_id not in latest_result:
                logger.error(f"[{device_id}] No results found in {device_result_path}. Qualification failed.")
                return False

            current_analysis_status = latest_result[device_id].get("analysis_status", "unknown")
            # Only "success" status is valid for qualification - timeout means incomplete analysis
            if current_analysis_status != "success":
                # Check if this is a timeout above baseline (overcommitment) or below baseline
                if current_num_streams > initial_streams and last_successful_streams > 0:
                    # Timeout/failure above baseline with previous success - likely overcommitment
                    logger.warning(
                        f"[{device_id}] Timeout/failure at {current_num_streams} streams (above baseline "
                        f"{initial_streams}). Using last successful: {last_successful_streams} streams."
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
                    # Failure at or below baseline - check if it's a timeout or other failure
                    result_data = latest_result.get(device_id, {})
                    metadata = result_data.get("metadata", {})
                    current_fps = metadata.get("per_stream_fps", 0.0)

                    # Distinguish between timeout and other failures
                    if current_analysis_status == "timeout":
                        # Timeout at or below baseline - immediately lower streams via binary search
                        consecutive_timeouts += 1
                        consecutive_failures = 0  # Reset non-timeout failure counter

                        logger.warning(
                            f"[{device_id}] Timeout at {current_num_streams} streams (baseline: {initial_streams}), "
                            f"consecutive timeouts: {consecutive_timeouts}/{consecutive_timeout_threshold}"
                        )

                        # Check if we've exceeded the consecutive timeout threshold
                        if consecutive_timeouts >= consecutive_timeout_threshold:
                            # Threshold reached - check if we have a previous successful configuration
                            if last_successful_streams > 0:
                                logger.warning(
                                    f"[{device_id}] Consecutive timeout threshold reached, but using last "
                                    f"successful configuration: {last_successful_streams} streams at "
                                    f"{last_successful_fps:.2f} FPS"
                                )
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
                                # No successful configuration found
                                error_reason = (
                                    f"Pipeline timeout at {current_num_streams} streams "
                                    f"(baseline: {initial_streams}) after {consecutive_timeouts} "
                                    f"consecutive timeouts. No successful configuration found."
                                )
                                logger.error(f"[{device_id}] {error_reason}. Device disqualified.")
                                device_data["pass"] = False
                                device_data["qualification_state"]["num_streams"] = -1
                                device_data["error_reason"] = error_reason
                                _save_device_result(device_result_path, device_id, device_data)
                                return False

                        # Haven't reached threshold yet - continue with binary search to lower streams
                        max_streams = current_num_streams
                        next_streams = (min_streams + current_num_streams) // 2
                        if next_streams == current_num_streams:
                            next_streams = current_num_streams - 1
                        if next_streams < 1:
                            next_streams = 1

                        # Check if we've converged or exhausted search space
                        if next_streams == current_num_streams or (min_streams >= max_streams):
                            # Can't go lower - fail device
                            error_reason = (
                                f"Pipeline timeout at {current_num_streams} streams "
                                f"(baseline: {initial_streams}). Cannot lower streams further."
                            )
                            logger.error(f"[{device_id}] {error_reason}. Device disqualified.")
                            device_data["pass"] = False
                            device_data["qualification_state"]["num_streams"] = -1
                            device_data["error_reason"] = error_reason
                            _save_device_result(device_result_path, device_id, device_data)
                            return False

                        # Continue with lower stream count via binary search
                        current_num_streams = next_streams
                        logger.info(
                            f"[{device_id}] Timeout detected, lowering streams via binary search: "
                            f"{current_num_streams} streams (range: {min_streams}-{max_streams}), "
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

            # Extract current_fps from metadata
            result_data = latest_result[device_id]
            metadata = result_data.get("metadata", {})
            current_fps = metadata.get("per_stream_fps", 0)
            device_data.update(latest_result[device_id])

            # Mark that we got valid analysis results (FPS extracted successfully)
            had_valid_analysis = True

        # Check all devices for pass/fail
        all_devices_pass = True
        failed_devices = []

        # Update all device metrics from latest results
        for dev_id in combined_analysis.keys():
            try:
                if is_multisocket and dev_id == "CPU":
                    # Aggregate for multi-socket CPU
                    all_socket_fps = []
                    for i in range(num_sockets):
                        socket_path = os.path.join(results_dir, f"total_streams_result_{i}_{dev_id}.json")
                        if not os.path.exists(socket_path):
                            logger.warning(f"Socket result file not found: {socket_path}")
                            continue

                        with open(socket_path, "r") as file:
                            socket_result = json.load(file)
                        if not socket_result or dev_id not in socket_result:
                            all_socket_fps.append(0)
                            continue
                        result_data = socket_result[dev_id]
                        metadata = result_data.get("metadata", {})
                        all_socket_fps.append(metadata.get("per_stream_fps", 0))

                    # Update CPU device metrics with aggregate data
                    if all_socket_fps:
                        dev_fps = min(all_socket_fps)
                        combined_analysis[dev_id]["qualification_state"]["per_stream_fps"] = dev_fps
                    else:
                        dev_fps = 0
                else:
                    result_path = os.path.join(results_dir, f"total_streams_result_0_{dev_id}.json")
                    if not os.path.exists(result_path):
                        logger.warning(f"Result file not found: {result_path}")
                        dev_fps = 0
                    else:
                        with open(result_path, "r") as file:
                            latest_result = json.load(file)
                        if dev_id in latest_result:
                            result_data = latest_result[dev_id]
                            metadata = result_data.get("metadata", {})
                            dev_fps = metadata.get("per_stream_fps", 0)
                            # Update the device metrics with latest data
                            combined_analysis[dev_id]["qualification_state"]["per_stream_fps"] = dev_fps
                        else:
                            dev_fps = 0

                # Check if this device meets the target
                if dev_fps < target_fps:
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

        # Binary search strategy with consecutive thresholds
        if all_devices_pass and current_fps >= target_fps:
            # Success - update lower bound
            last_successful_streams = current_num_streams
            last_successful_fps = current_fps
            # Store complete metadata for full restoration (includes per_stream_fps_list)
            last_successful_metadata = device_data.get("metadata", {}).copy() if "metadata" in device_data else None
            device_data["qualification_state"]["last_successful_fps"] = last_successful_fps
            consecutive_successes += 1
            consecutive_failures = 0  # Reset failure counter on success
            consecutive_timeouts = 0  # Reset timeout counter on success

            # Track last successful FPS for all other active devices
            for other_dev_id, other_dev_data in combined_analysis.items():
                if other_dev_id != device_id and other_dev_id in active_devices:
                    other_fps = other_dev_data.get("qualification_state", {}).get("per_stream_fps", 0)
                    if other_fps >= target_fps:
                        logger.debug(f"Saving last successful FPS for {other_dev_id}: {other_fps:.2f}")
                        active_devices[other_dev_id]["qualification_state"]["last_successful_fps"] = other_fps

            # Update lower bound
            min_streams = current_num_streams

            # Check if we can continue searching or should stop
            if max_streams - min_streams <= 1:
                # Converged - we found the maximum
                logger.info(
                    f"[{device_id}] Binary search converged. Qualified with {current_num_streams} "
                    f"streams at {current_fps:.2f} FPS."
                )
                device_data["pass"] = True
                device_data["qualification_state"]["per_stream_fps"] = current_fps

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

            # Check if we should try higher based on consecutive success threshold
            if consecutive_successes >= consecutive_success_threshold:
                # Determine increment strategy based on position relative to initial baseline
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
                        f"[{device_id}] At/above baseline ({initial_streams}), using +{increment} increment "
                        f"(limit: {max_allowed_streams}){' for multi-socket' if is_multisocket else ''}"
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
                        f"[{device_id}] Below baseline ({initial_streams}), using binary search to {next_streams}"
                    )

                current_num_streams = next_streams
                logger.info(
                    f"[{device_id}] Target met ({consecutive_successes}/{consecutive_success_threshold} "
                    f"consecutive successes), trying higher: {current_num_streams} streams "
                    f"(range: {min_streams}-{max_streams})."
                )
            else:
                # Not enough consecutive successes yet, stay at current level
                logger.info(
                    f"[{device_id}] Target met ({consecutive_successes}/{consecutive_success_threshold} "
                    f"consecutive successes), retesting with {current_num_streams} streams to confirm stability."
                )
        else:
            # Failed to meet target (non-timeout failure - FPS > 0 but below target)
            # Note: Timeout failures are handled earlier in the code when analysis_status != "success"
            consecutive_successes = 0  # Reset success counter on failure
            consecutive_timeouts = 0  # Reset timeout counter on FPS-based failure (pipeline ran but didn't meet target)
            consecutive_failures += 1

            # Check if we should try lower based on consecutive failure threshold
            if consecutive_failures >= consecutive_failure_threshold:
                # Update upper bound
                max_streams = current_num_streams

                # Validate range - if min >= max, we've converged
                if min_streams >= max_streams:
                    # Range exhausted - verify last successful configuration before qualifying
                    if last_successful_streams is not None and last_successful_streams > 0:
                        logger.info(
                            f"[{device_id}] Binary search range exhausted (min={min_streams}, max={max_streams}). "
                            f"Verifying last successful: {last_successful_streams} streams "
                            f"with {last_successful_fps:.2f} FPS."
                        )

                        # Run confirmation with last successful configuration
                        device_data["qualification_state"]["num_streams"] = last_successful_streams

                        # Execute confirmation run
                        run_concurrent_analysis(
                            docker_client=docker_client,
                            device_dict=device_dict,
                            analysis_tasks={device_id: device_data, **active_devices},
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
                        )

                        # Parse confirmation result
                        from .analysis import parse_device_result_file

                        confirmation_result = parse_device_result_file(device_id, results_dir)
                        confirmation_fps = confirmation_result.get("metadata", {}).get("per_stream_fps", 0)

                        # Check if confirmation passed
                        if confirmation_fps >= target_fps:
                            logger.info(
                                f"[{device_id}] Confirmation successful: {last_successful_streams} streams at "
                                f"{confirmation_fps:.2f} FPS (target: {target_fps:.2f})"
                            )
                            device_data["pass"] = True
                            device_data["qualification_state"]["per_stream_fps"] = confirmation_fps
                            device_data["qualification_state"]["num_streams"] = last_successful_streams
                            device_data["qualification_state"]["last_successful_fps"] = confirmation_fps

                            # Update metadata with confirmation result
                            if "metadata" in confirmation_result:
                                device_data["metadata"] = confirmation_result["metadata"]
                                logger.debug(
                                    f"[{device_id}] Updated metadata from confirmation run "
                                    f"(including per_stream_fps_list)"
                                )

                            # Restore last successful FPS to all active devices
                            for other_dev_id in active_devices:
                                other_qual_state = active_devices[other_dev_id].get("qualification_state", {})
                                if other_dev_id != device_id and "last_successful_fps" in other_qual_state:
                                    other_qual_state["per_stream_fps"] = other_qual_state["last_successful_fps"]

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
                            # Confirmation failed - restart binary search with last successful as upper limit
                            # Use the last_successful_streams (before confirmation) as new upper bound
                            # If no successful streams found, use baseline - 1
                            new_upper_limit = (
                                last_successful_streams if (last_successful_streams > 0) else (initial_streams - 1)
                            )

                            logger.warning(
                                f"[{device_id}] Confirmation failed: {confirmation_fps:.2f} < {target_fps:.2f}. "
                                f"Restarting binary search with last_successful={last_successful_streams} "
                                f"as upper limit."
                            )

                            # Reset binary search parameters
                            min_streams = 1
                            max_streams = new_upper_limit  # Use last successful streams as new upper bound
                            current_num_streams = (min_streams + max_streams) // 2

                            # Reset tracking variables
                            last_successful_streams = 0
                            last_successful_fps = 0
                            last_successful_metadata = None
                            consecutive_successes = 0
                            consecutive_failures = 0
                            consecutive_timeouts = 0

                            # Update device state
                            device_data["qualification_state"]["num_streams"] = current_num_streams

                            logger.info(
                                f"[{device_id}] Restarting binary search: "
                                f"range=[{min_streams}, {max_streams}], starting at {current_num_streams} streams"
                            )

                            # Continue the loop to restart binary search
                            continue
                    # No successful configuration found (or confirmation failed)
                    # Build detailed error message
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

                # Calculate next stream count to try
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
                consecutive_failures = 0  # Reset after taking action
                logger.info(
                    f"[{device_id}] Below target at {current_fps:.2f} FPS "
                    f"({consecutive_failure_threshold} consecutive failures), trying lower: {current_num_streams} "
                    f"streams (range: {min_streams}-{max_streams})."
                )
            else:
                # Not enough consecutive failures yet, stay at current level
                logger.info(
                    f"[{device_id}] Below target at {current_fps:.2f} FPS ({consecutive_failures}/"
                    f"{consecutive_failure_threshold} consecutive failures), retesting with {current_num_streams} "
                    f"streams to confirm."
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
