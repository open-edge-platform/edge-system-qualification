# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer device qualification and concurrent execution functions."""

import grp
import json
import logging
import os
import time
from typing import Any, Dict, List

import docker
import numa

from .container import run_dlstreamer_analyzer_container
from .pipeline import build_multi_pipeline_with_devices, get_sync_config, resolve_pipeline_placeholders
from .preparation import get_device_specific_docker_image
from .utils import update_device_metrics

logger = logging.getLogger(__name__)


def get_cpu_socket_numa_info(num_sockets: int = 1) -> List[Dict[str, Any]]:
    """Get CPU socket and NUMA node information for multi-socket systems."""
    num_of_numa_nodes = list(numa.info.numa_hardware_info()["node_cpu_info"].keys())
    bucket_size = int(len(num_of_numa_nodes) / num_sockets)

    bucket = []
    for i in range(0, len(num_of_numa_nodes), bucket_size):
        bucket.append(num_of_numa_nodes[i : i + bucket_size])

    cpu_socket = []
    for i, nodes in enumerate(bucket):
        cpu_ids = []
        for node in nodes:
            cpu_ids += numa.info.node_to_cpus(node)
        cpu_ids.sort()

        # Remove 2 CPUs, so won't overload the system
        cpu_ids.pop()  # remove 1 CPU
        cpu_ids.pop()  # remove 1 CPU

        cpu_socket.append({"nodes": nodes, "cpu_ids": cpu_ids})

    return cpu_socket


def wait_for_containers(containers: List[docker.models.containers.Container]) -> None:
    """Wait for a list of containers to complete."""
    LOG_FUTURES = []  # This should be passed from main context if needed

    for container in containers:
        try:
            container.wait()
            # remove the logging thread for this container
            for future in LOG_FUTURES:
                if future.done() and future.result() == container:
                    LOG_FUTURES.remove(future)
                    break
        except docker.errors.NotFound:
            pass
        except Exception as e:
            logger.error(f"An error occurred while waiting for container {container.name}: {e}")


def run_benchmark_container(
    docker_client,
    run_id: int,
    device_id: str,
    device_dict: Dict[str, Any],
    pipeline: str,
    pipeline_params: Dict[str, Dict[str, str]],
    docker_image_tag_analyzer: str,
    docker_container_prefix: str,
    data_dir: str,
    container_mnt_dir: str,
    pipeline_timeout: int,
    target_fps: float,
    combined_analysis: Dict[str, Any] = None,
    cpuset_cpus: str = "",
    cpuset_mems: str = "",
    num_streams: int = None,
    visualize_stream: bool = False,
    container_config: Dict[str, Any] = None,
) -> docker.models.containers.Container:
    """
    Helper to run a Docker container for benchmarking.
    Uses modular run_dlstreamer_analyzer_container with server mode for streaming analysis.
    """
    user_gid = os.getuid()
    render_gid = grp.getgrnam("render").gr_gid

    # Select device-specific Docker image
    if container_config:
        docker_image_tag_analyzer = get_device_specific_docker_image(
            device_id, container_config, docker_image_tag_analyzer, device_dict
        )
        logger.debug(f"Selected Docker image for device {device_id}: {docker_image_tag_analyzer}")

    # Set up container name
    container_name = f"{docker_container_prefix}-analyzer-total-{run_id}-{device_id}"
    if num_streams is not None:
        container_name += f"-{num_streams}-streams"

    logger.debug(f"Using pipeline: {pipeline}")

    # Use modular pipeline utilities
    resolved_pipeline = resolve_pipeline_placeholders(pipeline, pipeline_params, device_id, device_dict)
    if num_streams is not None:
        sync_value = get_sync_config(pipeline_params, device_id, device_dict)
        multi_pipeline, result_pipeline = build_multi_pipeline_with_devices(
            pipeline=resolved_pipeline,
            device_id=device_id,
            num_streams=num_streams,
            visualize_stream=visualize_stream,
            pipeline_sync=sync_value,
        )
    else:
        multi_pipeline = ""
        result_pipeline = ""

    # Generate the command
    device_data = combined_analysis.get(device_id, {}) if combined_analysis else {}

    command = [
        "total",
        "--run-id",
        str(run_id),
        "--target-device",
        device_id,
        "--target-fps",
        str(target_fps),
        "--multi-pipeline",
        multi_pipeline,
        "--result-pipeline",
        result_pipeline,
        "--pipeline-timeout",
        str(pipeline_timeout),
        "--combined-analysis",
        json.dumps(device_data.get("combined_analysis", {})),
    ]

    environment = {
        "XDG_RUNTIME_DIR": "/tmp",
        "DISPLAY": os.environ.get("DISPLAY"),
    }

    try:
        return run_dlstreamer_analyzer_container(
            docker_client=docker_client,
            docker_image_tag=docker_image_tag_analyzer,
            command=command,
            container_name=container_name,
            data_dir=data_dir,
            container_mnt_dir=container_mnt_dir,
            render_gid=render_gid,
            user_gid=user_gid,
            cpuset_cpus=cpuset_cpus,
            cpuset_mems=cpuset_mems,
            environment=environment,
            mode="server",
        )
    except RuntimeError as e:
        # Import here to avoid circular imports
        import pytest

        pytest.fail(str(e))


def run_concurrent_analysis(
    docker_client,
    device_dict: Dict[str, Any],
    analysis_tasks: Dict[str, Any],
    pipeline: str,
    pipeline_params: Dict[str, Dict[str, str]],
    docker_image_tag_analyzer: str,
    docker_container_prefix: str,
    data_dir: str,
    container_mnt_dir: str,
    pipeline_timeout: int,
    target_fps: float,
    num_sockets: int = 1,
    visualize_stream: bool = False,
    container_config: Dict[str, Any] = None,
) -> None:
    """
    Run analysis on multiple devices/configurations simultaneously.

    Args:
        analysis_tasks: Dictionary mapping device IDs to their configuration
        pipeline: Pipeline string to use for the benchmark
        pipeline_params: Dictionary of pipeline parameters
        num_sockets: Number of CPU sockets for multi-socket systems
    """
    logger.debug(f"Running concurrent analysis with tasks: {analysis_tasks}")
    containers = []

    for device_id, data in analysis_tasks.items():
        logger.info(f"[{device_id}] Running with {data['num_streams']} streams")
        # Handle special case for multi-socket CPU
        if device_id == "CPU" and num_sockets > 1:
            numa_info = get_cpu_socket_numa_info(num_sockets=num_sockets)
            # Divide streams evenly across sockets
            total_streams = data.get("num_streams", None)
            streams_per_socket = total_streams // num_sockets if total_streams is not None else None
            logger.info(
                f"[{device_id}] Multi-socket CPU: Dividing {total_streams} total streams "
                f"across {num_sockets} sockets ({streams_per_socket} streams per socket)"
            )
            for i, numa_node in enumerate(numa_info):
                # The container uses the run-id to create its unique result file, e.g., total_streams_result_0_CPU.json
                cpus = ",".join(map(str, numa_node["cpu_ids"]))
                mems = ",".join(map(str, numa_node["nodes"]))

                container = run_benchmark_container(
                    docker_client=docker_client,
                    run_id=i,
                    device_id=device_id,
                    device_dict=device_dict,
                    pipeline=pipeline,
                    pipeline_params=pipeline_params,
                    docker_image_tag_analyzer=docker_image_tag_analyzer,
                    docker_container_prefix=docker_container_prefix,
                    data_dir=data_dir,
                    container_mnt_dir=container_mnt_dir,
                    pipeline_timeout=pipeline_timeout,
                    target_fps=target_fps,
                    cpuset_cpus=cpus,
                    cpuset_mems=mems,
                    num_streams=streams_per_socket,
                    visualize_stream=visualize_stream,
                    container_config=container_config,
                )
                containers.append(container)
        else:
            container = run_benchmark_container(
                docker_client=docker_client,
                run_id=0,
                device_id=device_id,
                device_dict=device_dict,
                pipeline=pipeline,
                pipeline_params=pipeline_params,
                docker_image_tag_analyzer=docker_image_tag_analyzer,
                docker_container_prefix=docker_container_prefix,
                data_dir=data_dir,
                container_mnt_dir=container_mnt_dir,
                pipeline_timeout=pipeline_timeout,
                target_fps=target_fps,
                combined_analysis=analysis_tasks,
                num_streams=data.get("num_streams", None),
                visualize_stream=visualize_stream,
                container_config=container_config,
            )
            containers.append(container)

    wait_for_containers(containers)


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
    max_streams_above_baseline: int = 10,
    visualize_stream: bool = False,
    container_config: Dict[str, Any] = None,
) -> bool:
    """
    Iteratively find the max streams for a single device while others are active.
    Uses intelligent binary search strategy to quickly find optimal stream count.
    Handles multi-socket CPUs by running them concurrently and aggregating results.

    Args:
        consecutive_success_threshold: Number of consecutive successes before trying higher streams
        consecutive_failure_threshold: Number of consecutive failures before trying lower streams
        max_streams_above_baseline: Maximum streams to explore above baseline (limits +1 increment)
    """
    logger.debug(f"Active devices for qualification: {active_devices}")
    device_result_path = os.path.join(results_dir, f"total_streams_result_0_{device_id}.json")

    is_multisocket = device_id == "CPU" and num_sockets > 1

    # Initialize binary search bounds
    initial_streams = device_data["num_streams"]
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
    current_fps = 0.0
    current_analysis_status = "unknown"
    consecutive_successes = 0
    consecutive_failures = 0

    logger.debug(f"[{device_id}] Current FPS: {current_fps}")
    logger.info(
        f"[{device_id}] Starting intelligent qualification with {current_num_streams} streams "
        f"(search range: {min_streams}-{max_streams}), target FPS: {target_fps}"
    )
    logger.info(
        f"[{device_id}] Thresholds: {consecutive_success_threshold} consecutive successes to go higher, "
        f"{consecutive_failure_threshold} consecutive failures to go lower"
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
        previous_fps = device_data.get("per_stream_fps", 0)
        device_data["num_streams"] = current_num_streams
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
            visualize_stream=visualize_stream,
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
                all_socket_fps.append(result_data.get("per_stream_fps", 0))
                total_achieved_streams += result_data.get("num_streams", 0)

            current_fps = min(all_socket_fps) if all_socket_fps else 0
            device_data["per_stream_fps"] = current_fps
            device_data["num_streams"] = total_achieved_streams
        else:
            with open(device_result_path, "r") as file:
                latest_result = json.load(file)
            if device_id not in latest_result:
                logger.error(f"[{device_id}] No results found in {device_result_path}. Qualification failed.")
                return False

            current_analysis_status = latest_result[device_id].get("analysis_status", "unknown")
            if current_analysis_status != "success":
                # Check if this is a timeout above baseline (overcommitment) or below baseline (genuine failure)
                if current_num_streams > initial_streams and last_successful_streams > 0:
                    # Timeout above baseline with previous success - likely overcommitment
                    logger.warning(
                        f"[{device_id}] Timeout/failure at {current_num_streams} streams (above baseline "
                        f"{initial_streams}). Using last successful: {last_successful_streams} streams."
                    )
                    # Treat as convergence - use last successful configuration
                    device_data["pass"] = True
                    device_data["per_stream_fps"] = last_successful_fps
                    device_data["num_streams"] = last_successful_streams

                    # Restore last successful FPS to all active devices
                    for other_dev_id in active_devices:
                        if other_dev_id != device_id and "last_successful_fps" in active_devices[other_dev_id]:
                            active_devices[other_dev_id]["per_stream_fps"] = active_devices[other_dev_id][
                                "last_successful_fps"
                            ]

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
                    # Timeout at or below baseline - genuine failure
                    logger.error(
                        f"[{device_id}] Failed at {current_num_streams} streams (baseline: {initial_streams}) "
                        f"due to analysis status '{current_analysis_status}'. Device disqualified."
                    )
                    return False

            current_fps = latest_result[device_id].get("per_stream_fps", 0)
            device_data.update(latest_result[device_id])

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
                        all_socket_fps.append(result_data.get("per_stream_fps", 0))

                    # Update CPU device metrics with aggregate data
                    if all_socket_fps:
                        dev_fps = min(all_socket_fps)
                        combined_analysis[dev_id]["per_stream_fps"] = dev_fps
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
                            dev_fps = latest_result[dev_id].get("per_stream_fps", 0)
                            # Update the device metrics with latest data
                            combined_analysis[dev_id]["per_stream_fps"] = dev_fps
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
            device_data["last_successful_fps"] = last_successful_fps
            consecutive_successes += 1
            consecutive_failures = 0  # Reset failure counter on success

            # Track last successful FPS for all other active devices
            for other_dev_id, other_dev_data in combined_analysis.items():
                if other_dev_id != device_id and other_dev_id in active_devices:
                    other_fps = other_dev_data.get("per_stream_fps", 0)
                    if other_fps >= target_fps:
                        logger.debug(f"Saving last successful FPS for {other_dev_id}: {other_fps:.2f}")
                        active_devices[other_dev_id]["last_successful_fps"] = other_fps

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
                device_data["per_stream_fps"] = current_fps

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
                        device_data["per_stream_fps"] = current_fps
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
            # Failed to meet target - update failure counter
            consecutive_successes = 0  # Reset success counter on failure
            consecutive_failures += 1

            # Check if we should try lower based on consecutive failure threshold
            if consecutive_failures >= consecutive_failure_threshold:
                # Update upper bound
                max_streams = current_num_streams

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
                        device_data["per_stream_fps"] = last_successful_fps
                        device_data["num_streams"] = last_successful_streams

                        # Restore last successful FPS to all active devices
                        for other_dev_id in active_devices:
                            if other_dev_id != device_id and "last_successful_fps" in active_devices[other_dev_id]:
                                active_devices[other_dev_id]["per_stream_fps"] = active_devices[other_dev_id][
                                    "last_successful_fps"
                                ]

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
                        logger.warning(
                            f"[{device_id}] Cannot meet target FPS. "
                            f"Current FPS: {current_fps:.2f}, Target: {target_fps:.2f}"
                        )
                        for dev_id, dev_data in combined_analysis.items():
                            logger.warning(f"  - {dev_id}: {dev_data.get('per_stream_fps', 0):.2f} FPS")

                        device_data["pass"] = False
                        device_data["per_stream_fps"] = current_fps
                        device_data["num_streams"] = 0
                        logger.error(
                            f"[{device_id}] Disqualified after {iteration} iterations (max: {max_iterations})."
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
        device_data["num_streams"] = last_successful_streams
        device_data["per_stream_fps"] = last_successful_fps
        device_data["pass"] = True

        # Revert all active devices to their last successful FPS values
        logger.info("Reverting other active devices to their last successful FPS values")
        for other_dev_id, other_dev_data in active_devices.items():
            if other_dev_id != device_id and "last_successful_fps" in other_dev_data:
                other_last_fps = other_dev_data["last_successful_fps"]
                logger.info(f"  - Reverting {other_dev_id} to last successful FPS: {other_last_fps:.2f}")
                other_dev_data["per_stream_fps"] = other_last_fps

    # If there's absolutely no successful configuration, device failed
    else:
        logger.error(f"[{device_id}] No successful configuration found. Qualification failed.")
        device_data["num_streams"] = 0
        device_data["pass"] = False

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
