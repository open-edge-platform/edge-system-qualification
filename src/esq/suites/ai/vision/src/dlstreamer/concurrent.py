# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer concurrent analysis and container orchestration functions."""

import grp
import json
import logging
import os
from typing import Any, Dict, List

import docker
import numa

from .container import run_dlstreamer_analyzer_container
from .pipeline import (
    build_multi_pipeline_with_devices,
    get_fpscounter_config,
    get_sink_element_config,
    resolve_pipeline_placeholders,
)
from .preparation import get_device_specific_docker_image

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
        sink_element = get_sink_element_config(pipeline_params, device_id, device_dict)
        fpscounter_config = get_fpscounter_config(pipeline_params, device_id, device_dict)
        multi_pipeline, result_pipeline = build_multi_pipeline_with_devices(
            pipeline=resolved_pipeline,
            device_id=device_id,
            num_streams=num_streams,
            sink_element=sink_element,
            fpscounter_elements=fpscounter_config,
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
        qual_state = data.get("qualification_state", {})
        num_streams = qual_state.get("num_streams", 0)
        logger.info(f"[{device_id}] Running with {num_streams} streams")
        # Handle special case for multi-socket CPU
        if device_id == "CPU" and num_sockets > 1:
            numa_info = get_cpu_socket_numa_info(num_sockets=num_sockets)
            # Divide streams evenly across sockets
            total_streams = num_streams
            streams_per_socket = total_streams // num_sockets if total_streams is not None else None
            logger.info(
                f"[{device_id}] Multi-socket CPU: Dividing {total_streams} total streams "
                f"across {num_sockets} sockets ({streams_per_socket} streams per socket)"
            )
            for i, numa_node in enumerate(numa_info):
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
                num_streams=num_streams,
                container_config=container_config,
            )
            containers.append(container)

    wait_for_containers(containers)


def confirm_steady_state_concurrent_analysis(
    docker_client,
    device_dict: Dict[str, Any],
    qualified_devices: Dict[str, Any],
    pipeline: str,
    pipeline_params: Dict[str, Dict[str, str]],
    docker_image_tag_analyzer: str,
    docker_container_prefix: str,
    data_dir: str,
    container_mnt_dir: str,
    pipeline_timeout: int,
    results_dir: str,
    target_fps: float,
    num_sockets: int = 1,
    confirmation_threshold: int = 2,
    container_config: Dict[str, Any] = None,
) -> bool:
    """
    Run final concurrent analysis with all qualified devices and verify steady-state stability.

    This function ensures that the final concurrent metrics are stable by running multiple
    confirmation runs. If devices cannot consistently achieve target FPS, it reduces stream
    counts (only downward) and retries until a stable configuration is found.

    Args:
        docker_client: Docker client instance
        device_dict: Dictionary containing device information
        qualified_devices: Dictionary of qualified devices with their configurations
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
        confirmation_threshold: Number of consecutive successful runs required (default: 2)
        container_config: Container configuration with available images

    Returns:
        True if steady-state confirmed, False otherwise
    """
    if len(qualified_devices) <= 1:
        logger.info("Only one device qualified - skipping steady-state confirmation")
        return True

    logger.info(
        f"Running steady-state confirmation with {len(qualified_devices)} qualified devices "
        f"(confirmation threshold: {confirmation_threshold})"
    )

    # Track confirmation attempts for each device
    device_confirmation_counts = {device_id: 0 for device_id in qualified_devices.keys()}
    max_reduction_attempts = 10  # Maximum number of stream reduction attempts
    reduction_attempt = 0

    while reduction_attempt < max_reduction_attempts:
        # Check if all devices have been confirmed
        if all(count >= confirmation_threshold for count in device_confirmation_counts.values()):
            logger.info(
                f"Steady-state confirmed for all {len(qualified_devices)} devices after "
                f"{reduction_attempt} reduction attempts"
            )
            return True

        # Run concurrent analysis with current stream configurations
        logger.info(
            f"Running confirmation attempt (reduction iteration {reduction_attempt + 1}/{max_reduction_attempts})"
        )

        run_concurrent_analysis(
            docker_client=docker_client,
            device_dict=device_dict,
            analysis_tasks=qualified_devices,
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

        # Parse results and check if devices met target FPS
        devices_needing_reduction = []

        for device_id in qualified_devices.keys():
            result_file = os.path.join(results_dir, f"total_streams_result_0_{device_id}.json")

            if not os.path.exists(result_file):
                logger.warning(f"Result file not found for {device_id}: {result_file}")
                device_confirmation_counts[device_id] = 0
                devices_needing_reduction.append(device_id)
                continue

            try:
                with open(result_file, "r") as f:
                    result_data = json.load(f)

                if device_id not in result_data:
                    logger.warning(f"Device {device_id} not found in result file")
                    device_confirmation_counts[device_id] = 0
                    devices_needing_reduction.append(device_id)
                    continue

                device_data = result_data[device_id]
                metadata = device_data.get("metadata", {})
                per_stream_fps = metadata.get("per_stream_fps", 0)
                num_streams = metadata.get("num_streams", 0)

                # Check if device achieved target FPS
                if per_stream_fps >= target_fps:
                    device_confirmation_counts[device_id] += 1
                    logger.info(
                        f"[{device_id}] Confirmation {device_confirmation_counts[device_id]}/{confirmation_threshold}: "
                        f"{num_streams} streams at {per_stream_fps:.2f} FPS (target: {target_fps:.2f})"
                    )

                    # Update qualified devices metadata with confirmed metrics
                    if "metadata" in qualified_devices[device_id]:
                        qualified_devices[device_id]["metadata"].update(metadata)
                else:
                    # Reset confirmation count and mark for reduction
                    current_count = device_confirmation_counts[device_id]
                    if current_count > 0:
                        logger.debug(
                            f"[{device_id}] Failed to achieve target FPS: {per_stream_fps:.2f} < {target_fps:.2f} "
                            f"(resetting confirmation count from {current_count} to 0)"
                        )
                    else:
                        logger.debug(
                            f"[{device_id}] Failed to achieve target FPS: {per_stream_fps:.2f} < {target_fps:.2f}"
                        )
                    device_confirmation_counts[device_id] = 0
                    devices_needing_reduction.append(device_id)

            except Exception as e:
                logger.error(f"Failed to parse results for {device_id}: {e}")
                device_confirmation_counts[device_id] = 0
                devices_needing_reduction.append(device_id)

        # If all devices are confirmed, we're done
        if not devices_needing_reduction:
            logger.info("All devices confirmed - steady-state achieved")
            return True

        # Reduce streams for devices that didn't meet target FPS
        logger.info(f"Reducing streams for devices: {devices_needing_reduction}")

        for device_id in devices_needing_reduction:
            qual_state = qualified_devices[device_id].get("qualification_state", {})
            current_streams = qual_state.get("num_streams", 1)

            # Reduce by 1 stream (only downward adjustment)
            new_streams = max(1, current_streams - 1)

            if new_streams == current_streams:
                logger.warning(
                    f"[{device_id}] Already at minimum streams (1) - cannot reduce further. "
                    "Steady-state may not be achievable."
                )
                continue

            logger.info(
                f"[{device_id}] Reducing streams from {current_streams} to {new_streams} to find stable configuration"
            )

            # Update qualification state with new stream count
            qualified_devices[device_id]["qualification_state"]["num_streams"] = new_streams

            # Reset confirmation count for this device since we changed configuration
            device_confirmation_counts[device_id] = 0

        reduction_attempt += 1

    # Max reduction attempts reached
    logger.warning(
        f"Failed to achieve steady-state after {max_reduction_attempts} reduction attempts. "
        "Using last known configuration."
    )
    return False
