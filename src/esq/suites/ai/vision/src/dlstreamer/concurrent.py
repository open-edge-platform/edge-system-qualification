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
    """Get CPU socket and NUMA node information for multi-socket systems.

    For systems with insufficient NUMA nodes (e.g., 1-socket system testing dual-socket scenario),
    generates simulated NUMA configuration for testing purposes.
    """
    num_of_numa_nodes = list(numa.info.numa_hardware_info()["node_cpu_info"].keys())

    # Handle insufficient NUMA nodes - simulate configuration for testing
    if len(num_of_numa_nodes) < num_sockets:
        logger.warning(
            f"System has {len(num_of_numa_nodes)} NUMA node(s) but {num_sockets} socket(s) requested. "
            f"Generating simulated NUMA configuration for testing."
        )
        # Create simulated socket configurations with evenly distributed CPUs
        available_cpus = (
            numa.info.node_to_cpus(num_of_numa_nodes[0]) if num_of_numa_nodes else list(range(os.cpu_count()))
        )
        cpus_per_socket = max(1, len(available_cpus) // num_sockets)

        cpu_socket = []
        for i in range(num_sockets):
            start_cpu = i * cpus_per_socket
            end_cpu = start_cpu + cpus_per_socket if i < num_sockets - 1 else len(available_cpus)
            socket_cpus = available_cpus[start_cpu:end_cpu]

            # Remove 2 CPUs if we have enough, to avoid overloading
            if len(socket_cpus) > 2:
                socket_cpus = socket_cpus[:-2]

            cpu_socket.append({"nodes": [num_of_numa_nodes[0]] if num_of_numa_nodes else [0], "cpu_ids": socket_cpus})

        return cpu_socket

    # Normal path: sufficient NUMA nodes available
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
    base_run_id: int = 1,
) -> None:
    """
    Run analysis on multiple devices/configurations simultaneously.

    Args:
        analysis_tasks: Dictionary mapping device IDs to their configuration
        pipeline: Pipeline string to use for the benchmark
        pipeline_params: Dictionary of pipeline parameters
        num_sockets: Number of CPU sockets for multi-socket systems
        container_config: Container configuration with available images (optional)
        base_run_id: Base run ID for result file naming (default: 1)
                     - run_id=0: Reserved for qualification_state storage only (no actual execution)
                     - run_id=1: Primary execution (single device or socket 1 in multi-socket)
                     - run_id=2+: Additional sockets in multi-socket CPU scenarios

                     Standard practice: Always use base_run_id >= 1 for actual container execution.
                     The qualification state is separately saved to run_id=0 after execution completes.
                     - Use 0 only for qualification state storage (reserved)
                     - For multi-socket: run_id = base_run_id + socket_index
                       Example: base_run_id=1 → socket 0: run_id=1, socket 1: run_id=2
                     - For single-device: run_id = base_run_id

    Result File Naming Convention:
        - total_streams_result_0_{device_id}.json: Qualification state (reserved)
        - total_streams_result_1_{device_id}.json: Socket 0 or single-device results
        - total_streams_result_2_{device_id}.json: Socket 1 results (multi-socket only)
        - And so on for additional sockets
    """
    # Log concise summary of analysis tasks
    task_summary = ", ".join(
        f"{dev_id}:{data.get('qualification_state', {}).get('num_streams', 0)} streams"
        for dev_id, data in analysis_tasks.items()
    )
    logger.debug(f"Running concurrent analysis (run_id={base_run_id}): {task_summary}")
    containers = []

    for device_id, data in analysis_tasks.items():
        qual_state = data.get("qualification_state", {})
        num_streams = qual_state.get("num_streams", 0)
        logger.info(f"[{device_id}] Running with {num_streams} streams (base_run_id={base_run_id})")
        # Handle special case for multi-socket CPU
        if device_id == "CPU" and num_sockets > 1:
            numa_info = get_cpu_socket_numa_info(num_sockets=num_sockets)
            # Divide streams evenly across sockets
            total_streams = num_streams
            streams_per_socket = total_streams // num_sockets if total_streams is not None else None

            # Skip multi-socket execution if streams are insufficient (less than num_sockets)
            # Fall back to single-socket execution to avoid invalid 0-stream containers
            if streams_per_socket is None or streams_per_socket < 1:
                logger.warning(
                    f"[{device_id}] Insufficient streams ({total_streams}) for multi-socket execution "
                    f"across {num_sockets} sockets. Falling back to single-socket execution."
                )
                # Run on first socket only with all streams
                numa_node = numa_info[0]
                cpus = ",".join(map(str, numa_node["cpu_ids"]))
                mems = ",".join(map(str, numa_node["nodes"]))

                container = run_benchmark_container(
                    docker_client=docker_client,
                    run_id=base_run_id,  # Use base_run_id for single socket
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
                    num_streams=total_streams,  # Use all streams on single socket
                    container_config=container_config,
                )
                containers.append(container)
            else:
                # Normal multi-socket execution with sufficient streams
                logger.info(
                    f"[{device_id}] Multi-socket CPU: Dividing {total_streams} total streams "
                    f"across {num_sockets} sockets ({streams_per_socket} streams per socket)"
                )
                for i, numa_node in enumerate(numa_info):
                    cpus = ",".join(map(str, numa_node["cpu_ids"]))
                    mems = ",".join(map(str, numa_node["nodes"]))

                    container = run_benchmark_container(
                        docker_client=docker_client,
                        run_id=base_run_id + i,  # Use base_run_id + socket index
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
                run_id=base_run_id,  # Use base_run_id directly
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
