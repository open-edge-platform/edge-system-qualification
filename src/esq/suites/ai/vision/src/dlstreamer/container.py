# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer container management utilities."""

import json
import logging
import os

from sysagent.utils.infrastructure import DockerClient

logger = logging.getLogger(__name__)


def run_video_utils_container(
    docker_client: DockerClient,
    docker_image_tag,
    command,
    container_name,
    data_dir,
    container_mnt_dir,
    working_dir=None,
    result_file=None,
    container_result_file_dir=None,
    environment=None,
    volumes=None,
):
    """
    Run a container with the video utilities image.

    Args:
        docker_client: Docker client instance
        docker_image_tag: Docker image tag to use
        command: Command to run in the container
        container_name: Name for the container
        data_dir: Host data directory to mount
        container_mnt_dir: Container mount point
        result_file: Optional result file name
        container_result_file_dir: Optional directory for result file
        volumes: Additional volumes to mount (optional)

    Returns:
        Container execution result
    """
    user_gid = os.getgid()
    container_volumes = {
        data_dir: {"bind": container_mnt_dir, "mode": "rw"},
    }

    # Add any additional volumes if provided
    if volumes:
        container_volumes.update(volumes)

    # Ensure results directory exists and has correct permissions
    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    os.chmod(results_dir, 0o770)

    result = docker_client.run_container(
        name=container_name,
        image=docker_image_tag,
        volumes=container_volumes,
        environment=environment,
        group_add=[user_gid],
        mode="batch",
        entrypoint="/bin/bash",
        command=["-c", command],
        working_dir=working_dir,
        result_file=result_file,
        container_result_file_dir=container_result_file_dir,
    )

    if result.get("container_info", {}).get("exit_code") != 0:
        error_message = (
            f"Failed to run container {container_name}: {result.get('container_logs_text', 'Unknown error')}"
        )
        logger.error(error_message)
        raise RuntimeError(error_message)

    return result


def run_dlstreamer_analyzer_container(
    docker_client: DockerClient,
    docker_image_tag,
    command,
    container_name,
    data_dir,
    container_mnt_dir,
    render_gid,
    user_gid,
    cpuset_cpus="",
    cpuset_mems="",
    environment=None,
    mode="batch",
    result_file=None,
    container_result_file_dir=None,
    volumes=None,
):
    """
    Run a container with the DL Streamer analyzer image.

    Args:
        docker_client: Docker client instance
        docker_image_tag: Docker image tag to use
        command: Command to run in the container (or arguments if using entrypoint)
        container_name: Name for the container
        data_dir: Host data directory to mount
        container_mnt_dir: Container mount point
        render_gid: Render group ID for GPU access
        user_gid: User group ID
        cpuset_cpus: CPU set to use (optional)
        cpuset_mems: Memory nodes to use (optional)
        environment: Environment variables dictionary (optional)
        mode: Container execution mode ('batch' or 'server')
        result_file: Optional result file name
        container_result_file_dir: Optional directory for result file
        volumes: Additional volumes to mount (optional)

    Returns:
        Container or execution result depending on mode
    """
    # Base volumes - always mount the data directory
    container_volumes = {
        "/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "rw"},
        data_dir: {"bind": container_mnt_dir, "mode": "rw"},
    }

    # Add any additional volumes if provided
    if volumes:
        container_volumes.update(volumes)

    container_devices = ["/dev/dri:/dev/dri"]
    if os.path.exists("/dev/accel"):
        container_devices.append("/dev/accel:/dev/accel")

    # Ensure results directory exists and has correct permissions
    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    os.chmod(results_dir, 0o770)

    # Ensure logs directory exists and has correct permissions
    logs_dir = os.path.join(data_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    os.chmod(logs_dir, 0o770)

    if environment is None:
        environment = {}

    # enable GST debug based on env variable if set
    if os.getenv("GST_DEBUG"):
        logger.debug(f"Enabling GST_DEBUG with level: '{os.getenv('GST_DEBUG')}'")
        environment["GST_DEBUG"] = os.getenv("GST_DEBUG")
        environment["GST_DEBUG_NO_COLOR"] = "1"
    else:
        logger.debug("GST_DEBUG not set, running with default GStreamer logging")

    common_params = {
        "name": container_name,
        "image": docker_image_tag,
        "environment": environment,
        "volumes": container_volumes,
        "devices": container_devices,
        "group_add": [render_gid, user_gid],
        "cpuset_cpus": cpuset_cpus,
        "cpuset_mems": cpuset_mems,
        "command": command,  # Pass command as arguments to the entrypoint
        "working_dir": container_mnt_dir,
    }

    if mode == "batch":
        # Batch mode - run and wait for completion
        common_params["mode"] = "batch"
        common_params["result_file"] = result_file
        common_params["container_result_file_dir"] = container_result_file_dir

        result = docker_client.run_container(**common_params)

        # Exclude certain keys from result output in debug log
        exclude_keys = {"container_logs_text", "result_text"}
        result_log = {k: v for k, v in result.items() if k not in exclude_keys}
        logger.debug(f"Container {container_name} execution result: {json.dumps(result_log, indent=2)}")

        if result.get("container_info", {}).get("exit_code") != 0:
            error_message = (
                f"Failed to run container {container_name}: {result.get('container_logs_text', 'Unknown error')}"
            )
            logger.error(error_message)
            raise RuntimeError(error_message)
        else:
            logger.info(f"Container {container_name} completed successfully")

        return result
    else:  # server mode
        # Server mode - run in background
        common_params["mode"] = "server"

        container = docker_client.run_container(**common_params)

        if not container:
            error_message = f"Failed to run container {container_name} with command: {command}"
            logger.error(error_message)
            raise RuntimeError(error_message)

        logger.debug(f"Container {container_name} started successfully")
        docker_client.start_log_streaming(container, container_name)

        return container
