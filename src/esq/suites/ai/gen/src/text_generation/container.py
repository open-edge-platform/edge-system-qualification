# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Text Generation container management functions."""

import logging
import os
import time
from typing import Any, Dict

import requests
from sysagent.utils.infrastructure import DockerClient

logger = logging.getLogger(__name__)


def run_ovms_server_container(
    docker_client: DockerClient,
    device_id: str,
    docker_image_tag: str,
    docker_container_prefix: str,
    data_dir: str,
    container_mnt_dir: str,
    server_timeout: int = 300,
    model_id: str = "text_generation",
    port: int = 8000,
) -> Dict[str, Any]:
    """
    Run OVMS server container for text generation.

    Args:
        docker_client: Docker client instance
        device_id: Target device ID
        docker_image_tag: Docker image tag
        docker_container_prefix: Container name prefix
        data_dir: Host data directory
        container_mnt_dir: Container mount directory
        server_timeout: Server startup timeout
        model_id: Model identifier
        port: HTTP port for server

    Returns:
        Dict with container information and configuration
    """
    # Normalize model_id to use just the basename and make it lowercase
    model_basename = os.path.basename(model_id.replace("/", "_")).lower()
    container_name = f"{docker_container_prefix}server_{model_basename}_{device_id.replace('.', '_')}".lower()

    # Container configuration
    container_config = {
        "name": container_name,
        "image": docker_image_tag,
        "device_id": device_id,
        "model_id": model_id,
        "port": port,
        "timeout": server_timeout,
    }

    # Volume mounts - mount models directory
    models_dir = os.path.join(data_dir, "models")
    volumes = {models_dir: {"bind": "/workspace", "mode": "rw"}}

    # Port mapping - bind only to localhost (127.0.0.1) for security
    ports = {"8000/tcp": ("127.0.0.1", 8000)}

    # Device access for GPU - use original format
    container_devices = ["/dev/dri:/dev/dri"]
    if os.path.exists("/dev/accel"):
        container_devices.append("/dev/accel:/dev/accel")

    # Group access for GPU devices (like original implementation)
    import grp

    user_gid = os.getuid()
    render_gid = grp.getgrnam("render").gr_gid

    # OVMS command - use original fixed ports and paths
    ovms_cmd = ["--port", "9000", "--rest_port", "8000", "--config_path", "/workspace/config_all.json"]

    try:
        logger.info(f"Starting OVMS container: {container_name}")
        logger.debug(f"OVMS command: {' '.join(ovms_cmd)}")

        # Use docker_client.run_container method like original implementation
        container = docker_client.run_container(
            name=container_name,
            image=docker_image_tag,
            group_add=[render_gid, user_gid],
            ports=ports,
            volumes=volumes,
            devices=container_devices,
            mode="server",
            command=ovms_cmd,
        )

        container_config["container_id"] = container.id
        container_config["container"] = container

        logger.info(f"OVMS container started: {container.id[:12]}")

        # Start log streaming like original implementation
        docker_client.start_log_streaming(container, container_name)

        # Wait for container to be ready using original model availability check
        if not wait_for_ovms_model_ready(model_id, 8000, server_timeout):
            logger.error("OVMS server failed to become ready")
            docker_client.stop_log_streaming(container_name)
            docker_client.remove_container(container.id)
            raise RuntimeError("OVMS server startup failed")

        return container_config

    except Exception as e:
        logger.error(f"Failed to start OVMS container: {e}")
        raise


def wait_for_ovms_model_ready(model_id: str, port: int, timeout: int = 300) -> bool:
    """
    Wait for OVMS model to be ready - using original implementation logic.

    Args:
        model_id: Model identifier to check
        port: HTTP port to check
        timeout: Maximum wait time in seconds

    Returns:
        True if model becomes ready, False otherwise
    """
    api_url = f"http://localhost:{port}/v1/config"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                server_results = response.json()
                if (
                    server_results
                    and model_id in server_results
                    and "model_version_status" in server_results[model_id]
                    and server_results[model_id]["model_version_status"]
                    and len(server_results[model_id]["model_version_status"]) > 0
                    and "state" in server_results[model_id]["model_version_status"][0]
                    and server_results[model_id]["model_version_status"][0]["state"] == "AVAILABLE"
                ):
                    logger.info("OVMS model is available and ready.")
                    return True
                else:
                    logger.info("OVMS service is up but model not ready, retrying ...")
            else:
                logger.info(f"OVMS service not ready, status code: {response.status_code}")
        except requests.ConnectionError:
            logger.info("OVMS service not ready, retrying ...")
        except Exception as e:
            logger.error(f"Error checking OVMS service readiness: {e}")
        time.sleep(10)

    logger.error(f"Model initialization timed out after {timeout} seconds")
    return False


def run_benchmark_container(
    docker_client: DockerClient,
    docker_container_prefix: str,
    data_dir: str,
    model_id: str,
    model_precision: str,
    device_id: str,
    dataset_path: str,
    hf_dataset_filename: str,
    ovms_port: int,
    test_num_prompts: int,
    test_request_rate: int,
    test_max_concurrent_requests: int,
    benchmark_timeout: int = 300,
) -> Dict[str, Any]:
    """
    Run benchmark container for performance testing using pre-built image with uv package manager.

    Args:
        docker_client: Docker client instance
        docker_container_prefix: Container name prefix
        data_dir: Host data directory
        model_id: Model identifier
        model_precision: Model precision
        device_id: Device identifier
        dataset_path: Path to dataset file
        hf_dataset_filename: Dataset filename
        ovms_port: OVMS server port
        test_num_prompts: Number of prompts to test
        test_request_rate: Request rate
        test_max_concurrent_requests: Max concurrent requests
        benchmark_timeout: Benchmark timeout

    Returns:
        Dict with benchmark results
    """
    # Normalize model_id to use just the basename and make it lowercase
    model_basename = os.path.basename(model_id.replace("/", "_")).lower()
    container_name = f"{docker_container_prefix}benchmark_{model_basename}_{device_id.replace('.', '_')}".lower()

    # Results directory setup with proper permissions like original
    results_dir = os.path.join(data_dir, "results", "text_generation")
    os.makedirs(results_dir, exist_ok=True)

    # Ensure directories have correct permissions like original
    from sysagent.utils.config import ensure_dir_permissions

    ensure_dir_permissions(results_dir, uid=os.getuid(), gid=os.getgid(), mode=0o775)

    # Volume mounts - similar to original implementation
    volumes = {
        dataset_path: {"bind": f"/vllm-workspace/benchmarks/{hf_dataset_filename}", "mode": "ro"},
        results_dir: {"bind": "/vllm-workspace/results", "mode": "rw"},
    }

    # Build benchmark command using original implementation approach (runtime pip install + fixed port)
    benchmark_cmd = [
        "-c",
        (
            "python3 /vllm-workspace/benchmarks/benchmark_serving.py "
            f"--host 127.0.0.1 "  # Use localhost to access OVMS server bound to 127.0.0.1
            "--port 8000 "  # Use fixed port like original
            "--endpoint /v3/chat/completions "
            "--backend openai-chat "
            f"--model {model_id} "
            f"--dataset-path /vllm-workspace/benchmarks/{hf_dataset_filename} "
            f"--request-rate {test_request_rate} "
            f"--max-concurrency {test_max_concurrent_requests} "
            f"--num-prompts {test_num_prompts} "
            "--save-result "
            "--result-dir /vllm-workspace/results "
            f"--result-filename ovms-{model_id.replace('/', '_')}-{model_precision}-{device_id}.json"
        ),
    ]

    try:
        logger.info(f"Starting benchmark container: {container_name}")
        logger.debug(f"Benchmark command: {' '.join(benchmark_cmd)}")

        # Run the benchmark container using pre-built image
        benchmark_image_tag = "genai-benchmark:latest"
        logger.info(f"Starting OVMS benchmark container {container_name} with timeout {benchmark_timeout} seconds")

        # Set environment to ensure container logs are collected
        os.environ["CORE_SUPPRESS_CONTAINER_LOG_ATTACHMENTS"] = "1"

        # Get host user and group IDs for group_add approach
        render_gid = os.getgid()  # Host group ID
        user_gid = os.getgid()  # Host user group ID

        result = docker_client.run_container(
            image=benchmark_image_tag,
            name=container_name,
            volumes=volumes,
            mode="batch",
            entrypoint="/bin/bash",
            command=benchmark_cmd,
            timeout=benchmark_timeout,
            group_add=[render_gid, user_gid],  # Use group_add for better security model
            network_mode="host",  # Use host networking to access localhost services
        )

        logger.info(f"OVMS benchmark container {container_name} executed successfully")

        # Return result information with container details
        return {
            "container_name": container_name,
            "exit_code": result.get("container_info", {}).get("exit_code", 0),
            "logs": result.get("container_logs_text", ""),
            "success": True,
            "container_info": result.get("container_info", {}),
            "results_file": os.path.join(
                results_dir, f"ovms-{model_id.replace('/', '_')}-{model_precision}-{device_id}.json"
            ),
        }

    except Exception as e:
        logger.error(f"Failed to run benchmark container: {e}")
        raise


def cleanup_containers(docker_client: DockerClient, container_prefix: str) -> None:
    """
    Clean up containers with the given prefix.

    Args:
        docker_client: Docker client instance
        container_prefix: Container name prefix to match
    """
    try:
        containers = docker_client.client.containers.list(all=True)
        for container in containers:
            if container.name.startswith(container_prefix):
                try:
                    logger.info(f"Stopping container: {container.name}")
                    container.stop(timeout=10)
                    container.remove()
                except Exception as e:
                    logger.warning(f"Failed to clean up container {container.name}: {e}")
    except Exception as e:
        logger.error(f"Failed to cleanup containers: {e}")
