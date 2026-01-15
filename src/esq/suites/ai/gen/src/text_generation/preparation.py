# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Text Generation preparation functions for assets and environment setup."""

import logging
import os
from typing import Any, Dict

import allure
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Result
from sysagent.utils.infrastructure import DockerClient

logger = logging.getLogger(__name__)


def extract_benchmark_requirements_freeze(docker_client: DockerClient, image_tag: str) -> str:
    """
    Extract requirements freeze from benchmark container using uv pip freeze command.

    Args:
        docker_client: Docker client instance
        image_tag: Built container image tag

    Returns:
        Requirements freeze content as string
    """
    try:
        logger.info(f"Extracting requirements using 'uv pip freeze' from container {image_tag}")

        result = docker_client.client.containers.run(
            image=image_tag,
            entrypoint="/bin/bash",
            command=["-c", "uv pip freeze"],
            remove=True,
            stdout=True,
            stderr=True,
        )

        if isinstance(result, bytes):
            freeze_content = result.decode("utf-8").strip()
        else:
            freeze_content = str(result).strip()

        if freeze_content:
            logger.info(f"Successfully extracted {len(freeze_content.split())} lines using 'uv pip freeze'")
            return freeze_content
        else:
            logger.warning("uv pip freeze returned empty results")
            return ""

    except Exception as e:
        logger.error(f"Failed to extract requirements using 'uv pip freeze' from {image_tag}: {e}")
        return ""


def prepare_assets(
    docker_client: DockerClient,
    docker_image_tag: str,
    benchmark_docker_base_image: str,
    ovms_docker_base_image: str,
    dataset_repo: str,
    dataset_source: str,
    dataset_filename: str,
    dataset_path: str,
    models_dir: str,
    thirdparty_dir: str,
    src_dir: str,
    download_timeout: float = 300,
    docker_buildargs: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Prepare all necessary assets for the Text Generation test.

    Args:
        docker_client: Docker client instance
        docker_image_tag: Docker image tag for OVMS
        benchmark_docker_base_image: Benchmark Docker base image
        ovms_docker_base_image: OVMS Docker base image
        dataset_repo: Dataset repository ID (HuggingFace or ModelScope format)
        dataset_source: Dataset source ("huggingface" or "modelscope")
        dataset_filename: Filename within the dataset repository
        dataset_path: Local dataset path
        models_dir: Directory for model files
        thirdparty_dir: Directory for third-party data
        src_dir: Source directory
        download_timeout: Timeout for dataset download in seconds (default: 300)
        docker_buildargs: Docker build arguments

    Returns:
        Dict with preparation results and configuration
    """
    logger.info("Preparing assets for Text Generation test...")

    # Prepare directories
    ensure_dir_permissions(models_dir, uid=os.getuid(), gid=os.getgid())
    ensure_dir_permissions(thirdparty_dir, uid=os.getuid(), gid=os.getgid())

    # Pull base images first (required before extracting digests and building)
    logger.info("Pulling base Docker images")
    try:
        logger.info(f"Pulling OVMS base image: {ovms_docker_base_image}")
        docker_client.pull_image(ovms_docker_base_image)

        logger.info(f"Pulling benchmark base image: {benchmark_docker_base_image}")
        docker_client.pull_image(benchmark_docker_base_image)
    except Exception as e:
        logger.error(f"Failed to pull base images: {e}")
        raise

    # Extract base image digests (now that images are pulled)
    logger.info("Extracting base image digests")
    base_image_digests = {}

    # Extract OVMS base image digest
    ovms_digest = docker_client.extract_docker_image_digest(ovms_docker_base_image)
    if ovms_digest:
        base_image_digests["ovms_base"] = {"image": ovms_docker_base_image, "digest": f"sha256:{ovms_digest}"}

    # Extract benchmark base image digest
    benchmark_digest = docker_client.extract_docker_image_digest(benchmark_docker_base_image)
    if benchmark_digest:
        base_image_digests["benchmark_base"] = {
            "image": benchmark_docker_base_image,
            "digest": f"sha256:{benchmark_digest}",
        }

    preparation_results = {
        "base_image_digests": base_image_digests,
        "build_timestamp": os.environ.get("BUILD_TIMESTAMP", ""),
    }

    # Build OVMS Docker image
    try:
        logger.info(f"Building OVMS Docker image: {docker_image_tag}")

        # Use containers folder for build context
        container_src_dir = os.path.join(os.path.dirname(src_dir), "containers", "ovms_server")

        build_result = docker_client.build_image(
            path=container_src_dir, tag=docker_image_tag, buildargs=docker_buildargs
        )

        container_config = {
            "image_id": str(build_result.get("image_id", "")) if build_result.get("image_id") else "",
            "image_tag": str(docker_image_tag),
            "docker_image": str(build_result.get("docker_image", "")) if build_result.get("docker_image") else "",
        }

        if not container_config["image_id"]:
            raise RuntimeError(f"Failed to build Docker image: {docker_image_tag}")
        else:
            logger.info(f"Successfully built Docker image: {container_config['image_id'][:12]}")

        preparation_results["ovms_container"] = container_config

    except Exception as e:
        logger.error(f"Failed to build OVMS Docker image: {e}")
        raise

    # Build benchmark Docker image with uv package manager and pre-installed dependencies
    try:
        benchmark_dockerfile_dir = os.path.join(src_dir, "..", "containers", "benchmark")
        benchmark_image_tag = "genai-benchmark:latest"

        logger.info(f"Building benchmark Docker image: {benchmark_image_tag}")
        build_result = docker_client.build_image(
            path=benchmark_dockerfile_dir, tag=benchmark_image_tag, dockerfile="Dockerfile"
        )

        if not build_result.get("image_id"):
            raise RuntimeError(f"Failed to build benchmark Docker image: {benchmark_image_tag}")

        preparation_results["benchmark_container"] = {
            "image_id": str(build_result.get("image_id", "")),
            "image_tag": str(benchmark_image_tag),
            "docker_image": str(build_result.get("docker_image", "")),
        }

        # Extract requirements freeze from built container and attach to Allure
        logger.info("Extracting requirements freeze from benchmark container using uv pip freeze...")
        requirements_freeze = extract_benchmark_requirements_freeze(docker_client, benchmark_image_tag)

        if requirements_freeze:
            # Attach requirements freeze to Allure report as file attachment
            allure.attach(
                name="requirements-lock-benchmark.txt",
                body=requirements_freeze,
                attachment_type=allure.attachment_type.TEXT,
            )

            # Store in preparation results for metadata
            freeze_lines = [
                line for line in requirements_freeze.split("\n") if line.strip() and ("==" in line or "@" in line)
            ]
            logger.info(
                f"Successfully extracted requirements freeze with {len(freeze_lines)} packages using uv pip freeze"
            )
        else:
            logger.warning("Failed to extract requirements freeze from benchmark container")

        logger.info(f"Benchmark Docker image built successfully: {benchmark_image_tag}")

    except Exception as e:
        logger.error(f"Failed to build benchmark Docker image: {e}")
        raise

    # Download dataset if not already present
    try:
        if not os.path.exists(dataset_path):
            logger.info(f"Downloading dataset from {dataset_source}: {dataset_repo}/{dataset_filename}")
            os.makedirs(thirdparty_dir, exist_ok=True)

            from esq.utils.downloads import download_dataset_file

            # Use the source based on decision made at test level
            # source="modelscope" prevents fallback to HuggingFace (for restricted networks)
            # source=None allows automatic fallback if primary source fails
            download_source = dataset_source if dataset_source == "modelscope" else None

            download_dataset_file(
                dataset_id=dataset_repo,
                filename=dataset_filename,
                target_path=dataset_path,
                timeout=download_timeout,
                source=download_source,
            )

            logger.info(f"Dataset downloaded to {dataset_path}")
        else:
            logger.info(f"Dataset already exists at {dataset_path}")

    except TimeoutError as e:
        raise RuntimeError(f"Dataset download timed out after {download_timeout}s.") from e
    except Exception as e:
        raise RuntimeError(f"Dataset download failed: {e}") from e

    logger.info("Asset preparation completed successfully")

    # Return Result object
    result = Result(
        parameters={},
        metrics={},
        metadata={
            "status": True,
            "ovms_container": preparation_results.get("ovms_container", {}),
            "benchmark_container": preparation_results.get("benchmark_container", {}),
            "base_image_digests": preparation_results.get("base_image_digests", {}),
        },
    )

    return result


def prepare_model_config(model_id: str, model_precision: str, device_id: str, models_dir: str) -> Dict[str, Any]:
    """
    Prepare model configuration for OVMS.

    Args:
        model_id: Model identifier
        model_precision: Model precision (int4, fp16, etc.)
        device_id: Target device ID
        models_dir: Models directory

    Returns:
        Dict with model configuration
    """
    logger.info(f"Preparing model config for {model_id} on {device_id}")

    model_path = os.path.join(models_dir, model_id)

    # Create model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Model configuration for OVMS
    model_config = {
        "model_config_list": [
            {
                "config": {
                    "name": "text_generation",
                    "base_path": f"/mnt/models/{model_id}",
                    "target_device": device_id.upper(),
                    "nireq": 1,
                    "plugin_config": {
                        "PERFORMANCE_HINT": "THROUGHPUT",
                        "INFERENCE_PRECISION_HINT": model_precision.upper(),
                    },
                }
            }
        ]
    }

    return {
        "model_config": model_config,
        "model_path": model_path,
        "model_id": model_id,
        "precision": model_precision,
        "device": device_id,
    }
