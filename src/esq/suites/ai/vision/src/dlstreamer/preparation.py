# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer preparation functions for assets and baseline analysis."""

import grp
import hashlib
import logging
import os
from typing import Any, Dict
from urllib.parse import urlparse

import pytest
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Result
from sysagent.utils.infrastructure import DockerClient, download_file

from .container import run_dlstreamer_analyzer_container, run_video_utils_container
from .pipeline import build_baseline_pipeline, resolve_pipeline_placeholders

logger = logging.getLogger(__name__)


def get_device_specific_docker_image(
    device_id: str, container_config: dict, fallback_image: str, device_dict: dict = None
) -> str:
    """
    Get the appropriate Docker image for a specific device.

    Args:
        device_id: The device ID to check (e.g., 'dgpu', 'GPU.1', 'CPU')
        container_config: Container configuration with available images
        fallback_image: Default image to use if device-specific image not available
        device_dict: Dictionary containing device information (optional)

    Returns:
        Docker image tag appropriate for the device
    """
    # Check if the device is a dGPU device
    is_dgpu_device = False

    if device_dict and device_id in device_dict:
        # Check device type from device_dict for discrete GPUs
        device_type = device_dict[device_id].get("device_type", "")
        is_dgpu_device = "discrete" in device_type.lower()
    else:
        # Fallback to device_id pattern matching
        is_dgpu_device = device_id == "dgpu" or device_id.lower().startswith("dgpu") or "dgpu" in device_id.lower()

    if is_dgpu_device and "dgpu_analyzer_image" in container_config:
        image_name = container_config["dgpu_analyzer_image"]
        logger.debug(f"Using dGPU-specific Docker image for device {device_id}: {image_name}")
        return image_name
    elif "analyzer_image" in container_config:
        image_name = container_config["analyzer_image"]
        logger.debug(f"Using standard Docker image for device {device_id}: {image_name}")
        return image_name
    else:
        logger.warning(f"No device-specific image found for {device_id}, using fallback: {fallback_image}")
        return fallback_image


def prepare_assets(
    videos: list,
    configs: Dict[str, Any],
    models_dir: str,
    videos_dir: str,
    src_dir: str,
    docker_client: DockerClient,
    docker_image_tag_analyzer: str,
    docker_image_tag_utils: str,
    docker_container_prefix: str,
) -> Result:
    """
    Prepare all necessary assets (videos, models, etc.) for the DL Streamer test.

    Args:
        videos: List of video configurations
        configs: Test configuration dictionary
        models_dir: Directory for model files
        videos_dir: Directory for video files
        src_dir: Source directory
        docker_client: Docker client instance
        docker_image_tag_analyzer: Docker image tag for analyzer
        docker_image_tag_utils: Docker image tag for utilities
        docker_container_prefix: Prefix for container names

    Returns:
        Result object with preparation status and metadata
    """
    logger.info("Preparing assets for DL Streamer test...")

    # Derive data directory matching original logic
    core_data_dir = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "app_data"))
    data_dir = os.path.join(core_data_dir, "data", "suites", "ai", "vision")
    container_mnt_dir = "/mnt"  # Standard container mount directory

    # Prepare directories
    ensure_dir_permissions(models_dir, uid=os.getuid(), gid=os.getgid())
    ensure_dir_permissions(videos_dir, uid=os.getuid(), gid=os.getgid())

    # Prepare Docker images for DL Streamer analyzer
    container_config = {}

    # Build standard Docker image
    logger.info(f"Building standard Docker image: {docker_image_tag_analyzer}")
    build_analyzer_result = docker_client.build_image(
        path=f"{src_dir}/containers/dlstreamer_analyzer", tag=docker_image_tag_analyzer, extract_packages=True
    )
    if not build_analyzer_result.get("image_id"):
        pytest.fail(f"Failed to build Docker image: {docker_image_tag_analyzer}")
    else:
        logger.info(f"Standard Docker image built successfully: {docker_image_tag_analyzer}")

    container_config["analyzer_image"] = docker_image_tag_analyzer
    container_config["analyzer_image_id"] = build_analyzer_result.get("image_id", "")

    # Build dGPU-specific Docker image
    dgpu_image_tag = f"{docker_image_tag_analyzer.split(':')[0]}-dgpu:latest"
    logger.info(f"Building dGPU-specific Docker image with enhanced system packages: {dgpu_image_tag}")
    build_dgpu_result = docker_client.build_image(
        path=f"{src_dir}/containers/dlstreamer_analyzer",
        tag=dgpu_image_tag,
        dockerfile="Dockerfile.dgpu",
        extract_packages=True,
    )
    if not build_dgpu_result.get("image_id"):
        pytest.fail(f"Failed to build dGPU Docker image: {dgpu_image_tag}")
    else:
        logger.info(f"dGPU Docker image built successfully: {dgpu_image_tag}")

    container_config["dgpu_analyzer_image"] = dgpu_image_tag
    container_config["dgpu_analyzer_image_id"] = build_dgpu_result.get("image_id", "")

    # Prepare Docker image for DL Streamer utils
    build_utils_result = docker_client.build_image(
        path=f"{src_dir}/containers/dlstreamer_utils", tag=docker_image_tag_utils, extract_packages=True
    )
    if not build_utils_result.get("image_id"):
        pytest.fail(f"Failed to build Docker image: {docker_image_tag_utils}")
    else:
        logger.info(f"Docker image built successfully: {docker_image_tag_utils}")

    container_config.update(
        {
            "utils_image": docker_image_tag_utils,
            "utils_image_id": build_utils_result.get("image_id", ""),
        }
    )

    # Prepare all videos
    videos_config = []
    for video in videos:
        config = {
            "video_url": video.get("url", ""),
            "video_name": video.get("name", os.path.basename(urlparse(video.get("url", "")).path)),
            "video_sha256": video.get("sha256", ""),
        }
        videos_config.append(config)

        video_url = config["video_url"]
        video_name = config["video_name"]
        video_sha256 = config["video_sha256"]
        video_path = os.path.join(videos_dir, video_name)
        logger.info(f"Preparing video {len(videos_config)}/{len(videos)}: {video_name}")
        if not os.path.exists(video_path):
            # Download video
            if not video_url:
                pytest.fail(f"Test video URL is not provided for {video_name}")
            download_file(url=video_url, target_path=video_path, sha256sum=video_sha256)
            if not os.path.exists(video_path):
                pytest.fail(f"Failed to download test video: {config['video_name']}")

            # Convert video
            convert_required = ["fps", "width", "height"]
            if any(key in video for key in convert_required):
                for key in convert_required:
                    if key in video:
                        config[key] = video[key]
                    else:
                        config[key] = 1920 if key == "width" else 1080 if key == "height" else 15

                command = f"python3 main.py --video-name {video_name} convert \
                    --width {config['width']} \
                    --height {config['height']} \
                    --fps {config['fps']}"

                logger.info(f"Converting video {video_name} to target resolution and FPS")
                run_video_utils_container(
                    docker_client=docker_client,
                    docker_image_tag=docker_image_tag_utils,
                    command=command,
                    container_name=f"{docker_container_prefix}-utils-video-convert",
                    data_dir=data_dir,
                    container_mnt_dir=container_mnt_dir,
                )

                converted_sha256 = hashlib.sha256(open(video_path, "rb").read()).hexdigest()
                logger.debug(f"Original video SHA256: {video_sha256}")
                logger.debug(f"Converted video SHA256: {converted_sha256}")
                if converted_sha256 != video_sha256:
                    logger.debug(f"Video {config['video_name']} was converted correctly")
                else:
                    pytest.fail(f"Video {config['video_name']} was not converted correctly")

            # Trim video
            trim_required = ["duration"]
            if any(key in video for key in trim_required):
                for key in trim_required:
                    if key in video:
                        config[key] = video[key]
                logger.info(f"Trimming video {video_name} to {config['duration']} seconds")
                command = f"python3 main.py --video-name {video_name} trim \
                    --duration {config['duration']}"

                logger.info(f"Trimming video {video_name} to target duration")
                run_video_utils_container(
                    docker_client=docker_client,
                    docker_image_tag=docker_image_tag_utils,
                    command=command,
                    container_name=f"{docker_container_prefix}-utils-video-trim",
                    data_dir=data_dir,
                    container_mnt_dir=container_mnt_dir,
                )

                converted_sha256 = hashlib.sha256(open(video_path, "rb").read()).hexdigest()
                logger.debug(f"Original video SHA256: {video_sha256}")
                logger.debug(f"Converted video SHA256: {converted_sha256}")
                if converted_sha256 != video_sha256:
                    logger.debug(f"Video {config['video_name']} was trimmed correctly")
                else:
                    pytest.fail(f"Video {config['video_name']} was not trimmed correctly")

        else:
            logger.info(f" ✓ Video {config['video_name']} already exists")

    # Prepare all models
    models_config = []
    for model in configs.get("models", []):
        config = {
            "id": model.get("id", ""),
            "source": model.get("source", ""),
            "precision": model.get("precision", ""),
            "format": model.get("format", ""),
            "url": model.get("url", ""),
            "sha256": model.get("sha256", ""),
        }
        models_config.append(config)

    # Use the generalized batch model preparation function
    from esq.utils.models.setup_model import prepare_models_batch

    batch_result = prepare_models_batch(models_config, models_dir)

    if not batch_result["success"]:
        failed_models = ", ".join(batch_result["failed_models"])
        error_message = f"Failed to prepare models: {failed_models}"
        logger.error(error_message)
        pytest.fail(error_message)
    else:
        logger.info(f" ✓ All {len(models_config)} models prepared successfully")

    # Prepare all generic assets (labels, configs, etc.)
    assets_config = []
    for asset in configs.get("assets", []):
        config = {
            "id": asset.get("id", asset.get("name", "")),
            "url": asset.get("url", ""),
            "sha256": asset.get("sha256", ""),
            "path": asset.get("path", ""),
        }
        assets_config.append(config)

        asset_url = config["url"]
        asset_path_relative = config["path"]
        asset_sha256 = config["sha256"]
        asset_id = config["id"]

        if not asset_url or not asset_path_relative:
            logger.warning(f"Skipping asset {asset_id}: missing URL or path")
            continue

        # Convert relative path to absolute path within data directory
        if asset_path_relative.startswith("./"):
            asset_path_relative = asset_path_relative[2:]  # Remove './' prefix
        asset_path = os.path.join(data_dir, asset_path_relative)

        # Ensure the parent directory exists
        asset_dir = os.path.dirname(asset_path)
        ensure_dir_permissions(asset_dir, uid=os.getuid(), gid=os.getgid())

        logger.info(f"Preparing asset {len(assets_config)}/{len(configs.get('assets', []))}: {asset_id}")

        if not os.path.exists(asset_path):
            # Download asset
            logger.info(f"Downloading asset {asset_id} from {asset_url}")
            download_file(url=asset_url, target_path=asset_path, sha256sum=asset_sha256)
            if not os.path.exists(asset_path):
                pytest.fail(f"Failed to download asset: {asset_id}")
            else:
                logger.info(f" ✓ Asset {asset_id} downloaded successfully")
        else:
            logger.info(f" ✓ Asset {asset_id} already exists")

    if assets_config:
        logger.info(f" ✓ All {len(assets_config)} assets prepared successfully")

    result = Result(
        metadata={
            "status": True,
            "video_config": videos_config,
            "models_config": models_config,
            "assets_config": assets_config,
            "container_config": container_config,
        }
    )

    return result


def prepare_baseline(
    device_id: str,
    pipeline: str,
    target_fps: float,
    pipeline_params: Dict[str, Any],
    device_dict: Dict[str, Any],
    docker_client: DockerClient,
    docker_image_tag_analyzer: str,
    data_dir: str,
    container_mnt_dir: str,
    results_dir: str,
    docker_container_prefix: str,
    pipeline_timeout: int = 180,
    container_config: Dict[str, Any] = None,
) -> Result:
    """
    Prepare baseline streams analysis for a specific device.

    Args:
        device_id: Device ID to analyze
        pipeline: Pipeline configuration string
        target_fps: Target FPS for the test
        pipeline_params: Pipeline parameters dictionary
        device_dict: Dictionary containing device information
        docker_client: Docker client instance
        docker_image_tag_analyzer: Docker image tag for analyzer
        data_dir: Base data directory
        container_mnt_dir: Container mount directory
        results_dir: Directory for results
        docker_container_prefix: Prefix for container names
        pipeline_timeout: Timeout for pipeline execution
        container_config: Container configuration with available images (optional)

    Returns:
        Result object with baseline analysis results
    """
    logger.info(f"Preparing baseline streams analysis for device: {device_id}")

    if not pipeline:
        pytest.fail("Pipeline is not provided for baseline streams analysis")

    try:
        # Select device-specific Docker image
        if container_config:
            docker_image_tag_analyzer = get_device_specific_docker_image(
                device_id, container_config, docker_image_tag_analyzer, device_dict
            )

        baseline_streams = prepare_estimate_num_streams_for_device(
            device_id=device_id,
            pipeline=pipeline,
            target_fps=target_fps,
            pipeline_params=pipeline_params,
            device_dict=device_dict,
            docker_client=docker_client,
            docker_image_tag_analyzer=docker_image_tag_analyzer,
            data_dir=data_dir,
            container_mnt_dir=container_mnt_dir,
            results_dir=results_dir,
            docker_container_prefix=docker_container_prefix,
            pipeline_timeout=pipeline_timeout,
        )
    except KeyboardInterrupt:
        error_message = f"Interrupt detected during baseline streams analysis for device {device_id}"
        logger.warning(f"{error_message}")
        from .utils import cleanup_stale_containers, cleanup_thread_pool

        cleanup_stale_containers(docker_client, docker_container_prefix)
        cleanup_thread_pool()
        raise KeyboardInterrupt(error_message)
    except Exception as e:
        error_message = f"An error occurred during baseline streams analysis for device {device_id}: {e}"
        logger.error(error_message)
        from .utils import cleanup_stale_containers, cleanup_thread_pool

        cleanup_stale_containers(docker_client, docker_container_prefix)
        cleanup_thread_pool()
        pytest.fail(error_message)

    if not baseline_streams:
        error_message = (
            f"Baseline streams analysis for device {device_id} did not return valid results. "
            "Please check the pipeline configuration."
        )
        logger.error(error_message)
        from .utils import cleanup_stale_containers, cleanup_thread_pool

        cleanup_stale_containers(docker_client, docker_container_prefix)
        cleanup_thread_pool()
        pytest.fail(error_message)

    device_info = baseline_streams.get(device_id, {})
    num_streams = device_info.get("num_streams", 1)
    fps = device_info.get("per_stream_fps", 0.0)
    baseline_passed = fps >= target_fps

    logger.info(f"Baseline streams analysis completed for {device_id}: {num_streams} streams @ {fps:.2f} FPS")

    result = Result(
        metadata={
            "status": True,
            "device_id": device_id,
            "per_stream_fps": fps,
            "num_streams": num_streams,
            "pass": baseline_passed,
            "baseline_streams": baseline_streams,
        }
    )

    return result


def prepare_estimate_num_streams_for_device(
    device_id: str,
    pipeline: str,
    target_fps: float,
    pipeline_params: Dict[str, Any],
    device_dict: Dict[str, Any],
    docker_client: DockerClient,
    docker_image_tag_analyzer: str,
    data_dir: str,
    container_mnt_dir: str,
    results_dir: str,
    docker_container_prefix: str,
    pipeline_timeout: int = 180,
) -> Dict[str, Any]:
    """
    Estimate the number of streams a device can handle using baseline analysis.

    Args:
        device_id: Device ID to analyze
        pipeline: Pipeline configuration string
        target_fps: Target FPS for the test
        pipeline_params: Pipeline parameters dictionary
        device_dict: Dictionary containing device information
        docker_client: Docker client instance
        docker_image_tag_analyzer: Docker image tag for analyzer
        data_dir: Base data directory
        container_mnt_dir: Container mount directory
        results_dir: Directory for results
        docker_container_prefix: Prefix for container names
        pipeline_timeout: Timeout for pipeline execution

    Returns:
        Dictionary with baseline streams analysis results
    """
    try:
        logger.debug(f"Using pipeline: {pipeline}")

        # Use modular resolve_pipeline_placeholders from the imported module
        resolved_pipeline = resolve_pipeline_placeholders(pipeline, pipeline_params, device_id, device_dict)

        # Use modular build_baseline_pipeline from the imported module
        baseline_pipeline, result_pipeline = build_baseline_pipeline(pipeline=resolved_pipeline, sync_model=False)

        command = [
            "baseline",
            "--target-device",
            device_id,
            "--target-fps",
            str(target_fps),
            "--baseline-pipeline",
            baseline_pipeline,
            "--result-pipeline",
            result_pipeline,
            "--pipeline-timeout",
            str(pipeline_timeout),
        ]

        result = run_dlstreamer_analyzer_container(
            docker_client=docker_client,
            docker_image_tag=docker_image_tag_analyzer,
            command=command,
            container_name=f"{docker_container_prefix}-analyzer-baseline-{device_id}",
            data_dir=data_dir,
            container_mnt_dir=container_mnt_dir,
            render_gid=grp.getgrnam("render").gr_gid,
            user_gid=os.getuid(),
            mode="batch",
            result_file=f"baseline_streams_result_{str(device_id).replace('.', '_').lower()}.json",
            container_result_file_dir=f"{container_mnt_dir}/results",
        )

        # Extract and process results
        if result["result_json"]:
            baseline_streams = result["result_json"]
        else:
            raise RuntimeError("Could not extract test results from container.")

        device_info = baseline_streams.get(device_id, {})
        if not device_info.get("num_streams", 0) > 0 and not device_info.get("per_stream_fps", 0) > 0:
            error_message = (
                f"Baseline streams analysis for {device_id} did not return valid results. "
                "Please check the pipeline configuration."
            )
            logger.error(error_message)
            from .utils import cleanup_stale_containers, cleanup_thread_pool

            cleanup_stale_containers(docker_client, docker_container_prefix)
            cleanup_thread_pool()
            pytest.fail(error_message)

        return baseline_streams

    except Exception as e:
        logger.error(f"Exception in prepare_estimate_num_streams_for_device for {device_id}: {e}")
        from .utils import cleanup_stale_containers, cleanup_thread_pool

        cleanup_stale_containers(docker_client, docker_container_prefix)
        cleanup_thread_pool()
        pytest.fail(str(e))
