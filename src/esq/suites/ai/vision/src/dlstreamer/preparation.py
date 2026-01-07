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
from .utils import create_error_result

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
    Prepare all necessary assets (videos, models, files) for the DL Streamer test.

    All assets are defined in the 'assets' configuration parameter with a 'type' field
    that specifies whether they are 'video', 'model', or 'file' (default) assets.

    Args:
        configs: Test configuration dictionary containing 'assets' list
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

    # Prepare all assets (supporting typed assets: file, video, model)
    # Assets can be one of three types:
    # 1. 'file' (default) - Generic files like labels, configs, etc.
    #    Required: id, url, sha256, path
    # 2. 'video' - Video files with optional conversion/trimming
    #    Required: id, name, url, sha256
    #    Optional: fps, duration, width, height, codec (h264 or h265)
    # 3. 'model' - Model files with conversion options
    #    Required: id, source (ultralytics/zip/files), precision, format
    #    Optional: url, sha256 (for zip/files sources)
    assets_config = []
    video_assets = []
    model_assets = []
    file_assets = []

    for asset in configs.get("assets", []):
        asset_type = asset.get("type", "file").lower()
        asset_id = asset.get("id", asset.get("name", ""))

        logger.info(
            f"Preparing asset {len(assets_config) + 1}/{len(configs.get('assets', []))}: "
            f"{asset_id} (type: {asset_type})"
        )

        if asset_type == "video":
            # Handle video-type asset with conversion/trimming options
            video_config = {
                "video_url": asset.get("url", ""),
                "video_name": asset.get("name", os.path.basename(urlparse(asset.get("url", "")).path)),
                "video_sha256": asset.get("sha256", ""),
            }

            # Add optional video processing parameters
            if "fps" in asset:
                video_config["fps"] = asset["fps"]
            if "duration" in asset:
                video_config["duration"] = asset["duration"]
            if "width" in asset:
                video_config["width"] = asset["width"]
            if "height" in asset:
                video_config["height"] = asset["height"]

            video_assets.append(asset)
            assets_config.append({"id": asset_id, "type": "video", "config": video_config})

            # Process video asset (download, convert, trim)
            video_url = video_config["video_url"]
            video_name = video_config["video_name"]
            video_sha256 = video_config["video_sha256"]
            video_path = os.path.join(videos_dir, video_name)

            if not os.path.exists(video_path):
                # Download video
                if not video_url:
                    pytest.fail(f"Video asset URL is not provided for {video_name}")
                download_file(url=video_url, target_path=video_path, sha256sum=video_sha256)
                if not os.path.exists(video_path):
                    pytest.fail(f"Failed to download video asset: {video_name}")

                # Convert video if required (only if conversion parameters are specified)
                convert_params = ["fps", "width", "height", "codec"]
                specified_params = {key: asset[key] for key in convert_params if key in asset}

                if specified_params:
                    # Build command with only specified parameters
                    command_parts = ["python3 main.py", f"--video-name {video_name}", "convert"]

                    if "width" in specified_params:
                        command_parts.append(f"--width {specified_params['width']}")
                    if "height" in specified_params:
                        command_parts.append(f"--height {specified_params['height']}")
                    if "fps" in specified_params:
                        command_parts.append(f"--fps {specified_params['fps']}")
                    if "codec" in specified_params:
                        command_parts.append(f"--codec {specified_params['codec']}")

                    # Check global keep_original_videos flag from configs
                    keep_original = configs.get("keep_original_videos", False)
                    if keep_original:
                        command_parts.append("--keep-original")

                    command = " ".join(command_parts)

                    # Build descriptive log message
                    conversion_desc = []
                    if "width" in specified_params or "height" in specified_params:
                        w = specified_params.get("width", "auto")
                        h = specified_params.get("height", "auto")
                        conversion_desc.append(f"resolution: {w}x{h}")
                    if "fps" in specified_params:
                        conversion_desc.append(f"FPS: {specified_params['fps']}")
                    if "codec" in specified_params:
                        conversion_desc.append(f"codec: {specified_params['codec'].upper()}")

                    logger.info(f"Converting video asset {video_name} ({', '.join(conversion_desc)})")
                    run_video_utils_container(
                        docker_client=docker_client,
                        docker_image_tag=docker_image_tag_utils,
                        command=command,
                        container_name=f"{docker_container_prefix}-utils-video-convert",
                        data_dir=data_dir,
                        container_mnt_dir=container_mnt_dir,
                    )

                    converted_sha256 = hashlib.sha256(open(video_path, "rb").read()).hexdigest()
                    if converted_sha256 != video_sha256:
                        logger.debug(f"Video asset {video_name} was converted correctly")
                    else:
                        pytest.fail(f"Video asset {video_name} was not converted correctly")

                # Trim or extend video if required
                if "duration" in asset:
                    # Check global keep_original_videos flag from configs
                    keep_original = configs.get("keep_original_videos", False)
                    keep_flag = " --keep-original" if keep_original else ""

                    command = f"python3 main.py --video-name {video_name} trim \
                        --duration {asset['duration']}{keep_flag}"

                    logger.info(f"Trimming video asset {video_name} to target duration")
                    run_video_utils_container(
                        docker_client=docker_client,
                        docker_image_tag=docker_image_tag_utils,
                        command=command,
                        container_name=f"{docker_container_prefix}-utils-video-trim",
                        data_dir=data_dir,
                        container_mnt_dir=container_mnt_dir,
                    )

                    converted_sha256 = hashlib.sha256(open(video_path, "rb").read()).hexdigest()
                    if converted_sha256 != video_sha256:
                        logger.debug(f"Video asset {video_name} was trimmed correctly")
                    else:
                        pytest.fail(f"Video asset {video_name} was not trimmed correctly")

                # Extend video by looping if required
                if "loop" in asset:
                    # Check global keep_original_videos flag from configs
                    keep_original = configs.get("keep_original_videos", False)
                    keep_flag = " --keep-original" if keep_original else ""

                    command = f"python3 main.py --video-name {video_name} extend \
                        --target-duration {asset['loop']}{keep_flag}"

                    logger.info(f"Extending video asset {video_name} to {asset['loop']} seconds by looping")
                    run_video_utils_container(
                        docker_client=docker_client,
                        docker_image_tag=docker_image_tag_utils,
                        command=command,
                        container_name=f"{docker_container_prefix}-utils-video-extend",
                        data_dir=data_dir,
                        container_mnt_dir=container_mnt_dir,
                    )

                    converted_sha256 = hashlib.sha256(open(video_path, "rb").read()).hexdigest()
                    if converted_sha256 != video_sha256:
                        logger.debug(f"Video asset {video_name} was extended correctly")
                    else:
                        pytest.fail(f"Video asset {video_name} was not extended correctly")

                logger.info(f" ✓ Video asset {video_name} prepared successfully")
            else:
                logger.info(f" ✓ Video asset {video_name} already exists")

        elif asset_type == "model":
            # Handle model-type asset with conversion options
            model_config = {
                "id": asset_id,
                "source": asset.get("source", ""),
                "precision": asset.get("precision", "fp16"),
                "format": asset.get("format", "openvino"),
                "url": asset.get("url", ""),
                "sha256": asset.get("sha256", ""),
                "export_args": asset.get("export_args", None),
                # KaggleHub-specific parameters
                "kaggle_handle": asset.get("kaggle_handle", None),
                "convert_args": asset.get("convert_args", None),
                "quantize_args": asset.get("quantize_args", None),
            }

            model_assets.append(model_config)
            assets_config.append({"id": asset_id, "type": "model", "config": model_config})
            logger.info(f" → Model asset {asset_id} queued for batch preparation")

        else:
            # Handle generic file-type asset (default)
            config = {
                "id": asset_id,
                "url": asset.get("url", ""),
                "sha256": asset.get("sha256", ""),
                "path": asset.get("path", ""),
            }
            file_assets.append(config)
            assets_config.append({"id": asset_id, "type": "file", "config": config})

            asset_url = config["url"]
            asset_path_relative = config["path"]
            asset_sha256 = config["sha256"]

            if not asset_url or not asset_path_relative:
                logger.warning(f"Skipping file asset {asset_id}: missing URL or path")
                continue

            # Convert relative path to absolute path within data directory
            if asset_path_relative.startswith("./"):
                asset_path_relative = asset_path_relative[2:]  # Remove './' prefix
            asset_path = os.path.join(data_dir, asset_path_relative)

            # Ensure the parent directory exists
            asset_dir = os.path.dirname(asset_path)
            ensure_dir_permissions(asset_dir, uid=os.getuid(), gid=os.getgid())

            if not os.path.exists(asset_path):
                # Download asset
                logger.info(f"Downloading file asset {asset_id} from {asset_url}")
                download_file(url=asset_url, target_path=asset_path, sha256sum=asset_sha256)
                if not os.path.exists(asset_path):
                    pytest.fail(f"Failed to download file asset: {asset_id}")
                else:
                    logger.info(f" ✓ File asset {asset_id} downloaded successfully")
            else:
                logger.info(f" ✓ File asset {asset_id} already exists")

    # Batch process model assets using prepare_models_batch
    if model_assets:
        from esq.utils.models import prepare_models_batch

        logger.info(f"Batch processing {len(model_assets)} model assets")
        batch_result = prepare_models_batch(model_assets, models_dir)

        if not batch_result["success"]:
            failed_models = ", ".join(batch_result["failed_models"])
            error_message = f"Failed to prepare model assets: {failed_models}"
            logger.error(error_message)
            pytest.fail(error_message)
        else:
            logger.info(f" ✓ All {len(model_assets)} model assets prepared successfully")

    if assets_config:
        logger.info(
            f" ✓ All {len(assets_config)} assets prepared successfully "
            f"({len(video_assets)} videos, {len(model_assets)} models, {len(file_assets)} files)"
        )

    result = Result(
        metadata={
            "status": True,
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
        logger.warning(error_message)
        from .utils import cleanup_stale_containers, cleanup_thread_pool

        cleanup_stale_containers(docker_client, docker_container_prefix)
        cleanup_thread_pool()

        create_error_result(device_id, error_message)
        raise KeyboardInterrupt(error_message)

    except Exception as e:
        error_message = f"An error occurred during baseline streams analysis for device {device_id}: {e}"
        logger.error(error_message)
        from .utils import cleanup_stale_containers, cleanup_thread_pool

        cleanup_stale_containers(docker_client, docker_container_prefix)
        cleanup_thread_pool()
        return create_error_result(device_id, error_message)

    if not baseline_streams:
        error_message = (
            f"Baseline streams analysis for device {device_id} did not return valid results. "
            "Please check the pipeline configuration."
        )
        logger.error(error_message)
        from .utils import cleanup_stale_containers, cleanup_thread_pool

        cleanup_stale_containers(docker_client, docker_container_prefix)
        cleanup_thread_pool()
        return create_error_result(device_id, error_message)

    device_info = baseline_streams.get(device_id, {})
    num_streams = device_info.get("num_streams", -1)
    fps = device_info.get("per_stream_fps", 0.0)
    baseline_passed = fps >= target_fps

    if num_streams == -1:
        error_message = (
            f"Baseline streams analysis for {device_id} did not return valid results. "
            f"num_streams={num_streams}, per_stream_fps={fps}. "
            "Please check the pipeline configuration."
        )
        logger.error(error_message)
        return create_error_result(device_id, error_message, num_streams, fps)

    logger.info(f"Baseline streams analysis completed for {device_id}: {num_streams} streams @ {fps:.2f} FPS")

    return Result(
        metadata={
            "status": True,
            "device_id": device_id,
            "per_stream_fps": fps,
            "num_streams": num_streams,
            "pass": baseline_passed,
            "baseline_streams": baseline_streams,
        }
    )


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

        resolved_pipeline = resolve_pipeline_placeholders(pipeline, pipeline_params, device_id, device_dict)
        baseline_pipeline, result_pipeline = build_baseline_pipeline(
            pipeline=resolved_pipeline,
        )

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
        # Check for valid baseline results
        # Note: num_streams can be -1 for errors, 0 for ran-but-failed, or positive for success
        if not device_info.get("num_streams", 0) > 0 and not device_info.get("per_stream_fps", 0) > 0:
            error_message = (
                f"Baseline streams analysis for {device_id} did not return valid results. "
                f"num_streams={device_info.get('num_streams', 'N/A')}, "
                f"per_stream_fps={device_info.get('per_stream_fps', 'N/A')}. "
                "Please check the pipeline configuration."
            )
            logger.error(error_message)
            from .utils import cleanup_stale_containers, cleanup_thread_pool

            cleanup_stale_containers(docker_client, docker_container_prefix)
            cleanup_thread_pool()

            return {device_id: {"num_streams": -1, "per_stream_fps": 0.0, "error": error_message, "status": "error"}}

        return baseline_streams

    except Exception as e:
        logger.error(f"Exception in prepare_estimate_num_streams_for_device for {device_id}: {e}")
        from .utils import cleanup_stale_containers, cleanup_thread_pool

        cleanup_stale_containers(docker_client, docker_container_prefix)
        cleanup_thread_pool()

        return {device_id: {"num_streams": -1, "per_stream_fps": 0.0, "error": str(e), "status": "error"}}
