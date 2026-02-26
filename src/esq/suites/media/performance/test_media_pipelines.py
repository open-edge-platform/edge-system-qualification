# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import grp
import json
import logging
import os
from pathlib import Path

import allure
import pandas as pd
import pytest
from esq.utils.genutils import plot_grouped_bar_chart
from esq.utils.media import (
    detect_display_settings,
    get_x11_volumes,
    get_x11_environment,
    determine_display_output,
)
from esq.utils.media.validation import detect_platform_type
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result
from sysagent.utils.infrastructure import DockerClient
from sysagent.utils.system.ov_helper import get_available_devices_by_category

logger = logging.getLogger(__name__)

test_container_path = "src/containers/media_benchmark/"


def _normalize_device_categories(device_value, default: str = "igpu") -> list[str]:
    if isinstance(device_value, str):
        categories = [device_value]
    elif isinstance(device_value, (list, tuple, set)):
        categories = [str(category) for category in device_value]
    elif device_value is None:
        categories = [default]
    else:
        categories = [str(device_value)]

    normalized = [category.strip().lower() for category in categories if str(category).strip()]
    return normalized if normalized else [default]


def _resolve_esq_build_context(test_file_dir: Path, relative_dockerfile: str) -> Path:
    candidate_paths = [
        test_file_dir.parents[2],
        Path.cwd() / "src" / "esq",
        Path.cwd() / "esq",
    ]

    for candidate in candidate_paths:
        dockerfile_path = candidate / relative_dockerfile
        if candidate.is_dir() and dockerfile_path.is_file():
            return candidate

    raise FileNotFoundError(
        f"Unable to resolve Docker build context for '{relative_dockerfile}'. "
        f"Checked: {[str(path) for path in candidate_paths]}"
    )


def _create_media_metrics(value: str = "N/A", unit: str = None) -> dict:
    """
    Create media performance metrics dictionary.

    Args:
        value: Initial value for all metrics (default: "N/A")
        unit: Unit for metrics (default: None for N/A values)

    Returns:
        Dictionary of Metrics objects for media performance
    """
    return {
        "max_streams": Metrics(unit=unit, value=value, is_key_metric=True),
        "avg_fps": Metrics(unit=unit, value=value, is_key_metric=False),
        "gpu_freq_mhz": Metrics(unit=unit, value=value, is_key_metric=False),
        "pkg_power_w": Metrics(unit=unit, value=value, is_key_metric=False),
        "duration_s": Metrics(unit=unit, value=value, is_key_metric=False),
    }


def _generate_media_charts(media_results: str, configs: dict, logger, specific_csv: str = None):
    """
    Generate performance charts from aggregated test group CSV files.
    Each aggregated CSV contains results from comparable tests (same codec/operation).
    Charts show comparison across multiple test runs.

    Args:
        media_results: Path to directory containing aggregated CSV result files
        configs: Test configuration dictionary containing metadata
        logger: Logger instance for debug/error messages
        specific_csv: Optional specific CSV filename to process (e.g., "aggregated_h264_decode.csv")
                     If provided, only this CSV will be processed. If None, all CSVs are processed.
    """
    try:
        # Find aggregated CSV files to process
        if specific_csv:
            # Process only the specified CSV file
            specific_csv_path = Path(media_results) / specific_csv
            aggregated_csvs = [specific_csv_path] if specific_csv_path.exists() else []
            if not aggregated_csvs:
                logger.debug(f"Specific CSV not found: {specific_csv}")
                return
        else:
            # Find all aggregated CSV files: aggregated_{codec}_{operation}.csv
            aggregated_csvs = list(Path(media_results).glob("aggregated_*.csv"))

        if not aggregated_csvs:
            logger.debug("No aggregated CSV files found for chart generation")
            return

        # Extract reference values from CSV
        # Will be populated for each aggregated CSV based on actual test data
        reference_max_channels = None
        reference_platform = "Reference Platform"

        # Generate a chart for each aggregated CSV file
        for csv_path in aggregated_csvs:
            try:
                # Parse filename: aggregated_{codec}_{operation}.csv
                # Operation may contain spaces: "decode + compose" -> "decode + compose"
                filename = csv_path.stem  # e.g., "aggregated_h264_decode + compose"
                parts = filename.split("_")

                if len(parts) < 3:
                    logger.warning(f"Invalid aggregated CSV filename format: {csv_path.name}")
                    continue

                codec = parts[1]  # h264, h265
                # Rejoin remaining parts for operation (handles "decode_+_compose")
                operation = "_".join(parts[2:])  # encode, decode, decode_+_compose
                # Convert back: decode_+_compose -> decode + compose
                operation = operation.replace("_+_", " + ")

                # Read CSV file
                df = pd.read_csv(csv_path)
                logger.info(f"Processing aggregated CSV: {csv_path.name}, rows={len(df)}, operation='{operation}'")

                if df.empty:
                    logger.warning(f"Empty aggregated CSV: {csv_path.name}")
                    continue

                # Normalize column names (strip whitespace)
                df.columns = [col.strip() for col in df.columns]
                logger.debug(f"CSV columns: {list(df.columns)}")

                # Extract reference values from CSV
                # Look for "Reference Max Channels" and "Reference Platform" columns
                reference_max_channels = None
                reference_platform = "Reference Platform"

                if "Ref Max Channels" in df.columns and not df.empty:
                    # Get reference value from first row
                    ref_value = df.iloc[0]["Ref Max Channels"]
                    if pd.notna(ref_value) and ref_value != -1:
                        try:
                            reference_max_channels = float(ref_value)
                            logger.debug(f"Extracted reference max channels: {reference_max_channels}")
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid reference max channels value: {ref_value}")

                if "Ref Platform" in df.columns and not df.empty:
                    ref_platform = df.iloc[0]["Ref Platform"]
                    if pd.notna(ref_platform) and ref_platform != "N/A":
                        reference_platform = str(ref_platform)
                        logger.debug(f"Extracted reference platform: {reference_platform}")

                # Use actual CSV column names for chart generation (no renaming needed)
                # Check required columns exist
                required_cols = ["Device Used", "Max Channels"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(
                        f"Missing columns in {csv_path.name}: {missing_cols}. Available columns: {list(df.columns)}"
                    )
                    continue

                # Use Resolution or Bitrate for grouping (handle both old and new column names)
                if "Resolution" in df.columns:
                    group_col = "Resolution"
                elif "Input Resolution" in df.columns:
                    group_col = "Input Resolution"
                elif "Bitrate" in df.columns:
                    group_col = "Bitrate"
                elif "Input Bitrate" in df.columns:
                    group_col = "Input Bitrate"
                else:
                    group_col = None

                if not group_col:
                    logger.warning(
                        f"No grouping column (Resolution/Input Resolution/Bitrate/Input Bitrate) found in {csv_path.name}"
                    )
                    continue

                # Create chart output path
                # Normalize operation for filename (replace spaces with underscores)
                operation_filename = operation.replace(" ", "_").replace("+", "plus")
                chart_filename = f"chart_{operation_filename}_{codec}_comparison.png"
                chart_path = csv_path.parent / chart_filename

                logger.info(
                    f"Generating chart for {csv_path.name}: codec={codec}, operation='{operation}', "
                    f"group_by={group_col}, {len(df)} rows"
                )

                # Color map for common resolutions/bitrates
                resolution_colors = {
                    "1080p30": "#1f77b4",  # Blue
                    "1080p": "#1f77b4",  # Blue
                    "4k@30": "#ff7f0e",  # Orange
                    "4k": "#ff7f0e",  # Orange
                    "720p": "#2ca02c",  # Green
                    "2160p": "#d62728",  # Red
                }

                # Generate grouped bar chart using actual CSV column name "Device Used"
                plot_grouped_bar_chart(
                    csv_path=csv_path,
                    output_path=chart_path,
                    x_column="Device Used",
                    y_column="Max Channels",
                    group_column=group_col,
                    title=f"{codec.upper()} {operation.capitalize()} - Performance Comparison ({len(df)} tests)",
                    xlabel="Device",
                    ylabel="Max Channels (Stream Count)",
                    reference_value=reference_max_channels,
                    reference_label=(f"Reference ({reference_platform})" if reference_max_channels else "Reference"),
                    figsize=(12, 6),
                    rotation=0,
                    color_map=resolution_colors,
                )

                # Attach chart to Allure report
                if chart_path.exists():
                    with open(chart_path, "rb") as f:
                        allure.attach(
                            f.read(),
                            name=f"Performance Comparison - {codec.upper()} {operation.capitalize()} ({len(df)} tests)",
                            attachment_type=allure.attachment_type.PNG,
                        )
                    logger.info(f"Generated and attached chart: {chart_filename} with {len(df)} test results")

                    # Clean up chart file after attaching
                    chart_path.unlink()
                else:
                    logger.warning(f"Chart file not found after generation: {chart_path}")

            except Exception as chart_error:
                logger.error(f"Failed to generate chart for {csv_path.name}: {chart_error}", exc_info=True)

    except Exception as chart_error:
        logger.warning(f"Failed to generate media performance charts: {chart_error}", exc_info=True)


def _initialize_media_csv_files(output_dir: str):
    """
    Initialize CSV files with proper headers for media benchmarks.

    The container's update_csv function requires CSV files to exist with headers
    before it can write results (opens with "r+" mode).

    Args:
        output_dir: Output directory where CSV files will be created
    """
    csv_files_to_init = {
        "media_performance_benchmark.csv": (
            "Media Performance Benchmark,Device Used,Codec,Bitrate,Resolution,"
            "Num Monitors,Max Channels,"
            "GPU Freq,Pkg Power,Ref Platform,Ref Max Channels,Ref GPU Freq,Ref Pkg Power,Duration(s),Errors"
        ),
        "media_encode_performance_benchmark.csv": (
            "Media Performance Benchmark,Device Used,Codec,Bitrate,Resolution,"
            "Num Monitors,Max Channels,"
            "GPU Freq,Pkg Power,Ref Platform,Ref Max Channels,Ref GPU Freq,Ref Pkg Power,Duration(s),Errors"
        ),
    }

    for csv_file, header in csv_files_to_init.items():
        csv_path = Path(output_dir) / csv_file
        if not csv_path.exists():
            logger.info(f"Initializing CSV file: {csv_file}")
            csv_path.write_text(f"{header}\n")
            # Ensure proper permissions for container to write (owner/group rw, no world access)
            os.chmod(csv_path, 0o660)
        else:
            logger.debug(f"CSV file already exists: {csv_file}")


def _run_media_container(
    docker_client,
    container_name,
    image_name,
    image_tag,
    device_id,
    media_results,
    operation,
    codec,
    bitrate,
    resolution,
    platform_info,
    configs,
    timeout,
    data_dir,
):
    """
    Run media benchmark container using DockerClient API.

    Args:
        docker_client: DockerClient instance
        container_name: Name for the container
        image_name: Docker image name
        image_tag: Docker image tag
        device_id: Device identifier (e.g., "iGPU", "dGPU.0", "CPU")
        media_results: Path to results directory
        operation: Media operation ("decode", "encode", "compose")
        codec: Video codec ("h264", "h265")
        bitrate: Target bitrate
        resolution: Video resolution ("1080p", "4K")
        platform_info: Platform detection information
        configs: Test configuration dictionary (for reading display_output, etc.)
        timeout: Execution timeout in seconds
        data_dir: Base data directory for videos and other resources

    Returns:
        Container execution result dictionary
    """
    # Prepare volumes for input/output
    videos_path = Path(media_results) / "videos"
    volumes = {
        str(Path(media_results).resolve()): {"bind": "/home/dlstreamer/output", "mode": "rw"},
        str(videos_path.resolve()): {"bind": "/home/dlstreamer/sample_video", "mode": "rw"},
    }

    # Auto-detect display settings using shared utility
    host_display, display_available = detect_display_settings(logger)

    # Determine display_output setting with auto-fallback using shared utility
    config_display_output = configs.get("display_output", None)
    display_output_enabled = determine_display_output(
        config_display_output, display_available, logger
    )

    # Mount X11 volumes for display output (sockets and .Xauthority)
    if display_output_enabled:
        x11_volumes = get_x11_volumes(host_display, logger)
        volumes.update(x11_volumes)

    # Prepare environment variables using shared utility
    environment = get_x11_environment(host_display, display_output_enabled)

    # Prepare container devices for GPU access
    container_devices = []
    if device_id != "CPU":
        # Add GPU devices
        container_devices.extend(
            [
                "/dev/dri:/dev/dri",
            ]
        )

    # Get render group GID
    try:
        render_gid = grp.getgrnam("render").gr_gid
    except KeyError:
        render_gid = 109  # Default render GID
        logger.warning(f"'render' group not found, using default GID: {render_gid}")

    user_gid = os.getgid()

    # Build command arguments for media benchmark script
    # device_id parameter already contains the canonical name (iGPU, dGPU.0, CPU, etc.)
    # from the device mapping done in run_test()
    normalized_device = device_id
    logger.debug(f"Using device for benchmark: {normalized_device}")

    # Normalize codec format (H.264 -> h264, H.265 -> h265)
    codec_normalized = codec.replace(".", "").replace("-", "").lower()  # H.264 -> h264
    logger.debug(f"Normalized codec for container: {codec} → {codec_normalized}")

    # Normalize resolution format (1080p30 -> 1080p, 4k@30 -> 4K)
    # Container expects: ["1080p", "4K"]
    resolution_normalized = resolution.split("@")[0].split("30")[0].split("60")[0]
    if "4k" in resolution_normalized.lower():
        resolution_normalized = "4K"
    else:
        resolution_normalized = "1080p"
    logger.debug(f"Normalized resolution for container: {resolution} → {resolution_normalized}")

    # Normalize bitrate format (4Mbps -> 4000 kbps for container)
    # Container bitrates are in kbps: {"h264": [4000, 16000], "h265": [2000, 8000]}
    # Profile provides "4Mbps", "16Mbps", etc.
    # Extract number and convert to kbps (multiply by 1000)
    import re

    bitrate_match = re.match(r"(\d+)Mbps", bitrate)
    if bitrate_match:
        bitrate_normalized = str(int(bitrate_match.group(1)) * 1000)  # 4Mbps -> 4000 kbps
    else:
        bitrate_normalized = bitrate  # Keep as-is if format unexpected
    logger.debug(f"Normalized bitrate for container: {bitrate} → {bitrate_normalized} kbps")

    # Display output: 0=fakesink (no display), 1=xvimagesink (display enabled)
    # Use the auto-detected display_output_enabled value determined earlier
    display_output = "1" if display_output_enabled else "0"
    logger.debug(f"Display output setting: {display_output} (DISPLAY={host_display})")

    is_mtl = "true" if platform_info.get("is_mtl", False) else "false"
    has_igpu = "true" if platform_info.get("has_igpu", False) else "false"

    # Force entrypoint to bash because some FW custom base images define
    # ENTRYPOINT to a Python CLI (e.g., main.py baseline/total). Without
    # overriding entrypoint, docker treats command below as args to that
    # ENTRYPOINT and fails with: invalid choice: 'bash'.
    entrypoint = "/bin/bash"
    command = [
        "./run_media_benchmark_parallel.sh",
        normalized_device,
        operation,
        codec_normalized,
        bitrate_normalized,
        resolution_normalized,
        display_output,
        is_mtl,
        has_igpu,
    ]

    logger.debug(f"Running container {container_name}")
    logger.debug(f"Command: {' '.join(command)}")
    logger.debug(f"Devices: {container_devices}")
    logger.debug(f"Environment: {environment}")

    try:
        # Use DockerClient.run_container in batch mode
        result = docker_client.run_container(
            name=container_name,
            image=f"{image_name}:{image_tag}",
            entrypoint=entrypoint,
            command=command,
            volumes=volumes,
            devices=container_devices,
            environment=environment,
            user="root:root",  # Run as root for GPU access
            group_add=[render_gid, user_gid],
            privileged=True,  # validation-skip-privileged # Required for GPU access and benchmarking
            network_mode="host",  # Required for inter-process communication
            ipc_mode="host",  # Required for shared memory
            working_dir="/home/dlstreamer",
            mode="batch",
            detach=True,  # Required for batch mode
            remove=True,  # Remove container after completion
            attach_logs=True,  # Attach logs to Allure report
            timeout=timeout,
        )

        exit_code = result.get("container_info", {}).get("exit_code", 1)
        logger.info(f"Container {container_name} completed with exit code: {exit_code}")

        return result

    except Exception as e:
        logger.error(f"Failed to run container {container_name}: {e}")
        import traceback as tb

        allure.attach(
            tb.format_exc(),
            name=f"Container Execution Error - {container_name}",
            attachment_type=allure.attachment_type.TEXT,
        )
        raise


@allure.title("Media Performance via DL Streamer")
def test_media_pipelines(
    request,
    configs,
    cached_result,
    cache_result,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    # Request
    test_name = request.node.name.split("[")[0]

    # Parameters
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)
    force_run = configs.get("force_run", False)  # Control whether to force re-run even if results exist

    # Container execution control - read from profile config (default False to enable full test execution)
    skip_container_execution = configs.get("skip_container_execution", False)

    logger.info(
        f"Starting Media Benchmark Runner: {test_display_name} (force_run={force_run}, skip_container={skip_container_execution})"
    )

    dockerfile_name = configs.get("dockerfile_name", "Dockerfile")
    docker_image_tag = f"{configs.get('container_image', 'openvino_bm_runner')}:{configs.get('image_tag', '1.0')}"
    timeout = configs.get("timeout", 300)
    device = configs.get("device", "igpu")
    device_categories = _normalize_device_categories(device)
    device = device_categories[0]
    operation = configs.get("operation", "decode")
    codec = configs.get("codec", "H.264")
    bitrate = configs.get("bitrate", "4Mbps")

    # Setup
    test_dir = os.path.dirname(os.path.abspath(__file__))
    docker_dir = os.path.join(test_dir, test_container_path)

    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    core_data_dir = "".join(c for c in core_data_dir_tainted)
    data_dir = os.path.join(core_data_dir, "data", "media", "performance")
    media_results = os.path.join(data_dir, "results", "media")
    os.makedirs(media_results, exist_ok=True)

    # Ensure directories have correct permissions
    ensure_dir_permissions(media_results, uid=os.getuid(), gid=os.getgid(), mode=0o775)

    # Detect platform type for container configuration
    platform_info = detect_platform_type()
    logger.debug(f"Platform info: {platform_info}")

    # Step 1: Validate system requirements (including device requirements from config/profile)
    # This will skip the test if requirements are not met (instead of failing)
    validate_system_requirements_from_configs(configs)

    # Step 2: Get available devices based on device categories
    logger.info(f"Configured device categories: {device_categories}")
    device_dict = get_available_devices_by_category(device_categories=device_categories)

    logger.debug(f"Available devices: {device_dict}")
    if not device_dict:
        pytest.fail(f"No available devices found for device categories: {device}")

    # Log detailed device information
    for device_id, device_info in device_dict.items():
        logger.debug(f"Device {device_id}: Type={device_info['device_type']}, Name={device_info['full_name']}")

    docker_client = DockerClient()

    # Cleanup stale containers from previous interrupted runs
    container_prefix = configs.get("container_image", "media_perf_runner")
    logger.info(f"Cleaning up stale containers with prefix: {container_prefix}")
    try:
        docker_client.cleanup_containers_by_name_pattern(container_prefix)
    except Exception as cleanup_error:
        logger.warning(f"Failed to cleanup stale containers: {cleanup_error}")

    # Step 3: Initialize default metrics template (before test execution)
    # Create default metrics with N/A values that will be populated during test execution
    default_metrics = _create_media_metrics(value="N/A", unit="")

    # Initialize results template using from_test_config for automatic metadata application
    results = Result.from_test_config(
        configs=configs,
        parameters={
            "timeout (s)": timeout,
            "display_name": test_display_name,
            "device": device,
        },
        metrics=default_metrics,
    )

    # Initialize variables for finally block (moved to top for broader coverage)
    test_failed = False
    test_interrupted = False
    failure_message = ""
    container_name = None  # Track container name for cleanup in finally block

    try:
        # Step 3: Prepare test environment
        def prepare_assets():
            # Access outer scope variables
            nonlocal device

            # Build 1: Get FW custom device-specific images from dlstreamer preparation
            from esq.suites.ai.vision.src.dlstreamer.preparation import (
                prepare_assets as prepare_dlstreamer_assets,
            )

            test_dir_abs = os.path.dirname(os.path.abspath(__file__))
            # Navigate to ai/vision to access dlstreamer src
            ai_vision_dir = os.path.join(test_dir_abs, "..", "..", "ai", "vision")
            src_dir = os.path.join(ai_vision_dir, "src")

            # Setup models/videos dirs for dlstreamer prep
            core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
            core_data_dir = "".join(c for c in core_data_dir_tainted)
            prep_models_dir = os.path.join(core_data_dir, "data", "ai", "vision", "models")
            prep_videos_dir = os.path.join(core_data_dir, "data", "ai", "vision", "videos")

            logger.info("Build 1: Preparing FW custom device-specific DLStreamer images...")
            dlstreamer_result = prepare_dlstreamer_assets(
                configs=configs,
                models_dir=prep_models_dir,
                videos_dir=prep_videos_dir,
                src_dir=src_dir,
                docker_client=docker_client,
                docker_image_tag_analyzer="test-dlstreamer-analyzer:latest",
                docker_image_tag_utils="test-dlstreamer-utils:latest",
                docker_container_prefix="test",
            )

            fw_container_config = dlstreamer_result.metadata.get("container_config", {})
            logger.info(f"FW custom images available: {list(fw_container_config.keys())}")

            # Build 2: Select FW custom image based on device
            fw_custom_base_image = fw_container_config.get("analyzer_image", "intel/dlstreamer:2025.2.0-ubuntu24")

            # Check device parameter
            device_lower = device.lower()
            if "npu" in device_lower:
                fw_custom_base_image = fw_container_config.get(
                    "npu_analyzer_image",
                    fw_container_config.get("analyzer_image", "intel/dlstreamer:2025.2.0-ubuntu24")
                )
                logger.info("Detected NPU device, using NPU custom image")
            elif device_lower.startswith("gpu"):
                # Check for discrete GPU
                from sysagent.utils.system.hardware import collect_hardware_info
                hardware_info = collect_hardware_info()
                gpu_info = hardware_info.get("gpu", {})
                is_discrete = any(
                    gpu.get("is_discrete", False) for gpu in gpu_info.get("devices", [])
                )
                if is_discrete:
                    fw_custom_base_image = fw_container_config.get(
                        "dgpu_analyzer_image",
                        fw_container_config.get("analyzer_image", "intel/dlstreamer:2025.2.0-ubuntu24")
                    )
                    logger.info("Detected discrete GPU, using dGPU custom image")

            logger.info(f"Using FW image as base for media pipeline test: {fw_custom_base_image}")

            docker_nocache = configs.get("docker_nocache", False)
            logger.info(f"Docker build cache setting: nocache={docker_nocache}")
            logger.info(
                f"Build 2: Building test suite image '{docker_image_tag}' on top of FW custom image..."
            )

            # Download video files from GitHub
            logger.info("Downloading media benchmark video files from GitHub...")
            from pathlib import Path

            from esq.utils.models.media_resources import download_media_resources

            # Create videos directory in media results folder
            videos_path = Path(media_results) / "videos"
            videos_path.mkdir(parents=True, exist_ok=True)

            # Download required video files
            if not download_media_resources(str(videos_path)):
                logger.error("Failed to download media benchmark video files")
                raise RuntimeError("Media resource preparation failed")

            logger.info("Successfully downloaded all media benchmark video files")

            # Use extended Docker build context to include shared utilities
            # Build context set to esq/ directory to access utils/media/ directly
            logger.info("Preparing Docker build with extended build context...")

            test_file_dir = Path(__file__).resolve().parent

            # Dockerfile path relative to esq/ build context
            relative_dockerfile = "suites/media/performance/src/containers/media_benchmark/Dockerfile"

            build_context_dir = _resolve_esq_build_context(test_file_dir, relative_dockerfile)
            logger.debug(f"Build context directory: {build_context_dir}")
            logger.debug(f"Relative Dockerfile path: {relative_dockerfile}")

            build_args = {
                "COMMON_BASE_IMAGE": fw_custom_base_image,  # FW custom image
            }

            docker_client.build_image(
                path=str(build_context_dir),
                tag=docker_image_tag,
                nocache=docker_nocache,
                dockerfile=relative_dockerfile,
                buildargs=build_args,
            )

            result = Result(
                metadata={
                    "status": True,
                    "message": "Media Performance Runner",
                    "timeout (s)": timeout,
                    "display_name": test_display_name,
                }
            )

            return result

    except KeyboardInterrupt:
        failure_message = (
            f"User interrupt (Ctrl+C) detected during Media Benchmark test preparation. "
            f"Test: {test_display_name}, Device: {device}. "
            f"Partial setup may be incomplete."
        )
        logger.error(failure_message)
        raise

    except Exception as e:
        test_failed = True
        failure_message = (
            f"Unexpected error during test preparation: {type(e).__name__}: {str(e)}. "
            f"Test: {test_display_name}, Device: {device}, Docker image: {docker_image_tag}. "
            f"Check logs for full stack trace and error details."
        )
        logger.error(failure_message, exc_info=True)
        logger.debug(f"Preparation context - Docker dir: {docker_dir}")
        # Don't raise yet - create N/A result below

    try:
        prepare_test(test_name=test_name, configs=configs, prepare_func=prepare_assets, name="media_test_assets")
    except Exception as prep_error:
        # Handle docker build or other preparation failures
        test_failed = True
        failure_message = (
            f"Test preparation failed during asset setup: {type(prep_error).__name__}: {str(prep_error)}. "
            f"Possible causes: Docker build failure, network issues, or dependency problems. "
            f"Docker image: {docker_image_tag}. "
            f"Check logs for detailed error and verify Docker daemon is running."
        )
        logger.error(failure_message, exc_info=True)
        logger.debug(f"Preparation failed - Docker dir: {docker_dir}, Timeout: {timeout}s")

    # If preparation failed, return N/A metrics immediately
    if test_failed:
        metrics = _create_media_metrics(value="N/A", unit=None)

        results = Result.from_test_config(
            configs=configs,
            parameters={
                "timeout(s)": timeout,
                "display_name": test_display_name,
                "device": device,
            },
            metrics=metrics,
            metadata={
                "status": "N/A",
                "failure_reason": failure_message,
            },
        )

        # Summarize with N/A status and exit
        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
        pytest.fail(failure_message)

    # Normalize codec once for use in aggregated CSV filename (needed outside run_test())
    codec_normalized = codec.replace(".", "").lower()  # H.264 -> h264

    try:

        def run_test():
            resolution = configs.get("resolution", "1080p")  # Add resolution parameter

            # Track container name for cleanup (use nonlocal to access outer scope variable)
            nonlocal container_name

            # Define metrics with default values
            # Duration defaults to 0.0 instead of N/A since container doesn't write it
            metrics = _create_media_metrics(value="N/A", unit=None)
            metrics["duration_s"] = Metrics(unit="s", value=0.0, is_key_metric=False)

            # Mapping from CSV column names to metric keys
            csv_to_metric_map = {
                "Max Channels": "max_streams",  # Actual max channels from test execution
                "Average FPS": "avg_fps",  # Average FPS (backup to log extraction)
                "Ref GPU Freq": "gpu_freq_mhz",  # Reference GPU frequency (will be -1 for now)
                "Ref Pkg Power": "pkg_power_w",  # Reference package power (will be -1 for now)
                "Duration(s)": "duration_s",  # Test duration (optional - defaults to 0 if missing)
            }

            logger.debug(f"run_test() extracting metrics: {list(csv_to_metric_map.keys())}")

            # Initialize result template using from_test_config for automatic metadata application
            result = Result.from_test_config(
                configs=configs,
                parameters={
                    "test_id": test_id,
                    "device": device,
                    "display_name": test_display_name,
                },
                metrics=metrics,
                metadata={
                    "status": "N/A",
                },
            )

            # Check if devices are available
            if not device_dict:
                error_msg = (
                    f"No available devices found for configured device category: '{device}'. "
                    f"Expected device types: {device}. "
                    f"Verify hardware availability and driver installation (Intel GPU drivers required)."
                )
                logger.error(error_msg)
                logger.debug(f"Test configuration - device category: {device}, display_name: {test_display_name}")
                result.metadata["failure_reason"] = error_msg
                return result

            # Log detailed device information and execute test
            try:
                # Build device mapping with correct canonical names based on actual device types
                # Use common utility function for consistency with proxy test suite
                from esq.utils.media.validation import normalize_device_name

                device_canonical_map = {}
                dgpu_index = 0  # Counter for discrete GPUs
                for device_id, device_info in device_dict.items():
                    logger.info(
                        f"Device {device_id}: Type={device_info['device_type']}, Name={device_info['full_name']}"
                    )

                    # Use common utility to normalize device name
                    device_type = device_info.get("device_type")
                    canonical_name = normalize_device_name(device_id, device_type)

                    # If discrete GPU, add sequential index
                    if canonical_name == "dGPU":
                        canonical_name = f"dGPU.{dgpu_index}"
                        dgpu_index += 1

                    device_canonical_map[device_id] = canonical_name
                    logger.debug(f"Device {device_id} (Type={device_type}) mapped to {canonical_name}")

                for device_id, device_info in device_dict.items():
                    canonical_device_name = device_canonical_map[device_id]

                    # Check if specific test result exists in CSV (not just if CSV file exists)
                    # Determine which CSV file to check based on operation
                    if operation == "encode":
                        result_csv_path = Path(f"{media_results}/media_encode_performance_benchmark.csv")
                    else:
                        result_csv_path = Path(f"{media_results}/media_performance_benchmark.csv")

                    # Check if this specific device/codec/bitrate combination exists in CSV
                    result_exists_in_csv = False
                    if result_csv_path.exists():
                        try:
                            df = pd.read_csv(result_csv_path)
                            if not df.empty:
                                # Normalize for matching
                                codec_normalized = codec.replace(".", "").lower()
                                # Use canonical name from mapping (already determined by device type)
                                device_normalized = canonical_device_name.lower()

                                # Handle both old and new column names for backward compatibility
                                codec_col = "Codec" if "Codec" in df.columns else "Input Codec"
                                bitrate_col = "Bitrate" if "Bitrate" in df.columns else "Input Bitrate"

                                # Check if matching row exists
                                mask = (
                                    (df[codec_col].str.replace(".", "").str.lower() == codec_normalized)
                                    & (df[bitrate_col] == bitrate)
                                    & (df["Device Used"].str.lower() == device_normalized)
                                )

                                result_exists_in_csv = mask.any()
                                if result_exists_in_csv:
                                    logger.info(
                                        f"Result already exists in CSV for device={device_id}, "
                                        f"operation={operation}, codec={codec}, bitrate={bitrate}"
                                    )
                        except Exception as e:
                            logger.warning(f"Failed to check CSV for existing results: {e}")

                    # Determine if we need to run the benchmark
                    should_run_benchmark = (not result_exists_in_csv or force_run) and not skip_container_execution

                    if should_run_benchmark:
                        if force_run:
                            logger.info("Force run enabled. Running media benchmark container...")
                        else:
                            logger.info(
                                f"Result not found in CSV for device={device_id}, operation={operation}, "
                                f"codec={codec}, bitrate={bitrate}. Running media benchmark container..."
                            )
                    elif skip_container_execution:
                        logger.warning(
                            f"TEMP: Skipping container execution (skip_container_execution=True). "
                            f"Will use existing CSV results for device={device_id}, operation={operation}, "
                            f"codec={codec}, bitrate={bitrate}"
                        )

                    if should_run_benchmark:
                        # Initialize CSV files before running container
                        # The container's update_csv function requires CSV files to exist with headers
                        _initialize_media_csv_files(media_results)

                        try:
                            # Run media container using DockerClient API
                            # Track container name for cleanup in finally block
                            container_name = (
                                f"{configs.get('container_image', 'media_perf_runner')}_{device_id}_{test_id}"
                            )
                            container_result = _run_media_container(
                                docker_client=docker_client,
                                container_name=container_name,
                                image_name=configs.get("container_image", "media_perf_runner"),
                                image_tag=configs.get("image_tag", "1.0"),
                                device_id=canonical_device_name,
                                media_results=media_results,
                                operation=operation,
                                codec=codec,
                                bitrate=bitrate,
                                resolution=resolution,
                                platform_info=platform_info,
                                configs=configs,
                                timeout=timeout,
                                data_dir=data_dir,
                            )

                            # Check container exit code
                            exit_code = container_result.get("container_info", {}).get("exit_code", 1)
                            if exit_code != 0:
                                error_msg = (
                                    f"Container execution failed for device '{device_id}' with exit code {exit_code}. "
                                    f"Check container logs for detailed error information."
                                )
                                logger.error(error_msg)
                                logger.debug(
                                    f"Container parameters - Results dir: {media_results}, Device: {device_id}"
                                )
                                result.metadata["failure_reason"] = error_msg
                                result.metadata["status"] = "N/A"
                                return result

                        except Exception as container_error:
                            error_msg = (
                                f"Failed to run media container for device '{device_id}': {str(container_error)}. "
                                f"Check logs for detailed error information."
                            )
                            logger.error(error_msg, exc_info=True)
                            result.metadata["failure_reason"] = error_msg
                            result.metadata["status"] = "N/A"
                            return result

                    # Determine CSV file path based on operation
                    if operation == "encode":
                        csv_file_path = Path(f"{media_results}/media_encode_performance_benchmark.csv")
                    else:
                        # decode and decode + compose both use the main CSV
                        csv_file_path = Path(f"{media_results}/media_performance_benchmark.csv")

                    if csv_file_path.exists():
                        logger.info(f"Processing CSV file: {csv_file_path.name}")

                        try:
                            # Read CSV with pandas
                            df = pd.read_csv(csv_file_path)

                            if df.empty:
                                logger.warning(f"CSV file contains no data rows: {csv_file_path.name}")
                                continue

                            # Normalize column names (strip spaces)
                            df.columns = [col.strip() for col in df.columns]
                            logger.debug(f"CSV columns: {list(df.columns)}")

                            # Filter rows to find matching test result
                            # Use pandas query for cleaner filtering with normalized values
                            # Normalize codec for matching (H.264 -> h264)
                            codec_normalized = codec.replace(".", "").lower()  # H.264 -> h264

                            # Use canonical name from mapping (already determined by device type)
                            device_normalized = canonical_device_name.lower()

                            # Filter based on operation type
                            mask = pd.Series([True] * len(df))

                            # Handle both old column names (Input Codec) and new names (Codec) for backward compatibility
                            codec_col = "Codec" if "Codec" in df.columns else "Input Codec"
                            bitrate_col = "Bitrate" if "Bitrate" in df.columns else "Input Bitrate"

                            mask = mask & (df[codec_col].str.replace(".", "").str.lower() == codec_normalized)
                            mask = mask & (df[bitrate_col] == bitrate)
                            mask = mask & (df["Device Used"].str.lower() == device_normalized)

                            filtered_df = df[mask]

                            if filtered_df.empty:
                                logger.error(
                                    f"No matching data found. Looking for: operation={operation}, "
                                    f"device={device_normalized}, codec={codec_normalized}, bitrate={bitrate}"
                                )
                                logger.debug(f"Available devices in CSV: {df['Device Used'].unique()}")
                                logger.debug(f"Available codecs in CSV: {df['Codec'].unique()}")
                                logger.debug(f"Available bitrates in CSV: {df['Bitrate'].unique()}")
                                continue

                            # Take the first matching row
                            best_row = filtered_df.iloc[0]
                            logger.info(f"Found matching row: {len(filtered_df)} result(s), using first match")

                            # Extract metrics directly from CSV columns
                            for csv_col, metric_key in csv_to_metric_map.items():
                                if csv_col in best_row.index:
                                    value = best_row[csv_col]
                                    try:
                                        if csv_col == "Max Channels":
                                            # Max streams/channels (key metric)
                                            result.metrics[metric_key].value = (
                                                int(value)
                                                if str(value) not in ["N/A", "nan", "", "None"] and pd.notna(value)
                                                else 0
                                            )
                                            result.metrics[metric_key].unit = "streams"
                                            logger.info(f"Extracted {metric_key} = {value}")
                                        elif csv_col == "Average FPS":
                                            # Average FPS (backup if log extraction fails)
                                            fps_val = (
                                                float(value)
                                                if str(value) not in ["N/A", "nan", "", "-1"] and pd.notna(value)
                                                else 0.0
                                            )
                                            if fps_val > 0 and result.metrics[metric_key].value == "N/A":
                                                result.metrics[metric_key].value = fps_val
                                                result.metrics[metric_key].unit = "fps"
                                                logger.debug(f"Extracted {metric_key} = {value} from CSV (backup)")
                                        elif csv_col == "Duration(s)":
                                            # Duration in seconds
                                            duration_str = str(value).strip()
                                            if (
                                                duration_str
                                                and duration_str not in ["N/A", "nan", "", "-1", "None"]
                                                and pd.notna(value)
                                            ):
                                                result.metrics[metric_key].value = float(duration_str)
                                                result.metrics[metric_key].unit = "s"
                                                logger.debug(f"Extracted {metric_key} = {value}")
                                        elif csv_col in ["Ref GPU Freq"]:
                                            # Reference GPU frequency in MHz (will be -1 if not available)
                                            try:
                                                freq_val = (
                                                    float(value)
                                                    if pd.notna(value) and str(value) not in ["N/A", "nan", "", "None"]
                                                    else -1.0
                                                )
                                                result.metrics[metric_key].value = freq_val
                                                result.metrics[metric_key].unit = "MHz" if freq_val > 0 else None
                                                logger.debug(f"Extracted {metric_key} = {freq_val} MHz")
                                            except (ValueError, TypeError):
                                                result.metrics[metric_key].value = -1.0
                                                logger.debug(f"{metric_key} = -1 (not available)")
                                        elif csv_col in ["Ref Pkg Power"]:
                                            # Reference package power in watts (will be -1 if not available)
                                            try:
                                                power_val = (
                                                    float(value)
                                                    if pd.notna(value) and str(value) not in ["N/A", "nan", "", "None"]
                                                    else -1.0
                                                )
                                                result.metrics[metric_key].value = power_val
                                                result.metrics[metric_key].unit = "W" if power_val > 0 else None
                                                logger.debug(f"Extracted {metric_key} = {power_val} W")
                                            except (ValueError, TypeError):
                                                result.metrics[metric_key].value = -1.0
                                                logger.debug(f"{metric_key} = -1 (not available)")
                                    except (ValueError, TypeError) as e:
                                        logger.warning(f"Failed to convert {csv_col}={value}: {e}")
                                else:
                                    logger.debug(f"CSV column '{csv_col}' not found in CSV (may be optional)")

                            # Extract avg_fps from log file (not from CSV)
                            log_filename = "media_performance_benchmark_runner.log"
                            log_file_path = Path(media_results) / log_filename

                            if log_file_path.exists():
                                try:
                                    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as log_f:
                                        log_lines = log_f.readlines()

                                    # Pattern: "Average fps is X" or "Average FPS: X"
                                    import re

                                    fps_pattern = r"Average\s+fps\s+(?:is\s+|:\s*)([\d.]+)"

                                    fps_values = []
                                    for line in log_lines:
                                        match = re.search(fps_pattern, line, re.IGNORECASE)
                                        if match:
                                            fps_values.append(float(match.group(1)))

                                    if fps_values:
                                        result.metrics["avg_fps"].value = fps_values[-1]  # Use last/final FPS
                                        result.metrics["avg_fps"].unit = "fps"
                                        logger.info(
                                            f"Extracted avg_fps={fps_values[-1]} from {log_filename} ({len(fps_values)} entries)"
                                        )
                                    else:
                                        logger.warning(f"No FPS values found in log: {log_filename}")

                                    # Extract duration from log
                                    duration_pattern = r"(?:Duration|total\s+time)[:\s]+([\\d.]+)\\s*s(?:econds)?"
                                    duration_values = []
                                    for line in log_lines:
                                        match = re.search(duration_pattern, line, re.IGNORECASE)
                                        if match:
                                            duration_values.append(float(match.group(1)))

                                    if duration_values:
                                        result.metrics["duration_s"].value = duration_values[-1]
                                        result.metrics["duration_s"].unit = "s"
                                        logger.info(f"Extracted duration={duration_values[-1]}s from {log_filename}")
                                    else:
                                        logger.debug(f"No duration found in log: {log_filename}")
                                except Exception as log_err:
                                    logger.warning(f"Failed to extract metrics from {log_filename}: {log_err}")
                            else:
                                logger.warning(f"Log file not found: {log_file_path}")

                            # Verify key metric was assigned
                            if result.metrics["max_streams"].value == "N/A" or result.metrics["max_streams"].value == 0:
                                logger.warning("Key metric 'max_streams' not extracted or is zero")

                        except Exception as csv_error:
                            error_msg = (
                                f"Failed to parse CSV file: {type(csv_error).__name__}: {str(csv_error)}. "
                                f"CSV file: {csv_file_path}. "
                                f"Verify CSV format matches expected structure."
                            )
                            logger.error(error_msg, exc_info=True)
                            result.metadata["failure_reason"] = error_msg
                            result.metadata["status"] = "N/A"
                            return result
                    else:
                        # CSV file not found - keep all metrics as N/A and mark as failure
                        error_msg = (
                            f"Results CSV file not found at expected location: {csv_file_path}. "
                            f"Test container may have failed to generate results. "
                            f"Expected file: {csv_file_path.name} in {media_results}. "
                            f"Check container logs for execution errors."
                        )
                        logger.error(error_msg)
                        results_dir_contents = (
                            list(Path(media_results).iterdir())
                            if Path(media_results).exists()
                            else "Directory not found"
                        )
                        logger.debug(f"Results directory contents: {results_dir_contents}")
                        result.metadata["failure_reason"] = "Results CSV file not generated by test container"
                        result.metadata["status"] = "N/A"
                        return result

                valid_metrics = [m for m in result.metrics.values() if m.value != "N/A"]
                if not valid_metrics:
                    metric_names = list(result.metrics.keys())
                    error_msg = (
                        f"Test completed but no valid metrics were collected (all N/A). "
                        f"Expected metrics: {', '.join(metric_names)}. "
                        f"CSV file was found but metric extraction failed. "
                        f"Verify CSV format matches expected structure."
                    )
                    logger.error(error_msg)
                    logger.debug(f"CSV file location: {csv_file_path if 'csv_file_path' in locals() else 'undefined'}")
                    result.metadata["failure_reason"] = error_msg
                    result.metadata["status"] = "N/A"
                    return result

                # If successfully processed all devices and collected valid metrics, mark as success
                result.metadata["status"] = True
                result.metadata.pop("failure_reason", None)  # Remove failure_reason if test succeeded

            except Exception as exec_error:
                # Handle any execution errors (shell script failures, CSV parsing, etc.)
                error_msg = (
                    f"Test execution failed with exception: {type(exec_error).__name__}: {str(exec_error)}. "
                    f"Device: {device}, Operation: {operation}, Codec: {codec}. "
                    f"Check logs for stack trace and detailed error information."
                )
                logger.error(error_msg, exc_info=True)
                logger.debug(f"Execution context - Results dir: {media_results}")
                result.metadata["failure_reason"] = error_msg
                # Metrics remain as N/A
                return result
            logger.debug(f"Test results: {json.dumps(result.to_dict(), indent=2)}")

            return result

    except KeyboardInterrupt:
        failure_message = (
            f"User interrupt (Ctrl+C) detected during test execution. "
            f"Test: {test_display_name}, Device: {device}. "
            f"Test execution was terminated before completion."
        )
        test_interrupted = True
        logger.error(failure_message)
        # Cleanup any running containers with extended timeout for graceful shutdown
        if container_name:
            try:
                logger.debug(f"Cleaning up container: {container_name}")
                docker_client.cleanup_container(container_name, timeout=30)
                logger.debug(f"Successfully cleaned up container: {container_name}")
            except Exception as cleanup_err:
                logger.warning(
                    f"Container cleanup warning for {container_name}: {cleanup_err}. Container may need manual cleanup."
                )
        raise  # Re-raise KeyboardInterrupt to propagate to caller

    except Exception as e:
        test_failed = True
        failure_message = (
            f"Unexpected error during test execution: {type(e).__name__}: {str(e)}. "
            f"Test: {test_display_name}, Device: {device}. "
            f"Check logs for complete stack trace and error context."
        )
        logger.error(failure_message, exc_info=True)
        logger.debug(f"Execution context - Test ID: {test_id}, Operation: {operation}")

        # Attach traceback to Allure report
        try:
            import traceback

            tb_str = traceback.format_exc()
            allure.attach(
                tb_str,
                name=f"Execution Exception Traceback - {operation}",
                attachment_type=allure.attachment_type.TEXT,
            )
        except Exception as attach_error:
            logger.debug(f"Failed to attach traceback: {attach_error}")

    finally:
        # Cleanup: Ensure Docker containers are stopped/removed even if test fails or is interrupted
        if container_name:
            try:
                logger.debug(f"Finally block: Cleaning up container {container_name}")
                docker_client.cleanup_container(container_name, timeout=30)
                logger.info(f"Finally block: Successfully cleaned up container {container_name}")
            except Exception as cleanup_err:
                logger.warning(
                    f"Finally block: Failed to cleanup container {container_name}: {cleanup_err}. "
                    "Container may require manual cleanup with 'docker rm -f'."
                )
        else:
            logger.debug("Finally block: No container to cleanup (container_name is None)")

    # Execute the test with shared fixture
    results = execute_test_with_cache(
        cached_result=cached_result,
        cache_result=cache_result,
        test_name=test_name,
        configs=configs,
        run_test_func=run_test,
    )

    # Handle N/A status (missing hardware or test failures)
    if results.metadata.get("status") == "N/A" and "failure_reason" in results.metadata:
        failure_msg = results.metadata["failure_reason"]

        # Check if failure is due to missing hardware (not a test execution error)
        is_hardware_missing = "No available devices found" in failure_msg

        if is_hardware_missing:
            logger.error(f"Test failed - hardware not available: {failure_msg}")
            logger.info(f"Test summary - ID: {test_id}, Device: {device}, Operation: {operation}")
            logger.info(f"N/A metrics will be reported for {device.upper()} (hardware not present)")
        else:
            # Actual test execution error - log as error
            logger.error(f"Test failed with N/A status: {failure_msg}")
            logger.info(f"Test summary - ID: {test_id}, Device: {device}, Operation: {operation}")

        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )

        pytest.fail(f"Media Benchmark test failed - {failure_msg}")

    # Validate test results against KPI thresholds
    validate_test_results(results=results, configs=configs, get_kpi_config=get_kpi_config, test_name=test_name)
    try:
        logger.info(f"Generating test result visualizations (always executed) Results: {results}")

        # Summarize results using the shared fixture
        # NOTE: enable_visualizations NOT used - charts generated separately via _generate_media_charts()
        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as summary_error:
        error_msg = (
            f"Test result summarization failed: {type(summary_error).__name__}: {str(summary_error)}. "
            f"Test execution completed but report generation failed. "
            f"Results may be incomplete in final report."
        )
        logger.error(error_msg, exc_info=True)
        device_keys = list(device_dict.keys()) if device_dict else "None"
        logger.debug(f"Summary context - Results dir: {media_results}, Device dict: {device_keys}")

    # Create aggregated CSV for chart generation from main performance benchmark CSV
    # Read from the main CSV that container writes to (not the per-test CSV)
    try:
        # Main CSV file that container writes to
        # Note: For "decode + compose", CSV is still written to "decode" file
        # Map operation to CSV filename
        if operation == "encode":
            main_csv_name = "media_encode_performance_benchmark.csv"
        else:
            # Both "decode" and "decode + compose" use the main decode CSV
            main_csv_name = "media_performance_benchmark.csv"

        main_csv_path = Path(f"{media_results}/{main_csv_name}")

        if main_csv_path.exists():
            logger.info(f"Reading main CSV for aggregation: {main_csv_name}")
            # Read the main CSV
            main_df = pd.read_csv(main_csv_path)
            logger.debug(f"Main CSV has {len(main_df)} rows")

            # Get the actual device ID from device_dict (device config is a list/string, device_id is actual ID)
            # Extract first device from device_dict (e.g., "GPU.1" for dGPU)
            if device_dict:
                actual_device_id = list(device_dict.keys())[0]
                logger.debug(f"Using device ID for aggregation: {actual_device_id}")
            else:
                logger.warning("No devices in device_dict, cannot create aggregated CSV")
                raise ValueError("No devices available for aggregation")

            # Get canonical name for the device using common utility
            from esq.utils.media.validation import normalize_device_name

            device_type = device_dict[actual_device_id].get("device_type")
            device_normalized = normalize_device_name(actual_device_id, device_type)
            logger.debug(f"Canonical device for CSV matching: {actual_device_id} → {device_normalized}")

            # Normalize codec for matching (remove dots, lowercase)
            codec_normalized_match = codec.replace(".", "").replace("-", "").lower()

            # Handle both old and new column names for backward compatibility
            codec_col = "Codec" if "Codec" in main_df.columns else "Input Codec"
            bitrate_col = "Bitrate" if "Bitrate" in main_df.columns else "Input Bitrate"

            # Filter for matching row
            mask = (
                (main_df["Device Used"].str.contains(device_normalized, case=False, na=False))
                & (main_df[codec_col].str.replace(".", "").str.replace("-", "").str.lower() == codec_normalized_match)
                & (main_df[bitrate_col] == bitrate)
            )

            filtered_df = main_df[mask]

            if not filtered_df.empty:
                # Normalize operation name for filename (replace spaces with underscores)
                operation_normalized = operation.replace(" ", "_")
                aggregated_csv_path = Path(f"{media_results}/aggregated_{codec_normalized}_{operation_normalized}.csv")

                # Extract the test result row
                test_row = filtered_df.iloc[0].to_dict()

                # Add test_id for tracking
                test_row["Test Id"] = test_id
                test_row["Operation"] = operation

                # Read existing aggregated data or create new
                if aggregated_csv_path.exists():
                    agg_df = pd.read_csv(aggregated_csv_path)
                    # Remove old row with same test_id if exists (overwrite)
                    agg_df = agg_df[agg_df["Test Id"] != test_id]
                    # Append new row
                    agg_df = pd.concat([agg_df, pd.DataFrame([test_row])], ignore_index=True)
                else:
                    agg_df = pd.DataFrame([test_row])

                # Save aggregated CSV with original column names for chart generation
                agg_df.to_csv(aggregated_csv_path, index=False)
                logger.info(
                    f"Updated aggregated CSV: {aggregated_csv_path.name} "
                    f"(now has {len(agg_df)} rows, added test {test_id})"
                )
            else:
                logger.warning(
                    f"No matching row found in {main_csv_name} for device={device_normalized}, "
                    f"codec={codec_normalized_match}, bitrate={bitrate}"
                )
                logger.debug(f"Available devices in CSV: {main_df['Device Used'].unique()}")

    except Exception as agg_error:
        logger.warning(f"Failed to update aggregated CSV: {agg_error}", exc_info=True)

    # Attach CSV artifacts (only aggregated CSV)
    # Normalize operation name for filename (replace spaces with underscores)
    operation_normalized = operation.replace(" ", "_")
    aggregated_csv_path = Path(f"{media_results}/aggregated_{codec_normalized}_{operation_normalized}.csv")
    if aggregated_csv_path.exists():
        try:
            # Read aggregated CSV and filter to essential columns only
            df = pd.read_csv(aggregated_csv_path)

            # Keep only essential columns
            essential_cols = [
                "Media Performance Benchmark",
                "Device Used",
                "Codec",
                "Bitrate",
                "Resolution",
                "Num Monitors",
                "Max Channels",
                "GPU Freq",
                "Pkg Power",
                "Ref Platform",
                "Ref Max Channels",
                "Ref GPU Freq",
                "Ref Pkg Power",
                "Duration(s)",
            ]

            # Filter to available essential columns
            available_cols = [col for col in essential_cols if col in df.columns]
            df_filtered = df[available_cols]

            # Save filtered CSV temporarily
            filtered_csv_path = aggregated_csv_path.parent / f"filtered_{aggregated_csv_path.name}"
            df_filtered.to_csv(filtered_csv_path, index=False)

            # Attach filtered CSV
            file_name = os.path.basename(aggregated_csv_path)
            with open(filtered_csv_path, "rb") as f:
                allure.attach(f.read(), name=f"Media Results - {file_name}", attachment_type=allure.attachment_type.CSV)
            logger.info(f"Attached aggregated CSV (filtered): {file_name}")

            # Clean up filtered CSV
            filtered_csv_path.unlink()
        except Exception as attach_error:
            logger.warning(f"Failed to attach aggregated CSV: {attach_error}")
    else:
        logger.debug(f"Aggregated CSV not found: {aggregated_csv_path}")

    # Generate chart only for this test's aggregated CSV (not all CSVs to avoid duplicates)
    operation_normalized = operation.replace(" ", "_")
    current_aggregated_csv = Path(f"{media_results}/aggregated_{codec_normalized}_{operation_normalized}.csv")

    if current_aggregated_csv.exists():
        logger.info(f"Generating chart for current test: {current_aggregated_csv.name}")
        try:
            _generate_media_charts(
                media_results=media_results,
                configs=configs,
                logger=logger,
                specific_csv=current_aggregated_csv.name,  # Only process this specific CSV
            )
            logger.info("Chart generation completed")
        except Exception as chart_gen_error:
            logger.error(f"Chart generation failed: {chart_gen_error}", exc_info=True)
    else:
        logger.debug(f"No aggregated CSV found for chart generation: {current_aggregated_csv.name}")

    if test_failed:
        pytest.fail(failure_message)
