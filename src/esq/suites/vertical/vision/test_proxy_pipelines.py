# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import grp
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import allure
import pandas as pd
import pytest
from esq.utils.genutils import plot_grouped_bar_chart
from esq.utils.media.validation import detect_platform_type
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result
from sysagent.utils.infrastructure import DockerClient
from sysagent.utils.system.ov_helper import get_available_devices_by_category

# Use FW API for secure subprocess execution
try:
    from sysagent.utils.core.process import ProcessSecurityConfig, get_executor
except ModuleNotFoundError:
    # Fallback for container environments (should not happen in test suite)
    ProcessSecurityConfig = None
    get_executor = None

logger = logging.getLogger(__name__)

test_container_path = "src/containers/proxy_pipeline_benchmark/"


def _create_proxy_pipeline_metrics(suite_name: str, value: str = "N/A", unit: Optional[str] = None) -> dict:
    """
    Create proxy pipeline performance metrics dictionary based on suite type.

    Args:
        suite_name: Suite name - "SmartAIRunner", "VisualAIRunner", "VsaasAIRunner", or "LPRAIRunner"
        value: Initial value for all metrics (default: "N/A")
        unit: Unit for metrics (default: None for N/A values)

    Returns:
        Dictionary of Metrics objects for the specified suite
    """
    # Common metrics for all suites (based on CSV columns available from container)
    # Note: CPU/Memory metrics excluded - not provided by container CSV output
    base_metrics = {
        "max_streams": Metrics(unit=unit, value=value, is_key_metric=True),
        "avg_fps": Metrics(unit=unit, value=value, is_key_metric=False),
        "gpu_freq_mhz": Metrics(unit=unit, value=value, is_key_metric=False),
        "pkg_power_w": Metrics(unit=unit, value=value, is_key_metric=False),
        "duration_s": Metrics(unit=unit, value=value, is_key_metric=False),
    }

    return base_metrics


def folder_copy_all(src_dir, dst_dir):
    # Ensure destination exists
    os.makedirs(dst_dir, exist_ok=True)

    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)

        if os.path.isdir(src_path):
            # Recursively copy directories
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            # Copy individual files
            shutil.copy2(src_path, dst_path)  # preserves metadata

def proxy_pl_suite(suite_name: str, resources_base_dir: str):
    """
    Configure test suite and CSV files based on suite name.

    Args:
        suite_name: Name of the suite to run
        resources_base_dir: Base directory for downloading resources (under esq_data)

    Returns:
        tuple: (test_suite_name, list of expected CSV files, models_path, videos_path)
    """
    csvlist = []

    match suite_name:
        case "SmartAIRunner" | "VisualAIRunner" | "VsaasAIRunner":
            # Common resource preparation for Smart NVR, Headed Visual AI, and VSaaS
            # Resources stored under esq_data/data/vertical/vision/results/ppl/resources
            models_path = os.path.join(resources_base_dir, "models")
            video_path = os.path.join(resources_base_dir, "videos")
            os.makedirs(models_path, exist_ok=True)
            ensure_dir_permissions(models_path, uid=os.getuid(), gid=os.getgid(), mode=0o775)
            os.makedirs(video_path, exist_ok=True)
            ensure_dir_permissions(video_path, uid=os.getuid(), gid=os.getgid(), mode=0o775)

            # Download proxy pipeline models and videos using Python utility
            logger.info("Downloading proxy pipeline resources (models and videos)...")
            try:
                from esq.utils.models.proxy_pipeline_resources import download_proxy_pipeline_resources

                success = download_proxy_pipeline_resources(models_path, video_path)
                if not success:
                    raise RuntimeError("Failed to download proxy pipeline resources")

                logger.info(f"Proxy pipeline resources prepared: models={models_path}, videos={video_path}")

            except ImportError as e:
                logger.error(f"Failed to import proxy pipeline resource downloader: {e}")
                raise RuntimeError(f"Proxy pipeline resource downloader not available: {e}") from e
            except Exception as e:
                logger.error(f"Failed to download proxy pipeline resources: {e}")
                raise RuntimeError(f"Proxy pipeline resource download failed: {e}") from e

            # Set test suite name based on suite_name
            if suite_name == "SmartAIRunner":
                test_suite = "smart_nvr_benchmark"
                csvlist.append("smart_nvr_proxy_pipeline.csv")
            elif suite_name == "VisualAIRunner":
                test_suite = "headed_visual_ai_benchmark"
                csvlist.append("headed_visual_ai_proxy_pipeline.csv")
            else:  # VsaasAIRunner
                test_suite = "ai_vsaas_benchmark"
                csvlist.append("ai_vsaas_proxy_pipeline.csv")

        case "LPRAIRunner":
            test_suite = "lpr_benchmark"
            # LPR resources stored under esq_data/data/vertical/vision/results/ppl/resources
            models_path = os.path.join(resources_base_dir, "models")
            video_path = os.path.join(resources_base_dir, "videos")
            os.makedirs(models_path, exist_ok=True)
            ensure_dir_permissions(models_path, uid=os.getuid(), gid=os.getgid(), mode=0o775)
            os.makedirs(video_path, exist_ok=True)
            ensure_dir_permissions(video_path, uid=os.getuid(), gid=os.getgid(), mode=0o775)

            # Download LPR models and videos using Python utility
            logger.info("Downloading LPR resources (models and videos)...")
            try:
                from esq.utils.models.lpr_resources import download_lpr_resources

                success = download_lpr_resources(models_path, video_path)
                if not success:
                    raise RuntimeError("Failed to download LPR resources")

                logger.info(f"LPR resources prepared: models={models_path}, videos={video_path}")

            except ImportError as e:
                logger.error(f"Failed to import LPR resource downloader: {e}")
                raise RuntimeError(f"LPR resource downloader not available: {e}") from e
            except Exception as e:
                logger.error(f"Failed to download LPR resources: {e}")
                raise RuntimeError(f"LPR resource download failed: {e}") from e
            csvlist.append("lpr_proxy_pipeline.csv")

        case _:
            test_suite = "None"
            models_path = None
            video_path = None

    return test_suite, csvlist, models_path, video_path

def _generate_proxy_pipeline_charts(pp_results: str, suite_name: str, configs: dict, logger):
    """
    Generate performance charts for proxy pipeline benchmarks.

    Creates charts showing AI Channels/Streams vs Device/Mode, grouped by Model/Configuration.
    Chart layout varies by suite type:
    - Smart NVR, Headed Visual, VSaaS: AI Channels (y) vs Devices (x), grouped by Models
    - LPR: AI Channels (y) vs Modes (x), no grouping or grouped by configuration

    Args:
        pp_results: Path to directory containing CSV result files
        suite_name: Suite name (SmartAIRunner, VisualAIRunner, VsaasAIRunner, LPRAIRunner)
        configs: Test configuration dictionary containing metadata
        logger: Logger instance for debug/error messages
    """
    try:
        # Map suite names to CSV files and chart configurations
        suite_config = {
            "SmartAIRunner": {
                "csv_file": "smart_nvr_proxy_pipeline.csv",
                "chart_title": "Smart NVR - AI Channels Performance",
                "x_column": "Device",
                "y_column": "AI Channels",
                "group_column": "Model",
                "xlabel": "Device",
                "ylabel": "AI Channels (Stream Count)",
            },
            "VisualAIRunner": {
                "csv_file": "headed_visual_ai_proxy_pipeline.csv",
                "chart_title": "Headed Visual AI - AI Channels Performance",
                "x_column": "Device",
                "y_column": "AI Channels",
                "group_column": "Model",
                "xlabel": "Device",
                "ylabel": "AI Channels (Stream Count)",
            },
            "VsaasAIRunner": {
                "csv_file": "ai_vsaas_proxy_pipeline.csv",
                "chart_title": "VSaaS AI - AI Channels Performance",
                "x_column": "Device",
                "y_column": "AI Channels",
                "group_column": "Model",
                "xlabel": "Device",
                "ylabel": "AI Channels (Stream Count)",
            },
            "LPRAIRunner": {
                "csv_file": "lpr_proxy_pipeline.csv",
                "chart_title": "LPR Pipeline - AI Channels Performance",
                "x_column": "Mode",
                "y_column": "Streams",
                "group_column": "Mode",  # Same as x_column for no grouping
                "xlabel": "Mode",
                "ylabel": "AI Channels (Stream Count)",
            },
        }

        # Get configuration for this suite
        config = suite_config.get(suite_name)
        if not config:
            logger.warning(f"No chart configuration found for suite: {suite_name}")
            return

        csv_path = Path(f"{pp_results}/{config['csv_file']}")
        if not csv_path.exists():
            logger.debug(f"CSV file not found for {suite_name}: {csv_path}")
            return

        try:
            # Read CSV data
            df = pd.read_csv(csv_path)
            df = df.fillna(0.0)

            # Normalize column names
            df.columns = [col.strip() for col in df.columns]

            # Normalize string data in Model column (strip leading/trailing spaces)
            if "Model" in df.columns:
                df["Model"] = df["Model"].astype(str).str.strip()

            # Check if required columns exist
            required_cols = [config["x_column"], config["y_column"]]
            if config["x_column"] != config["group_column"]:
                required_cols.append(config["group_column"])

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in {suite_name} CSV: {missing_cols}")
                logger.debug(f"Available columns: {list(df.columns)}")
                return

            # Extract reference values - model-specific for all tests
            reference_value = None
            reference_platform = "Reference Platform"

            # For LPR, extract mode-specific reference values from CSV
            # Note: Column names are "Ref FPS" and "Ref Platform" after normalization
            if suite_name == "LPRAIRunner" and "Mode" in df.columns and "Ref FPS" in df.columns:
                # Build dict of mode -> reference value
                reference_value = {}
                for _, row in df.iterrows():
                    mode = row.get("Mode")
                    ref_fps = row.get("Ref FPS")
                    if mode and ref_fps and pd.notna(ref_fps):
                        # Use Mode as key (matches x_column for LPR)
                        reference_value[mode] = float(ref_fps)

                # Get reference platform from first row
                if "Ref Platform" in df.columns:
                    ref_platform_val = df["Ref Platform"].iloc[0]
                    if pd.notna(ref_platform_val):
                        reference_platform = str(ref_platform_val)

                logger.info(f"Extracted mode-specific reference values for {suite_name}: {reference_value}")
            else:
                # Smart NVR, Headed Visual AI, VSaaS: model-specific references
                # Actual CSV has "Ref AI Channels" and "Ref Platform" (not "Reference FPS"/"Reference Platform")
                if "Model" in df.columns and "Ref AI Channels" in df.columns:
                    model_reference_map = {}
                    reference_platforms = set()

                    # Extract reference value for each unique model from "Ref AI Channels" column
                    for model in df["Model"].unique():
                        model_df = df[df["Model"] == model]
                        if not model_df.empty:
                            # Ref AI Channels column contains AI Channel count reference values
                            ref_value = model_df.iloc[0].get("Ref AI Channels")
                            ref_platform = model_df.iloc[0].get("Ref Platform", "")

                            if not pd.isna(ref_value) and ref_value > 0:
                                model_reference_map[model] = float(ref_value)
                                if ref_platform:
                                    reference_platforms.add(ref_platform)

                    if model_reference_map:
                        reference_value = model_reference_map
                        # Use combined platform list if multiple platforms exist
                        reference_platform = ", ".join(sorted(reference_platforms))
                        logger.info(f"Model-specific reference values (AI Channels): {model_reference_map}")
                    else:
                        logger.warning(f"No model-specific reference values found in CSV for {suite_name}")
                else:
                    logger.warning(f"Required columns (Model, Ref AI Channels) not found in CSV for {suite_name}")

            # Create chart output path
            chart_filename = f"chart_{suite_name.lower()}.png"
            chart_path = csv_path.parent / chart_filename

            # Color map for models/devices
            color_map = {
                # Models
                "yolov5s": "#1f77b4",  # Blue
                "yolov8n": "#ff7f0e",  # Orange
                "efficientdet": "#2ca02c",  # Green
                "ssd": "#d62728",  # Red
                "mobilenet": "#9467bd",  # Purple
                "resnet": "#8c564b",  # Brown
                # Devices
                "iGPU": "#1f77b4",  # Blue
                "dGPU": "#ff7f0e",  # Orange
                "CPU": "#2ca02c",  # Green
                # Modes
                "Standard": "#1f77b4",  # Blue
                "Optimized": "#ff7f0e",  # Orange
                "High-Throughput": "#2ca02c",  # Green
            }

            # Generate grouped bar chart
            # For dict reference_value (mode-specific), adjust label
            if isinstance(reference_value, dict):
                ref_label = f"Reference ({reference_platform})"
            else:
                ref_label = f"Reference ({reference_platform})" if reference_value else "Reference"

            plot_grouped_bar_chart(
                csv_path=csv_path,
                output_path=chart_path,
                x_column=config["x_column"],
                y_column=config["y_column"],
                group_column=config["group_column"],
                title=config["chart_title"],
                xlabel=config["xlabel"],
                ylabel=config["ylabel"],
                reference_value=reference_value,
                reference_label=ref_label,
                figsize=(12, 6),
                rotation=0,
                color_map=color_map,
            )

            # Attach chart to Allure report
            if chart_path.exists():
                with open(chart_path, "rb") as f:
                    allure.attach(
                        f.read(),
                        name=f"Performance Chart - {suite_name}",
                        attachment_type=allure.attachment_type.PNG,
                    )
                logger.info(f"Generated and attached chart: {chart_filename}")

                # Clean up chart file
                chart_path.unlink()
            else:
                logger.error(f"Chart file was not created at: {chart_path}")
                logger.error("Chart generation may have failed - check logs above")

        except Exception as csv_error:
            logger.error(f"Failed to generate chart for {suite_name}: {csv_error}", exc_info=True)
            logger.error(f"CSV path: {csv_path}, Chart path: {chart_path if 'chart_path' in locals() else 'undefined'}")

    except Exception as chart_error:
        logger.warning(f"Failed to generate proxy pipeline charts: {chart_error}", exc_info=True)


def _initialize_csv_files(output_dir: str, csv_files: list):
    """
    Initialize CSV files with proper headers for proxy pipeline benchmarks.

    The container's update_csv function expects CSV files to exist with headers
    before it can write results (opens with "r+" mode).

    Args:
        output_dir: Output directory where CSV files will be created
        csv_files: List of CSV filenames to initialize
    """
    # CSV header formats for different proxy pipeline benchmarks
    csv_headers = {
        "lpr_proxy_pipeline.csv": (
            "Pipeline, Model, Mode, Devices, Streams, Average FPS,"
            "GPU Freq, Pkg Power,"
            "Ref Platform, Ref FPS, Ref GPU Freq, Ref Pkg Power,"
            "Duration(s), Errors"
        ),
        "headed_visual_ai_proxy_pipeline.csv": (
            "Pipeline, Device, Codec, Resolution, Model, Num Monitors, AI Channels,"
            "GPU Freq, Pkg Power,"
            "Ref Platform, Ref AI Channels, Ref GPU Freq, Ref Pkt Power,"
            "Duration(s), Errors"
        ),
        "ai_vsaas_proxy_pipeline.csv": (
            "Pipeline, Device, Codec, Resolution, Model, AI Channels,"
            "GPU Freq, Pkg Power,"
            "Ref Platform, Ref AI Channels, Ref GPU Freq, Ref Pkt Power,"
            "Duration(s), Errors"
        ),
        "smart_nvr_proxy_pipeline.csv": (
            "Pipeline, Device, Codec, Resolution, Model, Compose, Num Monitors, non-AI Channels, AI Channels,"
            "GPU Freq, Pkg Power,"
            "Ref Platform, Ref AI Channels, Ref GPU Freq, Ref Pkt Power,"
            "Duration(s), Errors"
        ),
        "default": (
            "Pipeline,Device Used,Input Codec,Input Resolution,Input Channels,"
            "Model,Compose,Number of Monitors,AI Channels,Average FPS,"
            "Reference Platform,Reference FPS,"
            "CPU Avg,CPU Peak,Mem Avg,Mem Peak,GPU Util Avg,GPU Util Peak,GPU Mem Avg,GPU Mem Peak,"
            "Duration(s),Errors"
        ),
    }

    for csv_file in csv_files:
        csv_path = Path(output_dir) / csv_file
        if not csv_path.exists():
            # Get appropriate header: LPR-specific or default for all others
            csv_header = csv_headers.get(csv_file, csv_headers["default"])
            logger.info(f"Initializing CSV file: {csv_file}")
            csv_path.write_text(f"{csv_header}\n")
            # Ensure proper permissions for container to write (owner/group rw, no world access)
            os.chmod(csv_path, 0o660)
        else:
            logger.debug(f"CSV file already exists: {csv_file}")


def _run_proxy_pipeline_container(
    docker_client: DockerClient,
    container_name: str,
    image_name: str,
    image_tag: str,
    benchmark_script: str,
    output_dir: str,
    devices: list,
    platform_info: dict,
    configs: dict,
    config_file: str = None,
    round_num: int = 1,
    fps: int = 30,
):
    """
    Run proxy pipeline benchmark container using DockerClient.

    Replaces run_container.sh functionality with Python DockerClient API.
    Uses consolidated utilities from esq/utils/media/ instead of local shell scripts.

    Args:
        docker_client: DockerClient instance
        container_name: Container name for tracking
        image_name: Docker image name
        image_tag: Docker image tag
        benchmark_script: Benchmark script name (e.g., "lpr_benchmark")
        output_dir: Host output directory
        devices: List of device strings (iGPU, dGPU.0, etc.)
        platform_info: Platform information from detect_platform_type()
        configs: Test configuration dictionary (for reading display_output, etc.)
        config_file: Optional configuration file path
        round_num: Test iteration number
        fps: Target FPS for benchmarking

    Returns:
        dict: Container execution result with keys: exit_code, container_logs_text, etc.
    """
    # Get render and user group IDs for GPU access
    render_gid = grp.getgrnam("render").gr_gid
    user_gid = os.getgid()

    # Prepare volumes
    sink_dir = os.path.join(output_dir, "sink")
    os.makedirs(sink_dir, exist_ok=True)
    ensure_dir_permissions(sink_dir, uid=os.getuid(), gid=user_gid, mode=0o775)

    # Base volumes
    volumes = {
        output_dir: {"bind": "/home/dlstreamer/output", "mode": "rw"},
        sink_dir: {"bind": "/home/dlstreamer/sink", "mode": "rw"},
        "/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "rw"},
    }

    # Mount models and videos directories from esq_data/data/vertical/vision/results/ppl/resources
    # Resources are stored in esq_data following FW team's recommendation for dynamic content
    # Paths are passed via configs from the test function
    models_path = configs.get("_models_path")  # Internal parameter passed from test
    videos_path = configs.get("_videos_path")  # Internal parameter passed from test

    # Mount models directory for ALL benchmarks (downloaded by resource utilities)
    if models_path and os.path.exists(models_path):
        volumes[models_path] = {"bind": "/home/dlstreamer/share/models", "mode": "ro"}
        logger.debug(f"Mounting models: {models_path} -> /home/dlstreamer/share/models")
    else:
        logger.warning(f"Models path not found or not provided: {models_path}")

    # Mount videos directory for ALL benchmarks (downloaded by resource utilities)
    # Note: Must be read-write (rw) because gst_loop_mp4.sh/gst_lpr_loop_mp4.sh scripts
    # create processed video files (e.g., *_180s_*.mp4, *_1min.mp4) in this directory
    if videos_path and os.path.exists(videos_path):
        volumes[videos_path] = {"bind": "/home/dlstreamer/sample_video", "mode": "rw"}
        logger.debug(f"Mounting videos: {videos_path} -> /home/dlstreamer/sample_video")
    else:
        logger.warning(f"Videos path not found or not provided: {videos_path}")

    # Add config file if specified
    if config_file and config_file != "none":
        if os.path.exists(config_file):
            config_basename = os.path.basename(config_file)
            volumes[config_file] = {"bind": f"/home/dlstreamer/{config_basename}", "mode": "ro"}

    # Prepare environment
    display = os.environ.get("DISPLAY", ":0")
    environment = {
        "DISPLAY": display,
        "DEVICES_LIST": " ".join(devices),
        "PL_NAME": benchmark_script,
        "OUTPUT_DIR": "/home/dlstreamer/output",
        "ROUND": str(round_num),
        "FPS": str(fps),
    }

    # Create Docker-compatible X authority file (required for xvimagesink/compositor)
    xauth_file = "/tmp/.docker.xauth"
    # Always recreate xauth file to ensure it matches current DISPLAY
    try:
        # Remove old file if exists
        if os.path.exists(xauth_file):
            os.remove(xauth_file)
        # Create empty xauth file
        Path(xauth_file).touch(mode=0o660)
        # Copy X11 authentication to Docker-compatible format

        # Sanitize display value using character-by-character copying to break taint chain
        # Allow only alphanumeric, colon, dot, and dash characters
        sanitized_display = "".join(c for c in display if c.isalnum() or c in ":.-")

        # Sanitize xauth_file path using character-by-character copying
        sanitized_xauth = "".join(c for c in xauth_file if c.isalnum() or c in "/._-")

        # Execute xauth command using explicit pipeline without shell=True
        # Chain three processes: xauth nlist | sed | xauth nmerge
        # Inputs are sanitized above to prevent command injection
        import subprocess  # nosec B404 - subprocess needed for xauth pipeline

        p1 = None
        p2 = None
        p3 = None
        try:
            # Process 1: xauth nlist - list X11 authorization entries
            p1 = subprocess.Popen(
                ["xauth", "nlist", sanitized_display],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Process 2: sed - replace first 4 chars with 'ffff' for wildcard hostname
            p2 = subprocess.Popen(
                ["sed", "-e", "s/^..../ffff/"],
                stdin=p1.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if p1.stdout:
                p1.stdout.close()  # Allow p1 to receive SIGPIPE if p2 exits

            # Process 3: xauth nmerge - merge modified entries into Docker xauth file
            p3 = subprocess.Popen(
                ["xauth", "-f", sanitized_xauth, "nmerge", "-"],
                stdin=p2.stdout,
                stderr=subprocess.PIPE,
                text=True,
            )
            if p2.stdout:
                p2.stdout.close()  # Allow p2 to receive SIGPIPE if p3 exits

            # Wait for final process to complete
            _, stderr = p3.communicate(timeout=10)
            returncode = p3.returncode

            # Log result for debugging
            if returncode != 0:
                logger.warning(f"xauth command failed with code {returncode}: {stderr}")
            else:
                logger.debug("xauth command succeeded")

            # Verify xauth file was created with content
            if os.path.exists(xauth_file):
                file_size = os.path.getsize(xauth_file)
                if file_size == 0:
                    logger.warning("xauth file created but empty - X11 display authorization may not work")
                else:
                    logger.debug(f"xauth file created successfully ({file_size} bytes)")
        except subprocess.TimeoutExpired:
            logger.warning("xauth command timed out after 10 seconds")
            # Clean up processes on timeout
            for proc in [p3, p2, p1]:
                try:
                    if proc is not None and proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=1)
                except Exception:
                    pass
        except Exception as xauth_err:
            logger.warning(f"xauth command failed: {xauth_err}")
        os.chmod(xauth_file, 0o660)
        logger.debug(f"Created X authority file for display {sanitized_display}")
    except Exception as e:
        logger.warning(f"Failed to create Docker X authority file: {e}")

    if os.path.exists(xauth_file):
        environment["XAUTHORITY"] = xauth_file
        volumes[xauth_file] = {"bind": xauth_file, "mode": "rw"}

    # GPU devices
    container_devices = ["/dev/dri:/dev/dri"]
    if os.path.exists("/dev/accel"):
        container_devices.append("/dev/accel:/dev/accel")

    # Build command - run parallel benchmark script
    # The script expects device names in canonical format (e.g., iGPU, dGPU.0, dGPU.1, CPU, NPU)
    # Convert OpenVINO device format to canonical format using centralized utility
    from esq.utils.media.validation import normalize_device_name

    device_args = [normalize_device_name(dev) for dev in devices]
    logger.debug(f"Normalized devices for benchmark: {devices} → {device_args}")

    # Display output: 0=fakesink (no display), 1=xvimagesink (display enabled)
    display_output = str(configs.get("display_output", 0))
    is_mtl = "true" if platform_info.get("is_mtl", False) else "false"
    has_igpu = "true" if platform_info.get("has_igpu", False) else "false"
    cfg_file_arg = config_file if config_file and config_file != "none" else "none"

    command = [
        "bash",
        "./run_proxy_pipeline_benchmark_parallel.sh",
        *device_args,
        display_output,
        is_mtl,
        has_igpu,
        cfg_file_arg,
        benchmark_script,
    ]

    logger.debug(f"Running container {container_name}")
    logger.debug(f"Command: {' '.join(command)}")
    logger.debug(f"Devices: {container_devices}")
    logger.debug(f"Environment: {environment}")

    try:
        # Use DockerClient.run_container in batch mode
        # Match old run_container.sh flags: privileged, net=host, ipc=host
        result = docker_client.run_container(
            name=container_name,
            image=f"{image_name}:{image_tag}",
            command=command,
            volumes=volumes,
            devices=container_devices,
            environment=environment,
            user="root:root",  # Run as root for GPU access
            group_add=[render_gid, user_gid],
            privileged=True,  # Required for GPU access and benchmarking
            network_mode="host",  # Required for inter-process communication
            ipc_mode="host",  # Required for shared memory
            working_dir="/home/dlstreamer",
            mode="batch",
            detach=True,  # Required for batch mode
            remove=False,  # Keep container for cleanup in finally block
            attach_logs=True,  # Attach logs to Allure report
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


@allure.title("Proxy Pipeline Benchmark Test")
def test_proxy_pipelines(
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
    suite_name = configs.get("pptest", "SmartAIRunner")
    force_run = configs.get("force_run", False)  # Control whether to force re-run even if results exist

    logger.info(f"Starting Proxy Pipelines Benchmark Runner: {test_display_name} (force_run={force_run})")

    dockerfile_name = configs.get("dockerfile_name", "Dockerfile")
    docker_image_tag = f"{configs.get('container_image', 'proxy_pl_bm_runner')}:{configs.get('image_tag', '1.0')}"
    timeout = int(configs.get("timeout", 300))
    base_image = configs.get("base_image", "intel/dlstreamer:2025.1.2-ubuntu24")
    devices = configs.get("devices", "igpu")

    # Setup
    test_dir = os.path.dirname(os.path.abspath(__file__))
    docker_dir = os.path.join(test_dir, test_container_path)
    logger.info(f"Docker directory: {docker_dir}")

    # Use CORE_DATA_DIR for results and resources: esq_data/data/vertical/vision/results/ppl
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    core_data_dir = "".join(c for c in core_data_dir_tainted)
    data_dir = os.path.join(core_data_dir, "data", "vertical", "vision")
    pp_results = os.path.join(data_dir, "results", "ppl")
    os.makedirs(pp_results, exist_ok=True)

    # Create resources directory for models and videos under esq_data (FW team recommendation)
    resources_dir = os.path.join(pp_results, "resources")
    os.makedirs(resources_dir, exist_ok=True)
    ensure_dir_permissions(resources_dir, uid=os.getuid(), gid=os.getgid(), mode=0o775)

    # Get test suite configuration and download resources
    test_suite, csvlist, models_path, videos_path = proxy_pl_suite(suite_name, resources_dir)

    # Ensure directories have correct permissions
    ensure_dir_permissions(pp_results, uid=os.getuid(), gid=os.getgid(), mode=0o775)

    # Step 1: Validate system requirements (CPU, memory, storage, Docker, etc.)
    # This will skip the test if requirements are not met (test framework decision)
    validate_system_requirements_from_configs(configs)

    # Step 2: Get available devices based on device categories
    logger.info(f"Configured device categories: {devices}")
    device_dict = get_available_devices_by_category(device_categories=devices)

    logger.debug(f"Available devices: {device_dict}")

    # Step 3: Handle missing hardware with N/A metrics (graceful failure, not skip)
    if not device_dict:
        logger.warning(
            f"Required {devices.upper()} hardware not available on this platform. "
            f"Test will complete with N/A metrics (hardware requirement not met)."
        )

        # Create N/A result immediately since hardware is not available
        metrics = _create_proxy_pipeline_metrics(suite_name, value="N/A", unit="")

        results = Result.from_test_config(
            configs=configs,
            parameters={
                "timeout(s)": timeout,
                "display_name": test_display_name,
                "suite_name": suite_name,
                "devices": devices,
            },
            metrics=metrics,
            metadata={
                "status": "N/A",
                "failure_reason": (
                    f"Required {devices.upper()} hardware not available on this platform. "
                    f"System does not meet hardware requirement: {devices}_required=true"
                ),
            },
        )

        # Summarize with N/A status
        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )

        # Fail test with clear message (ensures proper reporting)
        failure_msg = (
            f"Required {devices.upper()} hardware not available on this platform. "
            f"System does not meet hardware requirement: {devices}_required=true. "
            f"Test completed with N/A metrics."
        )
        logger.error(f"Test failed: {failure_msg}")
        pytest.fail(failure_msg)

    # Log detailed device information
    for device_id, device_info in device_dict.items():
        logger.debug(f"Device {device_id}: Type={device_info['device_type']}, Name={device_info['full_name']}")

    docker_client = DockerClient()

    # Initialize variables for finally block and exception handlers (moved to top for broader coverage)
    test_failed = False
    failure_message = ""
    results = None
    container_name = None  # Track container for cleanup on interrupts

    try:
        # Step 3: Prepare test environment
        def prepare_assets():
            # Access outer scope variables
            nonlocal base_image, docker_image_tag, dockerfile_name, docker_dir, timeout

            docker_nocache = configs.get("docker_nocache", False)
            logger.info(f"Docker build cache setting: nocache={docker_nocache}")

            # Validate platform and detect device types using Python validation
            logger.info("Validating platform configuration...")
            platform_info = detect_platform_type()
            logger.debug(
                f"Platform detection: iGPU={platform_info['has_igpu']}, "
                f"dGPU_count={platform_info['dgpu_count']}, MTL={platform_info['is_mtl']}"
            )

            # Copy consolidated utilities into Docker build context
            logger.info("Preparing Docker build context with consolidated utilities...")
            import shutil
            from pathlib import Path

            # Source paths (absolute paths relative to test file)
            test_file_dir = Path(__file__).resolve().parent
            # test_file_dir = .../src/esq/suites/vertical/vision/
            # parent.parent.parent = .../src/esq/
            esq_utils_src = test_file_dir.parent.parent.parent / "utils" / "media"

            # Destination paths (inside Docker build context)
            docker_build_path = Path(docker_dir)
            esq_utils_dst = docker_build_path / "esq_utils" / "media"

            # Create destination directories
            esq_utils_dst.mkdir(parents=True, exist_ok=True)

            # Copy ESQ utilities (including container_utils.py for subprocess execution)
            util_files = [
                "__init__.py",
                "pipeline_utils.py",
                "telemetry.py",
                "validation.py",
                "container_utils.py",
            ]
            for util_file in util_files:
                src = esq_utils_src / util_file
                dst = esq_utils_dst / util_file
                if src.exists():
                    shutil.copy2(src, dst)
                    logger.debug(f"Copied {util_file} to build context")
                else:
                    logger.warning(f"Utility file not found: {src}")

            # Build arguments
            build_args = {
                "COMMON_BASE_IMAGE": f"{base_image}",
            }

            build_result = docker_client.build_image(
                path=docker_dir,
                tag=docker_image_tag,
                nocache=docker_nocache,
                dockerfile=dockerfile_name,
                buildargs=build_args,
            )

            container_config = {
                "image_id": build_result.get("image_id", ""),
                "image_tag": docker_image_tag,
                "timeout": timeout,
                "dockerfile": os.path.join(docker_dir, dockerfile_name),
                "build_path": docker_dir,
            }

            result = Result(
                metadata={
                    "status": True,
                    "container_config": container_config,
                    "Timeout (s)": timeout,
                    "Display Name": test_display_name,
                }
            )

            return result

    except KeyboardInterrupt:
        failure_message = (
            f"User interrupt (Ctrl+C) detected during Proxy Pipeline test preparation. "
            f"Test: {test_display_name}, Suite: {suite_name}, Devices: {devices}. "
            f"Partial setup may be incomplete."
        )
        test_interrupted = True
        logger.error(failure_message)
        # No containers running yet during preparation phase
        # Cleanup will be handled by finally block if needed
        raise  # Re-raise KeyboardInterrupt to propagate to caller

    except Exception as e:
        test_failed = True
        failure_message = (
            f"Unexpected error during test preparation: {type(e).__name__}: {str(e)}. "
            f"Test: {test_display_name}, Suite: {suite_name}, Docker image: {docker_image_tag}. "
            f"Check logs for full stack trace and error details."
        )
        logger.error(failure_message, exc_info=True)
        logger.debug(f"Preparation context - Docker dir: {docker_dir}, Base image: {base_image}")

        # Attach traceback to Allure report
        try:
            import traceback

            tb_str = traceback.format_exc()
            allure.attach(
                tb_str,
                name=f"Preparation Exception Traceback - {suite_name}",
                attachment_type=allure.attachment_type.TEXT,
            )
        except Exception as attach_error:
            logger.debug(f"Failed to attach traceback: {attach_error}")
        # Don't raise yet - create N/A result below

    try:
        prepare_test(test_name=test_name, configs=configs, prepare_func=prepare_assets, name="proxy_pl_assets")
    except Exception as prep_error:
        # Handle docker build or other preparation failures
        test_failed = True
        failure_message = (
            f"Test preparation failed during asset setup: {type(prep_error).__name__}: {str(prep_error)}. "
            f"Possible causes: Docker build failure, network issues, or dependency problems. "
            f"Docker image: {docker_image_tag}, Suite: {suite_name}. "
            f"Check logs for detailed error and verify Docker daemon is running."
        )
        logger.error(failure_message, exc_info=True)
        logger.debug(f"Preparation failed - Docker dir: {docker_dir}, Base image: {base_image}, Timeout: {timeout}s")

    # If preparation failed, return N/A metrics immediately
    if test_failed:
        metrics = _create_proxy_pipeline_metrics(suite_name, value="N/A", unit=None)

        results = Result.from_test_config(
            configs=configs,
            parameters={
                "timeout(s)": timeout,
                "display_name": test_display_name,
                "suite_name": suite_name,
                "devices": devices,
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

    # Initialize results template using from_test_config for automatic metadata application
    results = Result.from_test_config(
        configs=configs,
        parameters={
            "timeout(s)": timeout,
            "display_name": test_display_name,
            "suite_name": suite_name,
        },
    )

    try:

        def run_test():
            # Define metrics with N/A as initial values (unit will be set when value is populated)
            metrics = _create_proxy_pipeline_metrics(suite_name, value="N/A", unit=None)

            # Initialize result template using from_test_config for automatic metadata application
            result = Result.from_test_config(
                configs=configs,
                parameters={
                    "test_id": test_id,
                    "suite_name": suite_name,
                    "devices": devices,
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
                    f"No available devices found for configured device category: '{devices}'. "
                    f"Expected device types: {devices}. "
                    f"Verify hardware availability and driver installation (Intel GPU drivers required)."
                )
                logger.error(error_msg)
                logger.debug(f"Test configuration - device category: {devices}, display_name: {test_display_name}")
                result.metadata["failure_reason"] = error_msg
                return result

            # Log detailed device information and execute test for each device
            try:
                # Check if CSV results already exist for this specific test suite
                csv_exists = all(Path(f"{pp_results}/{csv_file}").exists() for csv_file in csvlist)

                # Determine if we need to run the benchmark
                should_run_benchmark = not csv_exists or force_run

                if should_run_benchmark:
                    if force_run and csv_exists:
                        logger.info(f"Force run enabled. Re-running {suite_name} benchmark for all devices...")
                        # Clean up existing CSV files to start fresh
                        for csv_file in csvlist:
                            csv_path = Path(f"{pp_results}/{csv_file}")
                            if csv_path.exists():
                                logger.info(f"Removing existing CSV file: {csv_file}")
                                csv_path.unlink()
                    elif not csv_exists:
                        logger.info(f"CSV files not found. Running {suite_name} benchmark for all devices...")

                    # Run container using DockerClient API (replaces execute_shell_script)
                    logger.info(
                        f"Running {suite_name} benchmark using DockerClient. "
                        f"Testing devices: {list(device_dict.keys())}"
                    )

                    # Detect platform information
                    platform_info = detect_platform_type()
                    logger.debug(
                        f"Platform info: iGPU={platform_info['has_igpu']}, "
                        f"dGPU_count={platform_info['dgpu_count']}, MTL={platform_info['is_mtl']}"
                    )

                    # Initialize CSV files before running container
                    # The container's update_csv function requires CSV files to exist with headers
                    logger.info(f"Initializing CSV files for {suite_name}: {csvlist}")
                    _initialize_csv_files(pp_results, csvlist)

                    # Run container using helper function
                    try:
                        # Pass resource paths through configs for volume mounting
                        configs_with_paths = {**configs, "_models_path": models_path, "_videos_path": videos_path}

                        container_name = f"{configs.get('container_image', 'proxy_pl_bm_runner')}_{suite_name}"
                        container_result = _run_proxy_pipeline_container(
                            docker_client=docker_client,
                            container_name=container_name,
                            image_name=configs.get("container_image", "proxy_pl_bm_runner"),
                            image_tag=configs.get("image_tag", "1.0"),
                            benchmark_script=test_suite,
                            output_dir=pp_results,
                            devices=list(device_dict.keys()),
                            platform_info=platform_info,
                            configs=configs_with_paths,
                            config_file=None,
                            round_num=1,
                            fps=30,
                        )

                        # Check if container execution failed
                        exit_code = container_result.get("container_info", {}).get("exit_code", 1)
                        if exit_code != 0:
                            error_msg = (
                                f"Container execution failed for suite '{suite_name}' (exit code: {exit_code}). "
                                f"Container: {container_name}. "
                                f"Check attached container logs for error details."
                            )
                            logger.error(error_msg)
                            result.metadata["failure_reason"] = error_msg
                            result.metadata["status"] = "N/A"
                            return result
                    except Exception as container_err:
                        error_msg = (
                            f"Failed to run container for suite '{suite_name}': {type(container_err).__name__}: {str(container_err)}. "
                            f"Check logs for detailed error information."
                        )
                        logger.error(error_msg, exc_info=True)
                        result.metadata["failure_reason"] = error_msg
                        result.metadata["status"] = "N/A"
                        return result

                else:
                    logger.info(
                        f"CSV results already exist for {suite_name}. Skipping benchmark execution (use force_run=True to override)."
                    )

                    # Log which devices have results in the CSV
                    for device_id, device_info in device_dict.items():
                        logger.info(
                            f"Device {device_id}: Type={device_info['device_type']}, Name={device_info['full_name']}"
                        )

                # Process CSV files to extract metrics
                for csv_filename in csvlist:
                    csv_file_path = Path(f"{pp_results}/{csv_filename}")
                    if csv_file_path.exists():
                        logger.info(f"Processing CSV file: {csv_filename}")

                        try:
                            import pandas as pd

                            # Try reading CSV with header first (default pandas behavior)
                            has_header = False
                            df = None
                            try:
                                df = pd.read_csv(csv_file_path)
                                # Check if dataframe has data rows (not just header)
                                if not df.empty and len(df.columns) > 3:
                                    # Successfully read with headers and has data
                                    has_header = True
                                    logger.debug(f"CSV read with headers. Columns: {list(df.columns)}, Rows: {len(df)}")
                                elif df.empty:
                                    # CSV has only headers, no data rows
                                    logger.warning(f"CSV file contains only headers, no data rows: {csv_filename}")
                                    continue  # Skip to next CSV file
                                else:
                                    # Insufficient columns, try without header
                                    df = pd.read_csv(csv_file_path, header=None)
                                    logger.debug("CSV read without headers (insufficient columns)")
                            except Exception as e:
                                # Failed to read with header, try without
                                logger.debug(f"Failed to read CSV with header, trying without: {e}")
                                try:
                                    df = pd.read_csv(csv_file_path, header=None)
                                    has_header = False
                                except Exception as e2:
                                    logger.error(f"Failed to read CSV {csv_filename} in any format: {e2}")
                                    continue  # Skip to next CSV file

                            if df is not None and not df.empty:
                                # Normalize column names (strip spaces) - CSV has inconsistent spacing
                                if has_header:
                                    df.columns = [col.strip() for col in df.columns]
                                    logger.debug(f"Normalized column names: {list(df.columns)}")

                                # Select best result based on max_streams (highest AI Channels value)
                                if has_header:
                                    # CSV has header - use column names
                                    # Find row with maximum AI Channels (best device)
                                    best_row_idx = 0
                                    if "AI Channels" in df.columns:
                                        # Convert to numeric, handling N/A values
                                        df["AI Channels_numeric"] = pd.to_numeric(
                                            df["AI Channels"], errors="coerce"
                                        ).fillna(0)
                                        best_row_idx = df["AI Channels_numeric"].idxmax()
                                        logger.info(
                                            f"Found {len(df)} result rows. Selecting best device "
                                            f"(row {best_row_idx}) based on max AI Channels"
                                        )

                                    best_row = df.iloc[best_row_idx]

                                    logger.debug(f"CSV columns: {list(df.columns)}")
                                    logger.debug(f"Best row data: {best_row.to_dict()}")

                                    # Try to extract metrics by column name
                                    try:
                                        # Extract max streams (AI Channels = max concurrent streams tested)
                                        if "AI Channels" in df.columns:
                                            result.metrics["max_streams"].value = (
                                                int(best_row["AI Channels"])
                                                if str(best_row["AI Channels"]) != "N/A"
                                                else 0
                                            )
                                            result.metrics["max_streams"].unit = "streams"
                                            logger.info(f"Extracted max_streams={result.metrics['max_streams'].value}")
                                        elif "Streams" in df.columns:
                                            # LPR CSV uses "Streams" column instead of "AI Channels"
                                            result.metrics["max_streams"].value = (
                                                int(best_row["Streams"])
                                                if str(best_row["Streams"]) not in ["N/A", "nan", ""]
                                                else 0
                                            )
                                            result.metrics["max_streams"].unit = "streams"
                                            logger.info(
                                                f"Extracted max_streams={result.metrics['max_streams'].value} "
                                                f"from Streams column"
                                            )
                                        elif "Input Channels" in df.columns:
                                            result.metrics["max_streams"].value = int(best_row["Input Channels"])
                                            result.metrics["max_streams"].unit = "streams"
                                            logger.info(f"Extracted max_streams={result.metrics['max_streams'].value}")
                                        elif "Total_Streams" in df.columns:
                                            result.metrics["max_streams"].value = int(best_row["Total_Streams"])
                                            result.metrics["max_streams"].unit = "streams"
                                            logger.info(f"Extracted max_streams={result.metrics['max_streams'].value}")
                                        else:
                                            logger.warning(
                                                "No AI Channels/Streams/Input Channels/Total_Streams "
                                                "column found in CSV"
                                            )

                                        # Extract avg FPS from log file (not from CSV)
                                        # Map suite names to log file prefixes (must match FILE_PREFIX in run_container.sh)
                                        suite_to_log_prefix = {
                                            "LPRAIRunner": "lpr",
                                            "SmartAIRunner": "smart_nvr",
                                            "VisualAIRunner": "headed_visual_ai",
                                            "VsaasAIRunner": "ai_vsaas",
                                        }
                                        log_prefix = suite_to_log_prefix.get(
                                            suite_name, suite_name.lower().replace("airunner", "")
                                        )
                                        log_filename = f"{log_prefix}_proxy_pipeline_runner.log"
                                        log_file_path = Path(pp_results) / log_filename

                                        if log_file_path.exists():
                                            try:
                                                # Read log file and extract FPS
                                                with open(
                                                    log_file_path, "r", encoding="utf-8", errors="ignore"
                                                ) as log_f:
                                                    log_lines = log_f.readlines()

                                                # Pattern: "[INFO] LPR Average fps is X" or
                                                # "[INFO] Average fps is X"
                                                import re

                                                fps_pattern = (
                                                    r"\[INFO\]\s+(?:LPR\s+)?"
                                                    r"Average\s+fps\s+is\s+([\d.]+)"
                                                )

                                                # Find all FPS values in log
                                                fps_values = []
                                                for line in log_lines:
                                                    match = re.search(fps_pattern, line, re.IGNORECASE)
                                                    if match:
                                                        fps_values.append(float(match.group(1)))

                                                # Use last FPS (final/best config)
                                                if fps_values:
                                                    result.metrics["avg_fps"].value = fps_values[-1]
                                                    result.metrics["avg_fps"].unit = "fps"
                                                    logger.info(
                                                        f"Extracted avg_fps={fps_values[-1]} "
                                                        f"from {log_filename} (found "
                                                        f"{len(fps_values)} entries)"
                                                    )
                                                else:
                                                    logger.warning(f"No FPS in log: {log_filename}")
                                            except Exception as log_err:
                                                logger.warning(f"Failed to extract FPS from {log_filename}: {log_err}")
                                        else:
                                            logger.warning(f"Log file not found: {log_file_path}")

                                        # Extract telemetry metrics from CSV
                                        # Actual CSV format has: GPU Freq, Pkg Power (from container output)
                                        # Note: Container doesn't provide CPU/Mem/GPU Util metrics yet

                                        # GPU Frequency (MHz)
                                        if "GPU Freq" in df.columns:
                                            gpu_freq_val = (
                                                float(best_row["GPU Freq"])
                                                if str(best_row["GPU Freq"]) not in ["N/A", "nan", "", "-1"]
                                                else 0.0
                                            )
                                            if gpu_freq_val > 0:
                                                result.metrics["gpu_freq_mhz"].value = gpu_freq_val
                                                result.metrics["gpu_freq_mhz"].unit = "MHz"
                                                logger.info(
                                                    f"Extracted GPU Freq={result.metrics['gpu_freq_mhz'].value} MHz"
                                                )

                                        # Package Power (Watts)
                                        if "Pkg Power" in df.columns:
                                            pkg_power_val = (
                                                float(best_row["Pkg Power"])
                                                if str(best_row["Pkg Power"]) not in ["N/A", "nan", "", "-1"]
                                                else 0.0
                                            )
                                            if pkg_power_val > 0:
                                                result.metrics["pkg_power_w"].value = pkg_power_val
                                                result.metrics["pkg_power_w"].unit = "W"
                                                logger.info(
                                                    f"Extracted Pkg Power={result.metrics['pkg_power_w'].value} W"
                                                )

                                        # Duration
                                        if "Duration(s)" in df.columns:
                                            result.metrics["duration_s"].value = (
                                                float(best_row["Duration(s)"])
                                                if str(best_row["Duration(s)"]) not in ["N/A", "nan", ""]
                                                else 0.0
                                            )
                                            result.metrics["duration_s"].unit = "s"
                                            logger.info(f"Extracted duration_s={result.metrics['duration_s'].value}")

                                        logger.info(f"Extracted metrics from CSV with headers: {csv_filename}")
                                    except Exception as e:
                                        logger.error(
                                            f"Failed to extract metrics by column name from {csv_filename}: {e}",
                                            exc_info=True,
                                        )
                                        logger.debug(f"CSV columns: {list(df.columns) if has_header else 'No header'}")
                                        logger.debug(f"CSV shape: {df.shape}")
                                else:
                                    # CSV without header - use positional indices
                                    # Note: This is a fallback - prefer using CSV with headers
                                    first_row = df.iloc[0]
                                    logger.warning(
                                        "CSV without headers detected. Using positional indices for metric extraction. "
                                        "This may be unreliable - CSV should have headers for proper parsing."
                                    )

                                    if suite_name in ["LPRAIRunner"]:
                                        # LPR format: column 4 = AI Channels (max_streams), column 5 = Avg FPS
                                        if len(first_row) > 5:
                                            result.metrics["max_streams"].value = int(first_row[4])
                                            result.metrics["max_streams"].unit = "streams"
                                            result.metrics["avg_fps"].value = float(first_row[5])
                                            result.metrics["avg_fps"].unit = "fps"
                                            logger.info(
                                                f"Extracted {suite_name} metrics: max_streams={first_row[4]}, avg_fps={first_row[5]}"
                                            )
                                    else:
                                        # Smart NVR, Visual AI, VSaaS format
                                        if len(first_row) > 8:
                                            result.metrics["max_streams"].value = int(first_row[4])
                                            result.metrics["max_streams"].unit = "count"
                                            # Column 8 contains AI Channels (max concurrent streams)
                                            result.metrics["max_streams"].value = (
                                                int(first_row[8]) if str(first_row[8]) != "N/A" else 0
                                            )
                                            result.metrics["max_streams"].unit = "streams"
                                        logger.info(f"Extracted metrics: max_streams={first_row[8]}")
                            else:
                                logger.warning(f"CSV file is empty: {csv_filename}")
                        except Exception as csv_error:
                            logger.error(f"Failed to parse CSV {csv_filename}: {csv_error}", exc_info=True)
                    else:
                        logger.warning(f"CSV file not found: {csv_file_path}")

                # Check if at least one CSV was found
                found_csv = any((Path(f"{pp_results}/{csv_f}").exists() for csv_f in csvlist))
                if not found_csv:
                    error_msg = (
                        f"Results CSV files not found at expected location: {pp_results}. "
                        f"Expected files: {', '.join(csvlist)}. "
                        f"Test container may have failed to generate results. "
                        f"Check container logs for execution errors."
                    )
                    logger.error(error_msg)
                    results_dir_contents = (
                        list(Path(pp_results).iterdir()) if Path(pp_results).exists() else "Directory not found"
                    )
                    logger.debug(f"Results directory contents: {results_dir_contents}")
                    result.metadata["failure_reason"] = "Results CSV files not generated by test container"
                    result.metadata["status"] = "N/A"
                    return result

                valid_metrics = [m for m in result.metrics.values() if m.value != "N/A"]
                if not valid_metrics:
                    metric_names = list(result.metrics.keys())
                    error_msg = (
                        f"Test completed but no valid metrics were collected (all N/A). "
                        f"Expected metrics: {', '.join(metric_names)}. "
                        f"CSV files were found but metric extraction failed. "
                        f"Verify CSV format matches expected structure."
                    )
                    logger.error(error_msg)
                    result.metadata["failure_reason"] = error_msg
                    result.metadata["status"] = "N/A"
                    return result

                # If successfully processed and collected valid metrics, mark as success
                result.metadata["status"] = True
                result.metadata.pop("failure_reason", None)  # Remove failure_reason if test succeeded

            except Exception as exec_error:
                # Handle any execution errors (shell script failures, CSV parsing, etc.)
                error_msg = (
                    f"Test execution failed with exception: {type(exec_error).__name__}: {str(exec_error)}. "
                    f"Suite: {suite_name}, Devices: {devices}. "
                    f"Check logs for stack trace and detailed error information."
                )
                logger.error(error_msg, exc_info=True)
                result.metadata["failure_reason"] = error_msg
                # Metrics remain as N/A
                return result

            logger.debug(f"Test results: {json.dumps(result.to_dict(), indent=2)}")

            return result

    except KeyboardInterrupt:
        failure_message = (
            f"User interrupt (Ctrl+C) detected during test execution. "
            f"Test: {test_display_name}, Suite: {suite_name}, Devices: {devices}. "
            f"Test execution was terminated before completion."
        )
        test_interrupted = True
        logger.error(failure_message)
        # Cleanup any running containers
        if container_name:
            try:
                logger.debug(f"Cleaning up container: {container_name}")
                docker_client.cleanup_container(container_name, timeout=10)
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
            f"Test: {test_display_name}, Suite: {suite_name}, Devices: {devices}. "
            f"Check logs for complete stack trace and error context."
        )
        logger.error(failure_message, exc_info=True)
        logger.debug(f"Execution context - Test ID: {test_id}, Suite: {suite_name}")

        # Attach traceback to Allure report
        try:
            import traceback

            tb_str = traceback.format_exc()
            allure.attach(
                tb_str,
                name=f"Execution Exception Traceback - {suite_name}",
                attachment_type=allure.attachment_type.TEXT,
            )
        except Exception as attach_error:
            logger.debug(f"Failed to attach traceback: {attach_error}")

    finally:
        # Cleanup: Ensure Docker containers are stopped/removed even if test fails or is interrupted
        if container_name:
            try:
                logger.debug(f"Finally block: Cleaning up container {container_name}")
                docker_client.cleanup_container(container_name, timeout=10)
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
            logger.info(f"Test summary - ID: {test_id}, Suite: {suite_name}, Devices: {devices}")
            logger.info(f"N/A metrics will be reported for {devices.upper()} (hardware not present)")
        else:
            # Actual test execution error - log as error
            logger.error(f"Test failed with N/A status: {failure_msg}")
            logger.info(f"Test summary - ID: {test_id}, Suite: {suite_name}, Devices: {devices}")

        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )

        pytest.fail(f"Proxy Pipeline test failed - {failure_msg}")

    # Validate test results against KPIs
    validate_test_results(results=results, configs=configs, get_kpi_config=get_kpi_config, test_name=test_name)

    try:
        logger.info(f"Generating test result visualizations (always executed) Results: {results}")

        # Summarize results using the shared fixture
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
        logger.debug(f"Summary context - Results dir: {pp_results}, Suite: {suite_name}")

    # Attach artifacts (CSV files and tar archives) for all devices - works for both success and failure cases
    try:
        for csvf in csvlist:
            csv_file_path = Path(f"{pp_results}/{csvf}")
            if csv_file_path.exists():
                try:
                    file_name = os.path.basename(csv_file_path)
                    with open(csv_file_path, "rb") as f:
                        allure.attach(
                            f.read(),
                            name=f"Proxy Pipeline Results - {file_name}",
                            attachment_type=allure.attachment_type.CSV,
                        )
                    logger.debug(f"Attached CSV file: {file_name}")

                    # NOTE: Inline chart generation DISABLED
                    # Charts are generated centrally at end via _generate_proxy_pipeline_charts()
                    # This avoids duplicate/conflicting chart generation

                except Exception as attach_error:
                    logger.warning(f"Failed to attach CSV {csvf}: {attach_error}")
            else:
                logger.debug(f"CSV file not found: {csv_file_path}")

        # Attach log files for all suites
        suite_to_log_prefix = {
            "LPRAIRunner": "lpr",
            "SmartAIRunner": "smart_nvr",
            "VisualAIRunner": "headed_visual_ai",
            "VsaasAIRunner": "ai_vsaas",
        }

        for suite_name_check, log_prefix in suite_to_log_prefix.items():
            log_filename = f"{log_prefix}_proxy_pipeline_runner.log"
            log_file_path = Path(pp_results) / log_filename

            if log_file_path.exists():
                try:
                    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                        allure.attach(
                            f.read(),
                            name=f"Proxy Pipeline Log - {suite_name_check}",
                            attachment_type=allure.attachment_type.TEXT,
                        )
                    logger.debug(f"Attached log file: {log_filename}")
                except Exception as log_attach_error:
                    logger.warning(f"Failed to attach log {log_filename}: {log_attach_error}")

        # Attach tar archive if it exists (for LPR suite)
        tar_file_path = Path(f"{pp_results}/lpr_proxy_pipeline_runner.tar.gz")
        if tar_file_path.exists():
            try:
                file_name = os.path.basename(tar_file_path)
                with open(tar_file_path, "rb") as f:
                    allure.attach(
                        f.read(), name=f"LPR Pipeline Archive - {file_name}", attachment_type="application/gzip"
                    )
                logger.debug(f"Attached tar archive: {file_name}")
            except Exception as attach_error:
                logger.warning(f"Failed to attach tar archive: {attach_error}")
        else:
            logger.debug(f"Tar archive not found: {tar_file_path}")

    except Exception as attach_error:
        logger.warning(f"Failed to attach artifacts: {type(attach_error).__name__}: {str(attach_error)}")

    # Generate and attach performance charts with multi-model support
    logger.info(f"Generating performance charts for {suite_name}...")
    _generate_proxy_pipeline_charts(pp_results=pp_results, suite_name=suite_name, configs=configs, logger=logger)

    if test_failed:
        pytest.fail(failure_message)
