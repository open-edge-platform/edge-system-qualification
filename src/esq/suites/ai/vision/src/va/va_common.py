# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Video Analytics (VA) Common Utilities.

This module provides shared utilities for VA test suites (light/medium/heavy).
These functions are used across all VA pipeline tests to avoid code duplication.

Common functionality:
- Metrics creation and initialization
- CSV file management
- Docker container execution
- Chart generation for reports
"""

import grp
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import allure
import pandas as pd
from sysagent.utils.core import Metrics, Result
from sysagent.utils.infrastructure import DockerClient

from esq.utils.genutils import plot_grouped_bar_chart
from esq.utils.media.validation import normalize_device_name

logger = logging.getLogger(__name__)

# Path to container sources relative to test files
# From src/esq/suites/ai/vision/src/va/ go to src/esq/suites/ai/vision/src/containers/video_analytics/
VA_CONTAINER_PATH = "../containers/video_analytics/"

# VA compute modes: (decode_device, detect_device, classify_device)
# Modes 0-6: Standard modes where stages run on specified devices
# Modes 7-8: Concurrent modes where GPU and NPU pipelines run simultaneously
VA_COMPUTE_MODES = {
    "Mode 0": ("CPU", "CPU", "CPU"),
    "Mode 1": ("dGPU", "dGPU", "dGPU"),
    "Mode 2": ("iGPU", "iGPU", "iGPU"),
    "Mode 3": ("iGPU", "iGPU", "NPU"),
    "Mode 4": ("iGPU", "NPU", "NPU"),
    "Mode 5": ("dGPU", "dGPU", "NPU"),
    "Mode 6": ("dGPU", "NPU", "NPU"),
    # Concurrent modes: GPU and NPU pipelines run simultaneously
    # Tuple format: (decode_device, detect_device, classify_device, concurrent_device)
    # concurrent_device indicates this is a concurrent mode with NPU
    "Mode 7": ("iGPU", "iGPU", "iGPU", "NPU_CONCURRENT"),  # iGPU + NPU concurrent
    "Mode 8": ("dGPU", "dGPU", "dGPU", "NPU_CONCURRENT"),  # dGPU + NPU concurrent
}

# CSV file names for VA benchmarks
VA_CSV_FILES = ["va_proxy_pipeline.csv"]


def create_va_metrics(value: str = "N/A", unit: Optional[str] = None) -> dict:
    """
    Create video analytics performance metrics dictionary.

    Args:
        value: Initial value for all metrics (default: "N/A")
        unit: Unit for metrics (default: None for N/A values)

    Returns:
        Dictionary of Metrics objects for video analytics tests
    """
    base_metrics = {
        "max_streams": Metrics(unit=unit, value=value, is_key_metric=True),
        "avg_fps": Metrics(unit=unit, value=value, is_key_metric=False),
        "gpu_freq_mhz": Metrics(unit=unit, value=value, is_key_metric=False),
        "pkg_power_w": Metrics(unit=unit, value=value, is_key_metric=False),
        "duration_s": Metrics(unit=unit, value=value, is_key_metric=False),
    }

    return base_metrics


def initialize_csv_files(pp_results: str, csvlist: list = None) -> None:
    """
    Initialize CSV files with headers if they don't exist.

    Args:
        pp_results: Path to results directory
        csvlist: List of CSV filenames to initialize (default: VA_CSV_FILES)
    """
    if csvlist is None:
        csvlist = VA_CSV_FILES

    # VA CSV header format
    va_header = (
        "TC Name,Model,Mode,Devices,Streams,Average FPS,"
        "GPU Freq,Pkg Power,"
        "Ref Platform,Ref FPS,Ref GPU Freq,Ref Pkg Power,"
        "Duration(s),Errors\n"
    )

    for csv_file in csvlist:
        csv_path = Path(pp_results) / csv_file
        if not csv_path.exists():
            try:
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write(va_header)
                logger.info(f"Initialized CSV file: {csv_file}")
            except Exception as e:
                logger.warning(f"Failed to initialize CSV file {csv_file}: {e}")


def prepare_docker_build_context(
    test_file_dir: Path,
    docker_dir: str,
    util_files: list = None,
) -> None:
    """
    Copy consolidated utilities into Docker build context.

    Args:
        test_file_dir: Path to the test file directory
        docker_dir: Docker build directory path
        util_files: List of utility files to copy (default: common utils)
    """
    if util_files is None:
        util_files = [
            "__init__.py",
            "pipeline_utils.py",
            "telemetry.py",
            "validation.py",
            "container_utils.py",
            "memory_utils.py",
        ]

    # Navigate from src/esq/suites/ai/vision/src/va/ to src/esq/utils/media/
    esq_utils_src = test_file_dir.parent.parent.parent.parent.parent / "utils" / "media"

    docker_build_path = Path(docker_dir)
    esq_utils_dst = docker_build_path / "esq_utils" / "media"
    esq_utils_dst.mkdir(parents=True, exist_ok=True)

    for util_file in util_files:
        src = esq_utils_src / util_file
        dst = esq_utils_dst / util_file
        if src.exists():
            shutil.copy2(src, dst)
            logger.debug(f"Copied {util_file} to build context")
        else:
            logger.warning(f"Utility file not found: {src}")


def setup_x11_display(environment: dict, volumes: dict) -> None:
    """
    Setup X11 display for Docker container visualization.

    Args:
        environment: Environment dictionary to update with DISPLAY settings
        volumes: Volumes dictionary to update with X11 mounts
    """
    # X11 Display setup for visualization (if enabled)
    display_raw = os.environ.get("DISPLAY", ":0")
    # Sanitize display value to prevent command injection
    sanitized_display = "".join(c for c in display_raw if c.isalnum() or c in ":.")
    environment["DISPLAY"] = sanitized_display

    # Xauthority file for display access
    xauth_file = "/tmp/.docker.xauth"
    if os.path.exists("/tmp/.X11-unix"):
        volumes["/tmp/.X11-unix"] = {"bind": "/tmp/.X11-unix", "mode": "rw"}

    # Create xauth file for X11 access if display is available
    if os.environ.get("DISPLAY"):
        try:
            import subprocess  # nosec B404 # For xauth pipeline

            # Generate xauth entry for Docker container
            p1 = subprocess.Popen(["xauth", "nlist", sanitized_display], stdout=subprocess.PIPE)
            p2 = subprocess.Popen(["sed", "-e", "s/^..../ffff/"], stdin=p1.stdout, stdout=subprocess.PIPE)
            p3 = subprocess.Popen(
                ["xauth", "-f", xauth_file, "nmerge", "-"],
                stdin=p2.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                _, stderr = p3.communicate(timeout=10)
                returncode = p3.returncode

                if returncode != 0:
                    logger.warning(f"xauth command failed with code {returncode}: {stderr}")
                else:
                    logger.debug("xauth command succeeded")

            except subprocess.TimeoutExpired:
                logger.warning("xauth command timed out after 10 seconds")
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


def run_va_container(
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
    device_info: dict = None,
    csv_filename: str = None,
) -> dict:
    """
    Run video analytics benchmark container using DockerClient API.

    Args:
        docker_client: DockerClient instance
        container_name: Name for the container
        image_name: Docker image name
        image_tag: Docker image tag
        benchmark_script: Name of benchmark script to run
        output_dir: Host directory for output files
        devices: List of device IDs to test
        platform_info: Platform detection info (is_mtl, has_igpu, dgpu_count)
        configs: Test configuration dictionary
        config_file: Optional custom config file path
        round_num: Round number for output files
        fps: Target FPS
        device_info: Optional dict of device info (device_id -> {device_type, full_name})
        csv_filename: Optional custom CSV filename (without path, e.g., "va_heavy_pipeline_VA-HEAVY-001.csv")

    Returns:
        Container execution result dictionary
    """
    # Get paths from configs (set by caller)
    models_path = configs.get("_models_path", os.path.join(output_dir, "resources", "models"))
    videos_path = configs.get("_videos_path", os.path.join(output_dir, "resources", "videos"))

    # Setup volume mounts
    volumes = {
        output_dir: {"bind": "/home/dlstreamer/output", "mode": "rw"},
        models_path: {"bind": "/home/dlstreamer/share/models", "mode": "ro"},
        videos_path: {"bind": "/home/dlstreamer/sample_video", "mode": "ro"},
    }

    # Setup environment variables
    environment = {}

    # Pass custom CSV filename if provided
    if csv_filename:
        environment["VA_CSV_FILENAME"] = csv_filename
        logger.debug(f"Using custom CSV filename: {csv_filename}")

    # Get render group GID for GPU access
    try:
        render_gid = str(grp.getgrnam("render").gr_gid)
    except KeyError:
        render_gid = "109"  # Default render group GID
    user_gid = str(os.getgid())

    # Setup X11 display
    setup_x11_display(environment, volumes)

    # GPU devices
    container_devices = ["/dev/dri:/dev/dri"]
    if os.path.exists("/dev/accel"):
        container_devices.append("/dev/accel:/dev/accel")

    # Normalize device names for benchmark
    # Use device_info if available to get proper device_type for normalization
    device_args = []
    for dev in devices:
        if device_info and dev in device_info:
            dev_type = device_info[dev].get("device_type", None)
            device_args.append(normalize_device_name(dev, dev_type))
        else:
            device_args.append(normalize_device_name(dev))
    logger.debug(f"Normalized devices for benchmark: {devices} → {device_args}")

    # Display output: 0=fakesink (no display), 1=xvimagesink (display enabled)
    display_output = str(configs.get("display_output", 0))
    is_mtl = "true" if platform_info.get("is_mtl", False) else "false"
    has_igpu = "true" if platform_info.get("has_igpu", False) else "false"
    cfg_file_arg = config_file if config_file and config_file != "none" else "none"

    command = [
        "bash",
        "./run_video_analytics_benchmark.sh",
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
        result = docker_client.run_container(
            name=container_name,
            image=f"{image_name}:{image_tag}",
            command=command,
            volumes=volumes,
            devices=container_devices,
            environment=environment,
            group_add=[render_gid, user_gid],
            network_mode="host",
            ipc_mode="host",
            working_dir="/home/dlstreamer",
            mode="batch",
            detach=True,
            remove=False,
            attach_logs=True,
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


def generate_va_charts(pp_results: str, configs: dict, test_logger=None, csv_filename: str = None) -> None:
    """
    Generate performance charts for video analytics benchmarks.

    Creates charts showing AI Channels/Streams vs Mode, grouped by device configuration.

    Args:
        pp_results: Path to directory containing CSV result files
        configs: Test configuration dictionary
        test_logger: Logger instance for debug/error messages (default: module logger)
        csv_filename: Optional specific CSV filename to use (e.g., "va_heavy_pipeline_VA-HEAVY-001.csv")
    """
    log = test_logger or logger

    try:
        # Determine CSV filename based on pipeline type or use provided filename
        if csv_filename:
            csv_path = Path(pp_results) / csv_filename
        else:
            # Fallback: detect pipeline from configs
            pipeline_type = configs.get("pipeline", "light")
            if pipeline_type == "heavy":
                csv_name = "va_heavy_pipeline.csv"
            elif pipeline_type == "medium":
                csv_name = "va_medium_pipeline.csv"
            else:
                csv_name = "va_proxy_pipeline.csv"
            csv_path = Path(pp_results) / csv_name

        if not csv_path.exists():
            log.debug(f"CSV file not found for video analytics: {csv_path}")
            return

        # Read CSV data
        df = pd.read_csv(csv_path)
        df = df.fillna(0.0)

        # Normalize column names
        df.columns = [col.strip() for col in df.columns]

        # Check required columns
        required_cols = ["Mode", "Streams"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            log.warning(f"Missing columns in VA CSV: {missing_cols}")
            log.debug(f"Available columns: {list(df.columns)}")
            return

        # Generate grouped bar chart
        chart_title = "Video Analytics - Streams by Mode"
        x_column = "Mode"
        y_column = "Streams"

        # Create chart data
        chart_data = df[[x_column, y_column]].copy()
        chart_data = chart_data.groupby(x_column)[y_column].max().reset_index()

        # Add reference values if available
        reference_values = {}
        if "Ref FPS" in df.columns:
            ref_df = df[[x_column, "Ref FPS"]].drop_duplicates()
            for _, row in ref_df.iterrows():
                mode = row[x_column]
                ref_val = row["Ref FPS"]
                if pd.notna(ref_val) and ref_val > 0:
                    reference_values[mode] = ref_val

        # Plot chart
        try:
            chart_path = Path(pp_results) / "va_performance_chart.png"

            # Save chart data to temporary CSV for plot function
            chart_csv_path = Path(pp_results) / "va_chart_data.csv"
            chart_data.to_csv(chart_csv_path, index=False)

            # Get first reference value if available (for single reference line)
            ref_value = list(reference_values.values())[0] if reference_values else None

            plot_grouped_bar_chart(
                csv_path=chart_csv_path,
                output_path=chart_path,
                x_column=x_column,
                y_column=y_column,
                group_column=x_column,  # Same as x_column when no grouping
                title=chart_title,
                xlabel="Compute Mode",
                ylabel="Stream Count",
                reference_value=ref_value,
                reference_label="Reference FPS" if ref_value else None,
            )

            # Attach to Allure report
            if chart_path.exists():
                with open(chart_path, "rb") as f:
                    allure.attach(
                        f.read(),
                        name="VA Performance Chart",
                        attachment_type=allure.attachment_type.PNG,
                    )
                log.info(f"Generated and attached performance chart: {chart_path}")

        except Exception as chart_error:
            log.warning(f"Failed to generate chart: {chart_error}")

    except Exception as e:
        log.warning(f"Failed to generate video analytics charts: {e}")


def determine_expected_modes(device_dict: dict) -> list:
    """
    Determine which compute modes should run based on available devices.

    Platform-specific behavior:
    - Xeon platforms: Include CPU mode (Mode 0)
    - Client platforms (Core i-series): Skip CPU, focus on iGPU/dGPU/NPU

    Args:
        device_dict: Dictionary of available devices

    Returns:
        List of expected mode names (e.g., ["Mode 1", "Mode 2"])

    Compute Modes:
        Mode 0: CPU/CPU/CPU (all stages on CPU)
        Mode 1: dGPU/dGPU/dGPU (all stages on dGPU)
        Mode 2: iGPU/iGPU/iGPU (all stages on iGPU)
        Mode 3: iGPU/iGPU/NPU (decode+detect on iGPU, classify on NPU)
        Mode 4: iGPU/NPU/NPU (decode on iGPU, detect+classify on NPU)
        Mode 5: dGPU/dGPU/NPU (decode+detect on dGPU, classify on NPU)
        Mode 6: dGPU/NPU/NPU (decode on dGPU, detect+classify on NPU)
        Mode 7: iGPU + NPU concurrent (GPU and NPU pipelines run simultaneously)
        Mode 8: dGPU + NPU concurrent (GPU and NPU pipelines run simultaneously)
    """
    # Detect if this is a Xeon platform
    is_xeon = False
    try:
        from sysagent.utils.system import collect_system_info

        system_info = collect_system_info()
        cpu_brand = system_info.get("hardware", {}).get("cpu", {}).get("brand", "")
        is_xeon = "xeon" in cpu_brand.lower()
        logger.debug(f"CPU brand: {cpu_brand}, is_xeon: {is_xeon}")
    except Exception as e:
        logger.warning(f"Failed to detect CPU type: {e}, assuming client platform")
        is_xeon = False

    available_device_types = set()
    for dev_id in device_dict.keys():
        if "GPU.0" in dev_id or dev_id == "iGPU":
            available_device_types.add("iGPU")
        elif "GPU.1" in dev_id or dev_id == "dGPU":
            available_device_types.add("dGPU")
        elif dev_id == "CPU":
            available_device_types.add("CPU")
        elif dev_id == "NPU":
            available_device_types.add("NPU")

    # Define expected modes based on device configuration
    if "CPU" in available_device_types and len(available_device_types) == 1:
        return ["Mode 0"]
    elif (
        "iGPU" in available_device_types
        and "dGPU" not in available_device_types
        and "NPU" not in available_device_types
    ):
        return ["Mode 2"]
    elif (
        "dGPU" in available_device_types
        and "iGPU" not in available_device_types
        and "NPU" not in available_device_types
    ):
        return ["Mode 1"]
    elif "iGPU" in available_device_types and "NPU" in available_device_types and "dGPU" not in available_device_types:
        # iGPU + NPU: split modes (3, 4) and concurrent mode (7)
        return ["Mode 3", "Mode 4", "Mode 7"]
    elif "dGPU" in available_device_types and "NPU" in available_device_types and "iGPU" not in available_device_types:
        # dGPU + NPU: split modes (5, 6) and concurrent mode (8)
        return ["Mode 5", "Mode 6", "Mode 8"]
    elif "iGPU" in available_device_types and "dGPU" in available_device_types and "NPU" not in available_device_types:
        # iGPU + dGPU: both GPU modes
        # Include CPU only on Xeon platforms
        if is_xeon:
            return ["Mode 0", "Mode 1", "Mode 2"]
        else:
            return ["Mode 1", "Mode 2"]
    elif "iGPU" in available_device_types and "dGPU" in available_device_types and "NPU" in available_device_types:
        # All devices available (iGPU + dGPU + NPU)
        # Include CPU mode (Mode 0) only on Xeon platforms
        if is_xeon:
            return ["Mode 0", "Mode 1", "Mode 2", "Mode 3", "Mode 4", "Mode 5", "Mode 6", "Mode 7", "Mode 8"]
        else:
            # Client platform: skip CPU, focus on GPU/NPU modes
            return ["Mode 1", "Mode 2", "Mode 3", "Mode 4", "Mode 5", "Mode 6", "Mode 7", "Mode 8"]
    else:
        # Fallback: include Mode 0 only on Xeon
        if is_xeon:
            return ["Mode 0", "Mode 1", "Mode 2", "Mode 3", "Mode 4", "Mode 5", "Mode 6", "Mode 7", "Mode 8"]
        else:
            return ["Mode 1", "Mode 2", "Mode 3", "Mode 4", "Mode 5", "Mode 6", "Mode 7", "Mode 8"]


def extract_metrics_from_csv(
    result: Result,
    csv_file_path: Path,
    expected_modes: list,
    log: logging.Logger = None,
) -> None:
    """
    Extract metrics from VA CSV file and update result object.

    Args:
        result: Result object to update with metrics
        csv_file_path: Path to CSV file
        expected_modes: List of expected mode names to filter by
        log: Logger instance
    """
    log = log or logger

    try:
        df = pd.read_csv(csv_file_path)
        df.columns = [col.strip() for col in df.columns]

        if df.empty:
            log.warning("CSV file is empty")
            return

        # Filter by expected modes for this test case
        if "Mode" in df.columns and expected_modes:
            df_filtered = df[df["Mode"].isin(expected_modes)]
            if df_filtered.empty:
                log.warning(f"No results found for expected modes {expected_modes}")
                df_filtered = df  # Fallback to all data
        else:
            df_filtered = df

        log.info(f"Processing {len(df_filtered)} rows for modes: {expected_modes}")

        # Find best result (max streams) within filtered data
        # Priority: 1) Maximum stream count, 2) Among ties, maximum Avg FPS
        if "Streams" in df_filtered.columns:
            df_filtered = df_filtered.copy()
            df_filtered["Streams_numeric"] = pd.to_numeric(df_filtered["Streams"], errors="coerce").fillna(0)
            
            # Find maximum stream count
            max_streams = df_filtered["Streams_numeric"].max()
            
            # Filter rows with max stream count
            max_stream_rows = df_filtered[df_filtered["Streams_numeric"] == max_streams]
            
            # If multiple rows have same max streams, pick the one with best Avg FPS
            if len(max_stream_rows) > 1 and "Average FPS" in max_stream_rows.columns:
                max_stream_rows["Avg_FPS_numeric"] = pd.to_numeric(max_stream_rows["Average FPS"], errors="coerce").fillna(0)
                best_row_idx = max_stream_rows["Avg_FPS_numeric"].idxmax()
                log.info(f"Multiple rows with {max_streams} streams found, selecting best Avg FPS")
            else:
                best_row_idx = max_stream_rows.index[0]
            
            best_row = df_filtered.loc[best_row_idx]

            # Extract metrics from the best row (max streams)
            result.metrics["max_streams"].value = int(best_row["Streams"])
            result.metrics["max_streams"].unit = "streams"
            best_mode = best_row.get("Mode", "Unknown")
            log.info(f"Extracted max_streams={result.metrics['max_streams'].value} from {best_mode}")

            # Debug log for zero streams to help diagnose system capability issues
            if result.metrics["max_streams"].value == 0:
                log.warning(
                    f"[ZERO_STREAMS_DETECTED] Benchmark completed with 0 streams. "
                    f"Mode: {best_mode}. This indicates the system cannot meet target FPS requirements. "
                    f"Possible causes: 1) Insufficient CPU/GPU/NPU performance, "
                    f"2) Missing or outdated drivers, 3) Insufficient memory, "
                    f"4) Target FPS too high for hardware capabilities. "
                    f"Review benchmark logs for detailed failure information."
                )

            # Extract avg_fps from the same best row to ensure sync
            if "Average FPS" in best_row:
                avg_fps_val = best_row["Average FPS"]
                invalid_vals = ["N/A", "nan", ""]
                avg_fps_float = float(avg_fps_val) if str(avg_fps_val) not in invalid_vals else 0.0
                if avg_fps_float > 0:
                    result.metrics["avg_fps"].value = avg_fps_float
                    result.metrics["avg_fps"].unit = "fps"
                    log.info(f"Extracted avg_fps={avg_fps_float} from best row ({best_mode})")

            if "GPU Freq" in df.columns:
                gpu_val = best_row["GPU Freq"]
                invalid_vals = ["N/A", "nan", "", "-1"]
                gpu_freq_val = float(gpu_val) if str(gpu_val) not in invalid_vals else 0.0
                if gpu_freq_val > 0:
                    result.metrics["gpu_freq_mhz"].value = gpu_freq_val
                    result.metrics["gpu_freq_mhz"].unit = "MHz"

            if "Pkg Power" in df.columns:
                pwr_val = best_row["Pkg Power"]
                invalid_vals = ["N/A", "nan", "", "-1"]
                pkg_power_val = float(pwr_val) if str(pwr_val) not in invalid_vals else 0.0
                if pkg_power_val > 0:
                    result.metrics["pkg_power_w"].value = pkg_power_val
                    result.metrics["pkg_power_w"].unit = "W"

            if "Duration(s)" in df.columns:
                dur_val = best_row["Duration(s)"]
                invalid_vals = ["N/A", "nan", ""]
                dur_float = float(dur_val) if str(dur_val) not in invalid_vals else 0.0
                result.metrics["duration_s"].value = dur_float
                result.metrics["duration_s"].unit = "s"

    except Exception as e:
        log.error(f"Failed to extract metrics from CSV: {e}", exc_info=True)


def extract_fps_from_log(result: Result, log_file_path: Path, log: logging.Logger = None) -> None:
    """
    Extract average FPS from VA benchmark log file.

    Args:
        result: Result object to update with FPS metric
        log_file_path: Path to log file
        log: Logger instance
    """
    log = log or logger

    if not log_file_path.exists():
        log.warning(f"Log file not found: {log_file_path}")
        return

    try:
        with open(log_file_path, "r", encoding="utf-8", errors="ignore") as log_f:
            log_lines = log_f.readlines()

        # Pattern 1: "[INFO] VA Benchmark Average fps is X"
        # Pattern 2: "[PROGRESS] Pipeline completed. Average fps is X"
        # Pattern 3: "Best Average FPS: X" (final result summary)
        fps_pattern1 = r"\[INFO\]\s+(?:VA\s+)?(?:Benchmark\s+)?Average\s+fps\s+is\s+([\d.]+)"
        fps_pattern2 = r"\[PROGRESS\].*Average\s+fps\s+is\s+([\d.]+)"
        fps_pattern3 = r"Best\s+Average\s+FPS:\s+([\d.]+)"

        fps_values = []
        best_fps = None

        for line in log_lines:
            # Try pattern 3 first (Best Average FPS - final result)
            match = re.search(fps_pattern3, line, re.IGNORECASE)
            if match:
                best_fps = float(match.group(1))
                continue

            # Try pattern 1 (INFO logs)
            match = re.search(fps_pattern1, line, re.IGNORECASE)
            if match:
                fps_values.append(float(match.group(1)))
                continue

            # Try pattern 2 (PROGRESS logs)
            match = re.search(fps_pattern2, line, re.IGNORECASE)
            if match:
                fps_values.append(float(match.group(1)))

        # Use Best Average FPS if available, otherwise use last FPS
        final_fps = best_fps if best_fps is not None else (fps_values[-1] if fps_values else None)

        if final_fps is not None and final_fps > 0:
            result.metrics["avg_fps"].value = final_fps
            result.metrics["avg_fps"].unit = "fps"
            source = "Best Average FPS" if best_fps is not None else "last FPS value"
            log.info(
                f"Extracted avg_fps={final_fps} from log file ({source}, found {len(fps_values)} total FPS entries)"
            )
        else:
            log.warning("No valid FPS found in log file")

    except Exception as e:
        log.warning(f"Failed to extract FPS from log: {e}")


def attach_va_artifacts(va_results: str, csvlist: list = None, log: logging.Logger = None) -> None:
    """
    Attach VA benchmark artifacts to Allure report.

    Args:
        va_results: Path to VA results directory
        csvlist: List of CSV files to attach
        log: Logger instance
    """
    log = log or logger

    if csvlist is None:
        csvlist = VA_CSV_FILES

    try:
        for csvf in csvlist:
            csv_file_path = Path(va_results) / csvf
            if csv_file_path.exists():
                try:
                    with open(csv_file_path, "rb") as f:
                        allure.attach(
                            f.read(),
                            name=f"Video Analytics Results - {csvf}",
                            attachment_type=allure.attachment_type.CSV,
                        )
                except Exception as attach_error:
                    log.warning(f"Failed to attach CSV {csvf}: {attach_error}")

        # Attach log file
        log_file_path = Path(va_results) / "va_proxy_pipeline_runner.log"
        if log_file_path.exists():
            try:
                with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    allure.attach(
                        f.read(),
                        name="VA Pipeline Log",
                        attachment_type=allure.attachment_type.TEXT,
                    )
            except Exception as log_attach_error:
                log.warning(f"Failed to attach log: {log_attach_error}")

    except Exception as e:
        log.warning(f"Failed to attach artifacts: {e}")
