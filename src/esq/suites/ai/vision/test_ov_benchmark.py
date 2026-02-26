# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from pathlib import Path

import allure
import pytest
from esq.utils.genutils import extract_csv_values, plot_grouped_bar_chart
from esq.utils.media import get_platform_identifier, get_vdbox_count_for_device, match_platform
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result
from sysagent.utils.infrastructure import DockerClient
from sysagent.utils.system.ov_helper import get_available_devices_by_category

logger = logging.getLogger(__name__)

test_container_path = "src/containers/openvino_benchmark/"


def normalize_device_name_for_csv(device_id: str, csv_file_path, logger) -> str:
    """
    Normalize device name from test framework format to CSV format.

    Framework uses: iGPU, GPU.0, GPU.1, dGPU.0, CPU, NPU
    CSV may contain: GPU, GPU.0, GPU.1, CPU, NPU

    Strategy: Try exact match first, then try common aliases.

    Args:
        device_id: Device name from get_available_devices_by_category
        csv_file_path: Path to CSV file for checking available devices (str or Path)
        logger: Logger instance

    Returns:
        Device name that exists in CSV, or original device_id if no match
    """
    import pandas as pd

    # Read CSV to see what device names are available
    try:
        df = pd.read_csv(str(csv_file_path))
        if "Device" not in df.columns:
            logger.warning(f"'Device' column not found in CSV. Available columns: {df.columns.tolist()}")
            return device_id
        available_devices = df["Device"].unique().tolist()
        logger.info(f"Device normalization - Input: '{device_id}', CSV has: {available_devices}")
    except Exception as e:
        logger.warning(f"Could not read CSV to check device names: {e}")
        return device_id

    # Try exact match first
    if device_id in available_devices:
        logger.debug(f"Device '{device_id}' found in CSV (exact match)")
        return device_id

    # Define alias mapping: test_device_name -> [csv_alias_1, csv_alias_2, ...]
    # Priority order: try most specific first (GPU.0) before generic (GPU)
    # Note: get_available_devices_by_category returns OpenVINO names: "GPU", "GPU.0", "GPU.1"
    device_aliases = {
        "GPU": ["GPU.0", "GPU"],  # OpenVINO returns "GPU" for igpu
        "iGPU": ["GPU.0", "GPU"],
        "GPU.0": ["GPU.0", "GPU"],
        "GPU.1": ["GPU.1", "GPU"],
        "dGPU.0": ["GPU.0", "GPU.1", "GPU"],
        "dGPU.1": ["GPU.1", "GPU.2", "GPU"],
    }
    # Try aliases if device has them
    if device_id in device_aliases:
        for alias in device_aliases[device_id]:
            if alias in available_devices:
                logger.info(f"Device '{device_id}' mapped to '{alias}' for CSV query")
                return alias
        logger.warning(f"Device '{device_id}' not found in CSV. Tried aliases: {device_aliases[device_id]}")

    # Return original if no match found
    logger.warning(f"Device '{device_id}' not found in CSV (no alias match)")
    return device_id


def _create_ov_metrics(value: str = "N/A", unit: str = None) -> dict:
    """
    Create OpenVINO benchmark metrics dictionary with lowercase metric names.

    Args:
        value: Initial value for all metrics (default: "N/A")
        unit: Unit for metrics (default: None for N/A values)

    Returns:
        Dictionary of Metrics objects with lowercase metric names matching CSV columns
    """
    return {
        "throughput": Metrics(unit=unit, value=value, is_key_metric=True),
        "latency": Metrics(unit=unit, value=value, is_key_metric=False),
        "dev_avg_freq": Metrics(unit=unit, value=value, is_key_metric=False),
        "package_power": Metrics(unit=unit, value=value, is_key_metric=False),
        "duration": Metrics(unit=unit, value=value, is_key_metric=False),
    }

# Mapping between Python metric names (lowercase) and CSV column names
# Note: Container CSV only provides: Throughput, Latency, Dev Freq, Pkg Power, Duration(s)
CSV_COLUMN_MAPPING = {
    "throughput": "Throughput",
    "latency": "Latency",
    "dev_avg_freq": "Dev Freq",  # Device frequency (CPU or GPU depending on device)
    "package_power": "Pkg Power",  # Package power from container
    "duration": "Duration(s)",  # Test duration in seconds
}

# Platform identification and VD box detection now use utility modules:
# - get_platform_identifier() from esq.utils.media.platform
# - get_vdbox_count_for_device() from esq.utils.media.gpu_topology
# - match_platform() from esq.utils.media.platform


def load_reference_benchmarks(bcmk_ref_path: Path) -> dict:
    """
    Load reference benchmark values from bcmk_ref.csv.

    Args:
        bcmk_ref_path: Path to bcmk_ref.csv file

    Returns:
        Dictionary with (model, precision, device) tuples as keys and
        (reference_value, reference_platform, reference_freq, ref_vdbox) tuples as values
    """
    reference_data = {}

    if not bcmk_ref_path.exists():
        logger.warning(f"Reference benchmark file not found: {bcmk_ref_path}")
        return reference_data

    try:
        import csv

        with open(bcmk_ref_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                model = row.get("Model", "").strip()
                precision = row.get("Precision", "").strip()
                device = row.get("Device", "").strip()
                ref_value_str = row.get("Reference Value", "").strip()
                ref_platform = row.get("Reference Platform", "").strip()
                ref_freq_str = row.get("Reference Freq", "").strip()

                # Parse reference value (throughput)
                try:
                    ref_value = float(ref_value_str) if ref_value_str else None
                except (ValueError, TypeError):
                    ref_value = None

                # Parse reference frequency
                try:
                    ref_freq = float(ref_freq_str) if ref_freq_str else None
                except (ValueError, TypeError):
                    ref_freq = None

                # Parse VD box count (optional column for GPU topology validation)
                ref_vdbox_str = row.get("VD_Boxes", "").strip()
                ref_vdbox = None
                if ref_vdbox_str:
                    try:
                        ref_vdbox = int(ref_vdbox_str)
                    except (ValueError, TypeError):
                        ref_vdbox = None

                # Store using (model, precision, device) as key
                key = (model, precision, device)
                reference_data[key] = (ref_value, ref_platform, ref_freq, ref_vdbox)

        logger.info(f"Loaded {len(reference_data)} reference benchmark entries from {bcmk_ref_path}")

    except Exception as e:
        logger.warning(f"Failed to load reference benchmarks from {bcmk_ref_path}: {e}")

    return reference_data


def lookup_reference_benchmark(
    reference_data: dict,
    model: str,
    precision: str,
    device: str,
    system_cpu: str | None = None,  # type: ignore
    system_memory_gb: int = 0,
    system_vdbox: int | None = None,  # type: ignore
) -> tuple:
    """
    Lookup reference benchmark for given model, precision, device, and system platform.

    This function prioritizes matches based on:
    1. Exact platform + device match (highest priority)
    2. Close platform + device match (CPU matches, memory close)
    3. Partial platform + device match (CPU substring match)
    4. Any reference for this device (fallback)

    Optional VD box validation (secondary scoring):
    - For GPU devices, validates GPU topology matches reference (VD box count)
    - Adds scoring bonus for matching topology, penalty for mismatch
    - Gracefully degrades if VD box info unavailable

    Args:
        reference_data: Dictionary from load_reference_benchmarks()
        model: Model name
        precision: Precision (e.g., "INT8", "FP16")
        device: Device identifier (e.g., "CPU", "GPU.0", "NPU")
        system_cpu: System CPU model (e.g., "i7-1360P") for platform matching
        system_memory_gb: System memory in GB for platform matching
        system_vdbox: System VD box count for GPU topology validation (optional)

    Returns:
        Tuple of (reference_value, reference_platform, reference_freq) or (None, None, None)
    """
    # Find all matching entries for this model, precision, and device
    candidates = []

    for key, value in reference_data.items():
        ref_model, ref_precision, ref_device = key
        if ref_model == model and ref_precision == precision and ref_device == device:
            # Unpack reference data (may have 3 or 4 elements depending on CSV format)
            if len(value) == 4:
                ref_value, ref_platform, ref_freq, ref_vdbox = value
            else:
                ref_value, ref_platform, ref_freq = value
                ref_vdbox = None

            # Calculate platform match score if system info provided
            if system_cpu and system_cpu != "Unknown":
                score = match_platform(
                    system_cpu,
                    system_memory_gb,
                    ref_platform,
                    device=device,
                    system_vdbox=system_vdbox,
                    ref_vdbox=ref_vdbox,
                )
            else:
                score = 1  # Default score for fallback

            candidates.append((score, ref_value, ref_platform, ref_freq))

    # Log all candidates for debugging
    if candidates and system_cpu and system_cpu != "Unknown":
        logger.debug(f"Platform matching for {model}/{precision}/{device}:")
        logger.debug(f"  System: {system_cpu} ({system_memory_gb}GB)")
        logger.debug(f"  Found {len(candidates)} candidates:")
        for score, ref_value, ref_platform, ref_freq in sorted(candidates, key=lambda x: x[0], reverse=True):
            logger.debug(f"    Score {score}: {ref_platform} ({ref_value} FPS)")

    # Sort by score (descending) and return best match
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_match = candidates[0]
        return (best_match[1], best_match[2], best_match[3])  # value, platform, freq

    return (None, None, None)


def update_csv_references_with_platform_matching(df, model: str, precision: str, test_container_path: str) -> None:
    """
    Update reference platform and values in CSV dataframe using platform-aware matching.

    This ensures CSV table uses the same reference lookup logic as the performance chart.
    Updates "Ref Platform", "Ref Throughput", and "Ref Dev Freq" columns in-place.

    Args:
        df: Pandas DataFrame containing test results from container
        model: Model name for fallback if not in CSV columns
        precision: Precision for fallback if not in CSV columns
        test_container_path: Path to container directory (for loading bcmk_ref.csv)
    """
    try:
        # Get system platform identifier
        system_cpu, system_memory_gb = get_platform_identifier()

        # Detect VD box count for first GPU device (if testing GPU)
        # This is used as secondary validation for GPU platform matching
        system_vdbox = None
        if any("GPU" in str(df.at[idx, "Device"]) for idx in df.index):
            system_vdbox = get_vdbox_count_for_device("GPU.0")
            if system_vdbox:
                logger.info(f"Detected {system_vdbox} VD box(es) for GPU topology validation")

        # Load reference benchmarks
        test_dir = os.path.dirname(os.path.abspath(__file__))
        bcmk_ref_path = Path(test_dir) / test_container_path / "bcmk_ref.csv"
        reference_data = load_reference_benchmarks(bcmk_ref_path)

        # Update reference columns for each row based on device
        # Note: Container CSV uses 'Ref Platform', 'Ref Throughput', 'Ref Dev Freq', 'Ref Pkg Power'
        if "Ref Platform" in df.columns and "Device" in df.columns:
            logger.info("Updating CSV table with platform-aware reference values...")

            for idx, row in df.iterrows():
                device_id = row["Device"]
                row_model = row.get("Model", model)
                row_precision = row.get("Precision", precision)

                # Lookup best matching reference for this device (includes VD validation for GPU)
                ref_value, ref_platform, ref_freq = lookup_reference_benchmark(
                    reference_data,
                    row_model,
                    row_precision,
                    device_id,
                    system_cpu=system_cpu,
                    system_memory_gb=system_memory_gb,
                    system_vdbox=system_vdbox,
                )

                if ref_value is not None:
                    # Update reference columns in dataframe
                    # Note: df.at[row_index, column_name] is valid pandas syntax for scalar assignment
                    # Container uses 'Ref Platform', 'Ref Throughput', 'Ref Dev Freq', 'Ref Pkg Power'
                    df.at[idx, "Ref Platform"] = ref_platform  # type: ignore
                    df.at[idx, "Ref Throughput"] = ref_value  # type: ignore
                    if "Ref Dev Freq" in df.columns and ref_freq is not None:
                        df.at[idx, "Ref Dev Freq"] = ref_freq  # type: ignore

                    logger.debug(
                        f"Updated CSV row {idx}: Device={device_id}, Ref Platform={ref_platform}, Ref Value={ref_value}"
                    )
                else:
                    logger.debug(f"No reference found for CSV row {idx}: {row_model}/{row_precision}/{device_id}")

            logger.info("CSV reference values updated successfully (now consistent with chart)")
        else:
            logger.debug("CSV does not contain 'Ref Platform' column - skipping reference update")

    except Exception as ref_update_error:
        logger.warning(
            f"Failed to update CSV reference values: {ref_update_error}. CSV will use container's hardcoded references."
        )


def get_chart_reference_for_devices(device_dict: dict, model: str, precision: str, test_container_path: str) -> tuple:
    """
    Get reference benchmark for chart generation based on tested devices.

    Finds the best matching reference across all tested devices using platform-aware matching.

    Args:
        device_dict: Dictionary of device_id -> device_info for tested devices
        model: Model name
        precision: Precision (e.g., "INT8", "FP16")
        test_container_path: Path to container directory (for loading bcmk_ref.csv)

    Returns:
        Tuple of (reference_throughput, reference_platform, reference_freq) or (None, "Reference Platform", None)
    """
    try:
        # Get system platform identifier for reference matching
        system_cpu, system_memory_gb = get_platform_identifier()
        logger.info(f"System platform: {system_cpu} with {system_memory_gb}GB memory")

        # Detect VD box count for GPU devices (secondary validation)
        devices = list(device_dict.keys())
        system_vdbox = None
        if any("GPU" in dev for dev in devices):
            system_vdbox = get_vdbox_count_for_device("GPU.0")

        # Load reference benchmarks from bcmk_ref.csv
        test_dir = os.path.dirname(os.path.abspath(__file__))
        bcmk_ref_path = Path(test_dir) / test_container_path / "bcmk_ref.csv"
        reference_data = load_reference_benchmarks(bcmk_ref_path)

        # Try to find best matching reference entry (platform-aware with VD validation)
        for device_id in device_dict.keys():
            ref_value, ref_platform, ref_freq = lookup_reference_benchmark(
                reference_data,
                model,
                precision,
                device_id,
                system_cpu=system_cpu,
                system_memory_gb=system_memory_gb,
                system_vdbox=system_vdbox,
            )
            if ref_value is not None:
                logger.info(
                    f"Found reference benchmark: {model}/{precision}/{device_id} = "
                    f"{ref_value} FPS on {ref_platform} "
                    f"(matched with system: {system_cpu} {system_memory_gb}GB)"
                )
                return (ref_value, ref_platform if ref_platform else "Reference Platform", ref_freq)

        logger.debug(
            f"No reference benchmark found for {model}/{precision} with devices {list(device_dict.keys())} "
            f"on platform {system_cpu} ({system_memory_gb}GB)"
        )
        return (None, "Reference Platform", None)

    except Exception as e:
        logger.warning(f"Failed to get chart reference: {e}")
        return (None, "Reference Platform", None)


@allure.title("OpenVINO Benchmark - Model Inference Performance")
def test_ov_benchmark(
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
    test_display_name = configs.get("display_name", test_name)

    logger.info(f"Starting Openvino Benchmark Runner: {test_display_name}")

    test_id = configs.get("test_id", test_name)
    duration_secs = float(configs.get("duration_secs", 0.15))
    dockerfile_name = configs.get("dockerfile_name", "Dockerfile")
    docker_image_tag = f"{configs.get('container_image', 'openvino_bm_runner')}:{configs.get('image_tag', '1.0')}"
    timeout = int(configs.get("timeout", 300))
    devices = configs.get("devices", "igpu")
    model = configs.get("model", "resnet-50-tf")
    precision = configs.get("precision", "INT8")

    # Setup
    test_dir = os.path.dirname(os.path.abspath(__file__))
    docker_dir = os.path.join(test_dir, test_container_path)

    # Use CORE_DATA_DIR for results: esq_data/data/vertical/metro/results/openvino
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    core_data_dir = "".join(c for c in core_data_dir_tainted)
    data_dir = os.path.join(core_data_dir, "data", "vertical", "metro")
    ov_results = os.path.join(data_dir, "results", "openvino")
    os.makedirs(ov_results, exist_ok=True)
    os.makedirs(f"{ov_results}/ov_results", exist_ok=True)

    # Clean up only THIS test's device-specific rows from CSV at the START
    # This preserves results for other devices/models/precisions
    csv_file_path = Path(f"{ov_results}/ov_results/ov_result_{model}.csv")
    cleaned_csv_path = Path(f"{ov_results}/ov_results/cleaned_ov_result_{model}.csv")

    # Remove device-specific rows from the main CSV
    if csv_file_path.exists():
        try:
            import pandas as pd

            df = pd.read_csv(csv_file_path)

            # Get device list to remove
            devices_to_remove = devices if isinstance(devices, list) else [devices]
            initial_rows = len(df)

            # Remove rows matching this model, precision, and any of the test devices
            df = df[~((df["Model"] == model) & (df["Precision"] == precision) & (df["Device"].isin(devices_to_remove)))]

            removed_rows = initial_rows - len(df)
            if removed_rows > 0:
                df.to_csv(csv_file_path, index=False)
                logger.info(f"Cleaned up {removed_rows} previous row(s) for {model}/{precision}/{devices}")

        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup CSV rows: {cleanup_error}")

    # Remove cleaned CSV and chart for this model (will be regenerated with all devices/precisions)
    for file_to_remove in [
        cleaned_csv_path,
        Path(f"{ov_results}/ov_results/chart_{model}.png"),  # Model-level chart (all precisions)
    ]:
        if file_to_remove.exists():
            try:
                file_to_remove.unlink()
                logger.debug(f"Removed previous file: {file_to_remove.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_to_remove.name}: {e}")

    logger.info(f"Starting fresh test run for model: {model}, precision: {precision}, devices: {devices}")

    # Ensure directories have correct permissions
    ensure_dir_permissions(ov_results, uid=os.getuid(), gid=os.getgid(), mode=0o775)

    # Initialize result template early (BEFORE validation) to ensure test info is available even if skipped
    # This ensures Metro CSV shows proper test name and N/A metrics instead of "unknown"
    metrics = _create_ov_metrics(value="N/A", unit=None)

    results = Result.from_test_config(
        configs=configs,
        parameters={
            "timeout(s)": timeout,
            "display_name": test_display_name,
            "device": devices,
        },
        metrics=metrics,
        metadata={
            "status": "N/A",
        },
    )

    # Step 1: Validate system requirements (Computation devices, memory, storage, Docker, etc.)
    try:
        validate_system_requirements_from_configs(configs)
    except (pytest.skip.Exception, pytest.fail.Exception):
        # Test is being skipped or failed due to requirements not met
        # Ensure result is summarized before re-raising
        results.metadata["failure_reason"] = "System requirements not met (validated before device check)"
        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
        raise  # Re-raise to preserve skip/fail behavior

    # Get available devices to check after validation
    logger.info(f"Configured device categories: {devices}")
    device_dict = get_available_devices_by_category(device_categories=devices)

    if not device_dict:
        # Convert devices list to string for display
        devices_str = ", ".join(devices) if isinstance(devices, list) else str(devices)
        logger.warning(
            f"Required {devices_str.upper()} hardware not available to test. "
            f"Test will complete with N/A metrics (hardware requirement not met)."
        )

        # Update existing results object with failure reason
        results.metadata["status"] = "N/A"
        results.metadata["failure_reason"] = (
            f"Required {devices_str.upper()} hardware not available to test. "
            f"Hardware requirement set to {devices}_required=true"
        )

        # Summarize with N/A status
        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )

        # This ensures the error message is displayed in summary and report overview
        failure_msg = (
            f"Required {devices_str.upper()} hardware not available on this platform. "
            f"Hardware requirement set to {devices}_required=true. "
            f"Test completed with N/A metrics."
        )
        logger.error(f"Test failed: {failure_msg}")
        pytest.fail(failure_msg)

    docker_client = DockerClient()

    # Initialize variables for finally block (moved to top for broader coverage)
    test_failed = False
    failure_message = ""

    try:
        # Step 2: Prepare test environment
        def prepare_assets():
            """Prepare OpenVINO benchmark assets including model download and container build."""
            from esq.utils.models.openvino_model_utils import download_openvino_model

            # Download model
            logger.info(f"Downloading model {model} with precision {precision}...")
            model_path = download_openvino_model(
                model_id=model, precision=precision, force_download=configs.get("force_download", False)
            )

            if not model_path:
                raise RuntimeError(
                    f"Failed to download model {model} ({precision}). "
                    f"Check network connection and HuggingFace accessibility."
                )

            logger.info(f"Model ready: {model_path}")

            # Build 1: Get FW custom device-specific images from dlstreamer preparation
            # This builds: DLS base → DLS + device drivers (NPU/dGPU/standard)
            from esq.suites.ai.vision.src.dlstreamer.preparation import (
                prepare_assets as prepare_dlstreamer_assets,
            )

            test_dir_abs = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.join(test_dir_abs, "src")
            models_dir = os.path.join(data_dir, "models")
            videos_dir = os.path.join(data_dir, "videos")

            logger.info("Build 1: Preparing FW custom device-specific DLStreamer images...")
            dlstreamer_result = prepare_dlstreamer_assets(
                configs=configs,
                models_dir=models_dir,
                videos_dir=videos_dir,
                src_dir=src_dir,
                docker_client=docker_client,
                docker_image_tag_analyzer="test-dlstreamer-analyzer:latest",
                docker_image_tag_utils="test-dlstreamer-utils:latest",
                docker_container_prefix="test",
            )

            fw_container_config = dlstreamer_result.metadata.get("container_config", {})
            logger.info(f"FW custom images available: {list(fw_container_config.keys())}")

            # Build 2: Select appropriate FW custom image based on test devices
            # Determine which device-specific base image to use
            fw_custom_base_image = None
            if isinstance(devices, list):
                # Check for NPU or dGPU in device list
                if any("npu" in str(d).lower() for d in devices):
                    fw_custom_base_image = fw_container_config.get("npu_analyzer_image")
                    logger.info(f"Using FW NPU custom image: {fw_custom_base_image}")
                elif any("dgpu" in str(d).lower() or "gpu." in str(d).lower() for d in devices):
                    # Check if it's actually discrete GPU
                    fw_custom_base_image = fw_container_config.get("dgpu_analyzer_image")
                    if fw_custom_base_image:
                        logger.info(f"Using FW dGPU custom image: {fw_custom_base_image}")
            elif isinstance(devices, str):
                if "npu" in devices.lower():
                    fw_custom_base_image = fw_container_config.get("npu_analyzer_image")
                    logger.info(f"Using FW NPU custom image: {fw_custom_base_image}")
                elif "dgpu" in devices.lower():
                    fw_custom_base_image = fw_container_config.get("dgpu_analyzer_image")
                    logger.info(f"Using FW dGPU custom image: {fw_custom_base_image}")

            # Fallback to standard analyzer image
            if not fw_custom_base_image:
                fw_custom_base_image = fw_container_config.get("analyzer_image", "intel/dlstreamer:2025.2.0-ubuntu24")
                logger.info(f"Using FW standard image: {fw_custom_base_image}")

            # Build Docker container for benchmark execution using FW custom image as base
            docker_nocache = configs.get("docker_nocache", False)
            logger.info(
                f"Build 2: Building test suite image '{docker_image_tag}' on top of FW custom image '{fw_custom_base_image}'..."
            )

            build_args = {
                "COMMON_BASE_IMAGE": fw_custom_base_image,  # FW custom device-specific image
            }

            build_result = docker_client.build_image(
                path=docker_dir,
                tag=docker_image_tag,
                nocache=docker_nocache,
                dockerfile=dockerfile_name,
                buildargs=build_args,  # Pass FW custom image to test Dockerfile
            )

            container_config = {
                "image_id": build_result.get("image_id", ""),
                "image_tag": docker_image_tag,
                "timeout": timeout,
                "dockerfile": os.path.join(docker_dir, dockerfile_name),
                "build_path": docker_dir,
                "model_path": str(model_path),
                "model_dir": str(model_path.parent),
            }

            result = Result(
                metadata={
                    "status": True,
                    "message": "OpenVINO Benchmark environment prepared",
                    "container_config": container_config,
                    "timeout (s)": timeout,
                    "display_name": test_display_name,
                    "model_path": str(model_path),
                }
            )

            return result

        # Call prepare_test within the same try block
        prepare_test(test_name=test_name, configs=configs, prepare_func=prepare_assets, name="ov_assets")

    except KeyboardInterrupt:
        failure_message = (
            f"User interrupt (Ctrl+C) detected during OpenVINO Benchmark test preparation. "
            f"Test: {test_display_name}, Devices: {devices}. "
            f"Partial setup may be incomplete."
        )
        logger.error(failure_message)
        test_failed = True
        # No containers running yet during preparation phase
        raise  # Re-raise KeyboardInterrupt to propagate to caller

    except Exception as e:
        test_failed = True
        failure_message = (
            f"Unexpected error during test preparation: {type(e).__name__}: {str(e)}. "
            f"Test: {test_display_name}, Devices: {devices}, Docker image: {docker_image_tag}. "
            f"Check logs for full stack trace and error details."
        )
        logger.error(failure_message, exc_info=True)
        logger.debug(f"Preparation context - Docker dir: {docker_dir}")

    # If preparation failed, update existing results and exit
    if test_failed:
        results.metadata["status"] = "N/A"
        results.metadata["failure_reason"] = failure_message

        # Summarize with N/A status and exit
        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
        pytest.fail(failure_message)

    try:

        def run_test():
            # Get devices from configs (available in closure scope)
            test_devices = configs.get("devices", "igpu")

            # Define metrics with N/A as initial values (unit will be set when value is populated)
            metrics = _create_ov_metrics(value="N/A", unit=None)

            # Initialize result template using from_test_config for automatic metadata application
            result = Result.from_test_config(
                configs=configs,
                parameters={
                    "test_id": test_id,
                    "device": test_devices,
                    "test_duration_secs": duration_secs,
                    "display_name": test_display_name,
                },
                metrics=metrics,
                metadata={
                    "status": "N/A",
                },
            )

            # Check if devices are available
            if not device_dict:
                # Convert devices list to string for display
                devices_str = ", ".join(test_devices) if isinstance(test_devices, list) else str(test_devices)
                error_msg = (
                    f"No available devices found for configured device category: '{devices_str}'. "
                    f"Expected device types: {devices_str}. "
                    f"Verify hardware availability and driver installation (Intel GPU drivers required)."
                )
                logger.error(error_msg)
                logger.debug(f"Test configuration - device category: {devices_str}, display_name: {test_display_name}")
                result.metadata["failure_reason"] = error_msg
                return result

            # Log detailed device information and execute test
            # Initialize variables for exception handler scope
            container_image = configs.get("container_image", "openvino_bm_runner")
            container_tag = configs.get("image_tag", "1.0")
            container_full_tag = f"{container_image}:{container_tag}"

            try:
                logger.info(f"Processing {len(device_dict)} device(s): {list(device_dict.keys())}")
                for device_id, device_info in device_dict.items():
                    logger.info(
                        f"Device {device_id}: Type={device_info['device_type']}, Name={device_info['full_name']}"
                    )

                    # Run container using docker_client.run_container() API (not shell script)
                    logger.info(f"Running OpenVINO Benchmark container: {container_full_tag}")

                    # Setup mount paths using CORE_DATA_DIR structure: esq_data/data/vertical/metro
                    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
                    core_data_dir = "".join(c for c in core_data_dir_tainted)
                    data_dir = Path(core_data_dir) / "data" / "vertical" / "metro"
                    models_dir = data_dir / "models"
                    images_dir = data_dir / "images"
                    videos_dir = data_dir / "videos"

                    # Build volume mounts
                    volumes = {
                        str(ov_results): {"bind": "/home/dlstreamer/output", "mode": "rw"},
                        "/sys/bus/pci": {"bind": "/sys/bus/pci", "mode": "ro"},
                        "/proc/cpuinfo": {"bind": "/proc/cpuinfo", "mode": "ro"},
                    }

                    # Add models directory mount if it exists
                    if models_dir.exists():
                        volumes[str(models_dir)] = {"bind": "/home/dlstreamer/share/models", "mode": "ro"}
                        logger.info(f"Mounting models: {models_dir} -> /home/dlstreamer/share/models")
                    else:
                        logger.warning(f"Models directory not found: {models_dir}")

                    # Add images directory mount if it exists
                    if images_dir.exists():
                        volumes[str(images_dir)] = {"bind": "/home/dlstreamer/share/images", "mode": "ro"}
                        logger.debug(f"Mounting images: {images_dir} -> /home/dlstreamer/share/images")

                    # Add videos directory mount if it exists
                    if videos_dir.exists():
                        volumes[str(videos_dir)] = {"bind": "/home/dlstreamer/share/videos", "mode": "ro"}
                        logger.debug(f"Mounting videos: {videos_dir} -> /home/dlstreamer/share/videos")

                    # Build device mounts for GPU access
                    devices = ["/dev/dri:/dev/dri"]
                    if Path("/dev/accel").exists():
                        devices.append("/dev/accel:/dev/accel")

                    # Auto-detect MEI devices
                    import glob

                    mei_devices = glob.glob("/dev/mei*")
                    for mei_dev in mei_devices:
                        devices.append(f"{mei_dev}:{mei_dev}")
                    if mei_devices:
                        logger.debug(f"Detected MEI devices: {mei_devices}")

                    # Get renderD128 group ID for permissions
                    try:
                        render_stat = os.stat("/dev/dri/renderD128")
                        render_gid = render_stat.st_gid
                        group_add = [str(render_gid)]
                    except Exception as e:
                        logger.warning(f"Failed to get renderD128 group ID: {e}")
                        group_add = []

                    # Build container command - pass arguments for benchmark run
                    container_name = f"{container_image}_{device_id}_{model}_{precision}"
                    command = [
                        "-m",
                        model,
                        "-d",
                        str(device_id),
                        "-p",
                        precision,
                        "-t",
                        str(int(duration_secs)),
                    ]

                    try:
                        # Use framework's docker_client.run_container() API with cap_add support
                        logger.debug("Container parameters:")
                        logger.debug(f"  - Image: {container_full_tag}")
                        logger.debug(f"  - Name: {container_name}")
                        logger.debug(f"  - Command: {command}")
                        logger.debug(f"  - Volumes: {volumes}")
                        logger.debug(f"  - Devices: {devices}")
                        logger.debug(f"  - Group add: {group_add}")
                        logger.debug(f"  - Timeout: {timeout}s")
                        logger.debug("  - Capabilities: ['PERFMON', 'SYS_ADMIN']")
                        logger.debug("  - User: root:root")

                        # Use framework's run_container API with cap_add support
                        # Note: Framework automatically fails test if container exits with non-zero code
                        # CRITICAL: remove=False to preserve container until results are copied via volume mount
                        try:
                            container_result = docker_client.run_container(
                                name=container_name,
                                image=container_full_tag,
                                command=command,
                                volumes=volumes,
                                devices=devices,
                                user="root:root",
                                group_add=group_add,
                                cap_add=["PERFMON", "SYS_ADMIN"],  # Required for performance monitoring
                                timeout=timeout,
                                mode="batch",  # Wait for container to complete
                                attach_logs=True,  # Automatically attach logs to Allure
                                detach=True,  # Must be True for batch mode
                                remove=False,  # Keep container until results are read from volume
                            )
                        except (pytest.fail.Exception, Exception) as fail_ex:
                            # Handle container execution failures (non-zero exit code or other errors)
                            # Note: Framework may auto-fail on non-zero exit code
                            error_msg = (
                                f"Container execution failed: {str(fail_ex)}. "
                                f"Device: {device_id}, Container: {container_full_tag}. "
                                f"Check attached container logs for details."
                            )
                            logger.error(error_msg)
                            result.metadata["failure_reason"] = error_msg
                            result.metadata["status"] = "N/A"
                            continue

                        # Extract container info from result
                        container_info = container_result.get("container_info", {})
                        exit_code = container_info.get("exit_code", -1)

                        logger.debug(f"Container completed with exit code: {exit_code}")

                        logger.info("OpenVINO Benchmark execution completed successfully")

                    except KeyboardInterrupt:
                        # User interrupt during container execution
                        error_msg = (
                            f"User interrupt (Ctrl+C) during container execution for device '{device_id}'. "
                            f"Container: {container_name}, Test will be terminated and cleaned up."
                        )
                        logger.error(error_msg)
                        result.metadata["failure_reason"] = error_msg
                        result.metadata["status"] = "N/A"
                        # Set all metrics to -1 to indicate interrupted test
                        for metric_name in result.metrics.keys():
                            result.metrics[metric_name].value = -1
                            result.metrics[metric_name].unit = None
                        logger.info("-1 metrics will be reported (test interrupted before completion)")
                        # Let finally block handle cleanup, then re-raise to propagate interrupt
                        raise

                    except Exception as container_error:
                        error_msg = (
                            f"Container execution exception for device '{device_id}': "
                            f"{type(container_error).__name__}: {str(container_error)}. "
                            f"Container: {container_full_tag}, Model: {model}. "
                            f"Check if Docker image exists and GPU device is accessible."
                        )
                        logger.error(error_msg, exc_info=True)
                        logger.warning(
                            f"Skipping device {device_id} due to container error, continuing with remaining devices"
                        )
                        # Continue to next device instead of returning

                    finally:
                        # Always cleanup container, even on interrupts/exceptions
                        if container_name:
                            try:
                                logger.debug(f"Cleaning up container: {container_name}")
                                docker_client.cleanup_container(container_name, timeout=10)
                                logger.debug(f"Successfully cleaned up container: {container_name}")
                            except Exception as cleanup_err:
                                logger.warning(
                                    f"Container cleanup warning for {container_name}: {cleanup_err}. "
                                    "Container may need manual cleanup."
                                )

                # Strategy: Identify device with BEST throughput (max), use ALL metrics from that device
                csv_file_path = Path(f"{ov_results}/ov_results/ov_result_{model}.csv")
                logger.info(f"All devices processed. Extracting metrics from CSV: {csv_file_path}")

                if not csv_file_path.exists():
                    # CSV file not found - no devices successfully completed
                    error_msg = (
                        f"Results CSV file not found: {csv_file_path}. "
                        f"No devices completed successfully. "
                        f"Check container logs for execution errors."
                    )
                    logger.error(error_msg)
                    results_dir_contents = (
                        list(Path(f"{ov_results}/ov_results").iterdir())
                        if Path(f"{ov_results}/ov_results").exists()
                        else "Directory not found"
                    )
                    logger.debug(f"Results directory contents: {results_dir_contents}")
                    result.metadata["failure_reason"] = "No results CSV generated"
                    result.metadata["status"] = "N/A"
                    return result

                # CSV file exists - process it
                try:
                    import math
                    import pandas as pd

                    df = pd.read_csv(csv_file_path)
                    logger.debug(f"CSV columns: {list(df.columns)}")

                    if "Device" in df.columns:
                        devices_in_csv = df["Device"].unique().tolist()
                        logger.info(f"Found {len(devices_in_csv)} device(s) in CSV: {devices_in_csv}")
                    else:
                        logger.error("CSV missing 'Device' column")
                        result.metadata["failure_reason"] = "CSV format error: missing Device column"
                        result.metadata["status"] = "N/A"
                        return result

                    # Check for failures in Result column across all devices
                    failed_devices = []
                    if "Result" in df.columns:
                        for device_id in device_dict.keys():
                            device_rows = df[df["Device"] == device_id]
                            if not device_rows.empty:
                                result_status = device_rows["Result"].iloc[0]
                                if result_status and str(result_status).upper() == "FAIL":
                                    failed_devices.append(device_id)
                                    logger.error(f"Benchmark FAILED for {model} on {device_id}: {result_status}")
                                elif result_status and "ERROR" in str(result_status).upper():
                                    logger.warning(f"Benchmark error for {model} on {device_id}: {result_status}")

                    if failed_devices:
                        error_msg = (
                            f"Benchmark execution FAILED for device(s): {', '.join(failed_devices)}. "
                            f"Check container logs for detailed error information."
                        )
                        logger.error(error_msg)
                        result.metadata["failure_reason"] = error_msg
                        result.metadata["status"] = "N/A"
                        return result

                    # Extract metrics from device with best throughput
                    # Strategy: Find device with MAX throughput, use all metrics from that device
                    logger.info(f"Extracting metrics for {len(device_dict)} device(s)")

                    # Step 1: Find device with best (highest) throughput
                    throughput_column = CSV_COLUMN_MAPPING.get("throughput", "throughput")
                    device_throughputs = {}

                    for device_id in device_dict.keys():
                        # Normalize device name for CSV query (iGPU -> GPU, GPU.0 -> GPU, etc.)
                        csv_device_name = normalize_device_name_for_csv(device_id, csv_file_path, logger)
                        throughput_val = extract_csv_values(csv_file_path, "Device", csv_device_name, throughput_column)
                        if throughput_val is not None and not (
                            isinstance(throughput_val, float) and math.isnan(throughput_val)
                        ):
                            device_throughputs[device_id] = throughput_val
                            logger.debug(f"Throughput for {device_id}: {throughput_val} FPS")

                    if not device_throughputs:
                        logger.error("No valid throughput values found for any device")
                        result.metadata["failure_reason"] = "No valid throughput data"
                        result.metadata["status"] = "N/A"
                        return result

                    # Find device with maximum throughput
                    best_device = max(device_throughputs, key=device_throughputs.get)
                    best_throughput = device_throughputs[best_device]
                    logger.info(f"Best device: {best_device} with throughput={best_throughput:.2f} FPS")
                    logger.info(
                        f"All device throughputs: "
                        f"{[(dev, round(tput, 2)) for dev, tput in device_throughputs.items()]}"
                    )

                    # Step 2: Extract ALL metrics from the best device
                    # Normalize device name for CSV query
                    csv_best_device = normalize_device_name_for_csv(best_device, csv_file_path, logger)
                    logger.debug(
                        f"Using CSV device name '{csv_best_device}' for metric extraction (from '{best_device}')"
                    )

                    for metric_name in metrics.keys():
                        csv_column = CSV_COLUMN_MAPPING.get(metric_name, metric_name)

                        val = extract_csv_values(csv_file_path, "Device", csv_best_device, csv_column)

                        # Determine unit
                        if metric_name == "throughput":
                            unit = "FPS"
                        elif metric_name == "latency":
                            unit = "ms"
                        elif "freq" in metric_name:
                            unit = "GHz"
                        elif "power" in metric_name:
                            unit = "W"
                        elif metric_name == "duration":
                            unit = "s"
                        else:
                            unit = None

                        if val is not None and not (isinstance(val, float) and math.isnan(val)):
                            # Convert 0 or 0.0 values to -1 (indicating failed/missing data)
                            # This handles cases where tools couldn't retrieve the metric
                            if val == 0.0:
                                result.metrics[metric_name].value = -1
                                # Skip unit for -1 values (avoid "-1 FPS" etc.")
                                result.metrics[metric_name].unit = None
                                logger.debug(
                                    f"Converted 0 to -1 for {metric_name} from {best_device} (data not available)"
                                )
                            else:
                                result.metrics[metric_name].value = round(val, 2)
                                result.metrics[metric_name].unit = unit
                                logger.info(f"Metric {metric_name} from {best_device}: {val:.2f} {unit}")
                        else:
                            result.metrics[metric_name].value = -1
                            # Skip unit for -1 values
                            result.metrics[metric_name].unit = None
                            logger.warning(f"No valid value for {metric_name} from {best_device}, setting to -1")

                    # Post-process CSV: Convert 0.0 values to -1 in the CSV file
                    # This ensures the CSV attached to reports shows -1 for missing/failed data
                    try:
                        df = pd.read_csv(csv_file_path)

                        # Convert 0.0 to -1 for all numeric metric columns (exclude Device, Model, Reference columns, etc.)
                        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
                        metric_cols = ["throughput", "latency", "gpu_freq", "cpu_util", "gpu_util", "power"]
                        # Exclude reference columns from 0->-1 conversion (they may legitimately be 0)
                        # Container uses: Ref Platform, Ref Throughput, Ref Dev Freq, Ref Pkg Power
                        reference_cols = ["Ref Throughput", "Ref Dev Freq", "Ref Pkg Power", "Duration(s)"]
                        for col in numeric_columns:
                            # Only convert 0.0 to -1 for metric columns, not reference columns
                            if (col in CSV_COLUMN_MAPPING.values() or col in metric_cols) and col not in reference_cols:
                                # Replace 0.0 with -1 for metric columns
                                df[col] = df[col].apply(lambda x: -1 if x == 0.0 or x == 0 else x)

                        # Save updated CSV back
                        try:
                            df.to_csv(csv_file_path, index=False)
                            logger.debug(f"Post-processed CSV: converted 0.0 values to -1 in {csv_file_path}")
                        except PermissionError as perm_error:
                            logger.warning(
                                f"Permission denied writing CSV (metrics already collected in memory, test continues): "
                                f"{csv_file_path}. Error: {perm_error}"
                            )
                        except Exception as write_error:
                            logger.warning(f"Failed to write updated CSV: {write_error}")
                    except Exception as csv_update_error:
                        logger.warning(f"Failed to post-process CSV for 0->-1 conversion: {csv_update_error}")

                except Exception as csv_error:
                    error_msg = (
                        f"Failed to process CSV results: "
                        f"{type(csv_error).__name__}: {str(csv_error)}. "
                        f"CSV file: {csv_file_path}"
                    )
                    logger.error(error_msg, exc_info=True)
                    result.metadata["failure_reason"] = error_msg
                    result.metadata["status"] = "N/A"
                    return result

                # Check if we have any valid metrics collected
                valid_metrics = [m for m in result.metrics.values() if m.value != "N/A"]
                if not valid_metrics:
                    metric_names = list(result.metrics.keys())
                    logger.warning(
                        f"No valid metrics collected from CSV. All metrics are 'N/A'. "
                        f"Metrics checked: {metric_names}"
                    )

                # If successfully processed all devices and collected valid metrics, mark as success
                valid_metrics = [k for k, v in result.metrics.items() if v.value != "N/A"]
                result.metadata["status"] = True
                result.metadata.pop("failure_reason", None)  # Remove failure_reason if test succeeded
                logger.info(
                    f"Test completed successfully for {len(device_dict)} device(s) "
                    f"with {len(valid_metrics)} valid metrics"
                )
                logger.debug(f"Final aggregated metrics: {[(k, v.value, v.unit) for k, v in result.metrics.items()]}")

            except Exception as exec_error:
                # Handle any execution errors (container execution, CSV parsing, etc.)
                # Get test_devices for error message
                devices_display = ", ".join(test_devices) if isinstance(test_devices, list) else str(test_devices)
                error_msg = (
                    f"Test execution failed with exception: {type(exec_error).__name__}: {str(exec_error)}. "
                    f"Device category: {devices_display}, Model: {model}, Duration: {duration_secs}s. "
                    f"Check logs for stack trace and detailed error information."
                )
                logger.error(error_msg, exc_info=True)
                logger.debug(f"Execution context - Results dir: {ov_results}, Model: {model}")
                result.metadata["failure_reason"] = error_msg
                # Metrics remain as N/A
                return result

            logger.debug(f"Test results: {json.dumps(result.to_dict(), indent=2)}")

            return result
    except KeyboardInterrupt:
        failure_message = (
            f"User interrupt (Ctrl+C) detected during test execution. "
            f"Test: {test_display_name}, Devices: {devices}. "
            f"Test execution was terminated before completion."
        )
        logger.error(failure_message)
        # Note: Metrics already set to -1 in inner KeyboardInterrupt handler
        # Container cleanup already handled in finally block
        raise  # Re-raise KeyboardInterrupt to propagate to caller

    except Exception as e:
        test_failed = True
        failure_message = (
            f"Unexpected error during test execution: {type(e).__name__}: {str(e)}. "
            f"Test: {test_display_name}, Devices: {devices}. "
            f"Check logs for complete stack trace and error context."
        )
        logger.error(failure_message, exc_info=True)
        logger.debug(f"Execution context - Test ID: {test_id}, Duration: {duration_secs}s")

    # Execute the test with shared fixture
    results = execute_test_with_cache(
        cached_result=cached_result,
        cache_result=cache_result,
        test_name=test_name,
        configs=configs,
        run_test_func=run_test,
    )

    # Log results summary for debugging
    logger.info(f"Test execution completed with status: {results.metadata.get('status')}")
    metrics_summary = {k: (v.value, v.unit) for k, v in results.metrics.items()}
    logger.debug(f"Results metrics summary: {metrics_summary}")

    # Handle N/A status (missing hardware or test failures)
    if results.metadata.get("status") == "N/A" and "failure_reason" in results.metadata:
        failure_msg = results.metadata["failure_reason"]

        # Check if failure is due to missing hardware (not a test execution error)
        is_hardware_missing = "No available devices found" in failure_msg

        if is_hardware_missing:
            devices_str = ", ".join(devices) if isinstance(devices, list) else str(devices)
            logger.error(f"Test failed - hardware not available: {failure_msg}")
            logger.info(f"Test summary - ID: {test_id}, Devices: {devices}, Duration: {duration_secs}s")
            logger.info(f"N/A metrics will be reported for {devices_str.upper()} (hardware not present)")
        else:
            # Actual test execution error - log as error
            logger.error(f"Test failed with N/A status: {failure_msg}")
            logger.info(f"Test summary - ID: {test_id}, Devices: {devices}, Duration: {duration_secs}s")

        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )

        pytest.fail(f"OpenVINO Benchmark test failed - {failure_msg}")

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
        device_keys = list(device_dict.keys()) if device_dict else "None"
        logger.debug(f"Summary context - Results dir: {ov_results}, Device dict: {device_keys}")

    # Attach CSV artifacts if available
    csv_file_path = Path(f"{ov_results}/ov_results/ov_result_{model}.csv")
    if csv_file_path.exists():
        try:
            # Read CSV for attaching to report
            import pandas as pd

            df = pd.read_csv(csv_file_path)

            # Update reference platform and values using platform-aware matching
            # This ensures CSV table matches the graph (both use same reference lookup logic)
            update_csv_references_with_platform_matching(df, model, precision, test_container_path)

            # Now fill any remaining NaN values in non-metric columns for display purposes
            # Do NOT fill metric columns - they should remain as -1 if data wasn't collected
            # Only fill text/categorical columns that might have NaN for display
            text_columns = ["Model", "Precision", "Device", "Result", "Ref Platform"]
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].fillna("")

            # Save cleaned CSV to a temporary location
            cleaned_csv_path = csv_file_path.parent / f"cleaned_{csv_file_path.name}"
            df.to_csv(cleaned_csv_path, index=False)

            # Attach the cleaned CSV
            file_name = os.path.basename(csv_file_path)
            with open(cleaned_csv_path, "rb") as f:
                allure.attach(f.read(), name=file_name, attachment_type=allure.attachment_type.CSV)
            logger.debug(f"Attached cleaned CSV results with updated references: {file_name}")

            # Generate and attach performance chart
            try:
                # Get reference benchmark for chart using platform-aware matching
                reference_throughput, reference_platform, _ = get_chart_reference_for_devices(
                    device_dict, model, precision, test_container_path
                )

                # Create chart output path - model-level chart showing all devices and precisions
                chart_path = cleaned_csv_path.parent / f"chart_{model}.png"

                # Generate grouped bar chart: Throughput vs Device, grouped by Precision
                # This chart shows comparison across all devices for all precisions of this model
                plot_grouped_bar_chart(
                    csv_path=cleaned_csv_path,
                    output_path=chart_path,
                    x_column="Device",
                    y_column="Throughput",
                    group_column="Precision",
                    title=f"{model} - Performance Across Devices and Precisions",
                    xlabel="Device",
                    ylabel="Throughput (FPS)",
                    reference_value=reference_throughput,
                    reference_label=f"Reference ({reference_platform})",
                    figsize=(12, 6),
                    rotation=0,
                )

                # Attach chart to Allure report
                if chart_path.exists():
                    with open(chart_path, "rb") as f:
                        allure.attach(
                            f.read(),
                            name=f"Performance Chart - {model}",
                            attachment_type=allure.attachment_type.PNG,
                        )
                    logger.info(f"Performance chart generated and saved: {chart_path}")
                    logger.info(f"Chart available at: {chart_path.absolute()}")

            except Exception as chart_error:
                logger.warning(f"Failed to generate performance chart: {chart_error}")

            logger.info(f"Cleaned CSV file saved: {cleaned_csv_path.absolute()}")
        except Exception as attach_error:
            logger.warning(f"Failed to attach CSV results: {attach_error}")
    else:
        logger.debug(f"CSV file not found: {csv_file_path}")

    # Consolidate and attach additional log files from result directory
    # Note: Docker container runtime logs are already attached by framework's DockerClient
    # This collects additional log files created by the benchmark container (e.g., telemetry logs)
    try:
        result_path = Path(ov_results)
        if result_path.exists() and result_path.is_dir():
            # Find all .log files in result directory
            log_files = sorted(result_path.rglob("*.log"))

            if log_files:
                logger.debug(f"Found {len(log_files)} log file(s) in result directory")

                # Build consolidated log content
                consolidated_logs = []
                consolidated_logs.append("=" * 80)
                consolidated_logs.append(f"CONSOLIDATED LOG FILES: {test_display_name}")
                consolidated_logs.append("=" * 80)
                consolidated_logs.append(f"Result Directory: {result_path}")
                consolidated_logs.append(f"Total Log Files: {len(log_files)}")
                consolidated_logs.append("")

                for log_file in log_files:
                    try:
                        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                            log_content = f.read()

                        relative_path = log_file.relative_to(result_path)
                        consolidated_logs.append(f"\n{'─' * 80}")
                        consolidated_logs.append(f"Log File: {relative_path}")
                        consolidated_logs.append(f"Size: {log_file.stat().st_size} bytes")
                        consolidated_logs.append(f"{'─' * 80}")
                        consolidated_logs.append(log_content)

                    except Exception as e:
                        consolidated_logs.append(f"\n[ERROR] Failed to read {log_file}: {e}")

                # Attach consolidated logs to Allure
                consolidated_content = "\n".join(consolidated_logs)
                allure.attach(
                    consolidated_content,
                    name=f"{test_display_name}_additional_logs.log",
                    attachment_type=allure.attachment_type.TEXT,
                )
                logger.info(f"Attached {len(log_files)} consolidated log file(s) to Allure report")
            else:
                logger.debug("No additional .log files found in result directory")

    except Exception as log_consolidation_error:
        logger.warning(f"Failed to consolidate result directory logs: {log_consolidation_error}")

    # Results directory preserved for viewing - will be cleaned on next test run
    logger.info(f"Test results preserved in: {ov_results}")

    if test_failed:
        pytest.fail(failure_message)

    logger.info(f"OpenVINO Benchmark test '{test_name}' completed successfully")
