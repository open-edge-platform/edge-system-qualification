# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import allure
import pytest
from esq.utils.genutils import (
    extract_csv_values,
)
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result
from sysagent.utils.infrastructure import DockerClient
from sysagent.utils.system.ov_helper import get_available_devices_by_category

logger = logging.getLogger(__name__)

test_container_path = "src/containers/ai_frequency_measure/"


def _create_gpu_metrics(device: str, value: Any = -1, unit: Any = None) -> dict:
    """
    Create GPU performance metrics dictionary with ALL possible metrics.

    Creates metrics for both iGPU and dGPU, with the tested device marked as key metric.
    Includes both average performance AND stability metrics (stddev, range) for frequency.
    This ensures Metro CSV shows -1 for non-tested device metrics instead of empty columns.

    Note: When metrics cannot be retrieved (0 values from gpu_top tool, test interrupts, or failures),
    they are reported as -1 with no unit to avoid confusing displays like "-1 %".

    Args:
        device: Device type being tested - "dgpu" or "igpu"
        value: Initial value for all metrics (default: -1 for missing/unavailable data)
        unit: Unit for metrics (default: None for -1 values, set appropriately for valid data)

    Returns:
        Dictionary of Metrics objects for ALL device types (both iGPU and dGPU)
    """
    # Determine which device metric is the key metric based on device being tested
    is_dgpu = device == "dgpu"

    return {
        # dGPU metrics - Performance
        "frequency_max_dgpu": Metrics(unit=unit, value=value, is_key_metric=is_dgpu),
        "utilization_dgpu": Metrics(unit=unit, value=value, is_key_metric=False),
        "max_power_dgpu": Metrics(unit=unit, value=value, is_key_metric=False),
        # dGPU metrics - Stability (lower is better for stability)
        "frequency_stddev_dgpu": Metrics(unit=unit, value=value, is_key_metric=False),
        "frequency_min_dgpu": Metrics(unit=unit, value=value, is_key_metric=False),
        "frequency_range_dgpu": Metrics(unit=unit, value=value, is_key_metric=False),
        # iGPU metrics - Performance
        "frequency_max_igpu": Metrics(unit=unit, value=value, is_key_metric=not is_dgpu),
        "utilization_igpu": Metrics(unit=unit, value=value, is_key_metric=False),
        "max_power_igpu": Metrics(unit=unit, value=value, is_key_metric=False),
        # iGPU metrics - Stability (lower is better for stability)
        "frequency_stddev_igpu": Metrics(unit=unit, value=value, is_key_metric=False),
        "frequency_min_igpu": Metrics(unit=unit, value=value, is_key_metric=False),
        "frequency_range_igpu": Metrics(unit=unit, value=value, is_key_metric=False),
        # CPU metrics (common to both)
        "utilization_cpu": Metrics(unit=unit, value=value, is_key_metric=False),
        "frequency_max_cpu": Metrics(unit=unit, value=value, is_key_metric=False),
    }

@allure.title("System GPU Performance Test (OpenVINO)")
def test_system_gpu(
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

    logger.info(f"Starting AI Frequency Runner: {test_display_name}")

    duration_hrs = int(configs.get("duration_hrs", 0.15))
    dockerfile_name = configs.get("dockerfile_name", "Dockerfile")
    docker_image_tag = (
        f"{configs.get('container_image', 'metro_dqt_ai_freq_measure')}:{configs.get('image_tag', '3.0')}"
    )
    timeout = int(configs.get("timeout", 300))
    base_image = configs.get("base_image", "intel/dlstreamer:2025.1.2-ubuntu24")
    device = configs.get("device", "igpu")
    # Setup
    test_dir = os.path.dirname(os.path.abspath(__file__))
    docker_dir = os.path.join(test_dir, test_container_path)

    # Use CORE_DATA_DIR for results: esq_data/data/vertical/metro/results/gpu
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    core_data_dir = "".join(c for c in core_data_dir_tainted)
    data_dir = os.path.join(core_data_dir, "data", "vertical", "metro")
    aifreq_results = os.path.join(data_dir, "results", "gpu")
    os.makedirs(aifreq_results, exist_ok=True)

    # Ensure directories have correct permissions
    ensure_dir_permissions(aifreq_results, uid=os.getuid(), gid=os.getgid(), mode=0o775)

    # Initialize result template early (BEFORE validation) to ensure test info is available even if skipped
    # This ensures Metro CSV shows proper test name and -1 metrics (indicating data not available)
    metrics = _create_gpu_metrics(device, value=-1, unit=None)

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
        },
    )

    # Step 1: Validate system requirements (Computation devices, memory, storage, Docker, etc.)
    # If validation fails, test will be skipped but result object already initialized above
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

    # Get available devices to check before validation
    logger.info(f"Configured device categories: {device}")
    device_dict = get_available_devices_by_category(device_categories=device)

    if not device_dict:
        logger.warning(
            f"Required {device.upper()} hardware not available to test. "
            f"Test will complete with -1 metrics (hardware requirement not met)."
        )

        # Update existing results object with failure reason
        results.metadata["status"] = "N/A"
        results.metadata["failure_reason"] = (
            f"Required {device.upper()} hardware not available to test. "
            f"Hardware requirement set to {device}_required=true"
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
            f"Required {device.upper()} hardware not available on this platform. "
            f"Hardware requirement set to {device}_required=true. "
            f"Test completed with -1 metrics (data not available)."
        )
        logger.error(f"Test failed: {failure_msg}")
        pytest.fail(failure_msg)

    docker_client = DockerClient()

    # Initialize variables for finally block (moved to top for broader coverage)
    # Note: results object already initialized above before validation
    test_failed = False
    test_interrupted = False
    failure_message = ""

    try:
        # Step 2: Prepare test environment
        def prepare_assets():
            # Access outer scope variables
            nonlocal base_image, docker_image_tag, dockerfile_name, docker_dir, timeout

            docker_nocache = configs.get("docker_nocache", False)
            logger.info(f"Docker build cache setting: nocache={docker_nocache}")

            # Download models and assets outside container
            logger.info("Downloading OpenVINO models for GPU benchmarking...")
            # Use CORE_DATA_DIR structure: esq_data/data/vertical/metro/models and images
            core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
            core_data_dir = "".join(c for c in core_data_dir_tainted)
            data_dir = os.path.join(core_data_dir, "data", "vertical", "metro")
            models_dir = os.path.join(data_dir, "models")
            images_dir = os.path.join(data_dir, "images")
            videos_dir = os.path.join(data_dir, "videos")
            results_dir = os.path.join(data_dir, "results", "gpu")  # GPU suite results

            # Create directories
            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(videos_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)

            logger.info(f"Using data directory: {data_dir}")

            try:
                from esq.utils.models.yolo_model_utils import (
                    download_test_image,
                    download_yolo_model,
                    export_yolo_model,
                )
                from sysagent.utils.system.hardware import collect_cpu_info

                # Detect CPU class and select appropriate model
                def select_model_for_cpu(config_models):
                    """
                    Select appropriate YOLO model based on CPU class.

                    - Atom/Celeron: Use YOLOv5n (nano - smallest, fastest)
                    - Core/Xeon: Use YOLOv5s (small - default)
                    - Config override: Honor user-specified model

                    Returns:
                        str: Selected model name
                    """
                    # If config explicitly specifies a model, use it
                    if config_models and config_models != ["yolov5s"]:
                        logger.info(f"Using model from config: {config_models}")
                        return config_models

                    # Auto-detect CPU class
                    try:
                        cpu_info = collect_cpu_info()
                        cpu_brand = cpu_info.get("brand", "Unknown").lower()

                        # Check for Atom or Celeron (low-power CPUs)
                        if "atom" in cpu_brand or "celeron" in cpu_brand:
                            logger.info(f"Detected low-power CPU: {cpu_info.get('brand')}")
                            logger.info("Auto-selecting YOLOv5n (nano variant) for optimal performance")
                            return ["yolov5n"]
                        else:
                            logger.info(f"Detected CPU: {cpu_info.get('brand')}")
                            logger.info("Using YOLOv5s (small variant) - standard model")
                            return ["yolov5s"]
                    except Exception as e:
                        logger.warning(f"Failed to detect CPU class: {e}")
                        logger.info("Defaulting to YOLOv5s")
                        return ["yolov5s"]

                # Get model list from test configuration with CPU-based auto-selection
                # Default: YOLOv5s for Core/Xeon, YOLOv5n for Atom/Celeron
                config_models = configs.get("models", ["yolov5s"])
                required_models = select_model_for_cpu(config_models)
                precision = configs.get("precision", "FP16")

                downloaded_models = {}

                # Download each required model
                for model_name in required_models:
                    logger.info(f"Downloading YOLO model: {model_name}")
                    try:
                        # Download PyTorch weights (stays in Ultralytics cache)
                        weights_path = download_yolo_model(model_name, models_dir=models_dir)
                        if not weights_path:
                            raise RuntimeError(f"Failed to download {model_name} weights")
                        logger.info(f"YOLO weights location: {weights_path}")

                        # Export to OpenVINO IR format in esq_data/data/vertical/metro/models directory
                        model_xml_path = export_yolo_model(
                            model_id=model_name,
                            models_dir=models_dir,
                            model_precision=precision.upper(),
                            weights_path=weights_path,
                        )
                        if not model_xml_path:
                            raise RuntimeError(f"Failed to export {model_name} to OpenVINO format")
                        logger.info(f"Exported to OpenVINO IR: {model_xml_path}")

                        # Store the directory path (parent of .xml file) for consistency
                        model_path = model_xml_path.parent
                    except Exception as e:
                        logger.error(f"Failed to download/export YOLO model: {e}", exc_info=True)
                        raise

                    downloaded_models[model_name] = model_path
                    logger.info(f"Model {model_name}: prepared at {model_path}")

                # Download test image to images directory
                test_image = download_test_image("car.png", output_dir=images_dir)
                if test_image:
                    downloaded_models["test_image"] = test_image
                else:
                    logger.warning("Failed to download test image, test may fail")

                logger.debug(f"Downloaded models: {list(downloaded_models.keys())}")

                # Log directory structure for verification
                logger.debug("=" * 60)
                logger.debug(f"Data directory structure: {data_dir}")
                for subdir in ["models", "images", "videos", "results"]:
                    subdir_safe = "".join(c for c in subdir)
                    subdir_path = os.path.join(data_dir, subdir_safe)
                    if os.path.exists(subdir_path):
                        items = os.listdir(subdir_path)
                        logger.debug(f"  {subdir}/: {len(items)} items - {items[:5]}")
                    else:
                        logger.warning(f"  {subdir}/: NOT FOUND")
                logger.debug("=" * 60)

            except Exception as model_error:
                logger.error(f"Failed to download models: {model_error}", exc_info=True)
                raise RuntimeError(f"Model preparation failed: {model_error}") from model_error

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
            f"User interrupt (Ctrl+C) detected during AI Frequency test preparation. "
            f"Test: {test_display_name}, Device: {device}. "
            f"Partial setup may be incomplete."
        )
        test_interrupted = True
        logger.error(failure_message)
        # No containers running yet during preparation phase

    except Exception as e:
        test_failed = True
        failure_message = (
            f"Unexpected error during test preparation: {type(e).__name__}: {str(e)}. "
            f"Test: {test_display_name}, Device: {device}, Docker image: {docker_image_tag}. "
            f"Check logs for full stack trace and error details."
        )
        logger.error(failure_message, exc_info=True)
        logger.debug(f"Preparation context - Docker dir: {docker_dir}, Base image: {base_image}")
        # Don't raise yet - create N/A result below

    try:
        prepare_test(test_name=test_name, configs=configs, prepare_func=prepare_assets, name="AI_Freq_Assets")
    except Exception as prep_error:
        # Handle docker build or other preparation failures
        test_failed = True
        failure_message = (
            f"Test preparation failed during asset setup: {type(prep_error).__name__}: {str(prep_error)}. "
            f"Possible causes: Docker build failure, model download failure, network issues, or dependency problems. "
            f"Docker image: {docker_image_tag}, Models: {configs.get('models', ['yolov5s'])}. "
            f"Check logs for detailed error and verify Docker daemon is running."
        )
        logger.error(failure_message, exc_info=True)
        logger.debug(f"Preparation failed - Docker dir: {docker_dir}, Base image: {base_image}, Timeout: {timeout}s")

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
            # Define metrics with -1 as initial values (indicating data not available)
            # Unit will be set when valid value is populated (unit=None for -1 to avoid "-1 %" display)
            metrics = _create_gpu_metrics(device, value=-1, unit=None)

            # Initialize result template using from_test_config for automatic metadata application
            result = Result.from_test_config(
                configs=configs,
                parameters={
                    "test_id": test_id,
                    "device": device,
                    "test_duration_hrs": duration_hrs,
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
            # Initialize variables for exception handler scope and cleanup tracking
            model_name = configs.get("models", ["yolov5s"])[0]
            container_image = configs.get("container_image", "ai_freq_measure")
            container_tag = configs.get("image_tag", "1.0")
            container_full_tag = f"{container_image}:{container_tag}"
            container_name = None  # Track container for cleanup on interrupts

            try:
                # Container will create averages_summary.csv with results
                # CSV columns (in order): Function, frequency_max_igpu, utilization_igpu, max_power_igpu,
                #                         frequency_max_dgpu, utilization_dgpu, max_power_dgpu,
                #                         utilization_cpu, frequency_max_cpu
                csv_file_path = Path(f"{aifreq_results}/averages_summary.csv")
                logger.info(f"Results will be written to: {csv_file_path}")

                # Get container configuration
                container_image = configs.get("container_image", "ai_freq_measure")
                container_tag = configs.get("image_tag", "1.0")
                container_name = f"{container_image}_{device}_{test_id}"
                container_full_tag = f"{container_image}:{container_tag}"

                # Setup mount paths
                cl_cache_dir = Path(os.getcwd()) / "cl_cache"
                cl_cache_dir.mkdir(parents=True, exist_ok=True)

                # Build volume mounts (including system paths for intel_gpu_top)
                volumes = {
                    str(aifreq_results): {"bind": "/home/dlstreamer/output", "mode": "rw"},
                    str(cl_cache_dir): {"bind": "/home/dlstreamer/cl_cache", "mode": "rw"},
                    "/sys/bus/pci": {"bind": "/sys/bus/pci", "mode": "ro"},  # PCI device info for GPU
                    "/proc/cpuinfo": {"bind": "/proc/cpuinfo", "mode": "ro"},  # CPU info
                }

                # Add data directory mount (models/images from esq_data/data/vertical/metro)
                if os.path.exists(data_dir):
                    volumes[str(data_dir)] = {"bind": "/home/dlstreamer/share", "mode": "ro"}
                    logger.info(f"Mounting data directory: {data_dir} -> /home/dlstreamer/share")
                else:
                    logger.warning(f"Data directory not found: {data_dir}")

                # Build device mounts for GPU access
                devices = ["/dev/dri:/dev/dri"]
                if Path("/dev/accel").exists():
                    devices.append("/dev/accel:/dev/accel")

                # Get renderD128 and video group IDs for GPU access permissions
                group_add = []
                try:
                    render_stat = os.stat("/dev/dri/renderD128")
                    render_gid = render_stat.st_gid
                    group_add.append(str(render_gid))
                    logger.debug(f"Adding render group: {render_gid}")
                except Exception as e:
                    logger.warning(f"Failed to get renderD128 group ID: {e}")

                # Also add video group if different from render
                try:
                    import grp

                    video_group = grp.getgrnam("video")
                    video_gid = video_group.gr_gid
                    if str(video_gid) not in group_add:
                        group_add.append(str(video_gid))
                        logger.debug(f"Adding video group: {video_gid}")
                except Exception as e:
                    logger.debug(f"Video group not found or not needed: {e}")

                # Build environment variables
                environment = {
                    "MODEL_NAME": model_name,
                }

                for device_id, device_info in device_dict.items():
                    logger.info(
                        f"Device {device_id}: Type={device_info['device_type']}, Name={device_info['full_name']}"
                    )

                    # Run container using docker_client API
                    logger.info(f"Running FreqRunner container: {container_full_tag}")

                    try:
                        # Run container with CAP_PERFMON and CAP_SYS_ADMIN for intel_gpu_top performance monitoring
                        # Kernel requirement: 5.8+ (CAP_PERFMON), older kernels may need only SYS_ADMIN
                        # Running as root for now - GPU telemetry requires root for full access
                        logger.debug(
                            f"Running container: {container_full_tag} with CAP_PERFMON and CAP_SYS_ADMIN as root"
                        )

                        logger.debug("Container parameters:")
                        logger.debug(f"  - Image: {container_full_tag}")
                        logger.debug(f"  - Name: {container_name}")
                        logger.debug(f"  - Command: [{device_id}, {duration_hrs}]")
                        logger.debug(f"  - Volumes: {volumes}")
                        logger.debug(f"  - Devices: {devices}")
                        logger.debug(f"  - Environment: {environment}")
                        logger.debug("  - Capabilities: ['PERFMON', 'SYS_ADMIN']")
                        logger.debug("  - User: root:root")
                        logger.debug(f"  - Group add: {group_add}")

                        # Use framework's run_container API with cap_add support
                        # Note: Framework automatically fails test if container exits with non-zero code
                        # CRITICAL: remove=False to preserve container until results are copied via volume mount
                        try:
                            container_result = docker_client.run_container(
                                name=container_name,
                                image=container_full_tag,
                                command=[str(device_id), str(duration_hrs)],
                                volumes=volumes,
                                devices=devices,
                                environment=environment,
                                user="root:root",
                                group_add=group_add,
                                cap_add=["PERFMON", "SYS_ADMIN"],  # Required for intel_gpu_top
                                timeout=timeout,
                                mode="batch",  # Wait for container to complete
                                attach_logs=True,  # Automatically attach logs to Allure
                                detach=True,  # Must be True for batch mode
                                remove=False,  # Keep container until results are read from volume
                            )
                        except pytest.fail.Exception as fail_ex:
                            # Framework auto-fails on non-zero exit code, catch and handle ourselves
                            error_msg = (
                                f"Container execution failed: {str(fail_ex)}. "
                                f"Device: {device_id}, Container: {container_full_tag}. "
                                f"Check attached container logs for details."
                            )
                            logger.error(error_msg)
                            result.metadata["failure_reason"] = error_msg
                            result.metadata["status"] = "N/A"
                            return result

                        # Extract container info from result
                        container_info = container_result.get("container_info", {})
                        exit_code = container_info.get("exit_code", -1)

                        logger.debug(f"Container completed with exit code: {exit_code}")

                        logger.info("FreqRunner execution completed successfully")

                    except KeyboardInterrupt:
                        # User interrupt during container execution
                        error_msg = (
                            f"User interrupt (Ctrl+C) during container execution for device '{device_id}'. "
                            f"Container: {container_name}, Test will be terminated and cleaned up."
                        )
                        test_interrupted = True
                        logger.error(error_msg)
                        result.metadata["failure_reason"] = error_msg
                        result.metadata["status"] = "N/A"
                        # Let finally block handle cleanup

                    except Exception as container_error:
                        error_msg = (
                            f"Container execution exception for device '{device_id}': "
                            f"{type(container_error).__name__}: {str(container_error)}. "
                            f"Container: {container_full_tag}, Model: {model_name}. "
                            f"Check if Docker image exists and GPU device is accessible."
                        )
                        logger.error(error_msg, exc_info=True)
                        result.metadata["failure_reason"] = error_msg
                        result.metadata["status"] = "N/A"
                        return result

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

                csv_file_path = Path(f"{aifreq_results}/averages_summary.csv")
                logger.debug(f"Checking for CSV file at: {csv_file_path}")
                logger.debug(f"CSV file exists: {csv_file_path.exists()}")

                if csv_file_path.exists():
                    # Log CSV file contents for debugging
                    try:
                        with open(csv_file_path, "r") as f:
                            csv_contents = f.read()
                        logger.debug(f"CSV file contents ({len(csv_contents)} bytes):")
                        logger.debug(f"\n{csv_contents}")
                    except Exception as e:
                        logger.warning(f"Could not read CSV for debugging: {e}")

                    # Extract KPI values from test results
                    # Update metrics in-place (they're already part of result object)
                    logger.debug(f"Extracting {len(metrics)} metrics from CSV...")
                    for kpi_name, metric_obj in metrics.items():
                        logger.debug(f"Extracting metric: {kpi_name}")
                        val = extract_csv_values(csv_file_path, "Function", "AI-Freq-LR", kpi_name)
                        logger.debug(f"  -> Raw value from CSV: {val}")
                        if val is not None:
                            # Convert 0 or 0.0 values to -1 (indicating failed/missing data)
                            # This handles cases where gpu_top tool couldn't retrieve the metric
                            try:
                                numeric_val = float(val)
                                if numeric_val == 0.0:
                                    metric_obj.value = -1
                                    # Skip unit for -1 values (avoid "-1 %" etc.)
                                    metric_obj.unit = None
                                    logger.debug(f"  -> Converted 0 to -1 for {kpi_name} (data not available)")
                                else:
                                    metric_obj.value = val
                                    # Set appropriate unit based on metric type
                                    if "frequency" in kpi_name:
                                        metric_obj.unit = "GHz"
                                    elif "utilization" in kpi_name:
                                        metric_obj.unit = "%"
                                    elif "power" in kpi_name:
                                        metric_obj.unit = "W"
                                    logger.debug(f"  -> Updated KPI {kpi_name}: value={val}, unit={metric_obj.unit}")
                            except (ValueError, TypeError):
                                # Non-numeric value - keep as-is
                                metric_obj.value = val
                                logger.debug(f"  -> Non-numeric value for {kpi_name}: {val}")
                        else:
                            logger.debug(f"Metric {kpi_name} not found in CSV (may not be applicable for this device)")

                    # Validate key frequency metric - fail test if invalid
                    # Frequency is the key metric, must be valid for successful test
                    key_freq_metric = f"frequency_max_{device}"
                    freq_value = metrics.get(key_freq_metric)
                    if freq_value and (freq_value.value == -1 or freq_value.value <= 0.01):
                        error_msg = (
                            f"Test failed: Key frequency metric '{key_freq_metric}' has invalid value "
                            f"({freq_value.value} GHz). GPU telemetry tool reported 0.0 or "
                            f"invalid values, indicating telemetry collection failed. This typically occurs when: "
                            f"1) GPU driver issues, 2) Insufficient permissions for telemetry, or "
                            f"3) GPU hardware/firmware problems. Check container logs and verify GPU is accessible."
                        )
                        logger.error(error_msg)
                        result.metadata["failure_reason"] = error_msg
                        result.metadata["status"] = "N/A"
                        result.metadata["skip_attachment"] = True  # Skip report attachment for invalid results
                        return result

                    # Extract dGPU selection metadata from CSV (if present from container)
                    dgpu_device_id = extract_csv_values(csv_file_path, "Function", "AI-Freq-LR", "dgpu_device_id")
                    dgpu_count = extract_csv_values(csv_file_path, "Function", "AI-Freq-LR", "dgpu_count")

                    # Add metadata to result for visibility in reports
                    if dgpu_device_id and str(dgpu_device_id) != "N/A":
                        result.metadata["dgpu_device_id"] = str(dgpu_device_id)
                        logger.info(f"dGPU device selected for metrics: {dgpu_device_id}")

                    if dgpu_count is not None and str(dgpu_count) != "N/A":
                        try:
                            dgpu_count_int = int(float(str(dgpu_count)))
                            result.metadata["dgpu_count"] = dgpu_count_int
                            if dgpu_count_int > 1:
                                logger.info(f"Multiple dGPUs detected: {dgpu_count_int} devices")
                                logger.info(
                                    f"Best device selected ({dgpu_device_id}) based on: "
                                    "highest avg frequency + lowest stddev (stability)"
                                )
                            elif dgpu_count_int == 1:
                                logger.info("Single dGPU detected")
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Could not parse dgpu_count '{dgpu_count}': {e}")
                else:
                    # CSV file not found - keep all metrics as -1 and mark as failure
                    error_msg = (
                        f"Results CSV file not found at expected location: {csv_file_path}. "
                        f"Test container may have failed to generate results. "
                        f"Expected file: averages_summary.csv in {aifreq_results}. "
                        f"Check container logs for execution errors."
                    )
                    logger.error(error_msg)
                    results_dir_contents = (
                        list(Path(aifreq_results).iterdir()) if Path(aifreq_results).exists() else "Directory not found"
                    )
                    logger.debug(f"Results directory contents: {results_dir_contents}")
                    result.metadata["failure_reason"] = "Results CSV file not generated by test container"
                    result.metadata["status"] = "N/A"
                    return result

                valid_metrics = [m for m in metrics.values() if m.value != -1]
                if not valid_metrics:
                    metric_names = list(metrics.keys())
                    error_msg = (
                        f"Test completed but no valid metrics were collected (all -1). "
                        f"Expected metrics: {', '.join(metric_names)}. "
                        f"CSV file was found but metric extraction failed. "
                        f"Verify CSV format matches expected structure with 'Function' column."
                    )
                    logger.error(error_msg)
                    logger.debug(f"CSV file location: {csv_file_path}")
                    result.metadata["failure_reason"] = error_msg
                    result.metadata["status"] = "N/A"
                    return result

                # If successfully processed all devices and collected valid metrics, mark as success
                result.metadata["status"] = True
                result.metadata.pop("failure_reason", None)  # Remove failure_reason if test succeeded

            except Exception as exec_error:
                # Handle any execution errors (container execution, CSV parsing, etc.)
                error_msg = (
                    f"Test execution failed with exception: {type(exec_error).__name__}: {str(exec_error)}. "
                    f"Device: {device}, Model: {model_name}, Duration: {duration_hrs}hrs. "
                    f"Check logs for stack trace and detailed error information."
                )
                logger.error(error_msg, exc_info=True)
                logger.debug(f"Execution context - Results dir: {aifreq_results}, Container: {container_full_tag}")
                result.metadata["failure_reason"] = error_msg
                # Metrics remain as -1 (data not available)
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

    except Exception as e:
        test_failed = True
        failure_message = (
            f"Unexpected error during test execution: {type(e).__name__}: {str(e)}. "
            f"Test: {test_display_name}, Device: {device}. "
            f"Check logs for complete stack trace and error context."
        )
        logger.error(failure_message, exc_info=True)
        logger.debug(f"Execution context - Test ID: {test_id}, Duration: {duration_hrs}hrs")

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
            logger.info(f"Test summary - ID: {test_id}, Device: {device}, Duration: {duration_hrs}hrs")
            logger.info(f"-1 metrics will be reported for {device.upper()} (hardware not present)")
        else:
            # Actual test execution error - log as error
            logger.error(f"Test failed with N/A status: {failure_msg}")
            logger.info(f"Test summary - ID: {test_id}, Device: {device}, Duration: {duration_hrs}hrs")

        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )

        pytest.fail(f"GPU AI Frequency test failed - {failure_msg}")

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
        logger.debug(f"Summary context - Results dir: {aifreq_results}, Device dict: {device_keys}")

    # Attach artifacts (PNG and log files) for tested devices
    # Skip attachment if test failed due to invalid key metrics (telemetry failure)
    skip_attachment = results.metadata.get("skip_attachment", False)
    best_device_id = results.metadata.get("dgpu_device_id")
    dgpu_count = results.metadata.get("dgpu_count", 0)

    if skip_attachment:
        logger.warning(
            "Skipping report attachment - test failed due to invalid key frequency metric. "
            "GPU telemetry tool reported 0.0 or invalid values."
        )
    elif device_dict:
        # Determine which devices to attach artifacts for
        devices_to_attach = []
        dgpu_devices = []

        for device_id, device_info in device_dict.items():
            device_type = device_info.get("device_type", "unknown")
            if device_type == "dgpu":
                dgpu_devices.append(device_id)
            else:
                # Always attach non-dGPU devices (iGPU, etc.)
                devices_to_attach.append((device_id, device_info, ""))

        # For dGPUs: decide based on best device selection from container results
        if dgpu_devices:
            if dgpu_count > 1 and best_device_id:
                # Multiple dGPUs detected - attach only the best device
                devices_to_attach.append(
                    (best_device_id, device_dict[best_device_id], f" (Best of {dgpu_count} dGPUs)")
                )
                logger.info(f"Multiple dGPUs detected - attaching only best device: {best_device_id}")
            elif dgpu_count > 1 and not best_device_id:
                # Multiple dGPUs with no best device (all same performance) - attach first one only
                first_dgpu = dgpu_devices[0]
                devices_to_attach.append(
                    (first_dgpu, device_dict[first_dgpu], f" ({dgpu_count} dGPUs - Equal Performance)")
                )
                logger.info(f"Multiple dGPUs with equal performance - attaching first device: {first_dgpu}")
            elif dgpu_count == 1 and best_device_id:
                # Single dGPU identified in metadata - attach only that device
                devices_to_attach.append((best_device_id, device_dict[best_device_id], ""))
                logger.info(f"Single dGPU - attaching device: {best_device_id}")
            else:
                # Fallback: dgpu_count not available or invalid - attach first dGPU only
                first_dgpu = dgpu_devices[0]
                devices_to_attach.append((first_dgpu, device_dict[first_dgpu], ""))
                logger.warning(f"dGPU count metadata not available - attaching first device only: {first_dgpu}")

        logger.info(f"Attaching artifacts for {len(devices_to_attach)} device(s)")

        for device_id, device_info, title_suffix in devices_to_attach:
            img_prefix = re.sub(r"\W+", "", device_id)

            # Attach PNG report if available
            img_file_path = Path(f"{aifreq_results}/{img_prefix}_InferencePerformance_FreqPlot.png")
            if img_file_path.exists():
                try:
                    with open(img_file_path, "rb") as f:
                        allure.attach(
                            f.read(),
                            name=f"GPU AI Frequency Report - {device_id}{title_suffix}",
                            attachment_type=allure.attachment_type.PNG,
                        )
                    logger.debug(f"Attached PNG report for device: {device_id}")
                except Exception as attach_error:
                    logger.error(
                        f"Failed to attach PNG for device {device_id}: "
                        f"{type(attach_error).__name__}: {str(attach_error)}. Image file: {img_file_path}"
                    )
            else:
                logger.debug(f"PNG report not found for device {device_id}: {img_file_path}")

            # Attach log file if available
            log_file_path = Path(f"{aifreq_results}/{img_prefix}-IntelVideoAIBoxDetails.log")
            if log_file_path.exists():
                try:
                    with open(log_file_path, "r") as f:
                        allure.attach(
                            f.read(),
                            name=f"GPU Frequency Test Log - {device_id}{title_suffix}",
                            attachment_type=allure.attachment_type.TEXT,
                        )
                    logger.debug(f"Attached log file for device: {device_id}")
                except Exception as attach_error:
                    logger.warning(f"Failed to attach log for device {device_id}: {attach_error}")
            else:
                logger.debug(f"Log file not found for device {device_id}: {log_file_path}")
    else:
        logger.warning("No devices available for artifact attachment")

    # Note: Results directory cleanup is handled by 'esq clean --all' command
    # which clears esq_data/data/vertical/metro/results/gpu along with all other test data

    is_qualification = configs.get("labels", {}).get("type") == "qualification"

    if test_interrupted:
        logger.info("All running containers have been stopped and removed.")
        if is_qualification:
            pytest.fail(failure_message)
        else:
            logger.warning(f"Test interrupted but continuing (suite mode): {failure_message}")

    if test_failed:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            logger.warning(f"Test failed but continuing (suite mode): {failure_message}")

    logger.info(f"GPU AI Frequency test '{test_name}' completed successfully")
