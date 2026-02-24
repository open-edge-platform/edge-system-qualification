# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Video Analytics (VA) Heavy Pipeline Test.

This module provides the Heavy weight video analytics pipeline benchmark:
- Detection: YOLOv11m (medium-weight, higher accuracy)
- Classification: ResNet-v1-50 + MobileNet-v2 (dual classification)
- Video: bears.h265 (H265 codec for higher compression workload)

The VA Heavy pipeline is the most demanding workload:
- Uses H265 video codec (higher decode complexity)
- Uses YOLOv11m (larger than YOLOv11n in Light)
- Dual classification with ResNet-v1-50 and MobileNet-v2
- Higher starting-frame for FPS counter (2000 for longer warmup)

The VA pipeline supports different compute modes:
- Mode 0: CPU/CPU/CPU (all stages on CPU)
- Mode 1: dGPU/dGPU/dGPU (all stages on dGPU)
- Mode 2: iGPU/iGPU/iGPU (all stages on iGPU)
- Mode 3: iGPU/iGPU/NPU (decode+detect on iGPU, classify on NPU)
- Mode 4: iGPU/NPU/NPU (decode on iGPU, detect+classify on NPU)
- Mode 5: dGPU/dGPU/NPU (decode+detect on dGPU, classify on NPU)
- Mode 6: dGPU/NPU/NPU (decode on dGPU, detect+classify on NPU)
- Mode 7: iGPU + NPU concurrent (GPU and NPU pipelines run simultaneously)
- Mode 8: dGPU + NPU concurrent (GPU and NPU pipelines run simultaneously)
"""

import json
import logging
import os
from pathlib import Path

import allure
import pandas as pd
import pytest
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Result
from sysagent.utils.infrastructure import DockerClient
from sysagent.utils.system.ov_helper import get_available_devices_by_category

from esq.utils.media.validation import detect_platform_type

# Import shared VA utilities
from .va_common import (
    VA_CONTAINER_PATH,
    attach_va_artifacts,
    create_va_metrics,
    determine_expected_modes,
    extract_fps_from_log,
    extract_metrics_from_csv,
    generate_va_charts,
    initialize_csv_files,
    prepare_docker_build_context,
    run_va_container,
)

logger = logging.getLogger(__name__)

# CSV files for heavy pipeline - now dynamically generated per test_id
VA_HEAVY_CSV_FILES = ["va_heavy_pipeline.csv"]  # Default, overridden in test

def _download_va_heavy_resources(resources_dir: str) -> tuple:
    """
    Download VA Heavy pipeline resources.

    Args:
        resources_dir: Base directory for resources

    Returns:
        Tuple of (models_path, videos_path)
    """
    models_path = os.path.join(resources_dir, "models")
    videos_path = os.path.join(resources_dir, "videos")

    os.makedirs(models_path, exist_ok=True)
    ensure_dir_permissions(models_path, uid=os.getuid(), gid=os.getgid(), mode=0o775)
    os.makedirs(videos_path, exist_ok=True)
    ensure_dir_permissions(videos_path, uid=os.getuid(), gid=os.getgid(), mode=0o775)

    # Download VA Heavy models and videos using VA utility
    logger.info("Downloading VA Heavy pipeline resources (models and videos)...")
    try:
        from esq.utils.models.va_resources import download_va_heavy_resources

        results = download_va_heavy_resources(models_path, videos_path)
        if not results:
            raise RuntimeError("Failed to download VA Heavy resources - no results returned")

        logger.info("VA Heavy resources prepared:")
        logger.info(f"  Detection model: {results.get('detection_xml')}")
        logger.info(f"  ResNet classification model: {results.get('resnet_xml')}")
        logger.info(f"  MobileNet classification model: {results.get('mobilenet_xml')}")
        logger.info(f"  Video: {results.get('video')}")

    except ImportError as e:
        logger.error(f"Failed to import VA resource downloader: {e}")
        raise RuntimeError(f"VA resource downloader not available: {e}") from e
    except Exception as e:
        logger.error(f"Failed to download VA Heavy resources: {e}")
        raise RuntimeError(f"VA Heavy resource download failed: {e}") from e

    return models_path, videos_path


@allure.title("Video Analytics Heavy Pipeline Benchmark")
def test_va_heavy(
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
    """
    End-to-end Video Analytics Heavy Pipeline Benchmark using Docker container.

    Heavy Pipeline:
    - Detection: YOLOv11m (INT8, 640x640)
    - Classification: ResNet-v1-50 (INT8) + MobileNet-v2 (INT8)
    - Video: bears.h265 (H265 codec)

    This test runs multi-stage video analytics pipelines with configurable
    compute modes (CPU/iGPU/dGPU/NPU combinations for decode/detect/classify).
    """
    # Request
    test_name = request.node.name.split("[")[0]

    # Parameters
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", "VA Heavy Pipeline")
    force_run = configs.get("force_run", False)

    logger.info(f"Starting VA Heavy Pipeline Benchmark: {test_display_name} (force_run={force_run})")

    dockerfile_name = configs.get("dockerfile_name", "Dockerfile")
    docker_image_tag = f"{configs.get('container_image', 'va_heavy_bm')}:{configs.get('image_tag', '1.0')}"
    timeout = int(configs.get("timeout", 900))  # Heavy workload needs more time
    base_image = configs.get("base_image", "intel/dlstreamer:2025.2.0-ubuntu24")
    devices = configs.get("devices", "igpu")

    # Setup
    test_dir = os.path.dirname(os.path.abspath(__file__))
    docker_dir = os.path.join(test_dir, VA_CONTAINER_PATH)
    logger.info(f"Docker directory: {docker_dir}")

    # Use CORE_DATA_DIR for results and resources
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    core_data_dir = "".join(c for c in core_data_dir_tainted)
    data_dir = os.path.join(core_data_dir, "data", "ai", "vision")
    va_results = os.path.join(data_dir, "results", "va_heavy")
    os.makedirs(va_results, exist_ok=True)

    # Create resources directory for models and videos
    resources_dir = os.path.join(va_results, "resources")
    os.makedirs(resources_dir, exist_ok=True)
    ensure_dir_permissions(resources_dir, uid=os.getuid(), gid=os.getgid(), mode=0o775)

    # Step 1: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Step 2: Get available devices
    logger.info(f"Configured device categories: {devices}")
    device_dict = get_available_devices_by_category(device_categories=devices)
    logger.debug(f"Available devices: {device_dict}")

    # Step 3: Handle missing hardware
    if not device_dict:
        logger.warning(
            f"Required {devices.upper()} hardware not available on this platform. Test will complete with N/A metrics."
        )

        metrics = create_va_metrics(value="N/A", unit="")

        results = Result.from_test_config(
            configs=configs,
            parameters={
                "timeout(s)": timeout,
                "display_name": test_display_name,
                "devices": devices,
                "pipeline": "heavy",
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

        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )

        failure_msg = f"Required {devices.upper()} hardware not available. Test completed with N/A metrics."
        logger.error(f"Test failed: {failure_msg}")
        pytest.fail(failure_msg)

    # Log device information
    for device_id, device_info in device_dict.items():
        logger.debug(f"Device {device_id}: Type={device_info['device_type']}, Name={device_info['full_name']}")

    docker_client = DockerClient()

    # Initialize variables
    test_failed = False
    failure_message = ""
    results = None
    container_name = None

    try:
        # Step 3: Prepare test environment
        def prepare_assets():
            nonlocal base_image, docker_image_tag, dockerfile_name, docker_dir, timeout

            docker_nocache = configs.get("docker_nocache", False)
            logger.info(f"Docker build cache setting: nocache={docker_nocache}")

            # Download VA Heavy resources
            models_path, videos_path = _download_va_heavy_resources(resources_dir)

            # Validate platform
            logger.info("Validating platform configuration...")
            platform_info = detect_platform_type()
            logger.debug(
                f"Platform detection: iGPU={platform_info['has_igpu']}, "
                f"dGPU_count={platform_info['dgpu_count']}, MTL={platform_info['is_mtl']}"
            )

            # Check if Docker directory exists
            if not os.path.exists(docker_dir):
                logger.warning(f"Docker directory not found: {docker_dir}")
                logger.info("Creating minimal Docker structure for VA benchmark...")
                os.makedirs(docker_dir, exist_ok=True)

            # Copy consolidated utilities into Docker build context
            logger.info("Preparing Docker build context with consolidated utilities...")
            test_file_dir = Path(__file__).resolve().parent
            prepare_docker_build_context(test_file_dir, docker_dir)

            # Build Docker image if Dockerfile exists
            dockerfile_path = os.path.join(docker_dir, dockerfile_name)
            if os.path.exists(dockerfile_path):
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
                    "dockerfile": dockerfile_path,
                    "build_path": docker_dir,
                }
            else:
                logger.warning(f"Dockerfile not found: {dockerfile_path}")
                logger.info("Skipping Docker build - container will need to be pre-built")
                container_config = {
                    "image_tag": docker_image_tag,
                    "timeout": timeout,
                }

            result = Result(
                metadata={
                    "status": True,
                    "container_config": container_config,
                    "models_path": models_path,
                    "videos_path": videos_path,
                    "Timeout (s)": timeout,
                    "Display Name": test_display_name,
                    "Pipeline": "heavy",
                }
            )

            return result

        prepare_test(test_name=test_name, configs=configs, prepare_func=prepare_assets, name="va_heavy_assets")

    except KeyboardInterrupt:
        failure_message = (
            f"User interrupt (Ctrl+C) detected during test preparation. Test: {test_display_name}, Devices: {devices}."
        )
        logger.error(failure_message)
        raise

    except Exception as e:
        test_failed = True
        failure_message = (
            f"Unexpected error during test preparation: {type(e).__name__}: {str(e)}. "
            f"Test: {test_display_name}, Docker image: {docker_image_tag}."
        )
        logger.error(failure_message, exc_info=True)

    # If preparation failed, return N/A metrics
    if test_failed:
        metrics = create_va_metrics(value="N/A", unit=None)

        results = Result.from_test_config(
            configs=configs,
            parameters={
                "timeout(s)": timeout,
                "display_name": test_display_name,
                "devices": devices,
                "pipeline": "heavy",
            },
            metrics=metrics,
            metadata={
                "status": "N/A",
                "failure_reason": failure_message,
            },
        )

        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
        pytest.fail(failure_message)

    # Initialize results template
    results = Result.from_test_config(
        configs=configs,
        parameters={
            "timeout(s)": timeout,
            "display_name": test_display_name,
            "pipeline": "heavy",
        },
    )

    # Create test-specific CSV file list to prevent overwriting between test cases
    csv_filename = f"va_heavy_pipeline_{test_id}.csv"
    csvlist = [csv_filename]

    try:

        def run_test():
            nonlocal container_name
            nonlocal csv_filename

            # Define metrics with N/A as initial values
            metrics = create_va_metrics(value="N/A", unit=None)

            # Initialize result template
            result = Result.from_test_config(
                configs=configs,
                parameters={
                    "test_id": test_id,
                    "devices": devices,
                    "display_name": test_display_name,
                    "pipeline": "heavy",
                },
                metrics=metrics,
                metadata={
                    "status": "N/A",
                },
            )

            # Check if devices are available
            if not device_dict:
                error_msg = f"No available devices found for configured device category: '{devices}'."
                logger.error(error_msg)
                result.metadata["failure_reason"] = error_msg
                return result

            try:
                # Detect platform
                platform_info = detect_platform_type()
                logger.debug(
                    f"Platform info: iGPU={platform_info['has_igpu']}, "
                    f"dGPU_count={platform_info['dgpu_count']}, MTL={platform_info['is_mtl']}"
                )

                # Determine expected modes based on devices
                expected_modes = determine_expected_modes(device_dict)
                logger.info(f"Test {test_id} expects modes: {expected_modes} for devices: {list(device_dict.keys())}")

                # Check existing results in CSV
                result_csv_path = Path(f"{va_results}/{csv_filename}")
                modes_to_run = expected_modes.copy()

                if result_csv_path.exists() and not force_run:
                    try:
                        df = pd.read_csv(result_csv_path)
                        if not df.empty and "Mode" in df.columns:
                            existing_modes = df["Mode"].unique().tolist()
                            modes_to_run = [m for m in expected_modes if m not in existing_modes]
                            if modes_to_run:
                                logger.info(
                                    f"Modes already in CSV: {existing_modes}. Will run missing modes: {modes_to_run}"
                                )
                            else:
                                logger.info(f"All expected modes {expected_modes} already exist in CSV")
                    except Exception as e:
                        logger.warning(f"Failed to check CSV for existing results: {e}")

                should_run_benchmark = bool(modes_to_run) or force_run

                if should_run_benchmark:
                    if force_run:
                        logger.info("Force run enabled. Re-running VA Heavy benchmark for all devices...")
                        modes_to_run = expected_modes
                    else:
                        logger.info(f"Running VA Heavy benchmark for modes: {modes_to_run}")

                    # Initialize CSV files
                    logger.info(f"Ensuring CSV files exist: {csvlist}")
                    initialize_csv_files(va_results, csvlist)

                    # Get resource paths
                    models_path = os.path.join(resources_dir, "models")
                    videos_path = os.path.join(resources_dir, "videos")

                    # Run container
                    try:
                        configs_with_paths = {**configs, "_models_path": models_path, "_videos_path": videos_path}

                        container_name = f"{configs.get('container_image', 'va_heavy_bm')}_heavy"
                        container_result = run_va_container(
                            docker_client=docker_client,
                            container_name=container_name,
                            image_name=configs.get("container_image", "va_heavy_bm"),
                            image_tag=configs.get("image_tag", "1.0"),
                            benchmark_script="va_heavy",
                            output_dir=va_results,
                            devices=list(device_dict.keys()),
                            platform_info=platform_info,
                            configs=configs_with_paths,
                            config_file=None,
                            device_info=device_dict,  # Pass device info for proper normalization
                            csv_filename=csv_filename,  # Pass test-specific CSV filename
                        )

                        exit_code = container_result.get("container_info", {}).get("exit_code", 1)
                        if exit_code != 0:
                            error_msg = f"Container execution failed (exit code: {exit_code})."
                            logger.error(error_msg)
                            result.metadata["failure_reason"] = error_msg
                            result.metadata["status"] = "N/A"
                            return result

                    except Exception as container_err:
                        error_msg = f"Failed to run container: {type(container_err).__name__}: {str(container_err)}."
                        logger.error(error_msg, exc_info=True)
                        result.metadata["failure_reason"] = error_msg
                        result.metadata["status"] = "N/A"
                        return result

                else:
                    logger.info(
                        f"All expected modes {expected_modes} already exist in CSV. Skipping benchmark execution."
                    )

                # Process CSV files to extract metrics
                for csv_filename in csvlist:
                    csv_file_path = Path(f"{va_results}/{csv_filename}")
                    if csv_file_path.exists():
                        logger.info(f"Processing CSV file: {csv_filename} for modes: {expected_modes}")
                        extract_metrics_from_csv(result, csv_file_path, expected_modes, logger)

                # Extract FPS from log file
                log_file_path = Path(va_results) / "va_heavy_pipeline_runner.log"
                extract_fps_from_log(result, log_file_path, logger)

                # Check if metrics were collected
                valid_metrics = [m for m in result.metrics.values() if m.value != "N/A"]
                if not valid_metrics:
                    error_msg = "Test completed but no valid metrics were collected."
                    logger.error(error_msg)
                    result.metadata["failure_reason"] = error_msg
                    result.metadata["status"] = "N/A"
                    return result

                result.metadata["status"] = True
                result.metadata.pop("failure_reason", None)

            except Exception as exec_error:
                error_msg = f"Test execution failed: {type(exec_error).__name__}: {str(exec_error)}."
                logger.error(error_msg, exc_info=True)
                result.metadata["failure_reason"] = error_msg
                return result

            logger.debug(f"Test results: {json.dumps(result.to_dict(), indent=2)}")

            return result

        # Execute the test with caching
        results = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            test_name=test_name,
            configs=configs,
            run_test_func=run_test,
        )

    except KeyboardInterrupt:
        failure_message = f"User interrupt during test execution. Test: {test_display_name}."
        logger.error(failure_message)
        if container_name:
            try:
                docker_client.cleanup_container(container_name, timeout=10)
            except Exception as cleanup_err:
                logger.warning(f"Container cleanup warning: {cleanup_err}")
        raise

    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during test execution: {type(e).__name__}: {str(e)}."
        logger.error(failure_message, exc_info=True)

    finally:
        # Cleanup container
        if container_name:
            try:
                docker_client.cleanup_container(container_name, timeout=10)
            except Exception as cleanup_err:
                logger.warning(f"Container cleanup warning: {cleanup_err}")

    # Handle N/A status
    if results.metadata.get("status") == "N/A" and "failure_reason" in results.metadata:
        failure_msg = results.metadata["failure_reason"]
        logger.error(f"Test failed with N/A status: {failure_msg}")

        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )

        pytest.fail(f"VA Heavy test failed - {failure_msg}")

    # Validate test results
    validate_test_results(results=results, configs=configs, get_kpi_config=get_kpi_config, test_name=test_name)

    try:
        logger.info("Generating test result visualizations...")

        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as summary_error:
        logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)

    # Attach artifacts
    attach_va_artifacts(va_results, csvlist, logger)

    # Generate charts using test-specific CSV filename
    generate_va_charts(pp_results=va_results, configs=configs, test_logger=logger, csv_filename=csv_filename)

    if test_failed:
        pytest.fail(failure_message)
