# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from typing import Any, Dict

import pytest
from esq.suites.ai.vision.src.dlstreamer.execution import (
    finalize_device_metrics,
    process_device_results,
    run_device_test,
    update_device_pipeline_info,
    update_final_results_metadata,
    validate_final_streams_results,
)
from esq.suites.ai.vision.src.dlstreamer.preparation import prepare_assets, prepare_baseline

# Import DLStreamer modular utilities
from esq.suites.ai.vision.src.dlstreamer.utils import (
    cleanup_stale_containers,
    cleanup_thread_pool,
    sort_devices_by_priority,
)

# Import from sysagent utilities
from sysagent.utils.core import Metrics, Result, get_metric_name_for_device
from sysagent.utils.infrastructure import DockerClient
from sysagent.utils.system import SystemInfoCache
from sysagent.utils.system.ov_helper import get_available_devices_by_category

# Pipeline info handling is now in execution module

logger = logging.getLogger(__name__)


def test_dlstreamer(
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
    End-to-end DL Streamer Test using a Docker container.
    """
    # Request
    test_name = request.node.name.split("[")[0]

    # Parameters
    test_display_name = configs.get("display_name", test_name)
    kpi_validation_mode = configs.get("kpi_validation_mode", "all")
    videos = configs.get("videos", [])
    model_precision = configs.get("model_precision", "fp16")
    target_fps = configs.get("target_fps", 14.5)
    pipeline = configs.get("pipeline", None)
    pipeline_params = configs.get("pipeline_params", {})
    devices = configs.get("devices", [])
    timeout = configs.get("timeout", 300)
    pipeline_timeout = configs.get("pipeline_timeout", 180)
    visualize_stream = configs.get("visualize_stream", False)
    docker_image_tag_analyzer = configs.get(
        "docker_image_tag_analyzer",
        f"{configs.get('docker_image_name_analyzer', 'test-dlstreamer-analyzer')}:"
        f"{configs.get('docker_image_tag_analyzer', 'latest')}",
    )
    docker_image_tag_utils = configs.get(
        "docker_image_tag_utils",
        f"{configs.get('docker_image_name_utils', 'test-dlstreamer-utils')}:"
        f"{configs.get('docker_image_tag_utils', 'latest')}",
    )
    docker_container_prefix = configs.get("docker_container_prefix", "test-dlstreamer")
    max_plateau_iterations = configs.get("max_plateau_iterations", 3)

    # Setup
    test_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(test_dir, "src")
    core_data_dir = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "app_data"))
    data_dir = os.path.join(core_data_dir, "data", "suites", "ai", "vision")
    models_dir = os.path.join(data_dir, "models")
    videos_dir = os.path.join(data_dir, "videos")
    results_dir = os.path.join(data_dir, "results")

    container_mnt_dir = "/mnt"

    logger.info(f"Starting DL Streamer Test: {test_display_name}")

    # Step 1: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Verify docker client connection
    docker_client = DockerClient()

    # Get available devices based on device categories
    logger.info(f"Configured device categories: {devices}")
    device_dict = get_available_devices_by_category(device_categories=devices)

    # Extract device IDs to maintain compatibility with existing code
    device_list = list(device_dict.keys())
    logger.debug(f"Available devices: {device_dict}")
    if not device_dict:
        pytest.fail(f"No available devices found for device categories: {devices}")

    # TEMPORARY: Skip additional discrete GPUs until multi-dGPU support is fully enabled
    # Only run on the first discrete GPU if system has more than one
    discrete_gpus = [
        dev_id
        for dev_id, dev_info in device_dict.items()
        if "discrete" in dev_info.get("device_type", "").lower() and "GPU." in dev_id
    ]

    if len(discrete_gpus) > 1:
        # Keep only the first discrete GPU
        first_dgpu = discrete_gpus[0]
        skipped_dgpus = discrete_gpus[1:]

        logger.warning(
            f"TEMPORARY LIMITATION: System has {len(discrete_gpus)} discrete GPUs. "
            f"Only testing on {first_dgpu}. Skipping: {', '.join(skipped_dgpus)}. "
            "This limitation will be removed once multi-dGPU support is fully enabled."
        )

        # Remove additional discrete GPUs from device_dict and device_list
        for dgpu in skipped_dgpus:
            device_dict.pop(dgpu, None)
            if dgpu in device_list:
                device_list.remove(dgpu)

        logger.info(f"Updated device list after dGPU filtering: {device_list}")

    # Log detailed device information
    for device_id, device_info in device_dict.items():
        logger.debug(f"Device {device_id}: Type={device_info['device_type']}, Name={device_info['full_name']}")

    # Get current system info
    system_info = SystemInfoCache()
    hardware_info = system_info.get_hardware_info()
    num_sockets = hardware_info.get("cpu", {}).get("socket_count", 1)

    # Use modularized cleanup functions
    def cleanup() -> None:
        """Cleanup function to remove containers and thread pool."""
        cleanup_stale_containers(docker_client, docker_container_prefix)
        cleanup_thread_pool()

    # Initialize variables for finally block (moved to top for broader coverage)
    validation_results = {}
    test_failed = False
    test_interrupted = False
    failure_message = ""
    results = None
    qualified_devices = {}
    baseline_streams_results = []

    try:
        # Step 2: Prepare test using modular functions
        # Run asset preparation using modular function
        prepare_test(
            test_name=test_name,
            prepare_func=lambda: prepare_assets(
                videos=videos,
                configs=configs,
                models_dir=models_dir,
                videos_dir=videos_dir,
                src_dir=src_dir,
                docker_client=docker_client,
                docker_image_tag_analyzer=docker_image_tag_analyzer,
                docker_image_tag_utils=docker_image_tag_utils,
                docker_container_prefix=docker_container_prefix,
            ),
            configs=configs,
            name="Assets",
        )

        # Run baseline streams analysis each device
        baseline_streams_results = []
        for device_id in device_list:
            # Specific cache configurations for each device
            cache_configs = {
                "device_id": device_id,
                "target_fps": target_fps,
                "pipeline": pipeline,
                "pipeline_params": pipeline_params,
                "max_plateau_iterations": max_plateau_iterations,
                "visualize_stream": visualize_stream,
                "type": "baseline_streams",
            }

            result = prepare_test(
                test_name=test_name,
                configs=configs,
                prepare_func=lambda device_id=device_id: prepare_baseline(
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
                ),
                cached_result=cached_result,
                cache_result=cache_result,
                cache_configs=cache_configs,
                name=f"Baseline streams analysis for device {device_id}",
            )

            baseline_streams_results.append(result)

        for result in baseline_streams_results:
            if result.metadata.get("status", False):
                logger.info(
                    f"Device {result.metadata['device_id']} - Per Stream FPS: {result.metadata.get('per_stream_fps', 'N/A')}, Num Streams: {result.metadata.get('num_streams', 'N/A')}"
                )
            else:
                logger.error(f"Baseline streams analysis failed for device {result.metadata['device_id']}")

        # Step 3: Execute test
        qualified_devices: Dict[str, Any] = {}
        final_results: Dict[str, Any] = {}  # Prepare result template
        default_metrics = [(get_metric_name_for_device(dev, prefix="streams_max"), "streams") for dev in device_list]
        current_kpi_refs = configs.get("kpi_refs", [])
        if not current_kpi_refs:
            all_metrics = {name: unit for name, unit in default_metrics}
        else:
            all_metrics = {}
            for kpi in current_kpi_refs:
                all_metrics[kpi] = get_kpi_config(kpi).get("unit", "")

        # Check if we have multiple devices with streams_max metrics
        streams_max_devices = []
        for metric_name, unit in default_metrics:
            if metric_name.startswith("streams_max") and metric_name != "streams_max":
                streams_max_devices.append(metric_name)

        # Automatically add aggregate streams_max metric for multi-device scenarios
        if len(streams_max_devices) > 1:
            logger.info(
                f"Multi-device scenario detected with {len(streams_max_devices)} devices. Adding aggregate streams_max metric."
            )
            # Add individual device metrics first
            for metric_name, unit in default_metrics:
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = unit
            # Add aggregate streams_max metric
            if "streams_max" not in all_metrics:
                all_metrics["streams_max"] = "streams"
                logger.debug("Added aggregate streams_max metric for multi-device total")

        metrics = {
            kpi: Metrics(unit=unit, value=0.0 if kpi == "streams_max" else -1.0) for kpi, unit in all_metrics.items()
        }

        # Initialize results template using from_test_config for automatic metadata application
        results = Result.from_test_config(
            configs=configs,
            name=test_name,
            parameters={
                "Devices": devices,
                "Device List": device_list,
                "Detection Model Precision": model_precision,
                "Target FPS": target_fps,
                "Max Plateau Iterations": max_plateau_iterations,
                "Timeout (s)": timeout,
                "Display Name": test_display_name,
                "Pipeline": pipeline,
                "Pipeline Params": pipeline_params,
            },
            metrics=metrics,
            metadata={
                "status": True,
            },
        )

        # Optimize CPU baseline streams when running multiple devices
        baseline_streams = {res.metadata["device_id"]: res.metadata for res in baseline_streams_results}
        if len(device_list) > 1 and "CPU" in baseline_streams:
            original_cpu_streams = baseline_streams["CPU"].get("num_streams", 1)
            optimized_cpu_streams = max(1, original_cpu_streams // 3)  # Scale down by ratio of 3

            if optimized_cpu_streams != original_cpu_streams:
                logger.info("Optimizing CPU baseline streams for multi-device analysis:")
                logger.info(f"  Original CPU baseline: {original_cpu_streams} streams")
                logger.info(f"  Optimized CPU baseline: {optimized_cpu_streams} streams (scaled by 1/3)")
                baseline_streams["CPU"]["num_streams"] = optimized_cpu_streams

        # Sort devices by priority based on baseline streams results using modular function
        sorted_baseline = sort_devices_by_priority(baseline_streams, device_dict)
        sorted_devices = [d for d in sorted_baseline.keys() if d in device_list]
        logger.info(f"Processing devices in order: {sorted_devices}")

        # Cleanup before starting the test
        cleanup()

        # Run total streams analysis for all devices in device_list
        for device_id in sorted_devices:
            logger.info(f"\n{'=' * 60}\nProcessing device: {device_id}\n{'=' * 60}")
            # logger.info(f"Running test for device: {device_id}")

            # Prepare device-specific configurations
            metric_name = get_metric_name_for_device(device_id, prefix="streams_max")
            default_metrics = {metric_name: Metrics(unit="streams", value=-1.0)}

            # Specific cache configurations for each device
            cache_configs = {
                "device_id": device_id,
                "target_fps": target_fps,
                "pipeline": pipeline,
                "pipeline_params": pipeline_params,
                "max_plateau_iterations": max_plateau_iterations,
                "visualize_stream": visualize_stream,
                "devices": devices,
            }

            # Execute test with cache using modular function
            result = execute_test_with_cache(
                cached_result=cached_result,
                cache_result=cache_result,
                run_test_func=lambda: run_device_test(
                    docker_client=docker_client,
                    device_dict=device_dict,
                    device_id=device_id,
                    pipeline=pipeline,
                    pipeline_params=pipeline_params,
                    docker_image_tag_analyzer=docker_image_tag_analyzer,
                    docker_container_prefix=docker_container_prefix,
                    data_dir=data_dir,
                    container_mnt_dir=container_mnt_dir,
                    pipeline_timeout=pipeline_timeout,
                    results_dir=results_dir,
                    target_fps=target_fps,
                    num_sockets=num_sockets,
                    max_plateau_iterations=max_plateau_iterations,
                    qualified_devices=qualified_devices,
                    metrics=default_metrics,
                    baseline_streams=baseline_streams.get(device_id, {}),
                    visualize_stream=visualize_stream,
                ),
                test_name=test_name,
                configs=configs,
                cache_configs=cache_configs,
                name=f"Total streams analysis for device {device_id}",
            )

            if result.metadata.get("status", False):
                # If the result is from cache, update qualified_devices accordingly
                device_streams = result.metrics[metric_name].value
                stream_fps = result.metadata.get("Per Stream FPS", 0)
                dev_type = None
                if device_id in device_dict:
                    dev_type = device_dict[device_id]["device_type"]

                device_result = {
                    "device_type": dev_type,
                    "num_streams": device_streams,
                    "per_stream_fps": stream_fps,
                    "pass": True,
                }
                qualified_devices[device_id] = device_result

                logger.debug(f"[before update] DL Streamer Test results: {json.dumps(results.to_dict(), indent=2)}")
                logger.info(f"Device {device_id} qualified with {device_streams} streams at {stream_fps:.2f} FPS")
            else:
                logger.error(f"Test failed for device {device_id}: {result.metadata.get('error', 'Unknown error')}")
                continue

        # Process device results using modular function
        process_device_results(device_list, qualified_devices, results)

        # Finalize device metrics using modular function
        finalize_device_metrics(results, qualified_devices, device_list)

        # Validate final streams results using modular function
        validate_final_streams_results(results, qualified_devices, device_list)

        # Update pipeline information for qualified devices using modular function
        update_device_pipeline_info(results, qualified_devices, pipeline, pipeline_params, device_dict)

        # Update final results metadata using modular function
        update_final_results_metadata(results, qualified_devices, device_list)

        logger.debug(f"DL Streamer Test results: {json.dumps(results.to_dict(), indent=2)}")

        # Check if test failed and store failure info
        if not results.metadata.get("status", False):
            test_failed = True
            failure_message = f"DL Streamer Test failed: {results.metadata.get('error', 'Unknown error')}"

        # Step 4: Validate test results (always run to populate validation_results)
        validation_results = validate_test_results(
            results=results,
            configs=configs,
            get_kpi_config=get_kpi_config,
            test_name=test_name,
            mode=kpi_validation_mode,
        )

        # Update KPI validation status in result metadata
        results.update_kpi_validation_status(validation_results, kpi_validation_mode)

        # Add KPI configuration and validation results to the Result object
        current_kpi_refs = configs.get("kpi_refs", [])
        if current_kpi_refs:
            kpi_data = {}
            # Get the final validation mode based on validation results
            final_mode = results.get_final_validation_mode(validation_results, kpi_validation_mode)
            for kpi_name in current_kpi_refs:
                kpi_config = get_kpi_config(kpi_name)
                if kpi_config:
                    kpi_data[kpi_name] = {
                        "config": kpi_config,
                        "validation": validation_results.get("validations", {}).get(kpi_name, {}),
                        "mode": final_mode,
                    }
            results.kpis = kpi_data

    except KeyboardInterrupt:
        failure_message = "Interrupt detected during DL Streamer test execution"
        test_interrupted = True
        logger.error(failure_message)

    except Exception as e:
        # Catch any unhandled exceptions and ensure they don't prevent summarization
        test_failed = True
        failure_message = f"Unexpected error during DL Streamer test execution: {str(e)}"
        logger.error(failure_message, exc_info=True)

        # Create a minimal results object if none exists
        if results is None:
            default_metrics = [
                (get_metric_name_for_device(dev, prefix="streams_max"), "streams") for dev in device_list
            ]
            current_kpi_refs = configs.get("kpi_refs", [])
            if not current_kpi_refs:
                all_metrics = {name: unit for name, unit in default_metrics}
            else:
                all_metrics = {}
                for kpi in current_kpi_refs:
                    all_metrics[kpi] = get_kpi_config(kpi).get("unit", "")
            metrics = {
                kpi: Metrics(unit=unit, value=0.0 if kpi == "streams_max" else -1.0)
                for kpi, unit in all_metrics.items()
            }

            results = Result.from_test_config(
                configs=configs,
                name=test_name,
                parameters={
                    "Devices": devices,
                    "Device List": device_list,
                    "Detection Model Precision": model_precision,
                    "Target FPS": target_fps,
                    "Display Name": test_display_name,
                },
                metrics=metrics,
                metadata={"status": False, "error": str(e)},
            )
        else:
            results.metadata["status"] = False
            results.metadata["error"] = str(e)

        # Try to run validation even if test failed
        try:
            validation_results = validate_test_results(
                results=results,
                configs=configs,
                get_kpi_config=get_kpi_config,
                test_name=test_name,
                mode=kpi_validation_mode,
            )
            results.update_kpi_validation_status(validation_results, kpi_validation_mode)
        except Exception as validation_error:
            logger.error(f"Validation also failed: {validation_error}")
            validation_results = {"skipped": True, "skip_reason": "Validation failed due to test errors"}

    finally:
        # Step 5: Always summarize test results, regardless of test outcome
        try:
            if results is not None:
                summarize_test_results(
                    results=results,
                    configs=configs,
                    get_kpi_config=get_kpi_config,
                    test_name=test_name,
                )
                logger.info("✓ DL Streamer test result summary completed successfully")
            else:
                logger.error("No results to summarize")
        except Exception as summary_error:
            logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)

        try:
            cleanup()
            logger.info("✓ DL Streamer test cleanup completed")
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed: {cleanup_error}")

        is_qualification = configs.get("labels", {}).get("type") == "qualification"

        if test_interrupted:
            if is_qualification:
                pytest.fail(failure_message)
            else:
                raise RuntimeError(failure_message)
        if test_failed:
            pytest.fail(failure_message)

    logger.info(f"DL Streamer test '{test_name}' completed successfully")
