# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any, Dict

import pytest
from esq.suites.ai.vision.src.dlstreamer.execution import (
    run_multi_device_test,
)
from esq.suites.ai.vision.src.dlstreamer.preparation import prepare_assets, prepare_baseline
from esq.suites.ai.vision.src.dlstreamer.results import (
    update_final_results_metadata,
)

# Import DLStreamer modular utilities
from esq.suites.ai.vision.src.dlstreamer.utils import (
    cleanup_stale_containers,
    cleanup_thread_pool,
    create_device_metrics,
    sort_devices_by_priority,
    update_metrics_to_error_state,
)

# Import reference data utilities
from esq.utils.references import (
    add_reference_data_to_result,
    attach_reference_data_to_allure,
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
    target_fps = configs.get("target_fps", 14.5)
    pipeline = configs.get("pipeline", None)
    pipeline_params = configs.get("pipeline_params", {})
    devices = configs.get("devices", [])
    timeout = configs.get("timeout", 300)
    pipeline_timeout = configs.get("pipeline_timeout", 180)
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
    consecutive_timeout_threshold = configs.get("consecutive_timeout_threshold", 4)
    max_streams_above_baseline = configs.get("max_streams_above_baseline", 1)

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

    # Check for multi-stage single pipeline mode
    multi_stage_single_pipeline = configs.get("multi_stage_single_pipeline", False)

    if multi_stage_single_pipeline:
        # Special mode: Multi-stage pipeline with different devices per stage
        # Pipeline is treated as a single unit (not run per-device separately)
        # The 'devices' param validates that ALL required devices are available
        # Device property placeholders (${DEVICE_NPU}, ${DEVICE_IGPU}, etc.) are resolved
        # by the pipeline module based on system configuration
        logger.info("Multi-stage single pipeline mode enabled")
        logger.info(f"Validating required device categories: {devices}")

        # Get available devices to validate they exist
        device_dict = get_available_devices_by_category(device_categories=devices)

        if not device_dict:
            pytest.fail(f"Multi-stage pipeline requires devices {devices}, but none are available on this system.")

        # In multi-stage mode, use a composite device ID for the entire pipeline
        # This represents the multi-stage configuration as a single logical device
        device_list = ["multi_stage"]
        logger.info(f"Running multi-stage pipeline as single configuration: {device_list}")
    else:
        # Standard mode: use ${DEVICE_ID} placeholder, run separately per device
        # Device property placeholders in pipeline_params are resolved automatically
        logger.info("Standard mode: using device categories for dynamic device assignment")
        logger.info(f"Configured device categories: {devices}")
        device_dict = get_available_devices_by_category(device_categories=devices)

        # Extract device IDs to maintain compatibility with existing code
        device_list = list(device_dict.keys())

        if not device_dict:
            pytest.fail(f"No available devices found for device categories: {devices}")

    # Log detailed device information
    logger.debug(f"Available devices: {device_dict}")
    for device_id, device_info in device_dict.items():
        logger.debug(f"Device {device_id}: Type={device_info['device_type']}, Name={device_info['full_name']}")

    # Get current system info
    system_info = SystemInfoCache()
    hardware_info = system_info.get_hardware_info()
    num_sockets = hardware_info.get("cpu", {}).get("sockets") or 1

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
        asset_result = prepare_test(
            test_name=test_name,
            prepare_func=lambda: prepare_assets(
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

        # Extract container config for device-specific image selection
        container_config = asset_result.metadata.get("container_config", {})

        # Run baseline streams analysis each device
        baseline_streams_results = []
        for device_id in device_list:
            # Specific cache configurations for each device
            cache_configs = {
                "device_id": device_id,
                "target_fps": target_fps,
                "pipeline": pipeline,
                "pipeline_params": pipeline_params,
                "max_streams_above_baseline": max_streams_above_baseline,
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
                    container_config=container_config,
                ),
                cached_result=cached_result,
                cache_result=cache_result,
                cache_configs=cache_configs,
                name=f"Baseline streams analysis for device {device_id}",
            )

            baseline_streams_results.append(result)

            # Check immediately if this baseline analysis failed
            if not result.metadata.get("status", False):
                device_id = result.metadata.get("device_id", "Unknown")
                num_streams = result.metadata.get("num_streams", -1)
                per_stream_fps = result.metadata.get("per_stream_fps", 0.0)
                error_msg = result.metadata.get("error", "Unknown error")
                logger.error(
                    f"Baseline streams analysis failed for device {device_id}: "
                    f"num_streams={num_streams}, per_stream_fps={per_stream_fps}, error={error_msg}"
                )

                # Create failed result with error details using helper function
                results = Result.from_test_config(
                    configs=configs,
                    name=test_name,
                    parameters={
                        "Devices": devices,
                        "Device List": device_list,
                        "Target FPS": target_fps,
                    },
                    metrics=create_device_metrics(device_list),
                    metadata={
                        "status": False,
                        "error": f"Baseline streams analysis failed for device {device_id}",
                        "failed_device": device_id,
                        "baseline_error": error_msg,
                        "num_streams": num_streams,
                        "per_stream_fps": per_stream_fps,
                    },
                )
                pytest.fail(f"Baseline streams analysis failed for device {device_id}: {error_msg}")

        for result in baseline_streams_results:
            if result.metadata.get("status", False):
                logger.info(
                    f"Device {result.metadata['device_id']} - Per Stream FPS: "
                    f"{result.metadata.get('per_stream_fps', 'N/A')}, "
                    f"Num Streams: {result.metadata.get('num_streams', 'N/A')}"
                )
            else:
                device_id = result.metadata.get("device_id", "Unknown")
                num_streams = result.metadata.get("num_streams", -1)
                per_stream_fps = result.metadata.get("per_stream_fps", 0.0)
                error_msg = result.metadata.get("error", "Unknown error")
                logger.error(
                    f"Baseline streams analysis failed for device {device_id}: "
                    f"num_streams={num_streams}, per_stream_fps={per_stream_fps}, error={error_msg}"
                )

        # Step 3: Execute test
        qualified_devices: Dict[str, Any] = {}
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

        # Count dGPU devices detected (GPU.0, GPU.1, etc.)
        dgpu_device_count = sum(1 for dev_id in device_list if dev_id.upper().startswith("GPU."))

        # Determine if aggregate streams_max metric is needed:
        # 1. Multiple devices detected (multi-device scenario)
        # 2. KPI validation requires streams_max
        # 3. Multiple device categories requested in config (even if only 1 detected)
        # 4. Multiple dGPU devices detected (even if config only specifies "dgpu")
        needs_aggregate_metric = (
            len(streams_max_devices) > 1
            or "streams_max" in current_kpi_refs
            or len(devices) > 1
            or dgpu_device_count > 1
        )

        if needs_aggregate_metric:
            if len(streams_max_devices) > 1:
                logger.debug(
                    f"Multi-device scenario detected with {len(streams_max_devices)} devices. "
                    "Adding aggregate streams_max metric."
                )
            elif dgpu_device_count > 1:
                logger.debug(
                    f"Multiple dGPU devices detected ({dgpu_device_count} dGPUs). Adding aggregate streams_max metric."
                )
            elif len(devices) > 1:
                logger.debug(
                    f"Multiple device categories requested in config ({len(devices)} categories). "
                    "Adding aggregate streams_max metric."
                )
            elif "streams_max" in current_kpi_refs:
                logger.debug(
                    "KPI validation requires streams_max metric. "
                    "Adding aggregate streams_max metric for single device scenario."
                )

            # Add individual device metrics first
            for metric_name, unit in default_metrics:
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = unit
            # Add aggregate streams_max metric
            if "streams_max" not in all_metrics:
                all_metrics["streams_max"] = "streams"
                logger.debug("Added aggregate streams_max metric")

        metrics = {kpi: Metrics(unit=unit, value=-1) for kpi, unit in all_metrics.items()}

        # Check if any baseline analysis succeeded
        baseline_success_count = sum(1 for r in baseline_streams_results if r.metadata.get("status", False))
        if baseline_success_count == 0:
            # All baseline analyses failed - create result with error details and fail
            logger.error("All baseline streams analyses failed. Cannot proceed with test.")

            # Create failed result with error details
            results = Result.from_test_config(
                configs=configs,
                name=test_name,
                parameters={
                    "Devices": devices,
                    "Device List": device_list,
                    "Target FPS": target_fps,
                },
                metrics=metrics,
                metadata={
                    "status": False,
                    "error": "All baseline streams analyses failed",
                    "failed_devices": {
                        r.metadata.get("device_id"): {
                            "num_streams": r.metadata.get("num_streams", -1),
                            "per_stream_fps": r.metadata.get("per_stream_fps", 0.0),
                            "error": r.metadata.get("error", "Unknown"),
                        }
                        for r in baseline_streams_results
                    },
                },
            )
            pytest.fail("All baseline streams analyses failed")

        # Initialize results template using from_test_config for automatic metadata application
        results = Result.from_test_config(
            configs=configs,
            name=test_name,
            parameters={
                "Devices": devices,
                "Device List": device_list,
                "Target FPS": target_fps,
                "Max Streams Above Baseline": max_streams_above_baseline,
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

        # Prepare baseline streams dictionary
        baseline_streams = {res.metadata["device_id"]: res.metadata for res in baseline_streams_results}

        # Sort devices by priority based on baseline streams results using modular function
        sorted_baseline = sort_devices_by_priority(baseline_streams, device_dict)
        sorted_devices = [d for d in sorted_baseline.keys() if d in device_list]
        logger.info(f"Processing devices in order: {sorted_devices}")

        # Cleanup before starting the test
        cleanup()

        # Cache configuration for the overall multi-device test
        overall_cache_configs = {
            "target_fps": target_fps,
            "pipeline": pipeline,
            "pipeline_params": pipeline_params,
            "consecutive_timeout_threshold": consecutive_timeout_threshold,
            "max_streams_above_baseline": max_streams_above_baseline,
            "devices": sorted(devices),  # Sort to ensure consistent cache key
            "device_list": sorted(device_list),  # Include actual detected devices
            "type": "total_streams_multi_device",
        }

        # Execute the multi-device test with overall caching
        results = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=lambda: run_multi_device_test(
                docker_client=docker_client,
                device_dict=device_dict,
                sorted_devices=sorted_devices,
                device_list=device_list,
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
                consecutive_timeout_threshold=consecutive_timeout_threshold,
                max_streams_above_baseline=max_streams_above_baseline,
                baseline_streams=baseline_streams,
                container_config=container_config,
                configs=configs,
                test_name=test_name,
                devices=devices,
                timeout=timeout,
                test_display_name=test_display_name,
                metrics=metrics,
                baseline_streams_results=baseline_streams_results,
            ),
            test_name=test_name,
            configs=configs,
            cache_configs=overall_cache_configs,
            name="Multi-device total streams analysis",
        )

        # Extract qualified_devices from extended_metadata for downstream processing
        qualified_devices = results.extended_metadata.get("qualified_devices", {})

        logger.info(f"Multi-device test completed with {len(qualified_devices)} qualified devices")

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

        # Ensure metrics are set to -1 to indicate interrupted/error state
        if results is not None:
            # Set all stream metrics to -1 to indicate interruption
            update_metrics_to_error_state(results.metrics, value=-1, filter_prefix="streams_max")

            # Mark as failed with interrupt error
            results.metadata["status"] = False
            results.metadata["error"] = failure_message
            results.metadata["Target FPS"] = target_fps

        if results is not None and baseline_streams_results:
            try:
                update_final_results_metadata(
                    results,
                    qualified_devices,
                    device_list,
                    baseline_streams_results,
                    requested_device_categories=devices,
                    target_fps=target_fps,
                )
            except Exception as metadata_error:
                logger.warning(f"Failed to capture baseline metadata on interrupt: {metadata_error}")

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
            # Use -1 to indicate error/interrupted state
            metrics = {kpi: Metrics(unit=unit, value=-1) for kpi, unit in all_metrics.items()}

            results = Result.from_test_config(
                configs=configs,
                name=test_name,
                parameters={
                    "Devices": devices,
                    "Device List": device_list,
                    "Target FPS": target_fps,
                    "Display Name": test_display_name,
                },
                metrics=metrics,
                metadata={"status": False, "error": str(e)},
            )
        else:
            results.metadata["status"] = False
            results.metadata["error"] = str(e)
            results.metadata["Target FPS"] = target_fps
            # Set all stream metrics to -1 to indicate error state
            update_metrics_to_error_state(results.metrics, value=-1, filter_prefix="streams_max")

        if results is not None and baseline_streams_results:
            try:
                update_final_results_metadata(
                    results,
                    qualified_devices,
                    device_list,
                    baseline_streams_results,
                    requested_device_categories=devices,
                    target_fps=target_fps,
                )
            except Exception as metadata_error:
                logger.warning(f"Failed to capture baseline metadata on error: {metadata_error}")

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
        # Add verified reference data if available in configs
        try:
            reference_data = configs.get("verified_reference_data", {}).get("vision_ai", [])
            if reference_data and results is not None:
                logger.info("Processing verified reference data for Vision AI test")
                # Add to result extended metadata (filtered by generation)
                add_reference_data_to_result(
                    result=results,
                    reference_data=reference_data,
                    data_key="verified_reference_data",
                    filter_by_generation=True,
                )
                # Attach to Allure report
                attach_reference_data_to_allure(
                    reference_data=reference_data,
                    attachment_name="Vision AI - Verified Reference Data",
                    filter_by_generation=True,
                )
        except Exception as ref_data_error:
            logger.warning(f"Failed to process verified reference data: {ref_data_error}")

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
