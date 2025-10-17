# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os

import pytest
from esq.suites.ai.gen.src.text_generation.execution import process_device_results, run_device_test
from esq.suites.ai.gen.src.text_generation.preparation import prepare_assets

# Import Text Generation modular utilities
from esq.suites.ai.gen.src.text_generation.utils import cleanup_stale_containers, sort_devices_by_priority

# Import from sysagent utilities
from sysagent.utils.core import Metrics, Result, get_metric_name_for_device
from sysagent.utils.infrastructure import DockerClient
from sysagent.utils.system import SystemInfoCache
from sysagent.utils.system.ov_helper import get_available_devices_by_category

logger = logging.getLogger(__name__)


def test_text_generation(
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
    End-to-end Text Generation Test using a Docker container.
    """
    # Request
    test_name = request.node.name.split("[")[0]

    # Parameters
    test_display_name = configs.get("display_name", test_name)
    kpi_validation_mode = configs.get("kpi_validation_mode", "all")
    model_id = configs.get("model_id", "microsoft/Phi-4-mini-instruct")
    model_precision = configs.get("model_precision", "int4")
    test_num_prompts = configs.get("test_num_prompts", 5)
    test_request_rate = configs.get("test_request_rate", 1)
    test_max_concurrent_requests = configs.get("test_max_concurrent_requests", 1)
    devices = configs.get("devices", [])
    timeout = configs.get("timeout", 300)
    server_timeout = configs.get("server_timeout", 300)
    benchmark_timeout = configs.get("benchmark_timeout", 1800)
    hf_dataset_url = configs.get(
        "hf_dataset_url",
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    hf_dataset_url_path = hf_dataset_url.replace("https://huggingface.co/", "")
    hf_dataset_filename = hf_dataset_url.split("/")[-1]
    docker_image_tag = f"{configs.get('container_image_name', 'genai-ovms')}:{configs.get('container_tag', 'latest')}"
    docker_buildargs = {"HF_ENDPOINT": os.getenv("HF_ENDPOINT", "https://huggingface.co")}
    benchmark_docker_base_image = f"{configs.get('benchmark_container_image', 'vllm/vllm-openai:v0.9.2')}"
    ovms_docker_base_image = f"{configs.get('ovms_container_image', 'openvino/model_server:2025.2-gpu')}"

    # Setup
    test_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(test_dir, "src", "text_generation")
    core_data_dir = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "app_data"))
    data_dir = os.path.join(core_data_dir, "data", "suites", "ai", "gen")
    models_dir = os.path.join(data_dir, "models")
    thirdparty_dir = os.path.join(data_dir, "thirdparty")
    results_dir = os.path.join(data_dir, "results", "text_generation")
    dataset_path = os.path.join(thirdparty_dir, hf_dataset_filename)

    logger.info(f"Starting Text Generation Test: {test_display_name}")

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

    # Log detailed device information
    for device_id, device_info in device_dict.items():
        logger.debug(f"Device {device_id}: Type={device_info['device_type']}, Name={device_info['full_name']}")

    # Get current system info
    system_info = SystemInfoCache()
    hardware_info = system_info.get_hardware_info()

    # Use modularized cleanup functions
    def cleanup() -> None:
        """Cleanup function to remove containers."""
        cleanup_stale_containers(docker_client, f"{test_name}_")

    # Initialize variables for finally block (moved to top for broader coverage)
    validation_results = {}
    test_failed = False
    test_interrupted = False
    failure_message = ""
    results = None

    try:
        # Step 2: Prepare test using modular functions
        # Run asset preparation using modular function
        prepare_test(
            test_name=test_name,
            prepare_func=lambda: prepare_assets(
                docker_client=docker_client,
                docker_image_tag=docker_image_tag,
                docker_buildargs=docker_buildargs,
                benchmark_docker_base_image=benchmark_docker_base_image,
                ovms_docker_base_image=ovms_docker_base_image,
                hf_dataset_url=hf_dataset_url,
                dataset_path=dataset_path,
                models_dir=models_dir,
                thirdparty_dir=thirdparty_dir,
                src_dir=src_dir,
            ),
            configs=configs,
            name="Assets",
        )

        # Step 3: Execute test
        # Sort devices by priority
        sorted_device_list = sort_devices_by_priority(device_list)

        # Prepare result template
        default_metrics = [(get_metric_name_for_device(dev, prefix="throughput"), "tokens/sec") for dev in device_list]
        current_kpi_refs = configs.get("kpi_refs", [])
        if not current_kpi_refs:
            all_metrics = {name: unit for name, unit in default_metrics}
        else:
            all_metrics = {}
            for kpi in current_kpi_refs:
                all_metrics[kpi] = get_kpi_config(kpi).get("unit", "")
        metrics = {kpi: Metrics(unit=unit, value=-1.0) for kpi, unit in all_metrics.items()}

        # Initialize results template using from_test_config for automatic metadata application
        results = Result.from_test_config(
            configs=configs,
            parameters={
                "Device": devices,
                "Device List": device_list,
                "Model ID": model_id,
                "Model Precision": model_precision,
                "Number of Prompts": test_num_prompts,
                "Request Rate": test_request_rate,
                "Max Concurrent Requests": test_max_concurrent_requests,
                "Timeout (s)": timeout,
                "Server Timeout (s)": server_timeout,
                "Benchmark Timeout (s)": benchmark_timeout,
                "Display Name": test_display_name,
            },
            metrics=metrics,
            metadata={
                "status": True,
            },
        )
        logger.debug(f"Initial Results template: {json.dumps(results.to_dict(), indent=2)}")

        # Execute test for each device
        device_results = []
        for device_id in sorted_device_list:
            logger.info(f"Running test for device: {device_id}")

            # Prepare device-specific configurations
            metric_name = get_metric_name_for_device(device_id, prefix="throughput")
            default_metrics = {metric_name: Metrics(unit="tokens/sec", value=-1.0)}

            # Specific cache configurations for each device
            cache_configs = {
                "device_id": device_id,
                "model": model_id,
            }

            # Execute test with cache using modular function
            result = execute_test_with_cache(
                cached_result=cached_result,
                cache_result=cache_result,
                run_test_func=lambda device_id=device_id: run_device_test(
                    docker_client=docker_client,
                    device_id=device_id,
                    docker_image_tag=docker_image_tag,
                    docker_container_prefix=f"{test_name}_",
                    data_dir=data_dir,
                    container_mnt_dir="/mnt",
                    model_id=model_id,
                    model_precision=model_precision,
                    test_num_prompts=test_num_prompts,
                    test_request_rate=test_request_rate,
                    test_max_concurrent_requests=test_max_concurrent_requests,
                    server_timeout=server_timeout,
                    benchmark_timeout=benchmark_timeout,
                    dataset_path=dataset_path,
                    metrics=default_metrics,
                ),
                test_name=test_name,
                configs=configs,
                cache_configs=cache_configs,
            )

            device_results.append(result)

            # Update final results with device-specific metadata (matching original implementation)
            if not result.metadata.get("status", False):
                logger.error(f"Test failed for device {device_id}: {result.metadata.get('error', 'Unknown error')}")
                continue

            logger.info(f"Updating results for device {device_id}")
            results.metrics[metric_name].value = result.metrics[metric_name].value
            results.metadata[f"Device {device_id} status"] = result.metadata.get("status", False)
            results.metadata[f"Device {device_id} error"] = result.metadata.get("error", "No error")
            results.metadata[f"Device {device_id} Total Input Tokens"] = result.metadata.get("Total Input Tokens", -1.0)
            results.metadata[f"Device {device_id} Total Output Tokens"] = result.metadata.get(
                "Total Output Tokens", -1.0
            )
            results.metadata[f"Device {device_id} Output Throughput"] = result.metadata.get("Output Throughput", -1.0)
            results.metadata[f"Device {device_id} Mean TTFT (ms)"] = result.metadata.get("Mean TTFT (ms)", -1.0)
            results.metadata[f"Device {device_id} Mean TPOT (ms)"] = result.metadata.get("Mean TPOT (ms)", -1.0)

        # Process results using modular function
        process_device_results(
            results=device_results,
            device_list=device_list,
            final_results={"metrics": results.metrics, "metadata": results.metadata},
        )

        logger.debug(f"Text Generation Test results: {json.dumps(results.to_dict(), indent=2)}")

        # Check if test failed and store failure info
        if not results.metadata.get("status", False):
            test_failed = True
            error_message = results.metadata.get("error", "Unknown error")
            failure_message = f"Text Generation Test failed: {error_message}"

    except KeyboardInterrupt:
        failure_message = "Interrupt detected during Text Generation test execution"
        test_interrupted = True
        logger.error(failure_message)

    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during Text Generation test execution: {str(e)}"
        logger.error(failure_message, exc_info=True)

        # Create a minimal results object if none exists
        if results is None:
            default_metrics = [
                (get_metric_name_for_device(dev, prefix="throughput"), "tokens/sec") for dev in device_list
            ]
            metrics = {name: Metrics(unit=unit, value=-1.0) for name, unit in default_metrics}
            results = Result.from_test_config(
                configs=configs,
                parameters={
                    "Device": devices,
                    "Model ID": model_id,
                    "Error": str(e),
                },
                metrics=metrics,
                metadata={
                    "status": False,
                    "error": str(e),
                },
            )
        else:
            results.metadata["status"] = False
            results.metadata["error"] = str(e)

    finally:
        # Cleanup
        cleanup()

        # Step 4: Validate test results (always run to populate validation_results)
        try:
            validation_results = validate_test_results(
                results=results,
                configs=configs,
                get_kpi_config=get_kpi_config,
                test_name=test_name,
                mode=kpi_validation_mode,
            )

            # Update KPI validation status in result metadata
            results.update_kpi_validation_status(validation_results, kpi_validation_mode)

            # Automatically set key metric based on validation results and mode
            results.auto_set_key_metric(validation_results, kpi_validation_mode)

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
        except Exception as validation_error:
            logger.error(f"Validation failed: {validation_error}")
            validation_results = {"skipped": True, "skip_reason": "Validation failed due to errors"}

        # Step 5: Always summarize test results, regardless of test outcome
        try:
            logger.info("Generating test result summary")
            summarize_test_results(
                results=results,
                configs=configs,
                get_kpi_config=get_kpi_config,
                test_name=test_name,
            )
        except Exception as summary_error:
            logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)

        is_qualification = configs.get("labels", {}).get("type") == "qualification"

        if test_interrupted:
            if is_qualification:
                pytest.fail(failure_message)
            else:
                raise RuntimeError(failure_message)
        if test_failed:
            pytest.fail(failure_message)

    logger.info(f"Text Generation test '{test_name}' completed successfully")
