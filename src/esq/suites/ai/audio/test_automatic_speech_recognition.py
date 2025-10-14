# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import grp
import json
import logging
import os
import time
import traceback

import httpx
import pytest
import requests
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result
from sysagent.utils.infrastructure import download_and_prepare_audio
from sysagent.utils.system.ov_helper import get_available_devices_by_category

logger = logging.getLogger(__name__)


def test_automatic_speech_recognition(
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
    End-to-end ASR test: builds docker, runs inference, collects and validates results.
    """
    # Request
    test_name = request.node.name.split("[")[0]

    # Parameters
    test_display_name = configs.get("display_name", test_name)
    kpi_validation_mode = configs.get("kpi_validation_mode", "all")
    model_id = configs.get("model_id", "openai/whisper-tiny")
    devices = configs.get("devices", [])
    timeout = configs.get("timeout", 300)
    dockerfile_name = configs.get("dockerfile_name", "Dockerfile")
    docker_image_tag = f"{configs.get('container_image_name', 'automatic-speech-recognition')}:{configs.get('container_tag', 'latest')}"
    audio_url = configs.get(
        "audio_url",
        "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav",
    )
    audio_wav = configs.get("audio_wav", "how_are_you_doing_today.wav")
    audio_webm = configs.get("audio_webm", "how_are_you_doing_today.webm")

    # Setup
    core_data_dir = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "app_data"))
    asr_thirdparty_path = os.path.join(
        core_data_dir,
        "thirdparty",
        "edge-developer-kit-reference-scripts",
        "usecases",
        "ai",
        "microservices",
        "speech-to-text",
    )
    example_data_dir = os.path.join(core_data_dir, "data", "suites", "ai", "audio", "example_data")
    models_dir = os.path.join(core_data_dir, "data", "suites", "ai", "audio", "models")

    logger.info(f"Starting Automatic Speech Recognition Test: {test_display_name}")

    # Step 1: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Verify docker client connection
    from sysagent.utils.infrastructure import DockerClient

    docker_client = DockerClient()

    # Initialize variables for finally block (moved to top for broader coverage)
    validation_results = {}
    test_failed = False
    test_interrupted = False
    failure_message = ""
    results = None
    device_list = []

    try:
        # Get available devices based on device categories
        logger.info(f"Configured device categories: {devices}")
        device_dict = get_available_devices_by_category(device_categories=devices)

        # Extract device IDs to maintain compatibility with existing code
        device_list = list(device_dict.keys())
        logger.debug(f"Available devices: {device_dict}")
        if not device_dict:
            test_failed = True
            failure_message = f"No available devices found for device categories: {devices}"
            raise RuntimeError(failure_message)

        # Log detailed device information
        for device_id, device_info in device_dict.items():
            logger.debug(f"Device {device_id}: Type={device_info['device_type']}, Name={device_info['full_name']}")

        # Step 2: Prepare test environment
        def prepare_assets():
            build_result = docker_client.build_image(
                path=asr_thirdparty_path, tag=docker_image_tag, dockerfile=dockerfile_name
            )
            container_config = {
                "image_id": build_result.get("image_id", ""),
                "image_tag": docker_image_tag,
                "timeout": timeout,
                "dockerfile": os.path.join(asr_thirdparty_path, dockerfile_name),
                "build_path": asr_thirdparty_path,
            }
            audio_info = download_and_prepare_audio(
                url=audio_url,
                download_dir=example_data_dir,
                wav_filename=audio_wav,
                target_format="webm",
                target_filename=audio_webm,
            )
            result = Result(
                metadata={
                    "status": True,
                    "container_config": container_config,
                    "audio_info": audio_info,
                    "model_id": model_id,
                }
            )

            return result

        prepare_test(test_name=test_name, configs=configs, prepare_func=prepare_assets, name="Assets")

        # Clean up any leftover containers from previous test runs before starting
        try:
            logger.info("Cleaning up any leftover containers from previous test runs")
            docker_client.cleanup_containers_by_name_pattern("test_asr")
        except Exception as initial_cleanup_error:
            logger.warning(f"Initial cleanup failed: {initial_cleanup_error}")

        # Prepare result template for all devices
        default_metrics = [("inference_time", "secs")]
        current_kpi_refs = configs.get("kpi_refs", [])
        all_metrics = {name: unit for name, unit in default_metrics}
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
                "Timeout (s)": timeout,
                "Display Name": test_display_name,
            },
            metrics=metrics,
            metadata={
                "status": True,
            },
        )
        logger.debug(f"Initial Results template: {json.dumps(results.to_dict(), indent=2)}")
        # Step 3: Execute test for each device
        for device_id in device_list:
            logger.info(f"Running test for device: {device_id}")

            def run_test(device_id: str = device_id):
                logger.info(
                    f"Executing Automatic Speech Recognition test for model ID: {model_id} on device ID: {device_id}"
                )

                # Prepare result template for this device
                device_metrics = {kpi: Metrics(unit=unit, value=-1.0) for kpi, unit in all_metrics.items()}

                # Initialize result template
                result = Result(
                    parameters={
                        "Devices": devices,
                        "Device ID": device_id,
                        "Model ID": model_id,
                        "Display Name": test_display_name,
                    },
                    metrics=device_metrics,
                    metadata={
                        "status": False,
                    },
                )
                logger.debug(f"Initial Result template: {json.dumps(result.to_dict(), indent=2)}")

                # Prepare container environment
                container_name = f"test_asr_{model_id.replace('/', '_')}_{device_id.replace('.', '_')}".lower()
                environment = {
                    "DEFAULT_MODEL_ID": model_id,
                    "STT_DEVICE": device_id,
                }
                host_example_data_dir = example_data_dir
                host_models_dir = models_dir
                container_example_data_dir = "/usr/src/app/data/example_data"
                container_models_dir = "/usr/src/app/data/models"
                ports = {"5996/tcp": ("127.0.0.1", 5996)}
                user_gid = os.getuid()
                render_gid = grp.getgrnam("render").gr_gid
                volumes = {
                    host_example_data_dir: {"bind": container_example_data_dir, "mode": "rw"},
                    host_models_dir: {"bind": container_models_dir, "mode": "rw"},
                }

                # Ensure directories have correct permissions
                ensure_dir_permissions(host_example_data_dir, uid=os.getuid(), gid=os.getgid(), mode=0o775)
                ensure_dir_permissions(host_models_dir, uid=os.getuid(), gid=os.getgid(), mode=0o775)

                container_devices = ["/dev/dri:/dev/dri"]
                if os.path.exists("/dev/accel"):
                    container_devices.append("/dev/accel:/dev/accel")

                container = None
                try:
                    # Start container
                    container = docker_client.run_container(
                        name=container_name,
                        image=docker_image_tag,
                        environment=environment,
                        group_add=[render_gid, user_gid],
                        network_mode="bridge",
                        timeout=timeout,
                        ports=ports,
                        volumes=volumes,
                        devices=container_devices,
                        mode="server",
                    )

                    docker_client.start_log_streaming(container, container_name)

                    try:
                        api_host = "http://localhost:5996"
                        api_health = f"{api_host}/healthcheck"
                        api_url = f"{api_host}/v1/audio/transcriptions"

                        # Wait for service to be ready
                        def verify_service_ready(docker_client, container_name, url, max_retries=5, retry_interval=5):
                            retries = 0
                            while retries < max_retries:
                                if not docker_client.is_container_running(container_name):
                                    logger.error(f"Container {container_name} is not running.")
                                    return False
                                try:
                                    response = requests.get(url)
                                    if response.status_code == 200:
                                        logger.info(f"Service is ready at {url}")
                                        return True
                                except Exception as e:
                                    logger.debug(f"Health check failed: {e}")
                                retries += 1
                                if retries < max_retries:
                                    logger.info(f"Service not ready. Retrying {retries}/{max_retries}...")
                                    time.sleep(retry_interval)
                            logger.error(f"Health check failed after {max_retries} retries.")
                            return False

                        if not verify_service_ready(
                            docker_client, container_name, api_health, max_retries=20, retry_interval=30
                        ):
                            logger.error("Service is not ready. Skipping ...")
                            result.metadata["status"] = False
                            result.metadata["error"] = "Service not ready"
                            return result

                        # Now trigger ASR request
                        async def asr_request():
                            async with httpx.AsyncClient() as client:
                                audio_file_path = os.path.join(example_data_dir, audio_webm)
                                with open(audio_file_path, "rb") as audio_file:
                                    start_time = time.time()
                                    response = await client.post(
                                        api_url, files={"file": audio_file}, data={"model": model_id}
                                    )
                                    end_time = time.time()
                                    inference_time = round(end_time - start_time, 3)

                                if response.status_code == 200:
                                    response_data = response.json()
                                    result_text = response_data.get("text", "")
                                    status = True
                                else:
                                    result_text = f"Error: {response.status_code}"
                                    status = False
                                logger.debug(
                                    f"ASR response status: {response.status_code}, text: {result_text}, inference_time: {inference_time}"
                                )
                                result.metadata["status"] = status
                                result.metadata["Text"] = result_text
                                result.metrics["inference_time"].value = inference_time
                                logger.debug(
                                    f"ASR Result for {model_id} on {device_id}: {json.dumps(result.to_dict(), indent=2)}"
                                )
                                return result

                        try:
                            result = asyncio.run(asr_request())
                        except Exception as e:
                            exc_type = type(e).__name__
                            tb_str = traceback.format_exc()
                            error_message = f"Exception during ASR request: {exc_type}"
                            logger.error(error_message, exc_info=True)
                            result.metadata["error"] = error_message
                            allure.attach(
                                tb_str, name="ASR Exception Traceback", attachment_type=allure.attachment_type.TEXT
                            )
                        finally:
                            docker_client.stop_log_streaming(container_name)
                        return result

                    except Exception as e:
                        error_message = f"Exception in run_test for {model_id} on {device_id}: {e}"
                        logger.error(error_message, exc_info=True)
                        result.metadata["error"] = error_message
                        raise

                except Exception as inference_error:
                    error_message = f"Inference error for {model_id} on {device_id}: {inference_error}"
                    logger.error(error_message, exc_info=True)
                    result.metadata["error"] = error_message
                    result.metadata["status"] = False

                finally:
                    # Comprehensive container cleanup
                    logger.info(f"Stopping and removing ASR Docker container for {model_id} on {device_id}")
                    if container is not None:
                        try:
                            # Get container ID before cleanup
                            container_id = container.id
                            logger.info(f"Removing container: {container_id}")

                            # Use the proper cleanup method from DockerClient
                            docker_client.cleanup_container(container_name)

                        except Exception as cleanup_error:
                            warning_message = f"Failed to remove container {container.id if container else container_name} for {model_id}: {cleanup_error}"
                            logger.warning(warning_message)

                            # Fallback: try to forcefully clean up any containers with matching name
                            try:
                                logger.info(f"Attempting fallback cleanup for container: {container_name}")
                                docker_client.cleanup_containers_by_name_pattern(container_name)
                            except Exception as fallback_error:
                                logger.warning(f"Fallback cleanup also failed: {fallback_error}")
                    else:
                        logger.warning(f"Container was not created for {model_id}, checking for existing containers")
                        # Even if container creation failed, there might be leftover containers
                        try:
                            logger.info(f"Searching for leftover containers with name: {container_name}")
                            docker_client.cleanup_containers_by_name_pattern(container_name)
                        except Exception as leftover_search_error:
                            logger.warning(f"Could not search for leftover containers: {leftover_search_error}")

                return result

            # Execute the test with shared fixture
            device_result = execute_test_with_cache(
                cached_result=cached_result,
                cache_result=cache_result,
                test_name=test_name,
                configs=configs,
                run_test_func=lambda: run_test(device_id),
            )

            # Update main results with device-specific results
            if device_result and device_result.metadata.get("status", False):
                for metric_name, metric_obj in device_result.metrics.items():
                    if metric_name in results.metrics:
                        results.metrics[metric_name].value = metric_obj.value

                # Copy device-specific metadata
                for key, value in device_result.metadata.items():
                    if key.startswith("Device ") or key in ["Text", "status"]:
                        results.metadata[key] = value
            else:
                results.metadata["status"] = False
                if device_result and "error" in device_result.metadata:
                    results.metadata[f"Device {device_id} Error"] = device_result.metadata["error"]

            logger.debug(f"Device {device_id} test completed")

        # Final cleanup of any remaining containers
        try:
            logger.info("Performing final cleanup of any remaining containers")
            docker_client.cleanup_all_containers()

            # Additional cleanup for any containers that might have been missed
            docker_client.cleanup_containers_by_name_pattern("test_asr")

        except Exception as final_cleanup_error:
            logger.warning(f"Final cleanup failed: {final_cleanup_error}")

        logger.debug(f"ASR Test results: {json.dumps(results.to_dict(), indent=2)}")

        # Check if test failed and store failure info
        if not results.metadata.get("status", False):
            test_failed = True
            failure_message = f"ASR test failed: {results.metadata.get('error', 'Unknown error')}"

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

    except KeyboardInterrupt:
        failure_message = "Interrupt detected during ASR test execution"
        test_interrupted = True
        logger.error(failure_message, exc_info=True)

    except Exception as e:
        # Catch any unhandled exceptions and ensure they don't prevent visualization
        test_failed = True
        failure_message = f"Unexpected error during ASR test execution: {str(e)}"
        logger.error(failure_message, exc_info=True)

        # Create a minimal results object if none exists
        if results is None:
            default_metrics = [("inference_time", "secs")]
            current_kpi_refs = configs.get("kpi_refs", [])
            all_metrics = {name: unit for name, unit in default_metrics}
            for kpi in current_kpi_refs:
                kpi_config = get_kpi_config(kpi)
                if kpi_config:
                    all_metrics[kpi] = kpi_config.get("unit", "")
            metrics = {kpi: Metrics(unit=unit, value=-1.0) for kpi, unit in all_metrics.items()}

            results = Result.from_test_config(
                configs=configs,
                parameters={
                    "Device": devices,
                    "Device List": device_list,
                    "Model ID": model_id,
                    "Timeout (s)": timeout,
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
            results.auto_set_key_metric(validation_results, kpi_validation_mode)
        except Exception as validation_error:
            logger.error(f"Validation also failed: {validation_error}")
            validation_results = {"skipped": True, "skip_reason": "Validation failed due to test errors"}

    finally:
        # Step 5: Always summarize test results, regardless of test outcome
        try:
            if results is not None:
                logger.info("Generating test result summary (always executed)")
                summarize_test_results(
                    results=results, test_name=test_name, configs=configs, get_kpi_config=get_kpi_config
                )
            else:
                logger.warning("No results object available for summarization")
        except Exception as summary_error:
            logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)
            try:
                import allure

                if results is not None:
                    allure.attach(
                        json.dumps(results.to_dict(), indent=2),
                        name="Test Results (Fallback)",
                        attachment_type=allure.attachment_type.JSON,
                    )
                else:
                    allure.attach(
                        json.dumps(
                            {"error": "No results object available", "failure_message": failure_message}, indent=2
                        ),
                        name="Test Results (Fallback - No Results)",
                        attachment_type=allure.attachment_type.JSON,
                    )
            except Exception as fallback_error:
                logger.error(f"Even fallback attachment failed: {fallback_error}")

        if test_interrupted:
            raise RuntimeError(failure_message)
        if test_failed:
            pytest.fail(failure_message)

    logger.info(f"Automatic Speech Recognition test '{test_name}' completed successfully")
