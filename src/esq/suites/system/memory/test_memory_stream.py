# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System Memory Performance Test using STREAM benchmark.
"""

import logging
import os
from dataclasses import is_dataclass
from pathlib import Path
import allure
import pandas as pd
import pytest
from esq.suites.system.memory.memory_utils import init_csv_file, update_csv_from_result
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result
from sysagent.utils.infrastructure import DockerClient

logger = logging.getLogger(__name__)

test_container_path = "src/containers/memory_bcmk/"

def _create_mem_metrics(value: str = "N/A", unit: str = None) -> dict:
    """
    Create memory performance metrics dictionary.

    Args:
        value: Initial value for all metrics (default: "N/A")
        unit: Unit for metrics (default: None for N/A values)

    Returns:
        Dictionary of Metrics objects for memory performance
    """
    return {
        "best_rate": Metrics(unit=unit, value=value, is_key_metric=True),
        "avg_time": Metrics(unit=unit, value=value, is_key_metric=False),
        "min_time": Metrics(unit=unit, value=value, is_key_metric=False),
        "max_time": Metrics(unit=unit, value=value, is_key_metric=False),
    }

@allure.title("System Memory Performance Test (STREAM)")
def test_memory_stream(
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

    logger.info(f"Starting System Memory Performance Test (STREAM): {test_display_name}")

    operation = configs.get("operation", "Copy")
    dockerfile_name = configs.get("dockerfile_name", "Dockerfile")
    container_image = configs.get("container_image", "stream_memory_benchmark")
    image_tag = configs.get("image_tag", "1.0")
    docker_image_tag = f"{container_image}:{image_tag}"
    container_name = f"{container_image}_{operation.lower()}"

    stream_url = configs.get(
        "stream_git_url",
        "https://github.com/jeffhammond/STREAM.git",
    )

    # Ensure timeout is an integer (Converting in case YAML has string)
    timeout = int(configs.get("timeout", 300))

    # Step 1: Validate system requirements (CPU, memory, storage, Docker, etc.)
    validate_system_requirements_from_configs(configs)

    # Setup
    test_dir = os.path.dirname(os.path.abspath(__file__))
    docker_dir = os.path.join(test_dir, test_container_path)

    # Use esq_data folder for results (consistent with other suites)
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    core_data_dir = "".join(c for c in core_data_dir_tainted)
    data_dir = os.path.join(core_data_dir, "data", "system", "memory")
    mem_results = os.path.join(data_dir, "results", test_id)
    os.makedirs(mem_results, exist_ok=True)

    # Ensure directories have correct permissions
    ensure_dir_permissions(mem_results, uid=os.getuid(), gid=os.getgid(), mode=0o775)

    docker_client = DockerClient()

    # Initialize variables for error handling
    test_failed = False
    failure_message = ""
    results = None

    try:
        # Step 2: Prepare test environment
        def prepare_assets():
            # Access outer scope variables
            nonlocal docker_image_tag, dockerfile_name, docker_dir, timeout

            # Build 1: Get FW custom device-specific images from dlstreamer preparation
            from esq.suites.ai.vision.src.dlstreamer.preparation import (
                prepare_assets as prepare_dlstreamer_assets,
            )

            test_dir_abs = os.path.dirname(os.path.abspath(__file__))
            # Navigate to ai/vision to access dlstreamer src
            ai_vision_dir = os.path.join(test_dir_abs, "..", "..", "ai", "vision")
            src_dir = os.path.join(ai_vision_dir, "src")
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

            # Build 2: Select FW custom image based on device
            fw_custom_base_image = fw_container_config.get("analyzer_image", "intel/dlstreamer:2025.2.0-ubuntu24")
            logger.info(f"Using FW image as base for memory test: {fw_custom_base_image}")

            docker_nocache = configs.get("docker_nocache", False)
            logger.info(f"Docker build cache setting: nocache={docker_nocache}")
            logger.info(
                f"Build 2: Building test suite image '{docker_image_tag}' on top of FW custom image..."
            )

            build_args = {
                "COMMON_BASE_IMAGE": fw_custom_base_image,  # FW custom image
                "STREAM_GIT_URL": f"{stream_url}",
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
                    "message": "Memory BM - STREAM is the de facto industry standard benchmark",
                    "container_config": container_config,
                    "timeout(s)": timeout,
                    "display_name": test_display_name,
                }
            )

            return result

    except KeyboardInterrupt:
        failure_message = (
            f"User interrupt (Ctrl+C) detected during Memory Benchmark test preparation. "
            f"Test: {test_display_name}, Operation: {operation}. "
            f"Partial setup may be incomplete."
        )
        logger.error(failure_message)
        raise

    except Exception as e:
        test_failed = True
        failure_message = (
            f"Unexpected error during test preparation: {type(e).__name__}: {str(e)}. "
            f"Test: {test_display_name}, Operation: {operation}, Docker image: {docker_image_tag}. "
            f"Check logs for full stack trace and error details."
        )
        logger.error(failure_message, exc_info=True)
        logger.debug(f"Preparation context - Docker dir: {docker_dir}")
        # Don't raise yet - create N/A result below

    try:
        prepare_test(test_name=test_name, configs=configs, prepare_func=prepare_assets, name="Mem_BM_Assets")
    except Exception as prep_error:
        # Handle docker build or other preparation failures
        test_failed = True
        failure_message = (
            f"Test preparation failed during asset setup: {type(prep_error).__name__}: {str(prep_error)}. "
            f"Possible causes: Docker build failure, network issues, or dependency problems. "
            f"Docker image: {docker_image_tag}, STREAM URL: {stream_url}. "
            f"Check logs for detailed error and verify Docker daemon is running."
        )
        logger.error(failure_message, exc_info=True)
        logger.debug(f"Preparation failed - Docker dir: {docker_dir}, Timeout: {timeout}s")

    # If preparation failed, return N/A metrics immediately
    if test_failed:
        metrics = _create_mem_metrics(value="N/A", unit=None)

        results = Result.from_test_config(
            configs=configs,
            parameters={
                "timeout(s)": timeout,
                "display_name": test_display_name,
                "operation": operation,
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
        },
    )

    try:

        def run_test():
            # Access outer scope variables
            nonlocal docker_client, mem_results, docker_image_tag, container_name, timeout, operation

            # Get test case name from config for consistent use in CSV
            tc_name = configs.get("test_case_name", test_display_name)

            # Define metrics with N/A as initial values (unit will be set when value is populated)
            metrics = _create_mem_metrics(value="N/A", unit=None)

            remap_header = {
                "Best Rate MB/s": "best_rate(MB/s)",
                "Avg time": "avg_time(s)",
                "Min time": "min_time(s)",
                "Max time": "max_time(s)",
            }

            # Initialize result template using from_test_config for automatic metadata application
            result = Result.from_test_config(
                configs=configs,
                parameters={
                    "test_id": test_id,
                    "operation_type": operation,
                    "display_name": test_display_name,
                },
                metrics=metrics,
                metadata={
                    "status": "N/A",
                },
            )

            try:
                logger.info(f"Executing STREAM benchmark with operation: {operation}")

                # Initialize CSV file
                csv_file_path = Path(f"{mem_results}/memory_benchmark_runner.csv")
                init_csv_file(csv_file_path, tc_name=tc_name, force=True)

                # Setup container parameters
                container_home = "/home/dlstreamer"
                result_file_path = Path(f"{mem_results}/memory_benchmark_runner.result")
                volumes = {mem_results: {"bind": f"{container_home}/output", "mode": "rw"}}

                # Run container using docker_client
                try:
                    # Run container with automatic log attachment to Allure
                    docker_client.run_container(
                        name=container_name,
                        image=docker_image_tag,
                        volumes=volumes,
                        remove=True,
                        timeout=timeout,
                        mode="batch",
                        attach_logs=True,
                    )

                    # Update CSV from result file
                    test_passed = update_csv_from_result(
                        result_file_path=result_file_path,
                        csv_file_path=csv_file_path,
                        tc_name=tc_name,
                    )

                    if not test_passed:
                        error_msg = (
                            f"Container execution completed but test validation failed for operation '{operation}'. "
                            f"Container: {container_name}, Image: {docker_image_tag}. "
                            f"Check result file for error details: {result_file_path}"
                        )
                        logger.error(error_msg)

                        # Attach result file if available
                        if result_file_path.exists():
                            try:
                                with open(result_file_path, "r") as f:
                                    result_content = f.read()
                                allure.attach(
                                    result_content,
                                    name=f"Result File - {operation}",
                                    attachment_type=allure.attachment_type.TEXT,
                                )
                            except Exception as attach_err:
                                logger.warning(f"Failed to attach result file: {attach_err}")

                        result.metadata["failure_reason"] = error_msg
                        result.metadata["status"] = "N/A"
                        return result

                except Exception as container_error:
                    error_msg = (
                        f"Container execution failed for operation '{operation}': "
                        f"{type(container_error).__name__}: {str(container_error)}. "
                        f"Container: {container_name}, Image: {docker_image_tag}. "
                        f"Check logs for detailed error information."
                    )
                    logger.error(error_msg, exc_info=True)

                    result.metadata["failure_reason"] = error_msg
                    result.metadata["status"] = "N/A"
                    return result

                csv_res_path = Path(f"{mem_results}/memory_bm_{operation}.csv")

                if csv_file_path.exists():
                    df = pd.read_csv(csv_file_path)
                    # Filter rows where Function Name matches the given value
                    df_filtered = df[df["Function"].str.strip().str.upper() == operation.upper()]
                    # Rename column(s) based on the mapping, since no control over STREAM BMs result header
                    df_filtered = df_filtered.rename(columns=remap_header)
                    df_filtered.to_csv(csv_res_path, index=False)

                    if not df_filtered.empty:
                        row = df_filtered.iloc[0]  # extract the single row as Series

                        for key, val in result.metrics.items():
                            key_norm = key.replace("_", " ").strip().lower()

                            match = next(
                                (
                                    col
                                    for col in row.index
                                    if col.split("(")[0].replace("_", " ").strip().lower() == key_norm
                                ),
                                None,
                            )

                            if match:
                                new_val = row[match]
                                # Convert np.float64 or np.int64 to native Python types
                                if hasattr(new_val, "item"):
                                    new_val = new_val.item()
                                if is_dataclass(val):
                                    # Safely update the value field
                                    setattr(val, "value", new_val)
                                    # Set appropriate unit based on metric type
                                    if "rate" in key:
                                        setattr(val, "unit", "MB/s")
                                    elif "time" in key:
                                        setattr(val, "unit", "s")
                                    logger.debug(
                                        f"Updated metric '{key}' = {new_val}, unit={val.unit} from column '{match}'"
                                    )
                                else:
                                    # If val is not a Metrics dataclass, log warning and update directly
                                    logger.warning(f"Metric '{key}' is not a Metrics dataclass, updating directly")
                                    result.metrics[key] = new_val
                            else:
                                logger.warning(
                                    f"No matching column found for metric '{key}' (normalized: '{key_norm}')"
                                )

                        # Log final metrics state before returning
                        logger.info(f"Final metrics before return: {result.metrics}")
                    else:
                        error_msg = (
                            f"No data found for operation '{operation}' in CSV file. "
                            f"CSV file was found but no rows matched the operation filter. "
                            f"Verify CSV format and operation name match."
                        )
                        logger.error(error_msg)
                        result.metadata["failure_reason"] = error_msg
                        result.metadata["status"] = "N/A"
                        return result
                else:
                    # CSV file not found - keep all metrics as N/A and mark as failure
                    error_msg = (
                        f"Results CSV file not found at expected location: {csv_file_path}. "
                        f"Test container may have failed to generate results. "
                        f"Expected file: memory_benchmark_runner.csv in {mem_results}. "
                        f"Check container logs for execution errors."
                    )
                    logger.error(error_msg)
                    results_dir_contents = (
                        list(Path(mem_results).iterdir()) if Path(mem_results).exists() else "Directory not found"
                    )
                    logger.debug(f"Results directory contents: {results_dir_contents}")
                    result.metadata["failure_reason"] = "Results CSV file not generated by test container"
                    result.metadata["status"] = "N/A"
                    return result

                # Check if we collected valid metrics
                valid_metrics = [m for m in result.metrics.values() if m.value != "N/A"]
                if not valid_metrics:
                    metric_names = list(result.metrics.keys())
                    error_msg = (
                        f"Test completed but no valid metrics were collected (all N/A). "
                        f"Expected metrics: {', '.join(metric_names)}. "
                        f"CSV file was found but metric extraction failed. "
                        f"Verify CSV format matches expected structure with 'Function' column."
                    )
                    logger.error(error_msg)
                    logger.debug(f"CSV file location: {csv_file_path}")
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
                    f"Operation: {operation}, Test: {test_display_name}. "
                    f"Check logs for stack trace and detailed error information."
                )
                logger.error(error_msg, exc_info=True)
                logger.debug(f"Execution context - Container: {container_name}, Results dir: {mem_results}")
                result.metadata["failure_reason"] = error_msg
                # Metrics remain as N/A
                return result

            return result
    except KeyboardInterrupt:
        failure_message = "Interrupt detected during Memory Benchmark Test"
        logger.error(failure_message)

    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during Memory Benchmark Test: {str(e)}"
        logger.error(failure_message, exc_info=True)

    # Execute the test with shared fixture
    results = execute_test_with_cache(
        cached_result=cached_result,
        cache_result=cache_result,
        test_name=test_name,
        configs=configs,
        run_test_func=run_test,
    )

    # Handle N/A status (test failures)
    if results.metadata.get("status") == "N/A" and "failure_reason" in results.metadata:
        failure_msg = results.metadata["failure_reason"]
        logger.error(f"Test failed with N/A status: {failure_msg}")
        logger.info(f"Test summary - ID: {test_id}, Operation: {operation}")

        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )

        pytest.fail(f"Memory Benchmark test failed - {failure_msg}")

    # Validate test results against KPIs
    validate_test_results(results=results, configs=configs, get_kpi_config=get_kpi_config, test_name=test_name)
    try:
        logger.info(f"Generating test result visualizations (always executed) Results: {results}")

        # Attach CSV report if available
        csv_file_path = Path(f"{mem_results}/memory_bm_{operation}.csv")
        if csv_file_path.exists():
            try:
                df = pd.read_csv(csv_file_path)
                # Rename all columns: replace '_' with space, and title-case each word
                df.columns = [col.replace("_", " ").title() for col in df.columns]
                df.to_csv(csv_file_path, index=False)
                file_name = os.path.basename(csv_file_path)
                with open(csv_file_path, "rb") as f:
                    allure.attach(f.read(), name=file_name, attachment_type=allure.attachment_type.CSV)
                logger.debug(f"Attached CSV report: {file_name}")
            except Exception as attach_error:
                logger.warning(f"Failed to attach CSV report: {attach_error}")
        else:
            logger.debug(f"CSV report not found: {csv_file_path}")

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
        logger.debug(f"Summary context - Results dir: {mem_results}, Operation: {operation}")

    if test_failed:
        pytest.fail(failure_message)

    logger.info(f"System Memory Performance test '{test_name}' completed successfully")
