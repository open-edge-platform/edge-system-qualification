# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Robotics AI testing using FastMapping Benchmark
"""

import grp
import json
import logging
import os
from pathlib import Path

import allure
import pytest
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result
from sysagent.utils.infrastructure import DockerClient

logger = logging.getLogger(__name__)

test_directory = "fastmapping"
container_path = f"src/containers/{test_directory}/"


def _create_metrics(value: str = "N/A", unit: str = None) -> dict:  # type: ignore
    """
    Create performance metrics dictionary.

    Args:
        value: Initial value for all metrics (default: "N/A")
        unit: Unit for metrics (default: None for N/A values)

    Returns:
        Dictionary of Metrics objects for performance
    """
    return {
        "throughput": Metrics(unit=unit, value=value, is_key_metric=True),
        "mean_latency": Metrics(unit=unit, value=value, is_key_metric=False),
        "min_jitter": Metrics(unit=unit, value=value, is_key_metric=False),
        "max_jitter": Metrics(unit=unit, value=value, is_key_metric=False),
        "mean_jitter": Metrics(unit=unit, value=value, is_key_metric=False),
        "jitter_stdev": Metrics(unit=unit, value=value, is_key_metric=False),
    }


def _parse_results_file(results_file_path: Path) -> dict:  # type: ignore
    """
    Parse the benchmark results file and extract metrics.

    Args:
        results_file_path: Path to the benchmark results JSON file.

    Returns:
        Dictionary of Metrics objects extracted from the results file.
    """
    with open(results_file_path, "r", encoding="utf-8") as results_file:
        report = json.load(results_file)

    execution_results = report.get("execution_results", {})

    return {
        "throughput": Metrics(unit="Hz", value=float(execution_results["throughput"]), is_key_metric=True),
        "mean_latency": Metrics(unit="ms", value=float(execution_results["mean_latency"]), is_key_metric=False),
        "min_jitter": Metrics(unit="ms", value=float(execution_results["min_jitter"]), is_key_metric=False),
        "max_jitter": Metrics(unit="ms", value=float(execution_results["max_jitter"]), is_key_metric=False),
        "mean_jitter": Metrics(unit="ms", value=float(execution_results["mean_jitter"]), is_key_metric=False),
        "jitter_stdev": Metrics(unit="ms", value=float(execution_results["jitter_stdev"]), is_key_metric=False),
        "total_iterations": Metrics(
            unit="iterations",
            value=int(execution_results["iterations"]),
            is_key_metric=False,
        ),
    }


@allure.title("Robotics - FastMapping Benchmark")
def test_robotics_fastmapping(
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
    # ================================================================
    # STEP 1: Extract Parameters
    # ================================================================
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)
    timeout = int(configs.get("timeout", 300))
    device = configs.get("device", "gpu")
    operation = configs.get("operation", test_directory)
    dockerfile_name = configs.get("dockerfile_name", "Dockerfile")
    container_image = configs.get("container_image", f"robotics_{test_directory}_benchmark")
    image_tag = configs.get("image_tag", "1.0")
    docker_image_tag = f"{container_image}:{image_tag}"
    container_name = f"{container_image}_{operation.lower()}"

    # Qualification tests fail the run on interrupt; data-collection tests
    # surface it as a runtime error instead. This flag drives that decision.
    is_qualification = configs.get("labels", {}).get("type") == "qualification"

    logger.info(f"Starting test: {test_display_name}")

    # ================================================================
    # STEP 2: Validate System Requirements
    # ================================================================
    validate_system_requirements_from_configs(configs)

    # Setup
    test_dir = os.path.dirname(os.path.abspath(__file__))
    docker_dir = os.path.join(test_dir, container_path)

    # Use esq_data folder for results (consistent with other suites)
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    core_data_dir = "".join(c for c in core_data_dir_tainted)
    data_dir = os.path.join(core_data_dir, "data", "vertical", "robotics")
    test_results = os.path.join(data_dir, "results", test_id)
    os.makedirs(test_results, exist_ok=True)

    # Ensure directories have correct permissions
    ensure_dir_permissions(test_results, uid=os.getuid(), gid=os.getgid(), mode=0o775)

    docker_client = DockerClient()

    # Initialize variables for error handling
    results = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    def cleanup():
        try:
            docker_client.stop_container(container_name)
        except Exception as cleanup_error:
            logger.warning(f"Could not stop container '{container_name}' during cleanup: {cleanup_error}")

    try:
        # ============================================================
        # STEP 3: Prepare Assets/Dependencies
        # ============================================================
        def prepare_assets():
            # Access outer scope variables
            nonlocal docker_image_tag, dockerfile_name, docker_dir, timeout

            docker_base_image = configs.get("docker_base_image", "amd64/ros:jazzy-ros-base")
            docker_nocache = configs.get("docker_nocache", False)
            logger.info(f"Docker build cache setting: nocache={docker_nocache}")
            logger.info(f"Build 2: Building test suite image '{docker_image_tag}'.")

            build_args = {
                "COMMON_BASE_IMAGE": docker_base_image,
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
                name=f"{test_id} - Docker Image Build",
                metadata={
                    "status": True,
                    "message": test_display_name,
                    "container_config": container_config,
                    "timeout(s)": timeout,
                    "display_name": test_display_name,
                },
            )

            return result

        try:
            prepare_test(
                test_name=test_name,
                prepare_func=prepare_assets,
                configs=configs,
                name=f"{test_display_name}_Assets",
            )
        except Exception as prep_error:
            # Handle docker build or other preparation failures
            test_failed = True
            failure_message = (
                f"Test preparation failed during asset setup: {type(prep_error).__name__}: {str(prep_error)}. "
                f"Possible causes: Docker build failure, network issues, or dependency problems. "
                f"Docker image: {docker_image_tag}. "
                f"Check logs for detailed error and verify Docker daemon is running."
            )
            logger.error(failure_message, exc_info=True)
            logger.debug(f"Preparation failed - Docker dir: {docker_dir}, Timeout: {timeout}s")

        # ============================================================
        # STEP 4: Execute Test Logic (with caching)
        # ============================================================
        def execute_logic():
            # Access outer scope variables
            nonlocal docker_client, test_results, docker_image_tag, container_name, timeout, operation, device

            # Define metrics with N/A as initial values (unit will be set when value is populated)
            metrics = _create_metrics(value="N/A", unit=None)  # type: ignore

            # Initialize result template using from_test_config for automatic metadata application
            result = Result.from_test_config(
                name=f"{test_id} - {test_display_name}",
                configs=configs,
                parameters={
                    "device": device,
                    "operation_type": operation,
                },
                metrics=metrics,
                metadata={
                    "status": False,
                },
            )

            try:
                logger.info(f"Executing {test_display_name} with operation: {operation}")

                # Initialize results file
                results_file_path = Path(f"{test_results}/benchmark_report.json")

                # Remove stale benchmark artifacts from a previous run before launching
                # the container. The container writes these as its own unprivileged user,
                # whose UID can change when the image is rebuilt (created via `useradd -r`).
                # A leftover file owned by the previous build's UID is not writable by the
                # new container user and causes a "Permission denied" failure when the
                # benchmark tries to overwrite it. The host process owns the results
                # directory, so it can always clear the contents.
                for stale_file in Path(test_results).glob("benchmark*report*.json"):
                    try:
                        stale_file.unlink()
                    except OSError as cleanup_error:
                        logger.warning(f"Could not remove stale results file {stale_file}: {cleanup_error}")

                # Setup container parameters
                container_home = "/home/appuser"
                volumes = {test_results: {"bind": f"{container_home}/output", "mode": "rw"}}
                environment = {
                    "DEVICE": str(device),
                }

                # Prepare container devices for GPU access
                container_devices = []
                if device == "GPU":
                    # Add GPU devices
                    container_devices.extend(
                        [
                            "/dev/dri:/dev/dri",
                        ]
                    )

                if device == "NPU":
                    # Add NPU devices
                    container_devices.extend(
                        [
                            "/dev/accel/accel0:/dev/accel/accel0",
                        ]
                    )

                # Get render group GID
                try:
                    render_gid = grp.getgrnam("render").gr_gid
                except KeyError:
                    render_gid = 109  # Default render GID
                    logger.warning(f"'render' group not found, using default GID: {render_gid}")

                user_gid = os.getgid()

                logger.debug(f"Running container {container_name}")
                logger.debug(f"Devices: {container_devices}")
                logger.debug(f"Environment: {environment}")

                # Run container using docker_client
                try:
                    # Run container with automatic log attachment to Allure
                    docker_client.run_container(
                        name=container_name,
                        image=docker_image_tag,
                        volumes=volumes,
                        environment=environment,
                        devices=container_devices,
                        group_add=[render_gid, user_gid],
                        cap_add=[
                            "SYS_ADMIN",
                        ],
                        remove=True,
                        timeout=timeout,
                        attach_logs=True,
                    )

                except Exception as container_error:
                    error_msg = (
                        f"Container execution failed for operation '{operation}': "
                        f"{type(container_error).__name__}: {str(container_error)}. "
                        f"Container: {container_name}, Image: {docker_image_tag}. "
                        f"Check logs for detailed error information."
                    )
                    logger.error(error_msg, exc_info=True)

                    result.metadata["failure_reason"] = error_msg
                    result.metadata["status"] = False
                    return result

                if results_file_path.exists():
                    # Parse results file (JSON) and populate metrics
                    result.metrics = _parse_results_file(results_file_path)
                else:
                    # Result file not found - keep all metrics as N/A and mark as failure
                    error_msg = (
                        f"Results file not found at expected location: {results_file_path}. "
                        f"Test container may have failed to generate results. "
                        f"Expected file: benchmark_report.json in {test_results}. "
                        f"Check container logs for execution errors."
                    )
                    logger.error(error_msg)
                    results_dir_contents = (
                        list(Path(test_results).iterdir()) if Path(test_results).exists() else "Directory not found"
                    )
                    logger.debug(f"Results directory contents: {results_dir_contents}")
                    result.metadata["failure_reason"] = "Results file not generated by test container"
                    result.metadata["status"] = False
                    return result

                # Check if we collected valid metrics
                valid_metrics = [m for m in result.metrics.values() if m.value != "N/A"]
                if not valid_metrics:
                    metric_names = list(result.metrics.keys())
                    error_msg = (
                        f"Test completed but no valid metrics were collected (all N/A). "
                        f"Expected metrics: {', '.join(metric_names)}. "
                        f"Results file was found but metric extraction failed. "
                    )
                    logger.error(error_msg)
                    logger.debug(f"Results file location: {results_file_path}")
                    result.metadata["failure_reason"] = error_msg
                    result.metadata["status"] = False
                    return result

                # If successfully processed and collected valid metrics, mark as success
                result.metadata.pop("failure_reason", None)  # Remove failure_reason if test succeeded
                result.metadata["status"] = True
                result.update_timestamps()

            except Exception as exec_error:
                # Handle any execution errors (shell script failures, results file parsing, etc.)
                error_msg = (
                    f"Test execution failed with exception: {type(exec_error).__name__}: {str(exec_error)}. "
                    f"Operation: {operation}, Test: {test_display_name}. "
                    f"Check logs for stack trace and detailed error information."
                )
                logger.error(error_msg, exc_info=True)
                logger.debug(f"Execution context - Container: {container_name}, Results dir: {test_results}")
                result.metadata["failure_reason"] = error_msg
                result.metadata["status"] = False
                return result

            return result

        if not test_failed:
            results = execute_test_with_cache(
                cached_result=cached_result,
                cache_result=cache_result,
                run_test_func=execute_logic,
                test_name=test_name,
                configs=configs,
            )

        # Mark the test failed when the execution reported a non-passing status.
        if not results.metadata.get("status", False):
            test_failed = True
            failure_message = results.metadata.get("failure_reason", f"{test_display_name} failed")

    except KeyboardInterrupt:
        # Ctrl+C or run cancellation — record it and re-surface after cleanup.
        test_interrupted = True
        failure_message = "Interrupt detected during test execution"
        logger.error(failure_message)

    except Exception as error:
        test_failed = True
        # Any unexpected error — capture the message and continue to reporting.
        failure_message = f"Unexpected error during {test_display_name} Test: {str(error)}"
        logger.error(failure_message, exc_info=True)

    finally:
        # Always release resources, even on failure or interrupt.
        cleanup()

    # ================================================================
    # Ensure a result object always exists so validation and
    # summarization can record the outcome — even when execution was
    # interrupted before producing results.
    # ================================================================
    if results is None:
        metrics = _create_metrics(value="N/A", unit=None)  # type: ignore

        results = Result.from_test_config(
            name=f"{test_id} - {test_display_name}",
            configs=configs,
            parameters={
                "device": device,
                "operation_type": operation,
            },
            metrics=metrics,
            metadata={
                "status": False,
            },
            extended_metadata={
                "message": failure_message or "Test did not complete"
            },
        )

    # ================================================================
    # STEP 5: Validate Results Against KPIs (optional)
    # Only enforced when kpi_refs are configured for the test. Wrapped so a
    # validation error never masks the underlying test outcome.
    # ================================================================
    try:
        validate_test_results(
            results=results,
            configs=configs,
            get_kpi_config=get_kpi_config,
            test_name=test_name,
        )
    except Exception as validation_error:
        logger.error(f"Validation failed: {validation_error}")

    # ================================================================
    # STEP 6: Generate Summary (always runs)
    # ================================================================
    try:
        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as summary_error:
        logger.error(f"Summarization failed: {summary_error}", exc_info=True)

    cache_result(results)

    # ================================================================
    # STEP 7: Surface the outcome
    # Report interrupts and failures as proper pytest results instead of
    # leaving a broken/errored status behind.
    # ================================================================
    if test_interrupted:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)

    if test_failed:
        logger.error(f"Test failed with status: {failure_message}")
        logger.info(f"Test summary - ID: {test_id}, Operation: {operation}")
        pytest.fail(f"Robotics test '{test_name}' failed - {failure_message}")

    logger.info(f"Robotics test '{test_name}' completed successfully")
