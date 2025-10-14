# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Docker system performance test module using Sysbench.
"""

import json
import logging
import os

import allure
import pytest

from sysagent.utils.core import Metrics, Result

logger = logging.getLogger(__name__)


@allure.title("Docker System Performance Test (Sysbench)")
def test_docker_system_performance(
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
    Test system performance using sysbench within an Ubuntu 24.04 Docker container.

    This test builds a Docker container with sysbench, runs performance benchmarks
    inside it, and measures CPU, memory, file I/O, or mutex performance
    characteristics of the underlying system.
    """
    # Request
    test_name = request.node.name.split("[")[0]

    # Parameters
    test_name = configs.get("name", test_name)
    test_display_name = configs.get("display_name", test_name)
    kpi_validation_mode = configs.get("kpi_validation_mode", "all")
    test_duration = configs.get("test_duration", 10)
    test_threads = configs.get("test_threads", 1)
    test_type = configs.get("test_type", "cpu")
    memory_size = configs.get("memory_size", "1G")
    file_size = configs.get("file_size", "1G")
    timeout = configs.get("timeout", 300)
    docker_image_tag = (
        f"{configs.get('container_image_name', 'example-sysbench-test')}:{configs.get('container_tag', 'latest')}"
    )

    logger.info(f"Starting Docker System Performance Test (Sysbench): {test_name}")
    logger.debug(
        f"Test parameters: duration={test_duration}, threads={test_threads}, "
        f"type={test_type}, memory_size={memory_size}, file_size={file_size}, "
        f"timeout={timeout}"
    )

    # Step 1: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Verify docker client connection
    from sysagent.utils.infrastructure import DockerClient

    docker_client = DockerClient()

    # Step 2: Prepare test environment
    def prepare_test_function():
        # Get current directory (where the test is located)
        test_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(test_dir, "src")

        # Build docker image with shared fixture
        docker_nocache = configs.get("docker_nocache", False)
        build_result = docker_client.build_image(
            path=src_dir,
            tag=docker_image_tag,
            nocache=docker_nocache,
        )

        # Create container configuration attachment
        container_config = {
            "image_id": build_result.get("image_id", ""),
            "image_tag": docker_image_tag,
            "test_duration": test_duration,
            "test_threads": test_threads,
            "test_type": test_type,
            "memory_size": memory_size,
            "file_size": file_size,
            "timeout": timeout,
            "test_directory": test_dir,
        }

        result = Result(
            metadata={
                "status": True,
                "container_config": container_config,
            }
        )

        return result

    # Prepare test environment using shared fixture
    prepare_test(test_name=test_name, configs=configs, prepare_func=prepare_test_function, name="Assets")

    # Step 3: Execute test
    def run_test():
        logger.info("Running Sysbench benchmark in Docker container")

        # Dynamically build data keys from current_kpi_refs
        current_kpi_refs = configs.get("kpi_refs", [])
        metrics = {kpi: Metrics(unit=get_kpi_config(kpi).get("unit", ""), value=-1.0) for kpi in current_kpi_refs}

        # Initialize result template
        result = Result(
            parameters={
                "Test Type": test_type,
                "Test Thread": test_threads,
                "Memory Size": memory_size,
                "File Size": file_size,
                "Duration": test_duration,
                "Display Name": test_display_name,
            },
            metrics=metrics,
            metadata={
                "status": False,
            },
        )

        # Prepare environment and volume mount for results
        environment = {
            "TEST_DURATION": str(test_duration),
            "TEST_THREADS": str(test_threads),
            "TEST_TYPE": test_type,
            "MEMORY_SIZE": memory_size,
            "FILE_SIZE": file_size,
        }
        result_file = "results.json"
        container_result_file_dir = "/results"
        command = "/bin/bash -c 'cd /app && ./sysbench_benchmark.sh'"

        try:
            # Get host group ID for adding to container user
            host_gid = os.getgid()

            sysbench_result = docker_client.run_container(
                image=docker_image_tag,
                command=command,
                environment=environment,
                user="ubuntu",  # Use built-in Ubuntu 24.04 user
                group_add=[str(host_gid)],  # Add host group for file access
                network_mode="bridge",
                detach=True,
                remove=True,
                timeout=timeout,
                result_file=result_file,
                container_result_file_dir=container_result_file_dir,
            )
            logger.debug(f"Sysbench container result: {json.dumps(sysbench_result, indent=2)}")

            # Extract and process results
            if sysbench_result["result_json"]:
                test_results = sysbench_result["result_json"]

                # Exclude 'raw_output' from test_results when logging
                test_results_log = dict(test_results)
                test_results_log.pop("raw_output", None)
                logger.debug(f"Extracted test results: {json.dumps(test_results_log, indent=2)}")

                # Extract KPI units from kpi_refs and configs
                current_kpi_refs = configs.get("kpi_refs", [])
                logger.debug(f"Current KPI references: {current_kpi_refs}")
                for kpi_name in current_kpi_refs:
                    logger.debug(f"Processing KPI: {kpi_name}")
                    if kpi_name == "latency_avg_ms":
                        value = test_results.get("latency_ms", {}).get("avg", 0.0)
                    elif kpi_name == "latency_max_ms":
                        value = test_results.get("latency_ms", {}).get("max", 0.0)
                    else:
                        value = test_results.get(kpi_name, 0.0)
                    unit = get_kpi_config(kpi_name).get("unit", "")
                    result.metrics[kpi_name].value = value
                    result.metrics[kpi_name].unit = unit
                    logger.debug(f"KPI {kpi_name}: value={value}, unit={unit}")
                result.metadata["status"] = True

                return result
            else:
                raise RuntimeError("Could not extract test results from container.")
        except Exception as e:
            error_msg = f"Exception during sysbench result extraction: {e}"
            result.metadata["error"] = error_msg
            logger.error(error_msg, exc_info=True)
            raise

    # Initialize variables for finally block
    validation_results = {}
    results = None
    test_failed = False
    failure_message = ""

    try:
        # Execute the test with shared fixture
        results = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            test_name=test_name,
            configs=configs,
            run_test_func=run_test,
        )
        logger.debug(f"Test results: {json.dumps(results.to_dict(), indent=2)}")

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

    except Exception as e:
        # Catch any unhandled exceptions and ensure they don't prevent visualization
        test_failed = True
        failure_message = f"Unexpected error during Docker system performance test execution: {str(e)}"
        logger.error(failure_message, exc_info=True)

        # Create a minimal results object if none exists
        if results is None:
            current_kpi_refs = configs.get("kpi_refs", [])
            metrics = {kpi: Metrics(unit=get_kpi_config(kpi).get("unit", ""), value=-1.0) for kpi in current_kpi_refs}
            results = Result(
                parameters={
                    "Test Type": test_type,
                    "Test Thread": test_threads,
                    "Memory Size": memory_size,
                    "File Size": file_size,
                    "Duration": test_duration,
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
            validation_results = {
                "skipped": True,
                "skip_reason": "Validation failed due to test errors",
            }

    finally:
        try:
            logger.info("Generating test result visualizations (always executed)")

            # Summarize results using the shared fixture
            summarize_test_results(
                results=results,
                test_name=test_name,
                configs=configs,
                get_kpi_config=get_kpi_config,
            )
        except Exception as summary_error:
            logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)
            try:
                import allure

                if results:
                    allure.attach(
                        json.dumps(results.to_dict(), indent=2),
                        name="Test Results (Fallback)",
                        attachment_type=allure.attachment_type.JSON,
                    )
            except Exception as fallback_error:
                logger.error(f"Even fallback attachment failed: {fallback_error}")

        if test_failed:
            pytest.fail(failure_message)

    logger.info(f"Docker system performance test '{test_name}' completed successfully")
