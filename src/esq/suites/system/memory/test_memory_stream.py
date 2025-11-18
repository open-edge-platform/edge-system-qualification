# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System Memory Performance Test using STREAM benchmark.
"""

import logging
import os
import shutil
import allure
import pandas as pd
import pytest

from dataclasses import is_dataclass
from pathlib import Path
from esq.suites.summary.vertical.summary_utils import *
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result
from sysagent.utils.infrastructure import DockerClient

logger = logging.getLogger(__name__)

test_container_path = "src/containers/memory_bcmk/"

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
    docker_image_tag = f"{configs.get('container_image', 'stream_memory_benchmark')}:{configs.get('image_tag', '3.0')}"

    stream_url = configs.get(
        "stream_git_url",
        "https://github.com/jeffhammond/STREAM.git",
    )

    timeout = configs.get("timeout", 300)
    base_image = configs.get("base_image", "intel/dlstreamer:2025.1.2-dev-ubuntu24")

    # Step 1: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Setup
    test_dir = os.path.dirname(os.path.abspath(__file__))
    docker_dir = os.path.join(test_dir, test_container_path)

    mem_results = f"{docker_dir}/mem_results"
    os.makedirs(mem_results, exist_ok=True)

    # Ensure directories have correct permissions
    ensure_dir_permissions(mem_results, uid=os.getuid(), gid=os.getgid(), mode=0o775)

    docker_client = DockerClient()

    # Initialize variables for finally block (moved to top for broader coverage)
    validation_results = {}
    test_failed = False
    test_interrupted = False
    failure_message = ""
    results = None
    device_list = []

    try:
        # Step 2: Prepare test environment
        def prepare_assets():
            docker_nocache = configs.get("docker_nocache", False)

            build_args = {
                "COMMON_BASE_IMAGE": f"{base_image}",
                "STREAM_GIT_URL": f"{stream_url}"
            }

            build_result = docker_client.build_image(
                path=docker_dir,
                tag=docker_image_tag,
                nocache=docker_nocache,
                dockerfile=dockerfile_name,
                buildargs=build_args
            )
            container_config = {
                "image_id": build_result.get("image_id", ""),
                "image_tag": docker_image_tag,
                "timeout": timeout,
                "dockerfile": os.path.join(docker_dir, dockerfile_name),
                "build_path": docker_dir
            }
            result = Result(
                metadata={
                    "status": True,
                    "message": "Memory BM - STREAM is the de facto industry standard benchmark",
                    "container_config": container_config,
                    "timeout (s)": timeout,
                    "display Name": test_display_name
                }
            )

            return result

    except KeyboardInterrupt:
        failure_message = "Interrupt detected during Memory Benchmark Test"
        test_interrupted = True
        logger.error(failure_message)

    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during Memory Benchmark Test: {str(e)}"
        logger.error(failure_message, exc_info=True)

    prepare_test(test_name=test_name, configs=configs, prepare_func=prepare_assets, name="Mem_BM_Assets")

    # Initialize results template using from_test_config for automatic metadata application
    results = Result.from_test_config(
        configs=configs,
        parameters={
            "timeout (s)": timeout,
            "display Name": test_display_name
        }
    )

    try:

        def run_test():
            # Define metrics with default units
            metrics = {
                "best_rate": Metrics(unit="MB/s", value=-2.0),
                "avg_time": Metrics(unit="s", value=-1.0),
                "min_time": Metrics(unit="s", value=-2.0),
                "max_time": Metrics(unit="s", value=-1.0)
            }
            remap_header = {
                "Best Rate MB/s": "best_rate(MB/s)",
                "Avg time": "avg_time(s)",
                "Min time": "min_time(s)",
                "Max time": "max_time(s)"
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
                    "status": False,
                    "timeout (s)": timeout,
                    "display_name": test_display_name,
                },
            )

            run_shell_script(f"{test_dir}/src/run_mem_container.sh", mem_results)

            # Store simple metadata values (no nested objects)
            result.metadata["status"] = True

            csv_file_path = Path(f"{mem_results}/memory_benchmark_runner.csv")
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
                            None
                        )

                        if match:
                            new_val = row[match]
                            # Convert np.float64 or np.int64 to native Python types
                            if hasattr(new_val, "item"):
                                new_val = new_val.item()
                            if is_dataclass(val):
                                # Safely update the value field
                                setattr(val, "value", new_val)
                                logger.debug(f"Updated metric '{key}' = {new_val} from column '{match}'")
                            else:
                                # If val is not a Metrics dataclass, log warning and update directly
                                logger.warning(f"Metric '{key}' is not a Metrics dataclass, updating directly")
                                result.metrics[key] = new_val
                        else:
                            logger.warning(f"No matching column found for metric '{key}' (normalized: '{key_norm}')")

                    # Log final metrics state before returning
                    logger.info(f"Final metrics before return: {result.metrics}")

            else:
                logger.info("Warning: No metrics collected!.")
            return result
    except KeyboardInterrupt:
        failure_message = "Interrupt detected during Memory Benchmark Test"
        test_interrupted = True
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

    validation_results = validate_test_results(
        results=results, configs=configs, get_kpi_config=get_kpi_config, test_name=test_name
    )
    try:
        logger.info(f"Generating test result visualizations (always executed) Results: {results}")

        csv_file_path = Path(f"{mem_results}/memory_bm_{operation}.csv")

        if csv_file_path.exists():
            df = pd.read_csv(csv_file_path)
            # Rename all columns: replace '_' with space, and title-case each word
            df.columns = [col.replace("_", " ").title() for col in df.columns]
            df.to_csv(csv_file_path, index=False)
            file_name = os.path.basename(csv_file_path)
            with open(csv_file_path, "rb") as f:
                allure.attach(f.read(), name=file_name, attachment_type=allure.attachment_type.CSV)
        else:
            logger.error(f"Failed to find the output CSV File, {csv_file_path}")

        tar_file_path = Path(f"{mem_results}/memory_bm_runner.tar.gz")

        create_tar(mem_results, tar_file_path, True)

        if tar_file_path.exists():
            file_name = os.path.basename(tar_file_path)
            with open(tar_file_path, "rb") as f:
                allure.attach(f.read(), name=file_name, attachment_type="application/gzip")

        # Summarize results using the shared fixture
        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )

        shutil.rmtree(mem_results, ignore_errors=True)
    except Exception as summary_error:
        logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)

        if test_failed:
            pytest.fail(failure_message)

    logger.info(f"System Memory Performance test '{test_name}' completed successfully")
