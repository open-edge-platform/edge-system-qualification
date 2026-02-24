# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob
import logging
import os
import secrets
import string
import time
from pathlib import Path

import allure
import pytest
from sysagent.utils.core import Metrics, Result, run_command
from sysagent.utils.system.ov_helper import get_available_devices_by_category

logger = logging.getLogger(__name__)


def test_lp_vlm(
    request,
    configs,
    cached_result,
    cache_result,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """
    End-to-end Loss Prevention Visual Language Model Test.

    This test executes the Loss Prevention VLM workload which combines:
    - Object detection using YOLO models
    - Visual Language Model analysis for enhanced understanding
    - Message broker communication via RabbitMQ
    - Object storage via MinIO

    The test validates inference performance, accuracy, and system stability
    under VLM workload conditions.
    """
    # Initialize variables for finally block (moved to top for broader coverage)
    test_failed = False
    test_interrupted = False
    failure_message = ""
    results = None
    lp_base_dir = None

    # Step 1: Extract parameters from configs
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)
    timeout = configs.get("timeout", 3600)  # 60 minutes default for VLM
    devices = configs.get("devices", ["cpu"])
    camera_stream = configs.get("camera_stream", "camera_to_workload_vlm.json")
    commit_id = configs.get("commit_id", "2205dc9")  # Default fallback for backward compatibility
    huggingface_token = configs.get("huggingface_token", os.getenv("HUGGINGFACE_TOKEN"))
    logger.info(f"Starting Loss Prevention VLM Test: {test_display_name}")

    # Step 2: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Get available devices based on device categories
    logger.info(f"Configured device categories: {devices}")
    device_dict = get_available_devices_by_category(device_categories=devices)
    logger.debug(f"Available devices: {device_dict}")

    if not device_dict:
        pytest.fail(f"No available devices found for device categories: {devices}")

    # Setup paths
    core_data_dir = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    lp_base_dir = os.path.join(core_data_dir, "thirdparty", "loss-prevention")

    # Define cleanup function
    def cleanup():
        """Clean up Loss Prevention containers."""
        if lp_base_dir and os.path.exists(lp_base_dir):
            logger.info("Cleaning up Loss Prevention containers...")
            try:
                cleanup_cmd = ["make", "down-lp"]
                run_command(
                    cleanup_cmd,
                    cwd=lp_base_dir,
                    check=False,
                    stream_output=True,
                    timeout=60,
                )
                logger.info("✓ Loss Prevention container cleanup completed")
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")

    try:
        if not os.path.exists(lp_base_dir):
            # Clone repository if loss-prevention directory doesn't exist
            logger.info(f"Loss Prevention directory not found. Cloning repository to {lp_base_dir}")
            try:
                # Clone repository
                clone_cmd = ["git", "clone", "https://github.com/intel-retail/loss-prevention", lp_base_dir]
                logger.info(f"Executing: {' '.join(clone_cmd)}")
                clone_result = run_command(clone_cmd, check=True, stream_output=True, timeout=300)

                # Checkout specific commit
                checkout_cmd = ["git", "checkout", commit_id]
                logger.info(f"Checking out commit {commit_id}")
                checkout_result = run_command(checkout_cmd, cwd=lp_base_dir, check=True, stream_output=True, timeout=30)
                # Create necessary __pycache__ directories for Python imports
                # to prevent permission issues when running as non-root
                pycache_dirs = [
                    os.path.join(lp_base_dir, "lp-vlm", "src", "utils", "__pycache__"),
                    os.path.join(lp_base_dir, "lp-vlm", "src", "agent", "__pycache__"),
                ]

                for pycache_dir in pycache_dirs:
                    os.makedirs(pycache_dir, exist_ok=True)
                    logger.debug(f"Created __pycache__ directory: {pycache_dir}")

                checkout_result = run_command(checkout_cmd, cwd=lp_base_dir, check=True, stream_output=True, timeout=30)
                logger.info(f"Successfully cloned loss-prevention repository to {lp_base_dir}")
                # Combine clone and checkout outputs for allure attachment
                combined_output = (
                    f"=== Git Clone Output ===\n"
                    f"{clone_result.stdout + clone_result.stderr}\n\n"
                    f"=== Git Checkout Output ===\n"
                    f"{checkout_result.stdout + checkout_result.stderr}"
                )
                allure.attach(
                    combined_output,
                    name="Repository Setup Output",
                    attachment_type=allure.attachment_type.TEXT,
                )
            except Exception as e:
                error_msg = f"Error setting up repository: {str(e)}"
                logger.error(error_msg)
                pytest.fail(error_msg)
        else:
            logger.info(f"Loss Prevention directory already exists: {lp_base_dir}")

        # Step 3: Prepare assets/dependencies
        def prepare_assets():
            """Prepare Loss Prevention VLM assets and environment."""
            preparation_result = Result(name=f"{test_id} - Asset Preparation", metadata={"status": "in_progress"})

            try:
                # Validate configuration files exist
                camera_config_path = os.path.join(lp_base_dir, "configs", camera_stream)

                if not os.path.exists(camera_config_path):
                    raise FileNotFoundError(f"Camera config not found: {camera_config_path}")

                # Set required environment variables using character-by-character copying
                # for security (breaks Coverity taint chains)

                # Generate secure random passwords
                def generate_password(length=12):
                    """Generate secure random password using alphanumeric characters."""
                    alphabet = string.ascii_letters + string.digits
                    password = "".join(secrets.choice(alphabet) for _ in range(length))
                    return password

                env_vars = {
                    "MINIO_ROOT_USER": "user",
                    "MINIO_ROOT_PASSWORD": generate_password(),
                    "RABBITMQ_USER": "user",
                    "RABBITMQ_PASSWORD": generate_password(),
                    "GATED_MODEL": "true",
                    "HUGGINGFACE_TOKEN": huggingface_token if huggingface_token else "",
                    "DISPLAY": ":0",  # Ensure DISPLAY is set for any GUI components in VLM workload
                }

                # Create .env file in loss-prevention directory if it doesn't exist
                env_file_path = os.path.join(lp_base_dir, ".env")

                if not os.path.exists(env_file_path):
                    logger.info(f"Creating .env file with generated passwords at: {env_file_path}")
                    try:
                        with open(env_file_path, "w") as env_file:
                            for key, value in env_vars.items():
                                env_file.write(f"{key}={value}\n")
                        # Set restrictive file permissions for security
                        os.chmod(env_file_path, 0o600)
                        logger.info(f"Created .env file with {len(env_vars)} environment variables")
                    except Exception as e:
                        logger.warning(f"Failed to create .env file: {e}")
                        # Fall back to setting environment variables directly
                        for key, value in env_vars.items():
                            os.environ[key] = value
                            logger.debug(f"Set environment variable: {key}")
                else:
                    logger.info(f".env file already exists at: {env_file_path}")
                    # Read existing .env file and load into environment
                    try:
                        with open(env_file_path, "r") as env_file:
                            for line in env_file:
                                line = line.strip()
                                if line and not line.startswith("#"):
                                    if "=" in line:
                                        key, value = line.split("=", 1)
                                        os.environ[key] = value
                                        env_vars[key] = value
                        logger.info(f"Loaded {len(env_vars)} environment variables from existing .env file")
                    except Exception as e:
                        logger.warning(f"Failed to read existing .env file: {e}")

                # Ensure environment variables are set in current process for immediate use
                for key, value in env_vars.items():
                    os.environ[key] = value
                    logger.debug(f"Set environment variable: {key}")

                # Validate makefile exists
                makefile_path = os.path.join(lp_base_dir, "Makefile")
                if not os.path.exists(makefile_path):
                    raise FileNotFoundError(f"Makefile not found: {makefile_path}")

                preparation_result.metadata["status"] = "completed"
                preparation_result.metadata["lp_base_dir"] = lp_base_dir
                preparation_result.metadata["camera_config"] = camera_config_path
                preparation_result.metadata["environment_vars_set"] = list(env_vars.keys())

                logger.info("Loss Prevention VLM assets prepared successfully")
                return preparation_result

            except Exception as e:
                preparation_result.metadata["status"] = "failed"
                preparation_result.metadata["error"] = str(e)
                logger.error(f"Asset preparation failed: {e}")
                raise

        prepare_test(test_name=test_name, prepare_func=prepare_assets, configs=configs, name="Assets")

        # Step 4: Execute test logic (with caching)
        def execute_logic():
            """Execute the Loss Prevention VLM workload test."""
            results = Result(name=f"{test_id} - {test_display_name}")

            try:
                start_time = time.time()

                # Change to loss-prevention directory for make commands
                original_cwd = os.getcwd()
                os.chdir(lp_base_dir)

                try:
                    # Clean any existing containers first
                    logger.info("Cleaning any existing Loss Prevention containers...")
                    cleanup_cmd = ["make", "down-lp"]
                    run_command(
                        cleanup_cmd,
                        cwd=lp_base_dir,
                        check=False,  # Don't fail if nothing to clean
                        stream_output=True,
                        timeout=60,
                    )

                    # Prepare environment and run VLM workload
                    logger.info(f"Running Loss Prevention VLM workload with camera_stream={camera_stream}")

                    # Execute the VLM workload
                    lp_cmd = ["make", "run-lp", f"CAMERA_STREAM={camera_stream}", "RENDER_MODE=0", "REGISTRY=false"]

                    logger.info(f"Executing command: {' '.join(lp_cmd)}")
                    # Execute VLM workload with timeout
                    try:
                        lp_result = run_command(lp_cmd, cwd=lp_base_dir, timeout=timeout, stream_output=True)
                        if lp_result.returncode != 0:
                            logger.warning(f"Process ended with return code {lp_result.returncode}: {lp_result.stderr}")
                    except Exception as e:
                        logger.warning(f"VLM workload execution failed: {e}")

                    allure.attach(
                        lp_result.stdout + lp_result.stderr,
                        name="Pipeline Run Output",
                        attachment_type=allure.attachment_type.TEXT,
                    )
                    # Monitor vlm-pipeline-runner container logs to detect completion
                    logs_dir = os.path.join(core_data_dir, "data", "vertical", "retail")
                    os.makedirs(logs_dir, exist_ok=True)
                    logs_file = os.path.join(logs_dir, f"vlm_pipeline_logs_{int(time.time())}.txt")

                    max_attempts = 15
                    check_interval = 60
                    workload_completed = False
                    all_logs = ""

                    for attempt in range(max_attempts):
                        try:
                            # Get container logs using character-by-character copying for security
                            container_name = "vlm-pipeline-runner"

                            # Check container status first
                            inspect_cmd = [
                                "docker",
                                "inspect",
                                container_name,
                                "--format",
                                "{{.State.Status}} {{.State.ExitCode}}",
                            ]
                            logger.info(f"Checking container status (attempt {attempt + 1}/{max_attempts})")

                            inspect_result = run_command(inspect_cmd, stream_output=True, timeout=30)

                            container_status = None
                            container_exit_code = None

                            if inspect_result.returncode == 0:
                                status_output = inspect_result.stdout.strip().split()
                                if len(status_output) >= 2:
                                    container_status = status_output[0]
                                    try:
                                        container_exit_code = int(status_output[1])
                                    except (ValueError, IndexError):
                                        container_exit_code = None

                                    logger.info(
                                        f"Container status: {container_status}, exit code: {container_exit_code}"
                                    )

                            # Get container logs
                            logs_cmd = ["docker", "logs", container_name]
                            logs_result = run_command(logs_cmd, stream_output=True, timeout=30)

                            if logs_result.returncode == 0:
                                current_logs = logs_result.stdout + logs_result.stderr
                                all_logs = current_logs  # Keep latest full logs

                                # Check if container exited with code 0
                                if container_status == "exited" and container_exit_code == 0:
                                    logger.info("Container exited successfully with code 0")
                                    workload_completed = True
                                    break
                                elif container_status == "exited" and container_exit_code != 0:
                                    logger.warning(f"Container exited with non-zero code: {container_exit_code}")
                                    # Keep logs but don't mark as completed
                                    break
                            else:
                                logger.warning(f"Failed to get container logs: {logs_result.stderr}")

                            # Wait before next check (except on last attempt)
                            if attempt < max_attempts - 1:
                                logger.info(f"Waiting {check_interval} seconds before next check...")
                                time.sleep(check_interval)

                        except Exception as e:
                            logger.warning(f"Error checking container status/logs (attempt {attempt + 1}): {e}")

                    # Write collected logs to file
                    if all_logs:
                        try:
                            # Use restrictive permissions for security
                            with open(logs_file, "w") as f:
                                f.write(all_logs)
                            os.chmod(logs_file, 0o640)  # Restrictive file permissions
                            logger.info(f"Container logs written to: {logs_file}")

                            # Attach logs to allure report
                            allure.attach(
                                all_logs,
                                name="VLM Pipeline Container Logs",
                                attachment_type=allure.attachment_type.TEXT,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to write container logs: {e}")

                    run_duration = time.time() - start_time
                    if workload_completed:
                        logger.info(f"Workload completed early after {run_duration:.1f} seconds")
                    else:
                        logger.info(f"Workload completion not detected after {max_attempts} checks")

                    # Stop the pipeline
                    logger.info("Stopping Loss Prevention pipeline...")
                    stop_cmd = ["make", "down-lp"]
                    run_command(stop_cmd, cwd=lp_base_dir, stream_output=True, timeout=120)

                    end_time = time.time()
                    total_duration = end_time - start_time

                    # Collect metrics with KPI-compliant names
                    results.metadata["total_execution_time"] = round(total_duration, 2)

                    results.metadata["pipeline_run_duration"] = round(run_duration, 2)

                    # Extract VLM performance metrics from vlm_performance_metrics_*.txt files
                    results_dir = os.path.join(lp_base_dir, "results", "vlm-results")
                    vlm_metrics = {}
                    metric_counts = {}

                    try:
                        if os.path.exists(results_dir):
                            # Copy results to benchmark directory for preservation
                            import shutil

                            benchmark_dir = os.path.join(lp_base_dir, "benchmark")

                            # Check if results_dir is not empty
                            if os.listdir(results_dir):
                                # Create benchmark directory if it doesn't exist, clean it if it does
                                if os.path.exists(benchmark_dir):
                                    # Clean up existing files in benchmark directory
                                    for file_name in os.listdir(benchmark_dir):
                                        file_path = os.path.join(benchmark_dir, file_name)
                                        try:
                                            if os.path.isfile(file_path):
                                                os.unlink(file_path)
                                            elif os.path.isdir(file_path):
                                                shutil.rmtree(file_path)
                                        except Exception as cleanup_error:
                                            logger.warning(f"Failed to remove {file_path}: {cleanup_error}")
                                    logger.info("Cleaned up existing benchmark directory")
                                else:
                                    # Create benchmark directory
                                    os.makedirs(benchmark_dir, exist_ok=True)
                                    logger.info(f"Created benchmark directory: {benchmark_dir}")

                                # Copy all files from results_dir to benchmark_dir
                                try:
                                    for item_name in os.listdir(results_dir):
                                        src_path = os.path.join(results_dir, item_name)
                                        dst_path = os.path.join(benchmark_dir, item_name)
                                        if os.path.isfile(src_path):
                                            shutil.copy2(src_path, dst_path)
                                        elif os.path.isdir(src_path):
                                            shutil.copytree(src_path, dst_path)
                                    logger.info(
                                        f"Copied {len(os.listdir(results_dir))} items from results to benchmark directory"
                                    )
                                except Exception as copy_error:
                                    logger.warning(f"Failed to copy results to benchmark directory: {copy_error}")

                            get_latency_cmd = ["make", "consolidate-metrics"]
                            latency_results = run_command(
                                get_latency_cmd, cwd=lp_base_dir, stream_output=True, timeout=60
                            )
                            allure.attach(
                                latency_results.stdout + latency_results.stderr,
                                name="Consolidate Metrics Output",
                                attachment_type=allure.attachment_type.TEXT,
                            )
                            # Find all vlm_performance_metrics_*.txt files
                            performance_pattern = os.path.join(results_dir, "vlm_performance_metrics_*.txt")
                            performance_files = glob.glob(performance_pattern)

                            logger.info(f"Found {len(performance_files)} VLM performance metrics files")

                            # Process performance metrics files
                            for metrics_file in performance_files:
                                logger.debug(f"Processing performance metrics file: {metrics_file}")
                                with open(metrics_file, "r") as f:
                                    for line in f:
                                        if "vlm_metrics" in line and "Load_Time=" in line:
                                            # Parse metrics from log line
                                            # Format: timestamp - INFO - application=vlm_metrics ... Load_Time=X Generated_Tokens=Y ...
                                            parts = line.strip().split(" ")
                                            for part in parts:
                                                if (
                                                    "=" in part
                                                    and not part.startswith("application=")
                                                    and not part.startswith("timestamp_ms=")
                                                ):
                                                    try:
                                                        key, value = part.split("=", 1)
                                                        # Clean key name
                                                        clean_key = key.strip()
                                                        # Convert value to float
                                                        float_value = float(value)

                                                        # Skip negative values (Grammar_Compile metrics are -1.0 when not used)
                                                        if float_value >= 0:
                                                            if clean_key not in vlm_metrics:
                                                                vlm_metrics[clean_key] = 0.0
                                                                metric_counts[clean_key] = 0
                                                            vlm_metrics[clean_key] += float_value
                                                            metric_counts[clean_key] += 1
                                                    except (ValueError, IndexError):
                                                        continue

                        # Calculate averages and add to results
                        if vlm_metrics:
                            logger.info(f"Extracted VLM metrics from {sum(metric_counts.values())} total measurements")

                            for key, total_value in vlm_metrics.items():
                                count = metric_counts[key]
                                avg_value = total_value / count if count > 0 else 0.0

                                # Map metrics to standardized names and units
                                if key == "Throughput_Mean":
                                    results.metadata["throughput_mean"] = round(avg_value, 2)
                                elif key == "TTFT_Mean":
                                    results.metadata["ttft_mean"] = round(avg_value, 2) if avg_value > 0 else 0.0
                                    results.metrics["ft_throughput"] = Metrics(
                                        value=round(1000 / avg_value, 2) if avg_value > 0 else 0.0,
                                        unit="tokens/sec",
                                        is_key_metric=True,
                                    )
                                elif key == "TPOT_Mean":
                                    results.metadata["tpot_mean"] = round(avg_value, 2) if avg_value > 0 else 0.0
                                elif key == "Generate_Duration_Mean":
                                    results.metadata["generate_duration"] = round(avg_value, 2)
                                elif key == "Load_Time":
                                    results.metadata["load_time"] = round(avg_value, 2)
                                elif key == "Generated_Tokens":
                                    results.metadata["generated_tokens_avg"] = round(avg_value, 1)
                                elif key == "Input_Tokens":
                                    results.metadata["input_tokens_avg"] = round(avg_value, 1)
                            # Add total number of VLM inference calls after processing metrics
                            total_calls = max(metric_counts.values()) if metric_counts else 0
                            results.metadata["total_calls"] = total_calls

                            # Extract latency metrics from metrics.csv
                            metrics_csv_path = os.path.join(benchmark_dir, "metrics.csv")
                            if os.path.exists(metrics_csv_path):
                                logger.info(f"Processing latency metrics from: {metrics_csv_path}")
                                try:
                                    with open(metrics_csv_path, "r") as csvfile:
                                        import csv

                                        csv_reader = csv.reader(csvfile)
                                        for row in csv_reader:
                                            if len(row) >= 2:
                                                metric_name = row[0].strip()
                                                metric_value_str = row[1].strip()

                                                try:
                                                    metric_value = float(metric_value_str)

                                                    # Extract VLM verification latency
                                                    if "latency" in metric_name:
                                                        results.metrics["latency"] = Metrics(
                                                            value=round(metric_value, 2),
                                                            unit="ms",
                                                        )
                                                        logger.info(
                                                            f"Extracted VLM verification latency: {metric_value} ms"
                                                        )
                                                except ValueError:
                                                    # Skip non-numeric values
                                                    continue

                                    logger.info("Successfully processed latency metrics from metrics.csv")

                                except Exception as csv_error:
                                    logger.warning(f"Failed to process metrics.csv: {csv_error}")
                            else:
                                logger.info(f"metrics.csv not found at: {metrics_csv_path}")

                            # Set status to passed when VLM metrics are successfully extracted
                            results.metadata["status"] = "passed"
                            logger.info("VLM test completed successfully with metrics extracted")
                        else:
                            logger.warning("No VLM performance metrics found")
                            results.metadata["status"] = "failed"
                            results.metadata["error"] = "No VLM performance metrics found"

                    except Exception as e:
                        logger.error(f"Error extracting VLM metrics: {e}")
                        allure.attach(
                            str(e), name="VLM Metrics Extraction Error", attachment_type=allure.attachment_type.TEXT
                        )

                    # Add test parameters
                    results.parameters["Test ID"] = test_id
                    results.parameters["Display Name"] = test_display_name
                    results.parameters["Devices"] = ", ".join(devices)
                    results.parameters["Camera Stream Config"] = camera_stream
                    results.parameters["Run Duration"] = f"{run_duration}s"

                    # Update timestamps
                    results.update_timestamps()

                finally:
                    os.chdir(original_cwd)
                    # Find the performance metrics file dynamically
                    results_dir = os.path.join(lp_base_dir, "results", "vlm-results")
                    log_file_path = None

                    if os.path.exists(results_dir):
                        # Look for vlm_performance_metrics files
                        vlm_pattern = os.path.join(results_dir, "vlm_performance_metrics_*.txt")
                        vlm_files = glob.glob(vlm_pattern)

                        if vlm_files:
                            # Get the most recent file if multiple exist
                            log_file_path = Path(max(vlm_files, key=os.path.getctime))
                            logger.info(f"Found performance metrics file: {log_file_path}")
                        else:
                            logger.warning(f"No performance metrics files found in {results_dir}")

                    # Attach log file if found
                    if log_file_path and log_file_path.exists():
                        try:
                            with open(log_file_path, "r") as f:
                                allure.attach(
                                    f.read(),
                                    name="VLM Performance Metrics Log",
                                    attachment_type=allure.attachment_type.TEXT,
                                )
                            logger.debug(f"Attached performance metrics log: {log_file_path}")
                        except Exception as attach_error:
                            logger.warning(f"Failed to attach performance metrics log: {attach_error}")
                    else:
                        logger.debug("Performance metrics log file not found or doesn't exist")

            except Exception as e:
                results.metadata["status"] = "failed"
                results.metadata["error"] = str(e)
                logger.error(f"Loss Prevention VLM test execution failed: {e}")
                raise

            return results

        results = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=execute_logic,
            test_name=test_name,
            configs=configs,
        )

        # Check if test failed and store failure info
        if results.metadata.get("status") != "passed":
            test_failed = True
            failure_message = f"Loss Prevention VLM Test failed: {results.metadata.get('error', 'Test did not complete successfully')}"

    except KeyboardInterrupt:
        failure_message = "Interrupt detected during Loss Prevention VLM test execution"
        test_interrupted = True
        logger.error(failure_message)

        # Mark test as failed due to interruption
        if results is not None:
            results.metadata["status"] = False
            results.metadata["error"] = failure_message

            # Set metrics to indicate interruption
            for metric_name, metric in results.metrics.items():
                if hasattr(metric, "value"):
                    metric.value = -1

    except Exception as e:
        # Catch any unhandled exceptions and ensure they don't prevent summarization
        test_failed = True
        failure_message = f"Unexpected error during Loss Prevention VLM test execution: {str(e)}"
        logger.error(failure_message, exc_info=True)

        # Create minimal results object if none exists
        if results is None:
            results = Result(
                name=test_name,
                metadata={"status": False, "error": str(e)},
                metrics={},
                parameters={
                    "Test ID": test_id,
                    "Display Name": test_display_name,
                    "Devices": ", ".join(devices),
                    "Camera Stream Config": camera_stream,
                },
            )
        else:
            results.metadata["status"] = False
            results.metadata["error"] = str(e)

    finally:
        # Always run cleanup and summarization
        try:
            cleanup()
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed: {cleanup_error}")

        # Always summarize test results, regardless of test outcome
        try:
            if results is not None:
                summarize_test_results(results=results, test_name=test_name, configs=configs)
                logger.info("✓ Loss Prevention VLM test result summary completed successfully")
            else:
                logger.error("No results to summarize")
        except Exception as summary_error:
            logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)

        # Re-raise exceptions after cleanup
        if test_interrupted:
            pytest.fail(failure_message)
        elif test_failed:
            pytest.fail(failure_message)

    logger.info(f"Loss Prevention VLM test '{test_name}' completed successfully")
