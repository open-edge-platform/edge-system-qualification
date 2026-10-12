# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import glob
import logging
import os

import allure
import pytest
from sysagent.utils.core import Metrics, Result, run_command
from sysagent.utils.system.ov_helper import get_available_devices_by_category

logger = logging.getLogger(__name__)


@allure.description(
    "Run the Take-Away Order Accuracy fixed-workers benchmark to collect end-to-end "
    "VLM inference performance metrics. The workload ingests video streams, selects "
    "relevant frames with YOLO object detection, and validates orders using a "
    "VLM model via Intel® OpenVINO™ Model Server."
)
def test_ta_vlm(
    request,
    configs,
    cached_result,
    cache_result,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """Run the Take-Away Order Accuracy VLM benchmark to collect inference performance metrics."""
    test_failed = False
    test_interrupted = False
    failure_message = ""
    results = None
    ta_base_dir = None

    # ================================================================
    # STEP 1: Extract Parameters
    # ================================================================
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)
    timeout = configs.get("timeout", 1800)  # 30 minutes default
    devices = configs.get("devices", ["igpu"])
    commit_id = configs.get("commit_id", "65d1959")

    # Resolve actual OpenVINO device ID from category at runtime (matches test_text_generation pattern)
    device_dict = get_available_devices_by_category(device_categories=devices)
    if not device_dict:
        pytest.skip(f"No available devices found for device categories: {devices}")
    target_device = next(iter(device_dict))
    logger.info(f"Resolved device category {devices} -> {target_device}")

    # Sanitize numeric benchmark parameters
    benchmark_workers = max(1, int(configs.get("benchmark_workers", 1)))
    benchmark_duration = max(10, int(configs.get("benchmark_duration", 200)))
    benchmark_init_duration = max(5, int(configs.get("benchmark_init_duration", 10)))

    logger.info(f"Starting Take-Away Order Accuracy VLM Test: {test_display_name}")
    logger.info(
        f"Config: device={target_device}, workers={benchmark_workers}, "
        f"duration={benchmark_duration}s, init={benchmark_init_duration}s"
    )

    # ================================================================
    # STEP 2: Validate System Requirements
    # ================================================================
    validate_system_requirements_from_configs(configs)

    # Resolve paths
    core_data_dir = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    oa_base_dir = os.path.join(core_data_dir, "thirdparty", "order-accuracy")
    ta_base_dir = os.path.join(oa_base_dir, "take-away")

    def cleanup():
        """Stop Take-Away Docker* services and remove containers."""
        if ta_base_dir and os.path.exists(ta_base_dir):
            logger.info("Stopping Take-Away services...")
            try:
                run_command(
                    ["make", "down"],
                    cwd=ta_base_dir,
                    check=False,
                    stream_output=True,
                    timeout=60,
                )
                logger.info("Take-Away services stopped")
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")

    try:
        # Clone repository if not already present
        if not os.path.exists(oa_base_dir):
            logger.info(f"Cloning order-accuracy repository to {oa_base_dir}")
            try:
                clone_result = run_command(
                    ["git", "clone", "https://github.com/intel-retail/order-accuracy", oa_base_dir],
                    check=True,
                    stream_output=True,
                    timeout=300,
                )
                checkout_result = run_command(
                    ["git", "checkout", commit_id],
                    cwd=oa_base_dir,
                    check=True,
                    stream_output=True,
                    timeout=30,
                )
                submodule_result = run_command(
                    ["make", "update-submodules"],
                    cwd=ta_base_dir,
                    check=True,
                    stream_output=True,
                    timeout=120,
                )
                combined_output = (
                    "=== Clone ===\n"
                    f"{clone_result.stdout}{clone_result.stderr}\n\n"
                    "=== Checkout ===\n"
                    f"{checkout_result.stdout}{checkout_result.stderr}\n\n"
                    "=== Submodules ===\n"
                    f"{submodule_result.stdout}{submodule_result.stderr}"
                )
                allure.attach(
                    combined_output,
                    name="Repository Setup Output",
                    attachment_type=allure.attachment_type.TEXT,
                )
                logger.info(f"Cloned order-accuracy repository at commit {commit_id}")
            except Exception as clone_err:
                pytest.fail(f"Failed to clone order-accuracy repository: {clone_err}")
        else:
            logger.info(f"order-accuracy repository already exists: {oa_base_dir}")

        # ================================================================
        # STEP 3: Prepare Assets/Dependencies
        # ================================================================
        def prepare_assets():
            preparation_result = Result(
                name=f"{test_id} - Asset Preparation",
                metadata={"status": "in_progress"},
            )
            try:
                makefile_path = os.path.join(ta_base_dir, "Makefile")
                if not os.path.exists(makefile_path):
                    raise FileNotFoundError(f"Makefile not found: {makefile_path}")

                # Create .env via 'make init-env' (copies .env.example) only when it
                # doesn't already exist — the target prompts to overwrite an existing
                # .env interactively, which would hang an automated run.
                env_file_path = os.path.join(ta_base_dir, ".env")
                env_example_path = os.path.join(ta_base_dir, ".env.example")
                if not os.path.exists(env_file_path):
                    if not os.path.exists(env_example_path):
                        raise FileNotFoundError(f".env.example not found: {env_example_path}")
                    init_env_result = run_command(
                        ["make", "init-env"],
                        cwd=ta_base_dir,
                        check=False,
                        stream_output=True,
                        timeout=30,
                    )
                    allure.attach(
                        init_env_result.stdout + init_env_result.stderr,
                        name="Init Env Output",
                        attachment_type=allure.attachment_type.TEXT,
                    )
                    if not os.path.exists(env_file_path):
                        raise RuntimeError("make init-env did not create .env file")
                else:
                    logger.info(f".env already exists, skipping make init-env: {env_file_path}")

                # Apply device configuration on top of the template.
                # Both TARGET_DEVICE and OPENVINO_DEVICE must match per documentation.
                env_content = {}
                try:
                    with open(env_file_path, "r") as f:
                        for line in f:
                            stripped = line.strip()
                            if stripped and not stripped.startswith("#") and "=" in stripped:
                                key, value = stripped.split("=", 1)
                                env_content[key.strip()] = value.strip()
                except Exception as read_err:
                    logger.warning(f"Could not read {env_file_path}: {read_err}")

                device_str = target_device
                env_content["TARGET_DEVICE"] = device_str
                env_content["OPENVINO_DEVICE"] = device_str
                # Disable debug frame saving — frames written as root cause CI workspace
                # cleanup failures (permission denied). Not needed for benchmarking.
                env_content["DEBUG_FRAMES_ENABLED"] = "false"

                with open(env_file_path, "w") as f:
                    for key, value in env_content.items():
                        f.write(f"{key}={value}\n")
                os.chmod(env_file_path, 0o600)  # Restrictive file permissions for security
                logger.info(f"Configured .env: TARGET_DEVICE={device_str}, OPENVINO_DEVICE={device_str}")

                # Export model for the target device via ovms-service/setup_models.sh.
                # The script reads TARGET_DEVICE from take-away/.env (written above).
                # On repeat runs it detects the existing model and only patches
                # graph.pbtxt for the device — skipping the full HuggingFace download.
                ovms_script = os.path.join(oa_base_dir, "ovms-service", "setup_models.sh")
                if os.path.exists(ovms_script):
                    logger.info(f"Exporting/validating model for device {device_str}...")
                    export_result = run_command(
                        ["bash", ovms_script, "--app", "take-away"],
                        cwd=oa_base_dir,
                        check=False,
                        stream_output=False,
                        timeout=3600,  # 1 hour: first-time HuggingFace download + export
                    )
                    allure.attach(
                        export_result.stdout + export_result.stderr,
                        name="Model Export Output",
                        attachment_type=allure.attachment_type.TEXT,
                    )
                    if export_result.returncode != 0:
                        raise RuntimeError(
                            f"setup_models.sh failed (exit {export_result.returncode}): "
                            f"{export_result.stderr}"
                        )
                    logger.info("Model export/validation completed")
                else:
                    logger.warning(f"setup_models.sh not found at: {ovms_script}")

                # Create required directories (results/, metrics/, storage/videos/)
                run_command(
                    ["make", "setup-dirs"],
                    cwd=ta_base_dir,
                    check=False,
                    stream_output=True,
                    timeout=30,
                )

                # Download sample test video if not already present
                video_path = os.path.join(ta_base_dir, "storage", "videos", "test.mp4")
                if not os.path.exists(video_path):
                    logger.info("Downloading sample test video...")
                    download_result = run_command(
                        ["make", "download-sample-video"],
                        cwd=ta_base_dir,
                        check=False,
                        stream_output=False,
                        timeout=300,
                    )
                    allure.attach(
                        download_result.stdout + download_result.stderr,
                        name="Sample Video Download Output",
                        attachment_type=allure.attachment_type.TEXT,
                    )
                else:
                    logger.info(f"Sample video already present: {video_path}")

                preparation_result.metadata["status"] = "completed"
                preparation_result.metadata["ta_base_dir"] = ta_base_dir
                preparation_result.metadata["target_device"] = device_str
                logger.info("Take-Away assets prepared successfully")
                return preparation_result

            except Exception as prep_err:
                preparation_result.metadata["status"] = "failed"
                preparation_result.metadata["error"] = str(prep_err)
                logger.error(f"Asset preparation failed: {prep_err}")
                raise

        prepare_test(test_name=test_name, prepare_func=prepare_assets, configs=configs, name="Assets")

        # ================================================================
        # STEP 4: Execute Test Logic (with caching)
        # ================================================================
        def execute_logic():
            run_results = Result(name=f"{test_id} - {test_display_name}")

            try:
                # Run fixed-workers benchmark (benchmark-oa starts services automatically)
                logger.info(
                    f"Running benchmark-oa: workers={benchmark_workers}, "
                    f"duration={benchmark_duration}s, init={benchmark_init_duration}s, "
                    f"device={target_device}"
                )
                benchmark_cmd = [
                    "make",
                    "benchmark",
                    f"BENCHMARK_WORKERS={benchmark_workers}",
                    f"BENCHMARK_DURATION={benchmark_duration}",
                    f"BENCHMARK_INIT_DURATION={benchmark_init_duration}",
                    f"OPENVINO_DEVICE={target_device}",
                ]
                benchmark_result = run_command(
                    benchmark_cmd,
                    cwd=ta_base_dir,
                    check=False,
                    stream_output=True,
                    timeout=timeout,
                )
                allure.attach(
                    benchmark_result.stdout + benchmark_result.stderr,
                    name="Benchmark Run Output",
                    attachment_type=allure.attachment_type.TEXT,
                )

                results_dir = os.path.join(ta_base_dir, "results")

                run_results.metadata["benchmark_workers"] = benchmark_workers
                run_results.metadata["benchmark_duration"] = benchmark_duration
                run_results.metadata["target_device"] = target_device

                # ── Consolidate metrics and parse consolidated_metrics.csv ──
                logger.info("Consolidating benchmark metrics...")
                try:
                    consolidate_result = run_command(
                        ["make", "consolidate-metrics"],
                        cwd=ta_base_dir,
                        check=False,
                        stream_output=True,
                        timeout=60,
                    )
                    allure.attach(
                        consolidate_result.stdout + consolidate_result.stderr,
                        name="Consolidate Metrics Output",
                        attachment_type=allure.attachment_type.TEXT,
                    )
                except Exception as cons_err:
                    logger.warning(f"Metrics consolidation failed: {cons_err}")

                csv_path = os.path.join(results_dir, "consolidated_metrics.csv")
                if os.path.exists(csv_path):
                    try:
                        with open(csv_path, "r") as csvf:
                            reader = csv.reader(csvf)
                            for row in reader:
                                if len(row) < 2:
                                    continue
                                metric_key = row[0].strip()
                                try:
                                    metric_value = float(row[1].strip())
                                except ValueError:
                                    continue
                                if metric_key == "VLM_AVERAGE_CALL_DURATION":
                                    run_results.metrics["vlm_avg_call_duration"] = Metrics(
                                        value=round(metric_value, 4),
                                        unit="sec",
                                    )
                                elif metric_key == "VLM_TOTAL_CALLS":
                                    run_results.metadata["vlm_total_calls"] = int(metric_value)

                        with open(csv_path, "r") as csvf:
                            allure.attach(
                                csvf.read(),
                                name="Consolidated Metrics CSV",
                                attachment_type=allure.attachment_type.TEXT,
                            )
                    except Exception as csv_err:
                        logger.warning(f"Failed to parse consolidated_metrics.csv: {csv_err}")
                else:
                    logger.warning(f"consolidated_metrics.csv not found: {csv_path}")

                # ── Parse vlm_performance_metrics_*.txt ─────────────────────
                throughput_vals, tpot_vals, gen_dur_vals, gen_tok_vals = [], [], [], []
                perf_files = sorted(
                    glob.glob(os.path.join(results_dir, "vlm_performance_metrics_*.txt")),
                    key=os.path.getctime,
                )
                for perf_file in perf_files:
                    try:
                        with open(perf_file, "r") as pf:
                            for line in pf:
                                if "application=take-away-order-accuracy" not in line:
                                    continue
                                for token in line.strip().split():
                                    if "=" not in token:
                                        continue
                                    if token.startswith(("application=", "timestamp_ms=")):
                                        continue
                                    try:
                                        key, raw_val = token.split("=", 1)
                                        float_val = float(raw_val)
                                        if float_val < 0:
                                            continue
                                        if key == "throughput_mean_sec":
                                            throughput_vals.append(float_val)
                                        elif key == "tpot_sec":
                                            tpot_vals.append(float_val)
                                        elif key == "Generate_Duration_Mean":
                                            gen_dur_vals.append(float_val)
                                        elif key == "generated_tokens":
                                            gen_tok_vals.append(float_val)
                                    except (ValueError, IndexError):
                                        continue
                    except Exception as txt_err:
                        logger.warning(f"Could not parse {perf_file}: {txt_err}")

                if throughput_vals:
                    avg_tput = sum(throughput_vals) / len(throughput_vals)
                    run_results.metrics["throughput_mean_sec"] = Metrics(
                        value=round(avg_tput, 4),
                        unit="tokens/sec",
                        is_key_metric=True,
                    )
                    logger.info(
                        f"Average throughput: {avg_tput:.4f} tokens/sec "
                        f"({len(throughput_vals)} samples)"
                    )

                if tpot_vals:
                    run_results.metadata["avg_tpot_sec"] = round(sum(tpot_vals) / len(tpot_vals), 4)

                if gen_dur_vals:
                    run_results.metadata["avg_generate_duration_sec"] = round(sum(gen_dur_vals) / len(gen_dur_vals), 4)

                if gen_tok_vals:
                    run_results.metadata["avg_generated_tokens"] = round(sum(gen_tok_vals) / len(gen_tok_vals), 1)

                # ── Parse vlm_application_metrics_*.txt (ovms_vlm_request events) ──
                total_latency_vals, prompt_tok_vals, completion_tok_vals = [], [], []
                app_metric_files = sorted(
                    glob.glob(os.path.join(results_dir, "vlm_application_metrics_*.txt")),
                    key=os.path.getctime,
                )
                for app_file in app_metric_files:
                    try:
                        with open(app_file, "r") as af:
                            for line in af:
                                if "event=ovms_vlm_request" not in line:
                                    continue
                                for token in line.strip().split():
                                    if "=" not in token:
                                        continue
                                    if token.startswith(("application=", "timestamp_ms=", "id=", "event=")):
                                        continue
                                    try:
                                        key, raw_val = token.split("=", 1)
                                        float_val = float(raw_val)
                                        if float_val < 0:
                                            continue
                                        if key == "total_latency_sec":
                                            total_latency_vals.append(float_val)
                                        elif key == "prompt_tokens":
                                            prompt_tok_vals.append(float_val)
                                        elif key == "completion_tokens":
                                            completion_tok_vals.append(float_val)
                                    except (ValueError, IndexError):
                                        continue
                    except Exception as app_err:
                        logger.warning(f"Could not parse {app_file}: {app_err}")

                if total_latency_vals:
                    run_results.metadata["avg_total_latency_sec"] = round(sum(total_latency_vals) / len(total_latency_vals), 4)

                if prompt_tok_vals:
                    run_results.metadata["avg_prompt_tokens"] = round(sum(prompt_tok_vals) / len(prompt_tok_vals), 1)

                if completion_tok_vals:
                    run_results.metadata["avg_completion_tokens"] = round(sum(completion_tok_vals) / len(completion_tok_vals), 1)

                # Domain-specific result parameters (do not include test_id or display_name)
                run_results.parameters["Devices"] = ", ".join(devices)
                run_results.parameters["Target Device"] = target_device
                run_results.parameters["Benchmark Workers"] = str(benchmark_workers)
                run_results.parameters["Benchmark Duration"] = f"{benchmark_duration}s"

                has_metrics = bool(run_results.metrics)
                run_results.metadata["status"] = "passed" if has_metrics else "failed"
                if not has_metrics:
                    run_results.metadata["error"] = "No benchmark metrics were collected"

                run_results.update_timestamps()
                return run_results

            except Exception as exec_err:
                run_results.metadata["status"] = "failed"
                run_results.metadata["error"] = str(exec_err)
                logger.error(f"Benchmark execution error: {exec_err}")
                raise

            finally:
                # Always attach VLM performance and application metrics logs if present
                perf_files = sorted(
                    glob.glob(os.path.join(ta_base_dir, "results", "vlm_performance_metrics_*.txt")),
                    key=os.path.getctime,
                )
                if perf_files:
                    try:
                        with open(perf_files[-1], "r") as pf:
                            allure.attach(
                                pf.read(),
                                name="VLM Performance Metrics Log",
                                attachment_type=allure.attachment_type.TEXT,
                            )
                    except Exception as attach_err:
                        logger.warning(f"Failed to attach performance metrics log: {attach_err}")

                app_files = sorted(
                    glob.glob(os.path.join(ta_base_dir, "results", "vlm_application_metrics_*.txt")),
                    key=os.path.getctime,
                )
                if app_files:
                    try:
                        with open(app_files[-1], "r") as af:
                            allure.attach(
                                af.read(),
                                name="VLM Application Metrics Log",
                                attachment_type=allure.attachment_type.TEXT,
                            )
                    except Exception as attach_err:
                        logger.warning(f"Failed to attach application metrics log: {attach_err}")

        results = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=execute_logic,
            test_name=test_name,
            configs=configs,
        )

        if results.metadata.get("status") != "passed":
            test_failed = True
            failure_message = (
                f"Take-Away VLM benchmark failed: "
                f"{results.metadata.get('error', 'Test did not complete successfully')}"
            )

    except KeyboardInterrupt:
        failure_message = "Interrupt detected during Take-Away VLM test execution"
        test_interrupted = True
        logger.error(failure_message)
        if results is not None:
            results.metadata["status"] = False
            results.metadata["error"] = failure_message

    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during Take-Away VLM test execution: {str(e)}"
        logger.error(failure_message, exc_info=True)
        if results is None:
            results = Result(
                name=test_name,
                metadata={"status": False, "error": str(e)},
                metrics={},
                parameters={
                    "Devices": ", ".join(devices),
                    "Target Device": target_device,
                },
            )
        else:
            results.metadata["status"] = False
            results.metadata["error"] = str(e)

    finally:
        try:
            cleanup()
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed: {cleanup_error}")

        try:
            if results is not None:
                summarize_test_results(results=results, test_name=test_name, configs=configs)
                logger.info("Take-Away VLM test result summary completed")
            else:
                logger.error("No results to summarize")
        except Exception as summary_error:
            logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)

        if test_interrupted:
            pytest.fail(failure_message)
        elif test_failed:
            pytest.fail(failure_message)

    logger.info(f"Take-Away VLM test '{test_name}' completed successfully")
