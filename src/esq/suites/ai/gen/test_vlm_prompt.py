# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import logging
import os
import string

import allure
import pytest
from esq.suites.ai.gen.src.text_generation.container import run_ovms_server_container
from esq.suites.ai.gen.src.text_generation.utils import cleanup_stale_containers
from esq.suites.ai.gen.src.vlm_prompt.execution import (
    aggregate_records,
    aggregate_ttft,
    latency_metric_ms,
    read_records,
    run_client,
    run_stream_density,
)
from esq.suites.ai.gen.src.vlm_prompt.preparation import (
    prepare_media,
    prepare_model,
    resolve_rest_workers,
    setup_client_venv,
)
from sysagent.utils.core import Metrics, Result, TestResultCache, get_metric_name_for_device
from sysagent.utils.infrastructure import DockerClient
from sysagent.utils.system.ov_helper import get_available_devices_by_category

logger = logging.getLogger(__name__)


@allure.description(
    "Benchmarks a vision-language model (VLM) served by Intel® OpenVINO™ Model Server with the prompt and "
    "images or video of the profile. Fixed load mode runs concurrent workers for a set duration and reports "
    "VLM latency and token throughput. Stream density mode adds workers until the latency target is exceeded "
    "and reports the highest number of workers within it. Only the model server and a load generator run, so "
    "a prompt-based application, for example order accuracy, is described by its profile as a baseline "
    "without the rest of the application."
)
def test_vlm_prompt(
    request,
    configs,
    cached_result,
    cache_result,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """Benchmark a prompt-based VLM workload and report latency and token throughput."""
    test_failed = False
    test_interrupted = False
    failure_message = ""
    results = None

    # ================================================================
    # STEP 1: Extract Parameters
    # ================================================================
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)
    timeout = configs.get("timeout", 1800)
    devices = configs.get("devices", ["igpu"])

    model_id = configs.get("model_id")
    system_prompt = configs.get("system_prompt", "")
    prompt = configs.get("prompt")
    prompt_params = configs.get("prompt_params", {})
    media = configs.get("media", [])
    if not model_id or not prompt or not media:
        pytest.fail("Profile must define 'model_id', 'prompt' and 'media' for test_vlm_prompt")

    # Resolve the actual OpenVINO device IDs from the categories at runtime. A category can match
    # several devices (for example multiple dGPUs), and the test is run once on each of them.
    device_dict = get_available_devices_by_category(device_categories=devices)
    if not device_dict:
        pytest.skip(f"No available devices found for device categories: {devices}")
    device_list = list(device_dict.keys())
    logger.info(f"Resolved device category {devices} -> {device_list}")

    # benchmark_mode: "fixed_workers" runs benchmark_workers for benchmark_duration. "stream_density"
    # adds workers until the latency exceeds target_latency_ms and reports the highest passing count.
    # ttft_requests streaming requests are sent right after the server start: the first one gives the cold start
    # time to first token and the others the warm start. 0 turns the measurement off.
    ttft_requests = max(0, int(configs.get("ttft_requests", 4)))
    benchmark_mode = configs.get("benchmark_mode", "fixed_workers")
    if benchmark_mode not in ("fixed_workers", "stream_density"):
        pytest.fail(f"Unknown benchmark_mode '{benchmark_mode}', use 'fixed_workers' or 'stream_density'")
    # Defaults of the stream density benchmarks of the Order Accuracy applications (main branch)
    target_latency_ms = float(configs.get("target_latency_ms", 10000))
    latency_metric = configs.get("latency_metric", "avg")
    density_increment = max(1, int(configs.get("density_increment", 1)))
    min_requests = max(1, int(configs.get("min_requests", 3)))
    # Highest number of workers to test. 50 is the limit of the source applications (50 steps)
    max_workers = max(1, int(configs.get("max_workers", 50)))
    # "adaptive" finds the maximum with the latency-guided search of test_dlstreamer in a few steps.
    # "linear" tests 1, 2, 3, ... workers one step at a time like the source applications.
    density_search = configs.get("density_search", "adaptive")
    if density_search not in ("adaptive", "linear"):
        pytest.fail(f"Unknown density_search '{density_search}', use 'adaptive' or 'linear'")
    # Restart the model server before every density iteration, so each iteration measures a cold
    # model. The Take-Away application does this: its stream density benchmark restarts the whole
    # stack, including the model server, for every step.
    restart_server_per_iteration = bool(configs.get("restart_server_per_iteration", False))
    # Optional small request sent once before the first density iteration, like the warm-up the
    # application runs at startup: {prompt, image_size, max_new_tokens}
    warmup = configs.get("warmup")
    # Requests each worker sends per density iteration. 1 sends all workers' requests at once.
    requests_per_worker = max(1, int(configs.get("requests_per_worker", 1)))
    benchmark_workers = max(1, int(configs.get("benchmark_workers", 1)))
    benchmark_duration = max(10, int(configs.get("benchmark_duration", 200)))
    benchmark_init_duration = max(0, int(configs.get("benchmark_init_duration", 10)))
    images_per_request = max(1, int(configs.get("images_per_request", 3)))
    max_new_tokens = int(configs.get("max_new_tokens", 100))
    temperature = float(configs.get("temperature", 0.2))
    # Optional request fields are sent only when set: enable_thinking (chat_template_kwargs)
    # and response_format (structured output, for example a JSON schema)
    enable_thinking = configs.get("enable_thinking")
    if enable_thinking is not None:
        enable_thinking = bool(enable_thinking)
    response_format = configs.get("response_format")
    image_preprocessing = configs.get("image_preprocessing", {})
    image_max_size = int(configs.get("image_max_size", 448))
    jpeg_quality = int(configs.get("jpeg_quality", 82))
    request_timeout = int(configs.get("request_timeout", 360))
    server_timeout = int(configs.get("server_timeout", 900))
    download_timeout = int(configs.get("download_timeout", 18000))
    cache_size = int(configs.get("cache_size", 4))
    max_num_seqs = int(configs.get("max_num_seqs", 4))
    max_num_batched_tokens = int(configs.get("max_num_batched_tokens", 8192))
    enable_prefix_caching = bool(configs.get("enable_prefix_caching", True))
    # Number of REST worker threads of OVMS, used by every server start of the test (also the restart of
    # each density step). Unset leaves it to OVMS, which starts one worker per CPU core, like the application
    # docker-compose.yaml. Each worker handles one request until it responds, so a small value limits the
    # concurrent requests.
    rest_workers = int(configs.get("rest_workers") or 0) or None
    if rest_workers is not None and rest_workers < 0:
        pytest.fail(f"rest_workers must be a positive number or unset, got {rest_workers}")
    # Failsafe for systems with many cores: each worker needs about 3 file descriptors, so the default of
    # one worker per core can exceed the open files limit (1024) and fail with "Too many open files
    # (errno: 24)". When the number of workers would exceed max_rest_workers, max_rest_workers is used.
    # Unset means no limit.
    max_rest_workers = int(configs.get("max_rest_workers") or 0) or None
    if max_rest_workers is not None and max_rest_workers < 0:
        pytest.fail(f"max_rest_workers must be a positive number or unset, got {max_rest_workers}")
    rest_workers, rest_workers_info = resolve_rest_workers(rest_workers, max_rest_workers)
    logger.info(f"OVMS REST workers: {rest_workers_info}")
    plugin_config = configs.get("plugin_config", {})
    # Set model_precision to export and quantize a Hugging Face model with the shared OVMS
    # export flow. Leave it unset when model_id is already a pre-quantized OpenVINO model.
    model_precision = configs.get("model_precision")
    quantization_params = configs.get("quantization_params")
    export_timeout = int(configs.get("export_timeout", 3600))
    ovms_image = configs.get("ovms_container_image", "openvino/model_server:2026.3.1-gpu")

    # Placeholders in the prompt, such as ${INVENTORY}, are filled from prompt_params.
    # ${NUM_FRAMES} is resolved per request by the client.
    resolved_params = {k: "\n".join(v) if isinstance(v, list) else str(v) for k, v in prompt_params.items()}
    resolved_prompt = string.Template(prompt).safe_substitute(resolved_params)

    test_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(test_dir, "src", "vlm_prompt")
    core_data_dir = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "app_data"))
    # A dedicated directory keeps the OVMS config limited to this model
    data_dir = os.path.join(core_data_dir, "data", "suites", "ai", "gen", "vlm_prompt")
    models_dir = os.path.join(data_dir, "models")
    # One directory per media set, so entries with different media do not mix and cached files are reused
    media_key = hashlib.sha256("\n".join(m["url"] for m in media).encode()).hexdigest()[:12]
    media_dir = os.path.join(data_dir, "media", media_key)
    results_dir = os.path.join(data_dir, "results")
    client_script = os.path.join(src_dir, "vlm_client.py")
    client_requirements = os.path.join(src_dir, "requirements.txt")
    container_prefix = f"{test_name}_"

    logger.info(f"Starting VLM Prompt Test: {test_display_name}")
    logger.info(
        f"Config: model={model_id}, devices={device_list}, workers={benchmark_workers}, "
        f"duration={benchmark_duration}s, init={benchmark_init_duration}s"
    )

    # ================================================================
    # STEP 2: Validate System Requirements
    # ================================================================
    validate_system_requirements_from_configs(configs)
    docker_client = DockerClient()

    def cleanup():
        cleanup_stale_containers(docker_client, container_prefix)

    try:
        cleanup()

        # ================================================================
        # STEP 3: Prepare Assets/Dependencies
        # ================================================================
        def prepare_assets(target_device):
            result = Result(name=f"{test_id} - Asset Preparation ({target_device})", metadata={"status": "in_progress"})
            try:
                docker_client.pull_image(ovms_image)
                setup_client_venv(core_data_dir, client_requirements)
                servable = prepare_model(
                    model_id=model_id,
                    models_dir=models_dir,
                    device_id=target_device,
                    cache_size=cache_size,
                    max_num_seqs=max_num_seqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                    enable_prefix_caching=enable_prefix_caching,
                    plugin_config=plugin_config,
                    download_timeout=download_timeout,
                    model_precision=model_precision,
                    quantization_params=quantization_params,
                    export_timeout=export_timeout,
                    data_dir=os.path.dirname(data_dir),
                )
                prepare_media(media, media_dir)
                result.metadata["status"] = "completed"
                result.metadata["servable"] = servable
                return result
            except Exception as prep_err:
                result.metadata["status"] = "failed"
                result.metadata["error"] = str(prep_err)
                logger.error(f"Asset preparation failed: {prep_err}")
                raise

        # ================================================================
        # STEP 4: Execute Test Logic (with caching)
        # ================================================================
        def execute_logic(target_device, servable_name, client_manager, client_venv):
            run_results = Result(name=f"{test_id} - {test_display_name} ({target_device})")
            try:

                def start_server():
                    # Replaces a running server of the same name, so it also restarts it
                    started = run_ovms_server_container(
                        docker_client=docker_client,
                        device_id=target_device,
                        docker_image_tag=ovms_image,
                        docker_container_prefix=container_prefix,
                        data_dir=data_dir,
                        container_mnt_dir="/mnt",
                        server_timeout=server_timeout,
                        model_id=servable_name,
                        port=8000,
                        rest_workers=rest_workers,
                    )
                    logger.info(f"OVMS ready in {started['server_startup_duration_seconds']}s")

                start_server()

                client_config = {
                    "endpoint": "http://127.0.0.1:8000/v3/chat/completions",
                    "model": servable_name,
                    "system_prompt": system_prompt,
                    "prompt": resolved_prompt,
                    "images_per_request": images_per_request,
                    "image_max_size": image_max_size,
                    "jpeg_quality": jpeg_quality,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "enable_thinking": enable_thinking,
                    "response_format": response_format,
                    "image_preprocessing": image_preprocessing,
                    "request_timeout": request_timeout,
                    "workers": benchmark_workers,
                    "duration": benchmark_duration,
                    "init_duration": benchmark_init_duration,
                }
                density = None

                def run_once(workers, per_worker=None, warmup_config=None, ttft=0):
                    config = dict(client_config)
                    config["workers"] = workers
                    if ttft:
                        config["ttft_requests"] = ttft
                    if warmup_config:
                        config["warmup"] = warmup_config
                        config["warmup_only"] = True
                    if per_worker:
                        # Density iterations end when every worker has sent its requests
                        config["requests_per_worker"] = per_worker
                        config["duration"] = request_timeout
                        config["init_duration"] = 0
                    return run_client(
                        manager=client_manager,
                        venv_name=client_venv,
                        client_script=client_script,
                        results_dir=results_dir,
                        media_dir=media_dir,
                        client_config=config,
                        timeout=timeout,
                    )

                # Time to first token, measured before anything else so the first request is a cold start
                if ttft_requests:
                    try:
                        ttft_records = run_once(1, ttft=ttft_requests)
                        ttft = aggregate_ttft(ttft_records)
                        allure.attach(
                            "\n".join(json.dumps(r) for r in ttft_records),
                            name=f"TTFT Probe Records {target_device}",
                            attachment_type=allure.attachment_type.TEXT,
                        )
                        if ttft["cold"] is not None:
                            run_results.metrics[
                                get_metric_name_for_device(target_device, prefix="time_to_first_token_cold_start")
                            ] = Metrics(value=round(ttft["cold"], 4), unit="sec")
                        if ttft["warm"] is not None:
                            run_results.metrics[
                                get_metric_name_for_device(target_device, prefix="time_to_first_token_warm_start")
                            ] = Metrics(value=round(ttft["warm"], 4), unit="sec")
                            run_results.metadata["ttft_warm_requests"] = ttft["warm_requests"]
                        if ttft["failed"]:
                            logger.warning(f"{ttft['failed']} TTFT requests failed, for example: {ttft['errors']}")
                    except (RuntimeError, TimeoutError) as ttft_err:
                        logger.warning(f"TTFT measurement failed on {target_device}: {ttft_err}")

                if benchmark_mode == "stream_density":
                    if warmup:
                        run_once(1, warmup_config=warmup)
                    iteration_count = 0

                    def density_iteration(workers):
                        nonlocal iteration_count
                        iteration_count += 1
                        if restart_server_per_iteration and iteration_count > 1:
                            start_server()
                        return run_once(workers, requests_per_worker)

                    density = run_stream_density(
                        run_iteration=density_iteration,
                        target_latency_ms=target_latency_ms,
                        latency_metric=latency_metric,
                        density_increment=density_increment,
                        min_requests=min_requests,
                        max_workers=max_workers,
                        search=density_search,
                    )
                    # A failed density test (no step within the target) still reports the standard metrics,
                    # from the smallest step that was run
                    records = density["best_records"] or density["reference_records"]
                else:
                    records = run_once(benchmark_workers)
                allure.attach(
                    "\n".join(json.dumps(r) for r in records),
                    name=f"VLM Request Records {target_device}",
                    attachment_type=allure.attachment_type.TEXT,
                )
                stats = aggregate_records(records)
                if stats.get("failed"):
                    logger.warning(f"{stats['failed']} requests failed, for example: {stats['errors']}")

                if stats["calls"]:
                    run_results.metrics[get_metric_name_for_device(target_device, prefix="request_latency_avg")] = (
                        Metrics(value=round(stats["avg_latency"], 4), unit="sec")
                    )
                    run_results.metrics[get_metric_name_for_device(target_device, prefix="output_token_throughput")] = (
                        Metrics(
                            value=round(stats["output_token_throughput"], 4),
                            unit="tokens/sec",
                            is_key_metric=density is None,
                        )
                    )
                    run_results.metadata["request_count"] = stats["calls"]
                    # Not a standard metric, kept to compare with the Order Accuracy applications
                    run_results.metadata["throughput_mean_sec"] = round(stats["avg_throughput_per_request"], 4)
                    run_results.metadata["measurement_window_sec"] = round(stats["window_sec"], 2)
                    run_results.metadata["input_sequence_length_avg"] = round(stats["avg_prompt_tokens"], 1)
                    run_results.metadata["output_sequence_length_avg"] = round(stats["avg_completion_tokens"], 1)
                    allure.attach(
                        stats["sample_output"],
                        name=f"Sample VLM Output {target_device}",
                        attachment_type=allure.attachment_type.TEXT,
                    )

                if density is not None:
                    run_results.metrics[get_metric_name_for_device(target_device, prefix="streams_max")] = Metrics(
                        value=density["streams_max"], unit="streams", is_key_metric=True
                    )
                    run_results.metadata["target_latency_ms"] = target_latency_ms
                    run_results.metadata["latency_metric"] = latency_metric
                    run_results.metadata["density_search"] = density_search
                    run_results.metadata["max_workers"] = max_workers
                    run_results.metadata["worker_limit_reached"] = density["limit_reached"]
                    run_results.metadata["density_iterations"] = density["iterations"]
                run_results.metadata["benchmark_workers"] = benchmark_workers
                run_results.metadata["benchmark_duration"] = benchmark_duration
                run_results.metadata["target_device"] = target_device
                run_results.metadata["failed_requests"] = stats.get("failed", 0)

                has_metrics = bool(run_results.metrics)
                run_results.metadata["status"] = "passed" if has_metrics else "failed"
                if not has_metrics:
                    run_results.metadata["error"] = (
                        f"No VLM request completed on {target_device}: {len(records)} request(s) sent, "
                        f"{stats.get('failed', 0)} failed. Errors: {stats.get('errors') or 'none recorded'}. "
                        f"Check the model server container log and the request records."
                    )
                elif density is not None and density["streams_max"] == 0:
                    run_results.metadata["status"] = "failed"
                    reference_workers = density["reference_workers"]
                    measured_ms = latency_metric_ms(
                        [r["latency_sec"] for r in records if r.get("ok") and not r.get("warmup")], latency_metric
                    )
                    average = "" if latency_metric == "avg" else f", average {stats['avg_latency']:.2f}s"
                    run_results.metadata["error"] = (
                        f"The {latency_metric} latency target of {target_latency_ms / 1000:g}s is not met with "
                        f"{reference_workers} worker(s) on {target_device}: {latency_metric} call duration "
                        f"{measured_ms / 1000:.2f}s{average}, {stats['output_token_throughput']:.2f} tokens/sec, "
                        f"{stats['calls']} request(s) measured, {stats['failed']} failed."
                    )
                run_results.update_timestamps()
                return run_results
            except Exception as exec_err:
                run_results.metadata["status"] = "failed"
                run_results.metadata["error"] = str(exec_err)
                logger.error(f"Benchmark execution error: {exec_err}")
                # Keep what the load generator wrote before the failure
                partial_path = os.path.join(results_dir, "vlm_requests.jsonl")
                if os.path.exists(partial_path):
                    try:
                        partial = read_records(partial_path)
                        allure.attach(
                            "\n".join(json.dumps(r) for r in partial),
                            name=f"VLM Request Records {target_device} (partial)",
                            attachment_type=allure.attachment_type.TEXT,
                        )
                    except (OSError, ValueError) as partial_err:
                        logger.warning(f"Could not read the partial request records: {partial_err}")
                raise

        # Run the test on every resolved device one after another. The model server serves the VLM from a
        # single device, so it is stopped between devices. A failure on one device does not prevent the
        # remaining devices from being tested. Metrics are named per device, as in test_ta_vlm.
        results = Result(name=f"{test_id} - {test_display_name}")
        results.parameters["Devices"] = ", ".join(devices)
        results.parameters["Device List"] = ", ".join(device_list)
        results.parameters["Model ID"] = model_id
        results.parameters["Model Precision"] = model_precision or "pre-quantized"
        results.parameters["OVMS REST Workers"] = rest_workers_info
        if benchmark_mode == "stream_density":
            results.parameters["Benchmark Mode"] = "stream_density"
            results.parameters["Target Latency"] = f"{target_latency_ms / 1000:g}s ({latency_metric})"
        else:
            results.parameters["Benchmark Workers"] = str(benchmark_workers)
            results.parameters["Benchmark Duration"] = f"{benchmark_duration}s"

        failed_devices = {}
        limit_reached_devices = []
        for device_id in device_list:
            logger.info(f"Running VLM prompt test for device: {device_id}")
            try:
                prepared = prepare_test(
                    test_name=test_name,
                    prepare_func=lambda device_id=device_id: prepare_assets(device_id),
                    configs=configs,
                    name=f"Assets {device_id}",
                )
                # Idempotent: reuses the venv prepared above, also when the preparation result is cached
                client_manager, client_venv = setup_client_venv(core_data_dir, client_requirements)

                servable_name = prepared.metadata["servable"]

                def run_device(
                    device_id=device_id,
                    servable_name=servable_name,
                    client_manager=client_manager,
                    client_venv=client_venv,
                ):
                    return execute_logic(device_id, servable_name, client_manager, client_venv)

                device_result = execute_test_with_cache(
                    cached_result=cached_result,
                    cache_result=cache_result,
                    run_test_func=run_device,
                    test_name=test_name,
                    configs=configs,
                    cache_configs={
                        **{k: v for k, v in configs.items() if k not in TestResultCache.excluded_fields},
                        "device_id": device_id,
                    },
                    name=f"Analysis {device_id}",
                )
            except Exception as device_err:
                logger.exception(f"VLM prompt test failed for device {device_id}")
                failed_devices[device_id] = str(device_err)
                results.metadata[f"Device {device_id} status"] = "failed"
                results.metadata[f"Device {device_id} error"] = str(device_err)
                continue
            finally:
                cleanup()

            results.metrics.update(device_result.metrics)
            for meta_key, meta_value in device_result.metadata.items():
                results.metadata[f"Device {device_id} {meta_key}"] = meta_value
            if device_result.metadata.get("worker_limit_reached"):
                limit_reached_devices.append(device_id)

            if device_result.metadata.get("status") != "passed":
                failed_devices[device_id] = device_result.metadata.get("error", "Test did not complete successfully")

        if benchmark_mode == "stream_density":
            results.parameters["Worker Limit"] = (
                f"{max_workers} (reached on {', '.join(limit_reached_devices)}, the real maximum is higher)"
                if limit_reached_devices
                else str(max_workers)
            )
        results.metadata["status"] = "failed" if failed_devices else "passed"
        if failed_devices:
            results.metadata["error"] = "; ".join(f"{dev}: {err}" for dev, err in failed_devices.items())
        results.update_timestamps()

        if failed_devices:
            test_failed = True
            failure_message = f"VLM prompt benchmark failed: {results.metadata['error']}"

    except KeyboardInterrupt:
        failure_message = "Interrupt detected during VLM prompt test execution"
        test_interrupted = True
        logger.error(failure_message)
        if results is not None:
            results.metadata["status"] = False
            results.metadata["error"] = failure_message

    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during VLM prompt test execution: {e!s}"
        logger.exception(failure_message)
        if results is None:
            results = Result(
                name=test_name,
                metadata={"status": False, "error": str(e)},
                metrics={},
                parameters={"Devices": ", ".join(devices), "Device List": ", ".join(device_list)},
            )
        else:
            results.metadata["status"] = False
            results.metadata["error"] = str(e)

    finally:
        try:
            cleanup()
        except Exception as cleanup_error:  # noqa: BLE001
            logger.error(f"Cleanup failed: {cleanup_error}")

        try:
            if results is not None:
                summarize_test_results(results=results, test_name=test_name, configs=configs)
            else:
                logger.error("No results to summarize")
        except Exception:
            logger.exception("Test result summarization failed")

        if test_interrupted or test_failed:
            pytest.fail(failure_message)

    logger.info(f"VLM prompt test '{test_name}' completed successfully")
