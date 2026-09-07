# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import json
import logging
import os
import tarfile

import allure
import pytest

from sysagent.utils.core import Metrics, Result, get_metric_name_for_device
from sysagent.utils.infrastructure import DockerClient, download_file
from sysagent.utils.system.ov_helper import get_available_devices_by_category

from esq.suites.ai.audio.src.whisper.container import (
    build_whisper_image,
    run_openvino_export_container,
    run_openvino_whisper_container,
    run_pytorch_prepare_container,
    run_pytorch_whisper_container,
)

logger = logging.getLogger(__name__)

_DEFAULT_AUDIO_URL = "https://www.openslr.org/resources/12/dev-other.tar.gz"
_DEFAULT_AUDIO_FLAC_MEMBER = "LibriSpeech/dev-other/3915/57461/3915-57461-0003.flac"
_DEFAULT_REFERENCE = "here there is an arm of the sea which is crossed in ferry boats that start as soon as some twenty or thirty passengers are gathered together and in one of these boats the two travellers embarked"

# Maps model_size shorthand to the canonical HuggingFace model ID
_WHISPER_HF_IDS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
    "turbo": "openai/whisper-large-v3-turbo",
}


# Metric units used when assembling a Result from backend output
_METRIC_UNITS = {
    "latency_ms": "ms",
    "rtf": "ratio",
    "wer": "%",
    "cer": "%",
}


@allure.description(
    "Benchmarks Whisper ASR model inference performance by running transcription "
    "of a reference audio sample 5 times and measuring average latency, "
    "real-time factor (RTF), word error rate (WER), and character error rate (CER) "
    "across model sizes, backends (OpenVINO / PyTorch), and device targets (CPU, iGPU, dGPU, NPU)."
)
def test_whisper(
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
    """Benchmarks Whisper ASR model inference performance across model sizes, backends, and devices.

    Measures:
    - Latency: end-to-end transcription time in milliseconds, averaged across num_runs
      timed inference passes (default 5 runs).
    - RTF (Real-Time Factor): inference_time / audio_duration; lower is better.
    - WER (Word Error Rate): word-level transcription error percentage; lower is better.
    - CER (Character Error Rate): character-level transcription error percentage; lower is better.

    Supported model sizes: tiny, base, small, medium, large, turbo.
    Supported backends: openvino, pytorch.
    Supported devices: cpu, igpu, dgpu, npu.

    Profile parameters:
        model_size (str): Whisper model size.
        backend (str): Inference backend - "openvino" or "pytorch".
        devices (list): Device category list, e.g. [cpu], [igpu].
        model_id (str, optional): HuggingFace model ID to download and export.
            Defaults to the canonical ID for the given model_size (e.g. openai/whisper-base).
        model_dir (str, optional): Path to the exported OpenVINO model directory.
            Defaults to <data_dir>/models/whisper-<model_size>.
            If the directory does not exist, the model is downloaded and exported automatically.
        wav_file_path (str, optional): Path to a 16 kHz WAV audio file for benchmarking.
            Defaults to <data_dir>/audio/how_are_you_doing_today.wav (auto-downloaded).
        reference_transcript (str, optional): Ground-truth text used for WER/CER.
            Defaults to "how are you doing today".
        download_timeout (float, optional): Seconds to allow for HuggingFace model download. Default 3600.
        export_timeout (float, optional): Seconds to allow for optimum-cli export. Default 1800.
        warmup_runs (int, optional): Number of un-timed warm-up passes before measurement. Default 1.
        num_runs (int, optional): Number of timed inference passes; latency_ms is the average
            across these runs. Default 5.
    """

    # ================================================================
    # STEP 1: Extract Parameters
    # ================================================================
    test_name = request.node.name.split("[")[0]
    test_display_name = configs.get("display_name", test_name)
    kpi_validation_mode = configs.get("kpi_validation_mode", "all")
    timeout = configs.get("timeout", 300)  # noqa: F841
    devices = configs.get("devices", ["cpu"])
    model_size = configs.get("model_size", "tiny")
    backend = configs.get("backend", "openvino")
    reference_transcript = configs.get("reference_transcript", _DEFAULT_REFERENCE)
    download_timeout = configs.get("download_timeout", 3600)
    export_timeout = configs.get("export_timeout", 1800)
    inference_timeout = configs.get("inference_timeout", 900)
    warmup_runs = max(0, int(configs.get("warmup_runs", 1)))
    num_runs = max(1, int(configs.get("num_runs", 5)))
    docker_client_timeout = configs.get("docker_client_timeout", 180)

    # Docker image tags
    docker_image_tag = (
        f"{configs.get('container_image_name', f'whisper-{backend}')}:"
        f"{configs.get('container_tag', 'latest')}"
    )

    # Data directories
    test_dir = os.path.dirname(os.path.abspath(__file__))
    containers_dir = os.path.join(test_dir, "src", "containers")
    core_data_dir = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "app_data"))
    data_dir = os.path.join(core_data_dir, "data", "suites", "ai", "audio")

    # HuggingFace model ID - from profile param or derived from model_size
    model_hf_id = configs.get("model_id") or _WHISPER_HF_IDS.get(model_size, f"openai/whisper-{model_size}")

    # Model directory - from profile param or derived from model_size
    model_dir = configs.get("model_dir") or os.path.join(data_dir, "models", f"whisper-{model_size}")

    # Audio file - from profile param or default sample
    wav_file_path = configs.get("wav_file_path") or os.path.join(
        data_dir, "audio", "3915-57461-0003.wav"
    )

    # Map device category to backend-specific device string
    ov_device = None
    torch_device = None
    device_dict = {}
    device_list = []
    torch_ov_id_map = {}  # maps torch device string → OV device id for metric naming
    if backend == "openvino":
        device_dict = get_available_devices_by_category(device_categories=devices)
        if not device_dict:
            pytest.skip(f"No OpenVINO device available for categories: {devices}")
        device_list = list(device_dict.keys())
        ov_device = device_list[0]
        logger.info(f"OpenVINO devices resolved: {device_list} (from categories: {devices})")
    elif backend == "pytorch":
        from esq.suites.ai.audio.src.containers.utils.pytorch_helper import resolve_torch_devices
        torch_device_pairs = resolve_torch_devices(devices)
        device_list = [td for td, _ in torch_device_pairs]
        torch_ov_id_map = {td: ov_id for td, ov_id in torch_device_pairs}
        torch_device = device_list[0]
        logger.info(f"PyTorch devices resolved: {device_list} (from categories: {devices})")

    logger.info(f"Starting Whisper benchmark: {test_display_name}")
    logger.info(f"Model: whisper-{model_size}, Backend: {backend}, Device: {ov_device or devices}")

    # ================================================================
    # STEP 2: Validate System Requirements
    # ================================================================
    validate_system_requirements_from_configs(configs)

    # Verify Docker client connection
    docker_client = DockerClient(timeout=docker_client_timeout)

    # ================================================================
    # Initialise state variables (referenced in finally block)
    # ================================================================
    validation_results = {}
    test_failed = False
    test_interrupted = False
    failure_message = ""
    results = None

    try:
        # ============================================================
        # STEP 3: Prepare Assets / Dependencies
        # ============================================================
        def prepare_assets():
            # --- Build Docker image ---
            build_whisper_image(
                docker_client=docker_client,
                backend=backend,
                containers_dir=containers_dir,
                image_tag=docker_image_tag,
            )

            # --- Sample audio ---
            audio_dir = os.path.dirname(wav_file_path)
            os.makedirs(audio_dir, exist_ok=True)
            if not os.path.exists(wav_file_path):
                import soundfile as sf

                tar_path = os.path.join(audio_dir, "dev-other.tar.gz")
                logger.info(f"Downloading LibriSpeech dev-other archive to {tar_path}")
                download_file(url=_DEFAULT_AUDIO_URL, target_path=tar_path)

                logger.info(f"Extracting {_DEFAULT_AUDIO_FLAC_MEMBER} from archive")
                with tarfile.open(tar_path, "r:gz") as tar:
                    flac_file = tar.extractfile(_DEFAULT_AUDIO_FLAC_MEMBER)
                    if flac_file is None:
                        raise RuntimeError(f"Member not found in archive: {_DEFAULT_AUDIO_FLAC_MEMBER}")
                    flac_data = flac_file.read()

                data, samplerate = sf.read(io.BytesIO(flac_data))
                sf.write(wav_file_path, data, samplerate)
                logger.info(f"Converted FLAC to WAV: {wav_file_path}")
                os.remove(tar_path)

            # --- Model (backend-specific, run inside container) ---
            if backend == "openvino":
                run_openvino_export_container(
                    docker_client=docker_client,
                    image_tag=docker_image_tag,
                    container_name=f"{test_name}-export",
                    data_dir=data_dir,
                    model_hf_id=model_hf_id,
                    model_dir=model_dir,
                    export_timeout=int(export_timeout),
                )
            elif backend == "pytorch":
                pytorch_model_dir = os.path.join(data_dir, "models")
                run_pytorch_prepare_container(
                    docker_client=docker_client,
                    image_tag=docker_image_tag,
                    container_name=f"{test_name}-prepare",
                    data_dir=data_dir,
                    model_size=model_size,
                    pytorch_model_dir=pytorch_model_dir,
                    download_timeout=int(download_timeout),
                )

            return Result(
                name=f"{test_name} - Asset Preparation",
                metadata={"status": "completed"},
            )

        prepare_test(
            test_name=test_name,
            prepare_func=prepare_assets,
            configs=configs,
            name="Assets",
        )

        # ============================================================
        # STEP 4: Build initial results template and execute
        # ============================================================
        # Build per-device metrics; pytorch maps xpu:N → OV device id for consistent naming
        device_id_for_metric = torch_ov_id_map if backend == "pytorch" else {d: d for d in device_list}
        all_metrics = {}
        for dev in device_list:
            ov_id = device_id_for_metric.get(dev, dev)
            for metric_name, unit in _METRIC_UNITS.items():
                all_metrics[get_metric_name_for_device(ov_id, prefix=metric_name)] = unit
        metrics = {key: Metrics(unit=unit, value=-1.0) for key, unit in all_metrics.items()}

        device_id_str = ", ".join(device_list) if device_list else (ov_device or str(devices))
        results = Result.from_test_config(
            configs=configs,
            parameters={
                "Device": devices,
                "Device ID": device_id_str,
                "Model Size": model_size,
                "Backend": backend,
                "Warm-up Runs": warmup_runs,
                "Timed Runs": num_runs,
                "Display Name": test_display_name,
            },
            metrics=metrics,
            metadata={"status": True},
        )
        logger.debug(f"Initial Results template: {json.dumps(results.to_dict(), indent=2)}")

        # --------------------------------------------------------
        # Execute inference (with caching support)
        # --------------------------------------------------------
        if backend == "openvino":
            # Iterate over all available OpenVINO devices (handles multi-dGPU)
            for current_ov_device in device_list:
                def execute_for_device(ov_dev=current_ov_device):
                    dev_result = Result(name=f"{test_name} - {test_display_name} - {ov_dev}")
                    output = run_openvino_whisper_container(
                        docker_client=docker_client,
                        image_tag=docker_image_tag,
                        container_name=f"{test_name}-infer-{ov_dev.replace('.', '_').lower()}",
                        data_dir=data_dir,
                        model_dir=model_dir,
                        ov_device=ov_dev,
                        wav_file_path=wav_file_path,
                        warmup_runs=warmup_runs,
                        num_runs=num_runs,
                        reference_transcript=reference_transcript,
                        inference_timeout=inference_timeout,
                    )
                    for metric_name, value in output["metrics"].items():
                        dev_result.metrics[metric_name] = Metrics(
                            value=value, unit=_METRIC_UNITS[metric_name]
                        )
                    dev_result.parameters.update(output["parameters"])
                    dev_result.parameters["Warm-up Runs"] = warmup_runs
                    dev_result.parameters["Timed Runs"] = num_runs
                    dev_result.metadata.update(output["metadata"])
                    return dev_result

                cache_configs = {"device_id": current_ov_device, "model_size": model_size}
                device_result = execute_test_with_cache(
                    cached_result=cached_result,
                    cache_result=cache_result,
                    run_test_func=execute_for_device,
                    test_name=test_name,
                    configs=configs,
                    cache_configs=cache_configs,
                )

                # Merge into final results with per-device metric names
                for metric_name in _METRIC_UNITS:
                    if metric_name in device_result.metrics:
                        per_dev_metric = get_metric_name_for_device(current_ov_device, prefix=metric_name)
                        if per_dev_metric in results.metrics:
                            results.metrics[per_dev_metric].value = device_result.metrics[metric_name].value
                results.parameters.update(device_result.parameters)
                results.metadata.update(device_result.metadata)
                if not device_result.metadata.get("status", False):
                    results.metadata["status"] = False

            logger.info(f"Completed inference for {len(device_list)} OpenVINO device(s): {device_list}")

        else:  # pytorch
            for current_torch_device in device_list:
                ov_dev_id = torch_ov_id_map.get(current_torch_device, current_torch_device)

                def execute_for_pt(td=current_torch_device):
                    dev_result = Result(name=f"{test_name} - {test_display_name} - {td}")
                    pytorch_model_dir = os.path.join(data_dir, "models")
                    import re
                    safe_td = re.sub(r"[^a-zA-Z0-9_.-]", "_", td)
                    output = run_pytorch_whisper_container(
                        docker_client=docker_client,
                        image_tag=docker_image_tag,
                        container_name=f"{test_name}-infer-{safe_td}",
                        data_dir=data_dir,
                        model_size=model_size,
                        torch_device=td,
                        pytorch_model_dir=pytorch_model_dir,
                        wav_file_path=wav_file_path,
                        warmup_runs=warmup_runs,
                        num_runs=num_runs,
                        reference_transcript=reference_transcript,
                        inference_timeout=inference_timeout,
                    )
                    if output.get("metadata", {}).get("status") == "skip":
                        # Propagate skip signal via metadata; pytest.skip() called at test level.
                        dev_result.metadata["status"] = "skip"
                        dev_result.metadata["skip_reason"] = output["metadata"].get("skip_reason", "")
                        return dev_result
                    for metric_name, value in output["metrics"].items():
                        dev_result.metrics[metric_name] = Metrics(
                            value=value, unit=_METRIC_UNITS[metric_name]
                        )
                    dev_result.parameters.update(output["parameters"])
                    dev_result.parameters["Warm-up Runs"] = warmup_runs
                    dev_result.parameters["Timed Runs"] = num_runs
                    dev_result.metadata.update(output["metadata"])
                    return dev_result

                cache_configs = {"device_id": current_torch_device, "model_size": model_size}
                device_result = execute_test_with_cache(
                    cached_result=cached_result,
                    cache_result=cache_result,
                    run_test_func=execute_for_pt,
                    test_name=test_name,
                    configs=configs,
                    cache_configs=cache_configs,
                )

                for metric_name in _METRIC_UNITS:
                    if metric_name in device_result.metrics:
                        per_dev_metric = get_metric_name_for_device(ov_dev_id, prefix=metric_name)
                        if per_dev_metric in results.metrics:
                            results.metrics[per_dev_metric].value = device_result.metrics[metric_name].value
                results.parameters.update(device_result.parameters)
                results.metadata.update(device_result.metadata)
                if device_result.metadata.get("status") == "skip":
                    pytest.skip(device_result.metadata.get("skip_reason", "Device not available in container"))
                if not device_result.metadata.get("status", False):
                    results.metadata["status"] = False

            logger.info(f"Completed inference for {len(device_list)} PyTorch device(s): {device_list}")

        logger.debug(f"Whisper benchmark results: {json.dumps(results.to_dict(), indent=2)}")

        if not results.metadata.get("status", False):
            test_failed = True
            failure_message = results.metadata.get("error", f"{test_display_name} failed")

    except KeyboardInterrupt:
        test_interrupted = True
        failure_message = "Interrupt detected during Whisper benchmark execution"
        logger.error(failure_message)

    except Exception as error:
        test_failed = True
        failure_message = f"Unexpected error during test execution: {error}"
        logger.error(failure_message, exc_info=True)

        if results is None:
            metrics = {
                key: Metrics(unit=unit, value=-1.0) for key, unit in _METRIC_UNITS.items()
            }
            results = Result.from_test_config(
                configs=configs,
                parameters={
                    "Device": devices,
                    "Model Size": model_size,
                    "Error": str(error),
                },
                metrics=metrics,
                metadata={"status": False, "error": str(error)},
            )
        else:
            results.metadata["status"] = False
            results.metadata["error"] = str(error)

    finally:
        # Update timestamps and log duration
        if results:
            results.update_timestamps()
            logger.info(
                f"Test completed - Duration: {results.metadata.get('total_duration_seconds', 0):.2f} s"
            )

        # ============================================================
        # STEP 5: Validate Results Against KPIs (always runs)
        # ============================================================
        try:
            validation_results = validate_test_results(
                results=results,
                configs=configs,
                get_kpi_config=get_kpi_config,
                test_name=test_name,
                mode=kpi_validation_mode,
            )

            results.update_kpi_validation_status(validation_results, kpi_validation_mode)

            # Latency and error-rate metrics are lower-is-better
            results.auto_set_key_metric(
                validation_results,
                kpi_validation_mode,
                metric_direction="lower_is_better",
            )

            current_kpi_refs = configs.get("kpi_refs", [])
            if current_kpi_refs:
                kpi_data = {}
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

        # ============================================================
        # STEP 6: Generate Summary (always runs)
        # ============================================================
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

        # ============================================================
        # STEP 7: Surface the Outcome
        # ============================================================
        is_qualification = configs.get("labels", {}).get("type") == "qualification"

        if test_interrupted:
            if is_qualification:
                pytest.fail(failure_message)
            else:
                raise RuntimeError(failure_message)
        if test_failed:
            pytest.fail(failure_message)

    logger.info(f"Whisper benchmark '{test_name}' completed successfully")
