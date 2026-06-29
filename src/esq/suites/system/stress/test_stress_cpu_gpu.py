# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Host machine CPU + GPU stress test.

This test is intentionally host-only (no VM/QEMU lifecycle). It uses stress-ng
with configurable CPU and memory stressors, and optionally the GPU stressor
when supported on the platform.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import allure
import yaml
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result, run_command

logger = logging.getLogger(__name__)


def _check_command_available(command: str) -> bool:
    """Return True if a command exists in PATH."""
    result = run_command(["which", command], timeout=5)
    return bool(result and result.returncode == 0 and result.stdout.strip())


def _safe_int(value, default: int) -> int:
    """Parse integer safely with fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _detect_intel_gpu_cards() -> int:
    """Count Intel GPU cards under DRM."""
    count = 0
    for vendor_file in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            with open(vendor_file, "r", encoding="utf-8") as file:
                vendor = file.read().strip().lower()
            if vendor == "0x8086":
                count += 1
        except (IOError, OSError):
            continue
    return count


def _get_intel_drm_cards() -> List[str]:
    """Return Intel DRM card device nodes sorted by card index."""
    cards: List[Tuple[int, str]] = []
    for vendor_file in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            with open(vendor_file, "r", encoding="utf-8") as file:
                vendor = file.read().strip().lower()
            if vendor != "0x8086":
                continue
            card_name = vendor_file.parent.parent.name
            if not card_name.startswith("card"):
                continue
            card_index = int(card_name.replace("card", ""))
            devnode = f"/dev/dri/{card_name}"
            if os.path.exists(devnode):
                cards.append((card_index, devnode))
        except (IOError, OSError, ValueError):
            continue

    cards.sort(key=lambda item: item[0])
    return [devnode for _, devnode in cards]


def _resolve_gpu_devnode(gpu_device_index: int) -> str:
    """Resolve configured GPU index to a concrete DRM card node path."""
    if gpu_device_index < 0:
        return ""
    intel_cards = _get_intel_drm_cards()
    if gpu_device_index >= len(intel_cards):
        return ""
    return intel_cards[gpu_device_index]


def _command_to_text(command: Optional[List[str]]) -> str:
    """Convert command tokens to readable text safely."""
    if not command:
        return ""
    return " ".join(command)


def _build_cpu_stress_command(configs: Dict) -> Optional[List[str]]:
    """Build CPU/memory stress-ng command line from profile params."""
    duration = max(_safe_int(configs.get("stress_duration_seconds", 60), 60), 1)
    enable_cpu_stress = bool(configs.get("enable_cpu_stress", True))
    enable_memory_stress = bool(configs.get("enable_memory_stress", True))

    command: List[str] = [
        "stress-ng",
        "--timeout", f"{duration}s",
        "--metrics-brief",
    ]

    if enable_cpu_stress:
        cpu_workers = max(_safe_int(configs.get("stress_cpu_workers", 0), 0), 0)
        cpu_load = max(min(_safe_int(configs.get("stress_cpu_load", 90), 90), 100), 0)
        command.extend([
            "--cpu", str(cpu_workers),
            "--cpu-load", str(cpu_load),
        ])

    if enable_memory_stress:
        vm_workers = max(_safe_int(configs.get("stress_vm_workers", 0), 0), 0)
        vm_bytes = str(configs.get("stress_vm_bytes", "512M"))
        if vm_workers > 0:
            command.extend([
                "--vm", str(vm_workers),
                "--vm-bytes", vm_bytes,
            ])

    if len(command) <= 4:
        return None

    return command


def _metric_unit_from_name(metric_name: str) -> str:
    """Infer metric unit from stress-ng metric naming convention."""
    if metric_name.endswith("_bogo_ops_per_real_time") or metric_name.endswith("_bogo_ops_per_usr_sys_time"):
        return "ops/s"
    if metric_name.endswith("_bogo_ops"):
        return "ops"
    if metric_name.endswith("_secs"):
        return "s"
    return ""


def _resolve_key_metric_name(configs: Dict, gpu_requested: bool) -> str:
    """Resolve key metric name from profile config with sensible defaults."""
    configured_name = str(configs.get("key_metric_name", "")).strip()
    if configured_name:
        return configured_name
    return "gpu_bogo_ops_per_real_time" if gpu_requested else "cpu_bogo_ops_per_real_time"


def _normalize_key_metric(metrics: Dict[str, Metrics], key_metric_name: str, success: bool) -> Dict[str, Metrics]:
    """Mark one deterministic key metric and ensure it exists on failure."""
    normalized: Dict[str, Metrics] = {}
    for metric_name, metric in metrics.items():
        normalized[metric_name] = Metrics(value=metric.value, unit=metric.unit, is_key_metric=False)

    existing = normalized.get(key_metric_name)
    if success and existing is not None:
        normalized[key_metric_name] = Metrics(value=existing.value, unit=existing.unit, is_key_metric=True)
        return normalized

    unit = existing.unit if existing is not None else _metric_unit_from_name(key_metric_name)
    normalized[key_metric_name] = Metrics(value=-1.0, unit=unit, is_key_metric=True)
    return normalized


def _build_gpu_stress_command(configs: Dict, gpu_enabled: bool) -> Tuple[Optional[List[str]], str]:
    """Build GPU stress command.

    Supports:
    - stress-ng: GPU rendering workload (graphics stress)
    - custom: User-supplied command override

    Returns:
        (command, selected_tool)
    """
    if not gpu_enabled:
        return None, "disabled"

    duration = max(_safe_int(configs.get("stress_duration_seconds", 60), 60), 1)
    gpu_tool = str(configs.get("gpu_tool", "stress-ng")).strip().lower()

    if gpu_tool == "stress-ng":
        # stress-ng GPU rendering: --gpu stresses GPU graphics/rendering pipelines.
        # Workers=0 means auto/all available GPU devices.
        gpu_workers = max(_safe_int(configs.get("stress_gpu_workers", 0), 0), 0)
        gpu_ops = max(_safe_int(configs.get("stress_gpu_ops", 0), 0), 0)
        gpu_frag = max(_safe_int(configs.get("stress_gpu_frag", 0), 0), 0)
        gpu_upload = max(_safe_int(configs.get("stress_gpu_upload", 0), 0), 0)
        gpu_tex_size = max(_safe_int(configs.get("stress_gpu_tex_size", 0), 0), 0)
        gpu_xsize = max(_safe_int(configs.get("stress_gpu_xsize", 0), 0), 0)
        gpu_ysize = max(_safe_int(configs.get("stress_gpu_ysize", 0), 0), 0)
        gpu_device_index = _safe_int(configs.get("stress_gpu_device_index", -1), -1)
        gpu_devnode = _resolve_gpu_devnode(gpu_device_index)
        command = [
            "stress-ng",
            "--timeout", f"{duration}s",
            "--metrics-brief",
            "--gpu", str(gpu_workers),
        ]
        if gpu_devnode:
            command.extend(["--gpu-devnode", gpu_devnode])
        elif gpu_device_index >= 0:
            logger.warning(
                f"Configured stress_gpu_device_index={gpu_device_index} could not be resolved; "
                "running stress-ng GPU without explicit devnode"
            )
        if gpu_ops > 0:
            command.extend(["--gpu-ops", str(gpu_ops)])
        if gpu_frag > 0:
            command.extend(["--gpu-frag", str(gpu_frag)])
        if gpu_upload > 0:
            command.extend(["--gpu-upload", str(gpu_upload)])
        if gpu_tex_size > 0:
            command.extend(["--gpu-tex-size", str(gpu_tex_size)])
        if gpu_xsize > 0:
            command.extend(["--gpu-xsize", str(gpu_xsize)])
        if gpu_ysize > 0:
            command.extend(["--gpu-ysize", str(gpu_ysize)])
        return command, "stress-ng"

    # Support direct command override for external GPU tools (furmark/3dmark/occt etc.).
    custom = configs.get("gpu_custom_command")
    if isinstance(custom, list) and custom:
        return [str(token) for token in custom], gpu_tool

    if isinstance(custom, str) and custom.strip():
        return [custom.strip()], gpu_tool

    return None, gpu_tool


def _run_command_worker(command: List[str], timeout: int, sink: Dict, sink_key: str) -> None:
    """Run a command and store normalized result into sink."""
    result = run_command(command, timeout=timeout)
    sink[sink_key] = {
        "returncode": result.returncode if result else -1,
        "success": bool(result and result.returncode == 0),
        "stdout": result.stdout if result else "",
        "stderr": result.stderr if result else "",
        "command": command,
    }


def _parse_stress_ng_yaml_metrics(yaml_path: str) -> Tuple[Dict[str, Metrics], str]:
    """Parse stress-ng native YAML metrics file.

    Returns:
        (metrics, error_message). error_message is empty on success.
    """
    if not yaml_path:
        return {}, "missing yaml metrics path"
    if not os.path.exists(yaml_path):
        return {}, f"yaml metrics file not found: {yaml_path}"

    try:
        with open(yaml_path, "r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
    except (IOError, OSError, yaml.YAMLError) as error:
        return {}, f"failed to read yaml metrics file '{yaml_path}': {error}"

    metrics_section = payload.get("metrics")
    if not isinstance(metrics_section, list) or not metrics_section:
        return {}, f"yaml metrics file has no metrics entries: {yaml_path}"

    field_map = {
        "bogo-ops": "bogo_ops",
        "wall-clock-time": "real_time_secs",
        "user-time": "usr_time_secs",
        "system-time": "sys_time_secs",
        "bogo-ops-per-second-real-time": "bogo_ops_per_real_time",
        "bogo-ops-per-second-usr-sys-time": "bogo_ops_per_usr_sys_time",
    }

    units = {
        "bogo_ops": "ops",
        "real_time_secs": "s",
        "usr_time_secs": "s",
        "sys_time_secs": "s",
        "bogo_ops_per_real_time": "ops/s",
        "bogo_ops_per_usr_sys_time": "ops/s",
    }

    parsed: Dict[str, Metrics] = {}
    for entry in metrics_section:
        if not isinstance(entry, dict):
            continue
        stressor = str(entry.get("stressor", "")).strip().replace("-", "_").lower()
        if not stressor:
            continue

        for yaml_key, metric_key in field_map.items():
            if yaml_key not in entry:
                continue

            raw_value = entry.get(yaml_key)
            try:
                value = int(raw_value) if metric_key == "bogo_ops" else float(raw_value)
            except (TypeError, ValueError):
                continue

            full_metric_name = f"{stressor}_{metric_key}"
            parsed[full_metric_name] = Metrics(value=value, unit=units[metric_key], is_key_metric=False)

    if not parsed:
        return {}, f"yaml metrics file has no parseable numeric metrics: {yaml_path}"

    return parsed, ""


def _run_parallel_stress_and_sample(
    cpu_command: Optional[List[str]],
    gpu_command: Optional[List[str]],
    timeout: int,
) -> Tuple[Dict, Dict]:
    """Run CPU/GPU stress concurrently and return normalized command results."""
    results: Dict[str, Dict] = {}

    cpu_thread = None
    if cpu_command:
        cpu_thread = threading.Thread(
            target=_run_command_worker,
            args=(cpu_command, timeout, results, "cpu"),
            daemon=True,
        )
        cpu_thread.start()

    gpu_thread = None
    if gpu_command:
        gpu_thread = threading.Thread(
            target=_run_command_worker,
            args=(gpu_command, timeout, results, "gpu"),
            daemon=True,
        )
        gpu_thread.start()

    if cpu_thread:
        cpu_thread.join()
    if gpu_thread:
        gpu_thread.join()

    return results.get("cpu", {}), results.get("gpu", {})

def _write_command_logs(cpu_result: Dict, gpu_result: Dict, output_dir: str) -> Dict[str, str]:
    """Persist non-empty command logs only and return created file paths."""

    def _write_if_non_empty(stream_text: str, filename: str) -> str:
        if not stream_text:
            return ""
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(stream_text)
        return file_path

    return {
        "cpu_stderr_log": _write_if_non_empty(cpu_result.get("stderr", ""), "stress_cpu_stderr.log"),
        "gpu_stderr_log": _write_if_non_empty(gpu_result.get("stderr", ""), "stress_gpu_stderr.log"),
    }


def _attach_log_file(file_path: str, attachment_name: str) -> None:
    """Attach a log file to Allure report if it exists."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        allure.attach.file(
            file_path,
            name=attachment_name,
            attachment_type=allure.attachment_type.TEXT,
        )
    except Exception as error:  # pragma: no cover - report backend failure
        logger.warning(f"Failed to attach log file '{file_path}': {error}")


def _attach_native_yaml_file(file_path: str, attachment_name: str) -> None:
    """Attach native stress-ng YAML metrics file if it exists."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        allure.attach.file(
            file_path,
            name=attachment_name,
            attachment_type=allure.attachment_type.TEXT,
        )
    except Exception as error:  # pragma: no cover - report backend failure
        logger.warning(f"Failed to attach native YAML '{file_path}': {error}")


def _run_stress_command(configs: Dict, timeout: int, output_dir: str, gpu_enabled: bool) -> Dict:
    """Execute CPU and GPU stress concurrently."""
    cpu_command = _build_cpu_stress_command(configs)
    gpu_command, gpu_tool = _build_gpu_stress_command(configs, gpu_enabled=gpu_enabled)
    cpu_yaml_metrics = os.path.join(output_dir, "stress_cpu_metrics_native.yaml")
    gpu_yaml_metrics = os.path.join(output_dir, "stress_gpu_metrics_native.yaml")

    if cpu_command:
        cpu_command = list(cpu_command) + ["--yaml", cpu_yaml_metrics]
    if gpu_command:
        gpu_command = list(gpu_command) + ["--yaml", gpu_yaml_metrics]

    cpu_result, gpu_result = _run_parallel_stress_and_sample(
        cpu_command=cpu_command,
        gpu_command=gpu_command,
        timeout=timeout,
    )

    log_paths = {
        "cpu_stderr_log": "",
        "gpu_stderr_log": "",
    }
    try:
        log_paths = _write_command_logs(cpu_result, gpu_result, output_dir)
    except (IOError, OSError) as error:
        logger.warning(f"Failed to write stress logs: {error}")

    cpu_attempted = cpu_command is not None
    cpu_success = bool(cpu_result.get("success", False)) if cpu_attempted else True
    gpu_attempted = gpu_command is not None
    gpu_success = bool(gpu_result.get("success", False)) if gpu_attempted else True

    return {
        "returncode": cpu_result.get("returncode", -1),
        "success": cpu_success and gpu_success,
        "cpu_result": cpu_result,
        "cpu_attempted": cpu_attempted,
        "gpu_result": gpu_result,
        "gpu_attempted": gpu_attempted,
        "gpu_tool": gpu_tool,
        "cpu_command": cpu_result.get("command", cpu_command),
        "gpu_command": gpu_result.get("command", gpu_command or []),
        "cpu_stderr_log": log_paths.get("cpu_stderr_log", ""),
        "gpu_stderr_log": log_paths.get("gpu_stderr_log", ""),
        "cpu_yaml_metrics": cpu_yaml_metrics,
        "gpu_yaml_metrics": gpu_yaml_metrics,
    }


@allure.title("System Stress - Host CPU + GPU")
def test_stress_cpu_gpu(
    request,
    configs,
    cached_result,
    cache_result,
    execute_test_with_cache,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
):
    """Run host-only CPU + optional GPU stress test."""
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", "Host Stress")

    description = configs.get("description")
    if description:
        allure.dynamic.description(description)

    logger.info(f"Starting host stress test: {test_display_name}")

    validate_system_requirements_from_configs(configs)

    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    core_data_resolved = str(Path(core_data_dir_tainted).resolve())
    core_data_dir = "".join(ch for ch in core_data_resolved)

    expected_base = Path(os.getcwd()).resolve()
    if not Path(core_data_dir).resolve().is_relative_to(expected_base):
        core_data_dir = os.path.join(os.getcwd(), "esq_data")

    results_dir = os.path.join(core_data_dir, "data", "system", "stress", "results", test_id)
    results_resolved = str(Path(results_dir).resolve())
    results_dir = "".join(ch for ch in results_resolved)

    os.makedirs(results_dir, mode=0o770, exist_ok=True)
    ensure_dir_permissions(results_dir, uid=os.getuid(), gid=os.getgid(), mode=0o770)

    stress_ng_available = _check_command_available("stress-ng")
    intel_gpu_cards = _detect_intel_gpu_cards()
    gpu_requested = bool(configs.get("enable_gpu_stress", True))
    gpu_enabled = gpu_requested and intel_gpu_cards > 0
    key_metric_name = _resolve_key_metric_name(configs, gpu_requested=gpu_requested)

    duration = max(_safe_int(configs.get("stress_duration_seconds", 60), 60), 1)
    timeout = max(_safe_int(configs.get("timeout", duration + 120), duration + 120), duration + 30)

    if not stress_ng_available:
        base_metrics = _normalize_key_metric({}, key_metric_name=key_metric_name, success=False)
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={
                "status": False,
                "message": "stress-ng is not installed on host",
                "stress_success": False,
            },
            extended_metadata={
                "stress_output_dir": results_dir,
            },
            metrics=base_metrics,
        )
    elif gpu_requested and not gpu_enabled:
        base_metrics = _normalize_key_metric({}, key_metric_name=key_metric_name, success=False)
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={
                "status": False,
                "message": "GPU stress requested but no Intel iGPU detected on host",
                "stress_success": False,
            },
            extended_metadata={
                "stress_output_dir": results_dir,
            },
            metrics=base_metrics,
        )
    else:
        # Wrap stress execution in execute_test_with_cache to enable telemetry
        # sampling during the actual workload (scope=execution)
        def _stress_workload():
            """Execute stress commands and return result."""
            run_info = _run_stress_command(
                configs=configs,
                timeout=timeout,
                output_dir=results_dir,
                gpu_enabled=gpu_enabled,
            )

            # Fail immediately if GPU stress was requested but failed
            gpu_stderr = (run_info.get("gpu_result", {}).get("stderr") or "").lower()
            if gpu_enabled and run_info.get("gpu_attempted") and not run_info.get("gpu_result", {}).get("success", False):
                gpu_not_supported = (
                    "unrecognized option" in gpu_stderr
                    or "unknown stressor" in gpu_stderr
                    or "failed to find stressor 'gpu'" in gpu_stderr
                    or "gpu stressor" in gpu_stderr
                    or "not found" in gpu_stderr
                )
                if gpu_not_supported:
                    error_msg = f"GPU stressor failed: {gpu_stderr[:200] if gpu_stderr else 'unknown error'}"
                    metrics, cpu_yaml_error = _parse_stress_ng_yaml_metrics(run_info.get("cpu_yaml_metrics", ""))
                    if cpu_yaml_error:
                        error_msg = f"{error_msg}; CPU YAML parse error: {cpu_yaml_error}"
                    metrics = _normalize_key_metric(metrics, key_metric_name=key_metric_name, success=False)
                    _attach_native_yaml_file(
                        run_info.get("cpu_yaml_metrics", ""),
                        f"{test_id}_stress_cpu_metrics_native.yaml",
                    )
                    _attach_native_yaml_file(
                        run_info.get("gpu_yaml_metrics", ""),
                        f"{test_id}_stress_gpu_metrics_native.yaml",
                    )
                    _attach_log_file(run_info.get("cpu_stderr_log", ""), "stress_cpu_stderr.log")
                    _attach_log_file(run_info.get("gpu_stderr_log", ""), "stress_gpu_stderr.log")
                    return Result(
                        name=f"{test_id} - {test_display_name}",
                        metadata={
                            "status": False,
                            "message": error_msg,
                            "gpu_tool": run_info.get("gpu_tool", "unknown"),
                            "stress_success": False,
                        },
                        extended_metadata={
                            "stress_commands": {
                                "cpu": _command_to_text(run_info.get("cpu_command")),
                                "gpu": _command_to_text(run_info.get("gpu_command")),
                            },
                            "stress_output_dir": results_dir,
                            "stress_cpu_metrics_native_yaml": run_info.get("cpu_yaml_metrics", ""),
                            "stress_gpu_metrics_native_yaml": run_info.get("gpu_yaml_metrics", ""),
                        },
                        metrics=metrics,
                    )

            message = "Host stress completed" if run_info["success"] else "Host stress execution failed"
            metrics: Dict[str, Metrics] = {}
            yaml_errors: List[str] = []

            if run_info.get("cpu_attempted"):
                cpu_metrics, cpu_yaml_error = _parse_stress_ng_yaml_metrics(run_info.get("cpu_yaml_metrics", ""))
                if cpu_yaml_error:
                    yaml_errors.append(f"CPU YAML parse error: {cpu_yaml_error}")
                else:
                    metrics.update(cpu_metrics)

            if run_info.get("gpu_attempted"):
                gpu_metrics, gpu_yaml_error = _parse_stress_ng_yaml_metrics(run_info.get("gpu_yaml_metrics", ""))
                if gpu_yaml_error:
                    yaml_errors.append(f"GPU YAML parse error: {gpu_yaml_error}")
                else:
                    metrics.update(gpu_metrics)

            if yaml_errors:
                run_info["success"] = False
                message = "Host stress execution failed: " + "; ".join(yaml_errors)

            # If GPU stress was requested but the expected key metric is absent,
            # treat run as failed to avoid reporting "passed" with -1 key metric.
            if gpu_enabled and run_info.get("gpu_attempted") and key_metric_name not in metrics:
                run_info["success"] = False
                message = "Host stress execution failed: expected GPU metric not found"

            metrics = _normalize_key_metric(metrics, key_metric_name=key_metric_name, success=run_info["success"])
            _attach_native_yaml_file(
                run_info.get("cpu_yaml_metrics", ""),
                f"{test_id}_stress_cpu_metrics_native.yaml",
            )
            _attach_native_yaml_file(
                run_info.get("gpu_yaml_metrics", ""),
                f"{test_id}_stress_gpu_metrics_native.yaml",
            )
            _attach_log_file(run_info.get("cpu_stderr_log", ""), "stress_cpu_stderr.log")
            _attach_log_file(run_info.get("gpu_stderr_log", ""), "stress_gpu_stderr.log")

            return Result(
                name=f"{test_id} - {test_display_name}",
                metadata={
                    "status": run_info["success"],
                    "message": message,
                    "gpu_tool": run_info.get("gpu_tool", "unknown"),
                    "stress_success": run_info["success"],
                },
                extended_metadata={
                    "stress_commands": {
                        "cpu": _command_to_text(run_info.get("cpu_command")),
                        "gpu": _command_to_text(run_info.get("gpu_command")),
                    },
                    "stress_output_dir": results_dir,
                    "stress_cpu_metrics_native_yaml": run_info.get("cpu_yaml_metrics", ""),
                    "stress_gpu_metrics_native_yaml": run_info.get("gpu_yaml_metrics", ""),
                },
                metrics=metrics,
            )

        result = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=_stress_workload,
            test_name=test_name,
            configs=configs,
        )

    validate_test_results(
        test_name=test_name,
        results=result,
        configs=configs,
        get_kpi_config=get_kpi_config,
    )

    summarize_test_results(
        results=result,
        test_name=test_name,
        configs=configs,
        get_kpi_config=get_kpi_config,
    )

    cache_result(result)

    logger.info(f"Completed host stress test: {test_display_name}")
