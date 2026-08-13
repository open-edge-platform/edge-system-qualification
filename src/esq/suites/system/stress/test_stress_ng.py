# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Host machine stress-ng test.

Runs stress-ng directly on the host with configurable
CPU, memory, and GPU stressors. The active stressors, worker counts, load
percentages, and duration are all driven by profile parameters so the same
test function covers CPU-only, memory-only, iGPU-only, and mixed scenarios.

The profile duration is the default and can be overridden at runtime without
editing profiles via the per-suite environment variable
``ENV_SUITE_STRESS_NG_DURATION_SECONDS``. The variable is named after this test
file so it never collides with another suite's duration knob.

Metrics are collected from the native stress-ng YAML output file and exposed
as bogo-ops per second for KPI validation and telemetry correlation.
"""

import logging
import os
import threading
from pathlib import Path

import allure
import pytest
import yaml
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result, run_command
from sysagent.utils.system.drm import count_intel_drm_cards, resolve_intel_gpu_devnode

logger = logging.getLogger(__name__)

# Per-suite environment override for the stress duration. Named after this
# test file ("stress_ng") so it never collides with another suite's duration knob.
# The profile value is the default; export this to retune at runtime without
# editing profiles, e.g. ENV_SUITE_STRESS_NG_DURATION_SECONDS=300.
_DURATION_ENV_VAR = "ENV_SUITE_STRESS_NG_DURATION_SECONDS"


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


def _resolve_stress_duration(configs: dict) -> int:
    """Resolve the stress duration (seconds), honoring the per-suite env override.

    Priority: ``ENV_SUITE_STRESS_NG_DURATION_SECONDS`` env var > profile
    ``stress_duration_seconds`` > 60s default. Non-integer overrides are ignored
    with a warning. The resolved value is clamped to at least 1 second.
    """
    default_seconds = max(_safe_int(configs.get("stress_duration_seconds", 60), 60), 1)

    raw_override = os.environ.get(_DURATION_ENV_VAR)
    if raw_override is None:
        return default_seconds

    try:
        override = int(str(raw_override).strip())
    except (TypeError, ValueError):
        logger.warning(f"Ignoring non-integer {_DURATION_ENV_VAR}={raw_override!r}; using {default_seconds}s")
        return default_seconds

    override = max(override, 1)
    logger.info(f"Stress duration override from {_DURATION_ENV_VAR}: {override}s")
    return override


def _detect_intel_gpu_cards() -> int:
    """Count Intel GPU cards under DRM."""
    count = 0
    for vendor_file in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            with open(vendor_file, "r", encoding="utf-8") as file:
                vendor = file.read().strip().lower()
            if vendor == "0x8086":
                count += 1
        except OSError:
            continue
    return count


def _get_intel_drm_cards() -> list[str]:
    """Return Intel DRM card device nodes sorted by card index."""
    cards: list[tuple[int, str]] = []
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
        except (OSError, ValueError):
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


def _detect_intel_gpu_cards() -> int:
    """Count Intel GPU cards under DRM."""
    count = 0
    for vendor_file in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            with open(vendor_file, "r", encoding="utf-8") as file:
                vendor = file.read().strip().lower()
            if vendor == "0x8086":
                count += 1
        except OSError:
            continue
    return count


def _get_intel_drm_cards() -> list[str]:
    """Return Intel DRM card device nodes sorted by card index."""
    cards: list[tuple[int, str]] = []
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
        except (OSError, ValueError):
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


def _command_to_text(command: list[str] | None) -> str:
    """Convert command tokens to readable text safely."""
    if not command:
        return ""
    return " ".join(command)


def _build_cpu_stress_command(configs: dict) -> list[str] | None:
    """Build CPU/memory stress-ng command line from profile params."""
    duration = max(_safe_int(configs.get("stress_duration_seconds", 60), 60), 1)
    enable_cpu_stress = bool(configs.get("enable_cpu_stress", True))
    enable_memory_stress = bool(configs.get("enable_memory_stress", True))

    command: list[str] = [
        "stress-ng",
        "--timeout",
        f"{duration}s",
        "--metrics-brief",
    ]

    if enable_cpu_stress:
        cpu_workers = max(_safe_int(configs.get("stress_cpu_workers", 0), 0), 0)
        cpu_load = max(min(_safe_int(configs.get("stress_cpu_load", 90), 90), 100), 0)
        command.extend(
            [
                "--cpu",
                str(cpu_workers),
                "--cpu-load",
                str(cpu_load),
            ]
        )

    if enable_memory_stress:
        vm_workers = max(_safe_int(configs.get("stress_vm_workers", 0), 0), 0)
        # Default to a percentage of physical RAM so the workload scales with the
        # machine's physical memory instead of a fixed byte size. stress-ng treats
        # a "%" value for --vm-bytes as a percentage of total physical memory.
        vm_bytes = str(configs.get("stress_vm_bytes", "100%"))
        if vm_workers > 0:
            command.extend(
                [
                    "--vm",
                    str(vm_workers),
                    "--vm-bytes",
                    vm_bytes,
                    # Keep (and continuously re-dirty) the same mapping so the
                    # allocated pages stay resident in physical memory rather than
                    # being repeatedly unmapped/remapped, keeping the stress on RAM.
                    "--vm-keep",
                ]
            )

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


def _resolve_key_metric_name(configs: dict, gpu_requested: bool) -> str:
    """Resolve key metric name from profile config with sensible defaults."""
    configured_name = str(configs.get("key_metric_name", "")).strip()
    if configured_name:
        return configured_name
    return "gpu_bogo_ops_per_real_time" if gpu_requested else "cpu_bogo_ops_per_real_time"


def _normalize_key_metric(metrics: dict[str, Metrics], key_metric_name: str, success: bool) -> dict[str, Metrics]:
    """Mark one deterministic key metric and ensure it exists on failure."""
    normalized: dict[str, Metrics] = {}
    for metric_name, metric in metrics.items():
        normalized[metric_name] = Metrics(value=metric.value, unit=metric.unit, is_key_metric=False)

    existing = normalized.get(key_metric_name)
    if success and existing is not None:
        normalized[key_metric_name] = Metrics(value=existing.value, unit=existing.unit, is_key_metric=True)
        return normalized

    unit = existing.unit if existing is not None else _metric_unit_from_name(key_metric_name)
    normalized[key_metric_name] = Metrics(value=-1.0, unit=unit, is_key_metric=True)
    return normalized


def _build_gpu_stress_command(configs: dict, gpu_enabled: bool) -> tuple[list[str] | None, str]:
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
        gpu_devnode = resolve_intel_gpu_devnode(gpu_device_index)
        command = [
            "stress-ng",
            "--timeout",
            f"{duration}s",
            "--metrics-brief",
            "--gpu",
            str(gpu_workers),
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


def _run_command_worker(command: list[str], timeout: int, sink: dict, sink_key: str) -> None:
    """Run a command and store normalized result into sink."""
    result = run_command(command, timeout=timeout)
    sink[sink_key] = {
        "returncode": result.returncode if result else -1,
        "success": bool(result and result.returncode == 0),
        "stdout": result.stdout if result else "",
        "stderr": result.stderr if result else "",
        "command": command,
    }


def _parse_stress_ng_yaml_metrics(yaml_path: str) -> tuple[dict[str, Metrics], str]:
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
    except (OSError, yaml.YAMLError) as error:
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

    parsed: dict[str, Metrics] = {}
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
    cpu_command: list[str] | None,
    gpu_command: list[str] | None,
    timeout: int,
) -> tuple[dict, dict]:
    """Run CPU/GPU stress concurrently and return normalized command results."""
    results: dict[str, dict] = {}

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


def _write_command_logs(cpu_result: dict, gpu_result: dict, output_dir: str) -> dict[str, str]:
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


def _run_stress_command(configs: dict, timeout: int, output_dir: str, gpu_enabled: bool) -> dict:
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
    except OSError as error:
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


@allure.title("System Stress - stress-ng")
def test_stress_ng(
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
    """Run host-only stress-ng workload with configurable CPU, memory, and GPU stressors."""
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

    results_dir = os.path.join(core_data_dir, "data", "suites", "system", "stress", "results", test_id)
    results_resolved = str(Path(results_dir).resolve())
    results_dir = "".join(ch for ch in results_resolved)

    os.makedirs(results_dir, mode=0o770, exist_ok=True)
    ensure_dir_permissions(results_dir, uid=os.getuid(), gid=os.getgid(), mode=0o770)

    stress_ng_available = _check_command_available("stress-ng")
    intel_gpu_cards = count_intel_drm_cards()
    gpu_requested = bool(configs.get("enable_gpu_stress", True))
    gpu_enabled = gpu_requested and intel_gpu_cards > 0
    key_metric_name = _resolve_key_metric_name(configs, gpu_requested=gpu_requested)

    duration = _resolve_stress_duration(configs)
    # Propagate the resolved duration (default or env override) so the CPU and
    # GPU command builders driven by configs run for the same window.
    configs["stress_duration_seconds"] = duration
    timeout = max(_safe_int(configs.get("timeout", duration + 120), duration + 120), duration + 30)

    if not stress_ng_available:
        pytest.fail("stress-ng is not installed on host")

    if gpu_requested and not gpu_enabled:
        pytest.fail("GPU stress requested but no Intel iGPU detected on host")

    result = None
    test_interrupted = False
    test_failed = False
    failure_message = ""
    is_qualification = configs.get("labels", {}).get("type") == "qualification"

    # Wrap stress execution in execute_test_with_cache to enable telemetry
    # sampling scoped to the active workload window (scope=execution in profile).
    def _stress_workload():
        """Invoke stress-ng with the configured stressors and collect metrics."""
        _active = []
        if configs.get("enable_cpu_stress"):
            _active.append("cpu")
        if configs.get("enable_memory_stress"):
            _active.append("memory")
        if gpu_enabled:
            _active.append("gpu")
        logger.info(
            "Running stress-ng: duration=%ss, stressors=[%s]",
            duration,
            ", ".join(_active) if _active else "none",
        )
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
                        "gpu_tool": run_info.get("gpu_tool", "unknown"),
                        "stress_success": False,
                    },
                    extended_metadata={
                        "message": error_msg,
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
        metrics: dict[str, Metrics] = {}
        yaml_errors: list[str] = []

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
                "gpu_tool": run_info.get("gpu_tool", "unknown"),
                "stress_success": run_info["success"],
            },
            extended_metadata={
                "message": message,
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

    try:
        result = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=_stress_workload,
            test_name=test_name,
            configs=configs,
        )
    except KeyboardInterrupt:
        failure_message = "Interrupt detected during host stress test execution"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during host stress test execution: {e!s}"
        logger.error(failure_message, exc_info=True)

    # Ensure a result object always exists so the test terminates cleanly even
    # when interrupted before the workload produced a result.
    if result is None:
        base_metrics = _normalize_key_metric({}, key_metric_name=key_metric_name, success=False)
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={
                "status": False,
                "stress_success": False,
            },
            extended_metadata={
                "stress_output_dir": results_dir,
                "message": failure_message or "Host stress test did not complete",
            },
            metrics=base_metrics,
        )

    # When the stress-ng process itself failed (unexpected exit, subprocess
    # timeout), the result status is False and the test must also fail so the
    # outcome is not silently recorded as PASSED.
    if not test_interrupted and not test_failed:
        if not result.metadata.get("status", True):
            test_failed = True
            failure_message = result.extended_metadata.get("message", "stress-ng did not complete successfully")
            logger.error(f"stress-ng execution failed: {failure_message}")

    # Always validate and summarize so the result is recorded regardless of
    # interruption or unexpected errors during execution.
    try:
        validate_test_results(
            test_name=test_name,
            results=result,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as validation_error:
        logger.error(f"Validation failed: {validation_error}")

    try:
        summarize_test_results(
            results=result,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as summary_error:
        logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)

    # Only cache when the test completed without interruption or unexpected failure.
    # Caching an interrupted/failed placeholder result would cause subsequent runs
    # to find it in the cache and report a false "passed" status.
    if not test_interrupted and not test_failed:
        cache_result(result)

    logger.info(f"Completed host stress test: {test_display_name}")

    # Terminate cleanly: surface interrupts/errors as a proper test outcome
    # instead of leaving a broken/errored status behind.
    if test_interrupted:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
