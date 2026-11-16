# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Translate Metrics Manager's ``/api/v1/metrics/latest`` payload into the
canonical ``<prefix>_<metric>`` keys expected by
``esq.utils.telemetry.modules.platform_telemetry.PlatformTelemetryModule``.

Field names below were captured from a live ``intel/metrics-manager:2026.2.0``
container (Telegraf inputs: ``cpu``, ``mem``, ``temp`` filtered to the CPU
package sensor, ``read_cpu_freq.sh``, ``qmassa_reader.py``, ``npu_reader.py``)
exposed through its Prometheus relay:

* ``cpu_usage_user`` / ``cpu_usage_system`` / ``cpu_usage_idle`` — percent,
  tag ``cpu=cpu-total``.
* ``cpu_frequency_avg_frequency`` — **kHz** (``/sys/.../scaling_cur_freq``).
* ``temp_temp`` — degC, pre-filtered server-side to
  ``sensor=coretemp_package_id_*`` (CPU package only).
* ``mem_used_percent`` / ``mem_available_percent`` / ``mem_total`` / ``mem_used``.
* ``gpu_engine_usage_usage`` — percent (0-100), tags ``engine``, ``gpu_id``.
* ``gpu_frequency`` — **MHz already**, tags ``type=cur_freq``, ``gpu_id``.
* ``gpu_power`` — Watts, tags ``type=gpu_cur_power|pkg_cur_power``, ``gpu_id``.
* ``npu_utilization`` (percent), ``npu_frequency`` (MHz), ``npu_power`` (W),
  ``npu_temperature`` (degC), ``npu_memory_mb`` (MB already) — tag ``host`` only
  (the current ESQ profile mapping assumes one NPU). The 2026.2.0 API emitted
  NPU frequency in MHz during a live workload, despite the API documentation
  describing it as Hz.

Metrics Manager currently does not expose CPU package power (no RAPL input
in its Telegraf config), CPU/system bandwidth, GPU memory utilization, or GPU
temperature. ESQ's supplemental external collector only supplies these gaps.

``gpu_id`` is merely a renderD-node ordinal (``renderD128`` -> 0,
``renderD129`` -> 1, ...). It is correlated to the expanded capabilities
response by PCI BDF; the iGPU/dGPU category and software capabilities come
from Metrics Manager rather than a duplicate ESQ PCI-ID classifier.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .device_inventory import gpu_groups_by_render_id

logger = logging.getLogger(__name__)

_CPU_FREQ_NAME = "cpu_frequency_avg_frequency"
_TEMP_NAME = "temp_temp"
_MEM_USED_PERCENT_NAME = "mem_used_percent"
_GPU_ENGINE_USAGE_NAME = "gpu_engine_usage_usage"
_GPU_FREQUENCY_NAME = "gpu_frequency"
_GPU_POWER_NAME = "gpu_power"
_NPU_FIELD_NAMES = {
    "npu_utilization": "npu_utilization",
    "npu_frequency": "npu_frequency_mhz",
    "npu_power": "npu_power_w",
    "npu_temperature": "npu_temperature_c",
    "npu_memory_mb": "npu_memory_used_mb",
}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def flatten_latest_metrics(
    latest_metrics: Dict[str, Any], devices: List[Dict[str, Any]] | None = None
) -> Dict[str, float]:
    """Flatten a ``GET /api/v1/metrics/latest`` ``metrics`` dict into canonical keys."""
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for entry in (latest_metrics or {}).values():
        name = str(entry.get("name") or "")
        if not name:
            continue
        by_name.setdefault(name, []).append(entry)

    flattened: Dict[str, float] = {}

    _flatten_cpu(by_name, flattened)
    _flatten_memory(by_name, flattened)
    _flatten_npu(by_name, flattened)
    _flatten_gpu(by_name, flattened, devices or [])

    return flattened


def _single_value(by_name: Dict[str, List[Dict[str, Any]]], name: str) -> float | None:
    entries = by_name.get(name) or []
    if not entries:
        return None
    return _as_float((entries[0].get("fields") or {}).get("value"))


def _flatten_cpu(by_name: Dict[str, List[Dict[str, Any]]], out: Dict[str, float]) -> None:
    usage_user = _single_value(by_name, "cpu_usage_user")
    usage_system = _single_value(by_name, "cpu_usage_system")
    if usage_user is not None or usage_system is not None:
        out["cpu_utilization"] = (usage_user or 0.0) + (usage_system or 0.0)

    freq_khz = _single_value(by_name, _CPU_FREQ_NAME)
    if freq_khz is not None:
        out["cpu_frequency_mhz"] = freq_khz / 1000.0

    temp_c = _single_value(by_name, _TEMP_NAME)
    if temp_c is not None:
        out["cpu_temperature_c"] = temp_c


def _flatten_memory(by_name: Dict[str, List[Dict[str, Any]]], out: Dict[str, float]) -> None:
    used_percent = _single_value(by_name, _MEM_USED_PERCENT_NAME)
    if used_percent is not None:
        out["system_memory_utilization"] = used_percent


def _flatten_npu(by_name: Dict[str, List[Dict[str, Any]]], out: Dict[str, float]) -> None:
    for raw_name, canonical_key in _NPU_FIELD_NAMES.items():
        value = _single_value(by_name, raw_name)
        if value is None:
            continue
        out[canonical_key] = value


def _flatten_gpu(
    by_name: Dict[str, List[Dict[str, Any]]],
    out: Dict[str, float],
    devices: List[Dict[str, Any]],
) -> None:
    gpu_ids: set[int] = set()
    for measurement in (_GPU_ENGINE_USAGE_NAME, _GPU_FREQUENCY_NAME, _GPU_POWER_NAME):
        for entry in by_name.get(measurement, []):
            raw_gpu_id = (entry.get("tags") or {}).get("gpu_id")
            try:
                gpu_ids.add(int(raw_gpu_id))
            except (TypeError, ValueError):
                continue

    gpu_groups = gpu_groups_by_render_id(devices, gpu_ids)
    for gpu_id in sorted(gpu_ids):
        utilization_pct = _gpu_utilization_percent(by_name, gpu_id)
        frequency_mhz = _gpu_frequency_mhz(by_name, gpu_id)
        power_w = _gpu_power_w(by_name, gpu_id)
        device_group = gpu_groups.get(gpu_id)
        if device_group is None:
            logger.warning(
                "Metrics Manager reported GPU id %s without a matching expanded capability record; "
                "skipping its GPU metrics",
                gpu_id,
            )
            continue
        if device_group == "iGPU":
            prefix = "igpu"
        elif device_group == "dGPU":
            prefix = "dgpu"
        elif device_group.startswith("dGPU[") and device_group.endswith("]"):
            index = device_group[5:-1]
            prefix = f"dgpu_{index}"
        else:
            logger.warning("Unsupported Metrics Manager GPU category %r for id %s", device_group, gpu_id)
            continue

        if utilization_pct is not None:
            out[f"{prefix}_utilization"] = utilization_pct
        if frequency_mhz is not None:
            out[f"{prefix}_frequency_mhz"] = frequency_mhz
        if power_w is not None:
            out[f"{prefix}_power_w"] = power_w


def _gpu_utilization_percent(by_name: Dict[str, List[Dict[str, Any]]], gpu_id: int) -> float | None:
    best: float | None = None
    for entry in by_name.get(_GPU_ENGINE_USAGE_NAME, []):
        tags = entry.get("tags") or {}
        if str(tags.get("gpu_id")) != str(gpu_id):
            continue
        value = _as_float((entry.get("fields") or {}).get("value"))
        if value is None:
            continue
        if not 0.0 <= value <= 100.0:
            logger.warning(
                "Metrics Manager reported out-of-range GPU utilization %s for GPU id %s",
                value,
                gpu_id,
            )
            continue
        if best is None or value > best:
            best = value
    return best


def _gpu_frequency_mhz(by_name: Dict[str, List[Dict[str, Any]]], gpu_id: int) -> float | None:
    for entry in by_name.get(_GPU_FREQUENCY_NAME, []):
        tags = entry.get("tags") or {}
        if str(tags.get("gpu_id")) != str(gpu_id):
            continue
        if str(tags.get("type")) != "cur_freq":
            continue
        return _as_float((entry.get("fields") or {}).get("value"))
    return None


def _gpu_power_w(by_name: Dict[str, List[Dict[str, Any]]], gpu_id: int) -> float | None:
    by_type: Dict[str, float] = {}
    for entry in by_name.get(_GPU_POWER_NAME, []):
        tags = entry.get("tags") or {}
        if str(tags.get("gpu_id")) != str(gpu_id):
            continue
        value = _as_float((entry.get("fields") or {}).get("value"))
        if value is None:
            continue
        by_type[str(tags.get("type"))] = value
    # Prefer the GPU-domain reading; fall back to package-domain.
    return by_type.get("gpu_cur_power", by_type.get("pkg_cur_power"))
