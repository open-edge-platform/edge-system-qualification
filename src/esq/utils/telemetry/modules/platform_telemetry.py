# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Framework-level system telemetry module (scaffold).

This module is the ESQ integration point for provider-orchestrated telemetry
collection. The active backend is the telecollect collector; the provider
registry is structured so additional backends can be added later by
implementing ``BaseFrameworkTelemetryProvider`` and listing them in the
profile YAML ``provider_order``.

Phase-1 behavior:
- Accept and expose module-specific options from profile YAML.
- Resolve provider scaffolding and compose metadata.
- Accept target_device hint (auto/cpu/igpu/dgpu/npu) for provider filtering.
- Publish scaffold placeholder samples until full provider execution is implemented.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict

from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetryConfig, TelemetrySample
from sysagent.utils.system.cache import SystemInfoCache
from sysagent.utils.system.ov_helper import get_openvino_gpu_devices, get_openvino_npu_devices

from esq.utils.telemetry.providers import get_provider

logger = logging.getLogger(__name__)


class PlatformTelemetryModule(BaseTelemetryModule):
    """Scaffold module for framework-level telemetry orchestration."""

    module_name = "platform_telemetry"
    _PLACEHOLDER_METRIC = "scaffold_placeholder"

    _GROUP_RULES = {
        "cpu": lambda k: "cpu_" in k,
        "gpu": lambda k: "gpu_" in k,
        "memory": lambda k: "mem_" in k,
        "thermal": lambda k: "temp" in k,
    }

    # Device order and per-section metric layout. ``System`` holds
    # host-level metrics (system memory, thermals, chassis power).
    _DEVICE_ORDER = ["System", "CPU", "iGPU", "dGPU", "NPU"]
    _TELECOLECT_DEVICE_METRICS = {
        "System": [
            ("system.memory_utilization", "%"),
            ("system.power_w", "W"),
            ("system.bandwidth_mb_s", "MB/s"),
        ],
        "CPU": [
            ("cpu.utilization", "%"),
            ("cpu.frequency_mhz", "MHz"),
            ("cpu.temperature_c", "°C"),
        ],
        "iGPU": [
            ("gpu.utilization", "%"),
            ("gpu.frequency_mhz", "MHz"),
            ("gpu.power_w", "W"),
            ("gpu.memory_utilization", "%"),
            ("gpu.bandwidth_mb_s", "MB/s"),
            ("gpu.temperature_c", "°C"),
        ],
        "dGPU": [
            ("gpu.utilization", "%"),
            ("gpu.frequency_mhz", "MHz"),
            ("gpu.power_w", "W"),
            ("gpu.memory_utilization", "%"),
            ("gpu.bandwidth_mb_s", "MB/s"),
            ("gpu.temperature_c", "°C"),
        ],
        "NPU": [
            ("npu.utilization", "%"),
            ("npu.frequency_mhz", "MHz"),
            ("npu.power_w", "W"),
            # NPU plugin total memory is not exposed by the kernel; only
            # an absolute ``memory_used_mb`` size is published. Keep this
            # row size-explicit so the chart/CSV label cannot be confused
            # with a 0-100 % reading.
            ("npu.memory_used_mb", "MB"),
            ("npu.bandwidth_mb_s", "MB/s"),
            ("npu.temperature_c", "°C"),
        ],
    }

    # Unit mappings for common metric patterns
    _UNIT_MAPPING = {
        # Compound / specific patterns first — these MUST match before the
        # short single-word patterns below (e.g. "mb_s" before "mb"); dict
        # iteration order is insertion order in Python 3.7+.
        "bandwidth_mb_s": "MB/s",
        "mb_s": "MB/s",
        "gb_s": "GB/s",
        # CPU metrics
        "percent": "%",
        "usage": "%",
        "frequency": "MHz",
        "freq": "MHz",
        "mhz": "MHz",
        "clock": "MHz",
        "power": "W",
        "watts": "W",
        # GPU metrics
        "utilization": "%",
        "load": "%",
        # Memory metrics
        "bytes": "bytes",
        "kb": "KB",
        "mb": "MB",
        "gb": "GB",
        "used": "bytes",
        "available": "bytes",
        "free": "bytes",
        # Temperature metrics
        "celsius": "°C",
        "temp": "°C",
        "temperature": "°C",
        # Other
        "count": "count",
        "samples": "count",
        "ppm": "ppm",
    }

    def __init__(self, config: TelemetryConfig) -> None:
        super().__init__(config)
        self.options: Dict[str, Any] = dict(config.options or {})
        # provider_order lists the active backend(s); platform_telemetry is
        # the only registered backend
        self.provider_order = list(
            self.options.get(
                "provider_order",
                ["platform_telemetry"],
            )
        )
        self.target_device = self._parse_target_device(self.options.get("target_device", "auto"))
        self.target_devices = self._parse_target_devices(self.options.get("target_devices"))
        if not self.target_devices and self.target_device != "auto":
            legacy_device = self._canonical_target_device_label(self.target_device)
            if legacy_device is not None:
                self.target_devices = [legacy_device]

        # Keep normalized values available to provider implementations.
        self.options["target_device"] = self.target_device
        self.options["target_devices"] = list(self.target_devices)
        self._provider_instances = []
        self._active_provider = None
        self._active_provider_contributors: list[str] = []

        for provider_name in self.provider_order:
            provider = get_provider(provider_name, self.options)
            if provider is not None:
                self._provider_instances.append(provider)

    def _select_provider(self):
        if self._active_provider is not None:
            return self._active_provider
        for provider in self._provider_instances:
            try:
                if provider.health():
                    self._active_provider = provider
                    logger.debug("[%s] selected provider: %s", self.module_name, provider.name)
                    return provider
            except Exception as exc:
                logger.debug("[%s] provider health check failed for %s: %s", self.module_name, provider.name, exc)
        return None

    def _iter_provider_candidates(self):
        if self._active_provider is not None:
            yield self._active_provider
        for provider in self._provider_instances:
            if provider is self._active_provider:
                continue
            yield provider

    @staticmethod
    def _metric_completeness_score(values: Dict[str, Any]) -> int:
        canonical_metrics = set()
        for raw_name, raw_value in (values or {}).items():
            if not isinstance(raw_value, (int, float)):
                continue
            if str(raw_name).lower() in {"ms_service_up"}:
                continue
            normalized_name, _device_type, _device_group = PlatformTelemetryModule._normalize_metric_name(str(raw_name))
            metric_suffix = normalized_name.split(".", 1)[-1]
            if metric_suffix in {"utilization_idle", "utilization_sys"}:
                continue
            canonical_metrics.add(metric_suffix)
        return len(canonical_metrics)

    @staticmethod
    def _has_real_metrics(values: Dict[str, Any]) -> bool:
        for raw_name, raw_value in (values or {}).items():
            if not isinstance(raw_value, (int, float)):
                continue
            if str(raw_name).lower() == "ms_service_up":
                continue
            return True
        return False

    @staticmethod
    def _canonical_metric_key(device_group: str, normalized_metric_name: str) -> str:
        metric_suffix = normalized_metric_name.split(".", 1)[-1].replace(".", "_")
        if device_group == "System":
            return f"system_{metric_suffix}"
        if device_group == "CPU":
            return f"cpu_{metric_suffix}"
        if device_group == "iGPU":
            return f"igpu_{metric_suffix}"
        if device_group == "dGPU":
            return f"dgpu_{metric_suffix}"
        if device_group.startswith("dGPU[") and device_group.endswith("]"):
            index = device_group[5:-1].strip()
            if index.isdigit():
                return f"dgpu_{index}_{metric_suffix}"
        if device_group == "NPU":
            return f"npu_{metric_suffix}"
        return f"other_{metric_suffix}"

    def _categorize_metric(self, raw_name: str) -> tuple[str, str, str]:
        """Classify a metric into ``(normalized_name, device_type, device_group)``."""
        return self._normalize_metric_name(raw_name)

    def _canonicalize_provider_values(self, values: Dict[str, Any]) -> Dict[str, float]:
        grouped_metrics: Dict[str, Dict[str, float]] = {}

        for raw_name, raw_value in (values or {}).items():
            if not isinstance(raw_value, (int, float)):
                continue
            if str(raw_name).lower() == "ms_service_up":
                continue

            normalized_name, _device_type, device_group = self._categorize_metric(str(raw_name))
            if device_group not in grouped_metrics:
                grouped_metrics[device_group] = {}
            grouped_metrics[device_group][normalized_name] = self._scale_metric_value(normalized_name, float(raw_value))

        grouped_metrics = self._resolve_gpu_groups(grouped_metrics)

        canonical_values: Dict[str, float] = {}
        for device_group, metric_map in grouped_metrics.items():
            for normalized_name, scaled_value in metric_map.items():
                canonical_key = self._canonical_metric_key(device_group, normalized_name)
                canonical_values[canonical_key] = float(scaled_value)

        return canonical_values

    @staticmethod
    def _parse_target_device(raw_value: Any) -> str:
        allowed = {"auto", "cpu", "igpu", "dgpu", "npu"}
        value = str(raw_value or "auto").strip().lower()
        if value in allowed:
            return value
        logger.warning(
            "[%s] Unsupported target_device '%s'; falling back to 'auto'.",
            PlatformTelemetryModule.module_name,
            raw_value,
        )
        return "auto"

    @staticmethod
    def _canonical_target_device_label(raw_value: Any) -> str | None:
        value = str(raw_value or "").strip()
        if not value:
            return None

        lowered = value.lower()
        if lowered == "auto":
            return None
        if lowered == "system":
            return "System"
        if lowered == "cpu":
            return "CPU"
        if lowered == "igpu":
            return "iGPU"
        if lowered == "dgpu":
            return "dGPU"
        if lowered == "npu":
            return "NPU"

        match = re.fullmatch(r"dgpu\[(\d+)\]", lowered)
        if match:
            return f"dGPU[{match.group(1)}]"

        logger.warning(
            "[%s] Unsupported target_devices entry '%s'; ignoring.",
            PlatformTelemetryModule.module_name,
            raw_value,
        )
        return None

    @classmethod
    def _parse_target_devices(cls, raw_value: Any) -> list[str]:
        if raw_value is None:
            return []

        if isinstance(raw_value, str):
            raw_items = [item.strip() for item in raw_value.split(",") if item.strip()]
        elif isinstance(raw_value, (list, tuple, set)):
            raw_items = [str(item).strip() for item in raw_value if str(item).strip()]
        else:
            raw_items = [str(raw_value).strip()]

        normalized: list[str] = []
        for item in raw_items:
            canonical = cls._canonical_target_device_label(item)
            if canonical and canonical not in normalized:
                normalized.append(canonical)
        return normalized

    def _is_requested_device(self, device_group: str) -> bool:
        # ``System`` is a host-level section (package power, memory utilization,
        # memory bandwidth). It is always emitted regardless of the compute
        # ``target_devices`` filter so the report consistently surfaces it.
        if device_group == "System":
            return True
        if not self.target_devices:
            return True
        if device_group in self.target_devices:
            return True
        if device_group == "dGPU" and "dGPU[0]" in self.target_devices:
            return True
        if device_group == "dGPU[0]" and "dGPU" in self.target_devices:
            return True
        if device_group.startswith("dGPU[") and device_group.endswith("]") and "dGPU" in self.target_devices:
            return True
        return False

    def _filter_canonical_values_for_target_devices(self, values: Dict[str, float]) -> Dict[str, float]:
        if not self.target_devices:
            return values

        filtered: Dict[str, float] = {}
        for metric_name, metric_value in values.items():
            _normalized_name, _device_type, device_group = self._categorize_metric(metric_name)
            if self._is_requested_device(device_group):
                filtered[metric_name] = metric_value
        return filtered

    def _filter_device_layout_for_target_devices(self, device_groups: list[str]) -> list[str]:
        if not self.target_devices:
            return device_groups

        filtered = [device for device in device_groups if self._is_requested_device(device)]
        return filtered

    @staticmethod
    def _infer_unit(metric_name: str) -> str:
        """Infer metric unit from metric name pattern."""
        name_lower = str(metric_name or "").lower()
        for pattern, unit in PlatformTelemetryModule._UNIT_MAPPING.items():
            if pattern in name_lower:
                return unit
        return "value"

    @staticmethod
    def _normalize_metric_name(raw_name: str) -> tuple[str, str, str]:
        """Normalize raw metric name to telecollect format.

        Returns:
          (normalized_metric_name, device_type, device_group)
          - device_type: cpu|gpu|npu|unknown
          - device_group: CPU|iGPU|dGPU|NPU|Other|GPU_TOKEN::<token>
        """
        name_lower = raw_name.lower()
        device_group = "Other"
        device_token = ""

        if name_lower.startswith("igpu_"):
            metric_type = name_lower[len("igpu_") :]
            device_type = "gpu"
            device_group = "iGPU"
        elif name_lower.startswith("dgpu_"):
            remainder = name_lower[len("dgpu_") :]
            first, sep, rest = remainder.partition("_")
            device_type = "gpu"
            if sep and first.isdigit() and rest:
                device_group = f"dGPU[{first}]"
                metric_type = rest
            else:
                device_group = "dGPU"
                metric_type = remainder
        elif name_lower.startswith("cpu_"):
            metric_type = name_lower[len("cpu_") :]
            device_type = "cpu"
            device_group = "CPU"
        elif name_lower.startswith("npu_"):
            metric_type = name_lower[len("npu_") :]
            device_type = "npu"
            device_group = "NPU"
        elif name_lower.startswith("system_"):
            metric_type = name_lower[len("system_") :]
            device_type = "system"
            device_group = "System"
        else:
            # Strip known Prometheus provider prefixes before any other processing.
            for _pfx in ("node_", "dcgm_"):
                if name_lower.startswith(_pfx):
                    name_lower = name_lower[len(_pfx):]
                    break

            # Detect device type
            if "cpu_" in name_lower or "cpu." in name_lower:
                device_type = "cpu"
            elif "gpu_" in name_lower or "gpu." in name_lower or "dgpu" in name_lower or "igpu" in name_lower:
                device_type = "gpu"
            elif "npu_" in name_lower or "npu." in name_lower:
                device_type = "npu"
            elif name_lower.startswith("mem_"):
                # Host-level memory → System group by default.
                device_type = "system"
            elif name_lower.startswith("temp_"):
                # Host thermal-zone temperature stays under CPU.
                device_type = "cpu"
            else:
                device_type = "unknown"

            # Extract metric type
            # Remove device prefix
            metric_type = name_lower
            for prefix in [f"{device_type}_", f"{device_type}."]:
                if metric_type.startswith(prefix):
                    metric_type = metric_type[len(prefix):]
                    break

            # Parse provider-appended device token suffix (e.g. '__device_0000_03_00_0').
            for marker in ("__device_", "__gpu_", "__card_", "__adapter_", "__pci_", "__index_"):
                marker_idx = metric_type.find(marker)
                if marker_idx == -1:
                    continue
                token = metric_type[marker_idx + len(marker) :]
                metric_type = metric_type[:marker_idx]
                token = token.strip("_")
                if token:
                    device_token = token
                break

        # Normalize metric type names
        metric_type_mapping = {
            # Utilization
            "utilization": "utilization",
            "util": "utilization",
            "usage": "utilization",
            "load": "utilization",
            "eng_usage.render": "utilization",
            # Frequency
            "frequency": "frequency_mhz",
            "frequency_mhz": "frequency_mhz",
            "freq": "frequency_mhz",
            "freq_mhz": "frequency_mhz",
            "clock": "frequency_mhz",
            "clock_mhz": "frequency_mhz",
            # Power
            "power": "power_w",
            "power_w": "power_w",
            "watts": "power_w",
            # Memory/Bandwidth
            "memory_utilization": "memory_utilization",
            "mem_utilization": "memory_utilization",
            "memory_used": "memory_utilization",
            "bandwidth": "bandwidth_mb_s",
            "bandwidth_mb_s": "bandwidth_mb_s",
            "bw_mb_s": "bandwidth_mb_s",
            "throughput": "bandwidth_mb_s",
            # Temperature
            "temperature": "temperature_c",
            "temperature_c": "temperature_c",
            "temp": "temperature_c",
            "temp_c": "temperature_c",
            "celsius": "temperature_c",
            # Compound suffixes from Prometheus-style provider emissions.
            "frequency_avg_frequency": "frequency_mhz",   # cpu_frequency_avg_frequency
            "usage_user": "utilization",                   # cpu_usage_user  (primary CPU util)
            "usage_system": "utilization_sys",             # cpu_usage_system (secondary, not in METRIC_ORDER)
            "usage_idle": "utilization_idle",              # cpu_usage_idle   (secondary, not in METRIC_ORDER)
            "engine_usage_usage": "utilization",           # gpu_engine_usage_usage
            "mem_available_percent": "memory_utilization", # mem_available_percent
            "temp_temp": "temperature_c",                  # temp_temp
        }
        
        normalized_type = metric_type_mapping.get(metric_type, metric_type)

        # Device group inference
        if device_group not in {"System", "CPU", "iGPU", "dGPU", "NPU"} and not device_group.startswith("dGPU["):
            if device_type == "system":
                device_group = "System"
            elif device_type == "cpu":
                device_group = "CPU"
            elif device_type == "npu":
                device_group = "NPU"
            elif device_type == "gpu":
                if "igpu" in name_lower or "integrated" in name_lower:
                    device_group = "iGPU"
                elif "dgpu" in name_lower or "discrete" in name_lower:
                    device_group = "dGPU"
                elif device_token:
                    device_group = f"GPU_TOKEN::{device_token}"
                else:
                    device_group = "dGPU"
            else:
                device_group = "Other"

        # Route CPU package-level power/memory metrics into ``System``.
        if device_group == "CPU" and normalized_type in {"power_w", "memory_utilization", "bandwidth_mb_s"}:
            return (f"system.{normalized_type}", "system", "System")

        return (f"{device_type}.{normalized_type}", device_type, device_group)

    @staticmethod
    def _device_sort_key(device_group: str) -> tuple[int, int, str]:
        if device_group == "System":
            return (0, 0, device_group)
        if device_group == "CPU":
            return (1, 0, device_group)
        if device_group == "iGPU":
            return (2, 0, device_group)
        if device_group.startswith("dGPU[") and device_group.endswith("]"):
            try:
                idx = int(device_group[5:-1])
            except ValueError:
                idx = 0
            return (3, idx, device_group)
        if device_group == "dGPU":
            return (3, 0, device_group)
        if device_group == "NPU":
            return (4, 0, device_group)
        return (9, 0, device_group)

    @staticmethod
    def _scale_metric_value(normalized_metric_name: str, value: float) -> float:
        """Normalize provider values into display units expected by telecollect naming.

        Currently fixes frequency scale to MHz when provider emits Hz/kHz.
        """
        if not isinstance(value, (int, float)):
            return value

        # frequency_mhz should be in MHz. Some providers expose Hz (or kHz).
        if normalized_metric_name.endswith(".frequency_mhz"):
            # Typical ranges:
            # - kHz input: ~1_000_000 to ~6_000_000
            # - Hz input:  ~1_000_000_000 to ~6_000_000_000
            if value >= 100_000_000:
                return float(value) / 1_000_000.0  # Hz -> MHz
            if value >= 10_000:
                return float(value) / 1_000.0      # kHz -> MHz
        return float(value)

    def _resolve_gpu_groups(self, grouped_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Resolve GPU token groups into platform-aware iGPU/dGPU labels."""
        token_groups = sorted(k for k in grouped_metrics.keys() if k.startswith("GPU_TOKEN::"))
        if not token_groups:
            return grouped_metrics

        expected_devices = self._get_platform_expected_devices()
        has_igpu_slot = "iGPU" in expected_devices
        expected_dgpu_slots = [d for d in expected_devices if d == "dGPU" or d.startswith("dGPU[")]

        has_igpu = "iGPU" in grouped_metrics
        has_dgpu = any(k == "dGPU" or k.startswith("dGPU[") for k in grouped_metrics.keys())

        def _dgpu_label(slot_index: int) -> str:
            if expected_dgpu_slots and slot_index < len(expected_dgpu_slots):
                return expected_dgpu_slots[slot_index]
            if len(expected_dgpu_slots) > 1:
                return f"dGPU[{slot_index}]"
            return "dGPU"

        dgpu_index = 0
        if has_dgpu:
            existing_dgpu_indices = []
            for key in grouped_metrics.keys():
                if key.startswith("dGPU[") and key.endswith("]"):
                    try:
                        existing_dgpu_indices.append(int(key[5:-1]))
                    except ValueError:
                        continue
            if existing_dgpu_indices:
                dgpu_index = max(existing_dgpu_indices) + 1
            else:
                dgpu_index = 1

        for token_key in token_groups:
            metrics_for_token = grouped_metrics.pop(token_key, {})
            if not metrics_for_token:
                continue

            if has_igpu_slot and not has_igpu:
                target_group = "iGPU"
                has_igpu = True
            else:
                target_group = _dgpu_label(dgpu_index)
                dgpu_index += 1
                has_dgpu = True

            grouped_metrics.setdefault(target_group, {}).update(metrics_for_token)

        return grouped_metrics

    @staticmethod
    def _classify_openvino_device(device: Dict[str, Any]) -> str:
        """Classify OpenVINO device entry into igpu/dgpu/npu/unknown."""
        quick = device.get("quick_access", {}) or {}
        device_type = str(quick.get("device_type") or device.get("device_type") or "").lower()
        full_name = str(quick.get("full_device_name") or device.get("full_device_name") or "").lower()
        device_name = str(device.get("device_name") or "").lower()

        text = " ".join([device_type, full_name, device_name])
        if "npu" in text or "ai boost" in text:
            return "npu"
        if "integrated" in text or "(igpu)" in text or " igpu" in text:
            return "igpu"
        if "discrete" in text or "(dgpu)" in text or " dgpu" in text:
            return "dgpu"
        return "unknown"

    def _get_platform_expected_devices(self) -> list[str]:
        """Return device list driven by platform-supported hardware/OpenVINO info."""
        expected: list[str] = ["System", "CPU"]
        igpu_count = 0
        dgpu_count = 0
        npu_count = 0

        try:
            hw = SystemInfoCache().get_hardware_info() or {}
            gpu_devices = ((hw.get("gpu") or {}).get("devices") or [])
            npu_devices = ((hw.get("npu") or {}).get("devices") or [])

            for dev in gpu_devices:
                ov = (dev.get("openvino") or {}).copy()
                ov.setdefault("device_name", dev.get("device_name", ""))
                kind = self._classify_openvino_device(ov)
                if kind == "igpu":
                    igpu_count += 1
                elif kind == "dgpu":
                    dgpu_count += 1

            npu_count += len(npu_devices)
        except Exception as exc:
            logger.debug("[%s] hardware cache device detection failed: %s", self.module_name, exc)

        # Fallback to direct OpenVINO query when cache is incomplete.
        try:
            if igpu_count == 0 and dgpu_count == 0:
                for dev in get_openvino_gpu_devices():
                    kind = self._classify_openvino_device(dev)
                    if kind == "igpu":
                        igpu_count += 1
                    elif kind == "dgpu":
                        dgpu_count += 1
                    else:
                        # Unknown GPU kind defaults to discrete bucket.
                        dgpu_count += 1
        except Exception as exc:
            logger.debug("[%s] OpenVINO GPU detection fallback failed: %s", self.module_name, exc)

        try:
            if npu_count == 0:
                npu_count = len(get_openvino_npu_devices())
        except Exception as exc:
            logger.debug("[%s] OpenVINO NPU detection fallback failed: %s", self.module_name, exc)

        if igpu_count > 0:
            expected.append("iGPU")

        if dgpu_count <= 1:
            if dgpu_count == 1:
                expected.append("dGPU")
        else:
            for idx in range(dgpu_count):
                expected.append(f"dGPU[{idx}]")

        if npu_count > 0:
            expected.append("NPU")

        return expected

    @staticmethod
    def _infer_friendly_device_name(raw_device_name: str) -> str:
        """Map raw device identifiers to telecollect friendly names (CPU, iGPU, dGPU, NPU)."""
        name_lower = str(raw_device_name or "").lower()
        
        if "cpu" in name_lower or "processor" in name_lower:
            return "CPU"
        elif "igpu" in name_lower or "integrated" in name_lower:
            return "iGPU"
        elif "dgpu" in name_lower or "discrete" in name_lower or "gpu" in name_lower:
            return "dGPU"
        elif "npu" in name_lower:
            return "NPU"
        
        return "CPU"  # Default fallback

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "chart_type": "line",
            "title": {"display": True, "text": "Framework System Telemetry (Scaffold)"},
            "scales": {
                self._PLACEHOLDER_METRIC: {
                    "display": True,
                    "label": "Scaffold Placeholder",
                    "unit": "state",
                }
            },
        }

    def is_available(self) -> bool:
        # Keep this module loadable while it is still scaffolded so profile
        # validation and plumbing can be exercised end-to-end.
        return True

    def collect_sample(self) -> TelemetrySample:
        merged_values: Dict[str, Any] = {}
        active_provider = None
        contributors: list[str] = []

        for provider in self._iter_provider_candidates():
            try:
                if not provider.health():
                    continue
            except Exception as exc:
                logger.debug("[%s] provider health check failed for %s: %s", self.module_name, provider.name, exc)
                continue

            try:
                values = provider.collect_sample() or {}
            except Exception as exc:
                logger.debug(
                    "[%s] provider sample collection failed for %s: %s",
                    self.module_name,
                    provider.name,
                    exc,
                )
                continue

            canonical_values = self._canonicalize_provider_values(values)
            canonical_values = self._filter_canonical_values_for_target_devices(canonical_values)
            if not canonical_values:
                continue

            added_metrics = 0
            for metric_name, metric_value in canonical_values.items():
                if metric_name in merged_values:
                    continue
                merged_values[metric_name] = metric_value
                added_metrics += 1

            if added_metrics:
                if active_provider is None:
                    active_provider = provider
                contributors.append(provider.name)
                logger.debug(
                    "[%s] merged %s canonical metrics from provider %s",
                    self.module_name,
                    added_metrics,
                    provider.name,
                )

        if merged_values:
            self._active_provider = active_provider
            self._active_provider_contributors = contributors
            logger.debug(
                "[%s] merged sample across providers: primary=%s contributors=%s metrics=%s",
                self.module_name,
                active_provider.name if active_provider is not None else None,
                contributors,
                len(merged_values),
            )
            return TelemetrySample(timestamp=time.time(), values=merged_values)

        # Scaffold fallback path when provider metrics are not available yet.
        self._active_provider_contributors = []
        if self._provider_instances:
            logger.debug(
                "[%s] providers discovered (scaffold fallback, target_device=%s): %s",
                self.module_name,
                self.target_devices or self.target_device,
                [p.name for p in self._provider_instances],
            )
        return TelemetrySample(
            timestamp=time.time(),
            values={
                self._PLACEHOLDER_METRIC: 0,
            },
        )

    def close(self) -> None:
        """Tear down every provider this module instantiated.

        Idempotent: each provider is closed at most once, and exceptions
        are swallowed per-provider so one stuck teardown cannot block the
        rest.  After this call, every external resource (qmmd daemon,
        in-process collector threads) owned by ``platform_telemetry`` is
        released.
        """
        for provider in self._provider_instances:
            try:
                provider.close()
            except Exception as exc:
                logger.debug(
                    "[%s] provider %s close() raised: %s",
                    self.module_name,
                    getattr(provider, "name", type(provider).__name__),
                    exc,
                )
        self._provider_instances = []
        self._active_provider = None
        self._active_provider_contributors = []

    def get_summary(self) -> Dict[str, Any]:
        summary = super().get_summary()
        config = dict(summary.get("configs") or {})

        # When real provider metrics exist, replace scaffold chart config so
        # report renderers do not keep plotting the placeholder-only layout.
        metric_keys = [k for k in (summary.get("averages") or {}).keys() if k != self._PLACEHOLDER_METRIC]
        if not metric_keys:
            # Fallback to sample values in case averages are empty for short runs.
            for sample in summary.get("samples") or []:
                values = sample.get("values") or {}
                for key in values.keys():
                    if key != self._PLACEHOLDER_METRIC:
                        metric_keys.append(key)
            metric_keys = sorted(set(metric_keys))

        if metric_keys:
            scales = dict(config.get("scales") or {})
            scales.pop(self._PLACEHOLDER_METRIC, None)
            for key in metric_keys:
                scales.setdefault(
                    key,
                    {
                        "display": True,
                        "label": key,
                        "unit": self._infer_unit(key),
                    },
                )
            config.update(
                {
                    "metrics": metric_keys,
                    "title": {"display": True, "text": "Framework System Telemetry"},
                    "scales": scales,
                }
            )

        summary["config"] = config

        # Build grouped virtual modules using telecollect device structure
        virtual_modules: Dict[str, Dict[str, Any]] = {}
        if metric_keys:
            _raw_scales = config.get("scales")
            scales_cfg: Dict[str, Any] = _raw_scales if isinstance(_raw_scales, dict) else {}

            # First, normalize all metric names to telecollect format and group by device.
            normalized_metrics: Dict[str, Dict[str, str]] = {}
            for raw_metric_name in metric_keys:
                normalized_name, _device_type, device_group = self._categorize_metric(raw_metric_name)
                if device_group not in normalized_metrics:
                    normalized_metrics[device_group] = {}
                normalized_metrics[device_group][normalized_name] = raw_metric_name

            normalized_metrics = self._resolve_gpu_groups(normalized_metrics)
            normalized_metrics = {
                device_group: metric_map
                for device_group, metric_map in normalized_metrics.items()
                if self._is_requested_device(device_group)
            }

            def _build_normalized_samples(metric_map: Dict[str, str]) -> list[Dict[str, Any]]:
                """Build samples using normalized metric keys used by config.metrics/scales."""
                filtered = []
                for s in summary.get("samples") or []:
                    vals = s.get("values") or {}
                    sub_vals: Dict[str, Any] = {}
                    for normalized_name, raw_name in metric_map.items():
                        if raw_name not in vals:
                            continue
                        sub_vals[normalized_name] = self._scale_metric_value(normalized_name, vals[raw_name])
                    if sub_vals:
                        filtered.append({"timestamp": s.get("timestamp"), "values": sub_vals})
                return filtered

            def _build_module_suffix(device_group: str) -> str:
                lower_group = device_group.lower().replace("[", "_").replace("]", "")
                if lower_group == "system":
                    return "system"
                if lower_group == "cpu":
                    return "cpu"
                if lower_group == "igpu":
                    return "igpu"
                if lower_group.startswith("dgpu"):
                    return lower_group
                if lower_group == "npu":
                    return "npu"
                return (
                    lower_group.replace(" ", "_")
                    .replace("-", "_")
                    .replace(".", "_")
                    .replace("__", "_")
                )

            # Build virtual modules for each resolved device group.
            ordered_groups = sorted(normalized_metrics.keys(), key=self._device_sort_key)
            for device_group in ordered_groups:
                if device_group not in normalized_metrics:
                    continue

                normalized_by_device = normalized_metrics[device_group]
                if not normalized_by_device:
                    continue

                device_samples = _build_normalized_samples(normalized_by_device)
                if not device_samples:
                    continue

                # Create scales for this device using normalized names
                device_scales = {}
                for normalized_name, raw_name in normalized_by_device.items():
                    unit = (scales_cfg.get(raw_name, {}) or {}).get("unit")
                    if not unit:
                        # Infer from pattern
                        if ".utilization" in normalized_name:
                            unit = "%"
                        elif ".frequency_mhz" in normalized_name:
                            unit = "MHz"
                        elif ".power_w" in normalized_name:
                            unit = "W"
                        elif ".bandwidth_mb_s" in normalized_name:
                            unit = "MB/s"
                        elif ".temperature_c" in normalized_name:
                            unit = "°C"
                        elif ".memory_utilization" in normalized_name:
                            unit = "MB" if normalized_name.startswith("npu.") else "%"
                        else:
                            unit = "value"
                    
                    device_scales[normalized_name] = {
                        "display": True,
                        "label": normalized_name,
                        "unit": unit,
                    }

                # Get aggregates for normalized names (convert from raw names)
                device_averages = {}
                device_min_max = {}
                for normalized_name, raw_name in normalized_by_device.items():
                    if raw_name in (summary.get("averages") or {}):
                        device_averages[normalized_name] = self._scale_metric_value(
                            normalized_name,
                            summary["averages"][raw_name],
                        )
                    if raw_name in (summary.get("min_max") or {}):
                        raw_mm = summary["min_max"][raw_name] or {}
                        # min/max may carry the MISSING_VALUE (-1) sentinel
                        # when every observation was unavailable; pass it
                        # through unchanged.
                        mm_entry: dict = {}
                        if "min" in raw_mm and raw_mm.get("min") is not None:
                            mm_entry["min"] = self._scale_metric_value(normalized_name, raw_mm["min"])
                        if "max" in raw_mm and raw_mm.get("max") is not None:
                            mm_entry["max"] = self._scale_metric_value(normalized_name, raw_mm["max"])
                        if mm_entry:
                            device_min_max[normalized_name] = mm_entry

                # Determine friendly device name and stable module suffix.
                friendly_name = device_group
                module_suffix = _build_module_suffix(device_group)

                title = f"Framework System Telemetry - {friendly_name}"
                module_key = f"{self.module_name}_{module_suffix}"

                virtual_modules[module_key] = {
                    "module": module_key,
                    "device_name": friendly_name,
                    "configs": {
                        "metrics": list(normalized_by_device.keys()),
                        "thresholds": {},
                        "chart_type": config.get("chart_type", "line"),
                        "title": {"display": True, "text": title},
                        "scales": device_scales,
                        "axes": [],
                    },
                    "sample_count": len(device_samples),
                    "averages": device_averages,
                    "min_max": device_min_max,
                    "samples": device_samples,
                    "status": "active",
                    "active_provider": self._active_provider.name if self._active_provider else None,
                    "active_provider_contributors": list(self._active_provider_contributors),
                    "target_device": self.target_device,
                    "target_devices": list(self.target_devices),
                }

            # Keep dashboard layout stable for platform-supported devices even when
            # provider metrics are temporarily missing.
            expected_device_layout = self._get_platform_expected_devices()
            expected_device_layout = self._filter_device_layout_for_target_devices(expected_device_layout)
            for expected_device in expected_device_layout:
                expected_key = f"{self.module_name}_{_build_module_suffix(expected_device)}"
                if expected_key in virtual_modules:
                    continue

                expected_metric_group = "dGPU" if expected_device.startswith("dGPU[") else expected_device
                expected_rows = list(self._TELECOLECT_DEVICE_METRICS.get(expected_metric_group, []))
                expected_scales = {
                    metric_name: {
                        "display": True,
                        "label": metric_name,
                        "unit": metric_unit,
                    }
                    for metric_name, metric_unit in expected_rows
                }

                virtual_modules[expected_key] = {
                    "module": expected_key,
                    "device_name": expected_device,
                    "configs": {
                        "metrics": list(expected_scales.keys()),
                        "thresholds": {},
                        "chart_type": config.get("chart_type", "line"),
                        "title": {"display": True, "text": f"Framework System Telemetry - {expected_device}"},
                        "scales": expected_scales,
                        "axes": [],
                    },
                    "sample_count": 0,
                    "averages": {},
                    "min_max": {},
                    "samples": [],
                    "status": "inactive",
                    "active_provider": self._active_provider.name if self._active_provider else None,
                    "active_provider_contributors": list(self._active_provider_contributors),
                    "target_device": self.target_device,
                    "target_devices": list(self.target_devices),
                }

        if virtual_modules:
            # Re-order so device sections always follow ``_DEVICE_ORDER``
            # (System, CPU, iGPU, dGPU, NPU) regardless of which were
            # populated by provider samples vs. added by the fallback loop.
            virtual_modules = {
                k: virtual_modules[k]
                for k in sorted(
                    virtual_modules.keys(),
                    key=lambda mk: self._device_sort_key(
                        str(virtual_modules[mk].get("device_name") or mk)
                    ),
                )
            }
            summary["virtual_modules"] = virtual_modules
            # Consumers should render virtual modules only, not the raw parent module.
            summary["prefer_virtual_modules"] = True

        active_provider_name = self._active_provider.name if self._active_provider is not None else None
        provider_status = []
        for provider in self._provider_instances:
            try:
                provider_status.append(provider.get_status())
            except Exception as exc:
                provider_status.append({"name": provider.name, "health": False, "error": str(exc)})
        summary.update(
            {
                "status": "active" if active_provider_name and metric_keys else "scaffold",
                "placeholder_message": (
                    "platform_telemetry loaded successfully, but external provider runtime "
                    "collection is not implemented yet."
                    if not metric_keys
                    else ""
                ),
                "provider_order": list(self.provider_order),
                "providers_discovered": [provider.name for provider in self._provider_instances],
                "provider_status": provider_status,
                "active_provider": active_provider_name,
                "active_provider_contributors": list(self._active_provider_contributors),
                "target_device": self.target_device,
                "target_devices": list(self.target_devices),
            }
        )
        return summary
