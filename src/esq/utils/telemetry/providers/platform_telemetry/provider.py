# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Supplemental external telemetry provider for Metrics Manager coverage gaps."""

from __future__ import annotations

import logging
import re
from typing import Any

from ..base import BaseFrameworkTelemetryProvider
from .base import BaseCollector
from .cleanup import register
from .models import MetricSample
from .system.cpu_collector import CpuCollector
from .tools.gpu_collector import GpuCollector

logger = logging.getLogger(__name__)


class AdditionalMetricsProvider(BaseFrameworkTelemetryProvider):
    """Collect only CPU package power/bandwidth and GPU memory/temperature."""

    name = "additional_metrics"

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        self._cpu_collector = CpuCollector()
        self._gpu_collector = GpuCollector()
        self._collectors: list[BaseCollector] = [
            self._cpu_collector,
            self._gpu_collector,
        ]
        self._started = False
        self._last_error = ""
        register(self.close)

    def set_device_inventory(self, devices: list[dict[str, Any]]) -> None:
        self._gpu_collector.set_device_inventory(devices)

    def _ensure_started(self) -> None:
        if self._started:
            return
        for collector in self._collectors:
            collector.start()
        self._started = True

    def close(self) -> None:
        for collector in self._collectors:
            collector.stop()
        self._started = False

    def health(self) -> bool:
        self._ensure_started()
        return True

    def collect_sample(self) -> dict[str, float]:
        self._ensure_started()
        flattened: dict[str, float] = {}
        for collector in self._collectors:
            try:
                samples = collector.collect_once() or []
            except (OSError, RuntimeError, ValueError) as exc:
                self._last_error = f"{getattr(collector, 'name', '?')}: {exc}"
                logger.warning("[%s] collection failed: %s", self.name, self._last_error)
                continue

            for sample in samples:
                origin = str(sample.tags.get("metric_origin") or "")
                if sample.value < 0 or origin.startswith("placeholder_"):
                    logger.warning(
                        "[%s] skipping unavailable %s metric from %s",
                        self.name,
                        sample.metric_name,
                        getattr(collector, "name", type(collector).__name__),
                    )
                    continue
                if origin.endswith(("_warmup", "_reset")):
                    logger.debug(
                        "[%s] skipping transient %s metric from %s",
                        self.name,
                        sample.metric_name,
                        getattr(collector, "name", type(collector).__name__),
                    )
                    continue

                key = self._sample_to_key(sample)
                if key is not None:
                    flattened[key] = float(sample.value)
        return flattened

    def get_status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "collectors": [collector.name for collector in self._collectors],
            "started": self._started,
            "last_error": self._last_error,
            "health": self._started,
        }

    @staticmethod
    def _device_prefix(device_name: str) -> str:
        normalized = str(device_name or "").strip()
        if not normalized:
            return ""
        if normalized == "CPU":
            return "cpu"
        if normalized == "iGPU":
            return "igpu"
        if normalized == "dGPU":
            return "dgpu"
        if normalized == "NPU":
            return "npu"
        match = re.fullmatch(r"(iGPU|dGPU)(\d+)", normalized)
        if match:
            kind = "igpu" if match.group(1) == "iGPU" else "dgpu"
            index = int(match.group(2))
            return f"{kind}_{index - 1}" if index >= 2 else kind
        if normalized.startswith("dGPU[") and normalized.endswith("]"):
            index = normalized[5:-1].strip()
            if index.isdigit():
                return f"dgpu_{index}"
        return (
            normalized.lower()
            .replace("[", "_")
            .replace("]", "")
            .replace(" ", "_")
            .replace("-", "_")
        )

    @classmethod
    def _sample_to_key(cls, sample: MetricSample) -> str | None:
        prefix = cls._device_prefix(sample.device)
        if not prefix:
            return None
        metric = str(sample.metric_name or "")
        suffix = metric.split(".", 1)[1] if "." in metric else metric
        suffix = suffix.replace(".", "_")
        return f"{prefix}_{suffix}" if suffix else None
