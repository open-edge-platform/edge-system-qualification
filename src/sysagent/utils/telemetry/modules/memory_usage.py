# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Memory usage telemetry module.

Collects system RAM utilization via psutil. Lightweight and always available.
"""

import logging
import time

import psutil

from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetrySample

logger = logging.getLogger(__name__)

MODULE_NAME = "memory_usage"


class MemoryUsageModule(BaseTelemetryModule):
    """
    Collects system memory usage statistics using psutil.

    Metrics collected:
        used_percent: Percentage of RAM currently in use.
        available_gib: Available RAM in GiB (not used by any process).
        used_gib: RAM in active use by processes (GiB).
    """

    module_name = MODULE_NAME

    def get_default_config(self):
        return {
            "chart_type": "area",
            "title": {"display": True, "text": "Memory Usage"},
            "scales": {
                "used_percent": {"display": True, "label": "Usage", "unit": "%"},
                "available_gib": {"display": True, "label": "Available", "unit": "GB"},
                "used_gib": {"display": True, "label": "Usage", "unit": "GB"},
            },
            "thresholds": {
                "used_percent": {"warning": 90},
            },
            "axes": [
                {"id": "y", "position": "left", "metrics": ["used_percent"], "label": "Usage (%)"},
                {"id": "y1", "position": "right", "metrics": ["available_gib", "used_gib"], "label": "Memory (GB)"},
            ],
        }

    def is_available(self) -> bool:
        try:
            psutil.virtual_memory()
            return True
        except Exception:
            return False

    def collect_sample(self) -> TelemetrySample:
        raw: dict = {}
        try:
            vm = psutil.virtual_memory()
            raw["used_percent"] = round(float(vm.percent), 2)
            raw["available_gib"] = round(vm.available / (1024**3), 3)
            raw["used_gib"] = round(vm.used / (1024**3), 3)
        except Exception as exc:
            logger.debug("memory_usage sample error: %s", exc)

        values = self._filter_values(raw)
        sample = TelemetrySample(timestamp=time.time(), values=values)
        self.check_thresholds(values)
        return sample
