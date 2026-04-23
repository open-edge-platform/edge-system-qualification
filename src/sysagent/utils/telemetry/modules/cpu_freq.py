# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CPU frequency telemetry module.

Collects CPU frequency statistics via psutil. Reports current, min, and max
frequencies in MHz for the entire system (not per-core, to minimize overhead).
"""

import logging
import time

import psutil

from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetryConfig, TelemetrySample

logger = logging.getLogger(__name__)

MODULE_NAME = "cpu_freq"


class CpuFreqModule(BaseTelemetryModule):
    """
    Collects CPU frequency information using psutil.

    Metrics collected:
        current_mhz: Current CPU frequency in MHz.
    """

    module_name = MODULE_NAME

    def get_default_config(self):
        return {
            "title": {"display": True, "text": "CPU Frequency"},
            "scales": {
                "current_mhz": {"display": True, "label": "Current Frequency", "unit": "MHz"},
            },
        }

    def __init__(self, config: TelemetryConfig) -> None:
        super().__init__(config)
        # Prime psutil so first real call has no startup cost
        try:
            psutil.cpu_freq(percpu=False)
        except Exception:
            pass

    def is_available(self) -> bool:
        try:
            freq = psutil.cpu_freq(percpu=False)
            return freq is not None
        except Exception:
            return False

    def collect_sample(self) -> TelemetrySample:
        raw: dict = {}
        try:
            freq = psutil.cpu_freq(percpu=False)
            if freq is not None:
                raw["current_mhz"] = round(float(freq.current), 2)
        except Exception as exc:
            logger.debug("cpu_freq sample error: %s", exc)

        values = self._filter_values(raw)
        sample = TelemetrySample(timestamp=time.time(), values=values)
        self.check_thresholds(values)
        return sample
