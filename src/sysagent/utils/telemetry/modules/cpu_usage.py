# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CPU usage telemetry module.

Collects system-wide CPU utilization percentage via psutil.
Uses non-blocking (interval=None) polling which measures utilization
since the last call — this is suitable for background sampling threads.
"""

import logging
import time

import psutil

from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetryConfig, TelemetrySample

logger = logging.getLogger(__name__)

MODULE_NAME = "cpu_usage"


class CpuUsageModule(BaseTelemetryModule):
    """
    Collects CPU usage percentage using psutil.

    Uses interval=None (non-blocking) so the background thread is not
    stalled by psutil's internal sleep. The first call after process start
    returns 0.0 for all CPUs (this is normal psutil behaviour); subsequent
    calls return meaningful values.

    Metrics collected:
        total_percent: System-wide CPU utilization (%).
    """

    module_name = MODULE_NAME

    def get_default_config(self):
        return {
            "chart_type": "area",
            "title": {"display": True, "text": "CPU Usage"},
            "scales": {
                "total_percent": {"display": True, "label": "CPU Usage", "unit": "%"},
            },
            "thresholds": {
                "total_percent": {"warning": 95},
            },
        }

    def __init__(self, config: TelemetryConfig) -> None:
        super().__init__(config)
        # Prime psutil cpu_percent so the first real sample is meaningful
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def is_available(self) -> bool:
        try:
            psutil.cpu_percent(interval=None)
            return True
        except Exception:
            return False

    def collect_sample(self) -> TelemetrySample:
        raw: dict = {}
        try:
            total = psutil.cpu_percent(interval=None)
            raw["total_percent"] = round(float(total), 2)
        except Exception as exc:
            logger.debug("cpu_usage sample error: %s", exc)

        values = self._filter_values(raw)
        sample = TelemetrySample(timestamp=time.time(), values=values)
        self.check_thresholds(values)
        return sample
