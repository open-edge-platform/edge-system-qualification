# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CPU temperature telemetry module.

Collects CPU thermal readings via the Linux hwmon sysfs interface, reading
directly from ``/sys/class/hwmon/hwmon*/`` directories exposed by the
``coretemp`` driver (Intel) or ``k10temp`` driver (AMD) on Linux.

Falls back gracefully on systems where thermal sensors are not exposed
(e.g., some VMs or containers without host sensor passthrough).

Metrics collected:
    package_c:  CPU package temperature in Celsius (Package id 0 / Tdie).
    core_max_c: Maximum temperature across all individual CPU cores (°C).
                Not reported on AMD k10temp which does not expose per-core sensors.
"""

import glob
import logging
import os
import time

from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetryConfig, TelemetrySample

logger = logging.getLogger(__name__)

MODULE_NAME = "cpu_temp"

# Hwmon driver names that expose CPU temperature data
_CPU_HWMON_DRIVERS = frozenset({"coretemp", "k10temp"})


def _find_cpu_hwmon_dir():
    """
    Return ``(hwmon_dir, driver_name)`` for the first CPU thermal hwmon entry,
    or ``(None, None)`` if none is found.
    """
    for hwmon_dir in sorted(glob.glob("/sys/class/hwmon/hwmon*/")):
        name_file = os.path.join(hwmon_dir, "name")
        try:
            with open(name_file) as f:
                driver = f.read().strip()
            if driver in _CPU_HWMON_DRIVERS:
                return hwmon_dir.rstrip("/"), driver
        except OSError:
            pass
    return None, None


def _read_milli_celsius(path):
    """Read a sysfs ``temp*_input`` file (millidegrees) and return °C, or None."""
    try:
        with open(path) as f:
            return float(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


class CpuTempModule(BaseTelemetryModule):
    """
    Collects CPU temperature via the Linux hwmon sysfs interface.

    Reads ``/sys/class/hwmon/hwmon*/`` directories for ``coretemp`` (Intel)
    or ``k10temp`` (AMD) drivers. Temperature files are read directly from
    sysfs without any third-party library dependency.

    Metrics collected:
        package_c:  CPU package temperature in Celsius.
        core_max_c: Maximum individual core temperature in Celsius
                    (coretemp driver only).
    """

    module_name = MODULE_NAME

    def get_default_config(self):
        return {
            "title": {"display": True, "text": "CPU Temperature"},
            "scales": {
                "package_c": {"display": True, "label": "Package", "unit": "\u00b0C"},
                "core_max_c": {"display": True, "label": "Core Max", "unit": "\u00b0C"},
            },
            "thresholds": {
                "package_c": {"warning": 90},
                "core_max_c": {"warning": 90},
            },
        }

    def __init__(self, config: TelemetryConfig) -> None:
        super().__init__(config)

    def is_available(self) -> bool:
        hwmon_dir, _ = _find_cpu_hwmon_dir()
        return hwmon_dir is not None

    def collect_sample(self) -> TelemetrySample:
        raw: dict = {}
        hwmon_dir, _ = _find_cpu_hwmon_dir()
        if hwmon_dir:
            core_values = []
            for input_path in sorted(glob.glob(os.path.join(hwmon_dir, "temp*_input"))):
                temp = _read_milli_celsius(input_path)
                if temp is None:
                    continue
                label_path = input_path.replace("_input", "_label")
                try:
                    with open(label_path) as f:
                        label = f.read().strip().lower()
                except OSError:
                    label = ""

                if "package id" in label or label in ("package", "tdie", "tctl"):
                    # Prefer Tdie over Tctl for k10temp (Tdie is actual die temp)
                    if "package_c" not in raw or label == "tdie":
                        raw["package_c"] = round(temp, 1)
                elif label.startswith("core "):
                    core_values.append(temp)

            if core_values:
                raw["core_max_c"] = round(max(core_values), 1)

        values = self._filter_values(raw)
        sample = TelemetrySample(timestamp=time.time(), values=values)
        self.check_thresholds(values)
        return sample
