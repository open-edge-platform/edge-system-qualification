# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package power telemetry module (ESQ-specific).

Measures Intel CPU package power in real time using the Linux powercap RAPL
interface (sysfs energy counters). Two consecutive energy readings are
differenced to compute instantaneous power consumption.

No external tools (e.g., ``turbostat``, ``perf``) are required.
Requires read access to ``/sys/class/powercap/intel-rapl/.../energy_uj``.
Access can be granted without root via ``sudo scripts/system-setup.sh``.

Metrics collected:
    package_power_w: CPU package power (Watts).
    dram_power_w:    DRAM power (Watts), if the RAPL DRAM domain is exposed.
    core_power_w:    CPU core power (Watts), if available.
    uncore_power_w:  Uncore / GPU power (Watts), if available.
"""

import logging
import time
from pathlib import Path

from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetryConfig, TelemetrySample

logger = logging.getLogger(__name__)

MODULE_NAME = "package_power"

# Sysfs paths
_POWERCAP_BASE = "/sys/class/powercap"
_RAPL_ROOT = f"{_POWERCAP_BASE}/intel-rapl"


def _read_energy_uj(path: str) -> float:
    """Read energy counter in microjoules from a sysfs file."""
    try:
        with open(path, "r") as fh:
            return float(fh.read().strip())
    except (FileNotFoundError, PermissionError, ValueError, IOError):
        return -1.0


def _find_rapl_zones() -> dict:
    """
    Discover available RAPL zones and sub-zones on the current system.

    Returns a dict mapping logical name -> energy_uj sysfs path for:
    - "package" (intel-rapl:0)
    - "core"    (intel-rapl:0:0, zone name == "core")
    - "uncore"  (intel-rapl:0:0, zone name == "uncore")
    - "dram"    (intel-rapl:0:1 or whichever subzone is named "dram")
    """
    zones: dict = {}
    rapl_root = Path(_RAPL_ROOT)
    if not rapl_root.exists():
        return zones

    # Top-level package zones (intel-rapl:N)
    for pkg_path in sorted(rapl_root.glob("intel-rapl:[0-9]*")):
        if pkg_path.name.count(":") != 1:
            continue  # skip sub-zones at top level listing
        # Only use package 0 (physical package) for simplicity
        if not pkg_path.name.endswith(":0"):
            continue

        energy_path = str(pkg_path / "energy_uj")
        if _read_energy_uj(energy_path) >= 0:
            zones["package"] = energy_path

        # Sub-zones (intel-rapl:0:N)
        for sub_path in sorted(pkg_path.glob("intel-rapl:0:[0-9]*")):
            name_path = sub_path / "name"
            try:
                zone_name = name_path.read_text().strip().lower()
            except Exception:
                continue
            sub_energy_path = str(sub_path / "energy_uj")
            if _read_energy_uj(sub_energy_path) >= 0:
                zones[zone_name] = sub_energy_path  # "core", "uncore", "dram", etc.

    return zones


class PackagePowerModule(BaseTelemetryModule):
    """
    Measures CPU package power via Intel RAPL energy counters.

    Power is derived by differencing consecutive energy readings:
        power_W = (energy_t2 - energy_t1) / (t2 - t1) / 1e6

    The energy counter wraps at ``max_energy_range_uj``; this module handles
    wrap-around transparently.
    """

    module_name = MODULE_NAME

    def get_default_config(self):
        return {
            "chart_type": "area",
            "title": {"display": True, "text": "Package Power"},
            "scales": {
                "package_power_w": {"display": True, "label": "Package Power", "unit": "W"},
                "uncore_power_w": {"display": True, "label": "Uncore Power", "unit": "W"},
                "core_power_w": {"display": True, "label": "Core Power", "unit": "W"},
                "dram_power_w": {"display": True, "label": "DRAM Power", "unit": "W"},
            },
        }

    def __init__(self, config: TelemetryConfig) -> None:
        super().__init__(config)
        self._zones: dict = {}  # name -> energy_uj sysfs path
        self._max_energy: dict = {}  # name -> max_energy_range_uj (for wrap detection)
        self._prev_energy: dict = {}  # name -> last reading (µJ)
        self._prev_time: float = 0.0
        self._zones = _find_rapl_zones()
        self._read_max_energy()
        # Prime the first energy reading so the first real sample is meaningful
        if self._zones:
            self._take_snapshot()

    def _read_max_energy(self) -> None:
        """Read max_energy_range_uj for each zone (used for wrap detection)."""
        for name, energy_path in self._zones.items():
            max_path = energy_path.replace("energy_uj", "max_energy_range_uj")
            try:
                with open(max_path, "r") as fh:
                    self._max_energy[name] = float(fh.read().strip())
            except Exception:
                self._max_energy[name] = 2**32  # conservative fallback

    def _take_snapshot(self) -> None:
        """Record current energy readings and timestamp."""
        self._prev_time = time.monotonic()
        for name, path in self._zones.items():
            val = _read_energy_uj(path)
            if val >= 0:
                self._prev_energy[name] = val

    def is_available(self) -> bool:
        return bool(self._zones)

    def collect_sample(self) -> TelemetrySample:
        if not self._prev_energy:
            self._take_snapshot()
            return TelemetrySample(timestamp=time.time(), values={})

        now = time.monotonic()
        delta_t = now - self._prev_time
        if delta_t <= 0:
            return TelemetrySample(timestamp=time.time(), values={})

        raw: dict = {}
        for name, path in self._zones.items():
            curr = _read_energy_uj(path)
            if curr < 0:
                continue
            prev = self._prev_energy.get(name, -1)
            if prev < 0:
                self._prev_energy[name] = curr
                continue

            delta_uj = curr - prev
            # Handle counter wrap-around
            if delta_uj < 0:
                delta_uj += self._max_energy.get(name, 2**32)

            power_w = (delta_uj / 1e6) / delta_t
            metric_name = f"{name}_power_w"
            raw[metric_name] = round(power_w, 3)
            self._prev_energy[name] = curr

        self._prev_time = now
        values = self._filter_values(raw)
        sample = TelemetrySample(timestamp=time.time(), values=values)
        self.check_thresholds(values)
        return sample
