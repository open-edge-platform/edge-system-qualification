# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import List

import psutil

from esq.utils.telemetry.providers.platform_telemetry.base import BaseCollector
from esq.utils.telemetry.providers.platform_telemetry.models import MetricSample


class CpuCollector(BaseCollector):
    name = "cpu"

    def __init__(self):
        self._running = False
        self._last_energy_uj: float | None = None
        self._last_power_ts: float | None = None
        self._last_pgpg_kb: float | None = None
        self._last_bw_ts: float | None = None

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def collect_once(self) -> List[MetricSample]:
        if not self._running:
            return []

        now = datetime.utcnow().isoformat() + "Z"
        usage = psutil.cpu_percent(interval=None)
        freq = psutil.cpu_freq()
        current_freq = float(freq.current) if freq else 0.0
        mem = psutil.virtual_memory()
        memory_util = float(mem.percent)
        power_w, power_origin = self._read_cpu_package_power_w()
        temp_c, temp_origin = self._read_cpu_temperature_c()
        bandwidth_mb_s, bandwidth_origin = self._read_cpu_bandwidth_mb_s()
        return [
            MetricSample(
                timestamp_utc=now,
                collector=self.name,
                device="CPU",
                metric_name="cpu.utilization",
                value=float(usage),
                unit="%",
                tags={"vendor": "generic", "platform": "x86"},
            ),
            MetricSample(
                timestamp_utc=now,
                collector=self.name,
                device="CPU",
                metric_name="cpu.frequency_mhz",
                value=current_freq,
                unit="MHz",
                tags={"vendor": "generic", "platform": "x86"},
            ),
            MetricSample(
                timestamp_utc=now,
                collector=self.name,
                device="CPU",
                metric_name="cpu.memory_utilization",
                value=memory_util,
                unit="%",
                tags={"vendor": "generic", "platform": "x86"},
            ),
            MetricSample(
                timestamp_utc=now,
                collector=self.name,
                device="CPU",
                metric_name="cpu.power_w",
                value=power_w,
                unit="W",
                tags={
                    "vendor": "generic",
                    "platform": "x86",
                    "metric_origin": power_origin,
                },
            ),
            MetricSample(
                timestamp_utc=now,
                collector=self.name,
                device="CPU",
                metric_name="cpu.temperature_c",
                value=temp_c,
                unit="°C",
                tags={
                    "vendor": "generic",
                    "platform": "x86",
                    "metric_origin": temp_origin,
                },
            ),
            MetricSample(
                timestamp_utc=now,
                collector=self.name,
                device="CPU",
                metric_name="cpu.bandwidth_mb_s",
                value=bandwidth_mb_s,
                unit="MB/s",
                tags={
                    "vendor": "generic",
                    "platform": "x86",
                    "metric_origin": bandwidth_origin,
                },
            ),
        ]

    def _read_cpu_bandwidth_mb_s(self) -> tuple[float, str]:
        vmstat = Path("/proc/vmstat")
        if not vmstat.exists():
            return 0.0, "placeholder_cpu_bw_unavailable"

        try:
            rows = vmstat.read_text(encoding="utf-8").splitlines()
        except PermissionError:
            return 0.0, "placeholder_cpu_bw_permission_denied"
        except OSError:
            return 0.0, "placeholder_cpu_bw_unavailable"

        counters: dict[str, float] = {}
        for row in rows:
            parts = row.split()
            if len(parts) != 2:
                continue
            key, raw = parts
            if key not in {"pgpgin", "pgpgout"}:
                continue
            try:
                counters[key] = float(raw)
            except ValueError:
                continue

        if "pgpgin" not in counters or "pgpgout" not in counters:
            return 0.0, "placeholder_cpu_bw_unavailable"

        now = time.time()
        curr_total_kb = counters["pgpgin"] + counters["pgpgout"]

        if self._last_pgpg_kb is None or self._last_bw_ts is None:
            self._last_pgpg_kb = curr_total_kb
            self._last_bw_ts = now
            return 0.0, "proc_vmstat_warmup"

        delta_kb = curr_total_kb - self._last_pgpg_kb
        delta_s = max(now - self._last_bw_ts, 1e-6)

        self._last_pgpg_kb = curr_total_kb
        self._last_bw_ts = now

        if delta_kb < 0:
            return 0.0, "proc_vmstat_reset"

        # pgpgin/pgpgout are reported in KB since boot.
        mb_s = (delta_kb / 1024.0) / delta_s
        return max(0.0, float(mb_s)), "proc_vmstat_pgpg_io_approx"

    @staticmethod
    def _read_cpu_temperature_c() -> tuple[float, str]:
        permission_denied = False

        # Prefer package/CPU thermal zones when available.
        thermal_root = Path("/sys/class/thermal")
        if thermal_root.exists():
            for zone in sorted(thermal_root.glob("thermal_zone*")):
                type_path = zone / "type"
                temp_path = zone / "temp"
                if not temp_path.exists():
                    continue

                try:
                    zone_type = type_path.read_text(encoding="utf-8").strip().lower()
                except PermissionError:
                    permission_denied = True
                    continue
                except OSError:
                    continue

                if not any(token in zone_type for token in ("x86_pkg_temp", "package", "cpu", "soc")):
                    continue

                try:
                    raw = float(temp_path.read_text(encoding="utf-8").strip())
                    return max(0.0, raw / 1000.0), "sysfs_thermal_zone"
                except PermissionError:
                    permission_denied = True
                except (OSError, ValueError):
                    continue

        # Fall back to hwmon package/core temperatures.
        hwmon_root = Path("/sys/class/hwmon")
        if hwmon_root.exists():
            for hw in sorted(hwmon_root.glob("hwmon*")):
                name = ""
                try:
                    name = (hw / "name").read_text(encoding="utf-8").strip().lower()
                except PermissionError:
                    permission_denied = True
                    continue
                except OSError:
                    continue

                if name and name not in {"coretemp", "k10temp", "zenpower", "cpu_thermal", "soc_thermal"}:
                    continue

                for temp_file in sorted(hw.glob("temp*_input")):
                    try:
                        raw = float(temp_file.read_text(encoding="utf-8").strip())
                        return max(0.0, raw / 1000.0), "sysfs_hwmon"
                    except PermissionError:
                        permission_denied = True
                    except (OSError, ValueError):
                        continue

        # Last fallback: psutil sensors API when available.
        try:
            sensors = psutil.sensors_temperatures()
        except (AttributeError, OSError):
            sensors = {}

        if sensors:
            preferred = ("coretemp", "k10temp", "cpu_thermal", "soc_thermal")
            for key in preferred:
                entries = sensors.get(key, [])
                for entry in entries:
                    current = getattr(entry, "current", None)
                    if current is None:
                        continue
                    try:
                        return max(0.0, float(current)), "psutil_sensors"
                    except (TypeError, ValueError):
                        continue

            for entries in sensors.values():
                for entry in entries:
                    current = getattr(entry, "current", None)
                    if current is None:
                        continue
                    try:
                        return max(0.0, float(current)), "psutil_sensors"
                    except (TypeError, ValueError):
                        continue

        if permission_denied:
            return 0.0, "placeholder_cpu_temp_permission_denied"
        return 0.0, "placeholder_cpu_temp_unavailable"

    def _read_cpu_package_power_w(self) -> tuple[float, str]:
        energy, energy_status = self._read_total_rapl_energy_uj()
        now = time.time()

        if energy is None:
            instant, instant_status = self._read_total_rapl_instant_power_w()
            self._last_energy_uj = None
            self._last_power_ts = None
            if instant is not None:
                return instant, "sysfs_rapl_power_uw"
            if energy_status == "permission_denied" or instant_status == "permission_denied":
                return 0.0, "placeholder_rapl_permission_denied"
            return 0.0, "placeholder_rapl_unavailable"

        if self._last_energy_uj is None or self._last_power_ts is None:
            self._last_energy_uj = energy
            self._last_power_ts = now
            return 0.0, "sysfs_rapl_warmup"

        delta_energy = energy - self._last_energy_uj
        delta_time = max(now - self._last_power_ts, 1e-6)

        self._last_energy_uj = energy
        self._last_power_ts = now

        # Counter wrap/reset can happen; treat as warmup sample.
        if delta_energy < 0:
            return 0.0, "sysfs_rapl_reset"

        watts = (delta_energy / 1_000_000.0) / delta_time
        return max(0.0, float(watts)), "sysfs_rapl"

    @staticmethod
    def _read_total_rapl_energy_uj() -> tuple[float | None, str]:
        total = 0.0
        found = False
        saw_candidate = False
        permission_denied = False

        for energy_file in CpuCollector._primary_rapl_files("energy_uj"):
            saw_candidate = True
            try:
                total += float(energy_file.read_text(encoding="utf-8").strip())
                found = True
            except PermissionError:
                permission_denied = True
            except (OSError, ValueError):
                continue

        if found:
            return total, "ok"
        if permission_denied:
            return None, "permission_denied"
        if saw_candidate:
            return None, "read_error"
        return None, "unavailable"

    @staticmethod
    def _read_total_rapl_instant_power_w() -> tuple[float | None, str]:
        total = 0.0
        found = False
        saw_candidate = False
        permission_denied = False

        for power_file in CpuCollector._primary_rapl_files("power_uw"):
            saw_candidate = True
            try:
                total += float(power_file.read_text(encoding="utf-8").strip()) / 1_000_000.0
                found = True
            except PermissionError:
                permission_denied = True
            except (OSError, ValueError):
                continue

        if found:
            return total, "ok"
        if permission_denied:
            return None, "permission_denied"
        if saw_candidate:
            return None, "read_error"
        return None, "unavailable"

    @staticmethod
    def _primary_rapl_files(metric_file: str) -> list[Path]:
        """Return package-level RAPL files without subdomain/mmio double counting.

        Prefer intel-rapl package domains (intel-rapl:<N>) and use mmio only
        when intel-rapl package domains are unavailable.
        """
        candidate_roots = [
            # Native host path (works in host mode and some container setups).
            (Path("/sys/devices/virtual/powercap/intel-rapl"), "intel-rapl"),
            (Path("/sys/devices/virtual/powercap/intel-rapl-mmio"), "intel-rapl-mmio"),
            # Docker fallback path explicitly mounted by host-metrics overlay.
            (Path("/host_powercap/intel-rapl"), "intel-rapl"),
            (Path("/host_powercap/intel-rapl-mmio"), "intel-rapl-mmio"),
        ]

        def package_domains(root: Path, prefix: str) -> list[Path]:
            if not root.exists():
                return []
            domains: list[Path] = []
            for domain in root.glob(f"{prefix}:*"):
                # Keep only package domains like intel-rapl:0, not subdomains
                # such as intel-rapl:0:0 (core/uncore/etc.).
                if domain.name.count(":") != 1:
                    continue
                f = domain / metric_file
                if f.exists():
                    domains.append(f)
            return sorted(domains)

        for root, prefix in candidate_roots:
            files = package_domains(root, prefix)
            if files:
                return files

        return []
