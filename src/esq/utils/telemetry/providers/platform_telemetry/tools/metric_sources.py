# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unified ``MetricSource`` abstraction for the Intel GPU collector.

Historically the qmassa collector performed a per-metric waterfall with
provider-specific lookups inline (``xpu_util_by_pci.get(...)``,
``igt.get("utilization")``, ``self._sysfs_utilization(card)``). Each
provider exposed a different shape (PCI-keyed map vs. metric-keyed dict
vs. ``(value, origin)`` tuple), so the collector had to know each one
explicitly.

This module factors that out behind a single ``MetricSource`` protocol.
Every backfill provider (xpu-smi, intel_gpu_top, sysfs) implements the
same ``read(pci, metric)`` call and returns a ``MetricReading`` or
``None``. The collector keeps an ordered list of sources and falls
through them generically.

Values are returned in each metric's *native* unit:

* ``utilization``    — percent (0-100)
* ``frequency_mhz``  — MHz
* ``power_w``        — W
* ``temperature_c``  — °C
* ``bandwidth_mb_s`` — MB/s

The collector converts to its internal scale (fraction / Hz) at write
time, identical to the previous inline behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional, Protocol, Tuple, runtime_checkable

from esq.utils.telemetry.providers.platform_telemetry.tools.intel_gpu_top_utils import IntelGpuTopUtils
from esq.utils.telemetry.providers.platform_telemetry.tools.xpu_smi_utils import XpuSmiUtils


class MetricReading(NamedTuple):
    """A single per-(pci, metric) reading returned by a ``MetricSource``.

    ``domain`` is an optional sub-classifier carried only by metrics that
    need it (currently ``memory_utilization``, which distinguishes
    ``vram`` / ``gtt`` / ``smem`` / ``device``). For the simple
    single-value metrics it stays ``None``.
    """

    value: float
    origin: str
    domain: Optional[str] = None


# Canonical metric names recognised by every source.
METRIC_UTILIZATION = "utilization"
METRIC_FREQUENCY_MHZ = "frequency_mhz"
METRIC_POWER_W = "power_w"
METRIC_TEMPERATURE_C = "temperature_c"
METRIC_BANDWIDTH_MB_S = "bandwidth_mb_s"
METRIC_MEMORY_UTILIZATION = "memory_utilization"


@runtime_checkable
class MetricSource(Protocol):
    """Uniform per-(pci, metric) backfill source.

    Implementations adapt provider-specific shapes (PCI-keyed maps,
    metric-keyed dicts, ad-hoc sysfs reads) to a single ``read`` call so
    the collector can walk an ordered list of sources without knowing
    anything about how each one fetches its data.
    """

    name: str

    def refresh(self) -> None:
        """Called once per ``collect_once`` cycle. Sources that snapshot
        their data (e.g. by running a subprocess) should do it here so
        repeated ``read`` calls reuse the snapshot."""
        ...

    def read(
        self,
        pci: str,
        metric: str,
        *,
        card: Optional[Path] = None,
    ) -> Optional[MetricReading]:
        """Return a reading for ``metric`` on the device at PCI slot
        ``pci``, or ``None`` if this source has no data. ``card`` is the
        DRM ``Path`` resolved by the collector and is provided as a
        convenience for sysfs-style sources."""
        ...


# ── xpu-smi ──────────────────────────────────────────────────────────────────

class XpuSmiBackfillSource:
    """``MetricSource`` adapter over :class:`XpuSmiUtils`.

    Caches the four PCI-keyed maps for the lifetime of a collect cycle so
    repeated ``read()`` calls don't re-invoke the ``xpu-smi`` subprocess.
    """

    name = "xpu_smi"

    def __init__(self, utils: Optional[XpuSmiUtils] = None) -> None:
        self._utils = utils or XpuSmiUtils()
        self._power: Dict[str, float] = {}
        self._bandwidth: Dict[str, float] = {}
        self._utilization: Dict[str, float] = {}
        self._memory_util: Dict[str, float] = {}

    def refresh(self) -> None:
        self._power = self._utils.power_map_w()
        self._bandwidth = self._utils.bandwidth_map_mb_s()
        self._utilization = self._utils.utilization_map()
        self._memory_util = self._utils.memory_util_map()

    def read(self, pci: str, metric: str, *, card: Optional[Path] = None) -> Optional[MetricReading]:
        _ = card  # unused: xpu-smi is keyed by PCI
        key = pci.lower()
        if metric == METRIC_UTILIZATION:
            v = self._utilization.get(key)
        elif metric == METRIC_POWER_W:
            v = self._power.get(key)
        elif metric == METRIC_BANDWIDTH_MB_S:
            v = self._bandwidth.get(key)
        elif metric == METRIC_MEMORY_UTILIZATION:
            v = self._memory_util.get(key)
            if v is None:
                return None
            pct = max(0.0, min(100.0, float(v)))
            return MetricReading(pct, self.name, domain="device")
        else:
            return None
        return MetricReading(float(v), self.name) if v is not None else None


# ── intel_gpu_top ────────────────────────────────────────────────────────────

class IntelGpuTopBackfillSource:
    """``MetricSource`` adapter over :class:`IntelGpuTopUtils`."""

    name = "intel_gpu_top"

    def __init__(self, utils: Optional[IntelGpuTopUtils] = None) -> None:
        self._utils = utils or IntelGpuTopUtils()

    def refresh(self) -> None:
        # ``IntelGpuTopUtils`` snapshots per-PCI on demand and caches
        # internally; nothing to do at cycle start.
        return

    def read(self, pci: str, metric: str, *, card: Optional[Path] = None) -> Optional[MetricReading]:
        _ = card  # unused: igt is keyed by PCI
        snap = self._utils.snapshot(pci)
        if not snap:
            return None
        if metric == METRIC_UTILIZATION:
            v = snap.get("utilization")
        elif metric == METRIC_FREQUENCY_MHZ:
            v = snap.get("frequency_mhz")
            # 0 MHz from intel_gpu_top means "RC6 power-gated, no reading";
            # treat as missing so the next source (sysfs idle floor) is
            # given a chance.
            if v is not None and v <= 0:
                return None
        elif metric == METRIC_POWER_W:
            v = snap.get("power_w")
        elif metric == METRIC_BANDWIDTH_MB_S:
            v = snap.get("bandwidth_mb_s")
        else:
            return None
        return MetricReading(float(v), self.name) if v is not None else None


# ── sysfs ────────────────────────────────────────────────────────────────────

# Origin strings that mean "no usable reading" and should map to None at the
# protocol boundary. Kept as constants so they stay in sync with the legacy
# values previously produced by the inline ``_sysfs_*`` helpers.
_SYSFS_UNAVAILABLE = "placeholder_sysfs_unavailable"
_SYSFS_PARTIAL = "sysfs_partial"


class SysfsBackfillSource:
    """``MetricSource`` adapter that reads from DRM / hwmon / thermal sysfs.

    The helpers here are stateless and were previously static methods on
    ``QmassaCollector``. They are exposed publicly so the qmmd-offline
    fallback path can keep calling them directly with the same names.
    """

    name = "sysfs"

    def __init__(self, card_resolver: Callable[[str], Optional[Path]]) -> None:
        # The collector owns DRM card discovery; we ask it to resolve PCI
        # → ``Path`` on demand so this source stays free of side effects.
        self._card_resolver = card_resolver

    def refresh(self) -> None:
        return

    def read(self, pci: str, metric: str, *, card: Optional[Path] = None) -> Optional[MetricReading]:
        card_path = card if card is not None else self._card_resolver(pci)
        if card_path is None:
            return None
        if metric == METRIC_UTILIZATION:
            value, origin = self.utilization(card_path)
        elif metric == METRIC_FREQUENCY_MHZ:
            value, origin = self.frequency_mhz(card_path)
        elif metric == METRIC_POWER_W:
            value, origin = self.power_w(card_path)
        elif metric == METRIC_TEMPERATURE_C:
            value, origin = self.temperature_c(card_path)
        elif metric == METRIC_BANDWIDTH_MB_S:
            value, origin = self.bandwidth_mb_s(card_path)
        elif metric == METRIC_MEMORY_UTILIZATION:
            mem_pct, mem_origin, mem_domain, mem_available = (
                self.memory_utilization_from_sysfs(card_path)
            )
            if mem_available != "true":
                return None
            return MetricReading(float(mem_pct), mem_origin, domain=mem_domain)
        else:
            return None
        if value is None or origin in {_SYSFS_UNAVAILABLE, _SYSFS_PARTIAL}:
            return None
        return MetricReading(float(value), origin)

    # ── sysfs read helpers (moved verbatim from QmassaCollector) ─────────

    @staticmethod
    def utilization(card: Path) -> Tuple[Optional[float], str]:
        candidates = [
            card / "device" / "gpu_busy_percent",
            card / "gt" / "gt0" / "busy",
        ]
        for p in candidates:
            try:
                v = float(p.read_text().strip())
                return max(0.0, min(100.0, v)), "sysfs"
            except (OSError, ValueError):
                pass
        return None, _SYSFS_UNAVAILABLE

    @staticmethod
    def frequency_mhz(card: Path) -> Tuple[Optional[float], str]:
        candidates = [
            card / "gt" / "gt0" / "rps_cur_freq_mhz",
            card / "gt_cur_freq_mhz",
            card / "gt" / "gt0" / "rps_act_freq_mhz",
        ]
        for p in candidates:
            try:
                v = float(p.read_text().strip())
                if v > 0:
                    return v, "sysfs"
            except (OSError, ValueError):
                pass
        # xe driver: tile0/gt0
        tile_freq = card / "device" / "tile0" / "gt0" / "freq0" / "cur_freq"
        try:
            v = float(tile_freq.read_text().strip())
            if v > 0:
                return v, "sysfs_xe"
        except (OSError, ValueError):
            pass
        # Idle / RC6 powergated: report the configured minimum so graphs
        # show the floor instead of a flat zero.
        floor_candidates = [
            card / "gt" / "gt0" / "rps_min_freq_mhz",
            card / "gt_min_freq_mhz",
            card / "gt" / "gt0" / "rps_RPn_freq_mhz",
            card / "gt_RPn_freq_mhz",
            card / "device" / "tile0" / "gt0" / "freq0" / "min_freq",
        ]
        for p in floor_candidates:
            try:
                v = float(p.read_text().strip())
                if v > 0:
                    return v, "sysfs_idle_floor"
            except (OSError, ValueError):
                pass
        return None, _SYSFS_UNAVAILABLE

    @staticmethod
    def power_w(card: Path) -> Tuple[Optional[float], str]:
        hwmon_base = card / "device" / "hwmon"
        if hwmon_base.exists():
            patterns = ["power1_average", "power1_input", "power2_average", "power2_input"]
            for name in patterns:
                for pf in hwmon_base.glob(f"hwmon*/{name}"):
                    try:
                        return max(0.0, float(pf.read_text().strip()) / 1_000_000.0), "sysfs"
                    except (OSError, ValueError):
                        pass
        return None, _SYSFS_UNAVAILABLE

    @staticmethod
    def temperature_c(card: Path) -> Tuple[Optional[float], str]:
        hwmon_base = card / "device" / "hwmon"
        if not hwmon_base.exists():
            return None, _SYSFS_UNAVAILABLE

        for temp_file in sorted(hwmon_base.glob("hwmon*/temp*_input")):
            try:
                raw = float(temp_file.read_text().strip())
                return max(0.0, raw / 1000.0), "sysfs"
            except (OSError, ValueError):
                continue

        return None, _SYSFS_UNAVAILABLE

    @staticmethod
    def bandwidth_mb_s(card: Path) -> Tuple[Optional[float], str]:
        # Current Intel DRM sysfs does not expose a stable per-GPU memory
        # bandwidth counter across i915/xe.
        _ = card
        return None, _SYSFS_PARTIAL

    @staticmethod
    def cpu_package_temp_proxy_c() -> Tuple[Optional[float], str]:
        """Fallback temperature proxy for integrated GPUs on platforms without iGPU temp nodes."""
        thermal_root = Path("/sys/class/thermal")
        if not thermal_root.exists():
            return None, _SYSFS_UNAVAILABLE

        for zone in sorted(thermal_root.glob("thermal_zone*")):
            type_path = zone / "type"
            temp_path = zone / "temp"
            if not temp_path.exists():
                continue
            try:
                zone_type = type_path.read_text(encoding="utf-8").strip().lower()
            except OSError:
                continue

            if not any(token in zone_type for token in ("x86_pkg_temp", "package", "cpu", "soc")):
                continue

            try:
                raw = float(temp_path.read_text(encoding="utf-8").strip())
                return max(0.0, raw / 1000.0), "sysfs_cpu_pkg_proxy"
            except (OSError, ValueError):
                continue

        return None, _SYSFS_UNAVAILABLE

    @staticmethod
    def card_pci_address(card: Path) -> str:
        try:
            return (card / "device").resolve().name
        except OSError:
            return ""

    @staticmethod
    def is_integrated(card: Path) -> bool:
        """Return True if the card appears to be an integrated GPU."""
        import os
        import re

        pci = SysfsBackfillSource.card_pci_address(card)
        # On Intel client platforms, integrated graphics is typically 00:02.x.
        if pci.startswith("0000:00:02."):
            return True

        # Dedicated Intel GPUs usually expose VRAM capacity in sysfs.
        vram_total = card / "device" / "mem_info_vram_total"
        try:
            if int(vram_total.read_text().strip()) > 0:
                return False
        except (OSError, ValueError):
            pass

        # Discrete GPUs are commonly behind one or more PCI bridges.
        hop_count = 0
        try:
            bdf_hops = re.findall(r"/0000:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", str((card / "device").resolve()))
            hop_count = len(bdf_hops)
        except OSError:
            pass

        boot_vga = card / "device" / "boot_vga"
        try:
            if boot_vga.read_text().strip() == "1":
                return hop_count <= 1
        except OSError:
            pass

        try:
            driver = Path(os.readlink(card / "device" / "driver")).name
            if driver in {"i915", "xe"}:
                return hop_count <= 1
        except OSError:
            pass

        return False

    @staticmethod
    def memory_utilization_from_sysfs(card: Path) -> Tuple[float, str, str, str]:
        """Best-effort per-device memory utilization from DRM sysfs counters."""
        candidates = (
            ("vram", "mem_info_vram_used", "mem_info_vram_total"),
            ("gtt", "mem_info_gtt_used", "mem_info_gtt_total"),
            ("system", "mem_info_system_used", "mem_info_system_total"),
        )

        for mem_domain, used_name, total_name in candidates:
            used_path = card / "device" / used_name
            total_path = card / "device" / total_name
            try:
                used = float(used_path.read_text().strip())
                total = float(total_path.read_text().strip())
            except (OSError, ValueError):
                continue

            if total <= 0.0:
                continue

            pct = max(0.0, min(100.0, (used / total) * 100.0))
            return round(pct, 2), "sysfs", mem_domain, "true"

        return 0.0, "qmmd_partial", "unavailable", "false"
