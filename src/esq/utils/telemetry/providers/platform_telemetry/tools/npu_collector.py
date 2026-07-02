# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import List, Optional

from esq.utils.telemetry.providers.platform_telemetry.base import BaseCollector
from esq.utils.telemetry.providers.platform_telemetry.models import MetricSample

log = logging.getLogger(__name__)

# Sentinel emitted when a metric source cannot be read. -1 keeps missing
# data visible in JSON and distinct from a real 0 (e.g. idle NPU at 0%).
# Aggregation layers filter the sentinel out.
MISSING_VALUE: float = -1.0

# ── PMT GUID constants (one per platform generation) ─────────────────────────
_PMT_GUID_MTL   = "0x130670b2"   # Meteor Lake
_PMT_GUID_ARL   = "0x1306a0b3"   # Arrow Lake
_PMT_GUID_ARL_H = "0x1306a0b2"   # Arrow Lake-H
_PMT_GUID_ARL_S = "0x1306a0b4"   # Arrow Lake-S
_PMT_GUID_LNL   = "0x3072005"    # Lunar Lake
_PMT_GUID_PTL   = "0x3086000"    # Panther Lake

# Telemetry register offsets per platform
_REGS_MTL_ARL = {
    "VPU_ENERGY":    0x628,
    "SOC_TEMPS":     0x98,
    "VPU_WORKPOINT": 0x68,
}
_REGS_LNL = {
    "VPU_ENERGY":    0x5D0,
    "SOC_TEMPS":     0x70,
    "VPU_WORKPOINT": 0x18,
    "VPU_MEM_BW":    0xC18,
}
_REGS_PTL = {
    "VPU_ENERGY":    0x670,
    "SOC_TEMPS":     0x78,
    "VPU_WORKPOINT": 0x18,
    "VPU_MEM_BW":    0xC18,
}

_PMT_ROOT   = "/sys/class/intel_pmt"
_VPU_ROOT   = "/sys/bus/pci/drivers/intel_vpu"
_KB         = 1024


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sysfs_read(path: str) -> Optional[str]:
    # PMT sysfs nodes are made readable by the ESQ pre-requisites script
    # (system-setup.sh installs a udev rule that grants the runtime user
    # access). The collector itself runs unprivileged — if the read still
    # fails we surface the error rather than silently re-elevating.
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _pmt_read_u64_bits(buf: bytes, offset: int, msb: int, lsb: int) -> int:
    """Extract bit-field [msb:lsb] from an 8-byte little-endian word at *offset*."""
    raw = int.from_bytes(buf[offset: offset + 8], byteorder="little")
    msb_mask = 0xFFFF_FFFF_FFFF_FFFF & ((2 ** (msb + 1)) - 1)
    lsb_mask = 0xFFFF_FFFF_FFFF_FFFF & ((2 ** lsb) - 1)
    return (raw & (msb_mask & ~lsb_mask)) >> lsb


# ── Platform state ────────────────────────────────────────────────────────────

class _PmtState:
    """Holds one-time platform detection results and the live PMT buffer."""

    __slots__ = ("telem_path", "regs", "platform", "buf", "mem_util_supported")

    def __init__(self, telem_path: str, regs: dict, platform: str, mem_util_supported: bool):
        self.telem_path = telem_path
        self.regs = regs
        self.platform = platform
        self.buf: Optional[bytes] = None
        self.mem_util_supported = mem_util_supported

    def refresh(self) -> bool:
        """Re-read the PMT telemetry binary blob from sysfs."""
        try:
            with open(self.telem_path, "rb") as fh:
                self.buf = fh.read()
            return True
        except OSError as exc:
            log.warning("NpuCollector: PMT read failed: %s", exc)
            self.buf = None
            return False

    def read(self, reg_key: str, msb: int, lsb: int) -> Optional[int]:
        if self.buf is None:
            return None
        offset = self.regs.get(reg_key)
        if offset is None:
            return None
        return _pmt_read_u64_bits(self.buf, offset, msb, lsb)


# ── Collector ─────────────────────────────────────────────────────────────────

class NpuCollector(BaseCollector):
    """Intel NPU metrics collector using PMT registers and sysfs interfaces.

    Supports Meteor Lake (MTL), Arrow Lake (ARL), Lunar Lake (LNL), and
    Panther Lake (PTL) platforms.  The collector probes for the intel_vpu
    kernel driver and the Intel PMT sysfs tree on start().  If either is
    absent the collector silently returns empty samples, allowing the
    dashboard to display the NPU placeholder row rather than erroring.

    Delta-based metrics (utilization, power) require at least
    two consecutive collect_once() calls to produce non-zero values; the
    first call always initialises the baseline and returns sentinel values.
    """

    name = "npu"

    def __init__(self) -> None:
        self._running = False
        self._available = False
        self._pmt: Optional[_PmtState] = None
        self._dev_path: Optional[str] = None           # intel_vpu PCI device dir
        self._busy_time_path: Optional[str] = None
        self._mem_util_path: Optional[str] = None

        # Delta state (updated on every collect_once call)
        self._prev_busy_us: Optional[int] = None
        self._prev_energy_j: Optional[float] = None
        self._prev_ts: Optional[float] = None

    # ── BaseCollector interface ───────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._available = self._probe()
        if self._available:
            log.info("NpuCollector: NPU available on platform=%s", self._pmt.platform)  # type: ignore[union-attr]
        else:
            log.info("NpuCollector: NPU not available on this host (intel_vpu driver or PMT missing)")

    def stop(self) -> None:
        self._running = False

    def collect_once(self) -> List[MetricSample]:
        if not self._running or not self._available or self._pmt is None:
            return []

        if not self._pmt.refresh():
            return []

        now_ts = time.monotonic()
        now_utc = datetime.utcnow().isoformat() + "Z"
        samples: List[MetricSample] = []

        # ── Utilization (delta of npu_busy_time_us) ───────────────────────────
        utilization = self._compute_utilization(now_ts)

        # ── Power (delta of PMT energy counter, J → W) ────────────────────────
        power_w = self._compute_power_w(now_ts)

        # ── Frequency (PMT VPU_WORKPOINT register) ────────────────────────────
        freq_mhz = self._read_freq_mhz()

        # ── Temperature (PMT SOC_TEMPERATURES register) ───────────────────────
        temperature_c = self._read_temperature_c()

        # ── Memory utilisation (sysfs, PTL+ only) ─────────────────────────────
        mem_util_mb = self._read_memory_util_mb()

        # Update monotonic timestamp baseline
        self._prev_ts = now_ts

        # ── Emit MetricSamples ────────────────────────────────────────────────
        # Emit a MetricSample for every metric; unavailable readings emit
        # MISSING_VALUE so aggregations can filter them out.
        tags = {"vendor": "Intel", "metric_origin": "npu-pmt-sysfs", "platform": self._pmt.platform}

        samples.append(MetricSample(
            timestamp_utc=now_utc,
            collector=self.name,
            device="NPU",
            metric_name="npu.utilization",
            value=float(utilization) if utilization is not None else MISSING_VALUE,
            unit="%",
            tags=tags,
        ))

        samples.append(MetricSample(
            timestamp_utc=now_utc,
            collector=self.name,
            device="NPU",
            metric_name="npu.power_w",
            value=float(power_w) if power_w is not None else MISSING_VALUE,
            unit="W",
            tags=tags,
        ))

        samples.append(MetricSample(
            timestamp_utc=now_utc,
            collector=self.name,
            device="NPU",
            metric_name="npu.frequency_mhz",
            value=float(freq_mhz) if freq_mhz is not None else MISSING_VALUE,
            unit="MHz",
            tags=tags,
        ))

        samples.append(MetricSample(
            timestamp_utc=now_utc,
            collector=self.name,
            device="NPU",
            metric_name="npu.temperature_c",
            value=float(temperature_c) if temperature_c is not None else MISSING_VALUE,
            unit="°C",
            tags=tags,
        ))

        # Kernel ivpu exposes ``npu_memory_utilization`` as bytes currently
        # allocated (size, not a percentage). Publish under the size-explicit
        # canonical name ``npu.memory_used_mb`` so the chart/CSV label cannot
        # be confused with a 0-100 % reading. A true utilisation % would
        # require the NPU plugin total memory size
        # (OpenVINO NPU_DEVICE_TOTAL_MEM_SIZE), which is not available at
        # this layer.
        samples.append(MetricSample(
            timestamp_utc=now_utc,
            collector=self.name,
            device="NPU",
            metric_name="npu.memory_used_mb",
            value=float(mem_util_mb) if (mem_util_mb is not None and mem_util_mb >= 0) else MISSING_VALUE,
            unit="MB",
            tags=tags,
        ))

        return samples

    # ── Platform probe ────────────────────────────────────────────────────────

    def _probe(self) -> bool:
        """Detect intel_vpu device and matching PMT telemetry interface."""
        if not os.path.isdir(_VPU_ROOT):
            log.debug("NpuCollector: %s not found (intel_vpu driver not loaded)", _VPU_ROOT)
            return False

        # Find PCI slot entry under intel_vpu directory
        dev_path: Optional[str] = None
        try:
            for entry in os.listdir(_VPU_ROOT):
                if entry.startswith("0000:"):
                    dev_path = os.path.join(_VPU_ROOT, entry)
                    break
        except OSError as exc:
            log.debug("NpuCollector: Cannot list %s: %s", _VPU_ROOT, exc)
            return False

        if dev_path is None:
            log.debug("NpuCollector: No PCI device found under %s", _VPU_ROOT)
            return False

        self._dev_path = dev_path

        # Busy-time sysfs node
        busy_path = os.path.join(dev_path, "npu_busy_time_us")
        self._busy_time_path = busy_path if os.path.exists(busy_path) else None

        # Memory utilisation sysfs node (PTL+ only, detected after PMT GUID)
        mem_path = os.path.join(dev_path, "npu_memory_utilization")
        self._mem_util_path = mem_path if os.path.exists(mem_path) else None

        # Locate matching PMT telemetry interface
        pmt = self._find_pmt()
        if pmt is None:
            log.debug("NpuCollector: No supported PMT GUID found under %s", _PMT_ROOT)
            return False

        self._pmt = pmt

        if not os.access(pmt.telem_path, os.R_OK):
            log.warning(
                "NpuCollector: PMT telemetry path %s is not readable; "
                "run the ESQ pre-requisites script (system-setup.sh) to grant access",
                pmt.telem_path,
            )

        # Bootstrap delta baselines with first read
        if pmt.refresh():
            self._prev_busy_us = self._read_busy_time_us()
            self._prev_energy_j = self._read_energy_j()
            self._prev_ts = time.monotonic()

        return True

    @staticmethod
    def _find_pmt() -> Optional[_PmtState]:
        """Scan /sys/class/intel_pmt for a telem entry with a known NPU GUID."""
        if not os.path.isdir(_PMT_ROOT):
            return None

        try:
            entries = os.listdir(_PMT_ROOT)
        except OSError:
            return None

        for telem_dir in sorted(entries):
            if not telem_dir.startswith("telem"):
                continue

            telem_path = os.path.join(_PMT_ROOT, telem_dir)
            guid_path    = os.path.join(telem_path, "guid")
            telem_bin    = os.path.join(telem_path, "telem")
            size_path    = os.path.join(telem_path, "size")
            offset_path  = os.path.join(telem_path, "offset")

            if not all(os.path.exists(p) for p in [guid_path, telem_bin, size_path, offset_path]):
                continue

            guid = _sysfs_read(guid_path)
            if guid is None:
                continue

            if guid == _PMT_GUID_MTL:
                return _PmtState(telem_bin, _REGS_MTL_ARL, "MTL", mem_util_supported=False)
            if guid in (_PMT_GUID_ARL, _PMT_GUID_ARL_H, _PMT_GUID_ARL_S):
                return _PmtState(telem_bin, _REGS_MTL_ARL, "ARL", mem_util_supported=False)
            if guid == _PMT_GUID_LNL:
                return _PmtState(telem_bin, _REGS_LNL, "LNL", mem_util_supported=False)
            if guid == _PMT_GUID_PTL:
                return _PmtState(telem_bin, _REGS_PTL, "PTL", mem_util_supported=True)

        return None

    # ── Raw sysfs / PMT reads ─────────────────────────────────────────────────

    def _read_busy_time_us(self) -> Optional[int]:
        if self._busy_time_path is None:
            return None
        raw = _sysfs_read(self._busy_time_path)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _read_energy_j(self) -> Optional[float]:
        if self._pmt is None or self._pmt.buf is None:
            return None
        val = self._pmt.read("VPU_ENERGY", 63, 0)
        if val is None:
            return None
        # U32.18.14 fixed-point → float joules
        return (val >> 14) + ((val & ((1 << 14) - 1)) / (1 << 14))

    def _read_freq_mhz(self) -> Optional[float]:
        if self._pmt is None or self._pmt.buf is None:
            return None
        raw = self._pmt.read("VPU_WORKPOINT", 7, 0)
        if raw is None:
            return None
        if self._pmt.platform == "MTL":
            return 2 * raw / 3 / 10
        return 0.05 * raw

    def _read_temperature_c(self) -> Optional[int]:
        if self._pmt is None or self._pmt.buf is None:
            return None
        return self._pmt.read("SOC_TEMPS", 47, 40)

    def _read_memory_util_mb(self) -> Optional[float]:
        # Memory utilisation is exposed via sysfs when the driver/platform supports it.
        # Use the presence of the sysfs node as the source of truth rather than a
        # static platform allowlist so newer ARL builds exposing the node are read.
        if self._mem_util_path is None or not os.path.exists(self._mem_util_path):
            return None
        raw = _sysfs_read(self._mem_util_path)
        if raw is None:
            return None
        try:
            return float(int(raw)) / _KB / _KB  # bytes → MB
        except ValueError:
            return None

    # ── Delta computations ────────────────────────────────────────────────────

    def _compute_utilization(self, now_ts: float) -> Optional[float]:
        curr = self._read_busy_time_us()
        if curr is None or self._prev_busy_us is None or self._prev_ts is None:
            self._prev_busy_us = curr
            return None
        elapsed_us = (now_ts - self._prev_ts) * 1_000_000
        if elapsed_us <= 0:
            return None
        utilization = min(100.0, 100.0 * (curr - self._prev_busy_us) / elapsed_us)
        self._prev_busy_us = curr
        return max(0.0, utilization)

    def _compute_power_w(self, now_ts: float) -> Optional[float]:
        curr = self._read_energy_j()
        if curr is None or self._prev_energy_j is None or self._prev_ts is None:
            self._prev_energy_j = curr
            return None
        elapsed_s = now_ts - self._prev_ts
        if elapsed_s <= 0:
            return None
        power_w = (curr - self._prev_energy_j) / elapsed_s
        self._prev_energy_j = curr
        return max(0.0, power_w)
