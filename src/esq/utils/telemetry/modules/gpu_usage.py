# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
GPU utilization telemetry module (ESQ-specific).

Provides per-engine utilization for xe (Intel Arc) GPUs and GT-level
utilization for i915 (Intel integrated / legacy) GPUs in a single module.

xe GPUs — Intel Arc and Data Center GPUs (Linux 6.8+, xe driver)
-----------------------------------------------------------------
Uses Linux perf PMU ``engine-active-ticks`` / ``engine-total-ticks`` events
exposed by the xe kernel driver.  Reports one percentage per engine class:

    ``gpu_{idx}_render_pct``    — Render/3D engine (RCS)
    ``gpu_{idx}_compute_pct``   — Compute engine (CCS)
    ``gpu_{idx}_copy_pct``      — Blitter/Copy engine (BCS)
    ``gpu_{idx}_video_pct``     — Video codec engine (VCS)
    ``gpu_{idx}_video_enh_pct`` — Video Enhancement engine (VECS)

Multi-GT xe GPUs (e.g. Arc Pro B60) split engine classes across GTs (gt0 hosts
rcs/ccs/bcs, gt1 hosts vcs/vecs).  This is handled transparently — the metric
name encodes the engine function, not the GT index.

Requires ``perf_event_paranoid <= 0`` (set by ``system-setup.sh``) or
``CAP_PERFMON``.  If PMU access is unavailable, xe GPUs fall back to GT-level
RC6/C6 residency (same as i915).

i915 GPUs — Intel integrated / older discrete GPUs
---------------------------------------------------
Uses GT-level C6/RC6 idle-residency counters from sysfs.  Reports one
percentage per GT:

    ``gpu_{idx}_gt0_pct``  — GT 0 utilization
    ``gpu_{idx}_gt1_pct``  — GT 1 utilization (if present)

Formula (i915 / xe PMU-fallback)::

    usage_pct = (1 - delta_idle_ms / delta_wall_ms) * 100

Labels
------
xe engines use function names matching ``intel_gpu_top`` / ``qmassa``:
``Render/3D``, ``Compute``, ``Copy``, ``Video``, ``VideoEnh``.

i915 GTs use the GPU display name from the label map.  For a single-GT iGPU
only the GPU name is shown (e.g. ``"iGPU"``); multi-GT devices append the GT
index (e.g. ``"iGPU GT0"``).
"""

import ctypes
import glob
import logging
import os
import re
import struct
import time

from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetryConfig, TelemetrySample

from esq.utils.telemetry.modules._drm import _get_pci_slot, find_intel_gpu_drm_cards

logger = logging.getLogger(__name__)

MODULE_NAME = "gpu_usage"

# ---------------------------------------------------------------------------
# Linux perf_event_open constants (xe PMU engine ticks)
# ---------------------------------------------------------------------------

_NR_PERF_EVENT_OPEN = 298  # x86-64 syscall number
_PERF_EVENT_ATTR_SIZE = 128
_PERF_FORMAT_TOTAL_TIME_ENABLED = 1 << 1
_PERF_FORMAT_TOTAL_TIME_RUNNING = 1 << 2

_EVENT_ACTIVE_TICKS = 0x02
_EVENT_TOTAL_TICKS = 0x03

# Engine class codes (xe driver enum drm_xe_engine_class)
_ENGINE_CLASSES = {
    "rcs": 0,  # Render/3D
    "vcs": 1,  # Video Codec
    "bcs": 2,  # Blitter/Copy
    "vecs": 3,  # Video Enhancement
    "ccs": 4,  # Compute
}

# Metric name suffix — matches intel_gpu_top / qmassa / DRM convention
_ENGINE_METRIC_NAMES = {
    "rcs": "render",
    "vcs": "video",
    "bcs": "copy",
    "vecs": "video_enh",
    "ccs": "compute",
}

# Chart axis labels — mirror intel_gpu_top display strings
_ENGINE_LABELS = {
    "rcs": "Render/3D",
    "vcs": "Video",
    "bcs": "Copy",
    "vecs": "VideoEnh",
    "ccs": "Compute",
}


class _PerfEventAttr(ctypes.Structure):
    """Minimal perf_event_attr layout sufficient for device PMU counters."""

    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("bp_addr", ctypes.c_uint64),
        ("bp_len", ctypes.c_uint64),
        ("_pad", ctypes.c_uint8 * (_PERF_EVENT_ATTR_SIZE - 56)),
    ]


_libc = ctypes.CDLL(None, use_errno=True)


# ---------------------------------------------------------------------------
# PMU helpers
# ---------------------------------------------------------------------------


def _make_config(event: int, engine_class: int, engine_instance: int = 0, gt: int = 0) -> int:
    """
    Encode a xe PMU config value.

    Bit layout (from sysfs format/ directory):
        [0:11]   event
        [12:19]  engine_instance
        [20:27]  engine_class
        [60:63]  gt
    """
    return (event & 0xFFF) | (engine_instance << 12) | (engine_class << 20) | (gt << 60)


def _open_perf_fd(pmu_type: int, config: int) -> int:
    """Open a system-wide (pid=-1, cpu=0) device PMU counter fd."""
    attr = _PerfEventAttr()
    attr.type = pmu_type
    attr.size = _PERF_EVENT_ATTR_SIZE
    attr.config = config
    attr.read_format = _PERF_FORMAT_TOTAL_TIME_ENABLED | _PERF_FORMAT_TOTAL_TIME_RUNNING
    return _libc.syscall(_NR_PERF_EVENT_OPEN, ctypes.byref(attr), -1, 0, -1, 0)


def _read_perf_fd(fd: int):
    """Read the 64-bit counter value from a perf fd; returns None on error."""
    try:
        data = os.read(fd, 24)
        val, _enabled, _running = struct.unpack("QQQ", data)
        return val
    except OSError:
        return None


def _discover_xe_pmu_types() -> dict:
    """
    Return a mapping of PCI slot → xe PMU type for each xe GPU.

    Example: ``{"0000:04:00.0": 44, "0000:08:00.0": 45}``
    """
    result = {}
    for type_path in sorted(glob.glob("/sys/bus/event_source/devices/xe_*/type")):
        pmu_dir = os.path.basename(os.path.dirname(type_path))
        # xe_0000_04_00.0  →  "0000:04:00.0"
        m = re.match(r"^xe_([0-9a-f]{4})_([0-9a-f]{2})_([0-9a-f]{2})\.(\d+)$", pmu_dir)
        if not m:
            continue
        pci_slot = f"{m.group(1)}:{m.group(2)}:{m.group(3)}.{m.group(4)}"
        try:
            pmu_type = int(open(type_path).read().strip())
        except (OSError, ValueError):
            continue
        result[pci_slot] = pmu_type
    return result


def _discover_xe_gts(card_path: str) -> list:
    """Return sorted list of GT indices present under an xe DRM card's device."""
    device_path = os.path.realpath(os.path.join(card_path, "device"))
    gt_nums = set()
    for gt_dir in glob.glob(os.path.join(device_path, "tile*", "gt*")):
        if not os.path.isdir(gt_dir):
            continue
        m = re.match(r"^gt(\d+)$", os.path.basename(gt_dir))
        if m:
            gt_nums.add(int(m.group(1)))
    return sorted(gt_nums)


# ---------------------------------------------------------------------------
# RC6/C6 residency helpers (i915 / xe fallback)
# ---------------------------------------------------------------------------


def _read_ms(path):
    """Read an integer millisecond counter from a sysfs file, or return None."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _discover_idle_paths(skip_gpu_indices: set) -> list:
    """
    Discover GT idle-residency sysfs paths for GPUs not already covered by PMU.

    Returns list of ``(metric_name, sysfs_path)`` pairs, e.g.::

        [("gpu_0_gt0_pct", "/sys/class/drm/card0/gt/gt0/rc6_residency_ms")]
    """
    result = []
    for gpu_idx, (card_path, driver) in enumerate(find_intel_gpu_drm_cards()):
        if gpu_idx in skip_gpu_indices:
            continue
        if driver == "i915":
            for gt_dir in sorted(glob.glob(os.path.join(card_path, "gt", "gt*"))):
                m = re.match(r"^gt(\d+)$", os.path.basename(gt_dir))
                if not m:
                    continue
                gt_num = int(m.group(1))
                rc6_path = os.path.join(gt_dir, "rc6_residency_ms")
                if os.path.exists(rc6_path):
                    result.append((f"gpu_{gpu_idx}_gt{gt_num}_pct", rc6_path))
        elif driver == "xe":
            device_path = os.path.realpath(os.path.join(card_path, "device"))
            for gt_dir in sorted(glob.glob(os.path.join(device_path, "tile*", "gt*"))):
                if not os.path.isdir(gt_dir):
                    continue
                m = re.match(r"^gt(\d+)$", os.path.basename(gt_dir))
                if not m:
                    continue
                gt_num = int(m.group(1))
                idle_path = os.path.join(gt_dir, "gtidle", "idle_residency_ms")
                if os.path.exists(idle_path):
                    result.append((f"gpu_{gpu_idx}_gt{gt_num}_pct", idle_path))
    return result


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class GpuUsageModule(BaseTelemetryModule):
    """
    Combined Intel GPU utilization module.

    xe GPUs: per-engine PMU percentages (render, compute, copy, video, video_enh).
    i915 GPUs (and xe GPUs without PMU access): GT-level RC6/C6 residency percentages.
    """

    module_name = MODULE_NAME

    def __init__(self, config: TelemetryConfig) -> None:
        super().__init__(config)
        cards = find_intel_gpu_drm_cards()
        self._gpu_pci_slots: dict = {gpu_idx: _get_pci_slot(card_path) for gpu_idx, (card_path, _) in enumerate(cards)}

        # --- xe PMU engine fds (per engine class) ---
        # metric → {"act_fd": int, "tot_fd": int}
        self._fds: dict = {}
        # metric → (act_ticks, tot_ticks) at last sample
        self._pmu_prev: dict = {}
        # gpu indices fully covered by PMU (excluded from RC6 fallback)
        self._xe_pmu_indices: set = set()

        xe_pmu_types = _discover_xe_pmu_types()
        for gpu_idx, (card_path, driver) in enumerate(cards):
            if driver != "xe":
                continue
            pci_slot = self._gpu_pci_slots.get(gpu_idx, "")
            pmu_type = xe_pmu_types.get(pci_slot)
            if pmu_type is None:
                continue

            gt_nums = _discover_xe_gts(card_path)
            opened_any = False
            for gt_num in gt_nums:
                for eng_name, eng_class in _ENGINE_CLASSES.items():
                    fd_act = _open_perf_fd(pmu_type, _make_config(_EVENT_ACTIVE_TICKS, eng_class, gt=gt_num))
                    if fd_act < 0:
                        continue
                    fd_tot = _open_perf_fd(pmu_type, _make_config(_EVENT_TOTAL_TICKS, eng_class, gt=gt_num))
                    if fd_tot < 0:
                        os.close(fd_act)
                        continue
                    metric = f"gpu_{gpu_idx}_{_ENGINE_METRIC_NAMES[eng_name]}_pct"
                    self._fds[metric] = {"act_fd": fd_act, "tot_fd": fd_tot}
                    opened_any = True
            if opened_any:
                self._xe_pmu_indices.add(gpu_idx)

        if not xe_pmu_types and any(d == "xe" for _, d in cards):
            try:
                paranoid = open("/proc/sys/kernel/perf_event_paranoid").read().strip()
            except OSError:
                paranoid = "unknown"
            logger.debug(
                "gpu_usage: no xe PMU entries found; xe GPUs will use RC6/C6 fallback. "
                "perf_event_paranoid=%s (requires <= 0 — run 'sudo bash system-setup.sh')",
                paranoid,
            )

        # --- RC6/C6 residency paths (i915 + xe GPUs without PMU access) ---
        self._rc6_paths = _discover_idle_paths(skip_gpu_indices=self._xe_pmu_indices)
        # metric → (idle_ms, wall_timestamp) at last sample
        self._rc6_prev: dict = {}

        # Prime baselines
        if self._fds:
            self._prime_pmu()
        if self._rc6_paths:
            self._prime_rc6()

    def _prime_pmu(self) -> None:
        for metric, fd_pair in self._fds.items():
            act = _read_perf_fd(fd_pair["act_fd"])
            tot = _read_perf_fd(fd_pair["tot_fd"])
            if act is not None and tot is not None:
                self._pmu_prev[metric] = (act, tot)

    def _prime_rc6(self) -> None:
        now = time.time()
        for metric, path in self._rc6_paths:
            val = _read_ms(path)
            if val is not None:
                self._rc6_prev[metric] = (val, now)
        # Sleep 150 ms so the collector's immediate t=0 sample satisfies the
        # 100 ms minimum-window guard in collect_sample().  Without this pause
        # the baseline timestamp is only a few milliseconds old when the first
        # collect fires, causing the iGPU RC6 metrics to be silently skipped.
        time.sleep(0.15)

    def is_available(self) -> bool:
        return bool(self._fds) or bool(self._rc6_paths)

    def get_default_config(self):
        """Build default scales based on detected GPU metrics."""
        from esq.utils.telemetry.modules._drm import get_gpu_label_map

        label_map = get_gpu_label_map()
        scales = {}

        # PMU-based metrics (xe engine-class percentages)
        _metric_to_eng = {v: k for k, v in _ENGINE_METRIC_NAMES.items()}
        for metric in sorted(self._fds):
            m = re.match(r"gpu_(\d+)_([a-z_]+)_pct", metric)
            if not m:
                continue
            gpu_idx = int(m.group(1))
            metric_suffix = m.group(2)
            eng_name = _metric_to_eng.get(metric_suffix, metric_suffix)
            pci_slot = self._gpu_pci_slots.get(gpu_idx, "")
            gpu_label = label_map.get(pci_slot, f"GPU {gpu_idx}")
            eng_label = _ENGINE_LABELS.get(eng_name, metric_suffix.capitalize())
            scales[metric] = {
                "display": True,
                "label": f"{gpu_label} {eng_label}",
                "unit": "%",
            }

        # RC6/C6 residency metrics (i915 / xe fallback — GT-level)
        gt_counts: dict = {}
        for metric, _ in self._rc6_paths:
            m = re.match(r"gpu_(\d+)_gt\d+_pct", metric)
            if m:
                idx = int(m.group(1))
                gt_counts[idx] = gt_counts.get(idx, 0) + 1

        for metric, _ in self._rc6_paths:
            m = re.match(r"gpu_(\d+)_gt(\d+)_pct", metric)
            if not m:
                continue
            gpu_idx = int(m.group(1))
            gt_num = int(m.group(2))
            pci_slot = self._gpu_pci_slots.get(gpu_idx, "")
            gpu_label = label_map.get(pci_slot, f"GPU {gpu_idx}")
            if gt_counts.get(gpu_idx, 1) > 1:
                label = f"{gpu_label} GT{gt_num}"
            else:
                label = gpu_label
            scales[metric] = {"display": True, "label": label, "unit": "%"}

        return {
            "chart_type": "area",
            "title": {"display": True, "text": "GPU Usage"},
            "scales": scales,
        }

    def collect_sample(self) -> TelemetrySample:
        now = time.time()
        raw: dict = {}

        # PMU engine-tick percentages (xe GPUs)
        for metric, fd_pair in self._fds.items():
            act = _read_perf_fd(fd_pair["act_fd"])
            tot = _read_perf_fd(fd_pair["tot_fd"])
            if act is None or tot is None:
                continue
            if metric in self._pmu_prev:
                prev_act, prev_tot = self._pmu_prev[metric]
                delta_act = act - prev_act
                delta_tot = tot - prev_tot
                if delta_tot > 0:
                    pct = round(max(0.0, min(100.0, delta_act / delta_tot * 100.0)), 1)
                    raw[metric] = pct
            self._pmu_prev[metric] = (act, tot)

        # RC6/C6 residency percentages (i915 + xe fallback)
        for metric, path in self._rc6_paths:
            rc6_ms = _read_ms(path)
            if rc6_ms is None:
                continue
            if metric in self._rc6_prev:
                prev_rc6, prev_time = self._rc6_prev[metric]
                dt_ms = (now - prev_time) * 1000.0
                if dt_ms >= 100:
                    delta_rc6 = rc6_ms - prev_rc6
                    rc6_fraction = max(0.0, min(1.0, delta_rc6 / dt_ms))
                    raw[metric] = round((1.0 - rc6_fraction) * 100.0, 1)
            self._rc6_prev[metric] = (rc6_ms, now)

        values = self._filter_values(raw)
        sample = TelemetrySample(timestamp=now, values=values)
        self.check_thresholds(values)
        return sample

    def __del__(self):
        """Close all open perf file descriptors."""
        for fd_pair in getattr(self, "_fds", {}).values():
            for fd in fd_pair.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
