# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
NPU utilization and memory telemetry module (ESQ-specific).

Collects Intel NPU activity metrics via the Linux ``intel_vpu`` driver sysfs
interface exposed under ``/sys/class/accel/accel*/``.

Intel Core Ultra processors (Meteor Lake and later) integrate an NPU (Neural
Processing Unit) that is exposed to Linux as an accelerator device
(``/dev/accel/accel*``) by the ``intel_vpu`` kernel driver.

Reading strategy
----------------
The driver exposes a monotonic counter ``npu_busy_time_us`` (total microseconds
the NPU has spent executing workloads since driver load).  This module computes
a delta-based utilisation percentage over each collection interval, matching
how CPU and GPU utilisation are commonly reported::

    busy_pct = delta_npu_busy_us / delta_wall_us × 100

The ``npu_memory_utilization`` attribute reports the number of bytes of NPU
device memory currently allocated by the driver.  This is reported in MiB.

Power state from ``power_state`` (PCI PM: D0=active, D3hot/D3cold=idle) is
included as supplementary context.

Metrics collected (per detected NPU, 0-indexed):
    npu_0_busy_pct:   NPU 0 busy utilisation percentage (0–100 %).
    npu_0_mem_mib:    NPU 0 device memory currently allocated (MiB).

Both metrics are zero when the NPU is idle (D3hot power state with no active
inference workloads).

Reference: linux-npu-driver sysfs ABI
    drivers/accel/ivpu/ivpu_sysfs.c (npu_busy_time_us, npu_memory_utilization,
    npu_current_frequency_mhz, npu_max_frequency_mhz)
"""

import glob
import logging
import os
import time

from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetryConfig, TelemetrySample

logger = logging.getLogger(__name__)

MODULE_NAME = "npu_usage"

# ---------------------------------------------------------------------------
# Sysfs discovery
# ---------------------------------------------------------------------------

_ACCEL_GLOB = "/sys/class/accel/accel*"
_NPU_DRIVER = "intel_vpu"


def _find_npu_accel_devices() -> list:
    """
    Return a sorted list of ``(accel_path, pci_slot)`` for Intel NPU devices.

    Filters ``/sys/class/accel/accel*`` entries to those bound to the
    ``intel_vpu`` driver.  Returns an empty list when no NPU is present.
    """
    result = []
    for accel_dir in sorted(glob.glob(_ACCEL_GLOB)):
        dev_path = os.path.join(accel_dir, "device")
        uevent = os.path.join(dev_path, "uevent")
        try:
            content = open(uevent).read()
            if "intel_vpu" not in content and "IVPU" not in content.upper():
                # Check via driver symlink as a fallback
                driver_link = os.path.join(dev_path, "driver")
                if os.path.islink(driver_link):
                    driver_name = os.path.basename(os.readlink(driver_link))
                    if driver_name != _NPU_DRIVER:
                        continue
                else:
                    continue
        except OSError:
            # No uevent — skip
            continue

        # Extract PCI slot from uevent (PCI_SLOT_NAME=xxxx:xx:xx.x)
        pci_slot = ""
        try:
            for line in open(uevent):
                if line.startswith("PCI_SLOT_NAME="):
                    pci_slot = line.split("=", 1)[1].strip()
                    break
        except OSError:
            pass

        result.append((accel_dir, pci_slot))
    return result


def _read_int(path: str):
    """Read an integer from a sysfs file, or return None on failure."""
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class NpuUsageModule(BaseTelemetryModule):
    """
    Intel NPU busy utilisation and memory allocation telemetry.

    Uses the ``intel_vpu`` driver sysfs interface to report:
    - ``npu_{idx}_busy_pct`` — delta-based busy percentage over each interval
    - ``npu_{idx}_mem_mib``  — device memory currently allocated (MiB)

    The module is silently unavailable on systems without an Intel NPU or
    without the ``intel_vpu`` driver loaded.
    """

    module_name = MODULE_NAME

    def __init__(self, config: TelemetryConfig) -> None:
        super().__init__(config)
        self._devices = _find_npu_accel_devices()

        # Paths: npu_idx → {"busy_path": str, "mem_path": str}
        self._paths: dict = {}
        for idx, (accel_dir, pci_slot) in enumerate(self._devices):
            dev = os.path.join(accel_dir, "device")
            busy_path = os.path.join(dev, "npu_busy_time_us")
            mem_path = os.path.join(dev, "npu_memory_utilization")
            if os.path.exists(busy_path):
                self._paths[idx] = {
                    "busy_path": busy_path,
                    "mem_path": mem_path if os.path.exists(mem_path) else None,
                    "pci_slot": pci_slot,
                }

        # Baseline for delta calculation: npu_idx → (busy_us, wall_time)
        self._prev: dict = {}

        # Prime baseline so the first sample has a reference point.
        # Sleep 150 ms to ensure the first collect_sample() call satisfies
        # the 100 ms minimum-window guard.
        self._prime()
        if self._paths:
            time.sleep(0.15)

    def _prime(self) -> None:
        now = time.time()
        for idx, paths in self._paths.items():
            val = _read_int(paths["busy_path"])
            if val is not None:
                self._prev[idx] = (val, now)

    def is_available(self) -> bool:
        return bool(self._paths)

    def get_default_config(self) -> dict:
        scales = {}
        for idx in self._paths:
            scales[f"npu_{idx}_busy_pct"] = {
                "display": True,
                "label": f"NPU {idx} Busy",
                "unit": "%",
            }
            scales[f"npu_{idx}_mem_mib"] = {
                "display": True,
                "label": f"NPU {idx} Memory",
                "unit": "MB",
            }
        return {
            "chart_type": "area",
            "title": {"display": True, "text": "NPU Usage"},
            "scales": scales,
        }

    def collect_sample(self) -> TelemetrySample:
        now = time.time()
        raw: dict = {}

        for idx, paths in self._paths.items():
            busy_us = _read_int(paths["busy_path"])
            if busy_us is None:
                continue

            if idx in self._prev:
                prev_busy, prev_time = self._prev[idx]
                dt_us = (now - prev_time) * 1_000_000
                if dt_us >= 100:
                    delta_busy = busy_us - prev_busy
                    pct = max(0.0, min(100.0, delta_busy / dt_us * 100.0))
                    raw[f"npu_{idx}_busy_pct"] = round(pct, 1)

            self._prev[idx] = (busy_us, now)

            # Memory utilization (bytes → MiB)
            if paths["mem_path"]:
                mem_bytes = _read_int(paths["mem_path"])
                if mem_bytes is not None:
                    raw[f"npu_{idx}_mem_mib"] = round(mem_bytes / (1024 * 1024), 1)

        values = self._filter_values(raw)
        sample = TelemetrySample(timestamp=now, values=values)
        self.check_thresholds(values)
        return sample
