# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
NPU frequency telemetry module (ESQ-specific).

Collects Intel NPU operating frequency via the Linux ``intel_vpu`` driver
sysfs interface exposed under ``/sys/class/accel/accel*/``.

The ``intel_vpu`` driver exposes two frequency attributes per NPU device:

``npu_current_frequency_mhz``
    The clock frequency (MHz) currently requested or active for the NPU.
    Reads 0 when the NPU is in a low-power idle state (D3hot / D3cold).

``npu_max_frequency_mhz``
    The maximum supported clock frequency (MHz) for this device.  This is a
    read-only hardware capability value used as a reference.

Reading strategy
----------------
Only ``npu_current_frequency_mhz`` is collected as a time-series metric.
When the NPU is idle (power-gated), the driver returns 0 — the same
convention used by Intel GPU frequency modules for idle GTs.

Metrics collected (per detected NPU, 0-indexed):
    npu_0_freq_mhz:   Current operating frequency of NPU 0 in MHz.
                      Zero when the NPU is power-gated or idle.

Reference: linux-npu-driver sysfs ABI
    drivers/accel/ivpu/ivpu_sysfs.c (npu_current_frequency_mhz,
    npu_max_frequency_mhz)
"""

import glob
import logging
import os

from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetryConfig, TelemetrySample

logger = logging.getLogger(__name__)

MODULE_NAME = "npu_freq"

# ---------------------------------------------------------------------------
# Sysfs discovery  (reuses the same accel device discovery logic as npu_usage)
# ---------------------------------------------------------------------------

_ACCEL_GLOB = "/sys/class/accel/accel*"
_NPU_DRIVER = "intel_vpu"


def _find_npu_accel_devices() -> list:
    """
    Return a sorted list of ``(accel_path, pci_slot)`` for Intel NPU devices.

    Filters ``/sys/class/accel/accel*`` entries to those bound to the
    ``intel_vpu`` driver.
    """
    result = []
    for accel_dir in sorted(glob.glob(_ACCEL_GLOB)):
        dev_path = os.path.join(accel_dir, "device")
        uevent = os.path.join(dev_path, "uevent")
        try:
            content = open(uevent).read()
            if "intel_vpu" not in content and "IVPU" not in content.upper():
                driver_link = os.path.join(dev_path, "driver")
                if os.path.islink(driver_link):
                    if os.path.basename(os.readlink(driver_link)) != _NPU_DRIVER:
                        continue
                else:
                    continue
        except OSError:
            continue

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


def _read_mhz(path: str):
    """Read a sysfs frequency file and return the float value in MHz, or None."""
    try:
        return float(open(path).read().strip())
    except (OSError, ValueError):
        return None


def _is_npu_active(dev_path: str) -> bool:
    """
    Return True if the NPU device is in an active (non-suspended) power state.

    Reads ``power/runtime_status`` for the accel device.  When the NPU is
    in D3hot / runtime-suspended, the kernel may briefly resume the device
    to service a sysfs read of ``npu_current_frequency_mhz``, causing a
    spurious frequency spike even though no NPU work is running.  Checking
    runtime_status first and returning 0 avoids both the wakeup and the spike.
    """
    try:
        status = open(os.path.join(dev_path, "power", "runtime_status")).read().strip()
        return status == "active"
    except OSError:
        return True  # assume active if the file is not readable


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class NpuFreqModule(BaseTelemetryModule):
    """
    Intel NPU current operating frequency telemetry.

    Reads ``npu_current_frequency_mhz`` from the ``intel_vpu`` driver sysfs
    interface.  Reports 0 MHz when the NPU is power-gated (idle).

    Metrics are named ``npu_{idx}_freq_mhz`` where the index is 0-based
    among detected Intel NPU devices.
    """

    module_name = MODULE_NAME

    def __init__(self, config: TelemetryConfig) -> None:
        super().__init__(config)
        devices = _find_npu_accel_devices()

        # npu_idx → (freq_path, dev_path)
        self._freq_paths: dict = {}
        for idx, (accel_dir, pci_slot) in enumerate(devices):
            dev = os.path.join(accel_dir, "device")
            freq_path = os.path.join(dev, "npu_current_frequency_mhz")
            if os.path.exists(freq_path):
                self._freq_paths[idx] = (freq_path, dev)

    def is_available(self) -> bool:
        return bool(self._freq_paths)

    def get_default_config(self) -> dict:
        scales = {}
        for idx in self._freq_paths:
            scales[f"npu_{idx}_freq_mhz"] = {
                "display": True,
                "label": f"NPU {idx}",
                "unit": "MHz",
            }
        return {
            "title": {"display": True, "text": "NPU Frequency"},
            "scales": scales,
        }

    def collect_sample(self) -> TelemetrySample:
        now = __import__("time").time()
        raw: dict = {}
        for idx, (freq_path, dev_path) in self._freq_paths.items():
            if not _is_npu_active(dev_path):
                # NPU is runtime-suspended (D3hot); report 0 without reading
                # the frequency register, which would cause a spurious wakeup
                # and a transient frequency spike with no matching utilization.
                raw[f"npu_{idx}_freq_mhz"] = 0.0
            else:
                val = _read_mhz(freq_path)
                if val is not None:
                    raw[f"npu_{idx}_freq_mhz"] = round(val, 1)

        values = self._filter_values(raw)
        sample = TelemetrySample(timestamp=now, values=values)
        self.check_thresholds(values)
        return sample
