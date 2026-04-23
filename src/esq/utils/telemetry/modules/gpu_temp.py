# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
GPU temperature telemetry module (ESQ-specific).

Collects Intel GPU thermal readings via the Linux hwmon sysfs interface,
reading directly from ``/sys/class/hwmon/hwmon*/`` directories exposed by
the ``xe`` driver (Intel Arc / Data Center GPUs, Linux 6.8+) or the
``i915`` driver (Intel integrated and older discrete GPUs).

Each hwmon directory with a driver name of ``xe`` or ``i915`` is treated as
one GPU device. Multiple GPUs are indexed from 0 in hwmon-path order,
enabling stable metric names across samples.

Metrics collected (per detected GPU, 0-indexed):
    gpu_0_pkg_c:   Package temperature for GPU 0 in Celsius.
    gpu_0_vram_c:  VRAM temperature for GPU 0 in Celsius (if available).
    gpu_1_pkg_c:   Package temperature for GPU 1 in Celsius (if present).
    gpu_1_vram_c:  VRAM temperature for GPU 1 in Celsius (if present).
    ...

Systems with no compatible Intel GPU will return ``is_available() == False``
and the module will be silently skipped.
"""

import glob
import logging
import os
import time

from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetryConfig, TelemetrySample

logger = logging.getLogger(__name__)

MODULE_NAME = "gpu_temp"

# Hwmon driver names that expose Intel GPU temperature data
_GPU_HWMON_DRIVERS = frozenset({"xe", "i915"})


def _find_gpu_hwmon_dirs():
    """
    Return a sorted list of hwmon directories for Intel GPU drivers.

    Each entry represents one logical GPU device.  Sorted by path for stable
    0-based indexing across samples.
    """
    dirs = []
    for hwmon_dir in sorted(glob.glob("/sys/class/hwmon/hwmon*/")):
        name_file = os.path.join(hwmon_dir, "name")
        try:
            with open(name_file) as f:
                if f.read().strip() in _GPU_HWMON_DRIVERS:
                    dirs.append(hwmon_dir.rstrip("/"))
        except OSError:
            pass
    return dirs


def _read_milli_celsius(path):
    """Read a sysfs ``temp*_input`` file (millidegrees) and return °C, or None."""
    try:
        with open(path) as f:
            return float(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def _parse_gpu_temps():
    """
    Read GPU temperatures from hwmon sysfs.

    Each hwmon directory for an Intel GPU driver is treated as one GPU.
    Recognised labels: ``pkg`` / ``package`` → ``pkg_c``; ``vram`` → ``vram_c``.
    When no label file exists (some i915 versions), the first temp sensor is
    reported as ``pkg_c``.

    Returns:
        Dict mapping GPU index (int) to metric dict, e.g.::

            {0: {"pkg_c": 51.0, "vram_c": 50.0}, 1: {"pkg_c": 35.0}}
    """
    gpu_data: dict = {}
    for gpu_idx, hwmon_dir in enumerate(_find_gpu_hwmon_dirs()):
        temps: dict = {}
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

            if label in ("pkg", "package"):
                temps["pkg_c"] = round(temp, 1)
            elif label == "vram":
                temps["vram_c"] = round(temp, 1)
            elif not temps:
                # No recognised label — treat first sensor as package temperature
                temps["pkg_c"] = round(temp, 1)

        if temps:
            gpu_data[gpu_idx] = temps

    return gpu_data


class GpuTempModule(BaseTelemetryModule):
    """
    Collects Intel GPU temperature via the Linux hwmon sysfs interface.

    Supports the ``xe`` driver (Intel Arc / Data Center GPUs) and the
    ``i915`` driver (Intel integrated and older discrete GPUs).  Each hwmon
    directory for these drivers is treated as one GPU, indexed from 0.

    Metric names are dynamically indexed per GPU (0-based). On a system
    with two discrete GPUs the metrics are::

        gpu_0_pkg_c, gpu_0_vram_c, gpu_1_pkg_c, gpu_1_vram_c
    """

    module_name = MODULE_NAME

    def __init__(self, config: TelemetryConfig) -> None:
        super().__init__(config)
        # Cache hwmon dirs at construction for use in get_default_config().
        self._gpu_hwmon_dirs = _find_gpu_hwmon_dirs()
        # Cache PCI slot per GPU for human-readable label lookup.
        from esq.utils.telemetry.modules._drm import _get_pci_slot

        self._gpu_pci_slots = [_get_pci_slot(d) for d in self._gpu_hwmon_dirs]

    def get_default_config(self):
        """Build default scales and thresholds based on detected Intel GPU names."""
        from esq.utils.telemetry.modules._drm import get_gpu_label_map

        label_map = get_gpu_label_map()
        scales = {}
        for gpu_idx, pci_slot in enumerate(self._gpu_pci_slots):
            gpu_label = label_map.get(pci_slot, f"GPU {gpu_idx}")
            scales[f"gpu_{gpu_idx}_pkg_c"] = {
                "display": True,
                "label": f"{gpu_label} Pkg",
                "unit": "\u00b0C",
            }
            scales[f"gpu_{gpu_idx}_vram_c"] = {
                "display": True,
                "label": f"{gpu_label} VRAM",
                "unit": "\u00b0C",
            }
        return {
            "title": {"display": True, "text": "GPU Temperature"},
            "scales": scales,
        }

    def is_available(self) -> bool:
        return len(self._gpu_hwmon_dirs) > 0

    def collect_sample(self) -> TelemetrySample:
        raw: dict = {}
        try:
            for gpu_idx, gpu_metrics in _parse_gpu_temps().items():
                for key, value in gpu_metrics.items():
                    raw[f"gpu_{gpu_idx}_{key}"] = value
        except Exception as exc:
            logger.debug("gpu_temp sample error: %s", exc)

        values = self._filter_values(raw)
        sample = TelemetrySample(timestamp=time.time(), values=values)
        self.check_thresholds(values)
        return sample
