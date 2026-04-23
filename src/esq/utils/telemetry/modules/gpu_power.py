# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
GPU power telemetry module (ESQ-specific).

Collects Intel GPU power consumption in Watts via the Linux hwmon sysfs
interface, reading from ``/sys/class/hwmon/hwmon*/`` directories exposed
by the ``xe`` or ``i915`` kernel driver.

**Measurement method**

Power is derived from energy counter deltas — this is the same method used
by ``intel_gpu_top`` and ``qmassa``.  The xe PMU does *not* expose any
power or energy events (only engine-ticks, frequency, and C6-residency),
so the hwmon energy counters are the sole accurate power source.

Two measurement approaches are tried per GPU, in priority order:

1. **Direct power** — ``powerN_input`` (microwatts).  Available in some
   driver/kernel combinations where the driver reports instantaneous power.

2. **Energy counter delta** — ``energyN_input`` (microjoules).  The
   standard xe driver path.  Power is derived by differencing consecutive
   readings:

   .. code-block:: text

       power_W = (energy_t2_uJ - energy_t1_uJ) / 1e6 / (t2 - t1)

**Energy channels**

Some GPUs expose two distinct energy counters:

- ``pkg`` (package) — GPU die power only.  This is the primary metric and
  maps directly to what ``intel_gpu_top`` labels "GPU power".
- ``card`` (board) — total card power including VRAM and PCB regulators.
  Useful for thermal budget analysis.  Reported as a separate metric where
  available.

GPUs and their hwmon directories are discovered dynamically at module
construction.  Hwmon numbers are not hardcoded.

**Metrics collected** (per detected GPU, 0-indexed):

``gpu_0_w``
    GPU 0 package (die) power in Watts.  Falls back to board power if no
    ``pkg`` channel is available.

``gpu_0_card_w``
    GPU 0 total board power in Watts.  Only present when the driver exposes
    a distinct ``card`` energy counter alongside ``pkg``.

``gpu_1_w``, ``gpu_1_card_w``, …
    Same as above for additional GPUs.

The GPU index is 0-based among GPUs that expose any power data.  This
index matches the indexing used by ``gpu_temp`` (both are hwmon-based).

The first call to ``collect_sample()`` will return no power values because
the energy counter baseline is established in ``__init__``.
"""

import glob
import logging
import os
import time

from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetryConfig, TelemetrySample

logger = logging.getLogger(__name__)

MODULE_NAME = "gpu_power"

# Hwmon driver names that expose Intel GPU power/energy data
_GPU_HWMON_DRIVERS = frozenset({"xe", "i915"})

# Energy labels reported by the xe/i915 hwmon interface.
# "pkg"  = GPU die/package power only (primary; matches intel_gpu_top "GPU power").
# "card" = Total board power including VRAM and PCB regulators.
_LABEL_PKG = "pkg"
_LABEL_CARD = "card"


def _find_gpu_hwmon_dirs():
    """
    Return a sorted list of hwmon directories for Intel GPU drivers.

    Mirrors the discovery used by ``gpu_temp`` so that GPU indices are
    consistent between the two modules.
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


def _read_micro(path):
    """Read a sysfs value in micro-units and return as float, or None."""
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None


def _collect_labeled_channels(hwmon_dir, prefix):
    """
    Return a dict of ``{label: (source_type, path)}`` for all
    ``{prefix}N_input`` files that have a corresponding ``{prefix}N_label``.

    *source_type* is the *prefix* string (``"power"`` or ``"energy"``).
    """
    channels: dict = {}
    for label_path in sorted(glob.glob(os.path.join(hwmon_dir, f"{prefix}*_label"))):
        try:
            with open(label_path) as f:
                lbl = f.read().strip().lower()
        except OSError:
            continue
        input_path = label_path.replace("_label", "_input")
        if os.path.exists(input_path):
            channels[lbl] = (prefix, input_path)
    return channels


def _discover_power_sources():
    """
    Discover all available power/energy channels for each Intel GPU.

    For each GPU hwmon directory the function:

    1. Collects every labeled ``powerN_input`` channel.
    2. Collects every labeled ``energyN_input`` channel (skipping labels
       already covered by a power channel).
    3. Falls back to the first unlabeled ``energy*_input`` when nothing
       else is found.

    Returns:
        List of ``(gpu_idx, source_type, path, channel_label)`` tuples,
        where ``channel_label`` is ``"pkg"``, ``"card"``, or ``None``
        (unlabeled fallback).  Multiple tuples per GPU are possible when
        both ``pkg`` and ``card`` channels are available.

    Metric mapping (applied in ``collect_sample``):
        ``pkg`` or unlabeled → ``gpu_{idx}_w``   (primary / backward-compatible)
        ``card``              → ``gpu_{idx}_card_w``  (board-level, additive)
    """
    sources = []
    for gpu_idx, hwmon_dir in enumerate(_find_gpu_hwmon_dirs()):
        # Accumulate all labeled channels: power takes priority over energy
        all_channels: dict = {}  # label -> (source_type, path)

        for lbl, entry in _collect_labeled_channels(hwmon_dir, "power").items():
            all_channels[lbl] = entry
        for lbl, entry in _collect_labeled_channels(hwmon_dir, "energy").items():
            if lbl not in all_channels:
                all_channels[lbl] = entry

        if not all_channels:
            # Unlabeled fallback: use the first energy*_input found
            unlabeled = sorted(glob.glob(os.path.join(hwmon_dir, "energy*_input")))
            if unlabeled:
                sources.append((gpu_idx, "energy", unlabeled[0], None))
            continue

        # Primary channel: pkg preferred, else card, else first available
        for primary_label in (_LABEL_PKG, _LABEL_CARD):
            if primary_label in all_channels:
                src_type, path = all_channels[primary_label]
                sources.append((gpu_idx, src_type, path, primary_label))
                break
        else:
            # No pkg or card — use whatever is available (shouldn't happen normally)
            first_lbl, (src_type, path) = next(iter(all_channels.items()))
            sources.append((gpu_idx, src_type, path, first_lbl))

        # Secondary card channel: expose board power alongside pkg when both exist
        if _LABEL_CARD in all_channels and _LABEL_PKG in all_channels:
            src_type, path = all_channels[_LABEL_CARD]
            sources.append((gpu_idx, src_type, path, _LABEL_CARD))

    return sources


class GpuPowerModule(BaseTelemetryModule):
    """
    Collects Intel GPU power consumption via the Linux hwmon sysfs interface.

    Supports the ``xe`` driver (Intel Arc / Data Center) and the ``i915``
    driver.  Uses direct ``powerN_input`` when available; otherwise derives
    power from the ``energyN_input`` counter delta — identical to the
    method used by ``intel_gpu_top`` and ``qmassa``.

    Note: the xe PMU exposes no power or energy events (only engine-ticks,
    frequency, and C6-residency), so hwmon energy counters are the sole
    accurate power source on xe hardware.

    Hwmon numbering is discovered dynamically — this module is unaffected
    by hwmon renumbering across reboots.

    Metrics:
        ``gpu_{idx}_w``       GPU package (die) power in Watts.
        ``gpu_{idx}_card_w``  Total board power in Watts (where available).

    The GPU index matches the indexing used by ``gpu_temp`` (hwmon-based).
    """

    module_name = MODULE_NAME

    def __init__(self, config: TelemetryConfig) -> None:
        super().__init__(config)
        # Each source: (gpu_idx, source_type, path, metric_name)
        # metric_name is pre-computed: "gpu_{idx}_w" or "gpu_{idx}_card_w"
        raw_sources = _discover_power_sources()
        self._sources = self._assign_metric_names(raw_sources)
        # State for energy delta: metric_name -> (energy_uJ, timestamp)
        self._prev_energy: dict = {}
        # Cache PCI slot per GPU index for label lookup
        from esq.utils.telemetry.modules._drm import _get_pci_slot

        self._gpu_pci_slots: dict = {
            gpu_idx: _get_pci_slot(hwmon_dir) for gpu_idx, hwmon_dir in enumerate(_find_gpu_hwmon_dirs())
        }
        self._prime_energy_readings()

    @staticmethod
    def _assign_metric_names(raw_sources):
        """
        Convert raw ``(gpu_idx, src_type, path, channel_label)`` tuples into
        ``(gpu_idx, src_type, path, metric_name)`` tuples.

        Channel mapping:
          ``pkg`` or ``None`` (unlabeled) → ``gpu_{idx}_w``   (primary)
          ``card`` (when ``pkg`` also present for same GPU) → ``gpu_{idx}_card_w``

        When only ``card`` is available with no ``pkg``, it maps to
        ``gpu_{idx}_w`` so the primary metric is always populated.
        """
        # Determine which gpu_idx values have a pkg (or unlabeled) source
        has_primary: set = set()
        for gpu_idx, _, _, lbl in raw_sources:
            if lbl in (_LABEL_PKG, None):
                has_primary.add(gpu_idx)

        sources = []
        for gpu_idx, src_type, path, lbl in raw_sources:
            if lbl == _LABEL_CARD and gpu_idx in has_primary:
                metric = f"gpu_{gpu_idx}_card_w"
            else:
                metric = f"gpu_{gpu_idx}_w"
            sources.append((gpu_idx, src_type, path, metric))
        return sources

    def get_default_config(self):
        """Build default scales and thresholds for all detected power metrics."""
        from esq.utils.telemetry.modules._drm import get_gpu_label_map

        label_map = get_gpu_label_map()
        scales = {}
        for gpu_idx, _, _, metric in self._sources:
            if metric in scales:
                continue
            pci_slot = self._gpu_pci_slots.get(gpu_idx, "")
            gpu_label = label_map.get(pci_slot, f"GPU {gpu_idx}")
            if metric.endswith("_card_w"):
                label = f"{gpu_label} Board Power"
            else:
                label = f"{gpu_label} Power"
            scales[metric] = {"display": True, "label": label, "unit": "W"}
        return {
            "chart_type": "area",
            "title": {"display": True, "text": "GPU Power"},
            "scales": scales,
        }

    def _prime_energy_readings(self) -> None:
        now = time.time()
        for _gpu_idx, source_type, path, metric in self._sources:
            if source_type == "energy":
                val = _read_micro(path)
                if val is not None:
                    self._prev_energy[metric] = (val, now)

    def is_available(self) -> bool:
        return len(self._sources) > 0

    def collect_sample(self) -> TelemetrySample:
        now = time.time()
        raw: dict = {}

        for _gpu_idx, source_type, path, metric in self._sources:
            if source_type == "power":
                val = _read_micro(path)
                if val is not None and val >= 0:
                    raw[metric] = round(val / 1e6, 2)

            else:  # "energy" — compute from counter delta
                val = _read_micro(path)
                if val is None:
                    continue
                if metric in self._prev_energy:
                    prev_val, prev_time = self._prev_energy[metric]
                    dt = now - prev_time
                    if dt >= 0.1:  # guard against division by near-zero
                        power_w = (val - prev_val) / 1e6 / dt
                        if power_w >= 0:
                            raw[metric] = round(power_w, 2)
                self._prev_energy[metric] = (val, now)

        values = self._filter_values(raw)
        sample = TelemetrySample(timestamp=now, values=values)
        self.check_thresholds(values)
        return sample
