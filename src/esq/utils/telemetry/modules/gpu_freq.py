# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
GPU frequency telemetry module (ESQ-specific).

Collects Intel GPU operating frequency via the Linux DRM sysfs interface,
reading directly from ``/sys/class/drm/card*/`` directories.

Reading strategy per driver
---------------------------
intel_gpu_top and qmassa use the Linux perf PMU (``i915`` / ``xe_XXXX`` PMU
devices) to obtain a *time-weighted average* actual frequency over each
measurement interval.  This approach requires ``CAP_PERFMON`` or
``perf_event_paranoid <= 0``, which is not guaranteed in production systems.

This module uses a sysfs fallback that closely approximates the PMU result:

**i915 driver** (iGPU / older dGPU):
    Reads ``gt/gtN/rps_act_freq_mhz`` (actual hardware frequency).  Returns 0
    when the GT is in RC6 (render power-gate), which is the correct reading —
    the GT clock is stopped.

**xe driver** (Intel Arc / Data Center GPUs, Linux 6.8+):
    The xe GT exposes two power-save states.  ``gtidle/idle_status`` shows
    whether the GT is in C6 (full power-gate, clock stopped) or C0 (powered
    on, command processor active).

    * ``gt-c6`` → ``freq0/act_freq`` is 0 because the clock is genuinely off.
      This module reports 0 — the same as intel_gpu_top.
    * ``gt-c0`` → the command processor is running but ``freq0/act_freq`` only
      updates when the render engines actually clock up.  Between kernel
      dispatches the register reads back 0 even though the GT is powered and
      clocked at ``freq0/cur_freq`` (the governor-requested frequency).  This
      module uses ``cur_freq`` as the best available proxy for the true
      hardware clock in this state, matching intel_gpu_top's PMU view.

Metrics collected (per detected GPU and GT, 0-indexed):
    gpu_0_gt0_mhz:  Frequency of GPU 0, GT 0 in MHz.
    gpu_0_gt1_mhz:  Frequency of GPU 0, GT 1 in MHz (if present).
    gpu_1_gt0_mhz:  Frequency of GPU 1, GT 0 in MHz (if present).
    ...

The GPU index is 0-based among GPUs that expose any frequency data (not
among all Intel GPUs), so the metric numbering may differ from ``gpu_temp``
which uses hwmon-based indexing.
"""

import glob
import logging
import os
import re
import time

from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetryConfig, TelemetrySample

from esq.utils.telemetry.modules._drm import find_intel_gpu_drm_cards

logger = logging.getLogger(__name__)

MODULE_NAME = "gpu_freq"


def _read_mhz(path):
    """Read a sysfs frequency file and return the value in MHz, or None."""
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return None


def _discover_freq_paths():
    """
    Discover all readable GT frequency sysfs paths for Intel GPUs.

    GPU index is 0-based in DRM card sort order (card1, card2, …) so it
    stays aligned with the label map from ``_drm.get_gpu_label_map()``.
    GTs are numbered by the integer suffix of their sysfs directory name.

    Returns:
        List of dicts, each with keys:

        * ``metric``    – metric name, e.g. ``"gpu_0_gt0_mhz"``
        * ``act_path``  – sysfs path for actual hardware frequency
          (0 when GT clock is stopped: xe C6 or i915 RC6)
        * ``cur_path``  – sysfs path for governor-requested frequency
          (fallback for xe C0 with act=0); ``None`` if unavailable
        * ``idle_path`` – sysfs path for ``gtidle/idle_status`` (xe only);
          ``None`` for i915 (which has no instantaneous idle indicator)

    Example output on a system with one i915 iGPU and two xe dGPUs::

        [
          {"metric": "gpu_0_gt0_mhz",
           "act_path": ".../card1/gt/gt0/rps_act_freq_mhz",
           "cur_path": ".../card1/gt/gt0/rps_cur_freq_mhz",
           "idle_path": None},
          {"metric": "gpu_1_gt0_mhz",
           "act_path": ".../0000:04:00.0/tile0/gt0/freq0/act_freq",
           "cur_path": ".../0000:04:00.0/tile0/gt0/freq0/cur_freq",
           "idle_path": ".../0000:04:00.0/tile0/gt0/gtidle/idle_status"},
          ...
        ]
    """
    result = []
    for gpu_idx, (card_path, driver) in enumerate(find_intel_gpu_drm_cards()):
        if driver == "i915":
            # i915 exposes GT directories directly under the DRM card entry.
            for gt_dir in sorted(glob.glob(os.path.join(card_path, "gt", "gt*"))):
                m = re.match(r"^gt(\d+)$", os.path.basename(gt_dir))
                if not m:
                    continue
                gt_num = int(m.group(1))
                act_path = os.path.join(gt_dir, "rps_act_freq_mhz")
                if not os.path.exists(act_path):
                    continue
                cur_path = os.path.join(gt_dir, "rps_cur_freq_mhz")
                result.append(
                    {
                        "metric": f"gpu_{gpu_idx}_gt{gt_num}_mhz",
                        "act_path": act_path,
                        "cur_path": cur_path if os.path.exists(cur_path) else None,
                        "idle_path": None,  # i915 has no instantaneous idle indicator
                    }
                )
        elif driver == "xe":
            # xe exposes GT directories under the PCI device via tile subdirs.
            # The DRM card entry has no "gt/" directory for xe on Linux 6.x.
            device_path = os.path.realpath(os.path.join(card_path, "device"))
            for gt_dir in sorted(glob.glob(os.path.join(device_path, "tile*", "gt*"))):
                if not os.path.isdir(gt_dir):
                    continue
                m = re.match(r"^gt(\d+)$", os.path.basename(gt_dir))
                if not m:
                    continue
                gt_num = int(m.group(1))
                act_path = os.path.join(gt_dir, "freq0", "act_freq")
                if not os.path.exists(act_path):
                    continue
                cur_path = os.path.join(gt_dir, "freq0", "cur_freq")
                idle_path = os.path.join(gt_dir, "gtidle", "idle_status")
                result.append(
                    {
                        "metric": f"gpu_{gpu_idx}_gt{gt_num}_mhz",
                        "act_path": act_path,
                        "cur_path": cur_path if os.path.exists(cur_path) else None,
                        "idle_path": idle_path if os.path.exists(idle_path) else None,
                    }
                )
    return result


class GpuFreqModule(BaseTelemetryModule):
    """
    Collects Intel GPU actual operating frequency via Linux DRM sysfs.

    Supports the ``i915`` driver (``gt/gtN/rps_act_freq_mhz``) and the
    ``xe`` driver (``gt/gtN/freq0/act_freq``).  Multi-GT GPUs report one
    metric per GT.  GPUs without accessible frequency files are silently
    excluded.

    Metrics are named ``gpu_{gpu_idx}_gt{gt_num}_mhz``.  The GPU index
    is 0-based among GPUs with any frequency data.
    """

    module_name = MODULE_NAME

    def __init__(self, config: TelemetryConfig) -> None:
        super().__init__(config)
        # Discover frequency paths once at construction for efficiency
        self._freq_paths = _discover_freq_paths()
        # Build {gpu_idx: pci_slot} from DRM card order (matches discovery order).
        from esq.utils.telemetry.modules._drm import _get_pci_slot

        self._gpu_pci_slots: dict = {
            gpu_idx: _get_pci_slot(card_path) for gpu_idx, (card_path, _) in enumerate(find_intel_gpu_drm_cards())
        }

    def get_default_config(self):
        """Build default scales based on detected Intel GPU names and GT indices."""
        from esq.utils.telemetry.modules._drm import get_gpu_label_map

        label_map = get_gpu_label_map()
        scales = {}
        for entry in self._freq_paths:
            metric = entry["metric"]
            m = re.match(r"gpu_(\d+)_gt(\d+)_mhz", metric)
            if m:
                gpu_idx = int(m.group(1))
                gt_num = m.group(2)
                pci_slot = self._gpu_pci_slots.get(gpu_idx, "")
                gpu_label = label_map.get(pci_slot, f"GPU {gpu_idx}")
                scales[metric] = {
                    "display": True,
                    "label": f"{gpu_label} GT{gt_num}",
                    "unit": "MHz",
                }
        return {
            "title": {"display": True, "text": "GPU Frequency"},
            "scales": scales,
        }

    def is_available(self) -> bool:
        return len(self._freq_paths) > 0

    def collect_sample(self) -> TelemetrySample:
        raw: dict = {}
        for entry in self._freq_paths:
            metric = entry["metric"]
            act_val = _read_mhz(entry["act_path"]) or 0.0
            if act_val > 0.0:
                # Hardware is clocked and executing: use the actual frequency.
                raw[metric] = act_val
            else:
                # act_freq is 0 — determine whether the GT clock is truly stopped
                # or whether we are in the xe C0 sub-state (powered but engines idle).
                idle_path = entry.get("idle_path")
                cur_path = entry.get("cur_path")
                if idle_path:
                    # xe driver: use gtidle/idle_status to distinguish C6 vs C0.
                    try:
                        idle_status = open(idle_path).read().strip()
                    except OSError:
                        idle_status = None
                    if idle_status == "gt-c6":
                        # GT is fully power-gated: clock is genuinely stopped.
                        raw[metric] = 0.0
                    else:
                        # GT is in C0 (command processor active) but render engines
                        # are idle between dispatches — act_freq register reads 0.
                        # Use cur_freq (governor-requested frequency) as the best
                        # sysfs proxy for the true hardware clock, matching the
                        # non-zero reading that intel_gpu_top's PMU gives here.
                        raw[metric] = _read_mhz(cur_path) or 0.0
                else:
                    # i915 driver: no instantaneous idle indicator available.
                    # rps_act_freq_mhz is 0 when in RC6 (clock stopped), which is
                    # the correct reading for that state.
                    raw[metric] = 0.0

        values = self._filter_values(raw)
        sample = TelemetrySample(timestamp=time.time(), values=values)
        self.check_thresholds(values)
        return sample
