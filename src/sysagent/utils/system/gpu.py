# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Intel GPU static configuration via Linux sysfs.

Reads per-GPU frequency caps, power limits, and related configuration
directly from the DRM and hwmon sysfs interfaces exposed by the ``xe``
and ``i915`` kernel drivers.  This data is collected once at system-info
time (not continuously) and complements the live telemetry collected by
the ``gpu_freq`` and ``gpu_power`` modules.

**Why this matters for telemetry analysis**

When GPU frequency or power telemetry is being reviewed, the static limits
provide the reference upper bounds:

- If ``gpu_N_w`` (from ``gpu_power``) approaches ``power_limits.cap_w``,
  the GPU is running at its configured TDP — further load will trigger
  frequency throttling.
- If ``gpu_N_gt0_mhz`` (from ``gpu_freq``) is well below
  ``freq_limits.gt0.rp0_mhz``, it may indicate thermal or power throttling
  rather than light workload.

**Frequency limit fields** (per GT, all in MHz):

``rp0_mhz``
    Hardware maximum / boost frequency (RP0).  The highest clock the GPU
    can reach under any circumstances (absent throttling).

``max_mhz``
    Current software frequency ceiling — the operating system or driver
    may reduce this below RP0 for thermal / power reasons.  Writeable by
    root; may differ from ``rp0_mhz`` on thermally constrained systems.

``min_mhz``
    Current software frequency floor.

``rpe_mhz``
    Efficient frequency (RP1 / RPe).  The "knee" of the power-performance
    curve; the highest frequency at which the GPU is power-efficient.

``rpn_mhz``
    Hardware minimum frequency (RPn).

**Power limit fields** (in Watts):

``cap_w``
    Sustained power cap (PL1-equivalent), enforced by the driver / PCODE.
    GPU package or card power will not exceed this limit averaged over the
    cap window.  Present on cards where the driver exposes
    ``powerN_cap`` (e.g. Intel Arc Pro B60).

``crit_w``
    Hard critical power limit; the firmware will reduce clocks
    aggressively to stay below this.

``max_w``
    Maximum board power limit (may be a higher advisory limit than
    ``cap_w`` on some SKUs, e.g. Intel Arc A770 exposes ``powerN_max``
    rather than ``powerN_cap``).

``channels``
    Raw per-label power channel dict (``{"pkg": {...}, "card": {...}}``)
    for detailed inspection.
"""

import glob
import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DRM_DRIVERS = frozenset({"xe", "i915"})

# Sanity ceiling for frequency values (MHz).
# The xe driver has a known firmware bug on some Arc SKUs where rpa_freq
# is populated with a raw register value (e.g. 614400) instead of MHz.
# Any value above this ceiling is treated as invalid and discarded.
_FREQ_SANITY_MAX_MHZ = 5000

# Conversion factor from microwatts (sysfs units) to Watts.
_UW_TO_W = 1e-6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_int(path: str) -> Optional[int]:
    """Read a sysfs integer file, return None on error."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_mhz(path: str) -> Optional[int]:
    """
    Read a sysfs frequency file (integer MHz) and return the value, or None.

    Values outside the range (0, ``_FREQ_SANITY_MAX_MHZ``] are discarded to
    guard against driver bugs that emit raw register values instead of MHz.
    """
    val = _read_int(path)
    if val is not None and 0 < val <= _FREQ_SANITY_MAX_MHZ:
        return val
    if val is not None:
        logger.debug("Discarding implausible freq value %d from %s", val, path)
    return None


def _find_drm_cards():
    """
    Yield ``(pci_slot, driver, card_path)`` for each Intel GPU DRM card node.

    Connector sub-nodes (e.g. ``card1-DP-1``) are skipped — only the
    primary ``cardN`` entries are returned.
    """
    for entry in sorted(glob.glob("/sys/class/drm/card*/")):
        entry = entry.rstrip("/")
        # Skip connector nodes such as "card1-DP-1" or "card1-HDMI-A-1"
        if re.search(r"card\d+-", os.path.basename(entry)):
            continue
        uevent = os.path.join(entry, "device", "uevent")
        if not os.path.exists(uevent):
            continue
        pci_slot = driver = ""
        try:
            with open(uevent) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("PCI_SLOT_NAME="):
                        pci_slot = line.split("=", 1)[1]
                    elif line.startswith("DRIVER="):
                        driver = line.split("=", 1)[1]
        except OSError:
            continue
        if driver in _DRM_DRIVERS and pci_slot:
            yield pci_slot, driver, entry


def _find_hwmon_for_pci_slot(pci_slot: str) -> Optional[str]:
    """
    Return the hwmon directory path whose ``device`` symlink points to
    *pci_slot*, or ``None`` if no matching Intel GPU hwmon is found.
    """
    for hwmon in sorted(glob.glob("/sys/class/hwmon/hwmon*/")):
        hwmon = hwmon.rstrip("/")
        name_f = os.path.join(hwmon, "name")
        try:
            name = open(name_f).read().strip()
        except OSError:
            continue
        if name not in _DRM_DRIVERS:
            continue
        dev_link = os.path.join(hwmon, "device")
        try:
            dev_target = os.path.basename(os.readlink(dev_link))
        except OSError:
            continue
        if dev_target == pci_slot:
            return hwmon
    return None


# ---------------------------------------------------------------------------
# Per-driver frequency limit readers
# ---------------------------------------------------------------------------


def _collect_xe_freq_limits(card_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Read per-GT frequency limits for an xe-driver GPU.

    xe exposes each GT's frequency knobs under::

        {card}/device/tile{N}/gt{M}/freq0/{rp0_freq,max_freq,min_freq,...}

    Returns a dict keyed by GT label (e.g. ``"gt0"``, ``"gt1"``), each
    containing the subset of frequency fields that are available and valid.
    """
    gts: Dict[str, Dict[str, Any]] = {}
    for freq0_dir in sorted(glob.glob(os.path.join(card_path, "device", "tile*", "gt*", "freq0"))):
        if not os.path.isdir(freq0_dir):
            continue
        # GT label: parent directory name (e.g. "gt0")
        gt_name = os.path.basename(os.path.dirname(freq0_dir))
        entry = {
            "rp0_mhz": _read_mhz(os.path.join(freq0_dir, "rp0_freq")),
            "max_mhz": _read_mhz(os.path.join(freq0_dir, "max_freq")),
            "min_mhz": _read_mhz(os.path.join(freq0_dir, "min_freq")),
            "rpe_mhz": _read_mhz(os.path.join(freq0_dir, "rpe_freq")),
            "rpn_mhz": _read_mhz(os.path.join(freq0_dir, "rpn_freq")),
        }
        # Store only fields that returned valid values
        gts[gt_name] = {k: v for k, v in entry.items() if v is not None}
    return gts


def _collect_i915_freq_limits(card_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Read per-GT frequency limits for an i915-driver GPU.

    i915 exposes GT frequency knobs under::

        {card}/gt/gt{N}/rps_*_freq_mhz

    Returns a dict keyed by GT label (e.g. ``"gt0"``).
    """
    gts: Dict[str, Dict[str, Any]] = {}
    for gt_dir in sorted(glob.glob(os.path.join(card_path, "gt", "gt*"))):
        if not os.path.isdir(gt_dir):
            continue
        gt_name = os.path.basename(gt_dir)
        entry = {
            "rp0_mhz": _read_mhz(os.path.join(gt_dir, "rps_RP0_freq_mhz")),
            "max_mhz": _read_mhz(os.path.join(gt_dir, "rps_max_freq_mhz")),
            "min_mhz": _read_mhz(os.path.join(gt_dir, "rps_min_freq_mhz")),
            "rpe_mhz": _read_mhz(os.path.join(gt_dir, "rps_RP1_freq_mhz")),
            "rpn_mhz": _read_mhz(os.path.join(gt_dir, "rps_RPn_freq_mhz")),
        }
        gts[gt_name] = {k: v for k, v in entry.items() if v is not None}
    return gts


# ---------------------------------------------------------------------------
# Power limit reader
# ---------------------------------------------------------------------------


def _collect_power_limits(hwmon_dir: str) -> Dict[str, Any]:
    """
    Collect power limit configuration from a GPU hwmon directory.

    Reads all labeled ``power*`` sysfs files.  Returns a consolidated dict
    with the following top-level keys (``None`` when not available):

    ``cap_w``
        Sustained power cap in Watts (``powerN_cap``).  If present, the
        driver enforces this limit on the averaged GPU power draw.
        Typically the card-level (board) TDP.

    ``crit_w``
        Hard critical power limit in Watts (``powerN_crit``).  Exceeding
        this triggers aggressive clock reduction.

    ``max_w``
        Maximum allowed power in Watts (``powerN_max``).  Present on some
        SKUs instead of ``cap_w`` (e.g. Intel Arc A770).

    ``channels``
        Full per-label dict: ``{"card": {...}, "pkg": {...}}``.  Each
        channel dict contains whatever limit fields the driver exposes for
        that label (``cap_w``, ``crit_w``, ``max_w``).
    """
    channels: Dict[str, Dict[str, Any]] = {}

    for lf in sorted(glob.glob(os.path.join(hwmon_dir, "power*_label"))):
        try:
            label = open(lf).read().strip().lower()
        except OSError:
            continue
        # Derive the base path: remove "_label" suffix
        prefix = lf[: lf.rfind("_label")]
        ch: Dict[str, Any] = {}

        cap = _read_int(prefix + "_cap")
        if cap is not None:
            ch["cap_w"] = round(cap * _UW_TO_W, 1)

        crit = _read_int(prefix + "_crit")
        if crit is not None:
            ch["crit_w"] = round(crit * _UW_TO_W, 1)

        pw_max = _read_int(prefix + "_max")
        if pw_max is not None:
            ch["max_w"] = round(pw_max * _UW_TO_W, 1)

        if ch:
            channels[label] = ch

    if not channels:
        return {}

    # Build a flattened summary for quick access.
    # Priority order for each field: check "card" then "pkg" (or whichever has the value).
    result: Dict[str, Any] = {"channels": channels}
    for field in ("cap_w", "crit_w", "max_w"):
        for lbl in ("card", "pkg"):
            ch = channels.get(lbl, {})
            if field in ch and field not in result:
                result[field] = ch[field]
                break

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_gpu_sysfs_info() -> Dict[str, Any]:
    """
    Collect static GPU configuration from Linux sysfs for all Intel DRM GPUs.

    Returns a dict keyed by PCI slot string (e.g. ``"0000:04:00.0"``).
    Each value is a dict with:

    ``driver``
        Kernel driver name (``"xe"`` or ``"i915"``).

    ``freq_limits``
        Per-GT frequency limits dict, keyed by GT label
        (e.g. ``{"gt0": {"rp0_mhz": 2400, "max_mhz": 2400, ...}}``).
        See ``_collect_xe_freq_limits`` / ``_collect_i915_freq_limits``.

    ``power_limits``
        Hwmon power limit dict (see ``_collect_power_limits``).
        Empty dict if no hwmon entry is found for this GPU.

    Example output::

        {
            "0000:04:00.0": {        # Arc Pro B60
                "driver": "xe",
                "freq_limits": {
                    "gt0": {"rp0_mhz": 2400, "max_mhz": 2400,
                            "min_mhz": 1200, "rpe_mhz": 400, "rpn_mhz": 400},
                    "gt1": {"rp0_mhz": 1500, "max_mhz": 1500,
                            "min_mhz": 1200, "rpe_mhz": 400, "rpn_mhz": 400},
                },
                "power_limits": {
                    "cap_w": 200.0, "crit_w": 400.0,
                    "channels": {"card": {"cap_w": 200.0, "crit_w": 400.0}},
                },
            },
            "0000:08:00.0": {        # Arc A770
                "driver": "xe",
                "freq_limits": {
                    "gt0": {"rp0_mhz": 2400, "max_mhz": 2400,
                            "min_mhz": 600, "rpe_mhz": 600, "rpn_mhz": 300},
                },
                "power_limits": {
                    "max_w": 210.0,
                    "channels": {"pkg": {"max_w": 210.0}},
                },
            },
            "0000:00:02.0": {        # iGPU (i915)
                "driver": "i915",
                "freq_limits": {
                    "gt0": {"rp0_mhz": 2000, "max_mhz": 2000,
                            "min_mhz": 550, "rpe_mhz": 550, "rpn_mhz": 550},
                    "gt1": {"rp0_mhz": 1400, "max_mhz": 1400,
                            "min_mhz": 100, "rpe_mhz": 100, "rpn_mhz": 100},
                },
                "power_limits": {},
            },
        }
    """
    result: Dict[str, Any] = {}

    for pci_slot, driver, card_path in _find_drm_cards():
        gpu_data: Dict[str, Any] = {"driver": driver}

        if driver == "xe":
            gpu_data["freq_limits"] = _collect_xe_freq_limits(card_path)
        elif driver == "i915":
            gpu_data["freq_limits"] = _collect_i915_freq_limits(card_path)
        else:
            gpu_data["freq_limits"] = {}

        hwmon = _find_hwmon_for_pci_slot(pci_slot)
        gpu_data["power_limits"] = _collect_power_limits(hwmon) if hwmon else {}

        result[pci_slot] = gpu_data
        logger.debug(
            "GPU sysfs collected: %s (%s) — %d GT(s), power_limits=%s",
            pci_slot,
            driver,
            len(gpu_data["freq_limits"]),
            bool(gpu_data["power_limits"]),
        )

    return result
