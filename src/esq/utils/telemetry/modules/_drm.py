# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Shared DRM sysfs discovery helper for Intel GPU telemetry modules.

Provides a lightweight function to enumerate Intel GPU DRM cards
in a stable, sorted order by reading sysfs without any third-party
library dependency.  Card paths are discovered dynamically so the
module is not affected by changes in DRM card numbering across reboots.

Also provides GPU label helpers that map PCI slot addresses to short
human-readable labels (e.g. "Arc A770 (dGPU)", "iGPU") for use as
chart axis/scale labels in telemetry reports.
"""

import glob
import os
import re
from typing import Dict


def find_intel_gpu_drm_cards():
    """
    Return a sorted list of ``(card_dir, driver_name)`` for every Intel GPU
    DRM card present on the system.

    Discovery logic:
    - Enumerates ``/sys/class/drm/card*/`` in lexicographic order.
    - Skips connector entries (directory names containing ``-``).
    - Filters by ``device/vendor == 0x8086`` (Intel vendor ID).
    - Reads the driver name via the ``device/driver`` symlink.

    Returns:
        List of ``(str, str)`` tuples: ``(card_path_without_trailing_slash,
        driver_name)``.  ``driver_name`` is empty string if the symlink cannot
        be read.  Entries are sorted by card path for stable 0-based indexing.

    Example result on a system with one iGPU and one dGPU::

        [
            ("/sys/class/drm/card1", "i915"),
            ("/sys/class/drm/card2", "xe"),
        ]
    """
    cards = []
    for card in sorted(glob.glob("/sys/class/drm/card*/")):
        base = os.path.basename(card.rstrip("/"))
        # Connector entries look like "card1-HDMI-A-1" — skip them
        if "-" in base:
            continue
        # Confirm Intel vendor
        vendor_path = os.path.join(card, "device", "vendor")
        try:
            with open(vendor_path) as f:
                if f.read().strip() != "0x8086":
                    continue
        except OSError:
            continue
        # Resolve driver name from symlink
        driver_link = os.path.join(card, "device", "driver")
        try:
            driver = os.path.basename(os.readlink(driver_link))
        except OSError:
            driver = ""
        cards.append((card.rstrip("/"), driver))
    return cards


# ---------------------------------------------------------------------------
# GPU label helpers
# ---------------------------------------------------------------------------


def _get_pci_slot(sysfs_dir: str) -> str:
    """
    Resolve the PCI slot address for a sysfs device directory.

    Works for both hwmon directories (``/sys/class/hwmon/hwmonN/``) and DRM
    card directories (``/sys/class/drm/cardN/``) by reading the ``device``
    symlink and extracting its basename.

    Returns:
        PCI slot string (e.g. ``"0000:04:00.0"``), or empty string on failure.
    """
    device_link = os.path.join(sysfs_dir, "device")
    try:
        resolved = os.path.realpath(device_link)
        basename = os.path.basename(resolved)
        # Validate PCI slot format: XXXX:XX:XX.X
        if re.match(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$", basename):
            return basename
    except OSError:
        pass
    return ""


def _get_pci_slot_for_gt_path(gt_sysfs_path: str) -> str:
    """
    Extract the PCI slot from a DRM GT sysfs path.

    Handles both path formats produced by GPU telemetry discovery:
    - ``/sys/class/drm/card1/gt/gt0/rps_act_freq_mhz``
    - ``/sys/class/drm/card1/gt/gt0/freq0/act_freq``

    Returns:
        PCI slot string, or empty string on failure.
    """
    m = re.match(r"(/sys/class/drm/card\d+)/", gt_sysfs_path)
    if m:
        return _get_pci_slot(m.group(1))
    return ""


def _make_short_label(name: str, driver: str = "") -> str:
    """
    Convert a full GPU device name to a short human-readable label.

    Handles name strings from multiple sources:
    - OpenVINO ``full_device_name``: ``"Intel(R) Arc(TM) A770 Graphics (dGPU)"``
    - Internal GPU table: ``"Intel® Arc™ A770 Graphics"``
    - PCI device name: ``"DG2 [Arc A770]"``

    Examples::

        "Intel(R) Arc(TM) A770 Graphics (dGPU)"  →  "Arc A770"
        "Intel(R) Graphics (iGPU)"               →  "iGPU"
        "Intel® Arc™ B60 Graphics"               →  "Arc B60"
        "Intel® Graphics"          (driver=i915) →  "iGPU"

    Args:
        name:   Raw GPU name string.
        driver: Kernel driver name (``"i915"`` or ``"xe"``), used to
                distinguish iGPU when the name alone is ambiguous.
    """
    if not name:
        return "GPU"

    # Explicit iGPU markers
    if "(iGPU)" in name:
        return "iGPU"
    # Generic "Intel Graphics" without any Arc model number → iGPU
    if driver == "i915" and "Arc" not in name:
        return "iGPU"
    if re.match(r"^Intel[\(®\s]+[R®\)]*\s*Graphics\s*$", name.strip()):
        return "iGPU"

    label = name
    # Strip "Intel(R)", "Intel®", "Intel " prefixes
    label = re.sub(r"^Intel\s*[\(®][R®]?\)?\s*", "", label).strip()
    # "Arc(TM)" (OpenVINO style) or "Arc™" (Unicode) → "Arc"
    label = re.sub(r"Arc\s*(\(TM\)|™)", "Arc", label).strip()
    # Remove "Graphics" and the device-type suffix "(dGPU)" / "(iGPU)" / "(GPU)"
    label = re.sub(r"\bGraphics\b\s*", "", label).strip()
    label = re.sub(r"\s*\([di]?GPU\)\s*", "", label).strip()
    # Collapse multiple spaces
    label = re.sub(r"\s{2,}", " ", label).strip()

    return label or "GPU"


def get_gpu_label_map() -> Dict[str, str]:
    """
    Build a ``{pci_slot: short_label}`` mapping for all detected Intel GPUs.

    Lookup priority:
    1. **SystemInfoCache** (if already initialised) — uses the OpenVINO
       ``full_device_name`` (most accurate, e.g. ``"Intel(R) Arc(TM) A770
       Graphics (dGPU)"``).
    2. **Sysfs PCI device ID + GPU device table** — reads the PCI device ID
       from ``/sys/class/drm/cardN/device/device`` and looks it up in the
       internal ``GPU_DEVICE_LIST``.
    3. **Driver-based fallback** — ``"iGPU"`` for ``i915``, ``"dGPU"`` for
       ``xe``.

    This function is intentionally lightweight: it either reads from an
    in-memory singleton (zero I/O) or reads a handful of small sysfs files.
    It never triggers a full hardware scan.

    Returns:
        Dict mapping PCI slot strings to short labels, e.g.::

            {
                "0000:00:02.0": "iGPU",
                "0000:04:00.0": "Arc B60 (dGPU)",
                "0000:08:00.0": "Arc A770 (dGPU)",
            }
    """
    # 1. Try SystemInfoCache if already initialised (free — reads in-memory data)
    try:
        from sysagent.utils.system.cache import SystemInfoCache  # noqa: F401

        instance = SystemInfoCache._instance  # type: ignore[attr-defined]
        if instance is not None:
            hw = getattr(instance, "_info", {}).get("hardware", {})
            devices = hw.get("gpu", {}).get("devices", [])
            if devices:
                result: Dict[str, str] = {}
                for dev in devices:
                    pci_slot = dev.get("pci_slot", "")
                    if not pci_slot:
                        continue
                    ov = dev.get("openvino", {}) or {}
                    full_name = ov.get("full_device_name", "") or dev.get("device_name", "")
                    driver = dev.get("driver", "")
                    label = _make_short_label(full_name, driver)
                    result[pci_slot] = label
                if result:
                    return result
    except Exception:
        pass

    # 2. Sysfs PCI device ID + GPU_DEVICE_LIST table
    try:
        from sysagent.utils.system.gpu_devices import GPU_DEVICE_LIST  # noqa: F401

        pci_table: Dict[str, str] = {entry["pci_id"].upper(): entry["name"] for entry in GPU_DEVICE_LIST}
    except Exception:
        pci_table = {}

    result = {}
    for card, driver in find_intel_gpu_drm_cards():
        pci_slot = _get_pci_slot(card)
        if not pci_slot:
            continue
        try:
            with open(os.path.join(card, "device", "device")) as fh:
                raw_id = fh.read().strip().lstrip("0x").lstrip("0X").upper()
        except OSError:
            raw_id = ""

        table_name = pci_table.get(raw_id, "")
        if table_name:
            result[pci_slot] = _make_short_label(table_name, driver)
        else:
            # 3. Fallback
            result[pci_slot] = "iGPU" if driver == "i915" else "dGPU"

    return result
