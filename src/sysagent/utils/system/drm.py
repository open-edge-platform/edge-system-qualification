# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared DRM (Direct Rendering Manager) display and GPU enumeration helpers.

Centralizes the read-only probes over the Linux DRM subsystem
(``/sys/class/drm``) that were previously duplicated across the
display-connectivity, peripheral, stress, and virtualization test suites.

Provided capabilities:

- DRM card enumeration (:func:`get_drm_cards`)
- Display connector enumeration with status/type/mode
  (:func:`get_drm_connectors`, :func:`get_connector_modes`)
- Connected-display counting (:func:`count_connected_displays`)
- Display-controller-to-port mapping with PCI identity
  (:func:`get_display_controllers`)
- Intel GPU card discovery for stress targeting
  (:func:`count_intel_drm_cards`, :func:`get_intel_drm_cards`,
  :func:`resolve_intel_gpu_devnode`)

All probes are read-only and degrade gracefully (returning empty containers or
zero) when the DRM subsystem or related tooling is unavailable.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sysagent.utils.core import run_command

logger = logging.getLogger(__name__)

DRM_BASE = "/sys/class/drm"
INTEL_VENDOR_ID = "0x8086"
PHYSICAL_CONNECTOR_TYPES = ("HDMI", "DP", "DVI", "VGA", "eDP", "LVDS")


def is_drm_available() -> bool:
    """Return True when the DRM subsystem (``/sys/class/drm``) is present."""
    return os.path.exists(DRM_BASE)


def get_drm_cards() -> Dict[str, dict]:
    """
    Enumerate DRM card devices from ``/sys/class/drm``.

    Returns:
        Dict of {card_name: {"path": str, "available": bool}} for each
        ``card<N>`` device (connectors such as ``card0-HDMI-A-1`` are excluded).
    """
    devices: Dict[str, dict] = {}

    try:
        if not os.path.exists(DRM_BASE):
            logger.warning("DRM subsystem not available - /sys/class/drm not found")
            return devices

        for entry in os.listdir(DRM_BASE):
            # Only consider card* entries, not card*-* connectors
            if re.match(r"^card\d+$", entry):
                device_path = os.path.join(DRM_BASE, entry)
                devices[entry] = {
                    "path": device_path,
                    "available": os.path.isdir(device_path),
                }
    except (IOError, OSError) as e:
        logger.warning(f"Failed to enumerate DRM devices: {e}")

    return devices


def get_drm_connectors(card: Optional[str] = None) -> Dict[str, dict]:
    """
    Enumerate display connectors (ports) exposed by the DRM subsystem.

    Args:
        card: Specific DRM card (e.g. ``card0``) to filter by, or None for all.

    Returns:
        Dict of {connector_name: {
            "device": str,
            "path": str,
            "status": "connected"/"disconnected"/"unknown",
            "type": str (e.g. "HDMI-A", "DP"),
            "enabled": bool,
            "connector": str,
        }}
    """
    ports: Dict[str, dict] = {}

    try:
        if not os.path.exists(DRM_BASE):
            logger.warning("DRM subsystem not available")
            return ports

        entries = os.listdir(DRM_BASE)

        for entry in entries:
            # Match connector pattern: card0-HDMI-A-1, card1-DP-1, etc.
            match = re.match(r"^(card\d+)-(.+)$", entry)
            if not match:
                continue

            device = match.group(1)
            connector = match.group(2)

            # Filter by specific card if requested
            if card and device != card:
                continue

            port_path = os.path.join(DRM_BASE, entry)

            # Read connection status
            status_file = os.path.join(port_path, "status")
            status = "unknown"
            if os.path.exists(status_file):
                try:
                    with open(status_file, "r", encoding="utf-8") as f:
                        status = f.read().strip()
                except (IOError, OSError):
                    pass

            # Read enabled status
            enabled_file = os.path.join(port_path, "enabled")
            enabled = False
            if os.path.exists(enabled_file):
                try:
                    with open(enabled_file, "r", encoding="utf-8") as f:
                        enabled = f.read().strip() == "enabled"
                except (IOError, OSError):
                    pass

            # Determine port type (HDMI-A, DP, etc.)
            port_type_match = re.match(r"^([A-Z\-]+)-\d+$", connector)
            port_type = port_type_match.group(1) if port_type_match else connector

            ports[entry] = {
                "device": device,
                "path": port_path,
                "status": status,
                "type": port_type,
                "enabled": enabled,
                "connector": connector,
            }

    except (IOError, OSError) as e:
        logger.warning(f"Failed to enumerate display ports: {e}")

    return ports


def get_connector_modes(connector_path: str) -> List[str]:
    """
    Read available display modes for a DRM connector.

    Args:
        connector_path: Path to a connector under ``/sys/class/drm/``.

    Returns:
        List of resolution strings (e.g. ["1920x1080", "1280x720"]).
    """
    modes: List[str] = []
    modes_file = os.path.join(connector_path, "modes")

    if os.path.exists(modes_file):
        try:
            with open(modes_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    modes = content.split("\n")
        except (IOError, OSError) as e:
            logger.debug(f"Could not read modes for {connector_path}: {e}")

    return modes


def _get_pci_display_map() -> Dict[str, str]:
    """
    Map PCI addresses to display/GPU controller names via ``lspci``.

    Returns:
        Dict of {pci_address: "name"} for VGA/3D/display controllers.
    """
    names: Dict[str, str] = {}
    result = run_command(["lspci", "-D"], timeout=10)
    if not result or result.returncode != 0 or not result.stdout:
        return names
    for line in result.stdout.split("\n"):
        if not re.search(r"vga|3d|display", line, re.IGNORECASE):
            continue
        slot, _, rest = line.partition(" ")
        desc = rest.split(":", 1)[1].strip() if ":" in rest else rest.strip()
        names[slot] = desc
    return names


def get_display_controllers() -> List[dict]:
    """
    Enumerate display controllers and every physical port they expose.

    Reads the DRM subsystem so reports can show, per controller, the PCI
    address, GPU name, total available ports, connector type (HDMI/DP/etc.),
    and which ports are currently connected. Virtual and writeback connectors
    are excluded.

    Returns:
        List of {"card", "pci_id", "name", "ports": [{"name","type","status"}]}.
    """
    pci_names = _get_pci_display_map()
    controllers: Dict[str, dict] = {}
    try:
        for entry in sorted(os.listdir(DRM_BASE)):
            if "Virtual" in entry or "WRITEBACK" in entry:
                continue
            if not any(conn in entry for conn in PHYSICAL_CONNECTOR_TYPES):
                continue
            card, sep, port = entry.partition("-")
            port = port if sep else entry
            ptype = port.split("-")[0] if port else "unknown"
            status = "unknown"
            status_file = os.path.join(DRM_BASE, entry, "status")
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    status = f.read().strip()
            except (IOError, OSError):
                pass
            if card not in controllers:
                pci_id = "unknown"
                try:
                    dev_link = os.path.realpath(os.path.join(DRM_BASE, card, "device"))
                    matches = re.findall(r"([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d)", dev_link)
                    if matches:
                        pci_id = matches[-1]
                except OSError:
                    pass
                controllers[card] = {
                    "card": card,
                    "pci_id": pci_id,
                    "name": pci_names.get(pci_id, "unknown GPU"),
                    "ports": [],
                }
            controllers[card]["ports"].append({"name": port, "type": ptype, "status": status})
    except (IOError, OSError) as e:
        logger.debug(f"Failed to enumerate DRM controllers: {e}")
    return list(controllers.values())


def count_connected_displays() -> int:
    """Count physically connected displays via the DRM subsystem."""
    count = 0
    for controller in get_display_controllers():
        count += sum(1 for p in controller["ports"] if p["status"] == "connected")
    return count


def count_intel_drm_cards() -> int:
    """Count Intel GPU cards under the DRM subsystem."""
    count = 0
    for vendor_file in Path(DRM_BASE).glob("card*/device/vendor"):
        try:
            with open(vendor_file, "r", encoding="utf-8") as file:
                vendor = file.read().strip().lower()
            if vendor == INTEL_VENDOR_ID:
                count += 1
        except (IOError, OSError):
            continue
    return count


def get_intel_drm_cards() -> List[str]:
    """Return Intel DRM card device nodes sorted by card index."""
    cards: List[Tuple[int, str]] = []
    for vendor_file in Path(DRM_BASE).glob("card*/device/vendor"):
        try:
            with open(vendor_file, "r", encoding="utf-8") as file:
                vendor = file.read().strip().lower()
            if vendor != INTEL_VENDOR_ID:
                continue
            card_name = vendor_file.parent.parent.name
            if not card_name.startswith("card"):
                continue
            card_index = int(card_name.replace("card", ""))
            devnode = f"/dev/dri/{card_name}"
            if os.path.exists(devnode):
                cards.append((card_index, devnode))
        except (IOError, OSError, ValueError):
            continue

    cards.sort(key=lambda item: item[0])
    return [devnode for _, devnode in cards]


def resolve_intel_gpu_devnode(gpu_device_index: int) -> str:
    """Resolve a configured GPU index to a concrete DRM card node path."""
    if gpu_device_index < 0:
        return ""
    intel_cards = get_intel_drm_cards()
    if gpu_device_index >= len(intel_cards):
        return ""
    return intel_cards[gpu_device_index]
