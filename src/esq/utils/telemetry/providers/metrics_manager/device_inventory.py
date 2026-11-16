# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Device-group mappings sourced from Metrics Manager capabilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

_RENDER_NODE_BASE_MINOR = 128


def gpu_groups_by_pci_bdf(devices: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Map GPU PCI BDFs to Metrics Manager's iGPU/dGPU categories."""
    gpu_devices = []
    for device in devices:
        category = str(device.get("category") or "").lower()
        raw_details = device.get("details")
        details = raw_details if isinstance(raw_details, dict) else {}
        bdf = str(details.get("pci_bdf") or "").lower()
        if device.get("present") and category in {"igpu", "dgpu"} and bdf:
            gpu_devices.append((str(device.get("id") or ""), bdf, category))

    def device_order(device: tuple[str, str, str]) -> tuple[int, int | str]:
        device_id = device[0]
        suffix = device_id[4:] if device_id.startswith("card") else ""
        if suffix.isdigit():
            return (0, int(suffix))
        return (1, device_id)

    gpu_devices.sort(key=device_order)
    discrete_bdfs = [bdf for _, bdf, category in gpu_devices if category == "dgpu"]
    discrete_index = {bdf: index for index, bdf in enumerate(discrete_bdfs)}

    groups: dict[str, str] = {}
    for _, bdf, category in gpu_devices:
        if category == "igpu":
            groups[bdf] = "iGPU"
        elif len(discrete_bdfs) == 1:
            groups[bdf] = "dGPU"
        else:
            groups[bdf] = f"dGPU[{discrete_index[bdf]}]"
    return groups


def gpu_groups_by_render_id(
    devices: Iterable[dict[str, Any]], gpu_ids: Iterable[int]
) -> dict[int, str]:
    """Correlate Metrics Manager render-node ordinals with its PCI-ID categories."""
    groups_by_bdf = gpu_groups_by_pci_bdf(devices)
    groups: dict[int, str] = {}
    for gpu_id in gpu_ids:
        bdf = render_node_pci_bdf(gpu_id)
        if bdf and bdf in groups_by_bdf:
            groups[gpu_id] = groups_by_bdf[bdf]
    return groups


def render_node_pci_bdf(gpu_id: int) -> str | None:
    """Return a render node's PCI BDF without independently classifying it."""
    if gpu_id < 0:
        return None
    device_link = (
        Path("/sys/class/drm")
        / f"renderD{_RENDER_NODE_BASE_MINOR + gpu_id}"
        / "device"
    )
    try:
        return device_link.resolve(strict=True).name.lower()
    except OSError:
        return None
