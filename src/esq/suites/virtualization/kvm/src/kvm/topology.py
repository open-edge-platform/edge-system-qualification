# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Peripheral and display topology helpers for VM passthrough planning.

Enumerates connected input devices, display controllers and ports, and USB
controllers and their devices, then estimates how many VMs can each be given a
dedicated display + keyboard + mouse and renders an ASCII block diagram
summarizing the layout. These helpers are read-only probes over ``/sys``,
``/proc`` and the ``lspci``/``lsusb`` tools.
"""

import logging
import os
import re
from typing import Dict

from sysagent.utils.core import run_command
from sysagent.utils.system.drm import (
    count_connected_displays,
    get_display_controllers,
)

logger = logging.getLogger(__name__)


def get_input_device_pairs() -> tuple[int, int]:
    """
    Count physically connected keyboards and mouse (USB + PS/2).

    Returns:
        Tuple of (keyboards, mouse). Multiple input nodes from the same
        physical device are deduplicated by connection path.
    """
    try:
        with open("/proc/bus/input/devices", "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, OSError) as e:
        logger.debug(f"Failed to read /proc/bus/input/devices: {e}")
        return 0, 0

    keyboard_paths = set()
    mouse_paths = set()
    for block in content.split("\n\n"):
        name = ""
        phys = ""
        handlers = ""
        for line in block.split("\n"):
            if line.startswith("N: Name="):
                name = line[8:].strip().strip('"').lower()
            elif line.startswith("P: Phys="):
                phys = line[8:].strip().lower()
            elif line.startswith("H: Handlers="):
                handlers = line[12:].strip().lower()
        if not phys or not ("usb" in phys or "i8042" in phys or "serio" in phys):
            continue
        base = phys.split("/input")[0]
        if "keyboard" in name or ("kbd" in handlers and "event" in handlers):
            keyboard_paths.add(base)
        if "mouse" in name or ("mouse" in handlers and "event" in handlers):
            mouse_paths.add(base)

    return len(keyboard_paths), len(mouse_paths)


def _get_usb_device_id_map() -> Dict[str, dict]:
    """
    Map "bus:dev" to USB id, name, and HID roles via ``lsusb -v``.

    A single device may expose several Interface Descriptors (e.g. a KVM
    dongle advertising both Keyboard and Mouse). Reading ``bInterfaceProtocol``
    across all interfaces lets one physical device count as keyboard and/or
    mouse instead of being collapsed to a single role.

    Returns:
        Dict of {"bus:dev": {"id": "vid:pid", "name": str, "is_keyboard": bool,
        "is_mouse": bool}} for non-root devices.
    """
    devices: Dict[str, dict] = {}
    result = run_command(["lsusb", "-v"], timeout=20)
    if not result or result.returncode != 0 or not result.stdout:
        return devices
    key = ""
    current_class = ""
    for line in result.stdout.split("\n"):
        header = re.match(r"Bus\s+(\d+)\s+Device\s+(\d+):\s+ID\s+([0-9a-fA-F:]+)\s*(.*)", line)
        if header:
            key = f"{int(header.group(1))}:{int(header.group(2))}"
            current_class = ""
            devices[key] = {
                "id": header.group(3),
                "name": header.group(4).strip() or "unknown",
                "is_keyboard": False,
                "is_mouse": False,
            }
            continue
        if not key:
            continue
        cls = re.search(r"bInterfaceClass\s+(\d+)", line)
        if cls:
            current_class = cls.group(1)
            continue
        # Protocol 1/2 only denote keyboard/mouse on HID (class 3) interfaces;
        # other classes (e.g. Bluetooth/Wireless) reuse the same numbers.
        proto = re.search(r"bInterfaceProtocol\s+(\d+)\s+(\w+)", line)
        if proto and current_class == "3":
            if proto.group(1) == "1":
                devices[key]["is_keyboard"] = True
            elif proto.group(1) == "2":
                devices[key]["is_mouse"] = True
    return devices


def get_hub_input_pairs() -> int:
    """
    Count keyboard/mouse pairs grouped by USB controller via ``lsusb -t``.

    Devices wired through the same controller (bus) form one pair so a
    multi-port hub feeding one keyboard + one mouse counts once. Returns 0 when
    topology is unavailable so callers can fall back to host totals.
    """
    return sum(min(b["keyboards"], b["mouse"]) for b in get_usb_hub_branches().values())


def _get_usb_controller_pci(bus: str) -> str:
    """Resolve the PCI address of the host controller backing a USB bus."""
    try:
        link = os.path.realpath(f"/sys/bus/usb/devices/usb{bus}")
    except OSError:
        return "unknown"
    parent = os.path.basename(os.path.dirname(link))
    if re.match(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d$", parent):
        return parent
    return "unknown"


# lsusb reports link speed as bits/s suffixed with M; map the common rates to a
# human-readable USB generation so reviewers know why a device sits on a given
# bus (USB 2.0 vs SuperSpeed) and which hostbus to pass through.
_USB_SPEED_LABELS = {
    "1.5M": "USB1.0 Low-Speed",
    "12M": "USB1.1 Full-Speed",
    "480M": "USB2.0 Hi-Speed",
    "5000M": "USB3.0 SuperSpeed",
    "10000M": "USB3.1 SuperSpeed+",
    "20000M": "USB3.2 SuperSpeed+ 20G",
}


def _usb_speed_label(speed: str) -> str:
    """Return a USB-generation label for an lsusb link speed (e.g. ``480M``)."""
    if not speed:
        return ""
    base = speed.split("/", 1)[0]  # drop lane suffix such as "/x2"
    return _USB_SPEED_LABELS.get(base, speed)


def get_usb_hub_branches() -> Dict[str, dict]:
    """
    Group USB devices by the host controller (bus) they connect through.

    Each entry is keyed by bus number and records the backing USB controller's
    PCI address, driver, root-hub port count and link speed, plus every physical
    device on that bus. A single xHCI controller presents two host buses (one
    USB 2.0 and one SuperSpeed USB 3.x), so the same PCI address appears under
    two bus numbers and a device lands on whichever bus matches its link speed.
    Devices carry the identifiers needed to map them to a VM by any supported
    passthrough method: vendor:product id, hostbus+hostport (stable physical
    port path), hostbus+hostaddr (Dev number), or whole-controller PCI/VFIO
    passthrough.

    Returns:
        Dict of {bus: {"bus", "pci", "driver", "ports", "speed", "keyboards",
        "mouse", "devices": {dev: {...}}}}. ``ports`` is the root-hub port count,
        ``speed`` the bus link speed (e.g. ``480M``). ``keyboards``/``mouse`` are
        per-bus tallies used for peripheral-pair capacity; each device entry
        includes ``dev``, ``port`` (immediate), ``port_path`` (dotted hostport),
        ``depth``, ``speed``, ``id`` (vid:pid), ``name``, ``classes`` and HID
        roles.
    """
    branches: Dict[str, dict] = {}
    result = run_command(["lsusb", "-t"], timeout=10)
    if not result or result.returncode != 0 or not result.stdout:
        return branches

    id_map = _get_usb_device_id_map()

    current_bus = "0"
    port_stack: list = []
    for line in result.stdout.split("\n"):
        if not line.strip():
            continue

        # Root hub line marks a new bus / host controller. ``Driver=xhci_hcd/14p``
        # carries the port count; the trailing token (e.g. ``480M``) is the bus
        # link speed that determines which device speeds attach here.
        bus_match = re.match(r"^/:\s+Bus\s+(\d+)", line)
        if bus_match:
            current_bus = str(int(bus_match.group(1)))
            drv = re.search(r"Driver=([^/,\s]+)(?:/(\d+)p)?", line)
            spd = re.search(r"([\d.]+M(?:/x\d+)?)\s*$", line)
            branches[current_bus] = {
                "bus": current_bus,
                "pci": _get_usb_controller_pci(current_bus),
                "driver": drv.group(1) if drv else "unknown",
                "ports": int(drv.group(2)) if drv and drv.group(2) else 0,
                "speed": spd.group(1) if spd else "",
                "keyboards": 0,
                "mouse": 0,
                "devices": {},
            }
            port_stack = []
            continue

        # Child device/hub line: depth from indentation, port from "Port NNN".
        indent_match = re.match(r"^(\s*)\|__ ", line)
        if not indent_match:
            continue
        depth = max(1, len(indent_match.group(1)) // 4)
        port_match = re.search(r"Port\s+(\d+)", line)
        dev_match = re.search(r"Dev\s+(\d+)", line)
        if not port_match or not dev_match:
            continue
        port = str(int(port_match.group(1)))
        dev = int(dev_match.group(1))

        # Maintain the ancestor port chain so hostport reflects the physical
        # path (e.g. a device behind a hub on port 2, port 1 -> "2.1").
        del port_stack[depth - 1 :]
        port_stack.append(port)
        port_path = ".".join(port_stack)

        branch = branches.get(current_bus)
        if branch is None:
            continue
        devices = branch["devices"]
        cls = re.search(r"Class=([^,]+)", line)
        if dev not in devices:
            ident = id_map.get(f"{current_bus}:{dev}", {})
            spd = re.search(r"([\d.]+M(?:/x\d+)?)\s*$", line)
            devices[dev] = {
                "dev": dev,
                "port": port,
                "port_path": port_path,
                "depth": depth,
                "speed": spd.group(1) if spd else "",
                "id": ident.get("id", "????:????"),
                "name": ident.get("name", "unknown"),
                "classes": [],
                "is_keyboard": ident.get("is_keyboard", False),
                "is_mouse": ident.get("is_mouse", False),
            }
        if cls:
            cls_name = cls.group(1).strip()
            if cls_name not in devices[dev]["classes"]:
                devices[dev]["classes"].append(cls_name)

    # Tally one keyboard/mouse per physical device, not per interface.
    for info in branches.values():
        info["keyboards"] = sum(1 for d in info["devices"].values() if d["is_keyboard"])
        info["mouse"] = sum(1 for d in info["devices"].values() if d["is_mouse"])

    return branches


def compute_peripheral_vm_capacity(iommu_groups: int) -> tuple[int, int, int, int]:
    """
    Estimate how many VMs can each receive dedicated physical peripherals.

    Capacity here is specifically the number of VMs that can be given a
    pass-through set of one display, one keyboard, and one mouse. It is NOT a
    generic CPU/memory VM count — it requires physically connected peripherals
    plus at least one IOMMU group so they can be passed through. Hub-grouped
    input pairs are preferred; host keyboard/mouse totals are the fallback.

    Returns:
        Tuple of (peripheral_vm_capacity, keyboards, mouse, displays).
    """
    keyboards, mouse = get_input_device_pairs()
    displays = count_connected_displays()

    hub_pairs = get_hub_input_pairs()
    input_pairs = hub_pairs if hub_pairs > 0 else min(keyboards, mouse)

    capacity = min(input_pairs, displays)
    # Passthrough requires IOMMU isolation; no groups means no dedicated VMs.
    if iommu_groups <= 0:
        capacity = 0
    return capacity, keyboards, mouse, displays


def build_vm_topology_diagram(
    capacity: int,
    keyboards: int,
    mouse: int,
    displays: int,
    iommu_groups: int,
) -> str:
    """
    Build an ASCII block diagram visualizing peripheral VM capacity.

    Lays out each provisionable peripheral-passthrough VM (1 display +
    1 keyboard + 1 mouse) against available host peripherals, enumerates each
    display controller's ports (total/type/connected), and lists each USB
    controller's devices with the identifiers needed to assign them to a VM.
    """
    input_pairs = min(keyboards, mouse)
    if displays <= input_pairs and displays <= iommu_groups:
        limiter = "displays"
    elif input_pairs <= iommu_groups:
        limiter = "keyboard/mouse pairs"
    else:
        limiter = "IOMMU groups"

    lines = []
    lines.append("=" * 60)
    lines.append("  PERIPHERAL VM CAPACITY -- SYSTEM BLOCK DIAGRAM")
    lines.append("  (VMs each given 1 dedicated display + keyboard + mouse)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("  HOST RESOURCES")
    lines.append(f"    Displays connected ......... {displays}")
    lines.append(f"    Keyboards detected ......... {keyboards}")
    lines.append(f"    Mouse detected ............. {mouse}")
    lines.append(f"    Keyboard/mouse pairs ....... {input_pairs}")
    lines.append(f"    IOMMU groups (passthrough) . {iommu_groups}")
    lines.append("")
    lines.append(f"  PERIPHERAL VM CAPACITY = {capacity}   (limited by: {limiter})")
    lines.append("-" * 60)
    lines.append("")

    # Display controllers: total vs connected ports, by type.
    controllers = get_display_controllers()
    lines.append("  DISPLAY CONTROLLERS (ports total / connected by type)")
    if not controllers:
        lines.append("    [no display controllers detected]")
    for ctrl in controllers:
        ports = ctrl["ports"]
        connected = sum(1 for p in ports if p["status"] == "connected")
        lines.append(f"    {ctrl['card']} [{ctrl['pci_id']}] {ctrl['name']}: {len(ports)} ports, {connected} connected")
        for p in ports:
            mark = "[*]" if p["status"] == "connected" else "[ ]"
            lines.append(f"      {mark} {p['name']:<10} type={p['type']:<5} {p['status']}")
    lines.append("")

    # USB controllers & devices: surface every supported passthrough handle so
    # each device can be mapped to a VM by id, physical port, address, or the
    # whole controller (PCI/VFIO).
    branches = get_usb_hub_branches()
    lines.append("  USB CONTROLLERS & DEVICES (passthrough assignment options)")
    lines.append("  Map a device to a VM by any of:")
    lines.append("    - vendor:product id ....... usb-host,vendorid=0xVVVV,productid=0xPPPP")
    lines.append("    - host bus + port path .... usb-host,hostbus=B,hostport=P  (stable physical port)")
    lines.append("    - host bus + device addr .. usb-host,hostbus=B,hostaddr=D  (D changes on replug)")
    lines.append("    - whole controller ........ bind the controller PCI address to vfio-pci")
    lines.append("  Note: one xHCI controller exposes two host buses -- a USB 2.0 (480M) bus")
    lines.append("        and a SuperSpeed USB 3.x (5000M+) bus -- so the same PCI address can")
    lines.append("        appear under two bus numbers. A device attaches to the bus matching")
    lines.append("        its link speed; pick hostbus by the device speed shown below.")
    lines.append("")
    if not branches:
        lines.append("    [no USB topology available]")
    for info in branches.values():
        if not info["devices"]:
            continue
        ports = f"{info['ports']} ports" if info["ports"] else "ports n/a"
        speed = _usb_speed_label(info["speed"]) or "speed n/a"
        spec = f"{info['driver']}, {ports}, {speed}"
        lines.append(
            f"    USB Controller  PCI {info['pci']}  Bus {info['bus']}  "
            f"[{spec}]  --  {info['keyboards']} kbd, {info['mouse']} mouse"
        )
        for dev in info["devices"].values():
            roles = []
            if dev["is_keyboard"]:
                roles.append("keyboard")
            if dev["is_mouse"]:
                roles.append("mouse")
            if not roles:
                roles = dev["classes"] or ["device"]
            indent = "      " + "  " * (dev["depth"] - 1)
            dev_speed = dev.get("speed", "") or "-"
            lines.append(
                f"{indent}Dev {dev['dev']:<3} port {dev['port_path']:<7} "
                f"{dev_speed:<7} [{dev['id']}] {dev['name']} <{'+'.join(roles)}>"
            )
    lines.append("-" * 60)
    lines.append("")

    if capacity <= 0:
        lines.append("  [no complete VM station available]")
        lines.append("  Each VM requires 1 display + 1 keyboard + 1 mouse")
        lines.append("  and at least one IOMMU group for passthrough.")
    else:
        for i in range(1, capacity + 1):
            lines.append("  +--------------------------------------+")
            lines.append(f"  |  VM {i:<2}                              |")
            lines.append("  |   [Display] + [Keyboard] + [Mouse]   |")
            lines.append("  +--------------------------------------+")
        spare_disp = displays - capacity
        spare_kbd = keyboards - capacity
        spare_mouse = mouse - capacity
        lines.append("")
        lines.append(f"  Spare: displays={spare_disp}, keyboards={spare_kbd}, mouse={spare_mouse}")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
