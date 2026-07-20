# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Realtime Platform — PCI Multi-VC Detection.

Counts PCIe devices that expose the Virtual Channel (VC) extended capability
with Extended VC Count ≥ 1, indicating the hardware supports Multi-VC. When
Intel TCC Multi Virtual Channel Configuration is enabled, VC1 is actively
enabled with TC/VC mappings assigned, providing a dedicated real-time traffic
queue for deterministic memory-access latency.

Detection parses the output of ``lspci -vvvnn`` captured by
``system-setup.sh`` (PCI Device Info Dump).
Running lspci as root gives full 4096-byte PCIe extended config space access
via sysfs (kernel grants full access to processes with CAP_SYS_ADMIN).
The dump is stored under ``~/.esq/pci/lspci_verbose.txt`` with user-only
permissions and is shared across all ESQ tests that need PCI device
information, not just this VC test.

lspci Virtual Channel output format
====================================
Capabilities: [OFFSET v1] Virtual Channel
    Caps:   LPEVC=N ...
    VC0:    Caps:   ...
            Ctrl:   Enable+/- ID=N ArbSelect=Fixed TC/VC=HH
            Status: NegoPending+/- InProgress+/-
    VC1:    ...

Fields:
    LPEVC   -- Low Priority Extended VC Count (bits[6:4] of Cap Reg 1)
    Enable  -- VC resource enabled (bit[31] of VC Resource Control Reg)
    ID      -- VC identifier (bits[26:24] of VC Resource Control Reg)
    TC/VC   -- TC-to-VC map (bits[7:0] of VC Resource Control Reg, hex)
    NegoPending -- VC negotiation pending (bit[1] of VC Resource Status Reg)
"""

import csv
import io
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import allure
import pytest
from sysagent.utils.core import Metrics, Result

logger = logging.getLogger(__name__)

# lspci verbose dump captured by system-setup.sh (PCI Device Info Dump).
# Stored in the current user's home directory under ~/.esq/pci/.
# Protected with user-only permissions (0700 dir, 0600 file).
# Reusable by all ESQ tests that need PCI device information.
# Re-run system-setup.sh to refresh after hardware changes.
_LSPCI_DUMP = Path.home() / ".esq" / "pci" / "lspci_verbose.txt"

# --- lspci output parsing patterns -------------------------------------------
# Device header — lspci omits the domain (0000:) by default, so the BDF can
# appear as either "83:00.0" or "0000:83:00.0".  The optional domain group is
# normalised to "0000:" in _parse_lspci_vc_devices() so BDFs match sysfs names.
_RE_BDF_CLASS = re.compile(
    r"^((?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])\s.*\[([0-9a-f]{4})\]:",
    re.IGNORECASE,
)
# Virtual Channel extended capability header inside a device block
_RE_VC_CAP = re.compile(
    r"Capabilities:\s*\[([0-9a-f]+)\s+v\d+\]\s+Virtual Channel",
    re.IGNORECASE,
)
# Low Priority EVC from top-level Port VC Capability Caps line
_RE_LPEVC = re.compile(r"LPEVC=(\d+)")
# Per-VC resource entry header (VC0:, VC1:, ...)
_RE_VC_ENTRY_HDR = re.compile(r"^\s+VC\d+:\s+Caps:", re.MULTILINE)
# Per-VC Resource Control Register fields on the "Ctrl:" line
_RE_VC_CTRL = re.compile(
    r"Ctrl:\s+Enable(\+|-)\s+ID=(\d+)\s+\S+\s+TC/VC=([0-9a-f]+)",
    re.IGNORECASE,
)
# Per-VC Resource Status Register: NegoPending field on the "Status:" line
_RE_NEGO = re.compile(r"NegoPending(\+|-)")

# PCI class code → short label for attachment filenames.
# Key: (base_class, subclass) from the 4-digit hex class code in lspci output
# e.g. [0200] → base=0x02, sub=0x00 → "ethernet".
# Falls back to base-class-only label when subclass is not listed.
_CLASS_LABEL: Dict[Tuple[int, int], str] = {
    (0x01, 0x01): "ide",
    (0x01, 0x04): "raid",
    (0x01, 0x06): "sata",
    (0x01, 0x08): "nvme",
    (0x02, 0x00): "ethernet",
    (0x02, 0x80): "network",
    (0x03, 0x00): "vga",
    (0x03, 0x02): "gpu",
    (0x04, 0x01): "audio",
    (0x04, 0x03): "hdaudio",
    (0x06, 0x00): "host_bridge",
    (0x06, 0x01): "isa_bridge",
    (0x06, 0x04): "pcie_bridge",
    (0x06, 0x09): "pci_bridge",
    (0x0B, 0x40): "coprocessor",
    (0x0C, 0x00): "firewire",
    (0x0C, 0x03): "usb",
    (0x0C, 0x07): "smbus",
    (0x0C, 0x0A): "can",
    (0x0D, 0x11): "bluetooth",
    (0x0D, 0x20): "wifi",
    (0x10, 0x00): "crypto",
    (0x12, 0x00): "accel",
}
_BASE_CLASS_LABEL: Dict[int, str] = {
    0x01: "storage",
    0x02: "network",
    0x03: "display",
    0x04: "multimedia",
    0x05: "memory_ctrl",
    0x06: "bridge",
    0x07: "comm",
    0x08: "system",
    0x09: "input",
    0x0B: "processor",
    0x0C: "serial",
    0x0D: "wireless",
    0x10: "crypto",
    0x12: "accel",
}


def _pci_class_label_from_hex(class_hex: str) -> str:
    """
    Return a short, filesystem-safe class label from a 4-digit hex class code.

    ``class_hex`` is the 4-character hex string appearing in lspci ``-n`` output
    as ``[CCSS]:``, e.g. ``"0200"`` → ``"ethernet"``, ``"0604"`` → ``"pcie_bridge"``.
    """
    try:
        code = int(class_hex, 16)
        base = (code >> 8) & 0xFF
        sub = code & 0xFF
    except (ValueError, TypeError):
        return ""
    label = _CLASS_LABEL.get((base, sub))
    if label:
        return label
    return _BASE_CLASS_LABEL.get(base, f"cls{base:02x}")


def _parse_lspci_vc_devices(text: str) -> List[Dict]:
    """
    Parse ``lspci -vvvnn`` output and return devices with VC extended capability.

    Each returned dict has the same structure as ``_scan_vc_devices()`` produces:
    ``bdf``, ``class_label``, ``vc_cap_offset``, ``evc_count``, ``lpevc_count``,
    ``vcs`` (list of per-VC dicts with ``vc_id``, ``tc_vc_map``, ``enabled``,
    ``negotiation_pending``).

    The Extended VC Count (``evc_count``) is derived from the number of VC
    resource entries found (one per VC including VC0), so
    ``evc_count = len(vcs) - 1``.  ``lpevc_count`` is read from the
    ``LPEVC=N`` field of the Port VC Capability ``Caps:`` line.
    """
    devices: List[Dict] = []

    # Split into per-device blocks at lines beginning with a BDF address.
    # lspci omits the 0000: domain prefix by default, so both formats must match:
    #   "83:00.0 Ethernet..."    (no domain)
    #   "0000:83:00.0 Ethernet..." (explicit domain)
    blocks = re.split(
        r"(?=^(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]\s)",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        first_line = block.split("\n")[0]

        # Extract BDF and 4-digit class code from the device header line.
        bdf_cls = _RE_BDF_CLASS.match(first_line)
        if not bdf_cls:
            continue
        # Normalise to sysfs format: prepend "0000:" if domain was omitted
        raw_bdf = bdf_cls.group(1).lower()
        bdf = raw_bdf if ":" in raw_bdf[4:5] else f"0000:{raw_bdf}"
        class_label = _pci_class_label_from_hex(bdf_cls.group(2))

        # Find the Virtual Channel capability section within this block.
        vc_match = _RE_VC_CAP.search(block)
        if not vc_match:
            continue

        cap_offset = f"0x{int(vc_match.group(1), 16):03x}"

        # Isolate VC capability text: from the VC header to the next
        # "Capabilities:" line (if any), so we only parse the VC section.
        vc_start = vc_match.start()
        vc_text = block[vc_start:]
        next_cap = re.search(r"\n\s+Capabilities:", vc_text[20:])
        if next_cap:
            vc_text = vc_text[: 20 + next_cap.start()]

        # LPEVC from the top-level Port VC Capability "Caps:" line.
        lpevc_m = _RE_LPEVC.search(vc_text)
        lpevc = int(lpevc_m.group(1)) if lpevc_m else 0

        # Parse each VC resource entry (VC0, VC1, ...) line-by-line.
        vcs: List[Dict] = []
        current_vc: Optional[Dict] = None
        for line in vc_text.split("\n"):
            # VC entry header:  "        VC0:    Caps:   ..."
            if _RE_VC_ENTRY_HDR.match(line):
                if current_vc is not None:
                    vcs.append(current_vc)
                current_vc = {
                    "vc_id": len(vcs),
                    "tc_vc_map": "0x00",
                    "enabled": False,
                    "negotiation_pending": False,
                }
                continue

            if current_vc is None:
                continue

            # Per-VC Resource Control Register:
            # "        Ctrl:   Enable+ ID=0 ArbSelect=Fixed TC/VC=ff"
            ctrl_m = _RE_VC_CTRL.search(line)
            if ctrl_m:
                current_vc["enabled"] = ctrl_m.group(1) == "+"
                current_vc["vc_id"] = int(ctrl_m.group(2))
                current_vc["tc_vc_map"] = f"0x{int(ctrl_m.group(3), 16):02x}"
                continue

            # Per-VC Resource Status Register:
            # "        Status: NegoPending- InProgress-"
            nego_m = _RE_NEGO.search(line)
            if nego_m and "Status:" in line:
                current_vc["negotiation_pending"] = nego_m.group(1) == "+"

        if current_vc is not None:
            vcs.append(current_vc)

        if not vcs:
            continue

        devices.append(
            {
                "bdf": bdf,
                "class_label": class_label,
                "vc_cap_offset": cap_offset,
                "evc_count": len(vcs) - 1,  # VCs beyond VC0
                "lpevc_count": lpevc,
                "vcs": vcs,
            }
        )

    return devices


def _scan_vc_devices() -> Tuple[List[Dict], int]:
    """
    Scan PCIe devices for the Virtual Channel extended capability.

    Reads the ``lspci -vvvnn`` dump at ``~/.esq/pci/lspci_verbose.txt``
    captured by ``system-setup.sh``.  That dump is produced
    as root, giving access to the full 4096-byte PCIe extended config space
    including extended capabilities beyond offset 0x100.

    Returns ``(devices, 0)`` where ``devices`` is a list of dicts for every
    device that has the VC extended capability, with fields:

    - ``bdf``           \u2014 PCI BDF string, e.g. ``"0000:83:00.0"``
    - ``vc_cap_offset`` \u2014 hex offset of VC cap, e.g. ``"0x148"``
    - ``evc_count``     \u2014 Extended VC Count (VCs beyond VC0; 0 = VC0-only)
    - ``lpevc_count``   \u2014 Low Priority Extended VC Count
    - ``class_label``   \u2014 short PCI class name, e.g. ``"ethernet"``, ``"pcie_bridge"``
    - ``vcs``           \u2014 list of per-VC dicts (vc_id, tc_vc_map, enabled,
                          negotiation_pending) for ALL VCs including VC0

    The second return value is always 0 (permission errors are not applicable
    to text-file parsing); absence of the dump file is reported via a debug
    log — the caller is responsible for surfacing a user-visible message.
    """
    if not _LSPCI_DUMP.exists():
        logger.debug("lspci dump not found at %s", _LSPCI_DUMP)
        return [], 0

    try:
        text = _LSPCI_DUMP.read_text(errors="replace")
    except (OSError, IOError) as e:
        logger.warning("Could not read lspci dump %s: %s", _LSPCI_DUMP, e)
        return [], 0

    devices = _parse_lspci_vc_devices(text)
    logger.debug("Parsed %d VC-capable device(s) from %s", len(devices), _LSPCI_DUMP)
    return devices, 0


@allure.title("PCI Multi-VC Detection")
def test_pci_multi_vc(
    request,
    configs,
    cached_result,
    cache_result,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """
    Detect PCIe devices with Multi-VC capability (TCC VC configuration).

    Reports ``pci_multi_vc`` (count of PCIe devices exposing the VC extended
    capability with Extended VC Count \u2265 1). On Intel TCC platforms with Multi
    Virtual Channel enabled, these additional VCs provide a dedicated real-time
    traffic queue for deterministic memory-access latency.  For each VC-capable
    device a separate CSV attachment listing every VC channel's state
    (VC ID, TC/VC map, Enable, NegoPending) is added to the Allure report.

    Requires the ``lspci -vvvnn`` dump generated by
    ``system-setup.sh`` (PCI Device Info Dump).
    The dump is stored under ``~/.esq/pci/`` with user-only permissions and
    is reusable by all ESQ tests needing PCI info.
    """
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info(f"Starting PCI Multi-VC Detection: {test_display_name}")

    validate_system_requirements_from_configs(configs)

    # Prerequisite: lspci dump must exist (generated by system-setup.sh).
    # Skip rather than fail — the dump is a setup step, not a test failure.
    if not _LSPCI_DUMP.exists():
        pytest.skip(
            f"PCI device info not available ({_LSPCI_DUMP} not found). "
            "Run system-setup.sh to generate it — refer to the installation guide."
        )

    result = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    def _run_detection():
        devices, permission_errors = _scan_vc_devices()

        # pci_multi_vc counts devices where TCC Multi-VC is supported (EVC >= 1).
        # VC0-only devices are reported in the CSV but not counted here.
        multi_vc_devices = [d for d in devices if d["evc_count"] >= 1]
        count = len(multi_vc_devices)

        # Count extended VCs that are actively enabled across multi-VC devices.
        # VC0 has vc_id=0; extended VCs have vc_id >= 1.
        enabled_ext_vcs = sum(1 for dev in multi_vc_devices for vc in dev["vcs"] if vc["vc_id"] > 0 and vc["enabled"])

        logger.info(
            f"PCI VC scan: {len(devices)} VC-capable device(s) found, "
            f"{count} with multi-VC (EVC>=1), "
            f"{enabled_ext_vcs} extended VC(s) enabled"
            + (f"; {permission_errors} device(s) skipped (permission denied)" if permission_errors else "")
        )

        # CSV attachments: one file per VC-capable device.
        # Each CSV lists the VC channels for that device.  The attachment name
        # uses the BDF with colons replaced by underscores, e.g.
        # pci_vc_0000_83_00.0.csv
        _CSV_FIELDS = [
            "vc_id",
            "tc_vc_map",
            "vc_enabled",
            "negotiation_pending",
        ]
        for dev in devices:
            csv_buf = io.StringIO()
            writer = csv.DictWriter(csv_buf, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for vc in dev["vcs"]:
                writer.writerow(
                    {
                        "vc_id": vc["vc_id"],
                        "tc_vc_map": vc["tc_vc_map"],
                        "vc_enabled": "yes" if vc["enabled"] else "no",
                        "negotiation_pending": "yes" if vc["negotiation_pending"] else "no",
                    }
                )
            bdf_safe = dev["bdf"].replace(":", "_")
            class_label = dev.get("class_label", "")
            attach_name = f"pci_vc_{bdf_safe}_{class_label}.csv" if class_label else f"pci_vc_{bdf_safe}.csv"
            allure.attach(
                csv_buf.getvalue(),
                name=attach_name,
                attachment_type=allure.attachment_type.CSV,
            )

        return Result(
            name=f"{test_id} - {test_display_name}",
            extended_metadata={
                "pci_multi_vc_devices": [d["bdf"] for d in multi_vc_devices],
                "pci_vc_capable_devices": [d["bdf"] for d in devices],
                "pci_multi_vc_details": multi_vc_devices,
                "pci_config_permission_errors": permission_errors,
            },
            metrics={
                "pci_multi_vc": Metrics(unit="devices", value=count, is_key_metric=True),
            },
            # Always True: 0 multi-VC devices is valid data (hardware without
            # TCC Multi-VC), not a test failure. Only False on fatal scan errors.
            metadata={"status": True},
        )

    try:
        result = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=_run_detection,
            test_name=test_name,
            configs=configs,
        )
    except KeyboardInterrupt:
        failure_message = "Interrupt detected during PCI Multi-VC Detection"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during PCI Multi-VC Detection: {e}"
        logger.error(failure_message, exc_info=True)

    if result is None:
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "PCI Multi-VC detection did not complete"},
            metrics={},
        )

    try:
        validate_test_results(
            test_name=test_name,
            results=result,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as validation_error:
        logger.error(f"Validation failed: {validation_error}")

    try:
        summarize_test_results(
            results=result,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as summary_error:
        logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)

    logger.info(f"PCI Multi-VC Detection completed: {test_display_name}")

    if test_interrupted:
        if configs.get("labels", {}).get("type") == "qualification":
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
