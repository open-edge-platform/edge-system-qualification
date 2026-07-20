# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Real-time Platform — TSN (Time-Sensitive Networking) Detection.

Detects TSN-capable network interfaces by checking for the Intel ``igc``
driver. PTP hardware clocks are resolved to their PCI bus address via sysfs
symlinks, enabling per-interface clock association even when ethtool does not
report a bound hardware clock.

Detected capabilities
=====================

1. **TSN-capable interfaces** — Ethernet interfaces using the ``igc`` driver.
   The ``igc`` driver binds only to Intel NICs with native TSN hardware
   support (time-aware scheduling, hardware timestamping, PTP clock).

2. **PTP hardware clocks** — ``/sys/class/ptp/ptpX`` entries resolved to
   their PCI bus address for correlation with network interfaces.

Reported metrics
================
* ``tsn_capable_interfaces`` **(key metric)** — count of Ethernet interfaces
  using the ``igc`` driver; 0 when no TSN-capable NIC is present.

Extended metadata
=================
* ``tsn_capable_interfaces`` — list of igc-driver Ethernet interface names.
* ``ptp_clocks`` — dict mapping PTP clock name to PCI bus address,
  e.g. ``{"ptp0": "0000:84:00.0"}``.
* ``interface_details`` — per-interface dict for all non-loopback hardware
  interfaces; virtual/container interfaces (docker, veth, br-*) are excluded.
  Fields:

  - ``driver`` — kernel driver name (e.g. ``igc``, ``e1000e``, ``iwlwifi``).
  - ``bus_info`` — PCI bus address (e.g. ``0000:84:00.0``).
  - ``pci_vendor_id`` — PCI vendor ID hex string (e.g. ``0x8086``).
  - ``pci_device_id`` — PCI device ID hex string.
  - ``ptp_clock`` — associated PTP clock name (e.g. ``ptp0``), or
    ``"none"`` when no PTP clock resolves to this interface’s PCI address.
"""

import csv
import io
import logging
import re
from pathlib import Path
from typing import Dict, List

import allure
import pytest
from sysagent.utils.core import Metrics, Result, run_command

logger = logging.getLogger(__name__)


def _get_all_interfaces() -> List[str]:
    """Return all non-loopback network interface names."""
    interfaces: List[str] = []
    result = run_command(["ip", "-brief", "link"], timeout=10)
    if result and result.returncode == 0 and result.stdout:
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if not parts:
                continue
            iface = parts[0].split("@")[0]
            if iface == "lo":
                continue
            # Validate interface name (defence in depth for sysfs paths).
            if not re.match(r"^[A-Za-z0-9_\-.]+$", iface):
                continue
            interfaces.append(iface)
    return interfaces


def _is_ethernet_interface(iface: str) -> bool:
    """Return True when ``iface`` looks like a physical Ethernet interface."""
    return any(p in iface.lower() for p in ("eth", "enp", "eno", "ens", "enx"))


def _is_virtual_interface(iface: str) -> bool:
    """Return True when ``iface`` is a software-only virtual/container interface.

    Excludes Docker bridges (``docker0``), veth pairs (``veth*``), container
    bridge networks (``br-*``), and libvirt bridges (``virbr*``) from
    interface detail collection, as they are not hardware NICs.
    """
    return any(iface.startswith(p) for p in ("docker", "veth", "br-", "virbr", "cni"))


def _get_interface_driver_info(iface: str) -> Dict[str, str]:
    """
    Return driver and PCI identity information for a network interface.

    Uses ``ethtool -i`` for the kernel driver name and PCI bus address, and reads
    sysfs PCI ID files for vendor/device identification. Together these allow
    identifying the NIC hardware model (e.g. driver ``igc``, PCI
    ``0x8086:0x125c``) without requiring root or ``lspci``.

    Returns a dict with ``driver``, ``bus_info``, ``pci_vendor_id``, and
    ``pci_device_id``. All values default to ``"unknown"`` when unavailable.
    """
    info: Dict[str, str] = {
        "driver": "unknown",
        "bus_info": "unknown",
        "pci_vendor_id": "unknown",
        "pci_device_id": "unknown",
    }

    # ethtool -i: driver name and PCI bus address (e.g. 0000:56:00.0).
    result = run_command(["ethtool", "-i", iface], timeout=10)
    if result and result.returncode == 0 and result.stdout:
        for line in result.stdout.splitlines():
            if line.startswith("driver:"):
                info["driver"] = line.split(":", 1)[1].strip()
            elif line.startswith("bus-info:"):
                info["bus_info"] = line.split(":", 1)[1].strip()

    # Sysfs PCI ID files: hex strings e.g. "0x8086" / "0x125c".
    for field, sysfs_name in (("pci_vendor_id", "vendor"), ("pci_device_id", "device")):
        try:
            value = Path(f"/sys/class/net/{iface}/device/{sysfs_name}").read_text(encoding="utf-8").strip()
            if value:
                info[field] = value
        except (OSError, IOError):
            pass

    return info


def _get_ptp_clocks() -> Dict[str, str]:
    """Return PTP clock names mapped to their PCI bus address.

    Resolves each ``/sys/class/ptp/ptpX`` symlink to extract the PCI device
    address from the sysfs path (e.g. the path
    ``/sys/devices/pci.../0000:84:00.0/ptp/ptp0`` yields ``0000:84:00.0``).
    This enables correlating PTP clocks with network interfaces via
    ``bus_info`` even when ethtool does not report a bound hardware clock.

    Returns a dict mapping clock name to PCI bus address, e.g.
    ``{"ptp0": "0000:84:00.0"}``. The address is ``"unknown"`` when the sysfs
    path does not contain a recognisable PCI address.
    """
    clocks: Dict[str, str] = {}
    _pci_re = re.compile(r"\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]\b")
    try:
        ptp_sys = Path("/sys/class/ptp")
        if ptp_sys.exists():
            for p in sorted(ptp_sys.iterdir()):
                if not p.name.startswith("ptp"):
                    continue
                pci_addr = "unknown"
                try:
                    resolved = str(p.resolve())
                    ptp_idx = resolved.find("/ptp/")
                    if ptp_idx > 0:
                        matches = _pci_re.findall(resolved[:ptp_idx])
                        if matches:
                            pci_addr = matches[-1]
                except (OSError, IOError):
                    pass
                clocks[p.name] = pci_addr
    except (OSError, IOError) as e:
        logger.debug(f"Could not enumerate /sys/class/ptp: {e}")
    return clocks


@allure.title("TSN Detection")
def test_tsn(
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
    Detect Time-Sensitive Networking (TSN) capabilities.

    Reports ``tsn_capable_interfaces`` (count of Ethernet interfaces using
    the ``igc`` driver) as the key metric, alongside PTP clock count with
    per-interface PCI association.
    """
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info(f"Starting TSN Detection: {test_display_name}")

    # Step 1: Validate system requirements
    validate_system_requirements_from_configs(configs)

    is_qualification = configs.get("labels", {}).get("type") == "qualification"
    result = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    def _run_detection():
        all_interfaces = _get_all_interfaces()
        # Exclude software-only virtual/container interfaces from detail collection.
        iface_details: Dict[str, Dict[str, str]] = {
            iface: _get_interface_driver_info(iface) for iface in all_interfaces if not _is_virtual_interface(iface)
        }

        # PTP clocks: {clock_name: pci_bus_address} resolved from sysfs symlinks.
        ptp_clock_map = _get_ptp_clocks()

        # Reverse map: PCI address → clock name for per-interface annotation.
        pci_to_ptp: Dict[str, str] = {pci: clk for clk, pci in ptp_clock_map.items() if pci != "unknown"}

        # Annotate every interface with its associated PTP clock (if any).
        for details in iface_details.values():
            details["ptp_clock"] = pci_to_ptp.get(details["bus_info"], "none")

        # TSN-capable = Ethernet interfaces using the igc driver.
        igc_interfaces: List[str] = [
            iface
            for iface in iface_details
            if _is_ethernet_interface(iface) and iface_details[iface]["driver"] == "igc"
        ]

        tsn_iface_count = len(igc_interfaces)
        ptp_count = len(ptp_clock_map)
        is_tsn_capable = tsn_iface_count > 0

        logger.info(
            f"TSN detection: {'CAPABLE' if is_tsn_capable else 'NOT CAPABLE'} "
            f"(tsn_ifaces={tsn_iface_count}, ptp_clocks={ptp_count})"
        )

        # CSV attachment: one row per interface for easy PTP-to-interface correlation.
        csv_buf = io.StringIO()
        writer = csv.DictWriter(
            csv_buf,
            fieldnames=[
                "interface",
                "driver",
                "bus_info",
                "pci_vendor_id",
                "pci_device_id",
                "ptp_clock",
                "tsn_capable",
            ],
        )
        writer.writeheader()
        for iface, details in iface_details.items():
            writer.writerow(
                {
                    "interface": iface,
                    "driver": details["driver"],
                    "bus_info": details["bus_info"],
                    "pci_vendor_id": details["pci_vendor_id"],
                    "pci_device_id": details["pci_device_id"],
                    "ptp_clock": details["ptp_clock"],
                    "tsn_capable": "yes" if iface in igc_interfaces else "no",
                }
            )
        allure.attach(csv_buf.getvalue(), name="network_interfaces.csv", attachment_type=allure.attachment_type.CSV)

        return Result(
            name=f"{test_id} - {test_display_name}",
            extended_metadata={
                # igc-driver Ethernet interfaces = TSN-capable NICs.
                "tsn_capable_interfaces": igc_interfaces,
                # PTP clocks: {clock_name: pci_bus_address} for correlation.
                "ptp_clocks": ptp_clock_map,
                # Hardware interfaces (virtual/container excluded) with driver, PCI, and PTP annotation.
                "interface_details": iface_details,
            },
            metrics={
                # Count of igc-driver Ethernet interfaces (key metric).
                "tsn_capable_interfaces": Metrics(unit="interfaces", value=tsn_iface_count, is_key_metric=True),
            },
            metadata={"status": is_tsn_capable},
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
        failure_message = "Interrupt detected during TSN Detection"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during TSN Detection: {e}"
        logger.error(failure_message, exc_info=True)

    if result is None:
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "TSN detection did not complete"},
            metrics={},
        )

    # Step 2: KPI validation (only active when kpi_refs is set in profile)
    try:
        validate_test_results(
            test_name=test_name,
            results=result,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as validation_error:
        logger.error(f"Validation failed: {validation_error}")

    # Step 3: Summarize (always runs)
    try:
        summarize_test_results(
            results=result,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as summary_error:
        logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)

    # Caching is handled by execute_test_with_cache: it calls cache_result only when
    # result.metadata["status"] is not False. This means:
    #   tsn_capable_interfaces > 0 → status=True  → cached (TSN confirmed, stable result)
    #   tsn_capable_interfaces = 0 → status=False → not cached (hardware may be updated)
    # The explicit cache_result() call has been removed to avoid bypassing that guard.

    logger.info(f"TSN Detection completed: {test_display_name}")

    # Surface interrupts/errors as a proper pytest outcome.
    if test_interrupted:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
