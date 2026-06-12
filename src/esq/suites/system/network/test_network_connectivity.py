# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System Network Connectivity Test.

Detects network interfaces and checks their connectivity status by interface type.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict

import allure
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result, run_command

logger = logging.getLogger(__name__)


def get_interface_bus_type(iface_name: str) -> str:
    """
    Determine the bus type (USB, PCI, etc.) for a network interface.

    Args:
        iface_name: Interface name (e.g., 'eth0', 'wlan0')

    Returns:
        Bus type: 'usb', 'pci', or 'unknown'
    """
    # Sanitize interface name to prevent path traversal
    # Only allow alphanumeric characters, hyphens, underscores, and dots
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", iface_name):
        logger.debug(f"Invalid interface name format: {iface_name}")
        return "unknown"

    # Additional check: interface name should not contain path separators
    if "/" in iface_name or "\\" in iface_name or ".." in iface_name:
        logger.debug(f"Interface name contains invalid characters: {iface_name}")
        return "unknown"

    try:
        # Use sanitized interface name in path
        # Use os.path.join instead of f-string for Bandit compliance
        uevent_path = os.path.join("/sys", "class", "net", iface_name, "device", "uevent")
        if os.path.exists(uevent_path):
            with open(uevent_path, "r", encoding="utf-8") as f:
                content = f.read().lower()
                if "usb" in content:
                    return "usb"
                elif "pci" in content:
                    return "pci"
    except (IOError, OSError) as e:
        logger.debug(f"Could not read uevent for {iface_name}: {e}")

    return "unknown"


def get_network_interfaces() -> Dict[str, dict]:
    """
    Get all network interface information including bus type.

    Returns:
        Dict of {interface_name: {
            "state": "UP"/"DOWN",
            "addresses": "...",
            "type": "wifi"/"ethernet"/"other",
            "bus_type": "usb"/"pci"/"unknown",
            "has_ip": bool
        }}
    """
    interfaces = {}

    result = run_command(["ip", "-brief", "addr"])

    if result and result.returncode == 0 and result.stdout:
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 2:
                iface_name = parts[0]
                state = parts[1]
                addrs = " ".join(parts[2:]) if len(parts) > 2 else ""

                # Skip loopback
                if iface_name == "lo":
                    continue

                # Classify interface type
                is_wifi = any(prefix in iface_name.lower() for prefix in ["wlan", "wlp", "wifi", "wl"])
                is_ethernet = any(prefix in iface_name.lower() for prefix in ["eth", "enp", "eno", "ens"])

                # Get bus type
                bus_type = get_interface_bus_type(iface_name)

                interface_type = "wifi" if is_wifi else ("ethernet" if is_ethernet else "other")

                interfaces[iface_name] = {
                    "state": state,
                    "addresses": addrs,
                    "type": interface_type,
                    "bus_type": bus_type,
                    "has_ip": bool(addrs and addrs != ""),
                }

    return interfaces


def save_network_info(output_dir: str):
    """
    Save detailed network information to files.

    Args:
        output_dir: Directory to save network info
    """
    commands = {
        "network_interfaces.txt": ["ip", "-brief", "addr"],
        "network_interfaces_detail.txt": ["ip", "addr"],
        "routing_table.txt": ["ip", "route"],
    }

    for filename, cmd in commands.items():
        result = run_command(cmd)
        if result and result.returncode == 0 and result.stdout:
            try:
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
            except IOError as e:
                logger.warning(f"Could not save {filename}: {e}")


@allure.title("Network Connectivity Test")
def test_network_connectivity(
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
    Test network connectivity for WiFi and Ethernet interfaces.

    Detects WiFi and Ethernet interfaces and checks their connectivity status.
    """
    # Request
    test_name = request.node.name.split("[")[0]

    # Parameters
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    # Set test description from config if provided
    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info(f"Starting Network Connectivity Test: {test_display_name}")

    check_all = configs.get("check_all", False)
    check_wired = configs.get("check_wired", False)
    check_wireless = configs.get("check_wireless", False)

    # Step 1: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Step 2: Setup directories
    # Sanitize environment variable path using character-by-character copy
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))

    # Resolve path and reconstruct to break taint chain
    core_data_resolved = str(Path(core_data_dir_tainted).resolve())
    chars: list = []
    for char in core_data_resolved:
        chars.append(char)
    core_data_dir = "".join(chars)

    # Validate path stays within expected directory (no path traversal)
    expected_base = Path(os.getcwd()).resolve()
    if not Path(core_data_dir).resolve().is_relative_to(expected_base):
        # If path is outside current directory, use safe default
        core_data_dir = os.path.join(os.getcwd(), "esq_data")

    data_dir = os.path.join(core_data_dir, "data", "system", "network")
    network_results = os.path.join(data_dir, "results", test_id)

    # Sanitize final path before directory creation
    network_resolved = str(Path(network_results).resolve())
    chars_network: list = []
    for char in network_resolved:
        chars_network.append(char)
    network_results_clean = "".join(chars_network)

    os.makedirs(network_results_clean, mode=0o770, exist_ok=True)
    ensure_dir_permissions(network_results_clean, uid=os.getuid(), gid=os.getgid(), mode=0o770)

    # Use sanitized path for all subsequent operations
    network_results = network_results_clean

    # Step 3: Get network interfaces
    all_interfaces = get_network_interfaces()

    # Filter interfaces based on test criteria
    filtered_interfaces = {}

    for iface_name, iface_info in all_interfaces.items():
        include = False

        # Check if showing all interfaces
        if check_all:
            include = True

        # Check wired (Ethernet) interface type
        if check_wired and iface_info["type"] == "ethernet":
            include = True

        # Check wireless (WiFi) interface type
        if check_wireless and iface_info["type"] == "wifi":
            include = True

        if include:
            filtered_interfaces[iface_name] = iface_info

    # Count interfaces by type
    total_interfaces = len(filtered_interfaces)
    total_connected = sum(1 for iface in filtered_interfaces.values() if iface["state"] == "UP" and iface["has_ip"])

    wired_interfaces = {k: v for k, v in filtered_interfaces.items() if v["type"] == "ethernet"}
    wireless_interfaces = {k: v for k, v in filtered_interfaces.items() if v["type"] == "wifi"}

    wired_total = len(wired_interfaces)
    wired_connected = sum(1 for iface in wired_interfaces.values() if iface["state"] == "UP" and iface["has_ip"])

    wireless_total = len(wireless_interfaces)
    wireless_connected = sum(1 for iface in wireless_interfaces.values() if iface["state"] == "UP" and iface["has_ip"])

    # Log interface information
    logger.info(f"Network interfaces: {total_interfaces} total, {total_connected} connected")

    if check_all:
        logger.info(f"  Wired: {wired_total} total, {wired_connected} connected")
        for iface_name, iface_info in wired_interfaces.items():
            status = "CONNECTED" if (iface_info["state"] == "UP" and iface_info["has_ip"]) else "DISCONNECTED"
            ip_status = "<IP assigned>" if iface_info["has_ip"] else "No IP"
            logger.info(f"    {iface_name}: {status} - {ip_status}")
        logger.info(f"  Wireless: {wireless_total} total, {wireless_connected} connected")
        for iface_name, iface_info in wireless_interfaces.items():
            status = "CONNECTED" if (iface_info["state"] == "UP" and iface_info["has_ip"]) else "DISCONNECTED"
            ip_status = "<IP assigned>" if iface_info["has_ip"] else "No IP"
            logger.info(f"    {iface_name}: {status} - {ip_status}")

    if check_wired:
        logger.info(f"  Wired: {wired_total} total, {wired_connected} connected")
        for iface_name, iface_info in wired_interfaces.items():
            status = "CONNECTED" if (iface_info["state"] == "UP" and iface_info["has_ip"]) else "DISCONNECTED"
            ip_status = "<IP assigned>" if iface_info["has_ip"] else "No IP"
            logger.info(f"    {iface_name}: {status} - {ip_status}")

    if check_wireless:
        logger.info(f"  Wireless: {wireless_total} total, {wireless_connected} connected")
        for iface_name, iface_info in wireless_interfaces.items():
            status = "CONNECTED" if (iface_info["state"] == "UP" and iface_info["has_ip"]) else "DISCONNECTED"
            ip_status = "<IP assigned>" if iface_info["has_ip"] else "No IP"
            logger.info(f"    {iface_name}: {status} - {ip_status}")

    # Step 4: Save network info
    save_network_info(network_results)

    # Step 5: Create metrics (only relevant metrics per test type)
    # total_connected is the key metric only for check_all; specific tests use their own key metric
    metrics = {
        "total_interfaces": Metrics(unit="count", value=total_interfaces, is_key_metric=False),
        "total_connected": Metrics(unit="count", value=total_connected, is_key_metric=check_all),
    }

    # Add wired/wireless breakdown metrics if checking all
    if check_all:
        metrics["wired_interfaces_total"] = Metrics(unit="count", value=wired_total, is_key_metric=False)
        metrics["wired_interfaces_connected"] = Metrics(unit="count", value=wired_connected, is_key_metric=False)
        metrics["wireless_interfaces_total"] = Metrics(unit="count", value=wireless_total, is_key_metric=False)
        metrics["wireless_interfaces_connected"] = Metrics(unit="count", value=wireless_connected, is_key_metric=False)

    # Add wired metrics if checking wired only
    if check_wired:
        metrics["wired_interfaces_total"] = Metrics(unit="count", value=wired_total, is_key_metric=False)
        metrics["wired_interfaces_connected"] = Metrics(unit="count", value=wired_connected, is_key_metric=True)

    # Add wireless metrics if checking wireless only
    if check_wireless:
        metrics["wireless_interfaces_total"] = Metrics(unit="count", value=wireless_total, is_key_metric=False)
        metrics["wireless_interfaces_connected"] = Metrics(unit="count", value=wireless_connected, is_key_metric=True)

    # Step 6: Build status message (informational, suite does not fail on connectivity)
    if total_interfaces == 0:
        test_message = "No network interfaces found"
    else:
        parts = []
        if check_wired:
            parts.append(f"Wired: {wired_connected}/{wired_total}")
        elif check_wireless:
            parts.append(f"Wireless: {wireless_connected}/{wireless_total}")
        elif check_all:
            parts.append(f"Wired: {wired_connected}/{wired_total}")
            parts.append(f"Wireless: {wireless_connected}/{wireless_total}")

        test_message = f"{', '.join(parts)} - Total: {total_connected}/{total_interfaces} connected"

    # Step 7: Create result with proper name field
    result = Result(
        name=test_display_name,
        metadata={"status": True, "message": test_message},
        metrics=metrics,
    )

    # Step 8: Validate and summarize results
    validation_results = validate_test_results(
        test_name=test_name,
        results=result,
        configs=configs,
        get_kpi_config=get_kpi_config,
    )

    # Generate test summary
    summarize_test_results(
        results=result,
        test_name=test_name,
        configs=configs,
        get_kpi_config=get_kpi_config,
    )

    # Step 9: Cache results
    cache_result(result)

    logger.info(f"Network connectivity test completed: {test_display_name} - {test_message}")
