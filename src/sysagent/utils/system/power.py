# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Power information collection utilities.

Provides functions to collect power configuration information from Linux
powercap interface (RAPL - Running Average Power Limit).

This module reads power information directly from sysfs files without requiring
external dependencies or root permissions (when properly configured).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Powercap sysfs base paths
POWERCAP_PATH = "/sys/class/powercap"
INTEL_RAPL_PATH = f"{POWERCAP_PATH}/intel-rapl"
INTEL_RAPL_MMIO_PATH = f"{POWERCAP_PATH}/intel-rapl-mmio"


def _format_energy(microjoules: int) -> str:
    """
    Convert energy from microjoules to human-readable format.

    Args:
        microjoules: Energy value in microjoules

    Returns:
        Formatted string with appropriate unit (uJ, mJ, J, kJ)
    """
    if microjoules < 1000:
        return f"{microjoules} uJ"
    elif microjoules < 1_000_000:
        return f"{microjoules / 1000:.2f} mJ"
    elif microjoules < 1_000_000_000:
        return f"{microjoules / 1_000_000:.2f} J"
    else:
        return f"{microjoules / 1_000_000_000:.2f} kJ"


def _format_power(microwatts: int) -> str:
    """
    Convert power from microwatts to human-readable format.

    Args:
        microwatts: Power value in microwatts

    Returns:
        Formatted string with appropriate unit (uW, mW, W)
    """
    if microwatts < 1000:
        return f"{microwatts} uW"
    elif microwatts < 1_000_000:
        return f"{microwatts / 1000:.2f} mW"
    else:
        return f"{microwatts / 1_000_000:.2f} W"


def _format_time(microseconds: int) -> str:
    """
    Convert time from microseconds to human-readable format.

    Args:
        microseconds: Time value in microseconds

    Returns:
        Formatted string with appropriate unit (us, ms, s)
    """
    if microseconds < 1000:
        return f"{microseconds} us"
    elif microseconds < 1_000_000:
        return f"{microseconds / 1000:.2f} ms"
    else:
        return f"{microseconds / 1_000_000:.2f} s"


def _read_sysfs_file(filepath: str, default: Any = None) -> Optional[str]:
    """
    Read a sysfs file safely.

    Args:
        filepath: Path to sysfs file
        default: Default value to return on error

    Returns:
        File contents as string, or default value on error
    """
    try:
        with open(filepath, "r") as f:
            return f.read().strip()
    except (PermissionError, FileNotFoundError, IOError) as e:
        logger.debug(f"Cannot read {filepath}: {e}")
        return default


def _parse_zone_path(zone_path: str) -> Dict[str, Any]:
    """
    Parse a powercap zone directory and extract all available information.

    Args:
        zone_path: Path to zone directory (e.g., /sys/class/powercap/intel-rapl/intel-rapl:0)

    Returns:
        Dict containing zone information
    """
    zone_info = {}

    # Basic zone information
    zone_name = _read_sysfs_file(f"{zone_path}/name")
    if zone_name:
        zone_info["name"] = zone_name

    enabled = _read_sysfs_file(f"{zone_path}/enabled")
    if enabled is not None:
        zone_info["enabled"] = int(enabled)

    # Energy information
    max_energy_range = _read_sysfs_file(f"{zone_path}/max_energy_range_uj")
    if max_energy_range is not None:
        zone_info["max_energy_range_uj"] = int(max_energy_range)
        zone_info["max_energy_range_formatted"] = _format_energy(int(max_energy_range))

    energy = _read_sysfs_file(f"{zone_path}/energy_uj")
    if energy is not None:
        zone_info["energy_uj"] = int(energy)
        zone_info["energy_formatted"] = _format_energy(int(energy))

    # Power information (if available)
    max_power_range = _read_sysfs_file(f"{zone_path}/max_power_range_uw")
    if max_power_range is not None:
        zone_info["max_power_range_uw"] = int(max_power_range)
        zone_info["max_power_range_formatted"] = _format_power(int(max_power_range))

    # Constraints
    constraints = []
    constraint_idx = 0
    while True:
        constraint_name = _read_sysfs_file(f"{zone_path}/constraint_{constraint_idx}_name")
        if constraint_name is None:
            break

        constraint = {"name": constraint_name}

        power_limit = _read_sysfs_file(f"{zone_path}/constraint_{constraint_idx}_power_limit_uw")
        if power_limit is not None:
            constraint["power_limit_uw"] = int(power_limit)
            constraint["power_limit_formatted"] = _format_power(int(power_limit))

        time_window = _read_sysfs_file(f"{zone_path}/constraint_{constraint_idx}_time_window_us")
        if time_window is not None:
            constraint["time_window_us"] = int(time_window)
            constraint["time_window_formatted"] = _format_time(int(time_window))

        max_power = _read_sysfs_file(f"{zone_path}/constraint_{constraint_idx}_max_power_uw")
        if max_power is not None:
            constraint["max_power_uw"] = int(max_power)
            constraint["max_power_formatted"] = _format_power(int(max_power))

        constraints.append(constraint)
        constraint_idx += 1

    if constraints:
        zone_info["constraints"] = constraints

    # Recursively parse subzones
    subzones = []
    try:
        zone_dir = Path(zone_path)
        for subzone_path in sorted(zone_dir.glob("intel-rapl:*:*")):
            subzone_info = _parse_zone_path(str(subzone_path))
            if subzone_info:
                subzones.append(subzone_info)
    except Exception as e:
        logger.debug(f"Error scanning subzones in {zone_path}: {e}")

    if subzones:
        zone_info["subzones"] = subzones

    return zone_info


def _collect_rapl_control_type(base_path: str, control_type_name: str) -> Optional[Dict[str, Any]]:
    """
    Collect information for a RAPL control type (intel-rapl or intel-rapl-mmio).

    Args:
        base_path: Base path to control type directory
        control_type_name: Name of control type

    Returns:
        Dict containing control type information, or None if not available
    """
    if not os.path.exists(base_path):
        logger.debug(f"Control type {control_type_name} not available at {base_path}")
        return None

    try:
        control_type_info = {"name": control_type_name}

        # Check if enabled
        enabled = _read_sysfs_file(f"{base_path}/enabled")
        if enabled is not None:
            control_type_info["enabled"] = int(enabled)

        # Collect all zones
        zones = []
        base_dir = Path(base_path)
        for zone_path in sorted(base_dir.glob("intel-rapl:[0-9]*")):
            # Only match top-level zones (e.g., intel-rapl:0, not intel-rapl:0:0)
            if zone_path.name.count(":") == 1:
                zone_info = _parse_zone_path(str(zone_path))
                if zone_info:
                    zones.append(zone_info)

        if zones:
            control_type_info["zones"] = zones

        return control_type_info if zones else None

    except Exception as e:
        logger.warning(f"Error collecting {control_type_name} info: {e}")
        return None


def collect_power_info() -> Dict[str, Any]:
    """
    Collect platform power configuration information from Linux powercap interface.

    Reads RAPL (Running Average Power Limit) information from sysfs without
    requiring external dependencies. Supports both intel-rapl and intel-rapl-mmio
    control types.

    Returns:
        Dict containing power configuration information:
        {
            "control_types": [
                {
                    "name": "intel-rapl",
                    "enabled": 1,
                    "zones": [...]
                },
                ...
            ],
            "available": bool,
            "permission_issue": bool
        }
    """
    power_info = {"control_types": [], "available": False, "permission_issue": False}

    # Check if powercap is available
    if not os.path.exists(POWERCAP_PATH):
        logger.debug(f"Powercap interface not available at {POWERCAP_PATH}")
        return power_info

    power_info["available"] = True

    # Try to read intel-rapl-mmio first (if available)
    rapl_mmio_info = _collect_rapl_control_type(INTEL_RAPL_MMIO_PATH, "intel-rapl-mmio")
    if rapl_mmio_info:
        power_info["control_types"].append(rapl_mmio_info)

    # Read intel-rapl
    rapl_info = _collect_rapl_control_type(INTEL_RAPL_PATH, "intel-rapl")
    if rapl_info:
        power_info["control_types"].append(rapl_info)

    # Check for permission issues
    if power_info["available"] and not power_info["control_types"]:
        # Powercap exists but we couldn't read any zones - likely a permission issue
        test_file = f"{INTEL_RAPL_PATH}/intel-rapl:0/energy_uj"
        if os.path.exists(test_file):
            try:
                with open(test_file, "r") as f:
                    f.read()
            except PermissionError:
                power_info["permission_issue"] = True
                logger.warning(
                    "Permission denied reading powercap files. See documentation for setting up non-root access."
                )

    return power_info
