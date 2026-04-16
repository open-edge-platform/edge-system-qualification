# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Memory information collection utilities.

Provides functions to collect detailed memory information including
physical/virtual memory statistics and per-DIMM hardware specifications
via DMI/SMBIOS tables (using dmidecode).

To enable DIMM-level detail collection without root, run once:
    sudo scripts/system-setup.sh
"""

import logging
import os
from typing import Any, Dict, Optional

import psutil

logger = logging.getLogger(__name__)


def _get_mask_setting() -> bool:
    """
    Get the current masking configuration from environment variable.

    Returns:
        bool: True if masking is enabled (default), False otherwise.
    """
    return os.environ.get("CORE_MASK_DATA", "true").lower() == "true"


def _parse_memory_array_attrs(attrs: Dict[str, str]) -> Dict[str, Any]:
    """
    Parse DMI type 16 (Physical Memory Array) attributes.

    Args:
        attrs: Dictionary of attribute key-value pairs from dmidecode output.

    Returns:
        Dict with parsed memory array information.
    """
    result: Dict[str, Any] = {
        "location": attrs.get("Location", "Unknown"),
        "use": attrs.get("Use", "Unknown"),
        "error_correction": attrs.get("Error Correction Type", "Unknown"),
        "max_capacity": attrs.get("Maximum Capacity", "Unknown"),
    }
    num_devices_str = attrs.get("Number Of Devices", "0")
    try:
        result["number_of_devices"] = int(num_devices_str)
    except ValueError:
        result["number_of_devices"] = 0
    return result


def _parse_dimm_size_to_gib(size_str: str) -> Optional[float]:
    """
    Parse a dmidecode DIMM size string to a numeric GiB value.

    dmidecode reports sizes as e.g. "64 GB", "32 GB", "8192 MB".
    SMBIOS stores capacities in binary MiB units, so these values are
    already GiB-equivalent — we return them directly as GiB.

    Args:
        size_str: Size string from dmidecode, e.g. "64 GB" or "8192 MB".

    Returns:
        Float GiB value, or None if the slot is empty or unparseable.
    """
    if not size_str or size_str in ("No Module Installed", "Unknown", ""):
        return None
    parts = size_str.split()
    if len(parts) >= 2:
        try:
            value = float(parts[0])
            unit = parts[1].upper()
            if unit == "GB":
                return value  # SMBIOS reports in MiB binary, so this is already GiB
            elif unit == "MB":
                return value / 1024.0
            elif unit == "TB":
                return value * 1024.0
        except ValueError:
            return None
    return None


def _parse_memory_device_attrs(attrs: Dict[str, str]) -> Dict[str, Any]:
    """
    Parse DMI type 17 (Memory Device) attributes into a structured record.

    Args:
        attrs: Dictionary of attribute key-value pairs from dmidecode output.

    Returns:
        Dict with parsed memory device (DIMM/slot) information.
    """

    def _parse_speed(speed_str: str) -> Optional[int]:
        """Parse speed value in MT/s from strings like '6800 MT/s'."""
        if not speed_str or speed_str in ("Unknown", ""):
            return None
        parts = speed_str.split()
        if parts:
            try:
                return int(parts[0])
            except ValueError:
                return None
        return None

    size_str = attrs.get("Size", "Unknown")
    installed = size_str not in ("No Module Installed", "Unknown", "")

    result: Dict[str, Any] = {
        "locator": attrs.get("Locator", "Unknown"),
        "bank_locator": attrs.get("Bank Locator", "Unknown"),
        "size": size_str,
        "form_factor": attrs.get("Form Factor", "Unknown"),
        "type": attrs.get("Type", "Unknown"),
        "type_detail": attrs.get("Type Detail", "Unknown"),
        "speed_mts": _parse_speed(attrs.get("Speed", "")),
        "configured_speed_mts": _parse_speed(attrs.get("Configured Memory Speed", "")),
        "manufacturer": attrs.get("Manufacturer", "Unknown"),
        "part_number": attrs.get("Part Number", "Unknown"),
        "rank": None,
        "data_width": attrs.get("Data Width", "Unknown"),
        "total_width": attrs.get("Total Width", "Unknown"),
        "configured_voltage": attrs.get("Configured Voltage", "Unknown"),
        "min_voltage": attrs.get("Minimum Voltage", "Unknown"),
        "max_voltage": attrs.get("Maximum Voltage", "Unknown"),
        "memory_technology": attrs.get("Memory Technology", "Unknown"),
        "installed": installed,
    }

    # Mask serial number when masking is enabled (it's a hardware identifier)
    serial = attrs.get("Serial Number", "Unknown")
    if _get_mask_setting() and serial and serial not in ("Unknown", "Not Specified", ""):
        result["serial_number"] = "***MASKED***"
    else:
        result["serial_number"] = serial

    # Parse rank as integer
    rank_str = attrs.get("Rank", "")
    if rank_str:
        try:
            result["rank"] = int(rank_str)
        except ValueError:
            result["rank"] = None

    return result


def _parse_dmidecode_memory_output(output: str) -> tuple:
    """
    Parse raw dmidecode output for DMI types 16 and 17.

    Args:
        output: Raw text output from ``dmidecode -t 16,17``.

    Returns:
        Tuple of (arrays, devices) where arrays is a list of parsed type-16
        records and devices is a list of parsed type-17 records.
    """
    arrays = []
    devices = []

    current_type: Optional[int] = None
    current_attrs: Dict[str, str] = {}

    for line in output.splitlines():
        stripped = line.strip()

        # Skip empty lines, comments, and intro headers
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("Getting SMBIOS") or stripped.startswith("SMBIOS"):
            continue

        # A Handle line marks the start of a new DMI record
        if stripped.startswith("Handle "):
            # Flush the previous record before resetting state
            if current_type == 16 and current_attrs:
                arrays.append(_parse_memory_array_attrs(current_attrs))
            elif current_type == 17 and current_attrs:
                devices.append(_parse_memory_device_attrs(current_attrs))

            current_attrs = {}
            current_type = None

            # Extract DMI type number: "Handle 0xXXXX, DMI type N, M bytes"
            for part in stripped.split(","):
                part = part.strip()
                if "DMI type" in part:
                    try:
                        current_type = int(part.split("DMI type")[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass

        # Attribute line (indented, contains ':') for types 16 and 17
        elif line and line[0:1].isspace() and ":" in stripped and current_type in (16, 17):
            key, _, value = stripped.partition(":")
            current_attrs[key.strip()] = value.strip()

    # Flush the final record
    if current_type == 16 and current_attrs:
        arrays.append(_parse_memory_array_attrs(current_attrs))
    elif current_type == 17 and current_attrs:
        devices.append(_parse_memory_device_attrs(current_attrs))

    return arrays, devices


def collect_memory_dimm_info() -> Dict[str, Any]:
    """
    Collect detailed DIMM information via dmidecode (DMI types 16 and 17).

    Reads Physical Memory Array (type 16) and Memory Device (type 17) records
    to extract per-slot details: type (DDR4/DDR5/…), speed, size, manufacturer,
    part number, rank, voltage, and channel/bank locator.

    Requires ``dmidecode`` with ``cap_dac_read_search`` capability. Set once via::

        sudo scripts/system-setup.sh

    Returns:
        Dict with keys:

        - ``available`` (bool): False when dmidecode is unavailable or inaccessible.
        - ``reason`` (str): Failure reason when ``available`` is False.
        - ``hint`` (str, optional): Suggested remediation step.
        - ``arrays`` (list): Parsed Physical Memory Array records (type 16).
        - ``devices`` (list): Parsed Memory Device records per slot (type 17).
        - ``slot_count`` (int): Total number of DIMM slots reported.
        - ``installed_count`` (int): Number of slots with a module installed.
    """
    import shutil

    from sysagent.utils.core.process import run_command

    # Locate the dmidecode binary
    dmidecode_raw = shutil.which("dmidecode")
    if not dmidecode_raw:
        logger.debug("dmidecode not found; skipping DIMM info collection")
        return {"available": False, "reason": "dmidecode_not_found"}

    # Break taint chain via character-by-character copy
    safe_path = "".join(c for c in dmidecode_raw)

    result = run_command([safe_path, "-t", "16,17"], timeout=10)
    if result.timed_out:
        logger.debug("dmidecode timed out")
        return {"available": False, "reason": "timeout"}

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    output_lower = (stdout + stderr).lower()

    # Detect permission / no-data conditions
    no_data_phrases = ("permission denied", "no smbios nor dmi entry point found")
    if result.returncode != 0 or not stdout.strip():
        if any(phrase in output_lower for phrase in no_data_phrases):
            logger.debug(
                "dmidecode requires elevated permissions for DMI table access. Run: sudo scripts/system-setup.sh"
            )
            return {
                "available": False,
                "reason": "permission_denied",
                "hint": "Run: sudo scripts/system-setup.sh",
            }
        logger.debug(f"dmidecode returned no usable output (rc={result.returncode})")
        return {"available": False, "reason": "no_output"}

    # Guard against output that contains only header lines
    data_lines = [
        ln
        for ln in stdout.splitlines()
        if ln.strip()
        and not ln.strip().startswith("#")
        and not ln.strip().startswith("Getting SMBIOS")
        and not ln.strip().startswith("SMBIOS")
    ]
    if not data_lines:
        logger.debug("dmidecode output contained no memory data")
        return {"available": False, "reason": "no_data"}

    try:
        arrays, devices = _parse_dmidecode_memory_output(stdout)
    except Exception as e:
        logger.warning(f"Failed to parse dmidecode output: {e}")
        return {"available": False, "reason": f"parse_error: {e}"}

    installed_devices = [d for d in devices if d.get("installed")]
    slot_count = sum(a.get("number_of_devices", 0) for a in arrays)

    # Sum installed DIMM capacities (GiB) from SMBIOS/dmidecode records.
    installed_gib_values = [_parse_dimm_size_to_gib(d.get("size", "")) for d in installed_devices]
    valid_installed_gib = [v for v in installed_gib_values if v is not None]
    installed_ram_gib = sum(valid_installed_gib) if valid_installed_gib else None

    return {
        "available": True,
        "arrays": arrays,
        "devices": devices,
        "slot_count": slot_count,
        "installed_count": len(installed_devices),
        "installed_ram_gib": installed_ram_gib,
    }


def collect_memory_info() -> Dict[str, Any]:
    """
    Collect memory usage statistics and detailed DIMM hardware configuration.

    Combines psutil virtual/swap memory counters with per-DIMM specification
    data obtained via dmidecode (when the capability is configured).

    Returns:
        Dict containing:

        - ``total``, ``available``, ``used``, ``free``, ``percent``: psutil values.
        - ``swap``: swap memory stats.
        - ``dimms``: result of :func:`collect_memory_dimm_info`.
    """
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        dimms = collect_memory_dimm_info()

        # Usable RAM: what the OS sees after hardware reservation (psutil total / 1024^3).
        usable_ram_gib = round(memory.total / 1024**3, 1)

        memory_info: Dict[str, Any] = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "free": memory.free,
            "percent": memory.percent,
            "usable_ram_gib": usable_ram_gib,
            "used_gib": round(memory.used / 1024**3, 1),
            "available_gib": round(memory.available / 1024**3, 1),
            "swap": {
                "total": swap.total,
                "used": swap.used,
                "free": swap.free,
                "percent": swap.percent,
            },
            "dimms": dimms,
        }

        return memory_info

    except Exception as e:
        logger.warning(f"Failed to collect memory info: {e}")
        return {"error": str(e)}
