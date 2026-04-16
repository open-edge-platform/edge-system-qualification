# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System information formatting utilities.

Provides functions to format system information into human-readable reports
and summaries for display and debugging purposes.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def format_system_summary(hardware_info: Dict[str, Any], software_info: Dict[str, Any]) -> str:
    """
    Format consolidated system summary for both info command and test summaries.

    Args:
        hardware_info: Dictionary containing hardware information
        software_info: Dictionary containing software information

    Returns:
        Formatted system summary string
    """
    lines = []
    lines.append("SYSTEM INFORMATION")
    lines.append("-" * 40)

    # DMI System Information (Vendor, Product, Motherboard)
    dmi = hardware_info.get("dmi", {})
    if dmi:
        # System vendor and product
        system_dmi = dmi.get("system", {})
        if system_dmi.get("vendor") and system_dmi.get("product_name"):
            lines.append(f"System: {system_dmi['vendor']} {system_dmi['product_name']}")

        # Motherboard
        motherboard_dmi = dmi.get("motherboard", {})
        if motherboard_dmi.get("vendor") and motherboard_dmi.get("name"):
            lines.append(f"Motherboard: {motherboard_dmi['vendor']} {motherboard_dmi['name']}")
            if motherboard_dmi.get("version"):
                lines.append(f"  Version: {motherboard_dmi['version']}")

        # BIOS
        bios_dmi = dmi.get("bios", {})
        if bios_dmi.get("vendor") and bios_dmi.get("version"):
            bios_line = f"BIOS: {bios_dmi['vendor']} {bios_dmi['version']}"
            if bios_dmi.get("date"):
                bios_line += f" ({bios_dmi['date']})"
            lines.append(bios_line)

    # CPU Information
    cpu = hardware_info.get("cpu", {})
    if cpu.get("brand"):
        lines.append(f"CPU: {cpu['brand']}")
        # Add generation info (detailed multi-line format)
        gen_info = cpu.get("generation_info", {})
        if gen_info:
            codename = gen_info.get("codename", "Unknown")
            generation = gen_info.get("generation", "Unknown")
            product_collection = gen_info.get("product_collection", "Unknown")
            segment = gen_info.get("segment", "Unknown")
            is_supported = gen_info.get("is_supported", False)
            support_status = "Supported" if is_supported else "Not Supported"

            # Display each generation detail on separate line for clarity
            lines.append(f"  Codename: {codename}")
            lines.append(f"  Generation: {generation}")
            lines.append(f"  Product Collection: {product_collection}")
            lines.append(f"  Segment: {segment}")
            lines.append(f"  Support Status: {support_status}")
        if cpu.get("logical_cores") and cpu["logical_cores"] != "Unknown":
            lines.append(f"  Logical Cores: {cpu['logical_cores']}")
        if cpu.get("frequency_mhz") and cpu["frequency_mhz"] != "Unknown":
            freq_ghz = round(cpu["frequency_mhz"] / 1000, 2)
            lines.append(f"  Frequency: {freq_ghz} GHz")

    # Memory Information
    memory = hardware_info.get("memory", {})
    if memory.get("total_gib") and memory["total_gib"] != "Unknown":
        # Build the memory headline.
        # installed_ram_gib: sum of DIMM capacities from SMBIOS/dmidecode (GiB).
        # usable_ram_gib: psutil total / 1024^3 — what the OS sees (GiB).
        installed_ram_gib = memory.get("installed_ram_gib")
        usable_ram_gib = memory.get("usable_ram_gib")
        if installed_ram_gib is not None and usable_ram_gib is not None:
            installed_str = f"{installed_ram_gib:.1f} GB"
            usable_str = f"{usable_ram_gib:.1f} GB"
            lines.append(f"Memory: {installed_str} Installed ({usable_str} usable)")
        elif installed_ram_gib is not None:
            installed_str = f"{installed_ram_gib:.1f} GB"
            lines.append(f"Memory: {installed_str} Installed")
        else:
            lines.append(f"Memory: {memory['total_gib']} GB total")

        used_gib = memory.get("used_gib")
        available_gib = memory.get("available_gib")
        used_percent = memory.get("used_percent")
        if used_gib is not None and available_gib is not None and used_percent is not None:
            lines.append(f"  Usage: {used_gib} GB used ({used_percent}%) / {available_gib} GB available")
        elif used_percent and used_percent != "Unknown":
            lines.append(f"  Usage: {used_percent}%")

        # Show detailed DIMM configuration if available
        dimms = memory.get("dimms", {})
        if dimms.get("available"):
            arrays = dimms.get("arrays", [])
            if arrays:
                arr = arrays[0]
                if arr.get("error_correction") and arr["error_correction"] not in ("Unknown", "None"):
                    lines.append(f"  ECC: {arr['error_correction']}")

            installed_count = dimms.get("installed_count", 0)
            slot_count = dimms.get("slot_count", 0)
            if slot_count > 0:
                lines.append(f"  Slots: {installed_count}/{slot_count} populated")

            for device in dimms.get("devices", []):
                locator = device.get("locator", "")
                if not device.get("installed"):
                    lines.append(f"  [{locator}] (empty)")
                    continue
                size = device.get("size", "")
                mem_type = device.get("type", "")
                configured_speed = device.get("configured_speed_mts")
                speed = configured_speed or device.get("speed_mts")
                manufacturer = device.get("manufacturer", "")
                part_number = device.get("part_number", "")

                size_display = size
                slot_line = f"  [{locator}] {size_display}"
                if mem_type and mem_type != "Unknown":
                    slot_line += f" {mem_type}"
                if speed:
                    slot_line += f" @ {speed} MT/s"
                if manufacturer and manufacturer not in ("Unknown", "Not Specified"):
                    slot_line += f" - {manufacturer}"
                if part_number and part_number not in ("Unknown", "Not Specified"):
                    slot_line += f" {part_number}"
                lines.append(slot_line)
        elif dimms.get("reason") == "permission_denied":
            lines.append("  DIMM details unavailable (run setup: sudo scripts/system-setup.sh)")

    # GPU Information
    gpu = hardware_info.get("gpu", {})
    if gpu.get("device_count") and gpu["device_count"] > 0:
        lines.append(f"GPU: {gpu['device_count']} device(s)")
        devices = gpu.get("devices", [])
        for i, device in enumerate(devices):  # Show all GPU devices
            name = device.get("full_name", "Unknown")
            memory_gib = device.get("memory_gib")
            is_discrete = device.get("is_discrete", True)
            if memory_gib:
                # iGPU memory is GPU-accessible shared system RAM (CL_DEVICE_GLOBAL_MEM_SIZE),
                # not dedicated VRAM. Label it "(shared)" to avoid confusion with dGPU VRAM.
                mem_label = "GB" if is_discrete else "GB (shared)"
                mem_str = f" - {memory_gib:.1f} {mem_label}"
            else:
                mem_str = ""
            lines.append(f"  {i + 1}. {name}{mem_str}")

    # NPU Information
    npu = hardware_info.get("npu", {})
    if npu.get("device_count") and npu["device_count"] > 0:
        lines.append(f"NPU: {npu['device_count']} device(s)")
        devices = npu.get("devices", [])
        for i, device in enumerate(devices[:1]):  # Show first device
            name = device.get("full_name", "Unknown")
            lines.append(f"  {i + 1}. {name}")

    # Storage Information
    storage = hardware_info.get("storage", {})
    if storage.get("device_count") and storage["device_count"] > 0:
        lines.append(f"Storage: {storage['device_count']} device(s)")
        devices = storage.get("devices", [])
        for i, device in enumerate(devices):  # Show all storage devices
            name = device.get("model", "Unknown")
            interface = device.get("interface", "")
            size = device.get("size_gib", "")

            device_line = f"  {i + 1}. {name}"
            if interface:
                device_line += f" ({interface})"
            if size:
                device_line += f" - {size} GB"
            lines.append(device_line)

    # OS Information
    os_info = software_info.get("os", {})
    if os_info.get("name"):
        os_name = os_info.get("name", "Unknown")
        version = os_info.get("version", "")

        os_line = f"OS: {os_name}"
        if version and version != "Unknown":
            os_line += f" {version}"

        if os_info.get("pretty_name"):
            os_pretty = os_info["pretty_name"]
            os_line = f"OS: {os_pretty}"

        lines.append(os_line)

        if os_info.get("release") and os_info["release"] != "Unknown":
            lines.append(f"  Kernel: {os_info['release']}")

    # Power Information
    power = hardware_info.get("power", {})
    if power.get("available") and power.get("control_types"):
        # Zone metadata (name, enabled, max_energy_range, constraints) is always readable.
        # Only energy_uj (live energy counter) requires additional setup.
        if power.get("permission_issue"):
            lines.append("Power: RAPL power monitoring available (energy readings require setup)")
        else:
            lines.append("Power: RAPL power monitoring available")

        # Show all zones with constraints
        for control_type in power.get("control_types", []):
            zones = control_type.get("zones", [])

            # Show all zones, not just the first one
            for zone in zones:
                zone_name = zone.get("name", "Unknown")

                # Count subzones for this zone
                subzones = zone.get("subzones", [])
                subzone_count = len(subzones)

                # Build zone info line with subzone count if available
                zone_info_line = f"  {zone_name}"
                if subzone_count > 0:
                    zone_info_line += f" ({subzone_count} subzone{'s' if subzone_count != 1 else ''})"

                # Show energy if available
                energy_formatted = zone.get("energy_formatted")
                max_energy_formatted = zone.get("max_energy_range_formatted")

                if energy_formatted and max_energy_formatted:
                    energy_uj = zone.get("energy_uj", 0)
                    max_energy_uj = zone.get("max_energy_range_uj", 1)
                    percentage = (energy_uj / max_energy_uj * 100) if max_energy_uj > 0 else 0
                    zone_info_line += (
                        f" - Energy meter: {energy_formatted} / {max_energy_formatted} ({percentage:.1f}%)"
                    )
                elif energy_formatted:
                    zone_info_line += f" - Energy meter: {energy_formatted}"

                lines.append(zone_info_line)

                # Show all power constraints dynamically
                constraints = zone.get("constraints", [])
                for constraint in constraints:
                    power_limit = constraint.get("power_limit_formatted")
                    power_limit_uw = constraint.get("power_limit_uw", 0)
                    max_power = constraint.get("max_power_formatted")
                    max_power_uw = constraint.get("max_power_uw", 0)
                    constraint_name = constraint.get("name", "")
                    time_window = constraint.get("time_window_formatted")

                    if power_limit:
                        constraint_info = f"    Power Limit ({constraint_name}): {power_limit}"

                        # Show max power and utilization percentage if available
                        # Note: power_limit can exceed max_power on systems with overclocking support
                        if max_power and max_power_uw > 0:
                            utilization_pct = (power_limit_uw / max_power_uw * 100) if max_power_uw > 0 else 0
                            constraint_info += f" / {max_power} ({utilization_pct:.1f}%)"

                        # Show time window if available
                        if time_window:
                            constraint_info += f" (window: {time_window})"

                        lines.append(constraint_info)

        if power.get("permission_issue"):
            lines.append("  (Live energy readings unavailable - run: sudo scripts/system-setup.sh)")
    elif power.get("available") and power.get("permission_issue"):
        # Powercap exists but zone directories were unreadable entirely
        lines.append("Power: Available (setup required for non-root access)")

    # Network Information
    network = hardware_info.get("network", {})
    if network:
        internet_connected = network.get("internet_connected", False)
        restricted_access = network.get("restricted_access", False)
        restriction_details = network.get("restriction_details", {})

        if internet_connected:
            if restricted_access:
                lines.append("Network: Internet connected (Restricted)")
                if restriction_details:
                    # Derive blocked services from accessibility flags
                    blocked_services = [
                        service.replace("_accessible", "").capitalize()
                        for service, accessible in restriction_details.items()
                        if service.endswith("_accessible") and not accessible
                    ]
                    if blocked_services:
                        lines.append(f"  Blocked: {', '.join(blocked_services)}")
            else:
                lines.append("Network: Internet connected (Open)")
        else:
            lines.append("Network: No internet connection")

    lines.append("")  # Empty line after system info
    return "\n".join(lines)


def build_display_summary(hardware: Dict[str, Any], software: Dict[str, Any]) -> tuple:
    """
    Convert raw hardware/software dicts from SystemInfoCache into display-ready
    summary dicts for use with format_system_summary().

    This is the single source of truth for converting system info to display format,
    used by both generate_simple_report() and the test summary generator.

    Args:
        hardware: Raw hardware info dict from SystemInfoCache
        software: Raw software info dict from SystemInfoCache

    Returns:
        Tuple of (summary_hardware, summary_software) ready for format_system_summary()
    """
    summary_hardware = {}
    summary_software = {}

    # Convert CPU info
    if "cpu" in hardware:
        cpu = hardware["cpu"]
        summary_hardware["cpu"] = {
            "brand": cpu.get("brand", "Unknown"),
            "logical_cores": cpu.get("logical_count", cpu.get("count", "Unknown")),
            "frequency_mhz": cpu.get("frequency", {}).get("max", 0),
            "generation_info": cpu.get("generation_info", {}),
        }

    # Convert Memory info
    if "memory" in hardware:
        memory = hardware["memory"]
        dimms = memory.get("dimms", {})
        summary_hardware["memory"] = {
            "total_gib": round(memory.get("total", 0) / (1024**3)),
            "used_percent": round(memory.get("percent", 0), 1),
            "used_gib": memory.get("used_gib", round(memory.get("used", 0) / (1024**3), 1)),
            "available_gib": memory.get("available_gib", round(memory.get("available", 0) / (1024**3), 1)),
            "usable_ram_gib": memory.get("usable_ram_gib"),
            "installed_ram_gib": dimms.get("installed_ram_gib") if dimms.get("available") else None,
            "dimms": dimms,
        }

    # Convert GPU info
    if "gpu" in hardware:
        gpu = hardware["gpu"]
        summary_hardware["gpu"] = {
            "device_count": gpu.get("total_count", 0),
            "devices": [],
        }
        for device in gpu.get("devices", []):
            device_summary = {
                "full_name": device.get("device_name", "Unknown"),
                "is_discrete": device.get("is_discrete", False),
            }
            # Check for OpenVINO enhanced name and VRAM size
            if "openvino" in device:
                ov = device["openvino"]
                full_name = ov.get("full_device_name") or ov.get("quick_access", {}).get("full_name")
                if full_name and full_name != "Unknown":
                    device_summary["full_name"] = full_name
                memory_gib = ov.get("memory_gib")
                if memory_gib:
                    device_summary["memory_gib"] = round(memory_gib, 1)
            summary_hardware["gpu"]["devices"].append(device_summary)

    # Convert NPU info
    if "npu" in hardware:
        npu = hardware["npu"]
        summary_hardware["npu"] = {
            "device_count": npu.get("count", 0),
            "devices": [],
        }
        for device in npu.get("devices", []):
            device_summary = {"full_name": device.get("device_name", "Unknown")}
            # Check for OpenVINO enhanced name
            if "openvino" in device and "quick_access" in device["openvino"]:
                device_summary["full_name"] = device["openvino"]["quick_access"].get(
                    "full_name", device_summary["full_name"]
                )
            summary_hardware["npu"]["devices"].append(device_summary)

    # Convert Storage info
    if "storage" in hardware:
        storage = hardware["storage"]
        summary_hardware["storage"] = {
            "device_count": len(storage.get("devices", [])),
            "devices": [],
        }
        for device in storage.get("devices", []):
            device_summary = {
                "model": device.get("model", "Unknown"),
                "interface": device.get("interface", ""),
                "size_gib": round(device.get("size", 0) / (1024**3), 1) if device.get("size") else 0,
            }
            summary_hardware["storage"]["devices"].append(device_summary)

    # Convert DMI info (pass through as-is)
    if "dmi" in hardware:
        summary_hardware["dmi"] = hardware["dmi"]

    # Convert Power info (pass through as-is)
    if "power" in hardware:
        summary_hardware["power"] = hardware["power"]

    # Convert Network info (pass through as-is)
    if "network" in hardware:
        summary_hardware["network"] = hardware["network"]

    # Convert OS info
    if "os" in software:
        os_info = software["os"]
        dist = os_info.get("distribution", {})
        os_summary = {
            "name": dist.get("name", "Unknown") if isinstance(dist, dict) else str(dist),
            "version": dist.get("version_id", "") if isinstance(dist, dict) else "",
            "release": os_info.get("release", ""),
        }
        if isinstance(dist, dict) and dist.get("pretty_name"):
            os_summary["pretty_name"] = dist["pretty_name"]
        summary_software["os"] = os_summary

    return summary_hardware, summary_software


def generate_simple_report(system_info: Dict[str, Any]) -> str:
    """
    Generate a simple text report of system information.

    Args:
        system_info: Dictionary containing system information

    Returns:
        Formatted text report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SYSTEM INFORMATION REPORT")
    report_lines.append("=" * 80)

    # Get hardware and software data
    hardware = system_info.get("hardware", {})
    software = system_info.get("software", {})

    if hardware and software:
        # Convert raw hardware/software to summary format for unified display
        try:
            summary_hardware, summary_software = build_display_summary(hardware, software)
            system_summary_text = format_system_summary(summary_hardware, summary_software)
            report_lines.append("\n" + system_summary_text)

        except Exception as e:
            # Fallback to basic info if conversion fails
            report_lines.append(f"\nError processing system info: {e}")
            report_lines.append("Basic hardware/software information available in raw format.")

    # Add Python packages information if available
    if software:
        python_packages = software.get("python_packages", {})
        if python_packages:
            packages = python_packages.get("packages", {})
            # Show only installed packages (those with versions)
            installed_packages = {k: v for k, v in packages.items() if v}
            if installed_packages:
                report_lines.append("PYTHON PACKAGES:")
                report_lines.append("-" * 40)
                for pkg_name, version in list(installed_packages.items())[:10]:  # Show first 10
                    report_lines.append(f"{pkg_name}: {version}")

                total_installed = python_packages.get("total_installed", len(installed_packages))
                if len(installed_packages) > 10:
                    report_lines.append(f"... and {total_installed - 10} more packages")
                else:
                    report_lines.append(f"Total: {total_installed} packages")

    report_lines.append("\n" + "=" * 80)

    return "\n".join(report_lines)


def format_hardware_summary(hardware_info: Dict[str, Any]) -> str:
    """
    Format hardware information into a concise summary.

    Args:
        hardware_info: Dictionary containing hardware information

    Returns:
        Formatted hardware summary
    """
    summary_parts = []

    # CPU
    cpu = hardware_info.get("cpu", {})
    if cpu and not cpu.get("error"):
        brand = cpu.get("brand", "Unknown CPU")
        cores = cpu.get("count", "?")
        summary_parts.append(f"CPU: {brand} ({cores} cores)")

    # GPU
    gpu = hardware_info.get("gpu", {})
    if gpu and not gpu.get("error"):
        total = gpu.get("total_count", 0)
        discrete = gpu.get("discrete_count", 0)
        summary_parts.append(f"GPU: {total} total ({discrete} discrete)")

    # Memory
    memory = hardware_info.get("memory", {})
    if memory and not memory.get("error"):
        total_gib = memory.get("total", 0) / (1024**3)
        summary_parts.append(f"RAM: {total_gib:.1f} GB")

    return " | ".join(summary_parts) if summary_parts else "Hardware information unavailable"


def format_software_summary(software_info: Dict[str, Any]) -> str:
    """
    Format software information into a concise summary.

    Args:
        software_info: Dictionary containing software information

    Returns:
        Formatted software summary
    """
    summary_parts = []

    # OS
    os_info = software_info.get("os", {})
    if os_info and not os_info.get("error"):
        os_name = os_info.get("name", "Unknown")
        os_release = os_info.get("release", "")
        if os_name.lower() == "linux":
            dist = os_info.get("distribution", {})
            if dist and not dist.get("error"):
                dist_name = dist.get("name", os_name)
                summary_parts.append(f"OS: {dist_name} {os_release}")
            else:
                summary_parts.append(f"OS: {os_name} {os_release}")
        else:
            summary_parts.append(f"OS: {os_name} {os_release}")

    # Python
    python = software_info.get("python", {})
    if python and not python.get("error"):
        version_info = python.get("version_info", {})
        major = version_info.get("major", "?")
        minor = version_info.get("minor", "?")
        summary_parts.append(f"Python: {major}.{minor}")

    return " | ".join(summary_parts) if summary_parts else "Software information unavailable"
