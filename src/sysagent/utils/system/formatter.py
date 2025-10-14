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


def format_system_summary(
    hardware_info: Dict[str, Any], software_info: Dict[str, Any]
) -> str:
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
            lines.append(
                f"Motherboard: {motherboard_dmi['vendor']} {motherboard_dmi['name']}"
            )
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
        if cpu.get("logical_cores") and cpu["logical_cores"] != "Unknown":
            lines.append(f"  Logical Cores: {cpu['logical_cores']}")
        if cpu.get("frequency_mhz") and cpu["frequency_mhz"] != "Unknown":
            freq_ghz = round(cpu["frequency_mhz"] / 1000, 2)
            lines.append(f"  Frequency: {freq_ghz} GHz")

    # Memory Information
    memory = hardware_info.get("memory", {})
    if memory.get("total_gb") and memory["total_gb"] != "Unknown":
        lines.append(f"Memory: {memory['total_gb']} GB total")
        if memory.get("used_percent") and memory["used_percent"] != "Unknown":
            lines.append(f"  Usage: {memory['used_percent']}%")

    # GPU Information
    gpu = hardware_info.get("gpu", {})
    if gpu.get("device_count") and gpu["device_count"] > 0:
        lines.append(f"GPU: {gpu['device_count']} device(s)")
        devices = gpu.get("devices", [])
        for i, device in enumerate(devices[:2]):  # Show first 2 devices
            name = device.get("full_name", "Unknown")
            discrete = " (Discrete)" if device.get("is_discrete") else " (Integrated)"
            lines.append(f"  {i + 1}. {name}{discrete}")

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
        for i, device in enumerate(devices[:4]):  # Show first 4 devices
            name = device.get("model", "Unknown")
            interface = device.get("interface", "")
            size = device.get("size_gb", "")

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

    lines.append("")  # Empty line after system info
    return "\n".join(lines)


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
            summary_hardware = {}
            summary_software = {}

            # Convert CPU info
            if "cpu" in hardware:
                cpu = hardware["cpu"]
                summary_hardware["cpu"] = {
                    "brand": cpu.get("brand", "Unknown"),
                    "logical_cores": cpu.get(
                        "logical_count", cpu.get("count", "Unknown")
                    ),
                    "frequency_mhz": cpu.get("frequency", {}).get("max", 0),
                }

            # Convert Memory info
            if "memory" in hardware:
                memory = hardware["memory"]
                summary_hardware["memory"] = {
                    "total_gb": round(memory.get("total", 0) / (1000**3)),
                    "used_percent": round(memory.get("percent", 0), 1),
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
                    # Check for OpenVINO enhanced name
                    if "openvino" in device and "quick_access" in device["openvino"]:
                        device_summary["full_name"] = device["openvino"][
                            "quick_access"
                        ].get("full_name", device_summary["full_name"])
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
                        device_summary["full_name"] = device["openvino"][
                            "quick_access"
                        ].get("full_name", device_summary["full_name"])
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
                        "size_gb": round(device.get("size", 0) / (1000**3))
                        if device.get("size")
                        else 0,
                    }
                    summary_hardware["storage"]["devices"].append(device_summary)

            # Convert DMI info
            if "dmi" in hardware:
                summary_hardware["dmi"] = hardware["dmi"]

            # Convert OS info
            if "os" in software:
                os_info = software["os"]
                summary_software["os"] = {
                    "name": os_info.get("distribution", {}).get("name", "Unknown"),
                    "version": os_info.get("distribution", {}).get("version_id", ""),
                    "release": os_info.get("release", ""),
                }

            # Use the consolidated formatter
            system_summary_text = format_system_summary(
                summary_hardware, summary_software
            )
            report_lines.append("\n" + system_summary_text)

        except Exception as e:
            # Fallback to basic info if conversion fails
            report_lines.append(f"\nError processing system info: {e}")
            report_lines.append(
                "Basic hardware/software information available in raw format."
            )

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
                for pkg_name, version in list(installed_packages.items())[
                    :10
                ]:  # Show first 10
                    report_lines.append(f"{pkg_name}: {version}")

                total_installed = python_packages.get(
                    "total_installed", len(installed_packages)
                )
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
        total_gb = memory.get("total", 0) / (1000**3)
        summary_parts.append(f"RAM: {total_gb:.0f} GB")

    return (
        " | ".join(summary_parts)
        if summary_parts
        else "Hardware information unavailable"
    )


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

    return (
        " | ".join(summary_parts)
        if summary_parts
        else "Software information unavailable"
    )
