# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Hardware information collection utilities.

Provides functions to collect detailed hardware information about the system,
including CPU, GPU, NPU, memory, storage, and network interfaces.
"""

import logging
import os
import platform
from typing import Any, Dict

import cpuinfo
import psutil

from .ov_helper import collect_openvino_devices
from .pci_helper import get_pci_devices

logger = logging.getLogger(__name__)


def collect_hardware_info() -> Dict[str, Any]:
    """
    Collect comprehensive hardware information.

    Returns:
        Dict containing all hardware information
    """
    logger.debug("Collecting hardware system information")

    # Get PCI devices and OpenVINO devices first as they're used by multiple collectors
    pci_devices = get_pci_devices()
    openvino_gpu_info, openvino_cpu_info, openvino_npu_info = collect_openvino_devices()

    # Extract OpenVINO devices by type from the structured response
    openvino_cpu = [openvino_cpu_info] if openvino_cpu_info else []
    openvino_gpu = openvino_gpu_info.get("devices", [])
    openvino_npu = openvino_npu_info.get("devices", [])

    # Collect all OpenVINO devices for backward compatibility
    all_openvino_devices = []
    if openvino_cpu_info:
        all_openvino_devices.append(openvino_cpu_info)
    all_openvino_devices.extend(openvino_gpu)
    all_openvino_devices.extend(openvino_npu)

    hardware_info = {
        "cpu": collect_cpu_info(openvino_cpu),
        "gpu": collect_gpu_info(pci_devices, openvino_gpu),
        "npu": collect_npu_info(pci_devices, openvino_npu),
        "memory": collect_memory_info(),
        "storage": collect_storage_info(),
        "network": collect_network_info(),
        "dmi": collect_dmi_info(),
        "pci": pci_devices,
        "openvino": all_openvino_devices,
    }

    return hardware_info


def collect_cpu_info(openvino_cpu=None) -> Dict[str, Any]:
    """
    Collect CPU information including cores, architecture, and OpenVINO capabilities.

    Args:
        openvino_cpu: List of OpenVINO CPU devices

    Returns:
        Dict containing CPU information
    """
    try:
        cpu_info_data = cpuinfo.get_cpu_info()

        cpu_info = {
            "brand": cpu_info_data.get("brand_raw", "Unknown"),
            "architecture": cpu_info_data.get("arch", platform.machine()),
            "bits": cpu_info_data.get("bits", 64),
            "count": psutil.cpu_count(logical=False),  # Physical cores
            "logical_count": psutil.cpu_count(logical=True),  # Logical cores (including hyperthreading)
            "frequency": {
                "current": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "min": psutil.cpu_freq().min if psutil.cpu_freq() else None,
                "max": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            },
            "flags": cpu_info_data.get("flags", []),
            "vendor_id": cpu_info_data.get("vendor_id_raw", "Unknown"),
            "family": cpu_info_data.get("family", 0),
            "model": cpu_info_data.get("model", 0),
            "stepping": cpu_info_data.get("stepping", 0),
            "cache_size": cpu_info_data.get("l3_cache_size", "Unknown"),
            "microcode": cpu_info_data.get("microcode", "Unknown"),
        }

        # Add OpenVINO CPU device information if available
        if openvino_cpu:
            for ov_cpu in openvino_cpu:
                if ov_cpu:
                    # Extract from quick_access if present, otherwise from top level
                    quick_access = ov_cpu.get("quick_access", {})
                    device_info = ov_cpu.get("device", {})
                    all_props = device_info.get("all_properties", {})

                    # Include essential OpenVINO properties for direct access
                    cpu_openvino = {
                        "device_name": quick_access.get("device_name", ov_cpu.get("device_name", "CPU")),
                        "full_device_name": quick_access.get(
                            "full_device_name",
                            ov_cpu.get("full_device_name", "Unknown"),
                        ),
                        "device_type": quick_access.get("device_type", ov_cpu.get("device_type", "CPU")),
                        "capabilities": quick_access.get("capabilities", ov_cpu.get("capabilities", [])),
                        "vendor": quick_access.get("vendor", ov_cpu.get("vendor", "Unknown")),
                    }

                    # Add memory and performance related properties
                    if "CPU_THREADS_NUM" in all_props:
                        cpu_openvino["threads"] = all_props["CPU_THREADS_NUM"]
                    if "PERFORMANCE_HINT_NUM_REQUESTS" in all_props:
                        cpu_openvino["performance_hint_requests"] = all_props["PERFORMANCE_HINT_NUM_REQUESTS"]
                    if "INFERENCE_NUM_THREADS" in all_props:
                        cpu_openvino["inference_threads"] = all_props["INFERENCE_NUM_THREADS"]

                    # Store the flattened OpenVINO device info
                    cpu_info["openvino"] = cpu_openvino
                    break

        # Add extended CPU features detection
        extended_features = _detect_cpu_features(cpu_info_data)
        cpu_info.update(extended_features)

        return cpu_info

    except Exception as e:
        logger.warning(f"Failed to collect CPU info: {e}")
        return {"error": str(e)}


def collect_gpu_info(pci_devices, openvino_gpu=None) -> dict:
    """
    Collect GPU information from PCI devices and OpenVINO devices.

    OpenVINO GPU devices can be matched to PCI devices using the device_uuid field.
    The device_uuid is a 32-character hex string, where:
        - bytes 0-1: vendor_id (little endian, e.g. 0x80 0x86 -> 0x8086)
        - bytes 2-3: device_id (little endian, e.g. 0x55 0x7d -> 0x7d55)
        - byte 8:    bus
        - byte 10:   device

    To match, we normalize both PCI and OpenVINO devices to a string:
        "{vendor_id}{device_id}{bus}{dev}"
    All values are lower-case hex, zero-padded as needed, and without colons.

    Args:
        pci_devices: List of PCI devices
        openvino_gpu: List of OpenVINO GPU devices

    Returns:
        Dict containing GPU information
    """

    def normalize_pci_device(dev):
        """Normalize PCI device information."""
        return {
            "device_id": dev.get("device_id", "Unknown"),
            "vendor_id": dev.get("vendor_id", "Unknown"),
            "vendor_name": dev.get("vendor_name", "Unknown"),
            "device_name": dev.get("device_name", "Unknown"),
            "class_name": dev.get("class_name", "Unknown"),
            "driver": dev.get("driver", None),
            "subsystem_device": dev.get("subsystem_device", "Unknown"),
            "subsystem_vendor": dev.get("subsystem_vendor", "Unknown"),
            "revision": dev.get("revision", "Unknown"),
            "pci_slot": dev.get("pci_slot", "Unknown"),
        }

    def normalize_openvino_device(ovdev):
        """Normalize OpenVINO device information and extract key properties."""
        # Essential properties for direct access (flattened structure)
        logger.debug(f"Normalizing OpenVINO device: {ovdev}")

        # Extract from quick_access if present, otherwise from top level
        quick_access = ovdev.get("quick_access", {})
        device_info = ovdev.get("device", {})
        all_props = device_info.get("all_properties", {})

        # Build flattened device info structure
        normalized_device = {
            "device_name": quick_access.get("device_name", ovdev.get("device_name", "Unknown")),
            "full_device_name": quick_access.get("full_device_name", ovdev.get("full_device_name", "Unknown")),
            "device_type": quick_access.get("device_type", ovdev.get("device_type", "GPU")),
            "device_id": quick_access.get("device_id", ovdev.get("device_id", "Unknown")),
            "device_uuid": quick_access.get("device_uuid", ovdev.get("device_uuid", "Unknown")),
            "capabilities": quick_access.get("capabilities", ovdev.get("capabilities", [])),
            "vendor": quick_access.get("vendor", ovdev.get("vendor", "Unknown")),
        }

        # Extract memory information from all_properties
        if "GPU_DEVICE_TOTAL_MEM_SIZE" in all_props:
            mem_size = all_props["GPU_DEVICE_TOTAL_MEM_SIZE"]
            if isinstance(mem_size, (int, float)):
                normalized_device["memory_bytes"] = mem_size
                normalized_device["memory_gb"] = mem_size / (1000**3)  # Use GB (1000^3) standard

        # Extract performance-related properties
        if "GPU_EXECUTION_UNITS_COUNT" in all_props:
            normalized_device["execution_units"] = all_props["GPU_EXECUTION_UNITS_COUNT"]
        if "GPU_HW_EXECUTION_UNITS_COUNT" in all_props:
            normalized_device["hw_execution_units"] = all_props["GPU_HW_EXECUTION_UNITS_COUNT"]

        return normalized_device

    def determine_gpu_type_from_openvino(device_type):
        """Determine if GPU is discrete based on OpenVINO device_type."""
        if not device_type:
            return None

        device_type_str = str(device_type).lower()
        if "discrete" in device_type_str or device_type == "Type.DISCRETE":
            return True
        elif "integrated" in device_type_str or device_type == "Type.INTEGRATED":
            return False
        return None

    gpu_info = {"devices": [], "discrete_count": 0, "integrated_count": 0}

    # Get GPU devices from PCI (VGA compatible controller and Display controller)
    pci_gpus = []
    for dev in pci_devices:
        class_name = dev.get("class_name", "").lower()
        if "vga" in class_name or "display" in class_name or "3d" in class_name:
            pci_gpus.append(dev)  # Keep original device, not normalized

    # Get GPU devices from OpenVINO
    openvino_gpus = []
    if openvino_gpu:
        for ovdev in openvino_gpu:
            openvino_gpus.append(normalize_openvino_device(ovdev))

    # Helper function to add canonical GPU info from device table
    def add_canonical_gpu_info(gpu_device, pci_id):
        """Add canonical GPU info from device table if available."""
        from .gpu_devices import get_device_by_pci_id

        if pci_id:
            canonical_device = get_device_by_pci_id(pci_id)
            if canonical_device:
                gpu_device["table"] = canonical_device
                logger.debug(f"Enhanced GPU device with canonical info: {canonical_device.get('name', '')}")

    # Normalization functions for device matching
    def normalize_pci_device_for_matching(dev):
        """Normalize PCI device for UUID matching."""
        vendor_id = dev.get("vendor_id", "").lower().zfill(4)
        device_id = dev.get("device_id", "").lower().zfill(4)
        pci_slot = dev.get("pci_slot", "")
        bus = dev.get("bus")
        dev_num = dev.get("dev")

        if not (bus and dev_num) and pci_slot:
            try:
                # pci_slot format: "domain:bus:device.function" or "bus:device.function"
                if pci_slot.count(":") >= 2:
                    # Format: "domain:bus:device.function"
                    parts = pci_slot.split(":")
                    bus = parts[1].zfill(2)
                    device_func = parts[2].split(".")
                    dev_num = device_func[0].zfill(2)
                elif pci_slot.count(":") == 1:
                    # Format: "bus:device.function"
                    parts = pci_slot.split(":")
                    bus = parts[0].zfill(2)
                    device_func = parts[1].split(".")
                    dev_num = device_func[0].zfill(2)
            except Exception:
                bus = dev_num = None

        if not (bus and dev_num):
            return None
        return f"{vendor_id}{device_id}{bus}{dev_num}"

    def normalize_openvino_device_for_matching(ovdev):
        """Normalize OpenVINO device for UUID matching."""
        uuid = ovdev.get("device_uuid")
        if not uuid or len(uuid) < 20:
            return None
        # vendor_id: bytes 0-1 (little endian)
        vendor_id = uuid[2:4] + uuid[0:2]
        # device_id: bytes 4-5 (little endian)
        device_id = uuid[6:8] + uuid[4:6]
        # bus: byte 16-17 (8th byte, 2 chars)
        bus = uuid[16:18]
        # dev: byte 18-19 (9th byte, 2 chars)
        dev_num = uuid[18:20]
        return f"{vendor_id}{device_id}{bus}{dev_num}".lower()

    # Build a map of normalized PCI device strings to PCI device dicts
    pci_norm_map = {}
    for pci_gpu in pci_gpus:
        norm = normalize_pci_device_for_matching(pci_gpu)
        if norm:
            pci_norm_map[norm] = pci_gpu
            logger.debug(f"GPU PCI device normalized:  {pci_gpu.get('pci_slot', 'Unknown')} -> {norm}")

    # Match OpenVINO devices with PCI devices and build final GPU list
    all_gpus = []
    matched_norms = set()

    # Process OpenVINO devices first (try to match with PCI)
    for ov_gpu in openvino_gpus:
        matched = False
        norm = normalize_openvino_device_for_matching(ov_gpu)
        logger.debug(f"GPU OpenVINO dev normalized: {ov_gpu.get('device_uuid', 'Unknown')} -> {norm}")

        if norm and norm in pci_norm_map:
            # Perfect match found - create combined device
            pci_device = pci_norm_map[norm]
            gpu_device = normalize_pci_device(pci_device)
            gpu_device["source"] = "pci"
            gpu_device["openvino"] = ov_gpu

            # Add canonical GPU info from table
            add_canonical_gpu_info(gpu_device, pci_device.get("device_id"))

            all_gpus.append(gpu_device)
            matched_norms.add(norm)
            matched = True
            logger.debug(
                f"✅ GPU exact match: PCI {pci_device.get('pci_slot', 'Unknown')} <-> "
                f"OpenVINO {ov_gpu.get('device_name', 'Unknown')}"
            )

        if not matched:
            # Add OpenVINO-only device
            gpu_device = {
                "device_name": ov_gpu["device_name"],
                "device_type": ov_gpu["device_type"],
                "source": "openvino",
                "openvino": ov_gpu,
            }
            all_gpus.append(gpu_device)
            logger.debug(f"ℹ️ Added OpenVINO-only GPU dev: {ov_gpu.get('device_name', 'Unknown')}")

    # Add unmatched PCI devices
    for norm, pci_device in pci_norm_map.items():
        if norm not in matched_norms:
            gpu_device = normalize_pci_device(pci_device)
            gpu_device["source"] = "pci"

            # Add canonical GPU info from table
            add_canonical_gpu_info(gpu_device, pci_device.get("device_id"))

            all_gpus.append(gpu_device)
            logger.debug(f"ℹ️ Added PCI-only GPU device: {pci_device.get('pci_slot', 'Unknown')}")
    for gpu in all_gpus:
        is_discrete = None

        # First try to determine from OpenVINO device_type (most reliable)
        if "openvino" in gpu:
            ov_device_type = gpu["openvino"].get("device_type")
            is_discrete = determine_gpu_type_from_openvino(ov_device_type)

        # Default to integrated if still unknown
        if is_discrete is None:
            is_discrete = False

        gpu["is_discrete"] = is_discrete
        if is_discrete:
            gpu_info["discrete_count"] += 1
        else:
            gpu_info["integrated_count"] += 1

    gpu_info["devices"] = all_gpus
    gpu_info["total_count"] = len(all_gpus)

    return gpu_info


def collect_npu_info(pci_devices, openvino_npu=None) -> dict:
    """
    Collect NPU (Neural Processing Unit) information.

    NPU devices are identified by PCI class name "Processing accelerators".
    They may also be available through OpenVINO as NPU devices.

    Args:
        pci_devices: List of PCI devices
        openvino_npu: List of OpenVINO NPU devices

    Returns:
        Dict containing NPU information
    """

    def normalize_pci_device(dev):
        """Normalize PCI device information."""
        return {
            "device_id": dev.get("device_id", "Unknown"),
            "vendor_id": dev.get("vendor_id", "Unknown"),
            "vendor_name": dev.get("vendor_name", "Unknown"),
            "device_name": dev.get("device_name", "Unknown"),
            "class_name": dev.get("class_name", "Unknown"),
            "driver": dev.get("driver", None),
            "subsystem_device": dev.get("subsystem_device", "Unknown"),
            "subsystem_vendor": dev.get("subsystem_vendor", "Unknown"),
            "revision": dev.get("revision", "Unknown"),
            "pci_slot": dev.get("pci_slot", "Unknown"),
        }

    def normalize_openvino_device(ovdev):
        """Normalize OpenVINO device information and extract key properties."""
        # Essential properties for direct access (flattened structure)

        # Extract from quick_access if present, otherwise from top level
        quick_access = ovdev.get("quick_access", {})
        device_info = ovdev.get("device", {})
        all_props = device_info.get("all_properties", {})

        # Build flattened device info structure
        normalized_device = {
            "device_name": quick_access.get("device_name", ovdev.get("device_name", "Unknown")),
            "full_device_name": quick_access.get("full_device_name", ovdev.get("full_device_name", "Unknown")),
            "device_type": quick_access.get("device_type", ovdev.get("device_type", "NPU")),
            "device_id": quick_access.get("device_id", ovdev.get("device_id", "Unknown")),
            "device_uuid": quick_access.get("device_uuid", ovdev.get("device_uuid", "Unknown")),
            "capabilities": quick_access.get("capabilities", ovdev.get("capabilities", [])),
            "vendor": quick_access.get("vendor", ovdev.get("vendor", "Unknown")),
        }

        # Extract NPU-specific properties from all_properties
        if "NPU_DEVICE_TOTAL_MEM_SIZE" in all_props:
            mem_size = all_props["NPU_DEVICE_TOTAL_MEM_SIZE"]
            if isinstance(mem_size, (int, float)):
                normalized_device["memory_bytes"] = mem_size
                normalized_device["memory_gb"] = mem_size / (1000**3)  # Use GB (1000^3) standard

        # Extract performance-related properties
        if "NPU_DEVICE_ARCHITECTURE" in all_props:
            normalized_device["architecture"] = all_props["NPU_DEVICE_ARCHITECTURE"]
        if "DEVICE_ARCHITECTURE" in all_props:
            normalized_device["device_architecture"] = all_props["DEVICE_ARCHITECTURE"]

        return normalized_device

    npu_info = {"devices": [], "count": 0}

    # Get NPU devices from PCI (Processing accelerators)
    pci_npus = []
    for dev in pci_devices:
        class_name = dev.get("class_name", "").lower()
        device_name = dev.get("device_name", "").lower()

        # Look for processing accelerators or NPU-related keywords
        if (
            "processing accelerator" in class_name
            or "accelerator" in class_name
            or "npu" in device_name
            or "neural" in device_name
            or "ai" in device_name
        ):
            pci_npus.append(normalize_pci_device(dev))

    # Get NPU devices from OpenVINO
    openvino_npus = []
    if openvino_npu:
        for ovdev in openvino_npu:
            openvino_npus.append(normalize_openvino_device(ovdev))

    # Simplified NPU mapping logic - NPUs typically have 1-to-1 mapping
    all_npus = []

    logger.debug(f"NPU collection: {len(pci_npus)} PCI devs, {len(openvino_npus)} OpenVINO devs")

    # Simple case: If exactly one Processing accelerator and one OpenVINO NPU, map them
    processing_accelerators = [
        dev for dev in pci_npus if dev.get("class_name", "").lower() == "processing accelerators"
    ]

    if len(processing_accelerators) == 1 and len(openvino_npus) == 1:
        # Direct 1-to-1 mapping for single NPU devices
        pci_device = processing_accelerators[0]
        ov_device = openvino_npus[0]

        # Create combined device with both PCI and OpenVINO info
        npu_device = normalize_pci_device(pci_device)
        npu_device["source"] = "pci"
        npu_device["openvino"] = ov_device
        all_npus.append(npu_device)

        logger.debug(
            f"✅ NPU mapping: PCI {pci_device.get('pci_address', 'Unknown')} <-> "
            f"OpenVINO {ov_device.get('device_name', 'Unknown')}"
        )

        # Remove matched devices to avoid duplicates
        pci_npus = [dev for dev in pci_npus if dev != pci_device]
        openvino_npus = [dev for dev in openvino_npus if dev != ov_device]

    # Handle multiple NPUs or complex cases with UUID matching as fallback
    elif len(pci_npus) > 1 or len(openvino_npus) > 1:

        def normalize_pci_device_for_matching(dev):
            """Normalize PCI device for UUID matching."""
            vendor_id = dev.get("vendor_id", "").lower().zfill(4)
            device_id = dev.get("device_id", "").lower().zfill(4)
            pci_address = dev.get("pci_address", "")
            bus = dev.get("bus")
            dev_num = dev.get("dev")

            if not (bus and dev_num) and pci_address:
                try:
                    # pci_address format: "domain:bus:device.function"
                    if pci_address.count(":") >= 2:
                        parts = pci_address.split(":")
                        bus = parts[1].zfill(2)
                        device_func = parts[2].split(".")
                        dev_num = device_func[0].zfill(2)
                    elif pci_address.count(":") == 1:
                        parts = pci_address.split(":")
                        bus = parts[0].zfill(2)
                        device_func = parts[1].split(".")
                        dev_num = device_func[0].zfill(2)
                except Exception:
                    bus = dev_num = None

            if not (bus and dev_num):
                return None
            return f"{vendor_id}{device_id}{bus}{dev_num}"

        def normalize_openvino_device_for_matching(ovdev):
            """Normalize OpenVINO device for UUID matching."""
            uuid = ovdev.get("device_uuid")
            if not uuid or len(uuid) < 20:
                return None
            # vendor_id: bytes 0-1 (little endian)
            vendor_id = uuid[2:4] + uuid[0:2]
            # device_id: bytes 4-5 (little endian)
            device_id = uuid[6:8] + uuid[4:6]
            # bus: byte 16-17 (8th byte, 2 chars)
            bus = uuid[16:18]
            # dev: byte 18-19 (9th byte, 2 chars)
            dev_num = uuid[18:20]
            return f"{vendor_id}{device_id}{bus}{dev_num}".lower()

        # Try UUID-based matching for multiple NPUs
        pci_norm_map = {}
        for device in pci_npus:
            norm = normalize_pci_device_for_matching(device)
            if norm:
                pci_norm_map[norm] = device

        matched_npus = []
        remaining_openvino = openvino_npus.copy()

        for ov_npu in openvino_npus:
            norm = normalize_openvino_device_for_matching(ov_npu)

            if norm and norm in pci_norm_map:
                # UUID match found
                pci_device = pci_norm_map[norm]
                npu_device = normalize_pci_device(pci_device)
                npu_device["source"] = "pci"
                npu_device["openvino"] = ov_npu
                matched_npus.append(npu_device)
                remaining_openvino.remove(ov_npu)
                logger.debug(
                    f"✅ NPU match: PCI {pci_device.get('pci_address', 'Unknown')} <-> "
                    f"OpenVINO {ov_npu.get('device_name', 'Unknown')}"
                )

        all_npus.extend(matched_npus)
        openvino_npus = remaining_openvino

        # Remove matched PCI devices
        for matched_npu in matched_npus:
            pci_address = matched_npu.get("pci_address")
            pci_npus = [dev for dev in pci_npus if dev.get("pci_address") != pci_address]

    # Add remaining PCI devices (no OpenVINO match)
    for pci_npu in pci_npus:
        if not any(existing.get("pci_address") == pci_npu.get("pci_address") for existing in all_npus):
            npu_device = normalize_pci_device(pci_npu)
            npu_device["source"] = "pci"
            all_npus.append(npu_device)
            logger.debug(f"ℹ️ Added PCI-only NPU device: {pci_npu.get('pci_address', 'Unknown')}")

    # Add remaining OpenVINO devices (no PCI match)
    for ov_npu in openvino_npus:
        npu_device = {
            "device_name": ov_npu["device_name"],
            "device_type": ov_npu["device_type"],
            "source": "openvino",
            "openvino": ov_npu,
        }
        all_npus.append(npu_device)
        logger.debug(f"ℹ️ Added OpenVINO-only NPU device: {ov_npu.get('device_name', 'Unknown')}")

    npu_info["devices"] = all_npus
    npu_info["count"] = len(all_npus)

    return npu_info


def collect_memory_info() -> Dict[str, Any]:
    """
    Collect memory information including physical and virtual memory details.

    Returns:
        Dict containing memory information
    """
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "free": memory.free,
            "percent": memory.percent,
            "swap": {
                "total": swap.total,
                "used": swap.used,
                "free": swap.free,
                "percent": swap.percent,
            },
        }

        return memory_info

    except Exception as e:
        logger.warning(f"Failed to collect memory info: {e}")
        return {"error": str(e)}


def _get_disk_mapping() -> Dict[str, Dict[str, str]]:
    """
    Get mapping of physical disks from /dev/disk/by-id/ with model information.

    Returns:
        Dict mapping disk device names to their information (model, interface)
    """
    disk_mapping = {}

    try:
        import glob
        import re

        # Get disk IDs from /dev/disk/by-id/ (exclude partitions)
        if os.path.exists("/dev/disk/by-id/"):
            for disk_link in glob.glob("/dev/disk/by-id/*"):
                # Skip partition entries (contain -part)
                if "-part" in disk_link:
                    continue

                # Get the actual device name
                try:
                    real_device = os.path.realpath(disk_link)
                    device_name = os.path.basename(real_device)
                    link_name = os.path.basename(disk_link)

                    # Skip WWN entries if already have ATA/NVMe entries for same device
                    if link_name.startswith("wwn-") and device_name in disk_mapping:
                        continue

                    model = "Unknown"
                    interface = "Unknown"

                    if link_name.startswith("ata-"):
                        interface = "SATA"
                        # Extract model from ATA string (format: ata-MODEL_SERIAL)
                        model_parts = link_name.replace("ata-", "").split("_")
                        if model_parts:
                            model = model_parts[0]
                    elif link_name.startswith("nvme-"):
                        interface = "NVMe"
                        if link_name.startswith("nvme-eui."):
                            # EUI format - get model from sysfs
                            model = _get_disk_model_from_sysfs(device_name)
                        else:
                            # Extract model from NVMe string (format: nvme-MODEL_SERIAL)
                            model_parts = link_name.replace("nvme-", "").split("_")
                            if model_parts:
                                model = model_parts[0]
                    elif link_name.startswith("usb-"):
                        interface = "USB"
                        # Extract vendor and model from USB string
                        usb_parts = link_name.replace("usb-", "").split("_")
                        if len(usb_parts) >= 2:
                            model = f"{usb_parts[0]} {usb_parts[1]}"

                    # Get model from sysfs if not extracted from link name
                    if model == "Unknown":
                        model = _get_disk_model_from_sysfs(device_name)

                    disk_mapping[device_name] = {
                        "model": model,
                        "interface": interface,
                        "by_id_link": link_name,
                    }
                except (OSError, IOError):
                    continue

        # Fallback: get remaining disks from /sys/class/block/
        for device in glob.glob("/sys/class/block/*"):
            device_name = os.path.basename(device)

            # Skip loop devices, ram devices, and partition-like names
            if device_name.startswith(("loop", "ram", "dm-")) or re.match(r".*[0-9]+$", device_name):
                continue

            if device_name not in disk_mapping:
                model = _get_disk_model_from_sysfs(device_name)
                interface = _guess_interface_from_device_name(device_name)

                disk_mapping[device_name] = {
                    "model": model,
                    "interface": interface,
                    "by_id_link": None,
                }

    except Exception as e:
        logger.debug(f"Error getting disk mapping: {e}")

    return disk_mapping


def _get_disk_model_from_sysfs(device_name: str) -> str:
    """
    Get disk model from sysfs information.

    Args:
        device_name: Name of the block device (e.g., 'sda', 'nvme0n1')

    Returns:
        Disk model string or "Unknown"
    """
    try:
        model_path = f"/sys/class/block/{device_name}/device/model"
        if os.path.exists(model_path):
            with open(model_path, "r") as f:
                return f.read().strip()
    except (OSError, IOError):
        pass

    return "Unknown"


def _guess_interface_from_device_name(device_name: str) -> str:
    """
    Guess the interface type from device name.

    Args:
        device_name: Name of the block device

    Returns:
        Interface type string
    """
    if device_name.startswith("nvme"):
        return "NVMe"
    elif device_name.startswith(("sd", "hd")):
        return "SATA"
    elif device_name.startswith("mmc"):
        return "eMMC"
    else:
        return "Unknown"


def _get_parent_disk(partition_device: str) -> str:
    """
    Get the parent disk device name from a partition device.

    Args:
        partition_device: Partition device path (e.g., '/dev/sda1', '/dev/nvme0n1p1')

    Returns:
        Parent disk device name (e.g., 'sda', 'nvme0n1') or None
    """
    import re

    device_name = os.path.basename(partition_device)

    # Handle NVMe devices (nvme0n1p1 -> nvme0n1)
    if device_name.startswith("nvme"):
        match = re.match(r"(nvme\d+n\d+)", device_name)
        if match:
            return match.group(1)

    # Handle SATA/IDE devices (sda1 -> sda)
    elif device_name.startswith(("sd", "hd")):
        match = re.match(r"([a-z]+)", device_name)
        if match:
            return match.group(1)

    # Handle loop devices and other special cases
    elif device_name.startswith("loop"):
        return None  # Skip loop devices

    return None


def _get_disk_size(device_name: str) -> int:
    """
    Get the total size of a disk device in bytes.

    Args:
        device_name: Name of the block device (e.g., 'sda', 'nvme0n1')

    Returns:
        Disk size in bytes or 0 if unable to determine
    """
    try:
        size_path = f"/sys/class/block/{device_name}/size"
        if os.path.exists(size_path):
            with open(size_path, "r") as f:
                # Size is in 512-byte sectors
                sectors = int(f.read().strip())
                return sectors * 512
    except (OSError, IOError, ValueError):
        pass

    return 0


def collect_storage_info() -> Dict[str, Any]:
    """
    Collect storage information grouped by disk model/name with partitions
    as part of disk objects.

    Uses information from /dev/disk/by-id/ to group disks by model and includes
    partitions as part of each disk object for better storage summarization.

    Returns:
        Dict containing storage information grouped by disk devices
    """
    try:
        storage_info = {
            "devices": [],
            "total_size": 0,
            "total_used": 0,
            "total_free": 0,
        }

        # Get physical disk mapping from /dev/disk/by-id/
        disk_mapping = _get_disk_mapping()
        partitions = psutil.disk_partitions()

        # Find the root partition to determine the primary storage device
        root_partition = None
        root_disk_device = None

        for partition in partitions:
            if partition.mountpoint == "/":
                root_partition = partition
                root_disk_device = _get_parent_disk(partition.device)
                break

        # Group partitions by their parent disk device
        disk_partitions = {}
        for partition in partitions:
            parent_disk = _get_parent_disk(partition.device)
            if parent_disk and parent_disk in disk_mapping:
                if parent_disk not in disk_partitions:
                    disk_partitions[parent_disk] = []
                disk_partitions[parent_disk].append(partition)

        # Create storage device objects for each physical disk
        for disk_device, disk_info in disk_mapping.items():
            device_obj = {
                "device": disk_device,
                "model": disk_info.get("model", "Unknown"),
                "interface": disk_info.get("interface", "Unknown"),
                "size": _get_disk_size(disk_device),
                "partitions": [],
                "total_partition_size": 0,
                "total_partition_used": 0,
                "total_partition_free": 0,
            }

            # Add partitions for this disk device
            if disk_device in disk_partitions:
                for partition in disk_partitions[disk_device]:
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        partition_info = {
                            "device": partition.device,
                            "mountpoint": partition.mountpoint,
                            "fstype": partition.fstype,
                            "opts": partition.opts,
                            "total": usage.total,
                            "used": usage.used,
                            "free": usage.free,
                            "percent": (usage.used / usage.total) * 100 if usage.total > 0 else 0,
                        }
                        device_obj["partitions"].append(partition_info)
                        device_obj["total_partition_size"] += usage.total
                        device_obj["total_partition_used"] += usage.used
                        device_obj["total_partition_free"] += usage.free

                    except PermissionError:
                        # Skip inaccessible partitions
                        continue

            storage_info["devices"].append(device_obj)

        # Calculate global totals based on root disk only (for system validation)
        if root_disk_device and root_partition:
            try:
                # Use total disk size for the device containing root partition
                root_disk_size = _get_disk_size(root_disk_device)
                root_usage = psutil.disk_usage(root_partition.mountpoint)

                # Global totals should reflect the primary storage (root disk)
                storage_info["total_size"] = root_disk_size
                storage_info["total_used"] = root_usage.used
                storage_info["total_free"] = root_usage.free
            except Exception as e:
                logger.warning(f"Failed to get root disk usage: {e}")
                # Fallback: use root partition usage
                if root_partition:
                    try:
                        root_usage = psutil.disk_usage(root_partition.mountpoint)
                        storage_info["total_size"] = root_usage.total
                        storage_info["total_used"] = root_usage.used
                        storage_info["total_free"] = root_usage.free
                    except Exception:
                        pass

        # Sort devices with root mount disk first
        def sort_storage_devices(device):
            """Sort function to prioritize root mount disk."""
            # Check if any partition is mounted at root
            for partition in device.get("partitions", []):
                if partition.get("mountpoint") == "/":
                    return 0  # Root mount gets highest priority

            # Secondary priority: devices with mounted partitions
            if device.get("partitions"):
                return 1

            # Lowest priority: devices without mounted partitions
            return 2

        storage_info["devices"].sort(key=sort_storage_devices)

        return storage_info

    except Exception as e:
        logger.warning(f"Failed to collect storage info: {e}")
        return {"error": str(e)}


def collect_network_info() -> Dict[str, Any]:
    """
    Collect network interface information including addresses and statistics.

    Returns:
        Dict containing network information
    """
    try:
        interfaces = psutil.net_if_addrs()
        stats = psutil.net_if_stats()

        network_info = {
            "interfaces": [],
            "default_gateway": None,
            "internet_connected": _check_internet_connectivity(),
        }

        for interface_name, addresses in interfaces.items():
            interface_info = {
                "name": interface_name,
                "addresses": [],
                "type": _detect_interface_type(interface_name),
                "is_up": stats.get(interface_name, {}).isup if interface_name in stats else False,
                "speed": stats.get(interface_name, {}).speed if interface_name in stats else 0,
                "mtu": stats.get(interface_name, {}).mtu if interface_name in stats else 0,
            }

            for addr in addresses:
                addr_info = {
                    "family": addr.family.name,
                    "address": addr.address,
                    "netmask": addr.netmask,
                    "broadcast": addr.broadcast,
                    "ptp": addr.ptp,
                }
                interface_info["addresses"].append(addr_info)

            network_info["interfaces"].append(interface_info)

        return network_info

    except Exception as e:
        logger.warning(f"Failed to collect network info: {e}")
        return {"error": str(e)}


def collect_dmi_info() -> Dict[str, Any]:
    """
    Collect DMI (Desktop Management Interface) information about the system.

    Returns:
        Dict containing DMI information
    """
    dmi_info = {"system": {}, "bios": {}, "motherboard": {}, "chassis": {}}

    try:
        # Try to read DMI information from various sources
        dmi_files = {
            "system": {
                "vendor": "/sys/class/dmi/id/sys_vendor",
                "product_name": "/sys/class/dmi/id/product_name",
                "product_version": "/sys/class/dmi/id/product_version",
                "product_serial": "/sys/class/dmi/id/product_serial",
                "product_sku": "/sys/class/dmi/id/product_sku",
                "product_family": "/sys/class/dmi/id/product_family",
            },
            "bios": {
                "vendor": "/sys/class/dmi/id/bios_vendor",
                "version": "/sys/class/dmi/id/bios_version",
                "date": "/sys/class/dmi/id/bios_date",
            },
            "motherboard": {
                "vendor": "/sys/class/dmi/id/board_vendor",
                "name": "/sys/class/dmi/id/board_name",
                "version": "/sys/class/dmi/id/board_version",
                "serial": "/sys/class/dmi/id/board_serial",
                "asset_tag": "/sys/class/dmi/id/board_asset_tag",
            },
            "chassis": {
                "vendor": "/sys/class/dmi/id/chassis_vendor",
                "type": "/sys/class/dmi/id/chassis_type",
                "version": "/sys/class/dmi/id/chassis_version",
                "serial": "/sys/class/dmi/id/chassis_serial",
                "asset_tag": "/sys/class/dmi/id/chassis_asset_tag",
            },
        }

        for category, files in dmi_files.items():
            for key, filepath in files.items():
                try:
                    if os.path.exists(filepath):
                        with open(filepath, "r") as f:
                            value = f.read().strip()
                            if value and value != "Unknown" and value != "Not Specified":
                                dmi_info[category][key] = value
                except Exception:
                    continue

    except Exception as e:
        logger.warning(f"Failed to collect DMI info: {e}")
        dmi_info["error"] = str(e)

    return dmi_info


def _detect_cpu_features(cpu_info_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect advanced CPU features and capabilities.

    Args:
        cpu_info_data: Raw CPU information from cpuinfo

    Returns:
        Dict containing extended CPU features
    """
    features = {
        "virtualization": False,
        "hyper_threading": False,
        "turbo_boost": False,
        "aes_ni": False,
        "avx": False,
        "avx2": False,
        "avx512": False,
    }

    flags = cpu_info_data.get("flags", [])
    if isinstance(flags, list):
        flags_lower = [flag.lower() for flag in flags]

        # Check for virtualization support
        features["virtualization"] = any(flag in flags_lower for flag in ["vmx", "svm"])

        # Check for AES-NI support
        features["aes_ni"] = "aes" in flags_lower

        # Check for AVX support
        features["avx"] = "avx" in flags_lower
        features["avx2"] = "avx2" in flags_lower
        features["avx512"] = any("avx512" in flag for flag in flags_lower)

    # Check hyper-threading
    logical_cores = cpu_info_data.get("count", psutil.cpu_count(logical=True))
    physical_cores = psutil.cpu_count(logical=False)
    features["hyper_threading"] = logical_cores > physical_cores

    return features


def _detect_interface_type(interface_name: str, driver: str = None) -> str:
    """
    Detect the type of network interface based on name and driver.

    Args:
        interface_name: Name of the network interface
        driver: Driver name if available

    Returns:
        Interface type string
    """
    name_lower = interface_name.lower()

    # Wireless interfaces
    if any(prefix in name_lower for prefix in ["wl", "wlan", "wifi", "ath", "iwl"]):
        return "wireless"

    # Ethernet interfaces
    if any(prefix in name_lower for prefix in ["eth", "en", "em", "p", "eno", "ens", "enp"]):
        return "ethernet"

    # Loopback
    if "lo" in name_lower:
        return "loopback"

    # Bridge interfaces
    if any(prefix in name_lower for prefix in ["br", "bridge"]):
        return "bridge"

    # Virtual/tunnel interfaces
    if any(prefix in name_lower for prefix in ["tun", "tap", "veth", "docker", "vbox", "vmware"]):
        return "virtual"

    # Bluetooth
    if "bnep" in name_lower or "bt" in name_lower:
        return "bluetooth"

    # USB interfaces
    if "usb" in name_lower:
        return "usb"

    return "unknown"


def _check_internet_connectivity() -> bool:
    """
    Check if the system has internet connectivity using HTTPS requests.

    Uses multiple endpoints to handle different network environments,
    including regions where certain services may be blocked.

    Returns:
        True if internet is accessible, False otherwise
    """
    import requests

    # List of reliable endpoints to test connectivity
    test_endpoints = [
        "https://example.com",  # Global, minimal probe
        "https://www.cloudflare.com/cdn-cgi/trace",  # Global, lightweight
        "https://www.baidu.com",  # China-accessible
    ]

    for endpoint in test_endpoints:
        try:
            # Use HTTPS with proper TLS verification and short timeout
            response = requests.get(
                endpoint,
                timeout=5,
                verify=True,  # Enable TLS certificate verification
                allow_redirects=True,
            )
            if response.status_code == 200:
                logger.debug(f"Internet connectivity confirmed via {endpoint}")
                return True
        except requests.exceptions.RequestException:
            # Continue to next endpoint if this one fails
            continue

    logger.debug("No internet connectivity detected from any test endpoint")
    return False
