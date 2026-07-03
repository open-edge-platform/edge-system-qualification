# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Host virtualization-compatibility probes.

Each function inspects a single aspect of Intel virtualization support
(VT-x, VT-d/IOMMU, KVM/VFIO kernel modules, nested virtualization, the
``/dev/kvm`` device, and VFIO passthrough devices) and returns a simple,
serializable result so the pytest entry point can aggregate them.
"""

import logging
import os
import re
from typing import Dict

from sysagent.utils.core import run_command

logger = logging.getLogger(__name__)


def check_cpu_vt_x() -> tuple[bool, str]:
    """
    Check if Intel VT-x (vmx) is supported by the CPU.

    Returns:
        Tuple of (is_supported: bool, message: str)
    """
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            cpuinfo = f.read()

        if "vmx" in cpuinfo.lower():
            return True, "Intel VT-x (vmx) detected"
        else:
            return False, "Intel VT-x (vmx) not found in CPU flags"
    except (IOError, OSError) as e:
        logger.error(f"Failed to read /proc/cpuinfo: {e}")
        return False, f"Error reading CPU info: {e}"


def check_iommu_support() -> tuple[bool, str, Dict[str, any]]:
    """
    Check if Intel VT-d/IOMMU is enabled in kernel.

    Returns:
        Tuple of (is_enabled: bool, message: str, details: dict)
    """
    details = {
        "kernel_param": False,
        "iommu_groups": 0,
        "iommu_enabled_in_dmesg": False,
    }

    # Check kernel command line parameters
    try:
        with open("/proc/cmdline", "r", encoding="utf-8") as f:
            cmdline = f.read()

        if "intel_iommu=on" in cmdline or "iommu=pt" in cmdline:
            details["kernel_param"] = True
    except (IOError, OSError) as e:
        logger.debug(f"Failed to read /proc/cmdline: {e}")

    # Check IOMMU groups
    iommu_groups_path = "/sys/kernel/iommu_groups"
    if os.path.exists(iommu_groups_path):
        try:
            groups = [d for d in os.listdir(iommu_groups_path) if os.path.isdir(os.path.join(iommu_groups_path, d))]
            details["iommu_groups"] = len(groups)
        except (IOError, OSError) as e:
            logger.debug(f"Failed to read IOMMU groups: {e}")

    # Check dmesg for IOMMU initialization
    result = run_command(["dmesg"], timeout=10)
    if result and result.returncode == 0 and result.stdout:
        dmesg_output = result.stdout.lower()
        if "iommu" in dmesg_output and "enabled" in dmesg_output:
            details["iommu_enabled_in_dmesg"] = True

    # Determine overall status
    if details["kernel_param"] and details["iommu_groups"] > 0:
        return True, f"Intel VT-d/IOMMU enabled ({details['iommu_groups']} IOMMU groups)", details
    elif details["iommu_groups"] > 0:
        return True, f"IOMMU groups present ({details['iommu_groups']}) but kernel param not set", details
    else:
        return False, "Intel VT-d/IOMMU not enabled or not configured", details


def check_kernel_module(module_name: str) -> tuple[bool, str]:
    """
    Check if a kernel module is loaded.

    Args:
        module_name: Name of the kernel module (e.g., 'kvm', 'vfio')

    Returns:
        Tuple of (is_loaded: bool, message: str)
    """
    # Validate module name (alphanumeric, underscores, hyphens only)
    if not re.match(r"^[a-zA-Z0-9_\-]+$", module_name):
        logger.error(f"Invalid module name: {module_name}")
        return False, f"Invalid module name: {module_name}"

    result = run_command(["lsmod"], timeout=10)
    if result and result.returncode == 0 and result.stdout:
        if module_name in result.stdout:
            return True, f"{module_name} module loaded"
        else:
            return False, f"{module_name} module not loaded"
    else:
        return False, f"Failed to check {module_name} module status"


def check_nested_virtualization() -> tuple[bool, str]:
    """
    Check if nested virtualization is enabled for KVM Intel.

    Returns:
        Tuple of (is_enabled: bool, message: str)
    """
    nested_param_path = "/sys/module/kvm_intel/parameters/nested"

    if not os.path.exists(nested_param_path):
        return False, "kvm_intel module not loaded or nested parameter not available"

    try:
        with open(nested_param_path, "r", encoding="utf-8") as f:
            nested_value = f.read().strip()

        if nested_value in ["Y", "1"]:
            return True, "Nested virtualization enabled"
        else:
            return False, f"Nested virtualization disabled (value: {nested_value})"
    except (IOError, OSError) as e:
        logger.error(f"Failed to read nested parameter: {e}")
        return False, f"Error checking nested virtualization: {e}"


def check_kvm_device() -> tuple[bool, str]:
    """
    Check if /dev/kvm device exists and is accessible.

    Returns:
        Tuple of (is_accessible: bool, message: str)
    """
    kvm_dev_path = "/dev/kvm"

    if not os.path.exists(kvm_dev_path):
        return False, "/dev/kvm device not found"

    if os.access(kvm_dev_path, os.R_OK | os.W_OK):
        return True, "/dev/kvm accessible with read/write permissions"
    elif os.access(kvm_dev_path, os.R_OK):
        return False, "/dev/kvm accessible with read-only (needs write permission)"
    else:
        return False, "/dev/kvm exists but not accessible"


def check_vfio_devices() -> tuple[int, str]:
    """
    Check for VFIO devices in /dev/vfio.

    Returns:
        Tuple of (device_count: int, message: str)
    """
    vfio_path = "/dev/vfio"

    if not os.path.exists(vfio_path):
        return 0, "/dev/vfio directory not found"

    try:
        devices = [d for d in os.listdir(vfio_path) if d != "vfio"]
        device_count = len(devices)

        if device_count > 0:
            return device_count, f"{device_count} VFIO device(s) available"
        else:
            return 0, "No VFIO devices (normal if not using device passthrough)"
    except (IOError, OSError) as e:
        logger.debug(f"Failed to list VFIO devices: {e}")
        return 0, f"Error checking VFIO devices: {e}"
