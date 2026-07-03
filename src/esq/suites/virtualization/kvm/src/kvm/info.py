# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Persistence of detailed virtualization diagnostics.

Collects raw command output (CPU flags, loaded modules, IOMMU details) and
writes it alongside the test results so the compatibility verdict can be audited.
"""

import logging
import os

from sysagent.utils.core import run_command

logger = logging.getLogger(__name__)


def save_virtualization_info(output_dir: str):
    """
    Save detailed virtualization information to files.

    Args:
        output_dir: Directory to save output files
    """
    commands = {
        "cpuinfo_vmx.txt": ["grep", "-i", "vmx", "/proc/cpuinfo"],
        "lsmod_kvm.txt": ["lsmod"],
        "dmesg_iommu.txt": ["dmesg"],
        "iommu_groups.txt": ["find", "/sys/kernel/iommu_groups", "-type", "l"],
    }

    for filename, cmd in commands.items():
        result = run_command(cmd, timeout=10)
        if result and result.returncode == 0 and result.stdout:
            output_path = os.path.join(output_dir, filename)
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
                logger.debug(f"Saved {filename}")
            except IOError as e:
                logger.warning(f"Failed to save {filename}: {e}")
