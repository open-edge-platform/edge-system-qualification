# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System validation utilities for the core framework.

This module provides utilities to validate system requirements
against the collected system information.
"""

import logging
from typing import Any, Dict, List, Optional

from sysagent.utils.core.process import run_command
from sysagent.utils.system.cache import SystemInfoCache

# Setup logger
logger = logging.getLogger(__name__)


class SystemValidator:
    """
    Validates system requirements against collected system information.
    Optimized for tier validation with direct support for tier requirements.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the validator with system information.

        Args:
            cache_dir: Directory to store the cache file
        """
        self.system_info = SystemInfoCache(cache_dir).get_info()
        logger.debug("SystemValidator initialized with system info")

    def validate_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate system requirements against system info.

        Args:
            requirements: Dictionary of requirements to check

        Returns:
            Dictionary with validation results
        """
        logger.debug("Starting system requirements validation")
        results = {"passed": True, "checks": []}

        # Check hardware requirements
        if "hardware" in requirements:
            hw_results = self._validate_hardware(requirements["hardware"])
            results["checks"].extend(hw_results)
            if not all(check["passed"] for check in hw_results):
                results["passed"] = False

        # Check software requirements
        if "software" in requirements:
            sw_results = self._validate_software(requirements["software"])
            results["checks"].extend(sw_results)
            if not all(check["passed"] for check in sw_results):
                results["passed"] = False

        logger.debug(f"System validation completed: {results['passed']}")
        return results

    def _validate_hardware(
        self, hw_requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Validate hardware requirements using latest system info structure."""
        results = []
        hardware = self.system_info.get("hardware", {})

        # CPU Xeon requirement
        if (
            "cpu_xeon_required" in hw_requirements
            and hw_requirements["cpu_xeon_required"]
        ):
            cpu_info = hardware.get("cpu", {})
            cpu_brand = cpu_info.get("brand", "")
            is_xeon = "xeon" in cpu_brand.lower()
            results.append(
                {
                    "name": "Intel Xeon CPU",
                    "passed": is_xeon,
                    "actual": cpu_brand,
                    "required": "Intel Xeon CPU",
                    "category": "hardware.cpu.xeon",
                }
            )

        # CPU Core requirement
        if (
            "cpu_core_required" in hw_requirements
            and hw_requirements["cpu_core_required"]
        ):
            cpu_info = hardware.get("cpu", {})
            cpu_brand = cpu_info.get("brand", "")
            is_core = "core" in cpu_brand.lower()
            results.append(
                {
                    "name": "Intel Core CPU",
                    "passed": is_core,
                    "actual": cpu_brand,
                    "required": "Intel Core CPU",
                    "category": "hardware.cpu.core",
                }
            )

        # CPU Ultra requirement
        if (
            "cpu_ultra_required" in hw_requirements
            and hw_requirements["cpu_ultra_required"]
        ):
            cpu_info = hardware.get("cpu", {})
            cpu_brand = cpu_info.get("brand", "")
            is_ultra = "ultra" in cpu_brand.lower()
            results.append(
                {
                    "name": "Intel Ultra CPU",
                    "passed": is_ultra,
                    "actual": cpu_brand,
                    "required": "Intel Ultra CPU",
                    "category": "hardware.cpu.ultra",
                }
            )

        # CPU minimum cores
        if "cpu_min_cores" in hw_requirements:
            min_cores = hw_requirements["cpu_min_cores"]
            cpu_info = hardware.get("cpu", {})
            actual_cores = cpu_info.get("logical_count", 0)
            passed = actual_cores >= min_cores
            results.append(
                {
                    "name": f"CPU cores >= {min_cores}",
                    "passed": passed,
                    "actual": actual_cores,
                    "required": f">= {min_cores}",
                    "category": "hardware.cpu.cores",
                }
            )

        # Memory minimum requirement
        if "memory_min_gb" in hw_requirements:
            min_memory = hw_requirements["memory_min_gb"]
            memory_info = hardware.get("memory", {})
            total_bytes = memory_info.get("total", 0)
            actual_memory_gb = total_bytes / (1000**3)  # Use GB (1000^3) standard
            passed = actual_memory_gb >= min_memory
            results.append(
                {
                    "name": f"Memory >= {min_memory} GB",
                    "passed": passed,
                    "actual": f"{actual_memory_gb:.0f} GB",
                    "required": f">= {min_memory} GB",
                    "category": "hardware.memory.size",
                }
            )

        # Storage minimum requirement
        if "storage_min_gb" in hw_requirements:
            min_storage = hw_requirements["storage_min_gb"]
            storage_info = hardware.get("storage", {})
            total_free_bytes = storage_info.get("total_free", 0)
            actual_storage_gb = total_free_bytes / (1000**3)  # Use GB (1000^3) standard
            passed = actual_storage_gb >= min_storage
            results.append(
                {
                    "name": f"Free storage >= {min_storage} GB",
                    "passed": passed,
                    "actual": f"{actual_storage_gb:.0f} GB",
                    "required": f">= {min_storage} GB",
                    "category": "hardware.storage.free",
                }
            )

        # Integrated GPU requirement
        if "igpu_required" in hw_requirements and hw_requirements["igpu_required"]:
            gpu_info = hardware.get("gpu", {})
            integrated_count = gpu_info.get("integrated_count", 0)
            has_igpu = integrated_count > 0
            results.append(
                {
                    "name": "Integrated GPU required",
                    "passed": has_igpu,
                    "actual": f"{integrated_count} integrated GPU(s)",
                    "required": "At least 1 integrated GPU",
                    "category": "hardware.gpu.integrated",
                }
            )

        # Discrete GPU requirement
        if "dgpu_required" in hw_requirements and hw_requirements["dgpu_required"]:
            gpu_info = hardware.get("gpu", {})
        # Discrete GPU requirement
        if "dgpu_required" in hw_requirements and hw_requirements["dgpu_required"]:
            gpu_info = hardware.get("gpu", {})
            discrete_count = gpu_info.get("discrete_count", 0)
            has_dgpu = discrete_count > 0
            results.append(
                {
                    "name": "Discrete GPU required",
                    "passed": has_dgpu,
                    "actual": f"{discrete_count} discrete GPU(s)",
                    "required": "At least 1 discrete GPU",
                    "category": "hardware.gpu.discrete",
                }
            )

        # Discrete GPU minimum devices
        if "dgpu_min_devices" in hw_requirements:
            min_devices = hw_requirements["dgpu_min_devices"]
            gpu_info = hardware.get("gpu", {})
            discrete_count = gpu_info.get("discrete_count", 0)
            passed = discrete_count >= min_devices
            results.append(
                {
                    "name": f"Discrete GPUs >= {min_devices}",
                    "passed": passed,
                    "actual": f"{discrete_count} discrete GPU(s)",
                    "required": f">= {min_devices} discrete GPUs",
                    "category": "hardware.gpu.discrete_count",
                }
            )

        # Discrete GPU minimum VRAM
        if "dgpu_min_vram_gb" in hw_requirements:
            min_vram = hw_requirements["dgpu_min_vram_gb"]
            gpu_info = hardware.get("gpu", {})
            gpu_devices = gpu_info.get("devices", [])

            # Calculate total VRAM from discrete GPUs using new structure
            total_vram_gb = 0
            for device in gpu_devices:
                if device.get("is_discrete", False):
                    # Get VRAM from OpenVINO info in new structure
                    openvino_info = device.get("openvino", {})
                    if openvino_info and "memory_gb" in openvino_info:
                        vram_gb = openvino_info["memory_gb"]
                        total_vram_gb += vram_gb
                        logger.debug(f"Found discrete GPU VRAM: {vram_gb:.1f} GB")

            passed = total_vram_gb >= min_vram
            results.append(
                {
                    "name": f"Discrete GPU VRAM >= {min_vram} GB",
                    "passed": passed,
                    "actual": f"{total_vram_gb:.1f} GB total",
                    "required": f">= {min_vram} GB",
                    "category": "hardware.gpu.vram",
                }
            )

        # NPU requirement
        if "npu_required" in hw_requirements and hw_requirements["npu_required"]:
            npu_info = hardware.get("npu", {})
            npu_count = npu_info.get("count", 0)
            has_npu = npu_count > 0
            results.append(
                {
                    "name": "NPU required",
                    "passed": has_npu,
                    "actual": f"{npu_count} NPU(s)",
                    "required": "At least 1 NPU",
                    "category": "hardware.npu.required",
                }
            )

        # NPU minimum devices
        if "npu_min_devices" in hw_requirements:
            min_devices = hw_requirements["npu_min_devices"]
            npu_info = hardware.get("npu", {})
            npu_count = npu_info.get("count", 0)
            passed = npu_count >= min_devices
            results.append(
                {
                    "name": f"NPUs >= {min_devices}",
                    "passed": passed,
                    "actual": f"{npu_count} NPU(s)",
                    "required": f">= {min_devices} NPUs",
                    "category": "hardware.npu.count",
                }
            )

        return results

    def _validate_software(
        self, sw_requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Validate software requirements using latest system info structure."""
        results = []
        software = self.system_info.get("software", {})

        # OS type requirement
        if "os_type" in sw_requirements:
            required_os_types = sw_requirements["os_type"]
            if not isinstance(required_os_types, list):
                required_os_types = [required_os_types]

            os_info = software.get("os", {})
            actual_os = os_info.get("name", "").lower()

            passed = any(req_os.lower() in actual_os for req_os in required_os_types)
            results.append(
                {
                    "name": f"OS type in {required_os_types}",
                    "passed": passed,
                    "actual": actual_os,
                    "required": f"One of: {required_os_types}",
                    "category": "software.os.type",
                }
            )

        # Python version requirements
        if "min_python_version" in sw_requirements:
            min_version = sw_requirements["min_python_version"]
            python_info = software.get("python", {})
            version_info = python_info.get("version_info", {})

            # Create version tuple for comparison
            actual_version_tuple = (
                version_info.get("major", 0),
                version_info.get("minor", 0),
                version_info.get("micro", 0),
            )

            # Parse required version
            min_parts = min_version.split(".")
            min_version_tuple = (
                int(min_parts[0]) if len(min_parts) > 0 else 0,
                int(min_parts[1]) if len(min_parts) > 1 else 0,
                int(min_parts[2]) if len(min_parts) > 2 else 0,
            )

            passed = actual_version_tuple >= min_version_tuple
            actual_version_str = ".".join(map(str, actual_version_tuple))

            results.append(
                {
                    "name": f"Python >= {min_version}",
                    "passed": passed,
                    "actual": actual_version_str,
                    "required": f">= {min_version}",
                    "category": "software.python.version",
                }
            )

        # Docker requirement
        if "docker_required" in sw_requirements and sw_requirements["docker_required"]:
            # Check system packages first for docker
            system_packages = software.get("system_packages", {})
            packages = system_packages.get("packages", {})
            has_docker_package = ("docker" in packages and packages["docker"]) or (
                "docker.io" in packages and packages["docker.io"]
            )

            # Also check if docker is available via command
            docker_available = has_docker_package or self._check_docker_available()

            results.append(
                {
                    "name": "Docker required",
                    "passed": docker_available,
                    "actual": "Available" if docker_available else "Not available",
                    "required": "Docker available",
                    "category": "software.docker.required",
                }
            )

        # Required system packages
        if "required_system_packages" in sw_requirements:
            required_packages = sw_requirements["required_system_packages"]
            if not isinstance(required_packages, list):
                required_packages = [required_packages]

            system_packages = software.get("system_packages", {})
            packages = system_packages.get("packages", {})

            for package in required_packages:
                is_installed = (
                    package in packages and packages[package]
                )  # Check package exists and has version
                results.append(
                    {
                        "name": f"System package '{package}' required",
                        "passed": is_installed,
                        "actual": "Installed" if is_installed else "Not installed",
                        "required": f"Package '{package}' installed",
                        "category": "software.system_packages.required",
                    }
                )

        # Required Python packages
        if "required_python_packages" in sw_requirements:
            required_packages = sw_requirements["required_python_packages"]
            if not isinstance(required_packages, list):
                required_packages = [required_packages]

            python_packages = software.get("python_packages", {})
            packages = python_packages.get("packages", {})

            for package in required_packages:
                is_installed = (
                    package in packages and packages[package]
                )  # Check package exists and has version
                results.append(
                    {
                        "name": f"Python package '{package}' required",
                        "passed": is_installed,
                        "actual": "Installed" if is_installed else "Not installed",
                        "required": f"Package '{package}' installed",
                        "category": "software.python_packages.required",
                    }
                )

        return results

    def _check_docker_available(self) -> bool:
        """Check if Docker is available on the system."""
        try:
            result = run_command(["docker", "--version"], timeout=5)
            return result.success
        except (FileNotFoundError, OSError):
            return False


def validate_system_requirements(
    requirements: Dict[str, Any], cache_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to validate system requirements.

    Args:
        requirements: Dictionary of requirements to check
        cache_dir: Directory to store the cache file

    Returns:
        Dictionary with validation results
    """
    validator = SystemValidator(cache_dir)
    return validator.validate_requirements(requirements)


def check_system_ready_for_tests(
    requirements: Dict[str, Any], cache_dir: Optional[str] = None
) -> bool:
    """
    Check if the system meets all requirements for running tests.

    Args:
        requirements: Dictionary of requirements to check
        cache_dir: Directory to store the cache file

    Returns:
        True if system meets all requirements, False otherwise
    """
    results = validate_system_requirements(requirements, cache_dir)
    return results["passed"]
