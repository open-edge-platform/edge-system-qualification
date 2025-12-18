# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System validation utilities for the core framework.

This module provides utilities to validate system requirements
against the collected system information.
"""

import logging
import re
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

    def validate_requirements(
        self, requirements: Dict[str, Any], log_suggestions: bool = True, context: str = "system validator"
    ) -> Dict[str, Any]:
        """
        Validate system requirements against system info.

        Args:
            requirements: Dictionary of requirements to check
            log_suggestions: Whether to log fix suggestions for failed checks
            context: Context string for validation messages (e.g., "profile: my-profile")

        Returns:
            Dictionary with validation results
        """
        logger.debug("Starting system requirements validation")
        results = {"passed": True, "checks": []}

        # Check hardware requirements
        if "hardware" in requirements:
            hw_results = self._validate_hardware(requirements["hardware"])
            results["checks"].extend(hw_results)

            failed_hw_checks = [check for check in hw_results if not check["passed"]]
            if failed_hw_checks:
                results["passed"] = False

        # Check software requirements
        if "software" in requirements:
            sw_results = self._validate_software(requirements["software"])
            results["checks"].extend(sw_results)

            failed_sw_checks = [check for check in sw_results if not check["passed"]]
            if failed_sw_checks:
                results["passed"] = False

        # Summary logging
        failed_checks = [check for check in results["checks"] if not check["passed"]]

        # Provide user-friendly suggestions (only if requested)
        if not results["passed"] and log_suggestions:
            self._log_fix_suggestions(failed_checks, context)

        return results

    def _get_intel_devices_with_openvino(
        self, devices: List[Dict[str, Any]], device_filter: Optional[callable] = None
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Get Intel devices and filter for those detected by OpenVINO.

        Args:
            devices: List of device dictionaries
            device_filter: Optional filter function to apply before vendor check

        Returns:
            Tuple of (intel_devices, intel_devices_with_openvino)
        """
        # Apply custom filter if provided (e.g., discrete/integrated GPU check)
        filtered_devices = [dev for dev in devices if device_filter(dev)] if device_filter else devices

        # Find Intel devices (vendor_id 8086)
        intel_devices = [dev for dev in filtered_devices if dev.get("vendor_id", "").lower() == "8086"]

        # Find Intel devices detected by OpenVINO
        intel_devices_with_ov = [dev for dev in intel_devices if "openvino" in dev]

        return intel_devices, intel_devices_with_ov

    def _build_device_error_message(
        self,
        intel_devices: List[Dict[str, Any]],
        intel_devices_with_ov: List[Dict[str, Any]],
        all_devices: List[Dict[str, Any]],
        device_type: str,
        non_intel_check: Optional[callable] = None,
    ) -> str:
        """
        Build meaningful error message for device validation.

        Args:
            intel_devices: List of Intel devices found
            intel_devices_with_ov: List of Intel devices detected by OpenVINO
            all_devices: All devices for non-Intel check
            device_type: Type of device (e.g., "iGPU", "dGPU", "NPU")
            non_intel_check: Optional function to check for non-Intel devices

        Returns:
            Error message string
        """
        if len(intel_devices_with_ov) > 0:
            return f"{len(intel_devices_with_ov)} Intel {device_type}(s) detected by OpenVINO"

        # No Intel devices detected by OpenVINO
        if len(intel_devices) > 0:
            return (
                f"{len(intel_devices)} Intel {device_type}(s) found but not detected by OpenVINO "
                "(driver may not be installed/configured)"
            )

        # Check if non-Intel devices exist
        if non_intel_check and any(non_intel_check(dev) for dev in all_devices):
            return f"Non-Intel {device_type.lower()} found (Intel {device_type} required)"

        return f"No {device_type.lower()} found"

    def _validate_device_required(
        self,
        devices: List[Dict[str, Any]],
        device_type: str,
        category: str,
        device_filter: Optional[callable] = None,
        non_intel_check: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Validate that at least one Intel device detected by OpenVINO exists.

        Args:
            devices: List of device dictionaries
            device_type: Human-readable device type (e.g., "iGPU", "dGPU", "NPU")
            category: Category string for validation result
            device_filter: Optional filter to apply before vendor check
            non_intel_check: Optional check for non-Intel devices

        Returns:
            Validation result dictionary
        """
        intel_devices, intel_devices_with_ov = self._get_intel_devices_with_openvino(devices, device_filter)
        passed = len(intel_devices_with_ov) > 0
        actual_msg = self._build_device_error_message(
            intel_devices, intel_devices_with_ov, devices, device_type, non_intel_check
        )

        return {
            "name": f"{device_type} required",
            "passed": passed,
            "actual": actual_msg,
            "required": f"At least 1 Intel {device_type} detected by OpenVINO",
            "category": category,
        }

    def _validate_device_min_count(
        self,
        devices: List[Dict[str, Any]],
        min_count: int,
        device_type: str,
        category: str,
        device_filter: Optional[callable] = None,
        non_intel_check: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Validate minimum device count for Intel devices detected by OpenVINO.

        Args:
            devices: List of device dictionaries
            min_count: Minimum required device count
            device_type: Human-readable device type
            category: Category string for validation result
            device_filter: Optional filter to apply before vendor check
            non_intel_check: Optional check for non-Intel devices

        Returns:
            Validation result dictionary
        """
        intel_devices, intel_devices_with_ov = self._get_intel_devices_with_openvino(devices, device_filter)
        passed = len(intel_devices_with_ov) >= min_count

        if not passed:
            actual_msg = self._build_device_error_message(
                intel_devices, intel_devices_with_ov, devices, device_type, non_intel_check
            )
        else:
            actual_msg = f"{len(intel_devices_with_ov)} Intel {device_type}(s) detected by OpenVINO"

        return {
            "name": f"{device_type}s >= {min_count}",
            "passed": passed,
            "actual": actual_msg,
            "required": f">= {min_count} Intel {device_type}(s) detected by OpenVINO",
            "category": category,
        }

    def _validate_device_max_count(
        self,
        devices: List[Dict[str, Any]],
        max_count: int,
        device_type: str,
        category: str,
        device_filter: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Validate maximum device count for Intel devices detected by OpenVINO.

        Args:
            devices: List of device dictionaries
            max_count: Maximum allowed device count
            device_type: Human-readable device type
            category: Category string for validation result
            device_filter: Optional filter to apply before vendor check

        Returns:
            Validation result dictionary
        """
        intel_devices, intel_devices_with_ov = self._get_intel_devices_with_openvino(devices, device_filter)
        passed = len(intel_devices_with_ov) <= max_count
        actual_msg = f"{len(intel_devices_with_ov)} Intel {device_type}(s) detected by OpenVINO"

        return {
            "name": f"{device_type}s <= {max_count}",
            "passed": passed,
            "actual": actual_msg,
            "required": f"<= {max_count} Intel {device_type}(s) detected by OpenVINO",
            "category": category,
        }

    def _validate_hardware(self, hw_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate hardware requirements using latest system info structure."""
        results = []
        hardware = self.system_info.get("hardware", {})

        # CPU Xeon requirement
        if "cpu_xeon_required" in hw_requirements and hw_requirements["cpu_xeon_required"]:
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

        # CPU Core requirement (includes Ultra Desktop)
        if "cpu_core_required" in hw_requirements and hw_requirements["cpu_core_required"]:
            cpu_info = hardware.get("cpu", {})
            cpu_brand = cpu_info.get("brand", "")
            # Core requirement matches: "Core" keyword OR Ultra Desktop
            # Ultra Desktop falls under mainstream/core category
            is_core = "core" in cpu_brand.lower()
            is_ultra_desktop = self._is_ultra_desktop_cpu(cpu_brand) if "ultra" in cpu_brand.lower() else False
            passed = is_core or is_ultra_desktop
            results.append(
                {
                    "name": "Intel Core CPU",
                    "passed": passed,
                    "actual": cpu_brand,
                    "required": "Intel Core CPU (includes Ultra Desktop)",
                    "category": "hardware.cpu.core",
                }
            )

        # CPU Ultra requirement
        if "cpu_ultra_required" in hw_requirements and hw_requirements["cpu_ultra_required"]:
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

        # CPU Ultra Mobile requirement (suffixes: H, U, V, HX, P)
        if "cpu_ultra_mobile_required" in hw_requirements and hw_requirements["cpu_ultra_mobile_required"]:
            cpu_info = hardware.get("cpu", {})
            cpu_brand = cpu_info.get("brand", "")
            is_ultra_mobile = self._is_ultra_mobile_cpu(cpu_brand)
            results.append(
                {
                    "name": "Intel Ultra Mobile CPU",
                    "passed": is_ultra_mobile,
                    "actual": cpu_brand,
                    "required": "Intel Ultra CPU with mobile suffix (H, U, V, HX, P)",
                    "category": "hardware.cpu.ultra.mobile",
                }
            )

        # CPU Entry requirement (N-series, Atom x6000/x7000, Processor N-series)
        if "cpu_entry_required" in hw_requirements and hw_requirements["cpu_entry_required"]:
            cpu_info = hardware.get("cpu", {})
            cpu_brand = cpu_info.get("brand", "")
            is_entry = self._is_entry_cpu(cpu_brand)
            results.append(
                {
                    "name": "Intel Entry CPU",
                    "passed": is_entry,
                    "actual": cpu_brand,
                    "required": "Intel Entry CPU (N-series, Atom x6000/x7000, or Processor N-series)",
                    "category": "hardware.cpu.entry",
                }
            )

        # CPU Entry exclusion (for mainstream tier to exclude entry-level Core processors)
        if "cpu_entry_excluded" in hw_requirements and hw_requirements["cpu_entry_excluded"]:
            cpu_info = hardware.get("cpu", {})
            cpu_brand = cpu_info.get("brand", "")
            is_entry = self._is_entry_cpu(cpu_brand)
            # Pass if CPU is NOT entry-level (inverted logic)
            passed = not is_entry
            results.append(
                {
                    "name": "Entry CPU excluded",
                    "passed": passed,
                    "actual": cpu_brand,
                    "required": "Not an entry-level CPU (excludes N-series, Atom x6000/x7000, Processor N-series)",
                    "category": "hardware.cpu.entry.excluded",
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

        # Memory minimum free requirement
        if "memory_min_gb" in hw_requirements:
            min_memory = hw_requirements["memory_min_gb"]
            memory_info = hardware.get("memory", {})
            total_bytes = memory_info.get("total", 0)
            available_bytes = memory_info.get("available", 0)
            actual_memory_gb = total_bytes / (1000**3)  # Use GB (1000^3) standard
            available_memory_gb = available_bytes / (1000**3)
            passed = available_memory_gb >= min_memory
            results.append(
                {
                    "name": f"Available memory >= {min_memory} GB",
                    "passed": passed,
                    "actual": f"{available_memory_gb:.0f} GB",
                    "required": f">= {min_memory} GB",
                    "category": "hardware.memory.available",
                }
            )

        # Memory minimum total requirement
        if "memory_total_min_gb" in hw_requirements:
            min_total_memory = hw_requirements["memory_total_min_gb"]
            memory_info = hardware.get("memory", {})
            total_bytes = memory_info.get("total", 0)
            available_bytes = memory_info.get("available", 0)
            actual_memory_gb = total_bytes / (1000**3)  # Use GB (1000^3) standard
            passed = actual_memory_gb >= min_total_memory
            results.append(
                {
                    "name": f"Total memory >= {min_total_memory} GB",
                    "passed": passed,
                    "actual": f"{actual_memory_gb:.0f} GB",
                    "required": f">= {min_total_memory} GB",
                    "category": "hardware.memory.total",
                }
            )

        # Storage minimum free requirement
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

        # Storage minimum total requirement
        if "storage_total_min_gb" in hw_requirements:
            min_total_storage = hw_requirements["storage_total_min_gb"]
            storage_info = hardware.get("storage", {})
            total_bytes = storage_info.get("total_size", 0)
            actual_total_storage_gb = total_bytes / (1000**3)  # Use GB (1000^3) standard
            passed = actual_total_storage_gb >= min_total_storage
            results.append(
                {
                    "name": f"Total storage >= {min_total_storage} GB",
                    "passed": passed,
                    "actual": f"{actual_total_storage_gb:.0f} GB",
                    "required": f">= {min_total_storage} GB",
                    "category": "hardware.storage.total",
                }
            )

        # GPU requirement (either iGPU or dGPU)
        if "gpu_required" in hw_requirements and hw_requirements["gpu_required"]:
            gpu_info = hardware.get("gpu", {})
            devices = gpu_info.get("devices", [])

            # Check for Intel devices with OpenVINO support (either integrated or discrete)
            intel_devices = [dev for dev in devices if dev.get("vendor_id", "").lower() == "8086"]
            intel_devices_with_ov = [dev for dev in intel_devices if "openvino" in dev]

            passed = len(intel_devices_with_ov) > 0

            if not passed:
                if len(intel_devices) > 0:
                    actual_msg = f"Found {len(intel_devices)} Intel GPU(s) but not detected by OpenVINO"
                elif any(dev.get("vendor_id", "").lower() != "8086" for dev in devices):
                    actual_msg = "Found non-Intel GPU(s) only"
                else:
                    actual_msg = "No GPU found"
            else:
                # Count integrated and discrete GPUs
                igpu_count = len([dev for dev in intel_devices_with_ov if not dev.get("is_discrete", True)])
                dgpu_count = len([dev for dev in intel_devices_with_ov if dev.get("is_discrete", False)])
                if igpu_count > 0 and dgpu_count > 0:
                    actual_msg = f"{igpu_count} Intel iGPU(s) and {dgpu_count} Intel dGPU(s) detected by OpenVINO"
                elif igpu_count > 0:
                    actual_msg = f"{igpu_count} Intel iGPU(s) detected by OpenVINO"
                else:
                    actual_msg = f"{dgpu_count} Intel dGPU(s) detected by OpenVINO"

            results.append(
                {
                    "name": "GPU required",
                    "passed": passed,
                    "actual": actual_msg,
                    "required": "At least 1 Intel GPU (iGPU or dGPU) detected by OpenVINO",
                    "category": "hardware.gpu.required",
                }
            )

        # Integrated GPU requirement
        if "igpu_required" in hw_requirements and hw_requirements["igpu_required"]:
            gpu_info = hardware.get("gpu", {})
            devices = gpu_info.get("devices", [])
            results.append(
                self._validate_device_required(
                    devices=devices,
                    device_type="iGPU",
                    category="hardware.gpu.integrated",
                    device_filter=lambda dev: not dev.get("is_discrete", True),
                    non_intel_check=lambda dev: not dev.get("is_discrete", True),
                )
            )

        # Discrete GPU requirement
        if "dgpu_required" in hw_requirements and hw_requirements["dgpu_required"]:
            gpu_info = hardware.get("gpu", {})
            devices = gpu_info.get("devices", [])
            results.append(
                self._validate_device_required(
                    devices=devices,
                    device_type="dGPU",
                    category="hardware.gpu.discrete",
                    device_filter=lambda dev: dev.get("is_discrete", False),
                    non_intel_check=lambda dev: dev.get("is_discrete", False),
                )
            )

        # Discrete GPU minimum devices
        if "dgpu_min_devices" in hw_requirements:
            min_devices = hw_requirements["dgpu_min_devices"]
            gpu_info = hardware.get("gpu", {})
            devices = gpu_info.get("devices", [])
            results.append(
                self._validate_device_min_count(
                    devices=devices,
                    min_count=min_devices,
                    device_type="dGPU",
                    category="hardware.gpu.discrete_count_min",
                    device_filter=lambda dev: dev.get("is_discrete", False),
                    non_intel_check=lambda dev: dev.get("is_discrete", False),
                )
            )

        # Discrete GPU maximum devices
        if "dgpu_max_devices" in hw_requirements:
            max_devices = hw_requirements["dgpu_max_devices"]
            gpu_info = hardware.get("gpu", {})
            devices = gpu_info.get("devices", [])
            results.append(
                self._validate_device_max_count(
                    devices=devices,
                    max_count=max_devices,
                    device_type="dGPU",
                    category="hardware.gpu.discrete_count_max",
                    device_filter=lambda dev: dev.get("is_discrete", False),
                )
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
                    "category": "hardware.gpu.vram_min",
                }
            )

        # Discrete GPU maximum VRAM
        if "dgpu_max_vram_gb" in hw_requirements:
            max_vram = hw_requirements["dgpu_max_vram_gb"]
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

            passed = total_vram_gb <= max_vram
            results.append(
                {
                    "name": f"Discrete GPU VRAM <= {max_vram} GB",
                    "passed": passed,
                    "actual": f"{total_vram_gb:.1f} GB total",
                    "required": f"<= {max_vram} GB",
                    "category": "hardware.gpu.vram_max",
                }
            )

        # Discrete GPU minimum VRAM per device (each GPU must have minimum VRAM)
        if "dgpu_min_vram_per_device_gb" in hw_requirements:
            min_vram_per_device = hw_requirements["dgpu_min_vram_per_device_gb"]
            gpu_info = hardware.get("gpu", {})
            gpu_devices = gpu_info.get("devices", [])

            # Check each discrete GPU individually
            discrete_gpus = []
            insufficient_gpus = []
            for device in gpu_devices:
                if device.get("is_discrete", False):
                    openvino_info = device.get("openvino", {})
                    if openvino_info and "memory_gb" in openvino_info:
                        vram_gb = openvino_info["memory_gb"]
                        device_name = device.get("name", "Unknown")
                        discrete_gpus.append({"name": device_name, "vram_gb": vram_gb})
                        if vram_gb < min_vram_per_device:
                            insufficient_gpus.append(f"{device_name} ({vram_gb:.1f} GB)")
                        logger.debug(f"Discrete GPU {device_name}: {vram_gb:.1f} GB VRAM")

            passed = len(insufficient_gpus) == 0 and len(discrete_gpus) > 0
            actual_info = ", ".join([f"{gpu['name']}: {gpu['vram_gb']:.1f} GB" for gpu in discrete_gpus])
            if not actual_info:
                actual_info = "No discrete GPUs found"

            results.append(
                {
                    "name": f"Each discrete GPU VRAM >= {min_vram_per_device} GB",
                    "passed": passed,
                    "actual": actual_info,
                    "required": f"Each GPU >= {min_vram_per_device} GB",
                    "category": "hardware.gpu.vram_per_device_min",
                }
            )

        # Discrete GPU maximum VRAM per device (each GPU must not exceed maximum VRAM)
        if "dgpu_max_vram_per_device_gb" in hw_requirements:
            max_vram_per_device = hw_requirements["dgpu_max_vram_per_device_gb"]
            gpu_info = hardware.get("gpu", {})
            gpu_devices = gpu_info.get("devices", [])

            # Check each discrete GPU individually
            discrete_gpus = []
            excessive_gpus = []
            for device in gpu_devices:
                if device.get("is_discrete", False):
                    openvino_info = device.get("openvino", {})
                    if openvino_info and "memory_gb" in openvino_info:
                        vram_gb = openvino_info["memory_gb"]
                        device_name = device.get("name", "Unknown")
                        discrete_gpus.append({"name": device_name, "vram_gb": vram_gb})
                        if vram_gb > max_vram_per_device:
                            excessive_gpus.append(f"{device_name} ({vram_gb:.1f} GB)")
                        logger.debug(f"Discrete GPU {device_name}: {vram_gb:.1f} GB VRAM")

            passed = len(excessive_gpus) == 0 and len(discrete_gpus) > 0
            actual_info = ", ".join([f"{gpu['name']}: {gpu['vram_gb']:.1f} GB" for gpu in discrete_gpus])
            if not actual_info:
                actual_info = "No discrete GPUs found"

            results.append(
                {
                    "name": f"Each discrete GPU VRAM <= {max_vram_per_device} GB",
                    "passed": passed,
                    "actual": actual_info,
                    "required": f"Each GPU <= {max_vram_per_device} GB",
                    "category": "hardware.gpu.vram_per_device_max",
                }
            )

        # NPU requirement
        if "npu_required" in hw_requirements and hw_requirements["npu_required"]:
            npu_info = hardware.get("npu", {})
            devices = npu_info.get("devices", [])
            results.append(
                self._validate_device_required(
                    devices=devices,
                    device_type="NPU",
                    category="hardware.npu.required",
                    non_intel_check=lambda dev: dev.get("class_name", "").lower() == "processing accelerators",
                )
            )

        # NPU minimum devices
        if "npu_min_devices" in hw_requirements:
            min_devices = hw_requirements["npu_min_devices"]
            npu_info = hardware.get("npu", {})
            devices = npu_info.get("devices", [])
            results.append(
                self._validate_device_min_count(
                    devices=devices,
                    min_count=min_devices,
                    device_type="NPU",
                    category="hardware.npu.count",
                    non_intel_check=lambda dev: dev.get("class_name", "").lower() == "processing accelerators",
                )
            )

        # NPU maximum devices
        if "npu_max_devices" in hw_requirements:
            max_devices = hw_requirements["npu_max_devices"]
            npu_info = hardware.get("npu", {})
            devices = npu_info.get("devices", [])
            results.append(
                self._validate_device_max_count(
                    devices=devices,
                    max_count=max_devices,
                    device_type="NPU",
                    category="hardware.npu.count_max",
                )
            )

        # Integrated GPU minimum devices
        if "igpu_min_devices" in hw_requirements:
            min_devices = hw_requirements["igpu_min_devices"]
            gpu_info = hardware.get("gpu", {})
            devices = gpu_info.get("devices", [])
            results.append(
                self._validate_device_min_count(
                    devices=devices,
                    min_count=min_devices,
                    device_type="iGPU",
                    category="hardware.gpu.integrated_count_min",
                    device_filter=lambda dev: not dev.get("is_discrete", True),
                    non_intel_check=lambda dev: not dev.get("is_discrete", True),
                )
            )

        # Integrated GPU maximum devices
        if "igpu_max_devices" in hw_requirements:
            max_devices = hw_requirements["igpu_max_devices"]
            gpu_info = hardware.get("gpu", {})
            devices = gpu_info.get("devices", [])
            results.append(
                self._validate_device_max_count(
                    devices=devices,
                    max_count=max_devices,
                    device_type="iGPU",
                    category="hardware.gpu.integrated_count_max",
                    device_filter=lambda dev: not dev.get("is_discrete", True),
                )
            )

        return results

    def _validate_software(self, sw_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            has_docker_package = "docker-ce" in packages and packages["docker-ce"]
            docker_command_available = self._check_docker_available()
            docker_available = has_docker_package or docker_command_available

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
                is_installed = package in packages and packages[package]  # Check package exists and has version
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
                is_installed = package in packages and packages[package]  # Check package exists and has version
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

    def _is_ultra_mobile_cpu(self, cpu_brand: str) -> bool:
        """
        Check if CPU is Intel Ultra Mobile (suffixes: H, U, V, HX, P).

        Mobile suffixes take priority over desktop suffixes when both present.
        Examples:
        - Intel Core Ultra 7 processor 165H (mobile)
        - Intel Core Ultra 7 processor 288V (mobile)
        - Intel Core Ultra 5 processor 125U (mobile)
        - Intel Core Ultra 9 processor 285HX (mobile - HX for high performance mobile)

        Args:
            cpu_brand: CPU brand string from system info

        Returns:
            True if CPU is Ultra with any mobile suffix
        """
        if not cpu_brand or "ultra" not in cpu_brand.lower():
            return False

        # Check for mobile suffixes: H, U, V, HX, P
        # Mobile suffixes have priority - if ANY mobile suffix present, it's mobile
        # Pattern: number followed by mobile suffix (e.g., "165H", "288V", "125U", "285HX")

        mobile_pattern = r"\d+(HX|H|U|V|P)\b"
        return bool(re.search(mobile_pattern, cpu_brand, re.IGNORECASE))

    def _is_ultra_desktop_cpu(self, cpu_brand: str) -> bool:
        """
        Check if CPU is Intel Ultra Desktop (suffixes: K, F, KF, T, or no suffix).

        Desktop includes:
        - Suffixes: K, F, KF, T
        - No suffix: Core Ultra 9 285, Core Ultra 7 265 (default to desktop)

        Mobile takes priority: If ANY mobile suffix (H, U, V, HX, P) present, NOT desktop.

        Examples:
        - Intel Core Ultra 9 processor 285K (desktop with K)
        - Intel Core Ultra 7 processor 265KF (desktop with KF)
        - Intel Core Ultra 9 processor 285 (desktop, no suffix)
        - Intel Core Ultra 7 processor 265 (desktop, no suffix)

        Args:
            cpu_brand: CPU brand string from system info

        Returns:
            True if CPU is Ultra Desktop (with desktop suffix OR no suffix, excluding mobile)
        """
        if not cpu_brand or "ultra" not in cpu_brand.lower():
            return False

        # First check: if it has ANY mobile suffix, it's NOT desktop
        if self._is_ultra_mobile_cpu(cpu_brand):
            return False

        # Check for desktop suffixes: K, F, KF, T
        # Pattern: number followed by desktop suffix (e.g., "285K", "265KF")
        desktop_pattern = r"\d+(K|KF|F|T)\b"
        has_desktop_suffix = bool(re.search(desktop_pattern, cpu_brand, re.IGNORECASE))

        # Check for no suffix: "Ultra X processor YYY" where YYY is just numbers
        # Pattern: "ultra" followed by number, "processor", then just numbers (no suffix)
        no_suffix_pattern = r"ultra\s+\d+\s+processor\s+(\d+)\b"
        has_no_suffix = bool(re.search(no_suffix_pattern, cpu_brand, re.IGNORECASE))

        # Desktop if: has desktop suffix OR has no suffix (default to desktop)
        return has_desktop_suffix or has_no_suffix

    def _is_entry_cpu(self, cpu_brand: str) -> bool:
        """
        Check if CPU is Intel Entry level.

        Entry includes:
        - N-series: Core 3 N-series (e.g., Core i3-N305, N355)
        - Processor N-series: Intel Processor N150, N250
        - Atom x6000/x7000 series: Atom x6000E, x7000RE, etc.
        - Historical: Pentium, Celeron (pre-2023)

        Examples:
        - Intel Processor N150
        - Intel Core 3 Processor N355
        - Intel Atom x7000E
        - Intel Atom x6211E
        - Intel Core i3-N305
        - Intel Pentium Silver N6000
        - Intel Celeron N5105

        Args:
            cpu_brand: CPU brand string from system info

        Returns:
            True if CPU is entry-level
        """
        if not cpu_brand:
            return False

        brand_lower = cpu_brand.lower()

        # Check for N-series patterns

        # Pattern 1: "Processor N" followed by numbers
        if re.search(r"processor\s+n\d+", brand_lower):
            return True

        # Pattern 2: "Core 3" with N-series or any Core with N in model number
        if re.search(r"core\s+3.*n\d+", brand_lower):
            return True
        if re.search(r"core\s+i3-n\d+", brand_lower):
            return True
        if re.search(r"core.*\bn\d{3}\b", brand_lower):  # Core with N### pattern
            return True

        # Pattern 3: Atom x6000/x7000 series
        # Match "atom x" followed by 6 or 7, then digits, optionally followed by letters
        if re.search(r"atom.*\bx[67]\d{3}[a-z]*\b", brand_lower):
            return True

        # Pattern 4: Historical Pentium and Celeron
        if "pentium" in brand_lower or "celeron" in brand_lower:
            return True

        return False

    def _check_docker_available(self) -> bool:
        """Check if Docker is available on the system."""
        try:
            result = run_command(["docker", "--version"], timeout=5)
            return result.success
        except (FileNotFoundError, OSError):
            return False

    def _log_fix_suggestions(self, failed_checks: List[Dict[str, Any]], context: str = "system validator") -> None:
        """Log user-friendly suggestions for fixing failed system requirements."""
        from sysagent.utils.testing.validation_suggestions import log_validation_fix_suggestions

        log_validation_fix_suggestions(failed_checks, context=context)


def validate_system_requirements(requirements: Dict[str, Any], cache_dir: Optional[str] = None) -> Dict[str, Any]:
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


def check_system_ready_for_tests(requirements: Dict[str, Any], cache_dir: Optional[str] = None) -> bool:
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
