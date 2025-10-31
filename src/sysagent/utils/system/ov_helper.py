# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
OpenVINO helper utilities for the core framework.

This module provides utilities to collect OpenVINO device information,
optimized for Intel platforms including CPU and GPU devices.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

# Setup logger
logger = logging.getLogger(__name__)

# Try to import OpenVINO
HAVE_OPENVINO = False
try:
    import openvino

    HAVE_OPENVINO = True
except ImportError:
    logger.warning("OpenVINO package not found. GPU and OpenVINO CPU info will be limited.")


def is_openvino_available() -> bool:
    """
    Check if OpenVINO is available.

    Returns:
        bool: True if OpenVINO is available, False otherwise
    """
    return HAVE_OPENVINO


def serialize_openvino_value(val: Any) -> Any:
    """
    Serialize OpenVINO values for JSON storage.

    Args:
        val: OpenVINO value to serialize

    Returns:
        JSON-serializable value
    """
    # Handle common JSON types
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    # Lists or tuples - serialize each item
    if isinstance(val, (list, tuple)):
        return [serialize_openvino_value(v) for v in val]
    # Anything else - fallback to str()
    return str(val)


def get_openvino_device_properties(device_name: str, core=None) -> Dict[str, Any]:
    """
    Get OpenVINO device properties.

    Args:
        device_name: Name of the device (e.g., "CPU", "GPU.0", "NPU")
        core: OpenVINO Core instance (optional)

    Returns:
        Dict containing device properties with quick_access field
    """
    if not HAVE_OPENVINO:
        return {"error": "OpenVINO not available"}

    try:
        if core is None:
            core = openvino.Core()

        # Get all device properties
        all_properties = {}
        quick_access = {"device_name": device_name}

        # Properties to skip when collecting device properties
        skip_properties = {
            "CACHE_ENCRYPTION_CALLBACKS",
            "SUPPORTED_PROPERTIES",
            "AVAILABLE_DEVICES",
        }

        # Get basic device properties
        try:
            supported_properties = core.get_property(device_name, "SUPPORTED_PROPERTIES")
            logger.debug(f"OpenVINO device {device_name} supported properties: {len(supported_properties)}")

            for prop in supported_properties:
                try:
                    prop_name = str(prop)
                    if prop_name in skip_properties:
                        continue  # Skip these to avoid recursion

                    prop_value = core.get_property(device_name, prop)
                    serialized_value = serialize_openvino_value(prop_value)
                    all_properties[prop_name] = serialized_value

                    # Store key properties in quick_access for easy access
                    if prop_name == "DEVICE_TYPE":
                        quick_access["device_type"] = serialized_value
                    elif prop_name == "FULL_DEVICE_NAME":
                        quick_access["full_device_name"] = serialized_value
                    elif prop_name == "DEVICE_UUID":
                        quick_access["device_uuid"] = serialized_value
                    elif prop_name == "DEVICE_ID":
                        quick_access["device_id"] = serialized_value
                    elif prop_name == "DEVICE_ARCHITECTURE":
                        quick_access["device_architecture"] = serialized_value

                except Exception as e:
                    logger.debug(f"Could not get property {prop} for {device_name}: {e}")

        except Exception as e:
            logger.debug(f"Could not get supported properties for {device_name}: {e}")

        # Build complete device info structure
        device_info = {
            "device": {"name": device_name, "all_properties": all_properties},
            "quick_access": quick_access,
        }

        # Enhance device info with additional metadata
        _enhance_device_info(device_info, device_name, core)

        return device_info

    except Exception as e:
        logger.warning(f"Error getting OpenVINO device properties for {device_name}: {e}")
        return {"device_name": device_name, "error": str(e)}


def _enhance_device_info(device_info: Dict[str, Any], device_name: str, core) -> None:
    """
    Enhance device info with additional metadata.

    Args:
        device_info: Device info dict to enhance (has quick_access field)
        device_name: Name of the device
        core: OpenVINO Core instance
    """
    try:
        quick_access = device_info.get("quick_access", {})
        all_properties = device_info.get("device", {}).get("all_properties", {})

        # Add device UUID for GPU devices
        if device_name.startswith("GPU"):
            try:
                if "DEVICE_UUID" not in quick_access and "DEVICE_UUID" in all_properties:
                    quick_access["device_uuid"] = all_properties["DEVICE_UUID"]
            except Exception:
                pass

        # Add device architecture for supported devices
        if device_name.startswith(("GPU", "NPU")):
            try:
                if "DEVICE_ARCHITECTURE" not in quick_access and "DEVICE_ARCHITECTURE" in all_properties:
                    quick_access["device_architecture"] = all_properties["DEVICE_ARCHITECTURE"]
            except Exception:
                pass

        # Add memory info for relevant devices
        if device_name.startswith("GPU"):
            try:
                if "GPU_MEMORY_STATISTICS" in all_properties:
                    quick_access["memory_statistics"] = all_properties["GPU_MEMORY_STATISTICS"]
            except Exception:
                pass
        elif device_name.startswith("NPU"):
            try:
                if "NPU_DEVICE_TOTAL_MEM_SIZE" in all_properties:
                    quick_access["memory_statistics"] = all_properties["NPU_DEVICE_TOTAL_MEM_SIZE"]
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"Error enhancing device info for {device_name}: {e}")


def collect_openvino_devices() -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Collect OpenVINO device information.

    Returns:
        Tuple of (gpu_info, cpu_openvino_info, npu_info)
    """
    gpu_info = {"devices": []}
    cpu_openvino_info = None
    npu_info = {"devices": []}

    if not HAVE_OPENVINO:
        logger.debug("OpenVINO not available, skipping device collection")
        return gpu_info, cpu_openvino_info, npu_info

    try:
        core = openvino.Core()
        available_devices = core.available_devices
        logger.debug(f"OpenVINO detected devices: {available_devices}")

        # Process each device
        for device_name in available_devices:
            device_info = get_openvino_device_properties(device_name, core)

            # For CPU devices, store separately to add to CPU info later
            if device_name == "CPU":
                cpu_openvino_info = device_info
            # For NPU devices, add to NPU devices list
            elif device_name.startswith("NPU"):
                npu_info["devices"].append(device_info)
            # For GPU devices, add to GPU devices list
            elif device_name.startswith("GPU"):
                gpu_info["devices"].append(device_info)
            else:
                # Other devices for backward compatibility
                gpu_info["devices"].append(device_info)

        return gpu_info, cpu_openvino_info, npu_info

    except Exception as e:
        logger.warning(f"Error collecting device info with OpenVINO: {e}")
        return gpu_info, None, npu_info


def get_openvino_cpu_info() -> Optional[Dict[str, Any]]:
    """
    Get OpenVINO CPU device information.

    Returns:
        CPU device info dict or None if not available
    """
    if not HAVE_OPENVINO:
        return None

    try:
        core = openvino.Core()
        return get_openvino_device_properties("CPU", core)
    except Exception as e:
        logger.warning(f"Error getting OpenVINO CPU info: {e}")
        return None


def get_openvino_gpu_devices() -> List[Dict[str, Any]]:
    """
    Get OpenVINO GPU device information.

    Returns:
        List of GPU device info dicts
    """
    gpu_info, _, _ = collect_openvino_devices()
    return gpu_info.get("devices", [])


def get_openvino_npu_devices() -> List[Dict[str, Any]]:
    """
    Get OpenVINO NPU device information.

    Returns:
        List of NPU device info dicts
    """
    _, _, npu_info = collect_openvino_devices()
    return npu_info.get("devices", [])


def get_available_devices(device: str = None, device_type: str = None, filter_list: List[str] = []) -> List[str]:
    """
    Get list of available OpenVINO devices.

    Args:
        device: Specific device to check (optional)
        device_type: Device type to filter by (optional)
        filter_list: List of devices to exclude

    Returns:
        List of available device names
    """
    if not HAVE_OPENVINO:
        return []

    try:
        core = openvino.Core()
        available_devices = core.available_devices

        # Apply filters
        filtered_devices = [d for d in available_devices if d not in filter_list]

        if device:
            filtered_devices = [d for d in filtered_devices if device in d]

        if device_type:
            filtered_devices = [d for d in filtered_devices if d.startswith(device_type)]

        return filtered_devices

    except Exception as e:
        logger.warning(f"Error getting available devices: {e}")
        return []


def get_openvino_device_type(device_id: str) -> str:
    """
    Get the OpenVINO DEVICE_TYPE property for a given device id (e.g., "CPU", "GPU.0").
    Returns the device type string (e.g., "Type.INTEGRATED", "Type.DISCRETE").
    """
    if not HAVE_OPENVINO:
        return None
    try:
        core = openvino.Core()
        if device_id not in core.available_devices:
            logger.warning(f"Device {device_id} not found in available OpenVINO devices: {core.available_devices}")
            return None
        device_type = core.get_property(device_id, "DEVICE_TYPE")
        logger.debug(f"OpenVINO DEVICE_TYPE for {device_id}: {device_type}")
        return str(device_type)
    except Exception as e:
        logger.debug(f"Error getting DEVICE_TYPE for {device_id}: {e}")
        return None


def get_available_devices_by_category(
    device_categories: List[str], filter_list: List[str] = []
) -> Dict[str, Dict[str, str]]:
    """
    Get available devices organized by category.

    Supports hetero-dgpu category which returns a virtual HETERO device combining all available dGPUs.

    Args:
        device_categories: List of device categories to include (cpu, igpu, dgpu, npu, hetero-dgpu)
        filter_list: List of devices to exclude

    Returns:
        Dict with device ID as key and device info as value.
        For hetero-dgpu, returns a virtual device with format "HETERO:GPU.0,GPU.1,..."
    """
    device_dict = {}

    if not HAVE_OPENVINO:
        return device_dict

    try:
        core = openvino.Core()
        available_devices = get_available_devices(filter_list=filter_list)
        logger.debug(f"Available devices: {available_devices}")

        # First pass: collect discrete GPUs for hetero-dgpu
        discrete_gpus = []

        for ov_device in available_devices:
            if ov_device in filter_list:
                continue

            # Get device type and full name
            device_type = None
            full_name = None
            try:
                device_type = str(core.get_property(ov_device, "DEVICE_TYPE"))
                full_name = str(core.get_property(ov_device, "FULL_DEVICE_NAME"))
            except Exception as e:
                logger.debug(f"Error getting properties for {ov_device}: {e}")
                continue

            # Collect discrete GPUs for hetero support
            if ov_device.upper().startswith("GPU") and "discrete" in device_type.lower():
                discrete_gpus.append(
                    {
                        "device_id": ov_device,
                        "device_type": device_type,
                        "full_name": full_name,
                    }
                )

            # Check if device matches requested categories
            include_device = False

            # CPU device
            if "cpu" in device_categories and ov_device.upper() == "CPU":
                include_device = True

            # NPU device
            elif "npu" in device_categories and ov_device.upper() == "NPU":
                include_device = True

            # GPU devices - check if integrated or discrete
            elif ov_device.upper().startswith("GPU"):
                if "igpu" in device_categories and "integrated" in device_type.lower():
                    include_device = True
                elif "dgpu" in device_categories and "discrete" in device_type.lower():
                    include_device = True

            if include_device:
                device_dict[ov_device] = {
                    "device_type": device_type,
                    "full_name": full_name,
                }

        # Handle hetero-dgpu category: create virtual HETERO device combining all dGPUs
        if "hetero-dgpu" in device_categories and len(discrete_gpus) >= 2:
            # Build HETERO device string: "HETERO:GPU.0,GPU.1,..."
            hetero_device_list = [gpu["device_id"] for gpu in discrete_gpus]
            hetero_device_id = "HETERO:" + ",".join(hetero_device_list)

            # Create combined device name
            gpu_names = [gpu["full_name"] for gpu in discrete_gpus]
            combined_name = f"HETERO ({len(discrete_gpus)} dGPUs: {', '.join(gpu_names)})"

            device_dict[hetero_device_id] = {
                "device_type": "Type.DISCRETE",  # HETERO is considered discrete
                "full_name": combined_name,
                "is_hetero": True,
                "hetero_devices": hetero_device_list,
            }
            logger.info(f"Created HETERO device: {hetero_device_id}")
        elif "hetero-dgpu" in device_categories and len(discrete_gpus) < 2:
            logger.warning(
                f"hetero-dgpu category requested but only {len(discrete_gpus)} "
                f"discrete GPU(s) found. Need at least 2 for HETERO."
            )

        return device_dict

    except Exception as e:
        logger.warning(f"Error getting devices by category: {e}")
        return device_dict
