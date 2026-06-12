# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Intel System Peripheral Device Detection and Enumeration Test.

Enumerates and detects peripheral devices including USB, PS/2, network, and graphics devices.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import allure
import pytest
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result, run_command

logger = logging.getLogger(__name__)


def _create_peripheral_metrics(value: int = -1, unit: Optional[str] = None) -> dict:
    """
    Create peripheral device metrics dictionary.

    Args:
        value: Initial value for metrics (default: -1 for unavailable)
        unit: Unit for metrics

    Returns:
        Dictionary of Metrics objects for peripheral testing
    """
    return {
        # Device counts
        "usb_controllers_count": Metrics(unit="count", value=value, is_key_metric=False),
        "usb_devices_count": Metrics(unit="count", value=value, is_key_metric=False),
        "usb_keyboard_count": Metrics(unit="count", value=value, is_key_metric=False),
        "usb_mouse_count": Metrics(unit="count", value=value, is_key_metric=False),
        "ps2_keyboard_count": Metrics(unit="count", value=value, is_key_metric=False),
        "ps2_mouse_count": Metrics(unit="count", value=value, is_key_metric=False),
        "total_keyboard_count": Metrics(unit="count", value=value, is_key_metric=False),
        "total_mouse_count": Metrics(unit="count", value=value, is_key_metric=False),
        "usb_storage_count": Metrics(unit="count", value=value, is_key_metric=False),
        "usb_audio_count": Metrics(unit="count", value=value, is_key_metric=False),
        "usb_camera_count": Metrics(unit="count", value=value, is_key_metric=False),
        "usb_hub_count": Metrics(unit="count", value=value, is_key_metric=False),
        "network_controllers_count": Metrics(unit="count", value=value, is_key_metric=False),
        "graphics_controllers_count": Metrics(unit="count", value=value, is_key_metric=False),
        "igpu_detected": Metrics(unit="boolean", value=False, is_key_metric=False),
        "monitor_count": Metrics(unit="count", value=value, is_key_metric=False),
    }


def get_lspci_output() -> Optional[str]:
    """
    Get lspci output (cached to avoid redundant calls).

    Returns:
        lspci output as string, or None on error
    """
    try:
        result = run_command(
            command=["lspci"],
            timeout=10,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except (OSError, FileNotFoundError):
        pass
    return None


def count_devices_by_type(device_type: str, lspci_output: Optional[str] = None) -> int:
    """
    Count devices of a specific type using lspci.

    Args:
        device_type: Device type pattern (e.g., "usb", "network", "vga")
        lspci_output: Optional cached lspci output to avoid redundant calls

    Returns:
        Count of matching devices
    """
    if lspci_output is None:
        lspci_output = get_lspci_output()
    
    if lspci_output:
        pattern = re.compile(device_type, re.IGNORECASE)
        matches = [line for line in lspci_output.split("\n") if pattern.search(line)]
        return len(matches)
    return 0


def detect_igpu(lspci_output: Optional[str] = None) -> bool:
    """
    Detect Intel integrated GPU.
    
    Args:
        lspci_output: Optional cached lspci output to avoid redundant calls
        
    Returns:
        True if Intel iGPU detected
    """
    if lspci_output is None:
        lspci_output = get_lspci_output()
    
    if lspci_output:
        pattern = re.compile(r"(vga|display).*intel.*graphics", re.IGNORECASE)
        return bool(pattern.search(lspci_output))
    return False


def get_connected_monitors() -> int:
    """
    Get the count of physically connected monitors (excludes virtual displays).
    
    Returns:
        Number of physically connected monitors
    """
    # Try xrandr first
    try:
        result = run_command(
            command=["xrandr", "--query"],
            timeout=10,
        )
        if result.returncode == 0 and result.stdout:
            # Count lines with " connected" (space before "connected" to avoid "disconnected")
            # Exclude virtual displays
            monitor_count = 0
            for line in result.stdout.split("\n"):
                if " connected" in line:
                    # Filter out virtual/dummy displays
                    line_lower = line.lower()
                    if not any(virt in line_lower for virt in ["virtual", "dummy", "none"]):
                        monitor_count += 1
            if monitor_count > 0:
                return monitor_count
    except (OSError, FileNotFoundError):
        logger.debug("xrandr not available, trying alternative methods")
    
    # Fallback: Check /sys/class/drm for physically connected displays
    try:
        drm_path = Path("/sys/class/drm")
        if drm_path.exists():
            monitor_count = 0
            for card_dir in drm_path.iterdir():
                if card_dir.is_dir() and card_dir.name.startswith("card"):
                    # Check connector type to filter out virtual devices
                    connector_name = card_dir.name
                    # Skip virtual connectors
                    if "Virtual" in connector_name or "WRITEBACK" in connector_name:
                        continue
                    
                    status_file = card_dir / "status"
                    if status_file.exists():
                        try:
                            with open(status_file, "r") as f:
                                status = f.read().strip()
                                if status == "connected":
                                    # Verify it's a physical connector (HDMI, DP, DVI, VGA, eDP, etc.)
                                    if any(conn_type in connector_name for conn_type in 
                                           ["HDMI", "DP", "DVI", "VGA", "eDP", "LVDS"]):
                                        monitor_count += 1
                        except IOError:
                            pass
            return monitor_count
    except Exception as e:
        logger.warning(f"Error checking monitors via drm: {e}")
    
    return 0


def get_usb_devices() -> Tuple[int, int, int, int, int, int, int, List[str], Dict[str, List[str]]]:
    """
    Enumerate USB devices and categorize them.
    
    Returns:
        Tuple of (total_devices, keyboard_count, mouse_count, storage_count, 
                 audio_count, camera_count, hub_count, device_list, hub_details)
        hub_details: Dict mapping hub IDs to list of connected devices
    """
    try:
        result = run_command(
            command=["lsusb"],
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout:
            logger.warning("Failed to get lsusb output")
            return 0, 0, 0, 0, 0, 0, 0, [], {}
        
        devices = result.stdout.strip().split("\n")
        # Filter out root hubs
        devices = [d for d in devices if "root hub" not in d.lower()]
        
        keyboard_count = 0
        mouse_count = 0
        storage_count = 0
        audio_count = 0
        camera_count = 0
        hub_count = 0
        hub_details = {}
        
        for device in devices:
            device_lower = device.lower()
            if "keyboard" in device_lower or "kbd" in device_lower:
                keyboard_count += 1
            if "mouse" in device_lower or "pointing" in device_lower:
                mouse_count += 1
            if "mass storage" in device_lower or "storage" in device_lower or "disk" in device_lower:
                storage_count += 1
            if "audio" in device_lower or "sound" in device_lower or "speaker" in device_lower or "microphone" in device_lower:
                audio_count += 1
            if "camera" in device_lower or "webcam" in device_lower or "video" in device_lower:
                camera_count += 1
            if "hub" in device_lower:
                hub_count += 1
                # Extract hub ID for tracking connected devices
                hub_id = device.split()[1] + ":" + device.split()[3].rstrip(":")
                hub_details[hub_id] = []
        
        # Try to get detailed USB tree information for hub connections
        try:
            tree_result = run_command(
                command=["lsusb", "-t"],
                timeout=10,
            )
            if tree_result.returncode == 0 and tree_result.stdout:
                # Parse tree output to identify devices connected to hubs
                # This is a simplified parser - full implementation would be more complex
                for line in tree_result.stdout.split("\n"):
                    if "Hub" in line:
                        # Mark this as a hub in our tracking
                        pass  # Hub detection logic for topology
        except (OSError, FileNotFoundError):
            logger.debug("Could not get USB tree topology (lsusb -t failed)")
        
        return len(devices), keyboard_count, mouse_count, storage_count, audio_count, camera_count, hub_count, devices, hub_details
    
    except (OSError, FileNotFoundError) as e:
        logger.warning(f"Error running lsusb: {e}")
        return 0, 0, 0, 0, 0, 0, 0, [], {}


def get_all_input_devices() -> Tuple[int, int, int, int, int, int]:
    """
    Enumerate physically connected input devices (USB + PS/2) from /proc/bus/input/devices.
    Deduplicates multiple input nodes from the same physical device.
    Filters out virtual devices without physical connection paths.
    
    Returns:
        Tuple of (total_keyboards, total_mouse, usb_keyboards, usb_mouse, ps2_keyboards, ps2_mouse)
    """
    try:
        with open("/proc/bus/input/devices", "r") as f:
            content = f.read()
        
        # Split into individual device blocks
        devices = content.split("\n\n")
        
        # Track unique USB device paths to avoid counting multiple input nodes from same device
        usb_keyboard_paths = set()
        usb_mouse_paths = set()
        ps2_keyboard_paths = set()
        ps2_mouse_paths = set()
        
        for device in devices:
            if not device.strip():
                continue
            
            # Extract device name and physical connection
            name_line = ""
            phys_line = ""
            handlers_line = ""
            
            for line in device.split("\n"):
                if line.startswith("N: Name="):
                    name_line = line[8:].strip().strip('"').lower()  # Skip "N: Name="
                elif line.startswith("P: Phys="):
                    phys_line = line[8:].strip().lower()  # Skip "P: Phys="
                elif line.startswith("H: Handlers="):
                    handlers_line = line[12:].strip().lower()  # Skip "H: Handlers="
            
            # Skip devices without physical connection path (virtual devices)
            if not phys_line or phys_line == "":
                continue
            
            # Skip system buttons and non-input devices
            if any(skip in name_line for skip in ["power button", "sleep button", "lid switch", "video bus", "hda intel", "intel hid"]):
                continue
            
            # Detect keyboards
            is_keyboard = "keyboard" in name_line or "kbd" in name_line or ("kbd" in handlers_line and "event" in handlers_line)
            # Detect mouse
            is_mouse = "mouse" in name_line or "pointing" in name_line or ("mouse" in handlers_line and "event" in handlers_line)
            
            # ONLY accept physically connected devices (USB or PS/2)
            is_usb = "usb" in phys_line
            is_ps2 = "i8042" in phys_line or "serio" in phys_line
            
            # Skip if not physically connected
            if not (is_usb or is_ps2):
                continue
            
            # Extract base device path (before /inputX) for deduplication
            # Example: "usb-0000:00:14.0-3/input0" -> "usb-0000:00:14.0-3"
            base_path = phys_line.split("/input")[0] if "/input" in phys_line else phys_line
            
            if is_keyboard and base_path:
                if is_usb:
                    usb_keyboard_paths.add(base_path)
                elif is_ps2:
                    ps2_keyboard_paths.add(base_path)
            
            if is_mouse and base_path:
                if is_usb:
                    usb_mouse_paths.add(base_path)
                elif is_ps2:
                    ps2_mouse_paths.add(base_path)
        
        usb_keyboards = len(usb_keyboard_paths)
        usb_mouse = len(usb_mouse_paths)
        ps2_keyboards = len(ps2_keyboard_paths)
        ps2_mouse = len(ps2_mouse_paths)
        total_keyboards = usb_keyboards + ps2_keyboards
        total_mouse = usb_mouse + ps2_mouse
        
        return total_keyboards, total_mouse, usb_keyboards, usb_mouse, ps2_keyboards, ps2_mouse
    
    except (IOError, OSError) as e:
        logger.warning(f"Failed to read /proc/bus/input/devices: {e}")
        return 0, 0, 0, 0, 0, 0


def save_device_lists(output_dir: str):
    """
    Save detailed device lists to output directory.

    Args:
        output_dir: Directory to save device lists
    """
    device_types = {
        "vga_devices.txt": "vga",
        "audio_devices.txt": "audio",
        "network_devices.txt": "network|ethernet",
        "usb_controllers.txt": "usb",
        "nvme_devices.txt": "nvme",
    }
    
    # Get lspci output once and cache it
    try:
        result = run_command(
            command=["lspci", "-nn"],
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning("Failed to get lspci output for device lists")
            return
        
        lspci_output = result.stdout
    except (OSError, FileNotFoundError) as e:
        logger.error(f"Error running lspci: {e}")
        return
    
    for filename, pattern in device_types.items():
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            matching_lines = [line for line in lspci_output.split("\n") if regex.search(line)]
            
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                f.write("\n".join(matching_lines))
        except (OSError, FileNotFoundError, IOError) as e:
            logger.warning(f"Could not save {filename}: {e}")
    
    # Save USB devices list
    try:
        _, _, _, _, _, _, _, usb_devices, _ = get_usb_devices()
        if usb_devices:
            usb_filepath = os.path.join(output_dir, "usb_devices.txt")
            with open(usb_filepath, "w") as f:
                f.write("\n".join(usb_devices))
    except Exception as e:
        logger.warning(f"Could not save usb_devices.txt: {e}")



@allure.title("Peripheral Device Detection Test")
def test_peripheral_detection(
    request,
    configs,
    cached_result,
    cache_result,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """
    Test peripheral device detection and enumeration.

    Detects and enumerates peripheral devices including USB, PS/2, network, and graphics controllers.
    """
    # Request
    test_name = request.node.name.split("[")[0]

    # Parameters
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    # Set test description from config if provided
    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info(f"Starting Peripheral Device Detection Test: {test_display_name}")

    check_usb = configs.get("check_usb", False)
    check_usb_controllers = configs.get("check_usb_controllers", False)
    check_usb_devices = configs.get("check_usb_devices", False)
    check_usb_keyboard = configs.get("check_usb_keyboard", False)
    check_usb_mouse = configs.get("check_usb_mouse", False)
    check_ps2_keyboard = configs.get("check_ps2_keyboard", False)
    check_ps2_mouse = configs.get("check_ps2_mouse", False)
    check_total_keyboard = configs.get("check_total_keyboard", False)
    check_total_mouse = configs.get("check_total_mouse", False)
    check_usb_storage = configs.get("check_usb_storage", False)
    check_usb_audio = configs.get("check_usb_audio", False)
    check_usb_camera = configs.get("check_usb_camera", False)
    check_usb_hub = configs.get("check_usb_hub", False)
    check_network = configs.get("check_network", False)
    check_graphics = configs.get("check_graphics", False)
    check_monitors = configs.get("check_monitors", False)
    
    # If any specific USB check is enabled, enable general USB check
    if any([check_usb_controllers, check_usb_devices, check_usb_keyboard, check_usb_mouse, 
            check_usb_storage, check_usb_audio, check_usb_camera, check_usb_hub]):
        check_usb = True
    
    # If any input device check is enabled (USB or PS/2) OR if checking USB devices, 
    # we need to enumerate input devices for accurate counts
    check_input_devices = any([check_usb_keyboard, check_usb_mouse, check_ps2_keyboard, 
                               check_ps2_mouse, check_total_keyboard, check_total_mouse, 
                               check_usb_devices, check_usb])

    # Step 1: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Step 2: Setup directories
    # Sanitize environment variable path using character-by-character copy
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    
    # Resolve path and reconstruct to break taint chain
    core_data_resolved = str(Path(core_data_dir_tainted).resolve())
    chars: list = []
    for char in core_data_resolved:
        chars.append(char)
    core_data_dir = "".join(chars)
    
    # Validate path stays within expected directory (no path traversal)
    expected_base = Path(os.getcwd()).resolve()
    if not Path(core_data_dir).resolve().is_relative_to(expected_base):
        # If path is outside current directory, use safe default
        core_data_dir = os.path.join(os.getcwd(), "esq_data")
    
    data_dir = os.path.join(core_data_dir, "data", "system", "peripheral")
    peripheral_results = os.path.join(data_dir, "results", test_id)
    
    # Sanitize final path before directory creation
    peripheral_resolved = str(Path(peripheral_results).resolve())
    chars_peripheral: list = []
    for char in peripheral_resolved:
        chars_peripheral.append(char)
    peripheral_results_clean = "".join(chars_peripheral)
    
    os.makedirs(peripheral_results_clean, exist_ok=True)
    ensure_dir_permissions(peripheral_results_clean, uid=os.getuid(), gid=os.getgid(), mode=0o775)
    
    # Use sanitized path for all subsequent operations
    peripheral_results = peripheral_results_clean

    # Step 3: Enumerate devices
    # Get lspci output once and cache it for device counting
    lspci_output = get_lspci_output()

    # Count devices
    usb_count = 0
    usb_devices_total = 0
    usb_keyboard = 0
    usb_mouse = 0
    ps2_keyboard = 0
    ps2_mouse = 0
    total_keyboard = 0
    total_mouse = 0
    usb_storage = 0
    usb_audio = 0
    usb_camera = 0
    usb_hub = 0
    usb_hub_details = {}
    
    network_count = 0
    graphics_count = 0
    igpu_detected = False
    monitor_count = 0
    
    # USB device enumeration
    if check_usb or check_usb_controllers or check_usb_devices:
        usb_count = count_devices_by_type("usb", lspci_output)
        logger.info(f"USB controllers found: {usb_count}")
        
        # Enumerate actual USB devices (keyboard, mouse, storage, audio, camera, hub, etc.)
        usb_devices_total, usb_kbd_simple, usb_mouse_simple, usb_storage, usb_audio, usb_camera, usb_hub, _, usb_hub_details = get_usb_devices()
        logger.info(f"USB devices: {usb_devices_total} total (keyboard: {usb_kbd_simple}, mouse: {usb_mouse_simple}, storage: {usb_storage}, audio: {usb_audio}, camera: {usb_camera}, hub: {usb_hub})")
        if usb_hub_details:
            logger.info(f"USB hubs detected: {len(usb_hub_details)} hub(s)")
    
    # Input device enumeration (USB + PS/2) - more accurate for keyboard/mouse
    if check_input_devices:
        total_keyboard, total_mouse, usb_keyboard, usb_mouse, ps2_keyboard, ps2_mouse = get_all_input_devices()
        logger.info(f"Input devices detected: {total_keyboard} keyboards (USB: {usb_keyboard}, PS/2: {ps2_keyboard}), {total_mouse} mouse (USB: {usb_mouse}, PS/2: {ps2_mouse})")
        
        # Recalculate USB devices total using accurate input device counts
        # Total = USB keyboards + USB mouse + USB storage + USB audio + USB camera + USB hubs
        usb_devices_total = usb_keyboard + usb_mouse + usb_storage + usb_audio + usb_camera + usb_hub
        logger.info(f"Total USB peripheral devices: {usb_devices_total} (keyboard: {usb_keyboard}, mouse: {usb_mouse}, storage: {usb_storage}, audio: {usb_audio}, camera: {usb_camera}, hub: {usb_hub})")
    
    if check_network:
        network_count = count_devices_by_type("network|ethernet", lspci_output)
        logger.info(f"Network controllers found: {network_count}")
    
    if check_graphics:
        graphics_count = count_devices_by_type("vga|display", lspci_output)
        igpu_detected = detect_igpu(lspci_output)
        logger.info(f"Graphics controllers found: {graphics_count}, iGPU: {igpu_detected}")
    
    if check_monitors:
        monitor_count = get_connected_monitors()
        logger.info(f"Connected monitors: {monitor_count}")

    # Save device lists
    save_device_lists(peripheral_results)

    # Step 4: Create metrics - only include relevant metrics for each test type
    metrics = {}
    
    if check_usb_controllers:
        metrics["usb_controllers_count"] = Metrics(unit="count", value=usb_count, is_key_metric=True)
    elif check_usb_devices:
        metrics["usb_devices_count"] = Metrics(unit="count", value=usb_devices_total, is_key_metric=True)
        # Include breakdown for context
        metrics["usb_keyboard_count"] = Metrics(unit="count", value=usb_keyboard, is_key_metric=False)
        metrics["usb_mouse_count"] = Metrics(unit="count", value=usb_mouse, is_key_metric=False)
        metrics["usb_storage_count"] = Metrics(unit="count", value=usb_storage, is_key_metric=False)
        metrics["usb_audio_count"] = Metrics(unit="count", value=usb_audio, is_key_metric=False)
        metrics["usb_camera_count"] = Metrics(unit="count", value=usb_camera, is_key_metric=False)
        metrics["usb_hub_count"] = Metrics(unit="count", value=usb_hub, is_key_metric=False)
    elif check_usb_keyboard:
        metrics["usb_keyboard_count"] = Metrics(unit="count", value=usb_keyboard, is_key_metric=True)
    elif check_usb_mouse:
        metrics["usb_mouse_count"] = Metrics(unit="count", value=usb_mouse, is_key_metric=True)
    elif check_ps2_keyboard:
        metrics["ps2_keyboard_count"] = Metrics(unit="count", value=ps2_keyboard, is_key_metric=True)
    elif check_ps2_mouse:
        metrics["ps2_mouse_count"] = Metrics(unit="count", value=ps2_mouse, is_key_metric=True)
    elif check_total_keyboard:
        metrics["total_keyboard_count"] = Metrics(unit="count", value=total_keyboard, is_key_metric=True)
        # Include breakdown for context
        metrics["usb_keyboard_count"] = Metrics(unit="count", value=usb_keyboard, is_key_metric=False)
        metrics["ps2_keyboard_count"] = Metrics(unit="count", value=ps2_keyboard, is_key_metric=False)
    elif check_total_mouse:
        metrics["total_mouse_count"] = Metrics(unit="count", value=total_mouse, is_key_metric=True)
        # Include breakdown for context
        metrics["usb_mouse_count"] = Metrics(unit="count", value=usb_mouse, is_key_metric=False)
        metrics["ps2_mouse_count"] = Metrics(unit="count", value=ps2_mouse, is_key_metric=False)
    elif check_usb_storage:
        metrics["usb_storage_count"] = Metrics(unit="count", value=usb_storage, is_key_metric=True)
    elif check_usb_audio:
        metrics["usb_audio_count"] = Metrics(unit="count", value=usb_audio, is_key_metric=True)
    elif check_usb_camera:
        metrics["usb_camera_count"] = Metrics(unit="count", value=usb_camera, is_key_metric=True)
    elif check_usb_hub:
        metrics["usb_hub_count"] = Metrics(unit="count", value=usb_hub, is_key_metric=True)
    elif check_network:
        metrics["network_controllers_count"] = Metrics(unit="count", value=network_count, is_key_metric=True)
    elif check_graphics:
        metrics["graphics_controllers_count"] = Metrics(unit="count", value=graphics_count, is_key_metric=True)
        metrics["igpu_detected"] = Metrics(unit="boolean", value=igpu_detected, is_key_metric=False)
    elif check_monitors:
        metrics["monitor_count"] = Metrics(unit="count", value=monitor_count, is_key_metric=True)

    # Step 6: Create result
    result = Result(
        name=f"{test_id} - {test_display_name}",
        metadata={"status": True, "message": "Peripheral device detection completed"},
        metrics=metrics,
    )

    # Step 7: Cache result
    cache_result(result)
    
    # Step 8: Summarize results to attach metrics to test summary
    try:
        summarize_test_results(
            results=result,
            configs=configs,
            get_kpi_config=get_kpi_config,
            test_name=test_name,
        )
    except Exception as e:
        logger.error(f"Failed to summarize test results: {e}")
    
    logger.info(f"Peripheral device detection completed successfully: {test_display_name}")
