# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
PCI device information helper utilities.

This module provides utilities for working with PCI devices, including:
- Loading and parsing PCI ID database
- Mapping PCI IDs to human-readable device names
- Extracting PCI device information from sysfs

The PCI database files are stored in data/thirdparty/pci/ directory:
- pci.ids: Original PCI IDs database from pci-ids.ucw.cz
- pci_ids.json: Parsed database in JSON format (subsystems are excluded to reduce size)
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from sysagent.utils.config import setup_data_dir

# Setup logger
logger = logging.getLogger(__name__)

# PCI IDs database URL
PCI_IDS_URL = "https://pci-ids.ucw.cz/v2.2/pci.ids"
PCI_IDS_CACHE_TTL = 30 * 24 * 60 * 60  # 30 days in seconds


class PCIDeviceInfo:
    """
    Helper class for PCI device information management.

    This class handles:
    - Loading and parsing the PCI IDs database
    - Mapping PCI vendor/device IDs to human-readable names
    - Extracting PCI device information from sysfs
    """

    def __init__(
        self, thirdparty_dir: Optional[str] = None, cache_dir: Optional[str] = None
    ):
        """
        Initialize the PCI device info helper.

        Args:
            thirdparty_dir: Directory to store the PCI IDs database.
            cache_dir: Directory to store cached data.
        """
        # setup directory
        data_dir = setup_data_dir()
        thirdparty_dir = os.path.join(os.getcwd(), data_dir, "thirdparty")
        cache_dir = os.path.join(os.getcwd(), data_dir, "cache")

        self.cache_dir = cache_dir

        # Create thirdparty/pci subdirectory for PCI-related files
        pci_dir = os.path.join(thirdparty_dir, "pci")
        os.makedirs(pci_dir, exist_ok=True)
        self.thirdparty_dir = thirdparty_dir
        self.pci_dir = pci_dir

        # Set PCI IDs database path
        self.pci_ids_path = os.path.join(pci_dir, "pci.ids")
        self.pci_ids_json_path = os.path.join(cache_dir, "pci_ids.json")

        # Initialize database cache
        self._pci_db = None

    def _download_pci_ids(self) -> bool:
        """
        Download the PCI IDs database from the official repository.

        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            logger.info(f"Downloading PCI IDs database from {PCI_IDS_URL}")
            response = requests.get(PCI_IDS_URL, timeout=10)

            if response.status_code == 200:
                with open(self.pci_ids_path, "wb") as f:
                    f.write(response.content)
                logger.debug(
                    f"Successfully downloaded PCI IDs database to {self.pci_ids_path}"
                )
                return True
            else:
                logger.error(
                    f"Failed to download PCI IDs database: HTTP {response.status_code}"
                )
                return False

        except Exception as e:
            logger.error(f"Error downloading PCI IDs database: {e}")
            return False

    def _parse_pci_ids(self) -> Dict[str, Any]:
        """
        Parse the PCI IDs database file into a structured dictionary.

        Returns:
            Dict containing parsed PCI vendor and device information
        """
        if not os.path.exists(self.pci_ids_path):
            logger.error(f"PCI IDs database file not found at {self.pci_ids_path}")
            return {}

        logger.debug(f"Parsing PCI IDs database from {self.pci_ids_path}")

        pci_db = {"vendors": {}, "classes": {}, "subclasses": {}}

        current_vendor = None
        current_class = None
        current_subclass = None

        with open(self.pci_ids_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip()

                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Vendor
                if not line.startswith("\t"):
                    match = re.match(r"^([0-9a-f]{4})\s+(.+)$", line)
                    if match:
                        vendor_id, vendor_name = match.groups()
                        vendor_id = vendor_id.lower()
                        pci_db["vendors"][vendor_id] = {
                            "name": vendor_name,
                            "devices": {},
                        }
                        current_vendor = vendor_id
                    elif line.startswith("C "):
                        # Class definition
                        match = re.match(r"^C ([0-9a-f]{2})\s+(.+)$", line)
                        if match:
                            class_id, class_name = match.groups()
                            class_id = class_id.lower()
                            pci_db["classes"][class_id] = {
                                "name": class_name,
                                "subclasses": {},
                            }
                            current_class = class_id
                            current_subclass = None

                # Device or subclass
                elif line.startswith("\t") and not line.startswith("\t\t"):
                    if current_vendor is not None:
                        # Device under a vendor
                        match = re.match(r"^\t([0-9a-f]{4})\s+(.+)$", line)
                        if match:
                            device_id, device_name = match.groups()
                            device_id = device_id.lower()
                            pci_db["vendors"][current_vendor]["devices"][device_id] = {
                                "name": device_name
                            }
                    elif current_class is not None:
                        # Subclass under a class
                        match = re.match(r"^\t([0-9a-f]{2})\s+(.+)$", line)
                        if match:
                            subclass_id, subclass_name = match.groups()
                            subclass_id = subclass_id.lower()
                            pci_db["classes"][current_class]["subclasses"][
                                subclass_id
                            ] = {"name": subclass_name, "prog_ifs": {}}
                            current_subclass = subclass_id

                # Subsystem or programming interface
                elif line.startswith("\t\t"):
                    if current_vendor is not None:
                        # Skip subsystem information to keep JSON size manageable
                        pass
                    elif current_class is not None and current_subclass is not None:
                        # Programming interface under a subclass
                        match = re.match(r"^\t\t([0-9a-f]{2})\s+(.+)$", line)
                        if match:
                            prog_if_id, prog_if_name = match.groups()
                            prog_if_id = prog_if_id.lower()
                            pci_db["classes"][current_class]["subclasses"][
                                current_subclass
                            ]["prog_ifs"][prog_if_id] = {"name": prog_if_name}

        # Cache the parsed database to JSON for faster loading next time
        try:
            # Add metadata about the database
            timestamp = time.time()
            formatted_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(timestamp)
            )

            pci_db["metadata"] = {
                "timestamp": timestamp,
                "formatted_time": formatted_time,
                "version": "optimized",
                "source": PCI_IDS_URL,
                "note": "Subsystem information is excluded to reduce JSON size",
            }

            with open(self.pci_ids_json_path, "w") as f:
                json.dump(pci_db, f, indent=2)
            logger.debug(
                f"Cached parsed PCI IDs database to {self.pci_ids_json_path} "
                f"with timestamp {formatted_time}"
            )
        except Exception as e:
            logger.warning(f"Failed to cache parsed PCI IDs database: {e}")

        logger.debug(
            f"Loaded PCI IDs database with {len(pci_db['vendors'])} vendors and "
            f"{sum(len(v['devices']) for v in pci_db['vendors'].values())} devices"
        )

        return pci_db

    def _load_or_download_pci_db(self) -> Dict[str, Any]:
        """
        Load the PCI IDs database, downloading and parsing if necessary.

        Returns:
            Dict containing PCI database information
        """
        need_download = True

        # Check if we have a cached JSON version first (faster loading)
        if os.path.exists(self.pci_ids_json_path):
            try:
                # Check if JSON cache is recent enough
                with open(self.pci_ids_json_path, "r") as f:
                    cached_db = json.load(f)

                if "metadata" in cached_db and "timestamp" in cached_db["metadata"]:
                    cache_time = cached_db["metadata"]["timestamp"]
                    cache_age = time.time() - cache_time
                    formatted_age = f"{cache_age / 86400:.1f} days"

                    if cache_age < PCI_IDS_CACHE_TTL:
                        self._pci_db = cached_db
                        formatted_time = cached_db["metadata"].get(
                            "formatted_time", "unknown time"
                        )
                        vendor_count = len(cached_db["vendors"])
                        device_count = sum(
                            len(v["devices"]) for v in cached_db["vendors"].values()
                        )
                        logger.debug(
                            f"Loaded PCI IDs from cache: {self.pci_ids_json_path} "
                            f"(age: {formatted_age}, from: {formatted_time})"
                        )
                        logger.debug(
                            f"PCI database contains {vendor_count} vendors and "
                            f"{device_count} devices"
                        )
                        return self._pci_db
                    else:
                        logger.debug(
                            f"PCI IDs cache is {formatted_age} old, exceeding TTL of "
                            f"{PCI_IDS_CACHE_TTL / 86400:.1f} days"
                        )
                else:
                    logger.warning(
                        "PCI IDs cache lacks proper metadata, will regenerate"
                    )
            except Exception as e:
                logger.warning(
                    f"Error loading cached PCI JSON: {e}, will download fresh copy"
                )

        # Check if we need to download the database
        if os.path.exists(self.pci_ids_path):
            # Check if the file is recent enough
            mtime = os.path.getmtime(self.pci_ids_path)
            file_age = time.time() - mtime
            formatted_age = f"{file_age / 86400:.1f} days"

            if file_age < PCI_IDS_CACHE_TTL:
                need_download = False
                logger.debug(
                    f"Using existing PCI IDs database file: {self.pci_ids_path} "
                    f"(age: {formatted_age})"
                )
            else:
                logger.debug(
                    f"PCI IDs database is {formatted_age} old, downloading new version"
                )

        if need_download:
            success = self._download_pci_ids()
            if not success and not os.path.exists(self.pci_ids_path):
                logger.error(
                    "Failed to download PCI IDs database and no local copy exists"
                )
                return {}

        # Parse the database
        self._pci_db = self._parse_pci_ids()
        return self._pci_db

    def get_vendor_name(self, vendor_id: str) -> str:
        """
        Get the human-readable name for a PCI vendor ID.

        Args:
            vendor_id: PCI vendor ID (e.g., "8086" for Intel)

        Returns:
            str: Vendor name or "Unknown vendor" if not found
        """
        if self._pci_db is None:
            self._load_or_download_pci_db()

        vendor_id = vendor_id.lower()
        return (
            self._pci_db.get("vendors", {})
            .get(vendor_id, {})
            .get("name", "Unknown vendor")
        )

    def get_device_name(self, vendor_id: str, device_id: str) -> str:
        """
        Get the human-readable name for a PCI device.

        Args:
            vendor_id: PCI vendor ID
            device_id: PCI device ID

        Returns:
            str: Device name or "Unknown device" if not found
        """
        if self._pci_db is None:
            self._load_or_download_pci_db()

        vendor_id = vendor_id.lower()
        device_id = device_id.lower()

        vendor = self._pci_db.get("vendors", {}).get(vendor_id, {})
        return (
            vendor.get("devices", {}).get(device_id, {}).get("name", "Unknown device")
        )

    def get_class_info(
        self, class_id: str, subclass_id: str = None, prog_if: str = None
    ) -> Tuple[str, str, str]:
        """
        Get human-readable names for PCI class, subclass, and programming interface.

        Args:
            class_id: PCI class ID
            subclass_id: PCI subclass ID
            prog_if: PCI programming interface ID

        Returns:
            Tuple containing (class_name, subclass_name, prog_if_name)
        """
        if self._pci_db is None:
            self._load_or_download_pci_db()

        class_id = class_id.lower() if class_id else None
        subclass_id = subclass_id.lower() if subclass_id else None
        prog_if = prog_if.lower() if prog_if else None

        class_name = "Unknown class"
        subclass_name = "Unknown subclass"
        prog_if_name = "Unknown programming interface"

        if class_id and class_id in self._pci_db.get("classes", {}):
            class_info = self._pci_db["classes"][class_id]
            class_name = class_info.get("name", class_name)

            if subclass_id and subclass_id in class_info.get("subclasses", {}):
                subclass_info = class_info["subclasses"][subclass_id]
                subclass_name = subclass_info.get("name", subclass_name)

                if prog_if and prog_if in subclass_info.get("prog_ifs", {}):
                    prog_if_name = subclass_info["prog_ifs"][prog_if].get(
                        "name", prog_if_name
                    )

        return (class_name, subclass_name, prog_if_name)

    def extract_pci_devices(self) -> List[Dict[str, Any]]:
        """
        Extract all PCI devices from the system using sysfs.

        Returns:
            List of dictionaries containing PCI device information
        """
        devices = []
        sysfs_pci_path = "/sys/bus/pci/devices"

        if not os.path.exists(sysfs_pci_path):
            logger.warning(f"PCI sysfs path not found: {sysfs_pci_path}")
            return devices

        # Load PCI database if not already loaded
        if self._pci_db is None:
            self._load_or_download_pci_db()

        for device_dir in os.listdir(sysfs_pci_path):
            device_path = os.path.join(sysfs_pci_path, device_dir)

            try:
                # Extract basic device information
                device_info = {
                    "pci_slot": device_dir,  # PCI bus address (e.g., "0000:00:00.0")
                    "path": device_path,
                }

                # Vendor ID
                vendor_path = os.path.join(device_path, "vendor")
                if os.path.exists(vendor_path):
                    with open(vendor_path, "r") as f:
                        vendor_id = f.read().strip()[2:]  # Remove "0x" prefix
                        device_info["vendor_id"] = vendor_id
                        device_info["vendor_name"] = self.get_vendor_name(vendor_id)

                # Device ID
                device_id_path = os.path.join(device_path, "device")
                if os.path.exists(device_id_path):
                    with open(device_id_path, "r") as f:
                        device_id = f.read().strip()[2:]  # Remove "0x" prefix
                        device_info["device_id"] = device_id
                        if "vendor_id" in device_info:
                            device_info["device_name"] = self.get_device_name(
                                device_info["vendor_id"], device_id
                            )

                # Revision
                revision_path = os.path.join(device_path, "revision")
                if os.path.exists(revision_path):
                    with open(revision_path, "r") as f:
                        device_info["revision"] = f.read().strip()[
                            2:
                        ]  # Remove "0x" prefix

                # Class
                class_path = os.path.join(device_path, "class")
                if os.path.exists(class_path):
                    with open(class_path, "r") as f:
                        class_str = f.read().strip()[2:]  # Remove "0x" prefix
                        if len(class_str) >= 6:
                            # Class format: CCSSPP CC=class, SS=subclass, PP=prog-if
                            class_id = class_str[0:2]
                            subclass_id = class_str[2:4]
                            prog_if = class_str[4:6]

                            device_info["class_id"] = class_id
                            device_info["subclass_id"] = subclass_id
                            device_info["prog_if"] = prog_if

                            class_name, subclass_name, prog_if_name = (
                                self.get_class_info(class_id, subclass_id, prog_if)
                            )

                            device_info["class_name"] = class_name
                            device_info["subclass_name"] = subclass_name
                            device_info["prog_if_name"] = prog_if_name

                # Subsystem vendor and device
                subsys_vendor_path = os.path.join(device_path, "subsystem_vendor")
                subsys_device_path = os.path.join(device_path, "subsystem_device")

                if os.path.exists(subsys_vendor_path) and os.path.exists(
                    subsys_device_path
                ):
                    with open(subsys_vendor_path, "r") as f:
                        device_info["subsys_vendor_id"] = f.read().strip()[2:]

                    with open(subsys_device_path, "r") as f:
                        device_info["subsys_device_id"] = f.read().strip()[2:]

                # Driver in use
                driver_path = os.path.join(device_path, "driver")
                if os.path.exists(driver_path) and os.path.islink(driver_path):
                    driver_name = os.path.basename(os.readlink(driver_path))
                    device_info["driver"] = driver_name

                # Power state
                power_state_path = os.path.join(device_path, "power_state")
                if os.path.exists(power_state_path):
                    with open(power_state_path, "r") as f:
                        device_info["power_state"] = f.read().strip()

                # Device status (enabled/disabled)
                enable_path = os.path.join(device_path, "enable")
                if os.path.exists(enable_path):
                    with open(enable_path, "r") as f:
                        device_info["enabled"] = f.read().strip() == "1"

                # NUMA node
                numa_node_path = os.path.join(device_path, "numa_node")
                if os.path.exists(numa_node_path):
                    with open(numa_node_path, "r") as f:
                        numa_node = f.read().strip()
                        if numa_node != "-1":  # -1 means no NUMA node association
                            device_info["numa_node"] = int(numa_node)

                # Get device canonical name
                if "device_name" in device_info and "vendor_name" in device_info:
                    device_info["canonical_name"] = (
                        f"{device_info['vendor_name']} {device_info['device_name']}"
                    )

                # Add to devices list
                devices.append(device_info)

            except Exception as e:
                logger.warning(
                    f"Error extracting PCI device info for {device_dir}: {e}"
                )

        return devices


def get_pci_devices(
    thirdparty_dir: Optional[str] = None, cache_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Helper function to get all PCI devices in the system.

    Args:
        thirdparty_dir: Optional directory for PCI database storage
        cache_dir: Optional directory for cached data

    Returns:
        List of dictionaries containing PCI device information
    """
    pci_helper = PCIDeviceInfo(thirdparty_dir, cache_dir)
    return pci_helper.extract_pci_devices()
