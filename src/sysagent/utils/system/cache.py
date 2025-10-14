# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System information cache module.

Provides caching functionality for system information to avoid repeated
expensive system calls during test execution.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from sysagent.utils.config import setup_data_dir

logger = logging.getLogger(__name__)


class SystemInfoCache:
    """
    Singleton class for collecting and caching system information.

    The cache is stored in separate JSON files for hardware and software info,
    using the naming convention that matches the suite structure:
    - system_info_hardware.json: Contains hardware information
    - system_info_software.json: Contains software information

    Each cache file has a configurable TTL.
    """

    _instance = None
    _info: Dict[str, Any] = {}
    _cache_dir = None
    _hardware_cache_file = None
    _software_cache_file = None
    _cache_ttl = 3600  # Cache valid for 1 hour by default

    def __new__(cls, cache_dir: Optional[str] = None, ttl: int = 3600):
        """
        Create a new SystemInfoCache instance or return the existing one.

        Args:
            cache_dir: Directory to store the cache file
            ttl: Cache time-to-live in seconds
        """
        if cls._instance is None:
            cls._instance = super(SystemInfoCache, cls).__new__(cls)
            cls._instance._setup(cache_dir, ttl)
        return cls._instance

    def _setup(self, cache_dir: Optional[str], ttl: int) -> None:
        """
        Setup cache directory and TTL.

        Args:
            cache_dir: Directory to store the cache file
            ttl: Cache time-to-live in seconds
        """
        # Set cache TTL
        self._cache_ttl = ttl

        # Setup directories
        data_dir = setup_data_dir()
        cache_dir = os.path.join(os.getcwd(), data_dir, "cache")
        self._cache_dir = cache_dir

        # Set cache file paths
        self._hardware_cache_file = Path(cache_dir) / "system_info_hardware.json"
        self._software_cache_file = Path(cache_dir) / "system_info_software.json"

        # Load or collect info
        self._load_or_collect_info()

    def _load_or_collect_info(self) -> None:
        """Load from cache if valid, otherwise collect system info."""
        from .info import collect_hardware_info, collect_software_info

        hardware_loaded = False
        software_loaded = False

        # Try to load hardware info
        if self._hardware_cache_file and self._hardware_cache_file.exists():
            try:
                hardware_data = json.loads(self._hardware_cache_file.read_text())
                if time.time() - hardware_data.get("timestamp", 0) < self._cache_ttl:
                    logger.debug(
                        f"Using cached hardware info from {self._hardware_cache_file}"
                    )
                    self._info["hardware"] = hardware_data.get("hardware", {})
                    self._info["timestamp"] = hardware_data.get("timestamp")
                    hardware_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load hardware info cache: {e}")

        # Try to load software info
        if self._software_cache_file and self._software_cache_file.exists():
            try:
                software_data = json.loads(self._software_cache_file.read_text())
                if time.time() - software_data.get("timestamp", 0) < self._cache_ttl:
                    logger.debug(
                        f"Using cached software info from {self._software_cache_file}"
                    )
                    self._info["software"] = software_data.get("software", {})
                    if not hardware_loaded:
                        self._info["timestamp"] = software_data.get("timestamp")
                    software_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load software info cache: {e}")

        # Collect missing info
        if not hardware_loaded:
            logger.debug("Collecting hardware system information")
            self._info["hardware"] = collect_hardware_info()

        if not software_loaded:
            logger.debug("Collecting software system information")
            self._info["software"] = collect_software_info()

        # Set timestamp if not already set
        if "timestamp" not in self._info:
            self._info["timestamp"] = time.time()

        # Save to cache
        self._save_to_cache()

    def _save_to_cache(self) -> None:
        """Save system info to cache files."""
        if not self._cache_dir:
            return

        # Ensure cache directory exists
        os.makedirs(self._cache_dir, exist_ok=True)

        # Save hardware info
        try:
            hardware_data = {
                "hardware": self._info.get("hardware", {}),
                "timestamp": self._info.get("timestamp", time.time()),
            }
            self._hardware_cache_file.write_text(json.dumps(hardware_data, indent=2))
            logger.debug(f"Saved hardware info to cache: {self._hardware_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save hardware info cache: {e}")

        # Save software info
        try:
            software_data = {
                "software": self._info.get("software", {}),
                "timestamp": self._info.get("timestamp", time.time()),
            }
            self._software_cache_file.write_text(json.dumps(software_data, indent=2))
            logger.debug(f"Saved software info to cache: {self._software_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save software info cache: {e}")

    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        return self._info.get("hardware", {})

    def get_software_info(self) -> Dict[str, Any]:
        """Get software information."""
        return self._info.get("software", {})

    def get_info(self) -> Dict[str, Any]:
        """Get all system information."""
        return self._info.copy()

    def refresh(self) -> None:
        """Force refresh of system information."""
        from .info import collect_hardware_info, collect_software_info

        logger.debug("Refreshing system information cache")
        self._info["hardware"] = collect_hardware_info()
        self._info["software"] = collect_software_info()
        self._info["timestamp"] = time.time()
        self._save_to_cache()

    def generate_simple_report(self) -> str:
        """Generate a simple text report of system information."""
        from .formatter import generate_simple_report

        return generate_simple_report(self._info)
