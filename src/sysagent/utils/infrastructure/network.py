# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Network utilities for connectivity checks and restriction detection.

This module provides utilities to:
- Check internet connectivity
- Detect network restrictions (e.g., blocked services like HuggingFace)
- Automatically fallback to alternative services like ModelScope
"""

import logging
import os
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)


def check_internet_connectivity() -> bool:
    """
    Check if the system has internet connectivity using HTTPS requests.

    Uses multiple endpoints to handle different network environments,
    including regions where certain services may be blocked.

    Returns:
        True if internet is accessible, False otherwise
    """
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


def check_network_restrictions(timeout: float = 3.0) -> Tuple[bool, dict]:
    """
    Check network accessibility to common services.

    Tests connectivity to multiple services to determine if any are blocked or
    inaccessible from the current network. If any of these services cannot be
    reached, the network is considered to have restrictions.

    Tested services:
    - HuggingFace: AI model hosting
    - GitHub: Code hosting
    - Kaggle: Data science platform
    - Google: Search and services

    Args:
        timeout: Timeout in seconds for connectivity check

    Returns:
        Tuple of (is_restricted, details_dict)
        - is_restricted: True if network restrictions detected
        - details_dict: Dictionary with accessibility status for each service
                       and list of blocked services
    """
    # Services to test
    test_services = {
        "huggingface": "https://huggingface.co",
        "github": "https://github.com",
        "kaggle": "https://www.kaggle.com",
        "google": "https://www.google.com",
    }

    details = {
        "huggingface_accessible": False,
        "github_accessible": False,
        "kaggle_accessible": False,
        "google_accessible": False,
    }

    for service_name, url in test_services.items():
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)

            if response.status_code < 500:  # Any non-server-error is considered accessible
                details[f"{service_name}_accessible"] = True
                logger.debug(f"{service_name.capitalize()} is accessible")
            else:
                logger.debug(f"{service_name.capitalize()} returned server error: {response.status_code}")

        except Exception as e:
            logger.debug(f"{service_name.capitalize()} is not accessible: {e}")

    # Derive blocked services from boolean flags
    blocked_services = [
        service_name.capitalize() for service_name in test_services.keys() if not details[f"{service_name}_accessible"]
    ]

    # Network is restricted if ANY of the services are not accessible
    is_restricted = len(blocked_services) > 0

    if is_restricted:
        blocked_list = ", ".join(blocked_services)
        logger.info(f"Network restrictions detected (blocked: {blocked_list}) - will use alternative download sources")
    else:
        logger.debug("No network restrictions detected - all services accessible")

    return is_restricted, details


def get_cached_network_restrictions() -> Optional[bool]:
    """
    Get cached network restriction status from system info cache.

    Returns:
        None if not cached, True if restricted, False if not restricted
    """
    try:
        from sysagent.utils.system import SystemInfoCache

        system_info = SystemInfoCache()
        hardware_info = system_info.get_hardware_info()

        if hardware_info and "network" in hardware_info:
            network_info = hardware_info["network"]
            if "restricted_access" in network_info:
                return network_info["restricted_access"]

    except Exception as e:
        logger.debug(f"Could not get cached network restriction status: {e}")

    return None


def get_preferred_download_source(
    source_type: Optional[str] = None,
    default: str = "huggingface",
    valid_sources: Optional[list] = None,
) -> str:
    """
    Determine which download source to use based on preference flags in environment variables.

    Uses flag-based environment variables for flexible, discoverable source configuration:
    - PREFER_<SOURCE>_<TYPE>=1  (e.g., PREFER_HUGGINGFACE_MODEL=1)
    - PREFER_<SOURCE>=1         (e.g., PREFER_MODELSCOPE=1)

    Priority order:
    1. Type-specific preference flags (PREFER_<SOURCE>_<TYPE>=1)
    2. Global preference flags (PREFER_<SOURCE>=1)
    3. Function parameter default (caller-specified, defaults to "huggingface")

    This approach allows:
    - Easy discovery of new sources without modifying shared code
    - Test suites can add new sources by setting PREFER_<NEWSOURCE>=1
    - No hardcoded source names in the function
    - Simple boolean flags instead of string values

    Examples:
        # Prefer ModelScope globally
        export PREFER_MODELSCOPE=1

        # Prefer HuggingFace for models, ModelScope for datasets
        export PREFER_HUGGINGFACE_MODEL=1
        export PREFER_MODELSCOPE_DATASET=1

        # Prefer Ultralytics for video downloads
        export PREFER_ULTRALYTICS_VIDEO=1

    Common source types:
    - "model": Model downloads
    - "dataset": Dataset downloads
    - "video": Video/media downloads

    Args:
        source_type: Optional resource type (e.g., "model", "dataset", "video")
                    Used to check type-specific flags (PREFER_<SOURCE>_<TYPE>)
        default: Default source if no preference flag is set (default: "huggingface")
        valid_sources: Optional list of valid sources for this context
                      If provided, validates the source and warns on invalid values

    Returns:
        str: Download source identifier (lowercase)
    """
    source = None
    source_origin = None

    # Priority 1: Check type-specific preference flags (PREFER_<SOURCE>_<TYPE>)
    if source_type:
        type_suffix = f"_{source_type.upper()}"
        for env_var, env_value in os.environ.items():
            if env_var.startswith("PREFER_") and env_var.endswith(type_suffix):
                # Extract source name: PREFER_HUGGINGFACE_MODEL -> huggingface
                source_name = env_var[len("PREFER_") : -len(type_suffix)].lower()
                # Check if flag is set (any truthy value: 1, true, yes, etc.)
                if env_value.lower() in ("1", "true", "yes", "on"):
                    source = source_name
                    source_origin = env_var
                    break

    # Priority 2: Check global preference flags (PREFER_<SOURCE>)
    if source is None:
        for env_var, env_value in os.environ.items():
            if env_var.startswith("PREFER_") and "_" not in env_var[len("PREFER_") :]:
                # Extract source name: PREFER_MODELSCOPE -> modelscope
                source_name = env_var[len("PREFER_") :].lower()
                # Check if flag is set
                if env_value.lower() in ("1", "true", "yes", "on"):
                    source = source_name
                    source_origin = env_var
                    break

    # Priority 3: Use default
    if source is None:
        source = default.lower()
        source_origin = "default parameter"

    # Validate against valid_sources if provided
    if valid_sources:
        valid_sources_lower = [s.lower() for s in valid_sources]
        if source not in valid_sources_lower:
            logger.warning(
                f"Invalid download source '{source}' from {source_origin}. "
                f"Valid sources: {', '.join(valid_sources)}. "
                f"Defaulting to '{default}'."
            )
            source = default.lower()
            source_origin = "default (after validation)"

    # Log the selection
    type_label = f" for {source_type}" if source_type else ""
    logger.debug(f"Using '{source}' as download source{type_label} (from {source_origin})")

    return source


def test_connectivity(url: str, timeout: float = 5.0) -> bool:
    """
    Test if a URL is accessible.

    Args:
        url: URL to test
        timeout: Timeout in seconds

    Returns:
        True if accessible, False otherwise
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code < 500
    except Exception:
        return False
