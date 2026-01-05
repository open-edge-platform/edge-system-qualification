# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Shared validation suggestion utilities.

This module provides shared functions for generating user-friendly fix suggestions
for failed system requirements across different validation contexts.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def log_validation_fix_suggestions(
    failed_checks: List[Dict[str, Any]], context: str = "validation", deduplicate_by_category: bool = False
) -> None:
    """
    Log user-friendly suggestions for fixing failed system requirements.

    Args:
        failed_checks: List of failed check dictionaries containing name, category, etc.
        context: Context string to include in the header message (e.g., "profile: my-profile")
        deduplicate_by_category: Whether to deduplicate suggestions by category to avoid duplicates
    """
    if not failed_checks:
        return

    # Handle deduplication if requested
    checks_to_process = failed_checks
    if deduplicate_by_category:
        checks_to_process = _deduplicate_checks_by_category(failed_checks)

    # Format header based on context type
    if context.startswith("profile:"):
        profile_name = context.split(":", 1)[1].strip()
        logger.info("")
        logger.info(f"╭─ Validation Failed: {profile_name}")
        logger.info(f"│  Missing requirements ({len(checks_to_process)}):")
    else:
        logger.info("")
        logger.info(f"Missing requirements ({context}):")

    for check in checks_to_process:
        category = check.get("category", "")
        name = check["name"]

        _log_single_suggestion(category, name, check, is_in_profile_context=context.startswith("profile:"))

    # Add bottom border for profile context
    if context.startswith("profile:"):
        logger.info("╰─")


def _deduplicate_checks_by_category(failed_checks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate suggestions based on category to avoid redundant messages.

    Args:
        failed_checks: List of failed check dictionaries

    Returns:
        List of unique checks based on category
    """
    seen_categories = set()
    unique_checks = []

    for check in failed_checks:
        category = check.get("category", "")

        if category not in seen_categories:
            seen_categories.add(category)
            unique_checks.append(check)

    return unique_checks


def _log_single_suggestion(
    category: str, name: str, check: Dict[str, Any], is_in_profile_context: bool = False
) -> None:
    """
    Log a single fix suggestion based on the requirement category.

    Args:
        category: Category string (e.g., "software.docker.required")
        name: Human-readable name of the requirement
        check: Full check dictionary for additional context
        is_in_profile_context: Whether this is being logged within a profile validation context
    """
    # Use different prefix based on context
    prefix = "│  " if is_in_profile_context else ""
    if category == "software.docker.required":
        logger.info(f"{prefix}• {name}: Refer to Docker official installation guide")
        logger.info(f"{prefix}  Then add user to docker group: 'sudo usermod -aG docker $USER && newgrp docker'")
    elif category == "hardware.gpu.discrete":
        logger.info(f"{prefix}• {name}: This test requires a discrete GPU (dGPU). Check if:")
        logger.info(f"{prefix}  - A discrete GPU is properly installed and seated")
        logger.info(f"{prefix}  - GPU drivers are installed (Intel Arc, AMD, or NVIDIA)")
        logger.info(f"{prefix}  - GPU is detected by the system (check with 'lspci | grep -i vga')")
    elif category == "hardware.gpu.integrated":
        logger.info(f"{prefix}• {name}: This test requires an integrated GPU (iGPU). Check if:")
        logger.info(f"{prefix}  - Integrated graphics are enabled in BIOS/UEFI")
        logger.info(f"{prefix}  - Intel graphics drivers are installed")
        logger.info(f"{prefix}  - CPU supports integrated graphics")
    elif category == "hardware.cpu.cores":
        logger.info(f"{prefix}• {name}: Upgrade to a CPU with more cores")
    elif category == "hardware.memory.available":
        logger.info(f"{prefix}• {name}: Free up system memory by closing applications or add more RAM")
    elif category == "hardware.memory.total":
        logger.info(f"{prefix}• {name}: Add more system RAM to meet total memory requirements")
    elif category == "hardware.storage.free":
        logger.info(f"{prefix}• {name}: Free up disk space or add more storage")
    elif category == "hardware.storage.total":
        logger.info(f"{prefix}• {name}: Upgrade to larger storage capacity")
    elif category == "software.os.type":
        logger.info(f"{prefix}• {name}: This test requires a different operating system")
    elif category == "software.python.version":
        logger.info(f"{prefix}• {name}: Upgrade Python version")
    elif "system_packages" in category:
        logger.info(f"{prefix}• {name}: Ensure dependency is installed")
    elif "python_packages" in category:
        logger.info(f"{prefix}• {name}: Ensure dependency is installed")
    else:
        logger.info(f"{prefix}• {name}: Check system documentation for {check['required']}")


def get_fix_suggestion_for_category(category: str, name: str, check: Dict[str, Any]) -> List[str]:
    """
    Get fix suggestions for a specific category as a list of strings (for testing or other uses).

    Args:
        category: Category string (e.g., "software.docker.required")
        name: Human-readable name of the requirement
        check: Full check dictionary for additional context

    Returns:
        List of suggestion strings
    """
    suggestions = []

    if category == "software.docker.required":
        suggestions.append(
            f"{name}: Install Docker with 'sudo apt install docker.io' or 'curl -fsSL https://get.docker.com | sh'"
        )
        suggestions.append("Then add user to docker group: 'sudo usermod -aG docker $USER && newgrp docker'")
    elif category == "hardware.gpu.discrete":
        suggestions.append(f"{name}: This test requires a discrete GPU (dGPU). Check if:")
        suggestions.append("- A discrete GPU is properly installed and seated")
        suggestions.append("- GPU drivers are installed (Intel Arc, AMD, or NVIDIA)")
        suggestions.append("- GPU is detected by the system (check with 'lspci | grep -i vga')")
    elif category == "hardware.gpu.integrated":
        suggestions.append(f"{name}: This test requires an integrated GPU (iGPU). Check if:")
        suggestions.append("- Integrated graphics are enabled in BIOS/UEFI")
        suggestions.append("- Intel graphics drivers are installed")
        suggestions.append("- CPU supports integrated graphics")
    elif category == "hardware.cpu.cores":
        suggestions.append(f"{name}: Upgrade to a CPU with more cores, or use a different test profile")
    elif category == "hardware.memory.available":
        suggestions.append(f"{name}: Free up system memory by closing applications or add more RAM")
    elif category == "hardware.memory.total":
        suggestions.append(f"{name}: Add more system RAM to meet total memory requirements")
    elif category == "hardware.storage.free":
        suggestions.append(f"{name}: Free up disk space or add more storage")
    elif category == "hardware.storage.total":
        suggestions.append(f"{name}: Upgrade to larger storage capacity")
    elif category == "software.os.type":
        suggestions.append(f"{name}: This test requires a different operating system")
    elif category == "software.python.version":
        suggestions.append(f"{name}: Upgrade Python with 'sudo apt install python3' or use pyenv/conda")
    elif "system_packages" in category:
        package = name.split("'")[1] if "'" in name else "unknown"
        suggestions.append(f"{name}: Install with 'sudo apt install {package}' (Ubuntu/Debian)")
    elif "python_packages" in category:
        package = name.split("'")[1] if "'" in name else "unknown"
        suggestions.append(f"{name}: Install with 'pip install {package}' or 'uv pip install {package}'")
    else:
        suggestions.append(f"{name}: Check system documentation for {check['required']}")

    return suggestions
