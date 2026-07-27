# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Validation reporting utilities.

This module provides shared functions for reporting failed system requirements
across different validation contexts, showing required vs actual values.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def log_failed_requirements(
    failed_checks: List[Dict[str, Any]], context: str = "validation", deduplicate_by_category: bool = False
) -> None:
    """
    Log failed system requirements showing required vs actual values.

    Args:
        failed_checks: List of failed check dictionaries containing name, category, etc.
        context: Context string to include in the header message (e.g., "profile: my-profile")
        deduplicate_by_category: Whether to deduplicate checks by category to avoid redundant messages
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

        _log_requirement_failure(category, name, check, is_in_profile_context=context.startswith("profile:"))

    # Add bottom border for profile context
    if context.startswith("profile:"):
        logger.info("╰─")


def _deduplicate_checks_by_category(failed_checks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate checks based on category to avoid redundant messages.

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


def _log_requirement_failure(
    category: str, name: str, check: Dict[str, Any], is_in_profile_context: bool = False
) -> None:
    """
    Log a single failed requirement showing required vs actual values.

    Args:
        category: Category string (e.g., "software.docker.required")
        name: Human-readable name of the requirement
        check: Full check dictionary for additional context
        is_in_profile_context: Whether this is being logged within a profile validation context
    """
    prefix = "│  " if is_in_profile_context else ""

    # For binary presence checks the 'actual' value adds no information
    # (e.g. "Not set", "Not available", "Not installed") — show the name only.
    _name_only_categories = {
        "software.env.required",
        "software.docker.required",
        "software.system_packages.required",
        "software.python_packages.required",
    }
    if category in _name_only_categories:
        logger.info(f"{prefix}• {name}")
    else:
        required = check.get("required", "")
        actual = check.get("actual", "")
        logger.info(f"{prefix}• Required: {required} | Actual: {actual}")


def format_requirement_failure(category: str, name: str, check: Dict[str, Any]) -> List[str]:
    """
    Format a failed requirement as a list of strings (for testing or other uses).

    Args:
        category: Category string (e.g., "software.docker.required")
        name: Human-readable name of the requirement
        check: Full check dictionary for additional context

    Returns:
        List of strings describing the failed requirement
    """
    _name_only_categories = {
        "software.env.required",
        "software.docker.required",
        "software.system_packages.required",
        "software.python_packages.required",
    }
    if category in _name_only_categories:
        return [name]

    required = check.get("required", "")
    actual = check.get("actual", "")
    return [f"Required: {required} | Actual: {actual}"]
