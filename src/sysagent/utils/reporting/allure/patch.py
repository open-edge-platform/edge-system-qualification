# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Allure patch handling utilities.

This module contains functions for applying patches to allure report configurations
using the patch command utility.
"""

import logging
import os

# Import secure process execution utilities
from sysagent.utils.core.process import (
    check_command_available,
    run_command,
)

logger = logging.getLogger(__name__)


def _is_already_applied(*texts: str) -> bool:
    """Return True when patch output indicates the patch was previously applied."""
    text = "\n".join(t or "" for t in texts).lower()
    return any(
        phrase in text
        for phrase in [
            "reversed (or previously applied)",
            "already applied",
            "previously applied",
            "patch detected!  skipping patch",
            "skipping patch",
            "already exists",
        ]
    )


def _apply_patch_with_patch_command(patch_path: str, cwd: str) -> bool:
    """
    Apply a patch file using the patch command utility.

    Args:
        patch_path: Path to the patch file
        cwd: Working directory for patch commands

    Returns:
        bool: True if patch was applied successfully, False if already applied

    Raises:
        Exception: If patch application fails
    """
    patch_name = os.path.basename(patch_path)

    # First, run a dry-run for observability. Do not fail immediately on dry-run
    # mismatch because `patch` often reports details on stdout and --forward apply
    # can still provide the definitive result.
    try:
        result = run_command(
            ["patch", "--dry-run", "--strip=1", "--input=" + patch_path],
            cwd=cwd,
            check=False,
        )

        if result.success:
            logger.debug(f"Patch dry-run passed: {patch_name}")
        else:
            if _is_already_applied(result.stdout, result.stderr):
                logger.debug(f"Patch already applied: {patch_name}")
                return False
            logger.debug(
                "Patch dry-run reported mismatches for %s; proceeding to --forward apply. "
                "stdout: %s stderr: %s",
                patch_name,
                (result.stdout or "<empty>").strip(),
                (result.stderr or "<empty>").strip(),
            )
    except Exception as e:
        raise Exception(f"Could not validate patch {patch_name}: {e}") from e

    # Apply the patch using patch command
    try:
        result = run_command(
            [
                "patch",
                "--strip=1",
                "--force",
                "--backup",
                "--forward",
                "--input=" + patch_path,
            ],
            cwd=cwd,
            check=False,
        )

        if result.success:
            logger.debug(f"Patch applied: {patch_name}")
            return True
        else:
            if _is_already_applied(result.stdout, result.stderr):
                logger.debug(f"Patch already applied: {patch_name}")
                return False
            raise Exception(
                f"Patch apply failed for {patch_name}. "
                f"stdout: {result.stdout or '<empty>'}; stderr: {result.stderr or '<empty>'}"
            )

    except Exception as e:
        raise Exception(f"Patch command raised exception for {patch_name}: {e}") from e


def apply_patch(patch_path: str, cwd: str) -> bool:
    """
    Apply a patch file using the patch command utility.

    Args:
        patch_path: Path to the patch file
        cwd: Working directory for patch commands

    Returns:
        bool: True if patch was applied successfully, False if already applied

    Raises:
        Exception: If patch application fails or patch command is not available
    """
    patch_name = os.path.basename(patch_path)

    # Check if patch command utility is available
    try:
        check_command_available("patch")
        logger.debug(f"Using patch utility for: {patch_name}")
    except (Exception, FileNotFoundError):
        raise Exception(
            f"patch command utility not found. Cannot apply {patch_name}. "
            "Please install the 'patch' system utility. "
            "Refer to your system package manager or project documentation."
        )

    # Use patch command utility
    return _apply_patch_with_patch_command(patch_path, cwd)
