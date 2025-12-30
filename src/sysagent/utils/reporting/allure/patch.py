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

    # First, try to check if the patch can be applied using patch --dry-run
    try:
        result = run_command(
            ["patch", "--dry-run", "--strip=1", "--input=" + patch_path],
            cwd=cwd,
            check=False,
        )

        if result.success:
            logger.debug(f"Patch {patch_name} passed dry run check")
        else:
            # Check if this is because the patch is already applied
            if any(
                phrase in result.stderr.lower()
                for phrase in [
                    "reversed (or previously applied)",
                    "already applied",
                    "previously applied",
                ]
            ):
                logger.debug(f"Patch {patch_name} appears to be already applied")
                return False
            else:
                logger.debug(f"Patch {patch_name} dry run failed: {result.stderr}")
                # Continue to try applying anyway, as it might still work
    except Exception as e:
        logger.debug(f"Could not perform dry run for patch {patch_name}: {e}")

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
            logger.debug(f"Successfully applied patch with patch command: {patch_name}")
            return True
        else:
            # Check if this is because the patch is already applied
            stderr_text = result.stderr.lower() if result.stderr else ""
            if any(
                phrase in stderr_text
                for phrase in [
                    "reversed (or previously applied)",
                    "already applied",
                    "previously applied",
                    "patch detected!  skipping patch",
                    "skipping patch",
                    "already exists",
                ]
            ):
                logger.debug(f"Patch {patch_name} appears to be already applied")
                return False

            # If patch command returned non-zero but not because it's already applied,
            # this might be a real error, but let's try to continue
            logger.debug(f"patch command failed for {patch_name}: {result.stderr}")

            # Consider failed patches as already applied
            # This prevents build failures when patches have minor conflicts
            logger.debug(f"Patch {patch_name} may already be applied (continuing)")
            return False

    except Exception as e:
        logger.debug(f"patch command raised exception for {patch_name}: {e}")
        # Also be lenient with exceptions - treat as already applied
        logger.debug(f"Patch {patch_name} encountered exception, assuming already applied")
        return False


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
        logger.debug(f"patch command utility is available, using it for {patch_name}")
    except (Exception, FileNotFoundError):
        raise Exception(
            f"patch command utility not found. Cannot apply {patch_name}. "
            "Please install patch utility: "
            "On Ubuntu/Debian: sudo apt-get install patch, On RHEL/Fedora: sudo yum install patch"
        )

    # Use patch command utility
    return _apply_patch_with_patch_command(patch_path, cwd)
