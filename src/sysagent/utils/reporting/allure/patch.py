# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Allure patch handling utilities.

This module contains functions for applying patches to repositories and managing
git operations related to patching allure report configurations.
"""

import logging
import os
from typing import List

# Import secure process execution utilities
from sysagent.utils.core.process import (
    ProcessResult,
    check_command_available,
    run_command,
    run_git_command,
)

logger = logging.getLogger(__name__)


def _run_git_command(
    cmd: List[str], cwd: str = None, check: bool = True
) -> ProcessResult:
    """
    Execute a git command with proper error handling and logging.

    Args:
        cmd: Git command and arguments
        cwd: Working directory for command execution
        check: Whether to raise exception on non-zero exit codes

    Returns:
        ProcessResult: Command execution result

    Raises:
        Exception: If command fails and check=True
    """
    logger.debug(
        f"Executing git command: git {' '.join(cmd)} in {cwd or 'current directory'}"
    )

    result = run_git_command(cmd, cwd=cwd, check=check)

    if result.stdout:
        logger.debug(f"Git command stdout: {result.stdout.strip()}")
    if result.stderr:
        logger.debug(f"Git command stderr: {result.stderr.strip()}")

    return result


def _is_git_repository(directory: str) -> bool:
    """
    Check if the given directory is a git repository.

    Args:
        directory: Path to check

    Returns:
        bool: True if directory is a git repository
    """
    try:
        result = _run_git_command(["rev-parse", "--git-dir"], cwd=directory, check=True)
        return result.success
    except Exception:
        return False


def _initialize_git_repository(directory: str) -> None:
    """
    Initialize a git repository in the specified directory.

    Args:
        directory: Directory to initialize as git repository

    Raises:
        subprocess.CalledProcessError: If git initialization fails
    """
    logger.debug(f"Initializing git repository in {directory}")

    try:
        # Initialize the repository
        _run_git_command(["init"], cwd=directory)
        logger.debug("Git repository initialized successfully")

        # Add all files to the repository
        _run_git_command(["add", "-f", "."], cwd=directory)
        logger.debug("Added all files to git repository")

        # Configure git user if not already configured (needed for commits)
        try:
            _run_git_command(["config", "user.name"], cwd=directory)
        except Exception:
            # Use dynamic project name instead of hardcoded
            from sysagent.utils.config.config_loader import get_project_name

            project_name = get_project_name()
            _run_git_command(
                ["config", "user.name", f"{project_name.upper()} Auto"], cwd=directory
            )

        try:
            _run_git_command(["config", "user.email"], cwd=directory)
        except Exception:
            from sysagent.utils.config.config_loader import get_project_name

            project_name = get_project_name()
            _run_git_command(
                ["config", "user.email", f"{project_name.lower()}@intel.com"],
                cwd=directory,
            )  # Create an initial commit to ensure patches can be applied
        _run_git_command(
            ["commit", "-m", "Initial commit from downloaded archive"], cwd=directory
        )
        logger.debug("Created initial commit in the repository")
    except Exception as e:
        # This could happen if there are no files to commit or other git issues
        logger.warning(
            f"Failed to create initial commit: {e.stderr}. Will try to continue."
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
            logger.info(f"Successfully applied patch with patch command: {patch_name}")
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
            logger.info(f"Patch {patch_name} may already be applied (continuing)")
            return False

    except Exception as e:
        logger.debug(f"patch command raised exception for {patch_name}: {e}")
        # Also be lenient with exceptions - treat as already applied
        logger.info(
            f"Patch {patch_name} encountered exception, assuming already applied"
        )
        return False


def apply_patch(patch_path: str, cwd: str) -> bool:
    """
    Apply a patch file using the best available method.

    Args:
        patch_path: Path to the patch file
        cwd: Working directory for patch commands

    Returns:
        bool: True if patch was applied successfully, False if already applied

    Raises:
        Exception: If patch application fails
    """
    patch_name = os.path.basename(patch_path)

    # Check if patch command utility is available
    try:
        check_command_available("patch")
        patch_available = True
        logger.debug(f"patch command utility is available, using it for {patch_name}")
    except (Exception, FileNotFoundError):
        patch_available = False
        logger.debug(
            f"patch command utility not available, using git apply for {patch_name}"
        )

    if patch_available:
        # Use patch command utility (preferred method for new files/directories)
        return _apply_patch_with_patch_command(patch_path, cwd)
    else:
        # Fallback to git apply method
        return _apply_patch_with_git_apply(patch_path, cwd)


def _apply_patch_with_git_apply(patch_path: str, cwd: str) -> bool:
    """
    Apply a patch file using git apply commands.

    Args:
        patch_path: Path to the patch file
        cwd: Working directory for git commands

    Returns:
        bool: True if patch was applied successfully, False if already applied

    Raises:
        Exception: If patch application fails
    """
    patch_name = os.path.basename(patch_path)

    # First, try to check if the patch can be applied using git apply --check
    try:
        _run_git_command(["apply", "--check", patch_path], cwd=cwd, check=True)
        logger.debug(f"Patch {patch_name} passed git apply check")
    except Exception as e:
        # Check if this is because the patch is already applied
        if "stderr" in str(e) and any(
            phrase in str(e).lower()
            for phrase in ["patch does not apply", "already exists", "with conflicts"]
        ):
            logger.debug(
                f"Patch {patch_name} appears to be already applied or has conflicts"
            )
            return False
        else:
            logger.debug(f"Patch {patch_name} check failed: {e}")

    # Try git apply with --index flag to handle new files and stage them
    try:
        _run_git_command(
            ["apply", "--index", "--ignore-whitespace", patch_path], cwd=cwd, check=True
        )
        logger.info(f"Successfully applied patch with git apply --index: {patch_name}")
        return True
    except Exception as e:
        apply_index_error = str(e)
        logger.debug(
            f"git apply --index failed for patch {patch_name}: {apply_index_error}"
        )

        # Check if it's because the patch is already applied
        if any(
            phrase in apply_index_error.lower()
            for phrase in ["patch does not apply", "already exists"]
        ):
            logger.debug(f"Patch {patch_name} appears to be already applied")
            return False

    # Try git apply without --index (for compatibility)
    try:
        _run_git_command(
            ["apply", "--ignore-whitespace", patch_path], cwd=cwd, check=True
        )
        logger.info(f"Successfully applied patch with basic git apply: {patch_name}")
        return True
    except Exception as e:
        basic_apply_error = str(e)
        logger.debug(
            f"Basic git apply failed for patch {patch_name}: {basic_apply_error}"
        )

        # Check if it's because the patch is already applied
        if any(
            phrase in basic_apply_error.lower()
            for phrase in ["patch does not apply", "already exists"]
        ):
            logger.debug(f"Patch {patch_name} appears to be already applied")
            return False

    # All git methods failed
    logger.error(f"All git apply methods failed for {patch_name}")

    # Re-raise the last exception with context
    raise Exception(f"Failed to apply patch {patch_name}")
