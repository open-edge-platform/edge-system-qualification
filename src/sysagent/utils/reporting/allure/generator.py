# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Allure report generation utilities.

This module contains functions for generating Allure reports,
including setup, installation, an    # Verify 'yarn allure' command works
    try:
        result = run_command([yarn_bin, "allure", "--version"],
            cwd=allure_repo_dir,
            check=True,
            env=env
        )
        logger.debug("Verified 'yarn allure' command works")
    except Exception as e:
        logger.error(f"Failed to verify 'yarn allure' command: {str(e)}")
        raisetion.
"""

import logging
import os
import shutil
from typing import Optional

# Import secure process execution utilities
from sysagent.utils.core.process import run_command

logger = logging.getLogger(__name__)

# Default directory for Allure
ALLURE_DIR_NAME = "allure3"


def _verify_allure_installation(allure_repo_dir: str, node_dir: str) -> bool:
    """Verify that Allure CLI is properly installed and functional.

    Args:
        allure_repo_dir: Path to Allure repository
        node_dir: Path to Node.js installation

    Returns:
        bool: True if Allure is properly installed and functional
    """
    from sysagent.utils.infrastructure import get_node_binary_paths

    # Check if critical files exist
    cli_dist_path = os.path.join(allure_repo_dir, "packages", "cli", "dist")
    index_js_path = os.path.join(cli_dist_path, "index.js")

    if not os.path.exists(index_js_path):
        logger.debug(f"Allure CLI index.js not found at {index_js_path}")
        return False

    # Verify that 'yarn allure' command actually works
    try:
        node_bin, _, yarn_bin = get_node_binary_paths(node_dir)
        node_dir_path = os.path.dirname(node_bin)
        env = os.environ.copy()
        env["PATH"] = f"{node_dir_path}{os.pathsep}{env.get('PATH', '')}"

        result = run_command([yarn_bin, "allure", "--help"], cwd=allure_repo_dir, check=False, env=env, timeout=10)

        if not result.success:
            logger.debug(f"Allure CLI verification failed: {result.stderr}")
            return False

        logger.debug("Allure CLI verification successful")
        return True

    except Exception as e:
        logger.debug(f"Allure CLI verification failed with exception: {e}")
        return False


def _cleanup_corrupted_allure(allure_repo_dir: str) -> None:
    """Clean up corrupted or incomplete Allure installation.

    Removes the entire allure3 directory to ensure a fresh start.
    The installation process will re-download and rebuild from scratch.

    Args:
        allure_repo_dir: Path to Allure repository to remove
    """
    import shutil

    logger.debug(f"Removing corrupted Allure installation at {allure_repo_dir}")

    if not os.path.exists(allure_repo_dir):
        logger.debug("Allure directory does not exist, nothing to clean")
        return

    try:
        shutil.rmtree(allure_repo_dir)
        logger.debug("Removed entire Allure directory for fresh reinstall")
    except Exception as e:
        logger.warning(f"Failed to remove Allure directory: {e}")


def install_allure_cli_from_repo(node_dir: str, force_reinstall: bool = False) -> str:
    """
    Install Allure CLI from the repository with patching support
    s
    Args:
        node_dir: Directory where Node.js is installed
        force_reinstall: Force reinstallation even if Allure CLI is already installed

    Returns:
        str: Command prefix for running Allure CLI (yarn allure)

    Raises:
        FileNotFoundError: If Allure3 repository is not found
        subprocess.CalledProcessError: If installation fails
    """
    from sysagent.utils.config import get_thirdparty_dir
    from sysagent.utils.infrastructure import get_node_binary_paths

    from .patch import apply_patch

    # Get Node.js binary paths
    node_bin, _, yarn_bin = get_node_binary_paths(node_dir)
    node_dir_path = os.path.dirname(node_bin)

    # Get the Allure repo directory in thirdparty
    thirdparty_dir = get_thirdparty_dir()
    allure_repo_dir = os.path.join(thirdparty_dir, ALLURE_DIR_NAME)

    # Command to run Allure using yarn
    allure_cmd = "yarn allure"

    # Check if Allure3 repository is already built
    cli_dist_path = os.path.join(allure_repo_dir, "packages", "cli", "dist")

    # Verify if Allure CLI is properly installed and functional (not just checking if folder exists)
    if not force_reinstall and os.path.exists(allure_repo_dir):
        logger.debug("Checking existing Allure installation...")
        if _verify_allure_installation(allure_repo_dir, node_dir):
            logger.debug("Allure CLI is already installed and functional")
            return allure_cmd
        else:
            logger.debug("Existing Allure installation is corrupted or incomplete")
            _cleanup_corrupted_allure(allure_repo_dir)

    logger.debug("Installing Allure CLI from repository")

    # Check if the repo directory exists, re-download if completely missing
    if not os.path.exists(allure_repo_dir) or not os.path.isdir(allure_repo_dir):
        logger.warning(f"Allure3 repository not found at {allure_repo_dir}")
        logger.debug("Attempting to download Allure3 repository...")
        try:
            # Import here to avoid circular dependency
            from sysagent.utils.infrastructure import download_github_repo

            download_github_repo()

            # Verify download succeeded
            if not os.path.exists(allure_repo_dir) or not os.path.isdir(allure_repo_dir):
                raise FileNotFoundError(f"Failed to download Allure3 repository to {allure_repo_dir}.")
            logger.debug("Allure3 repository downloaded successfully")
        except Exception as e:
            raise FileNotFoundError(
                f"Allure3 repository not found at {allure_repo_dir} and download failed: {e}. "
            ) from e

    # Find all patch files in the patches directory
    # Always use sysagent core directory for allure3 patches (not extension packages)
    from sysagent.utils.config import get_sysagent_core_directory

    core_dir = get_sysagent_core_directory()
    patches_found = False
    patches_dir = None

    if core_dir:
        patches_dir = os.path.join(core_dir, "patches", "allure3")
        if os.path.exists(patches_dir):
            patches_found = True

    if patches_found:
        patch_files = [f for f in os.listdir(patches_dir) if f.endswith(".patch")]
        logger.debug(f"Found {len(patch_files)} patch files in {patches_dir}")

        # Apply patches using subprocess
        for patch_file in patch_files:
            patch_path = os.path.join(patches_dir, patch_file)
            logger.debug(f"Applying patch: {patch_file}")

            try:
                patch_applied = apply_patch(patch_path, allure_repo_dir)
                if patch_applied:
                    logger.debug(f"Successfully applied patch: {patch_file}")
                else:
                    logger.debug(f"Patch already applied or skipped: {patch_file}")
            except Exception as e:
                logger.error(f"Failed to apply patch {patch_file}: {str(e)}")
                raise
    else:
        logger.warning(f"Patches directory not found: {patches_dir}")

    # Install dependencies and build the project
    env = os.environ.copy()
    env["PATH"] = f"{node_dir_path}{os.pathsep}{env.get('PATH', '')}"

    # Step 1: Install dependencies with yarn
    allure_repo_relative = os.path.relpath(allure_repo_dir, os.getcwd())
    logger.debug(f"Installing dependencies in {allure_repo_relative}")
    try:
        result = run_command([yarn_bin, "install"], cwd=allure_repo_dir, check=False, env=env)

        if not result.success:
            logger.error(f"Dependency installation failed with exit code {result.returncode}")
            if result.stdout:
                logger.error(f"Install stdout:\n{result.stdout}")
            if result.stderr:
                logger.error(f"Install stderr:\n{result.stderr}")
            logger.debug("Cleaning up failed installation...")
            _cleanup_corrupted_allure(allure_repo_dir)
            raise Exception(f"yarn install failed with exit code {result.returncode}")

        logger.debug("Dependencies installed successfully")
    except Exception as e:
        if "yarn install failed" not in str(e):
            logger.error(f"Failed to install dependencies: {e}")
            logger.debug("Cleaning up failed installation...")
            _cleanup_corrupted_allure(allure_repo_dir)
        raise

    # Step 2: Build the project
    logger.debug("Building Allure3 project")
    try:
        result = run_command([yarn_bin, "build"], cwd=allure_repo_dir, check=False, env=env)

        if not result.success:
            # Log the actual build error output
            logger.error(f"Allure3 build failed with exit code {result.returncode}")
            if result.stdout:
                logger.error(f"Build stdout:\n{result.stdout}")
            if result.stderr:
                logger.error(f"Build stderr:\n{result.stderr}")
            logger.debug("Cleaning up failed build...")
            _cleanup_corrupted_allure(allure_repo_dir)
            raise Exception(f"yarn build failed with exit code {result.returncode}")

        logger.debug("Allure3 project built successfully")
    except Exception as e:
        if "yarn build failed" not in str(e):
            logger.error(f"Failed to build Allure3 project: {e}")
            logger.debug("Cleaning up failed build...")
            _cleanup_corrupted_allure(allure_repo_dir)
        raise

    # Verify that 'yarn allure' works
    try:
        result = run_command([yarn_bin, "allure", "--help"], cwd=allure_repo_dir, check=True, env=env)
        if not result.success:
            raise RuntimeError(f"Unable to run 'yarn allure': {result.stderr}")
        logger.debug("Verified 'yarn allure' command works")
    except Exception as e:
        logger.error(f"Failed to verify 'yarn allure' command: {str(e)}")
        logger.debug("Cleaning up failed installation...")
        _cleanup_corrupted_allure(allure_repo_dir)
        raise

    # Final verification that the installation is complete and functional
    if not _verify_allure_installation(allure_repo_dir, node_dir):
        logger.error("Allure CLI installation verification failed after build")
        logger.debug("Cleaning up failed installation...")
        _cleanup_corrupted_allure(allure_repo_dir)
        raise RuntimeError("Allure CLI installation verification failed")

    logger.debug("Allure CLI installed and verified successfully")
    return allure_cmd


def generate_allure_report(
    node_dir: str,
    results_dir: str,
    report_dir: str,
    report_name: Optional[str] = None,
    report_version: Optional[str] = None,
    force_reinstall: bool = False,
    debug: bool = False,
) -> int:
    """
    Generate an Allure report using the Allure3 repository.

    Args:
        node_dir: Directory where Node.js is installed
        results_dir: Directory containing Allure results
        report_dir: Directory where the report should be generated
        report_name: Custom name for the Allure report (if provided)
        report_version: Version string to use for allureVersion in config (if provided)
        force_reinstall: Force reinstallation of Allure CLI
        debug: Whether to show debug level logs

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    from sysagent.utils.config import get_thirdparty_dir
    from sysagent.utils.infrastructure import get_node_binary_paths

    from .config import update_allure_config

    # Get the Allure repository directory in thirdparty
    thirdparty_dir = get_thirdparty_dir()
    allure_repo_dir = os.path.join(thirdparty_dir, ALLURE_DIR_NAME)

    # Install/setup Allure CLI if needed
    try:
        logger.debug(f"Setting up Allure CLI from repository: {allure_repo_dir} and node directory: {node_dir}")
        install_allure_cli_from_repo(node_dir, force_reinstall)
    except Exception as e:
        logger.error(f"Failed to setup Allure CLI from repository: {e}")
        return 1

    # Get the path to the Allure configuration file
    from sysagent.utils.config import get_reports_directory

    reports_dir = get_reports_directory()
    allure_config_path = os.path.join(reports_dir, "allure", "allurerc.mjs")

    # Copy and update the configuration file
    try:
        project_config_path = update_allure_config(allure_config_path, allure_repo_dir, report_name, report_version)
        allure_config_path = project_config_path
    except Exception as e:
        logger.error(f"Failed to setup Allure configuration: {e}")
        return 1

    # Generate report
    logger.debug(f"Generating Allure report from {results_dir}")
    logger.debug(f"Using Allure configuration from {allure_config_path}")

    # Get Node.js binary paths for environment setup
    node_bin, _, yarn_bin = get_node_binary_paths(node_dir)
    node_dir_path = os.path.dirname(node_bin)

    try:
        cmd_str = f"{yarn_bin} allure generate {results_dir} -o {report_dir}"

        if allure_config_path:
            logger.debug(f"Setting custom allure config path: {allure_config_path}")
            cmd_str += " -c allurerc.mjs"

        cmd = cmd_str.split()
        env = os.environ.copy()
        env["PATH"] = f"{node_dir_path}{os.pathsep}{env.get('PATH', '')}"

        logger.debug(f"Repo directory: {allure_repo_dir}")
        result = run_command(
            cmd,
            cwd=allure_repo_dir,
            check=False,  # Don't raise an exception on non-zero exit code
            env=env,
        )

        if not result.success:
            logger.error(f"Failed to generate Allure report: {result.stderr}")
            return result.returncode

        logger.debug(f"Allure report generated successfully at {report_dir}")

        # Generate final timestamped copy with system information
        final_report_path = generate_final_report_copy(report_dir, debug)
        if final_report_path:
            # Convert to relative path for cleaner CLI output
            relative_path = os.path.relpath(final_report_path, os.getcwd())
            logger.info(f"\nReport available at: {relative_path}")
        else:
            logger.warning("Failed to generate final report copy")
            # Convert to relative path for cleaner CLI output
            fallback_path = os.path.relpath(os.path.join(report_dir, "index.html"), os.getcwd())
            logger.info(f"\nReport available at: {fallback_path}")

        return 0
    except Exception as e:
        logger.error(f"Error generating Allure report: {e}")
        return 1


def generate_final_report_copy(report_dir: str, debug: bool = False) -> Optional[str]:
    """
    Generate a timestamped final copy of the Allure report.
    Format: <appname>_report_<system_and_productname>_<normalize_cpu_brand>
    _<list_of_discrete_gpus_if_available>_<timestamp>.html

    Args:
        report_dir: Directory containing the generated Allure report
        debug: Whether to show debug level logs

    Returns:
        str: Path to the final report file, or None if generation failed
    """
    from .config import (
        cleanup_old_final_reports,
        generate_final_report_filename,
        get_comprehensive_system_info_for_filename,
    )

    try:
        # Check if the standard report exists
        source_report = os.path.join(report_dir, "index.html")
        if not os.path.exists(source_report):
            logger.error(f"Source report not found: {source_report}")
            return None

        # Clean up old final reports first to keep only the latest
        cleanup_old_final_reports(report_dir, debug)

        # Generate components for filename
        app_name = get_app_name()
        system_info = get_comprehensive_system_info_for_filename()

        logger.debug(f"System info for filename: {system_info}")
        logger.debug(f"App name for filename: {app_name}")

        # Generate final filename
        final_filename = generate_final_report_filename(app_name, system_info)
        final_report_path = os.path.join(report_dir, final_filename)

        # Copy the report file
        shutil.copy2(source_report, final_report_path)

        logger.debug(f"Final report copy created: {final_report_path}")
        return final_report_path

    except Exception as e:
        logger.error(f"Failed to generate final report copy: {e}")
        if debug:
            logger.error(f"Error details: {e}", exc_info=True)
        return None


def get_app_name() -> str:
    """
    Get the application name for filename prefix.
    Uses CLI-aware project name to properly prefix reports based on the CLI extension package.

    Returns:
        str: Application name or 'app' as fallback
    """
    from sysagent.utils.config import get_cli_aware_project_name

    from .config import normalize_filename_component

    try:
        # Use CLI-aware project name to get the correct prefix (e.g., 'esq', 'test-esq', 'sysagent')
        app_name = get_cli_aware_project_name()
        if app_name:
            # Clean up app name for filename (remove problematic characters)
            cleaned = normalize_filename_component(app_name)
            return cleaned or "app"
        return "app"
    except Exception:
        return "app"
