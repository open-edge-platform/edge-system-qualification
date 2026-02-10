# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Virtual Environment Management Utilities.

This module provides a centralized, secure way to manage isolated virtual environments
for test suites with specific dependency requirements. It uses the 'uv' package manager
for fast and reliable environment creation and dependency installation.

Key Features:
- Create isolated virtual environments for test suites
- Install dependencies from requirements.txt files
- Run commands in isolated environments
- Manage multiple venvs in data directory
- Cleanup and maintenance utilities

All venv operations use the secure process utilities for consistency and security.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VenvManager:
    """
    Manager for isolated virtual environments using uv package manager.

    This class provides comprehensive venv management for test suites that require
    specific dependency versions isolated from the main environment.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the VenvManager.

        Args:
            data_dir: Base data directory where venvs will be stored
        """
        self.data_dir = Path(data_dir)
        self.venvs_dir = self.data_dir / "venvs"
        self.venvs_dir.mkdir(parents=True, exist_ok=True)

        # Check if uv is available
        self._check_uv_available()

    def _check_uv_available(self) -> bool:
        """
        Check if uv command is available.

        Returns:
            bool: True if uv is available, False otherwise

        Raises:
            RuntimeError: If uv is not available
        """
        from sysagent.utils.core.process import check_command_available

        if not check_command_available("uv"):
            raise RuntimeError(
                "uv package manager is not available. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
            )
        return True

    def get_venv_name(self, suite_path: str, requirements_hash: Optional[str] = None) -> str:
        """
        Generate a unique venv name based on suite path and requirements hash.

        Args:
            suite_path: Path to the test suite (e.g., "esq/suites/ai/gen")
            requirements_hash: Optional hash of requirements.txt content

        Returns:
            str: Unique venv name
        """
        # Extract path starting from the folder after src/esq or src/sysagent
        path_parts = Path(suite_path).parts

        # Try to find src/esq or src/sysagent and start from the next folder
        start_index = 0
        for i in range(len(path_parts) - 1):
            if path_parts[i] == "src" and path_parts[i + 1] in ["esq", "sysagent"]:
                # Start from the folder after 'esq' or 'sysagent'
                start_index = i + 2
                break

        # If we found a match, use the path from that point
        if start_index > 0 and start_index < len(path_parts):
            suite_path = str(Path(*path_parts[start_index:]))

        # Normalize suite path
        normalized_path = suite_path.replace("/", "_").replace("\\", "_").strip("_")

        # Add hash suffix if provided for requirements versioning
        if requirements_hash:
            return f"{normalized_path}_{requirements_hash[:8]}"
        return normalized_path

    def get_venv_path(self, venv_name: str) -> Path:
        """
        Get the full path to a venv directory.

        Args:
            venv_name: Name of the virtual environment

        Returns:
            Path: Full path to venv directory
        """
        return self.venvs_dir / venv_name

    def get_requirements_hash(self, requirements_file: str) -> str:
        """
        Calculate hash of requirements.txt file content.

        Args:
            requirements_file: Path to requirements.txt file

        Returns:
            str: SHA256 hash of file content
        """
        requirements_path = Path(requirements_file)
        if not requirements_path.exists():
            logger.warning(f"Requirements file not found: {requirements_file}")
            return "no_requirements"

        with open(requirements_path, "rb") as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()

    def venv_exists(self, venv_name: str) -> bool:
        """
        Check if a venv exists and is valid.

        Args:
            venv_name: Name of the virtual environment

        Returns:
            bool: True if venv exists and is valid
        """
        venv_path = self.get_venv_path(venv_name)

        # Check if directory exists
        if not venv_path.exists():
            return False

        # Check for key venv files
        python_bin = venv_path / "bin" / "python"
        activate_script = venv_path / "bin" / "activate"

        return python_bin.exists() and activate_script.exists()

    def create_venv(
        self, venv_name: str, python_version: Optional[str] = None, force: bool = False
    ) -> Tuple[bool, str]:
        """
        Create a new virtual environment using uv.

        Args:
            venv_name: Name for the virtual environment
            python_version: Optional Python version (e.g., "3.10", "3.11")
            force: Whether to recreate if venv already exists

        Returns:
            Tuple[bool, str]: (success, message)
        """
        from sysagent.utils.core.process import run_command

        venv_path = self.get_venv_path(venv_name)

        # Check if already exists
        if self.venv_exists(venv_name):
            if not force:
                logger.info(f"Virtual environment already exists: {venv_name}")
                return True, f"Venv already exists: {venv_path}"
            else:
                logger.info(f"Recreating virtual environment: {venv_name}")
                self.remove_venv(venv_name)

        # Build uv venv command
        cmd = ["uv", "venv", str(venv_path)]

        if python_version:
            cmd.extend(["--python", python_version])

        logger.info(f"Creating virtual environment: {venv_name}")
        logger.debug(f"Command: {' '.join(cmd)}")

        # Create venv
        result = run_command(command=cmd, timeout=120.0, check=False, stream_output=True)

        if result.success:
            logger.info(f"Successfully created venv: {venv_name}")
            return True, f"Created venv at: {venv_path}"
        else:
            error_msg = f"Failed to create venv: {result.stderr}"
            logger.error(error_msg)
            return False, error_msg

    def install_requirements(self, venv_name: str, requirements_file: str, timeout: float = 600.0) -> Tuple[bool, str]:
        """
        Install dependencies from requirements.txt into venv.

        Args:
            venv_name: Name of the virtual environment
            requirements_file: Path to requirements.txt file
            timeout: Timeout for installation in seconds

        Returns:
            Tuple[bool, str]: (success, message)
        """
        from sysagent.utils.core.process import run_command

        # Verify venv exists
        if not self.venv_exists(venv_name):
            error_msg = f"Virtual environment does not exist: {venv_name}"
            logger.error(error_msg)
            return False, error_msg

        # Verify requirements file exists
        requirements_path = Path(requirements_file)
        if not requirements_path.exists():
            error_msg = f"Requirements file not found: {requirements_file}"
            logger.error(error_msg)
            return False, error_msg

        venv_path = self.get_venv_path(venv_name)
        python_bin = venv_path / "bin" / "python"

        # Use uv pip to install requirements
        cmd = ["uv", "pip", "install", "-r", str(requirements_path), "--python", str(python_bin)]

        logger.info(f"Installing requirements into venv: {venv_name}")
        logger.debug(f"Requirements file: {requirements_file}")
        logger.debug(f"Command: {' '.join(cmd)}")

        result = run_command(command=cmd, timeout=timeout, check=False, stream_output=True)

        if result.success:
            logger.info(f"Successfully installed requirements into: {venv_name}")
            return True, "Requirements installed successfully"
        else:
            error_msg = f"Failed to install requirements: {result.stderr}"
            logger.error(error_msg)
            return False, error_msg

    def install_project_packages(self, venv_name: str, timeout: float = 300.0) -> Tuple[bool, str]:
        """
        Install the esq package which will automatically install sysagent as a dependency.

        This method handles multiple installation scenarios:
        1. Development mode: Project root with src/esq - installs in editable mode
        2. uv tool install: Package in site-packages directory - creates .pth link
        3. pip install: Package in site-packages directory - creates .pth link
        4. Mixed environments: Any combination of the above

        Note: Only esq is installed directly as it declares sysagent as a dependency in pyproject.toml.

        Args:
            venv_name: Name of the virtual environment
            timeout: Installation timeout in seconds

        Returns:
            Tuple[bool, str]: (success, message)
        """
        import importlib.metadata

        from sysagent.utils.core.process import run_command

        if not self.venv_exists(venv_name):
            error_msg = f"Venv does not exist: {venv_name}"
            logger.error(error_msg)
            return False, error_msg

        venv_path = self.get_venv_path(venv_name)
        python_bin = venv_path / "bin" / "python"

        # Find the esq package root to install
        # sysagent will be installed automatically as a dependency
        pkg_root = None
        pkg_name = "esq"

        # Strategy: Try to find esq package and determine its installation path
        # Use importlib.metadata to get accurate package location
        # Find the esq package root to install
        # sysagent will be installed automatically as a dependency
        pkg_root = None
        pkg_name = "esq"

        # Strategy: Try to find esq package and determine its installation path
        # Use importlib.metadata to get accurate package location
        try:
            # Method 1: Use importlib.metadata.distribution to find package location
            # This works for both installed packages and editable installs
            try:
                dist = importlib.metadata.distribution(pkg_name)

                # Try to get the editable install location from direct_url.json
                try:
                    # Check if this is an editable install
                    files = dist.files
                    if files:
                        # For editable installs, look for .pth or direct_url.json
                        first_file = files[0]
                        pkg_location = first_file.locate().parent

                        # Check current directory first, then walk up to find pyproject.toml
                        candidates_to_check = [pkg_location] + list(pkg_location.parents)
                        for candidate in candidates_to_check:
                            pyproject = candidate / "pyproject.toml"
                            if pyproject.exists():
                                # Check if this is a project root with src/ structure
                                src_dir = candidate / "src"
                                if src_dir.exists() and (src_dir / pkg_name).exists():
                                    # Development mode: found src/ structure
                                    pkg_root = candidate
                                    logger.debug(f"Found {pkg_name} in development mode: {candidate}")
                                    break
                                # Check if this is an installed package with its own pyproject.toml
                                elif (candidate / pkg_name).exists() or candidate.name == pkg_name:
                                    # Installed mode: package has its own pyproject.toml
                                    pkg_root = candidate
                                    logger.debug(f"Found {pkg_name} as installed package: {candidate}")
                                    break
                except (AttributeError, TypeError, OSError):
                    # Fall back to module import method
                    pass

            except importlib.metadata.PackageNotFoundError:
                logger.debug(f"Package {pkg_name} not found via importlib.metadata")

            # Method 2: Try direct import and walk up from module location
            if pkg_root is None:
                try:
                    module = importlib.import_module(pkg_name)
                    module_file = Path(module.__file__)
                    pkg_dir = module_file.parent

                    # Check current directory first, then walk up to find pyproject.toml
                    candidates_to_check = [pkg_dir] + list(pkg_dir.parents)
                    for candidate in candidates_to_check:
                        pyproject = candidate / "pyproject.toml"
                        if pyproject.exists():
                            # Check for development mode (src/ structure)
                            src_dir = candidate / "src"
                            if src_dir.exists() and (src_dir / pkg_name).exists():
                                pkg_root = candidate
                                logger.debug(f"Found {pkg_name} in dev mode via import: {candidate}")
                                break
                            # Check for installed package mode (has pyproject.toml in package dir)
                            elif (candidate / pkg_name).exists() or candidate.name == pkg_name:
                                pkg_root = candidate
                                logger.debug(f"Found {pkg_name} as installed via import: {candidate}")
                                break
                except ImportError:
                    logger.debug(f"Could not import {pkg_name}")

        except Exception as e:
            logger.debug(f"Error locating {pkg_name}: {e}")

        except Exception as e:
            logger.debug(f"Error locating {pkg_name}: {e}")

        # Method 3: Fallback to CWD search (for development mode when package not yet installed)
        if pkg_root is None:
            logger.debug("Package not found via installed modules, checking CWD for development mode")
            cwd_root = Path.cwd()
            candidates_checked = []

            while cwd_root != cwd_root.parent:
                pyproject = cwd_root / "pyproject.toml"
                candidates_checked.append(str(cwd_root))

                if pyproject.exists():
                    src_dir = cwd_root / "src"
                    if src_dir.exists() and (src_dir / pkg_name).exists():
                        pkg_root = cwd_root
                        logger.debug(f"Found development project root via CWD: {cwd_root}")
                        break
                cwd_root = cwd_root.parent

            if pkg_root is None:
                logger.debug(f"Checked CWD candidates: {candidates_checked}")

        if pkg_root is None:
            error_msg = f"Could not find {pkg_name} package to install"
            logger.error(error_msg)
            return False, error_msg

        # Install the package (esq will automatically install sysagent as dependency)
        logger.info(f"Installing {pkg_name} package in venv: {venv_name}")

        # Determine if this is development mode or installed mode
        # Development mode: has src/ directory with package
        # Installed mode: package is in site-packages without src/ structure
        is_dev_mode = (pkg_root / "src").exists() and (pkg_root / "src" / pkg_name).exists()

        # Determine if this is development mode or installed mode
        # Development mode: has src/ directory with package
        # Installed mode: package is in site-packages without src/ structure
        is_dev_mode = (pkg_root / "src").exists() and (pkg_root / "src" / pkg_name).exists()

        if is_dev_mode:
            # Development mode: install in editable mode
            # This will also install sysagent as a dependency from pyproject.toml
            cmd = ["uv", "pip", "install", "-e", str(pkg_root), "--python", str(python_bin)]
            logger.debug(f"Installing in editable mode (development): {pkg_root}")
            logger.debug(f"Command: {' '.join(cmd)}")

            result = run_command(command=cmd, timeout=timeout, check=False, stream_output=True)

            if not result.success:
                error_msg = f"Failed to install package from {pkg_root}: {result.stderr}"
                logger.error(error_msg)
                return False, error_msg

            logger.debug(f"Successfully installed package from: {pkg_root}")
        else:
            # Installed mode: package is already in site-packages
            # Instead of reinstalling (which fails due to setuptools-scm), create .pth file
            # This makes the installed packages available to the venv
            venv_site_packages = (
                venv_path / "lib" / f"python3.{Path(python_bin).resolve().name.split('.')[1]}" / "site-packages"
            )
            if not venv_site_packages.exists():
                # Try to find the actual site-packages path
                for lib_dir in (venv_path / "lib").iterdir():
                    if lib_dir.name.startswith("python3."):
                        sp = lib_dir / "site-packages"
                        if sp.exists():
                            venv_site_packages = sp
                            break

            # Get the parent directory of the package (the site-packages directory)
            parent_site_packages = pkg_root.parent

            # Create .pth file to add the parent site-packages to Python path
            # This makes both esq and sysagent available (as they're in the same site-packages)
            pth_file = venv_site_packages / f"_esq_packages_{pkg_root.name}.pth"
            try:
                with open(pth_file, "w") as f:
                    f.write(f"{parent_site_packages}\n")
                logger.debug(f"Created .pth file linking to: {parent_site_packages}")
                logger.debug(f"Successfully made {pkg_root.name} available to venv")
            except Exception as e:
                error_msg = f"Failed to create .pth file for {pkg_root}: {e}"
                logger.error(error_msg)
                return False, error_msg

        logger.info(f"Successfully installed {pkg_name} package (with sysagent as dependency) in: {venv_name}")
        return True, f"{pkg_name} package installed successfully"

    def setup_venv(
        self, suite_path: str, requirements_file: str, python_version: Optional[str] = None, force: bool = False
    ) -> Tuple[bool, str, str]:
        """
        Complete venv setup: create venv and install requirements.

        Args:
            suite_path: Path to test suite (e.g., "esq/suites/ai/gen")
            requirements_file: Path to requirements.txt file
            python_version: Optional Python version
            force: Whether to recreate if venv already exists

        Returns:
            Tuple[bool, str, str]: (success, venv_name, message)
        """
        # Calculate requirements hash for versioning
        req_hash = self.get_requirements_hash(requirements_file)

        # Generate venv name
        venv_name = self.get_venv_name(suite_path, req_hash)

        logger.info(f"Setting up isolated venv for suite: {suite_path}")
        logger.info(f"Venv name: {venv_name}")

        # Check if venv already exists with correct requirements
        if self.venv_exists(venv_name) and not force:
            logger.info(f"Using existing venv: {venv_name}")
            return True, venv_name, f"Using existing venv: {venv_name}"

        # Create venv
        success, message = self.create_venv(venv_name, python_version, force)
        if not success:
            return False, venv_name, message

        # Install project packages (esq, sysagent) FIRST in editable mode
        # This installs the base dependencies from pyproject.toml
        success, message = self.install_project_packages(venv_name)
        if not success:
            return False, venv_name, message

        # Install requirements SECOND - this will override any versions from project
        # This ensures the test suite gets its specific required versions
        success, message = self.install_requirements(venv_name, requirements_file)
        if not success:
            return False, venv_name, message

        return True, venv_name, f"Venv setup complete: {venv_name}"

    def get_python_executable(self, venv_name: str) -> Optional[str]:
        """
        Get path to Python executable in venv.

        Args:
            venv_name: Name of the virtual environment

        Returns:
            Optional[str]: Path to Python executable, or None if venv doesn't exist
        """
        if not self.venv_exists(venv_name):
            logger.error(f"Venv does not exist: {venv_name}")
            return None

        venv_path = self.get_venv_path(venv_name)
        python_bin = venv_path / "bin" / "python"

        return str(python_bin)

    def run_command_in_venv(
        self,
        venv_name: str,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        check: bool = False,
        capture_output: bool = True,
        stream_output: bool = False,
    ):
        """
        Run a command in the isolated venv.

        Args:
            venv_name: Name of the virtual environment
            command: Command to run (list of arguments)
            cwd: Working directory
            env: Environment variables
            timeout: Command timeout
            check: Whether to raise exception on failure
            capture_output: Whether to capture output (ignored if stream_output=True)
            stream_output: Whether to stream output in real-time

        Returns:
            ProcessResult: Result of command execution
        """
        from sysagent.utils.core.process import run_command

        if not self.venv_exists(venv_name):
            raise RuntimeError(f"Venv does not exist: {venv_name}")

        # Prepare environment variables
        venv_env = env.copy() if env else os.environ.copy()
        venv_path = self.get_venv_path(venv_name)

        # Set VIRTUAL_ENV to activate the venv
        venv_env["VIRTUAL_ENV"] = str(venv_path)

        # Update PATH to use venv binaries
        bin_dir = venv_path / "bin"
        venv_env["PATH"] = f"{bin_dir}:{venv_env.get('PATH', '')}"

        # Unset PYTHONHOME if set
        venv_env.pop("PYTHONHOME", None)

        logger.debug(f"Running command in venv {venv_name}: {' '.join(command)}")

        return run_command(
            command=command,
            cwd=cwd,
            env=venv_env,
            timeout=timeout,
            check=check,
            capture_output=capture_output,
            stream_output=stream_output,
        )

    def run_pytest_in_venv(
        self, venv_name: str, pytest_args: List[str], cwd: Optional[str] = None, timeout: Optional[float] = None
    ) -> int:
        """
        Run pytest in the isolated venv.

        Args:
            venv_name: Name of the virtual environment
            pytest_args: Arguments to pass to pytest
            cwd: Working directory
            timeout: Test execution timeout

        Returns:
            int: Pytest exit code
        """
        python_bin = self.get_python_executable(venv_name)
        if not python_bin:
            logger.error(f"Cannot run pytest: venv not found: {venv_name}")
            return 1

        # Build pytest command using venv's Python
        cmd = [python_bin, "-m", "pytest"] + pytest_args

        logger.info(f"Running pytest in venv: {venv_name}")
        logger.debug(f"Pytest args: {pytest_args}")

        # Use stream_output=True to allow real-time output to terminal and log files
        # Pytest is already configured with --capture=tee-sys, --log-cli-level, and --log-file
        # This ensures pytest output appears in both terminal and parent process logs
        result = self.run_command_in_venv(
            venv_name=venv_name,
            command=cmd,
            cwd=cwd,
            timeout=timeout,
            check=False,
            capture_output=False,
            stream_output=True,  # Stream output for real-time display and logging
        )

        return result.returncode

    def list_venvs(self) -> List[Dict[str, str]]:
        """
        List all managed virtual environments.

        Returns:
            List[Dict[str, str]]: List of venv info dictionaries
        """
        venvs = []

        if not self.venvs_dir.exists():
            return venvs

        for venv_dir in self.venvs_dir.iterdir():
            if venv_dir.is_dir():
                python_bin = venv_dir / "bin" / "python"
                if python_bin.exists():
                    venvs.append({"name": venv_dir.name, "path": str(venv_dir), "python": str(python_bin)})

        return venvs

    def remove_venv(self, venv_name: str) -> Tuple[bool, str]:
        """
        Remove a virtual environment.

        Args:
            venv_name: Name of the virtual environment to remove

        Returns:
            Tuple[bool, str]: (success, message)
        """
        import shutil

        venv_path = self.get_venv_path(venv_name)

        if not venv_path.exists():
            return False, f"Venv does not exist: {venv_name}"

        try:
            logger.info(f"Removing venv: {venv_name}")
            shutil.rmtree(venv_path)
            return True, f"Removed venv: {venv_name}"
        except Exception as e:
            error_msg = f"Failed to remove venv: {e}"
            logger.error(error_msg)
            return False, error_msg

    def cleanup_all_venvs(self) -> Tuple[int, int]:
        """
        Remove all managed virtual environments.

        Returns:
            Tuple[int, int]: (success_count, failure_count)
        """
        venvs = self.list_venvs()
        success_count = 0
        failure_count = 0

        logger.info(f"Cleaning up {len(venvs)} virtual environments")

        for venv_info in venvs:
            success, message = self.remove_venv(venv_info["name"])
            if success:
                success_count += 1
            else:
                failure_count += 1
                logger.error(message)

        logger.info(f"Cleanup complete: {success_count} removed, {failure_count} failed")
        return success_count, failure_count


# Convenience functions


def get_venv_manager(data_dir: Optional[str] = None) -> VenvManager:
    """
    Get a VenvManager instance.

    Args:
        data_dir: Base data directory (defaults to CORE_DATA_DIR env var or "./app_data")

    Returns:
        VenvManager: Configured venv manager instance
    """
    if data_dir is None:
        data_dir = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "app_data"))

    return VenvManager(data_dir)


def setup_suite_venv(
    suite_path: str,
    requirements_file: str,
    data_dir: Optional[str] = None,
    python_version: Optional[str] = None,
    force: bool = False,
) -> Tuple[bool, str, str]:
    """
    Setup an isolated venv for a test suite.

    Args:
        suite_path: Path to test suite (e.g., "esq/suites/ai/gen")
        requirements_file: Path to requirements.txt file
        data_dir: Base data directory
        python_version: Optional Python version
        force: Whether to recreate if venv already exists

    Returns:
        Tuple[bool, str, str]: (success, venv_name, message)
    """
    manager = get_venv_manager(data_dir)
    return manager.setup_venv(suite_path, requirements_file, python_version, force)


def get_suite_python_executable(
    suite_path: str, requirements_file: str, data_dir: Optional[str] = None
) -> Optional[str]:
    """
    Get Python executable for a test suite's venv.

    Args:
        suite_path: Path to test suite
        requirements_file: Path to requirements.txt file
        data_dir: Base data directory

    Returns:
        Optional[str]: Path to Python executable, or None if venv doesn't exist
    """
    manager = get_venv_manager(data_dir)
    req_hash = manager.get_requirements_hash(requirements_file)
    venv_name = manager.get_venv_name(suite_path, req_hash)
    return manager.get_python_executable(venv_name)


def run_pytest_in_suite_venv(
    suite_path: str,
    requirements_file: str,
    pytest_args: List[str],
    data_dir: Optional[str] = None,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
) -> int:
    """
    Run pytest in a test suite's isolated venv.

    Args:
        suite_path: Path to test suite
        requirements_file: Path to requirements.txt file
        pytest_args: Arguments to pass to pytest
        data_dir: Base data directory
        cwd: Working directory
        timeout: Test execution timeout

    Returns:
        int: Pytest exit code
    """
    manager = get_venv_manager(data_dir)
    req_hash = manager.get_requirements_hash(requirements_file)
    venv_name = manager.get_venv_name(suite_path, req_hash)

    return manager.run_pytest_in_venv(venv_name, pytest_args, cwd, timeout)
