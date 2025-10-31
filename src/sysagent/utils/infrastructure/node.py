# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for managing Node.js installation.
"""

import logging
import os
import platform
import shutil
import tarfile
import zipfile
from typing import Tuple

# Import secure process execution utilities
from sysagent.utils.core.process import run_command

logger = logging.getLogger(__name__)

# Node.js configuration
NODE_DIR_NAME = "node"


def get_node_version():
    """Get the Node.js version from configuration."""
    try:
        from sysagent.utils.config import get_node_version

        return get_node_version()
    except ImportError:
        # Fallback version if config loader not available
        return "v20.11.0"


NODE_VERSION = get_node_version()
NODE_LINUX_URL = f"https://nodejs.org/dist/{NODE_VERSION}/node-{NODE_VERSION}-linux-x64.tar.gz"
NODE_WINDOWS_URL = f"https://nodejs.org/dist/{NODE_VERSION}/node-{NODE_VERSION}-win-x64.zip"


def _validate_url_scheme(url: str) -> None:
    """
    Validate that URL uses only safe schemes (http/https).

    Args:
        url: URL to validate

    Raises:
        ValueError: If URL scheme is not safe
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(f"Unsafe URL scheme '{parsed.scheme}'. Only http/https are allowed.")


def download_file(url: str, target_path: str) -> None:
    """
    Download a file from a URL to a target path.

    Args:
        url: URL to download from
        target_path: Path to save the downloaded file
    """
    logger.debug(f"Downloading {url} to {target_path}")
    _validate_url_scheme(url)

    try:
        import requests

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.debug(f"Download completed: {target_path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


def is_nodejs_installed(node_dir: str) -> bool:
    """
    Check if Node.js is installed in the specified directory.

    Args:
        node_dir: Directory where Node.js should be installed

    Returns:
        bool: True if Node.js is installed, False otherwise
    """
    # Check if node/npm executables exist
    node_exe = (
        os.path.join(node_dir, "bin", "node") if platform.system() != "Windows" else os.path.join(node_dir, "node.exe")
    )
    npm_exe = (
        os.path.join(node_dir, "bin", "npm") if platform.system() != "Windows" else os.path.join(node_dir, "npm.cmd")
    )

    return os.path.exists(node_exe) and os.path.exists(npm_exe)


def get_node_binary_paths(node_dir: str) -> Tuple[str, str, str]:
    """
    Get paths to node, npm, and yarn binaries.

    Args:
        node_dir: Directory where Node.js is installed

    Returns:
        Tuple[str, str, str]: Paths to node, npm, and yarn binaries
    """
    if platform.system() == "Windows":
        node_bin = os.path.join(node_dir, "node.exe")
        npm_bin = os.path.join(node_dir, "npm.cmd")
    else:
        node_bin = os.path.join(node_dir, "bin", "node")
        npm_bin = os.path.join(node_dir, "bin", "npm")

    logger.debug(f"Node binary path: {node_bin}")
    logger.debug(f"npm binary path: {npm_bin}")

    if not os.path.exists(node_bin) or not os.path.exists(npm_bin):
        raise FileNotFoundError(f"Node.js binaries not found in {node_dir}")

    # Ensure Yarn is installed globally before using it
    yarn_bin = install_yarn_global(node_dir, node_bin, npm_bin)
    logger.debug(f"Yarn binary path: {yarn_bin}")

    logger.debug(f"Node binary path: {node_bin}")
    logger.debug(f"npm binary path: {npm_bin}")

    return node_bin, npm_bin, yarn_bin


def get_yarn_binary_path(node_dir: str) -> str:
    """
    Get the path to the globally installed yarn binary.
    Args:
        node_dir: Directory where Node.js is installed
    Returns:
        str: Path to yarn binary, or empty string if not found
    """
    if platform.system() == "Windows":
        yarn_bin = os.path.join(node_dir, "yarn.cmd")
    else:
        yarn_bin = os.path.join(node_dir, "bin", "yarn")
    logger.debug(f"Yarn binary path: {yarn_bin}")
    return yarn_bin if os.path.exists(yarn_bin) else ""


def install_yarn_global(node_dir: str, node_bin: str, npm_bin: str) -> str:
    """
    Install Yarn globally if it's not already installed.

    Args:
        node_dir: Directory where Node.js is installed
        node_bin: Path to node binary
        npm_bin: Path to npm binary

    Returns:
        str: Path to yarn binary
    """
    yarn_bin = get_yarn_binary_path(node_dir)

    if yarn_bin and os.path.exists(yarn_bin):
        logger.debug("Yarn is already installed globally")
        return yarn_bin

    logger.debug("Installing Yarn globally")
    try:
        # Prepare environment with Node.js path
        env = os.environ.copy()
        node_dir_path = os.path.dirname(node_bin)
        env["PATH"] = f"{node_dir_path}{os.pathsep}{env.get('PATH', '')}"

        # Install yarn globally using npm
        cmd = [npm_bin, "install", "-g", "yarn"]
        logger.debug(f"Running command: {cmd}")
        result = run_command(cmd, env=env, check=True)
        logger.debug(f"Installation exit code: {result.returncode}")
        logger.debug("Yarn installed successfully")

        # Get the yarn binary path after installation
        yarn_bin = get_yarn_binary_path(node_dir)
        logger.debug(f"Yarn binary path after installation: {yarn_bin}")
        if not yarn_bin or not os.path.exists(yarn_bin):
            raise RuntimeError("Yarn binary not found after installation")

        return yarn_bin

    except Exception as e:
        logger.error(f"Failed to install Yarn: {e}")
        if hasattr(e, "stderr"):
            logger.error(f"npm stderr: {e.stderr}")
        raise RuntimeError(f"Failed to install Yarn: {e}")


def extract_nodejs_archive(archive_path: str, extract_dir: str) -> str:
    """
    Extract Node.js archive to specified directory.

    Args:
        archive_path: Path to the downloaded archive
        extract_dir: Directory to extract to

    Returns:
        str: Path to the extracted Node.js directory
    """
    logger.debug(f"Extracting {archive_path} to {extract_dir}")

    if archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)
            # Get the name of the extracted directory
            extracted_name = tar.getnames()[0].split("/")[0]
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
            # Get the name of the extracted directory
            extracted_name = zip_ref.namelist()[0].split("/")[0]
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    extracted_path = os.path.join(extract_dir, extracted_name)
    logger.debug(f"Extracted to: {extracted_path}")
    return extracted_path


def install_nodejs(thirdparty_dir: str) -> str:
    """
    Install Node.js to the thirdparty directory.

    Args:
        thirdparty_dir: Directory where thirdparty tools are installed

    Returns:
        str: Path to the Node.js installation directory
    """
    node_dir = os.path.join(thirdparty_dir, NODE_DIR_NAME)

    # Check if already installed
    if is_nodejs_installed(node_dir):
        logger.debug(f"Node.js is already installed at {node_dir}")
        return node_dir

    logger.debug(f"Installing Node.js {NODE_VERSION} to {node_dir}")

    # Determine download URL based on platform
    if platform.system() == "Windows":
        download_url = NODE_WINDOWS_URL
        archive_name = f"node-{NODE_VERSION}-win-x64.zip"
    else:
        download_url = NODE_LINUX_URL
        archive_name = f"node-{NODE_VERSION}-linux-x64.tar.gz"

    # Create thirdparty directory if it doesn't exist
    os.makedirs(thirdparty_dir, exist_ok=True)

    # Download Node.js
    archive_path = os.path.join(thirdparty_dir, archive_name)
    logger.debug(f"Downloading Node.js from {download_url} to {archive_path}")
    download_file(download_url, archive_path)

    try:
        # Extract the archive
        extracted_path = extract_nodejs_archive(archive_path, thirdparty_dir)
        logger.debug(f"Extracted Node.js to: {extracted_path}")

        # Remove existing node directory if it exists
        if os.path.exists(node_dir):
            shutil.rmtree(node_dir)

        # Rename extracted directory to standard name
        os.rename(extracted_path, node_dir)

        # Verify installation
        if not is_nodejs_installed(node_dir):
            raise RuntimeError("Node.js installation verification failed")

        logger.debug(f"Node.js installed successfully at {node_dir}")
        return node_dir

    finally:
        # Clean up downloaded archive
        if os.path.exists(archive_path):
            os.remove(archive_path)
            logger.debug(f"Cleaned up archive: {archive_path}")


def setup_nodejs() -> str:
    """
    Set up Node.js and return binary paths.

    Returns:
        str: Path to the Node.js installation directory
    """
    logger.debug("Setting up Node.js environment")
    try:
        from sysagent.utils.config import get_thirdparty_dir

        thirdparty_dir = get_thirdparty_dir()
    except ImportError:
        logger.debug("Config not available, using fallback third-party directory.")
        # Fallback if config not available
        thirdparty_dir = os.path.join(os.getcwd(), "thirdparty")

    logger.debug(f"Third-party directory: {thirdparty_dir}")
    node_dir = install_nodejs(thirdparty_dir)
    logger.debug(f"Node.js installed at: {node_dir}")
    return node_dir


def verify_nodejs_installation(node_dir: str) -> bool:
    """
    Verify that Node.js installation is working correctly.

    Args:
        node_dir: Directory where Node.js is installed

    Returns:
        bool: True if installation is working, False otherwise
    """
    try:
        node_bin, npm_bin, yarn_bin = get_node_binary_paths(node_dir)

        # Test node
        result = run_command([node_bin, "--version"])
        if not result.success:
            logger.error(f"Node.js version check failed: {result.stderr}")
            return False
        logger.debug(f"Node.js version: {result.stdout.strip()}")

        # Test npm
        result = run_command([npm_bin, "--version"])
        if not result.success:
            logger.error(f"npm version check failed: {result.stderr}")
            return False
        logger.debug(f"npm version: {result.stdout.strip()}")

        # Test yarn
        if yarn_bin:
            result = run_command([yarn_bin, "--version"])
            if not result.success:
                logger.warning(f"Yarn version check failed: {result.stderr}")
            else:
                logger.debug(f"Yarn version: {result.stdout.strip()}")

        return True

    except Exception as e:
        logger.error(f"Node.js installation verification failed: {e}")
        return False


def get_node_env_vars(node_dir: str) -> dict:
    """
    Get environment variables needed for Node.js.

    Args:
        node_dir: Directory where Node.js is installed

    Returns:
        dict: Environment variables to set
    """
    env_vars = {}

    if platform.system() == "Windows":
        # Add node directory to PATH
        env_vars["PATH"] = f"{node_dir};{os.environ.get('PATH', '')}"
    else:
        # Add node bin directory to PATH
        bin_dir = os.path.join(node_dir, "bin")
        env_vars["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

    return env_vars
