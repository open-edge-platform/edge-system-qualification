# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Dependency setup utilities for third-party tools.

This module provides utilities for downloading, extracting, and setting up
third-party dependencies and tools required by the system.
"""

import hashlib
import logging
import os
import platform
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
        raise ValueError(
            f"Unsafe URL scheme '{parsed.scheme}'. Only http/https are allowed."
        )


def _download_and_extract(url: str, dest_dir: Path):
    """
    Download a tar.gz or zip archive from a URL and extract it to dest_dir.
    The contents will be moved up if the archive contains a single top-level folder.
    """
    logger.info(f"Downloading: {url}")
    _validate_url_scheme(url)

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        shutil.copyfileobj(response.raw, tmp_file)
        tmp_path = tmp_file.name

    # Try to extract as zip or tar.gz
    try:
        if url.endswith(".zip"):
            with zipfile.ZipFile(tmp_path, "r") as zip_ref:
                zip_ref.extractall(dest_dir)
        else:
            with tarfile.open(tmp_path, "r:gz") as tar_ref:
                tar_ref.extractall(dest_dir)
        logger.info(f"Extracted to: {dest_dir}")

        # Flatten if there's a single top-level folder
        items = list(dest_dir.iterdir())
        if len(items) == 1 and items[0].is_dir():
            top_level = items[0]
            for item in top_level.iterdir():
                shutil.move(str(item), str(dest_dir))
            top_level.rmdir()
    finally:
        os.unlink(tmp_path)


def _download_with_progress(url: str, dest_path: Path):
    """
    Download a file with progress bar.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with (
        open(dest_path, "wb") as file,
        tqdm(
            desc=dest_path.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))


def _verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """
    Verify file checksum.
    """
    if not expected_checksum:
        return True

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    actual_checksum = sha256_hash.hexdigest()
    return actual_checksum == expected_checksum


def load_tool_config() -> Dict[str, Any]:
    """
    Load merged tool configuration from all packages.
    """
    try:
        from sysagent.utils.config.config_loader import load_merged_tool_config

        return load_merged_tool_config("tool.sysagent")
    except ImportError:
        logger.warning("Config loader not available, using fallback configuration")
        return {}


def get_thirdparty_dir() -> str:
    """
    Get the path to the third-party tools directory.

    Uses CLI-aware project name to determine the data directory:
    - esq CLI -> esq_data/thirdparty
    - sysagent CLI -> sysagent_data/thirdparty

    Returns:
        str: Path to the third-party tools directory
    """
    from sysagent.utils.config.config_loader import get_cli_aware_project_name

    project_name = get_cli_aware_project_name()
    return os.path.join(os.getcwd(), f"{project_name}_data", "thirdparty")


def setup_allure() -> str:
    """
    Set up Allure reporting tool.

    Returns:
        str: Path to Allure installation directory
    """
    tool_config = load_tool_config()
    allure_config = tool_config.get("allure", {})

    version = allure_config.get("version", "2.24.0")
    url = allure_config.get(
        "url",
        f"https://github.com/allure-framework/allure2/releases/download/{version}/allure-{version}.tgz",
    )

    thirdparty_dir = Path(get_thirdparty_dir())
    allure_dir = thirdparty_dir / "allure3"

    if allure_dir.exists():
        logger.info(f"Allure already installed at {allure_dir}")
        return str(allure_dir)

    logger.info(f"Setting up Allure {version}")
    allure_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract
    _download_and_extract(url, allure_dir)

    # Verify installation
    allure_bin = allure_dir / "bin" / "allure"
    if platform.system() == "Windows":
        allure_bin = allure_dir / "bin" / "allure.bat"

    if not allure_bin.exists():
        raise RuntimeError(f"Allure binary not found at {allure_bin}")

    logger.info(f"Allure installed successfully at {allure_dir}")
    return str(allure_dir)


def setup_edge_developer_kit() -> str:
    """
    Set up Edge Developer Kit scripts.

    Returns:
        str: Path to Edge Developer Kit installation directory
    """
    tool_config = load_tool_config()
    edk_config = tool_config.get("edge_developer_kit", {})

    url = edk_config.get(
        "url",
        "https://github.com/intel-ai-reference-kits/edge-developer-kit-reference-scripts/archive/refs/heads/main.zip",
    )

    thirdparty_dir = Path(get_thirdparty_dir())
    edk_dir = thirdparty_dir / "edge-developer-kit-reference-scripts"

    if edk_dir.exists():
        logger.info(f"Edge Developer Kit already installed at {edk_dir}")
        return str(edk_dir)

    logger.info("Setting up Edge Developer Kit reference scripts")
    edk_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract
    _download_and_extract(url, edk_dir)

    logger.info(f"Edge Developer Kit installed successfully at {edk_dir}")
    return str(edk_dir)


def setup_node() -> str:
    """
    Set up Node.js.

    Returns:
        str: Path to Node.js installation directory
    """
    from .node import setup_nodejs

    node_dir = setup_nodejs()
    return node_dir


def setup_dependency(dependency_name: str) -> str:
    """
    Set up a specific dependency.

    Args:
        dependency_name: Name of the dependency to set up

    Returns:
        str: Path to the dependency installation directory
    """
    dependency_name = dependency_name.lower()

    setup_functions = {
        "allure": setup_allure,
        "allure3": setup_allure,
        "edge-developer-kit": setup_edge_developer_kit,
        "edge_developer_kit": setup_edge_developer_kit,
        "edk": setup_edge_developer_kit,
        "node": setup_node,
        "nodejs": setup_node,
        "node.js": setup_node,
    }

    setup_func = setup_functions.get(dependency_name)
    if not setup_func:
        raise ValueError(f"Unknown dependency: {dependency_name}")

    return setup_func()


def setup_all_dependencies() -> Dict[str, str]:
    """
    Set up all configured dependencies.

    Returns:
        Dict[str, str]: Mapping of dependency names to installation paths
    """
    results = {}

    dependencies = ["allure", "edge-developer-kit", "node"]

    for dep in dependencies:
        try:
            path = setup_dependency(dep)
            results[dep] = path
            logger.info(f"Successfully set up {dep} at {path}")
        except Exception as e:
            logger.error(f"Failed to set up {dep}: {e}")
            results[dep] = None

    return results


def verify_dependency_installation(dependency_name: str, install_path: str) -> bool:
    """
    Verify that a dependency is correctly installed.

    Args:
        dependency_name: Name of the dependency
        install_path: Path where the dependency is installed

    Returns:
        bool: True if installation is valid, False otherwise
    """
    install_path = Path(install_path)

    if not install_path.exists():
        return False

    dependency_name = dependency_name.lower()

    if dependency_name in ["allure", "allure3"]:
        # Check for allure binary
        allure_bin = install_path / "bin" / "allure"
        if platform.system() == "Windows":
            allure_bin = install_path / "bin" / "allure.bat"
        return allure_bin.exists()

    elif dependency_name in ["edge-developer-kit", "edge_developer_kit", "edk"]:
        # Check for reference scripts
        return (install_path / "README.md").exists()

    elif dependency_name in ["node", "nodejs", "node.js"]:
        # Check for node binary
        from .node import is_nodejs_installed

        return is_nodejs_installed(str(install_path))

    return False


def cleanup_dependency(dependency_name: str) -> bool:
    """
    Remove a dependency installation.

    Args:
        dependency_name: Name of the dependency to remove

    Returns:
        bool: True if cleanup was successful, False otherwise
    """
    thirdparty_dir = Path(get_thirdparty_dir())

    dependency_paths = {
        "allure": thirdparty_dir / "allure3",
        "allure3": thirdparty_dir / "allure3",
        "edge-developer-kit": thirdparty_dir / "edge-developer-kit-reference-scripts",
        "edge_developer_kit": thirdparty_dir / "edge-developer-kit-reference-scripts",
        "edk": thirdparty_dir / "edge-developer-kit-reference-scripts",
        "node": thirdparty_dir / "node",
        "nodejs": thirdparty_dir / "node",
        "node.js": thirdparty_dir / "node",
    }

    dep_path = dependency_paths.get(dependency_name.lower())
    if not dep_path:
        logger.error(f"Unknown dependency: {dependency_name}")
        return False

    try:
        if dep_path.exists():
            shutil.rmtree(dep_path)
            logger.info(f"Removed {dependency_name} from {dep_path}")
        else:
            logger.info(f"{dependency_name} not found at {dep_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove {dependency_name}: {e}")
        return False


def list_installed_dependencies() -> Dict[str, Optional[str]]:
    """
    List all installed dependencies and their paths.

    Returns:
        Dict[str, Optional[str]]: Mapping of dependency names to paths
    """
    thirdparty_dir = Path(get_thirdparty_dir())

    dependencies = {
        "allure": thirdparty_dir / "allure3",
        "edge-developer-kit": thirdparty_dir / "edge-developer-kit-reference-scripts",
        "node": thirdparty_dir / "node",
    }

    results = {}
    for name, path in dependencies.items():
        if path.exists() and verify_dependency_installation(name, str(path)):
            results[name] = str(path)
        else:
            results[name] = None

    return results


def get_dependency_info(dependency_name: str) -> Dict[str, Any]:
    """
    Get information about a dependency.

    Args:
        dependency_name: Name of the dependency

    Returns:
        Dict[str, Any]: Information about the dependency
    """
    tool_config = load_tool_config()
    thirdparty_dir = Path(get_thirdparty_dir())

    dependency_name = dependency_name.lower()

    if dependency_name in ["allure", "allure3"]:
        config = tool_config.get("allure", {})
        path = thirdparty_dir / "allure3"
    elif dependency_name in ["edge-developer-kit", "edge_developer_kit", "edk"]:
        config = tool_config.get("edge_developer_kit", {})
        path = thirdparty_dir / "edge-developer-kit-reference-scripts"
    elif dependency_name in ["node", "nodejs", "node.js"]:
        config = {"version": "v20.11.0"}  # Default version
        path = thirdparty_dir / "node"
    else:
        return {"error": f"Unknown dependency: {dependency_name}"}

    return {
        "name": dependency_name,
        "config": config,
        "path": str(path),
        "installed": path.exists()
        and verify_dependency_installation(dependency_name, str(path)),
        "version": config.get("version", "unknown"),
    }


def download_github_repo(section: str = "tool.sysagent", package: str = "sysagent"):
    """
    Download all GitHub repositories defined in merged pyproject.toml configurations.
    The section should contain a 'github_dependencies' array of tables:
      github_dependencies = [
        { name = "...", url = "...", ref = "...", archive = "zip" }
      ]

    Uses merged configuration from all packages to support modular dependencies.
    """
    from sysagent.utils.config.config_loader import load_merged_tool_config

    config = load_merged_tool_config(section=section)
    github_deps = config.get("github_dependencies", [])
    if not github_deps:
        logger.info("No GitHub dependencies found in merged config.")
        return

    thirdparty_dir = get_thirdparty_dir()  # Use the CLI-aware function

    for dep in github_deps:
        dep_name = dep.get("name")
        url = dep.get("url")
        ref = dep.get("ref", "main")
        archive_format = dep.get("archive", "zip")
        if not dep_name or not url:
            logger.warning(f"Dependency missing 'name' or 'url'. Skipping: {dep}")
            continue

        # Parse repo from url (expects https://github.com/owner/repo or ...git)
        if url.startswith("https://github.com/"):
            repo = (
                url.replace("https://github.com/", "").replace(".git", "").rstrip("/")
            )
        else:
            logger.warning(f"Dependency '{dep_name}' has invalid GitHub URL: {url}")
            continue

        archive_url = f"https://github.com/{repo}/archive/{ref}.{archive_format}"
        dep_dest = Path(thirdparty_dir) / dep_name
        if dep_dest.exists() and any(dep_dest.iterdir()):
            logger.debug(
                f"Dependency '{dep_name}' already installed at {dep_dest}. Skipping."
            )
            continue

        dep_dest.mkdir(parents=True, exist_ok=True)
        try:
            # Use the robust download and extract function
            logger.info(f"Installing '{dep_name}' from {archive_url}")
            _download_and_extract(archive_url, dep_dest)
            logger.info(f"Installed '{dep_name}' to {dep_dest}")
        except Exception as e:
            logger.error(f"Failed to install '{dep_name}': {e}")


def extract_zip_file(zip_path: str, dest_dir: str):
    """
    Extracts a zip file to the destination directory.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    return str(dest_dir)


def download_file(url: str, target_path: str, sha256sum: str = "") -> Dict[str, Any]:
    """
    Download a file from a URL to the specified target path.

    Args:
        url (str): URL to download from.
        target_path (str): Path to save the downloaded file.
        sha256sum (str): Optional SHA256 checksum to verify the downloaded file.
    Returns:
        dict: Contains the path, file size in bytes, and URL.
    """
    dest_path = Path(target_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = dest_path.exists()
    file_size = dest_path.stat().st_size if file_exists else 0

    def is_valid_file(path, checksum):
        if not path.exists() or path.stat().st_size == 0:
            return False
        if checksum:
            with open(path, "rb") as fp:
                file_content = fp.read()
                sha256_hash = hashlib.sha256(file_content).hexdigest()
            return sha256_hash == checksum
        return True

    if file_exists and is_valid_file(dest_path, sha256sum):
        logger.info(f"File already exists: {os.path.basename(dest_path)}")
    else:
        if file_exists and (
            file_size == 0 or (sha256sum and not is_valid_file(dest_path, sha256sum))
        ):
            logger.warning(
                "File exists but is empty or checksum mismatch, "
                f"re-downloading: {dest_path}"
            )
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with (
                open(dest_path, "wb") as f,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=os.path.basename(target_path),
                    ncols=80,
                ) as bar,
            ):
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            logger.info(f"Downloaded file to: {dest_path}")
        except Exception as e:
            if dest_path.exists():
                dest_path.unlink()  # Remove incomplete file
            logger.error(f"Failed to download file from {url}: {e}")
            raise RuntimeError(f"Failed to download file from {url}: {e}")

        # Check file size after download
        file_size = dest_path.stat().st_size
        if file_size == 0:
            logger.error(f"Downloaded file is empty: {dest_path}")
            dest_path.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded file is empty: {dest_path}")

        # Check checksum after download
        if sha256sum and not is_valid_file(dest_path, sha256sum):
            logger.error(f"Downloaded file checksum mismatch: {dest_path}")
            dest_path.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded file checksum mismatch: {dest_path}")

    logger.debug(f"File details: size={file_size} bytes, path={dest_path}")

    return {"path": str(dest_path), "file_size_bytes": file_size, "url": url}


def download_and_prepare_audio(
    url: str,
    download_dir: str,
    wav_filename: str = "audio.wav",
    target_format: str = "webm",
    target_filename: str = "audio.webm",
) -> dict:
    """
    Download an audio file from a URL, save as WAV, and convert to target format.
    Returns a dict with details: paths, length (s), size (bytes), etc.
    """
    os.makedirs(download_dir, exist_ok=True)
    wav_path = os.path.join(download_dir, wav_filename)
    target_path = os.path.join(download_dir, target_filename)

    if not os.path.exists(wav_path):
        logger.info(f"Downloading audio from {url} to {wav_path}")
        r = requests.get(url)
        if r.status_code == 200:
            with open(wav_path, "wb") as f:
                f.write(r.content)
        else:
            raise RuntimeError(f"Failed to download audio: {url}")

    from pydub import AudioSegment

    if not os.path.exists(target_path):
        logger.info(f"Converting {wav_path} to {target_format} as {target_path}")
        audio = AudioSegment.from_wav(wav_path)
        audio.export(target_path, format=target_format)
    else:
        audio = AudioSegment.from_wav(wav_path)

    # Gather audio details
    audio_file_length_s = len(audio) / 1000.0
    file_size_bytes = os.path.getsize(target_path)
    sample_rate = audio.frame_rate
    channels = audio.channels

    return {
        "wav_path": wav_path,
        "converted_path": target_path,
        "length_seconds": audio_file_length_s,
        "file_size_bytes": file_size_bytes,
        "sample_rate": sample_rate,
        "channels": channels,
        "url": url,
    }
