# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
LPR (License Plate Recognition) Resource Downloader.

This module handles downloading LPR models and video files required for
proxy pipeline benchmarks.
"""

import logging
from pathlib import Path

from esq.utils.genutils import download_file_from_url, extract_zip_archive

logger = logging.getLogger(__name__)


def download_lpr_resources(models_dir: str, videos_dir: str) -> bool:
    """
    Download LPR models and videos from GitHub.

    Args:
        models_dir: Directory to save models
        videos_dir: Directory to save videos

    Returns:
        bool: True if all downloads successful
    """
    models_path = Path(models_dir)
    videos_path = Path(videos_dir)

    # LPR resource URLs
    # Using specific commit hash (6d452bf) for reproducibility - resources added on Aug 6, 2025
    # Commit: "Add License Plate Reader models and video (#7)"
    resources = {
        "models": {
            "url": "https://github.com/open-edge-platform/edge-ai-resources/raw/6d452bf87bb1707630f747774d2d15caa1a6f7aa/models/license-plate-reader.zip",
            "dest": models_path / "lpr",
            "is_zip": True,
        },
        "video": {
            "url": "https://github.com/open-edge-platform/edge-ai-resources/raw/6d452bf87bb1707630f747774d2d15caa1a6f7aa/videos/ParkingVideo.mp4",
            # Base video (gst_lpr_loop_mp4.sh will create _1min version)
            "dest": videos_path / "lpr" / "ParkingVideo.mp4",
            "is_zip": False,
        },
    }

    success = True

    for resource_name, resource_info in resources.items():
        url = resource_info["url"]
        dest = resource_info["dest"]
        is_zip = resource_info["is_zip"]

        if is_zip:
            # Check if already extracted
            if dest.exists() and any(dest.iterdir()):
                logger.info(f"Models already exist at: {dest}")
                continue

            # Download zip file
            zip_path = dest.parent / f"{resource_name}.zip"
            if not download_file_from_url(url, zip_path):
                success = False
                continue

            # Extract entire zip preserving directory structure
            # The zip contains a "models" subdirectory which will be extracted to dest/
            if not extract_zip_archive(zip_path, dest, search_dir=None):
                success = False
                continue

            # Clean up zip file
            zip_path.unlink(missing_ok=True)

        else:
            # Direct file download
            if dest.exists():
                logger.info(f"File already exists: {dest}")
                continue

            if not download_file_from_url(url, dest):
                success = False
                continue

    if success:
        logger.info("All LPR resources downloaded successfully")
    else:
        logger.error("Some LPR resources failed to download")

    return success


if __name__ == "__main__":
    # For testing
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if len(sys.argv) != 3:
        print("Usage: python lpr_resources.py <models_dir> <videos_dir>")
        sys.exit(1)

    models_dir = sys.argv[1]
    videos_dir = sys.argv[2]

    success = download_lpr_resources(models_dir, videos_dir)
    sys.exit(0 if success else 1)
