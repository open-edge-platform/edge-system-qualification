# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Media Benchmark Resource Downloader.

This module handles downloading video files required for media benchmarks.
Videos are downloaded from the edge-ai-resources GitHub repository.
"""

import logging
from pathlib import Path

from esq.utils.genutils import download_file_from_url

logger = logging.getLogger(__name__)


def download_media_resources(videos_dir: str) -> bool:
    """
    Download media benchmark video files from GitHub.

    Videos are downloaded from edge-ai-resources repository:
    https://github.com/open-edge-platform/edge-ai-resources

    Args:
        videos_dir: Directory to save videos

    Returns:
        bool: True if all downloads successful
    """
    videos_path = Path(videos_dir)

    # Ensure directory exists
    videos_path.mkdir(parents=True, exist_ok=True)

    success = True

    # Download video files from edge-ai-resources repository
    logger.info("Downloading media benchmark video files from GitHub...")

    # Define required video files with their GitHub URLs
    # Using specific commit hash (b219993) for reproducibility - videos added on Dec 15, 2025
    # Commit: "Added 10 cars-in-a-cycle benchmarking videos (#18)"
    video_files = {
        "car_1080p30_10s_h264.mp4": "https://github.com/open-edge-platform/edge-ai-resources/raw/b219993299792993944ee9a6892f0e1bf3e4f4b0/videos/car_1080p30_10s_h264.mp4",
        "car_1080p30_10s_h265.mp4": "https://github.com/open-edge-platform/edge-ai-resources/raw/b219993299792993944ee9a6892f0e1bf3e4f4b0/videos/car_1080p30_10s_h265.mp4",
        "car_4K30_10s_h264.mp4": "https://github.com/open-edge-platform/edge-ai-resources/raw/b219993299792993944ee9a6892f0e1bf3e4f4b0/videos/car_4K30_10s_h264.mp4",
        "car_4K30_10s_h265.mp4": "https://github.com/open-edge-platform/edge-ai-resources/raw/b219993299792993944ee9a6892f0e1bf3e4f4b0/videos/car_4K30_10s_h265.mp4",
    }

    # Download each video file
    for video_name, video_url in video_files.items():
        dest_video = videos_path / video_name

        if dest_video.exists():
            logger.info(f"Video already exists: {video_name}")
            continue

        logger.info(f"Downloading {video_name} from GitHub...")
        if not download_file_from_url(video_url, dest_video):
            logger.error(f"Failed to download video: {video_name}")
            success = False
        else:
            logger.info(f"Successfully downloaded: {video_name}")

    if success:
        logger.info("All media benchmark resources prepared successfully")
    else:
        logger.error("Some media benchmark resources failed to prepare")

    return success


if __name__ == "__main__":
    # For testing
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if len(sys.argv) != 2:
        print("Usage: python media_resources.py <videos_dir>")
        sys.exit(1)

    videos_dir = sys.argv[1]

    success = download_media_resources(videos_dir)
    sys.exit(0 if success else 1)
