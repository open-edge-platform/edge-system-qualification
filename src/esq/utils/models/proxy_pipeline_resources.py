# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Proxy Pipeline Resource Downloader.

This module handles downloading models and video files required for
proxy pipeline benchmarks (Smart NVR, Headed Visual AI, VSaaS).
"""

import logging
import tarfile
from pathlib import Path

from esq.utils.genutils import download_file_from_url

logger = logging.getLogger(__name__)


def download_proxy_pipeline_resources(models_dir: str, videos_dir: str) -> bool:
    """
    Download proxy pipeline models and videos.

    Models are downloaded from pipeline-zoo-models repository.
    Videos are downloaded from edge-ai-resources repository.

    Args:
        models_dir: Directory to save models (will create subdirectories)
        videos_dir: Directory to save videos

    Returns:
        bool: True if all downloads successful
    """
    models_path = Path(models_dir)
    videos_path = Path(videos_dir)

    # Ensure directories exist
    models_path.mkdir(parents=True, exist_ok=True)
    videos_path.mkdir(parents=True, exist_ok=True)

    success = True

    # Step 1: Download and extract pipeline-zoo-models
    logger.info("Downloading pipeline-zoo-models...")

    pipeline_zoo_url = "https://github.com/dlstreamer/pipeline-zoo-models/archive/refs/tags/v0.0.9.tar.gz"
    pipeline_zoo_tar = models_path.parent / "pipeline-zoo-models.tar.gz"
    pipeline_zoo_dir = models_path.parent / "pipeline-zoo-models"

    # Check if models already exist
    required_models = ["yolov5s-416_INT8", "yolov5m-416_INT8", "efficientnet-b0_INT8"]
    all_models_exist = all((models_path / model).exists() for model in required_models)

    if all_models_exist:
        logger.info("All required models already exist, skipping download")
    else:
        # Download tar.gz
        if not download_file_from_url(pipeline_zoo_url, pipeline_zoo_tar):
            logger.error("Failed to download pipeline-zoo-models")
            success = False
        else:
            # Extract tar.gz
            try:
                logger.info(f"Extracting pipeline-zoo-models to {pipeline_zoo_dir}...")
                with tarfile.open(pipeline_zoo_tar, "r:gz") as tar:
                    # Extract to parent directory
                    tar.extractall(path=models_path.parent)

                # Rename extracted directory (removes version suffix)
                extracted_dir = models_path.parent / "pipeline-zoo-models-0.0.9"
                if extracted_dir.exists():
                    extracted_dir.rename(pipeline_zoo_dir)

                # Move required models from storage/ to models_dir
                storage_dir = pipeline_zoo_dir / "storage"
                if storage_dir.exists():
                    logger.info("Moving required models to target directory...")
                    for model_name in required_models:
                        src = storage_dir / model_name
                        dst = models_path / model_name
                        if src.exists():
                            if dst.exists():
                                logger.info(f"Model already exists: {model_name}")
                            else:
                                # Use shutil.move for cross-filesystem compatibility
                                import shutil

                                shutil.move(str(src), str(dst))
                                logger.info(f"Moved model: {model_name}")
                        else:
                            logger.warning(f"Model not found in archive: {model_name}")

                    # Clean up extracted directory
                    import shutil

                    shutil.rmtree(pipeline_zoo_dir, ignore_errors=True)
                    logger.info("Cleaned up temporary extraction directory")
                else:
                    logger.error("Storage directory not found in extracted archive")
                    success = False

                # Clean up tar file
                pipeline_zoo_tar.unlink(missing_ok=True)

            except Exception as e:
                logger.error(f"Failed to extract pipeline-zoo-models: {e}")
                success = False

    # Step 2: Download video files from edge-ai-resources repository
    logger.info("Downloading video files from GitHub...")

    # Define required video files with their GitHub URLs
    # Using specific commit hash (b219993) for reproducibility - videos added on Dec 15, 2025
    # Commit: "Added 10 cars-in-a-cycle benchmarking videos (#18)"
    video_files = {
        "car_1080p20_10s_h264.mp4": "https://github.com/open-edge-platform/edge-ai-resources/raw/b219993299792993944ee9a6892f0e1bf3e4f4b0/videos/car_1080p20_10s_h264.mp4",
        "car_1080p30_10s_h264.mp4": "https://github.com/open-edge-platform/edge-ai-resources/raw/b219993299792993944ee9a6892f0e1bf3e4f4b0/videos/car_1080p30_10s_h264.mp4",
    }

    # Download each video file
    for video_name, video_url in video_files.items():
        dest_video = videos_path / video_name

        if dest_video.exists():
            logger.info(f"Video already exists: {video_name}")
            continue

        logger.info(f"Downloading {video_name} from {video_url}")
        if not download_file_from_url(video_url, dest_video):
            logger.error(f"Failed to download video: {video_name}")
            success = False
        else:
            logger.info(f"Successfully downloaded: {video_name}")

    if success:
        logger.info("All proxy pipeline resources prepared successfully")
    else:
        logger.error("Some proxy pipeline resources failed to prepare")

    return success


if __name__ == "__main__":
    # For testing
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if len(sys.argv) != 3:
        print("Usage: python proxy_pipeline_resources.py <models_dir> <videos_dir>")
        sys.exit(1)

    models_dir = sys.argv[1]
    videos_dir = sys.argv[2]

    success = download_proxy_pipeline_resources(models_dir, videos_dir)
    sys.exit(0 if success else 1)
