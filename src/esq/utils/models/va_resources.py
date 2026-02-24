# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Video Analytics (VA) Pipeline Resource Downloader.

This module handles downloading VA models and video files required for
VA video analytics benchmarks (light/medium/heavy).

Uses ESQ's model download utilities (yolo_model_utils, openvino_model_utils)
for programmatic model downloads, following the pattern from lpr_resources.py.

VA Pipeline Configurations:
- Light: YOLOv11n (detection) + ResNet-50 (classification), H264 video
- Medium: YOLOv5m (detection) + ResNet-50 + MobileNet-v2 (dual classification), H265 video
- Heavy: Future - larger models with additional stages

VA Pipeline: Multi-stage video analytics pipeline with:
- Decode stage: H264/H265 video decoding
- Detection stage: YOLO object detection
- Tracking stage: Object tracking (short-term-imageless)
- Classification stage: ResNet-50/MobileNet-v2 classification
"""

import logging
from pathlib import Path
from typing import Dict

from esq.utils.genutils import download_file_from_url
from esq.utils.models.openvino_model_utils import download_openvino_model
from esq.utils.models.yolo_model_utils import download_yolo_model, export_yolo_model

logger = logging.getLogger(__name__)

# VA Light pipeline video URL - H.265 encoded bears video
VIDEO_LIGHT_URL = "https://videos.pexels.com/video-files/18856748/18856748-uhd_3840_2160_60fps.mp4"

# VA Medium pipeline video - H.265 encoded bears video
VIDEO_MEDIUM_URL = "https://videos.pexels.com/video-files/18856748/18856748-uhd_3840_2160_60fps.mp4"


def download_va_resources(models_dir: str, videos_dir: str) -> Dict[str, Path]:
    """
    Download VA Light pipeline models and videos using ESQ utilities.

    Downloads:
    - YOLOv11n detection model (640x640, INT8) via yolo_model_utils
    - ResNet-50 classification model (INT8) via openvino_model_utils
    - 18856748-bears_1920_1080_30fps_30s.h265 video

    Args:
        models_dir: Directory to save models (e.g., esq_data/data/ai/vision/models/va/light)
        videos_dir: Directory to save videos (e.g., esq_data/data/ai/vision/videos/va/light)

    Returns:
        Dictionary with paths to downloaded resources
    """
    # Create light subdirectory structure
    models_path = Path(models_dir) / "light"
    videos_path = Path(videos_dir) / "light"

    logger.info("Downloading VA Light pipeline resources...")

    results = {}

    # ========== Download YOLOv11n Detection Model ==========
    # YOLOv11n is the latest YOLO version with improved performance
    detection_dir = models_path / "detection" / "yolov11n" / "int8"
    detection_dir.mkdir(parents=True, exist_ok=True)

    # export_yolo_model creates: {models_dir}/{model_id}/{PRECISION}/{model_id}.xml
    yolo_xml_expected = detection_dir / "yolov11n" / "INT8" / "yolov11n.xml"

    if yolo_xml_expected.exists():
        logger.info(f"YOLOv11n model already exists: {yolo_xml_expected}")
        results["detection_xml"] = yolo_xml_expected
        results["detection_bin"] = yolo_xml_expected.with_suffix(".bin")
    else:
        try:
            logger.info("Downloading YOLOv11n detection model...")
            # Download .pt weights
            pt_path = download_yolo_model("yolov11n", str(models_path.parent.parent))

            logger.info("Exporting YOLOv11n to OpenVINO INT8 format with dynamic batch support...")
            # Export to OpenVINO with INT8 precision and dynamic batch
            # dynamic=True allows variable batch size, half=True for FP16 precision
            yolo_xml_result = export_yolo_model(
                model_id="yolov11n",
                models_dir=str(detection_dir),
                model_precision="int8",
                weights_path=pt_path,
                export_args={"dynamic": True, "half": True, "imgsz": 640},
            )

            if not yolo_xml_result:
                raise FileNotFoundError("YOLOv11n export failed - no path returned")

            # Convert to Path object
            yolo_xml = Path(yolo_xml_result) if isinstance(yolo_xml_result, str) else yolo_xml_result

            if not yolo_xml.exists():
                raise FileNotFoundError(f"YOLOv11n export failed - XML not found at {yolo_xml}")

            logger.info(f"YOLOv11n model exported: {yolo_xml}")
            results["detection_xml"] = yolo_xml
            results["detection_bin"] = yolo_xml.with_suffix(".bin")

        except Exception as e:
            logger.error(f"Failed to download YOLOv11n model: {e}", exc_info=True)
            raise

    # ========== Download ResNet-50 Classification Model ==========
    try:
        logger.info("Downloading ResNet-50 classification model...")
        resnet_result = download_openvino_model(
            model_id="resnet-50-tf", precision="INT8", models_dir=str(models_path / "classification")
        )

        if not resnet_result:
            raise FileNotFoundError("ResNet-50 download failed - no path returned")

        # Convert to Path object
        resnet_xml = Path(resnet_result) if isinstance(resnet_result, str) else resnet_result

        if not resnet_xml.exists():
            raise FileNotFoundError(f"ResNet-50 download failed - XML not found at {resnet_xml}")

        logger.info(f"ResNet-50 model ready: {resnet_xml}")
        results["classification_xml"] = resnet_xml
        results["classification_bin"] = resnet_xml.with_suffix(".bin")

    except Exception as e:
        logger.error(f"Failed to download ResNet-50 model: {e}", exc_info=True)
        raise

    # ========== Download Video ==========
    # Create video subdirectory: videos/light/video/
    video_dir = videos_path / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "18856748-bears_1920_1080_30fps_30s.h265"

    if video_path.exists():
        logger.info(f"Video already exists: {video_path}")
    else:
        logger.info(f"Downloading VA Light video: {VIDEO_LIGHT_URL}")
        # download_file_from_url expects a Path object, not a string
        if not download_file_from_url(VIDEO_LIGHT_URL, video_path):
            raise RuntimeError(f"Failed to download video from {VIDEO_LIGHT_URL}")
        logger.info(f"Video downloaded: {video_path}")

    results["video"] = video_path

    logger.info("All VA Light resources downloaded successfully")
    return results


def download_va_medium_resources(models_dir: str, videos_dir: str) -> Dict[str, Path]:
    """
    Download VA Medium pipeline models and videos using ESQ utilities.

    Medium pipeline is more complex than Light with:
    - YOLOv5m detection model (larger than YOLO11n)
    - Dual classification: ResNet-50 + MobileNet-v2
    - H265 encoded video (more compression than H264)

    Downloads:
    - YOLOv5m detection model (640x640, INT8) via yolo_model_utils
    - ResNet-50 classification model (INT8) via openvino_model_utils
    - MobileNet-v2 classification model (INT8) via openvino_model_utils
    - apple.h265 video (bears footage, 1920x1080@30fps)

    Args:
        models_dir: Directory to save models (e.g., esq_data/data/ai/vision/models/va/medium)
        videos_dir: Directory to save videos (e.g., esq_data/data/ai/vision/videos/va/medium)

    Returns:
        Dictionary with paths to downloaded resources
    """
    # Create medium subdirectory structure
    models_path = Path(models_dir) / "medium"
    videos_path = Path(videos_dir) / "medium"

    logger.info("Downloading VA Medium pipeline resources...")

    results = {}

    # ========== Download YOLOv5m Detection Model ==========
    detection_dir = models_path / "detection" / "yolov5m_640x640" / "INT8"
    detection_dir.mkdir(parents=True, exist_ok=True)

    # export_yolo_model creates: {models_dir}/{model_id}/{PRECISION}/{model_id}.xml
    yolo_xml_expected = detection_dir / "yolov5m" / "INT8" / "yolov5m.xml"

    if yolo_xml_expected.exists():
        logger.info(f"YOLOv5m model already exists: {yolo_xml_expected}")
        results["detection_xml"] = yolo_xml_expected
        results["detection_bin"] = yolo_xml_expected.with_suffix(".bin")
    else:
        try:
            logger.info("Downloading YOLOv5m detection model...")
            # Download .pt weights
            pt_path = download_yolo_model("yolov5m", str(models_path.parent.parent))

            logger.info("Exporting YOLOv5m to OpenVINO INT8 format with dynamic batch support...")
            # Export to OpenVINO with INT8 precision and dynamic batch
            yolo_xml_result = export_yolo_model(
                model_id="yolov5m",
                models_dir=str(detection_dir),
                model_precision="int8",
                weights_path=pt_path,
                export_args={"dynamic": True, "half": True, "imgsz": 640},
            )

            if not yolo_xml_result:
                raise FileNotFoundError("YOLOv5m export failed - no path returned")

            yolo_xml = Path(yolo_xml_result) if isinstance(yolo_xml_result, str) else yolo_xml_result

            if not yolo_xml.exists():
                raise FileNotFoundError(f"YOLOv5m export failed - XML not found at {yolo_xml}")

            logger.info(f"YOLOv5m model exported: {yolo_xml}")
            results["detection_xml"] = yolo_xml
            results["detection_bin"] = yolo_xml.with_suffix(".bin")

        except Exception as e:
            logger.error(f"Failed to download YOLOv5m model: {e}", exc_info=True)
            raise

    # ========== Download ResNet-50 Classification Model ==========
    try:
        logger.info("Downloading ResNet-50 classification model...")
        resnet_result = download_openvino_model(
            model_id="resnet-50-tf", precision="INT8", models_dir=str(models_path / "classification")
        )

        if not resnet_result:
            raise FileNotFoundError("ResNet-50 download failed - no path returned")

        resnet_xml = Path(resnet_result) if isinstance(resnet_result, str) else resnet_result

        if not resnet_xml.exists():
            raise FileNotFoundError(f"ResNet-50 download failed - XML not found at {resnet_xml}")

        logger.info(f"ResNet-50 model ready: {resnet_xml}")
        results["classification_resnet_xml"] = resnet_xml
        results["classification_resnet_bin"] = resnet_xml.with_suffix(".bin")

    except Exception as e:
        logger.error(f"Failed to download ResNet-50 model: {e}", exc_info=True)
        raise

    # ========== Download MobileNet-v2 Classification Model ==========
    try:
        logger.info("Downloading MobileNet-v2 classification model...")
        mobilenet_result = download_openvino_model(
            model_id="mobilenet-v2-pytorch", precision="INT8", models_dir=str(models_path / "classification")
        )

        if not mobilenet_result:
            raise FileNotFoundError("MobileNet-v2 download failed - no path returned")

        mobilenet_xml = Path(mobilenet_result) if isinstance(mobilenet_result, str) else mobilenet_result

        if not mobilenet_xml.exists():
            raise FileNotFoundError(f"MobileNet-v2 download failed - XML not found at {mobilenet_xml}")

        logger.info(f"MobileNet-v2 model ready: {mobilenet_xml}")
        results["classification_mobilenet_xml"] = mobilenet_xml
        results["classification_mobilenet_bin"] = mobilenet_xml.with_suffix(".bin")

    except Exception as e:
        logger.error(f"Failed to download MobileNet-v2 model: {e}", exc_info=True)
        raise

    # ========== Download Video ==========
    # Create video subdirectory: videos/medium/video/
    video_dir = videos_path / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "apple.h265"

    if video_path.exists():
        logger.info(f"Video already exists: {video_path}")
    else:
        logger.info(f"Downloading VA Medium video: {VIDEO_MEDIUM_URL}")
        if not download_file_from_url(VIDEO_MEDIUM_URL, video_path):
            raise RuntimeError(f"Failed to download video from {VIDEO_MEDIUM_URL}")
        logger.info(f"Video downloaded: {video_path}")

    results["video"] = video_path

    logger.info("All VA Medium resources downloaded successfully")
    return results


# VA Heavy pipeline video URL - H.265 encoded bears video
VIDEO_HEAVY_URL = "https://videos.pexels.com/video-files/18856748/18856748-uhd_3840_2160_60fps.mp4"


def download_va_heavy_resources(models_dir: str, videos_dir: str) -> Dict[str, Path]:
    """
    Download VA Heavy pipeline models and videos using ESQ utilities.

    Heavy pipeline is the most demanding with:
    - YOLO11m detection model (medium YOLO11, provides good accuracy vs speed)
    - Dual classification: ResNet-50-tf + MobileNet-v2-pytorch
    - H264 encoded video

    Downloads:
    - YOLO11m detection model (640x640, INT8) via yolo_model_utils
    - ResNet-50-tf classification model (INT8) via openvino_model_utils
    - MobileNet-v2-pytorch classification model (INT8) via openvino_model_utils
    - 18856748-bears_1920_1080_30fps_30s.h265 video

    Args:
        models_dir: Directory to save models (e.g., esq_data/data/ai/vision/models/va/heavy)
        videos_dir: Directory to save videos (e.g., esq_data/data/ai/vision/videos/va/heavy)

    Returns:
        Dictionary with paths to downloaded resources
    """
    # Create heavy subdirectory structure
    models_path = Path(models_dir) / "heavy"
    videos_path = Path(videos_dir) / "heavy"

    logger.info("Downloading VA Heavy pipeline resources...")

    results = {}

    # ========== Download YOLO11m Detection Model ==========
    # Using YOLO11m with yolo-v8.json post-processor (compatible with YOLOv11)
    detection_dir = models_path / "detection" / "yolo11m_640x640" / "INT8"
    detection_dir.mkdir(parents=True, exist_ok=True)

    # export_yolo_model creates: {models_dir}/{model_id}/{PRECISION}/{model_id}.xml
    yolo_xml_expected = detection_dir / "yolo11m" / "INT8" / "yolo11m.xml"

    if yolo_xml_expected.exists():
        logger.info(f"YOLO11m model already exists: {yolo_xml_expected}")
        results["detection_xml"] = yolo_xml_expected
        results["detection_bin"] = yolo_xml_expected.with_suffix(".bin")
    else:
        try:
            logger.info("Downloading YOLO11m detection model...")
            # Download .pt weights using "yolo11m" identifier (Ultralytics naming without 'v')
            pt_path = download_yolo_model("yolo11m", str(models_path.parent.parent))

            logger.info("Exporting YOLO11m to OpenVINO INT8 format with dynamic batch support...")
            # Export to OpenVINO with INT8 precision and dynamic batch
            yolo_xml_result = export_yolo_model(
                model_id="yolo11m",
                models_dir=str(detection_dir),
                model_precision="int8",
                weights_path=pt_path,
                export_args={"dynamic": True, "half": True, "imgsz": 640},
            )

            if not yolo_xml_result:
                raise FileNotFoundError("YOLO11m export failed - no path returned")

            yolo_xml = Path(yolo_xml_result) if isinstance(yolo_xml_result, str) else yolo_xml_result

            if not yolo_xml.exists():
                raise FileNotFoundError(f"YOLO11m export failed - XML not found at {yolo_xml}")

            logger.info(f"YOLO11m model exported: {yolo_xml}")
            results["detection_xml"] = yolo_xml
            results["detection_bin"] = yolo_xml.with_suffix(".bin")

        except Exception as e:
            logger.error(f"Failed to download YOLO11m model: {e}", exc_info=True)
            raise

    # ========== Download ResNet-50 Classification Model ==========
    # Note: Using resnet-50-tf (same as medium) since resnet-v1-50-tf is not available
    try:
        logger.info("Downloading ResNet-50 classification model for heavy pipeline...")
        resnet_result = download_openvino_model(
            model_id="resnet-50-tf", precision="INT8", models_dir=str(models_path / "classification")
        )

        if not resnet_result:
            raise FileNotFoundError("ResNet-50 download failed - no path returned")

        resnet_xml = Path(resnet_result) if isinstance(resnet_result, str) else resnet_result

        if not resnet_xml.exists():
            raise FileNotFoundError(f"ResNet-50 download failed - XML not found at {resnet_xml}")

        logger.info(f"ResNet-50 model ready: {resnet_xml}")
        results["resnet_xml"] = resnet_xml
        results["resnet_bin"] = resnet_xml.with_suffix(".bin")

    except Exception as e:
        logger.error(f"Failed to download ResNet-50 model: {e}", exc_info=True)
        raise

    # ========== Download MobileNet-v2 Classification Model ==========
    # Note: Using mobilenet-v2-pytorch (same as medium) since mobilenet-v2-1.0-224-tf is not available
    try:
        logger.info("Downloading MobileNet-v2 classification model for heavy pipeline...")
        mobilenet_result = download_openvino_model(
            model_id="mobilenet-v2-pytorch", precision="INT8", models_dir=str(models_path / "classification")
        )

        if not mobilenet_result:
            raise FileNotFoundError("MobileNet-v2 download failed - no path returned")

        mobilenet_xml = Path(mobilenet_result) if isinstance(mobilenet_result, str) else mobilenet_result

        if not mobilenet_xml.exists():
            raise FileNotFoundError(f"MobileNet-v2 download failed - XML not found at {mobilenet_xml}")

        logger.info(f"MobileNet-v2 model ready: {mobilenet_xml}")
        results["mobilenet_xml"] = mobilenet_xml
        results["mobilenet_bin"] = mobilenet_xml.with_suffix(".bin")

    except Exception as e:
        logger.error(f"Failed to download MobileNet-v2-1.0-224 model: {e}", exc_info=True)
        raise

    # ========== Download Video ==========
    # Create video subdirectory: videos/heavy/video/
    video_dir = videos_path / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "18856748-bears_1920_1080_30fps_30s.h265"

    if video_path.exists():
        logger.info(f"Video already exists: {video_path}")
    else:
        logger.info(f"Downloading VA Heavy video: {VIDEO_HEAVY_URL}")
        if not download_file_from_url(VIDEO_HEAVY_URL, video_path):
            raise RuntimeError(f"Failed to download video from {VIDEO_HEAVY_URL}")
        logger.info(f"Video downloaded: {video_path}")

    results["video"] = video_path

    logger.info("All VA Heavy resources downloaded successfully")
    return results


if __name__ == "__main__":
    # For testing
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if len(sys.argv) != 3:
        print("Usage: python va_resources.py <models_dir> <videos_dir>")
        sys.exit(1)

    models_dir = sys.argv[1]
    videos_dir = sys.argv[2]

    try:
        results = download_va_resources(models_dir, videos_dir)
        print("\nDownloaded resources:")
        for key, path in results.items():
            print(f"  {key}: {path}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to download resources: {e}")
        sys.exit(1)
