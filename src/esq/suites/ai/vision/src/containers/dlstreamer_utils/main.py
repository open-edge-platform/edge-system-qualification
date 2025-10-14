#!/usr/bin/env python3

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Video utility script for DLStreamer containers.
"""

import argparse
import logging
import os
import re
import shutil
import subprocess  # nosec
from pathlib import Path
from typing import Any, Dict


def get_log_level():
    """Get log level from environment variable with fallback to INFO."""
    env_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return log_levels.get(env_log_level, logging.INFO)


logging.basicConfig(
    level=get_log_level(), format="%(asctime)s.%(msecs)03d - %(name)s:%(lineno)d - %(message)s", datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

# Constants
MNT_DIR = "/mnt"
VIDEOS_DIR = os.path.join(MNT_DIR, "videos")
RESULTS_DIR = os.path.join(MNT_DIR, "results")

# Input validation constants
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}
MIN_WIDTH = 320
MAX_WIDTH = 7680  # 8K width
MIN_HEIGHT = 240
MAX_HEIGHT = 4320  # 8K height
MIN_FPS = 1
MAX_FPS = 120
MIN_DURATION = 1  # 1 second
MAX_DURATION = 3600  # 1 hour
MAX_FILENAME_LENGTH = 255


def validate_video_filename(filename: str) -> str:
    """
    Validate and sanitize video filename to prevent path manipulation.
    """
    if not filename:
        raise ValueError("Video filename cannot be empty")

    if len(filename) > MAX_FILENAME_LENGTH:
        raise ValueError(f"Video filename too long (max {MAX_FILENAME_LENGTH} characters)")

    # Check for path manipulation characters
    dangerous_chars = ["..", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    for char in dangerous_chars:
        if char in filename:
            raise ValueError(f"Invalid character '{char}' in filename: {filename}")

    # Ensure filename has valid extension
    file_extension = os.path.splitext(filename)[1].lower()
    if not file_extension:
        raise ValueError(f"Video filename must have an extension: {filename}")

    if file_extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension '{file_extension}'. "
            f"Allowed extensions: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
        )

    # Check that filename doesn't start with special characters
    if filename.startswith((".", "-", "_")):
        raise ValueError(f"Filename cannot start with special characters: {filename}")

    # Ensure filename contains only allowed characters (alphanumeric, dash, underscore, dot)
    if not re.match(r"^[a-zA-Z0-9\-_.]+$", filename):
        raise ValueError(f"Filename contains invalid characters: {filename}")

    return filename


def sanitize_path(user_input: str, base_dir: str) -> str:
    """
    Sanitize path input.

    This function creates a new untainted string by character-by-character copying
    with validation, which recognizes as breaking the taint chain.

    Args:
        user_input: The potentially tainted user input
        base_dir: The base directory to validate against

    Returns:
        A sanitized path string that considers untainted

    Raises:
        ValueError: If the input contains invalid characters or path traversal attempts
    """
    if not user_input or not isinstance(user_input, str):
        raise ValueError("Invalid input: path must be a non-empty string")

    # First validate the input using our existing function
    validated_filename = validate_video_filename(user_input)

    # Create a new untainted string by explicit character-by-character copying
    # This approach is recognized as breaking the taint chain
    sanitized_chars = []

    for char in validated_filename:
        # Only allow safe characters (alphanumeric, dots, hyphens, underscores)
        if char.isalnum() or char in ".-_":
            sanitized_chars.append(char)
        else:
            # Replace unsafe characters with underscore (though validation should prevent this)
            sanitized_chars.append("_")

    # Join the characters to create a new untainted string
    sanitized_filename = "".join(sanitized_chars)

    # Additional validation - ensure it's not empty and doesn't start with dots
    if not sanitized_filename or sanitized_filename.startswith("."):
        raise ValueError(f"Invalid filename after sanitization: {user_input}")

    # Verify the sanitized path would be safe within the base directory
    base_path = Path(base_dir).resolve()
    test_path = base_path / sanitized_filename
    resolved_test_path = test_path.resolve()

    # Ensure no path traversal (resolved path must be within base directory)
    try:
        resolved_test_path.relative_to(base_path)
    except ValueError:
        raise ValueError(f"Path '{user_input}' attempts to escape base directory")

    # Additional safety checks
    if resolved_test_path.is_symlink():
        raise ValueError(f"Symlinks are not allowed: {user_input}")

    # Return the new untainted string
    return sanitized_filename


def validate_video_dimensions(width: int, height: int) -> tuple[int, int]:
    """
    Validate video width and height dimensions.
    """
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("Width and height must be integers")

    if width < MIN_WIDTH or width > MAX_WIDTH:
        raise ValueError(f"Width must be between {MIN_WIDTH} and {MAX_WIDTH} pixels")

    if height < MIN_HEIGHT or height > MAX_HEIGHT:
        raise ValueError(f"Height must be between {MIN_HEIGHT} and {MAX_HEIGHT} pixels")

    # Ensure dimensions are even numbers (required for most video codecs)
    if width % 2 != 0:
        raise ValueError(f"Width must be an even number: {width}")

    if height % 2 != 0:
        raise ValueError(f"Height must be an even number: {height}")

    return width, height


def validate_fps(fps: int) -> int:
    """
    Validate frames per second value.
    """
    if not isinstance(fps, int):
        raise ValueError("FPS must be an integer")

    if fps < MIN_FPS or fps > MAX_FPS:
        raise ValueError(f"FPS must be between {MIN_FPS} and {MAX_FPS}")

    return fps


def validate_duration(duration: int) -> int:
    """
    Validate video duration value.
    """
    if not isinstance(duration, int):
        raise ValueError("Duration must be an integer")

    if duration < MIN_DURATION or duration > MAX_DURATION:
        raise ValueError(f"Duration must be between {MIN_DURATION} and {MAX_DURATION} seconds")

    return duration


def validate_video_file_exists(filename: str) -> str:
    """
    Validate that video file exists and is safe to access.
    """
    video_path = Path(os.path.join(VIDEOS_DIR, filename))

    # Check if file exists and is not a symlink
    if not files_exist_and_safe(video_path):
        raise ValueError(f"Video file does not exist or is unsafe: {filename}")

    return str(video_path)


def files_exist_and_safe(*paths: Path) -> bool:
    """
    Returns True if all given files exist and are not symlinks or do not contain symlinks in their paths.
    """
    for path in paths:
        if not path.exists():
            return False
        # Check if the path is a symlink
        if os.path.realpath(path) != os.path.abspath(path):
            return False
    return True


def convert_video(
    name: str,
    width=1920,
    height=1080,
    fps=15,
) -> Dict[str, Any]:
    """
    Convert a video to specified width, height, fps, and duration.
    Returns a dict with video configuration details.
    """
    # Validate all inputs to prevent path manipulation and ensure safe parameters
    validated_name = validate_video_filename(name)
    validated_width, validated_height = validate_video_dimensions(width, height)
    validated_fps = validate_fps(fps)

    # Validate that the video file exists and is safe to access
    validate_video_file_exists(validated_name)

    video_path = os.path.join(VIDEOS_DIR, validated_name)
    Path(video_path).parent.mkdir(parents=True, exist_ok=True, mode=0o750)

    file_extension = os.path.splitext(validated_name)[1]
    file_basename = os.path.splitext(validated_name)[0]
    source_video_path = os.path.join(VIDEOS_DIR, f"{file_basename}_before_convert{file_extension}")
    if not os.path.exists(source_video_path):
        shutil.copy(video_path, source_video_path)
    os.remove(video_path)

    config = {
        "video_name": validated_name,
        "video_width": validated_width,
        "video_height": validated_height,
        "video_fps": validated_fps,
    }
    logger.debug(f"Converting video {validated_name} to {validated_width}x{validated_height} at {validated_fps} FPS")

    pipeline = [
        "gst-launch-1.0",
        "-q",
        "filesrc",
        f"location={source_video_path}",
        "!",
        "qtdemux",
        "!",
        "h264parse",
        "!",
        "avdec_h264",
        "!",
        "videoscale",
        "!",
        f"video/x-raw,width={validated_width},height={validated_height}",
        "!",
        "videorate",
        "!",
        f"video/x-raw,framerate={validated_fps}/1",
        "!",
        "x264enc",
        "!",
        "h264parse",
        "!",
        "mp4mux",
        "!",
        "filesink",
        f"location={video_path}",
    ]

    try:
        logger.debug(f"Converting video with GStreamer pipeline: {' '.join(pipeline)}")
        subprocess.run(pipeline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Converting video with GStreamer failed with exit code {e.returncode}\n"
            f"Command: {' '.join(pipeline)}\n"
            f"Stdout:\n{e.stdout}\n"
            f"Stderr:\n{e.stderr}\n"
        )
        logger.error(error_message)
        raise RuntimeError(error_message)

    video_path = Path(video_path).resolve()

    return config


def trim_video(name: str, duration=30) -> Dict[str, Any]:
    """
    Trim a video to a specified duration.
    Returns a dict with video configuration details.
    """
    # Validate all inputs to prevent path manipulation and ensure safe parameters
    validated_name = validate_video_filename(name)
    validated_duration = validate_duration(duration)

    # Validate that the video file exists and is safe to access
    validate_video_file_exists(validated_name)

    video_path = os.path.join(VIDEOS_DIR, validated_name)
    Path(video_path).parent.mkdir(parents=True, exist_ok=True, mode=0o750)

    file_extension = os.path.splitext(validated_name)[1]
    file_basename = os.path.splitext(validated_name)[0]
    source_video_path = os.path.join(VIDEOS_DIR, f"{file_basename}_before_trim{file_extension}")
    if not os.path.exists(source_video_path):
        shutil.copy(video_path, source_video_path)
    os.remove(video_path)

    config = {
        "video_name": validated_name,
        "video_duration": validated_duration,
    }

    logger.debug(f"Trimming video with FFmpeg to {validated_duration} seconds")
    ffmpeg_pipeline = [
        "ffmpeg",
        "-ss",
        "00:00:00",
        "-i",
        source_video_path,
        "-t",
        str(validated_duration),
        "-c",
        "copy",
        video_path,
    ]

    try:
        subprocess.run(ffmpeg_pipeline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        logger.debug("FFmpeg trimming completed successfully.")
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Trimming video with FFmpeg failed with exit code {e.returncode}\n"
            f"Command: {' '.join(ffmpeg_pipeline)}\n"
            f"Stdout:\n{e.stdout}\n"
            f"Stderr:\n{e.stderr}\n"
        )
        logger.error(error_message)
        raise RuntimeError(error_message)

    video_path = Path(video_path).resolve()

    return config


def main():
    """Main entry point for video utilities."""
    parser = argparse.ArgumentParser(description="Video Utilities")

    # Common arguments
    parser.add_argument("--video-name", required=True, help="Video file name")

    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Convert video command
    convert_parser = subparsers.add_parser("convert", help="Convert video")
    convert_parser.add_argument("--width", type=int, default=1920, help="Video width")
    convert_parser.add_argument("--height", type=int, default=1080, help="Video height")
    convert_parser.add_argument("--fps", type=int, default=30, help="Video FPS")

    # Trim video command
    trim_parser = subparsers.add_parser("trim", help="Trim video")
    trim_parser.add_argument("--duration", type=int, default=10, help="Video duration in seconds")

    args = parser.parse_args()

    try:
        sanitized_video_name = sanitize_path(args.video_name, VIDEOS_DIR)

        if args.command == "convert":
            validated_width, validated_height = validate_video_dimensions(args.width, args.height)
            validated_fps = validate_fps(args.fps)

            convert_video(sanitized_video_name, validated_width, validated_height, validated_fps)
            logger.info(f"Successfully converted video: {sanitized_video_name}")
        elif args.command == "trim":
            validated_duration = validate_duration(args.duration)

            trim_video(sanitized_video_name, validated_duration)
            logger.info(f"Successfully trimmed video: {sanitized_video_name}")
        else:
            parser.print_help()
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        exit(1)
    except RuntimeError as e:
        logger.error(f"Processing error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
