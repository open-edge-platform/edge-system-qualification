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
import subprocess  # nosec B404 # Subprocess needed for video processing with ffmpeg
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
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".h264", ".h265"}
ALLOWED_CODECS = {"h264", "h265"}
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
        raise ValueError(f"Duration must be an integer, got {type(duration).__name__}")

    if duration < MIN_DURATION or duration > MAX_DURATION:
        raise ValueError(f"Duration must be between {MIN_DURATION} and {MAX_DURATION} seconds")

    return duration


def validate_codec(codec: str) -> str:
    """
    Validate video codec value.
    """
    if not isinstance(codec, str):
        raise ValueError(f"Codec must be a string, got {type(codec).__name__}")

    codec_lower = codec.lower()
    if codec_lower not in ALLOWED_CODECS:
        raise ValueError(f"Codec must be one of {ALLOWED_CODECS}, got '{codec}'")

    return codec_lower


def validate_video_file_exists(filename: str) -> str:
    """
    Validate that video file exists and is safe to access.
    """
    video_path = Path(os.path.join(VIDEOS_DIR, filename))

    # Check if file exists and is not a symlink
    if not files_exist_and_safe(video_path):
        raise ValueError(f"Video file does not exist or is unsafe: {filename}")

    return str(video_path)


def detect_video_codec(video_path: str) -> str:
    """
    Detect the codec of a video file using ffprobe.
    Returns 'h264' or 'h265', or 'unknown' if unable to detect.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        codec = result.stdout.strip().lower()

        # Map common codec names to our standard names
        if codec in ["h264", "avc", "avc1"]:
            return "h264"
        elif codec in ["h265", "hevc", "hev1"]:
            return "h265"
        else:
            logger.warning(f"Unknown codec detected: {codec}, defaulting to h264")
            return "h264"
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to detect codec with ffprobe: {e}, defaulting to h264")
        return "h264"


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
    width=None,
    height=None,
    fps=None,
    codec=None,
    keep_original=False,
) -> Dict[str, Any]:
    """
    Convert a video with specified transformations.
    Only applies transformations for parameters that are explicitly provided.
    Supports H.264 and H.265 (HEVC) codecs.
    Returns a dict with video configuration details.

    Args:
        name: Video filename
        width: Target width (optional - if not provided, preserves original)
        height: Target height (optional - if not provided, preserves original)
        fps: Target FPS (optional - if not provided, preserves original)
        codec: Target codec (optional - h264 or h265, if not provided, preserves original)
        keep_original: If True, move original to debug/ subfolder; if False, delete it (default)
    """
    # Validate all inputs to prevent path manipulation and ensure safe parameters
    validated_name = validate_video_filename(name)

    # Validate provided parameters only
    validated_width = None
    validated_height = None
    validated_fps = None
    validated_codec = None

    if width is not None and height is not None:
        validated_width, validated_height = validate_video_dimensions(width, height)
    elif width is not None or height is not None:
        raise ValueError("Both width and height must be specified together, or neither")

    if fps is not None:
        validated_fps = validate_fps(fps)

    if codec is not None:
        validated_codec = validate_codec(codec)

    # Validate that the video file exists and is safe to access
    validate_video_file_exists(validated_name)

    video_path = os.path.join(VIDEOS_DIR, validated_name)
    Path(video_path).parent.mkdir(parents=True, exist_ok=True, mode=0o750)

    file_extension = os.path.splitext(validated_name)[1]
    file_basename = os.path.splitext(validated_name)[0]

    # Determine output extension based on target codec
    if validated_codec == "h265":
        # Use .h265 extension for raw H.265 elementary stream
        output_extension = ".h265"
    elif validated_codec == "h264" or (validated_codec is None and validated_fps is None and validated_width is None):
        # Keep original extension if no conversion
        output_extension = file_extension
    else:
        # Default to .mp4 for H.264 conversion
        output_extension = ".mp4"

    # Update video_path with correct extension
    video_path = os.path.join(VIDEOS_DIR, f"{file_basename}{output_extension}")

    source_video_path = os.path.join(VIDEOS_DIR, f"{file_basename}_before_convert{file_extension}")
    if not os.path.exists(source_video_path):
        shutil.copy(os.path.join(VIDEOS_DIR, validated_name), source_video_path)

    # Remove original file
    original_path = os.path.join(VIDEOS_DIR, validated_name)
    if os.path.exists(original_path):
        os.remove(original_path)

    config = {
        "video_name": validated_name,
    }
    if validated_width is not None:
        config["video_width"] = validated_width
    if validated_height is not None:
        config["video_height"] = validated_height
    if validated_fps is not None:
        config["video_fps"] = validated_fps
    if validated_codec is not None:
        config["video_codec"] = validated_codec

    # Build description of transformations
    transformations = []
    if validated_width and validated_height:
        transformations.append(f"{validated_width}x{validated_height}")
    if validated_fps:
        transformations.append(f"{validated_fps} FPS")
    if validated_codec:
        transformations.append(f"{validated_codec.upper()} codec")

    logger.debug(f"Converting video {validated_name} with transformations: {', '.join(transformations)}")

    # Determine target codec
    if validated_codec is None:
        source_codec = detect_video_codec(source_video_path)
        logger.debug(f"No codec conversion specified, preserving source codec: {source_codec}")
        target_codec = source_codec
    else:
        target_codec = validated_codec

    # Build dynamic GStreamer pipeline based on specified parameters and target codec
    pipeline = [
        "gst-launch-1.0",
        "-q",
        "filesrc",
        f"location={source_video_path}",
        "!",
    ]

    if target_codec == "h265":
        # H.265 pipeline using software encoding (x265enc)
        logger.info("Using software-based H.265 encoding (x265enc)")
        pipeline.extend(["decodebin", "!"])

        # Always need videoconvert for format compatibility
        pipeline.extend(["videoconvert", "!"])

        # Add FPS conversion if specified
        if validated_fps is not None:
            pipeline.extend(["videorate", "!", f"video/x-raw,framerate={validated_fps}/1", "!"])

        # Add resolution scaling if specified
        if validated_width is not None and validated_height is not None:
            pipeline.extend(["videoscale", "!", f"video/x-raw,width={validated_width},height={validated_height}", "!"])

        # Software H.265 encoding with x265enc
        # speed-preset=medium (balance quality/speed), bitrate=2000 (2 Mbps), key-int-max=60 (GOP size)
        pipeline.extend(["x265enc", "speed-preset=medium", "bitrate=2000", "key-int-max=60", "!"])

        # Parse H.265 stream
        pipeline.extend(["h265parse", "!"])

        # Output raw H.265 elementary stream (no muxing)
        pipeline.extend(["filesink", f"location={video_path}"])

    else:  # h264
        # H.264 pipeline using software encoding (existing method)
        pipeline.extend(["decodebin", "!"])

        # Always need videoconvert for format compatibility
        pipeline.extend(["videoconvert", "!"])

        # Add FPS conversion if specified
        if validated_fps is not None:
            pipeline.extend(["videorate", "!", f"video/x-raw,framerate={validated_fps}/1", "!"])

        # Add resolution scaling if specified
        if validated_width is not None and validated_height is not None:
            pipeline.extend(["videoscale", "!", f"video/x-raw,width={validated_width},height={validated_height}", "!"])

        # Software H.264 encoding
        pipeline.extend(["x264enc", "!", "h264parse", "!"])

        # Mux to MP4 container
        pipeline.extend(["mp4mux", "!", "filesink", f"location={video_path}"])

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

    # Handle original video cleanup based on keep_original flag
    if keep_original:
        # Move to debug subfolder for debugging
        debug_dir = os.path.join(VIDEOS_DIR, "debug")
        Path(debug_dir).mkdir(parents=True, exist_ok=True, mode=0o750)
        debug_video_path = os.path.join(debug_dir, f"{file_basename}_before_convert{file_extension}")
        shutil.move(source_video_path, debug_video_path)
        logger.debug(f"Moved original video to debug folder: {debug_video_path}")
    else:
        # Delete to optimize storage
        os.remove(source_video_path)
        logger.debug(f"Deleted original video to optimize storage: {source_video_path}")

    video_path = Path(video_path).resolve()

    return config


def trim_video(name: str, duration=30, keep_original=False) -> Dict[str, Any]:
    """
    Trim a video to a specified duration.
    Returns a dict with video configuration details.

    Args:
        name: Video filename
        duration: Target duration in seconds
        keep_original: If True, move original to debug/ subfolder; if False, delete it (default)
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

    # Handle original video cleanup based on keep_original flag
    if keep_original:
        # Move to debug subfolder for debugging
        debug_dir = os.path.join(VIDEOS_DIR, "debug")
        Path(debug_dir).mkdir(parents=True, exist_ok=True, mode=0o750)
        debug_video_path = os.path.join(debug_dir, f"{file_basename}_before_trim{file_extension}")
        shutil.move(source_video_path, debug_video_path)
        logger.debug(f"Moved original video to debug folder: {debug_video_path}")
    else:
        # Delete to optimize storage
        os.remove(source_video_path)
        logger.debug(f"Deleted original video to optimize storage: {source_video_path}")

    video_path = Path(video_path).resolve()

    return config


def extend_video(name: str, target_duration: int, keep_original=False) -> Dict[str, Any]:
    """
    Extend a video to reach target duration by looping/repeating the content.
    If the original video is longer than target_duration, it will be trimmed instead.

    Args:
        name: Video filename
        target_duration: Target duration in seconds
        keep_original: If True, move original to debug/ subfolder; if False, delete it (default)

    Returns:
        Dict with video configuration details
    """
    # Validate inputs
    validated_name = validate_video_filename(name)
    validated_target_duration = validate_duration(target_duration)

    # Paths
    source_video_path = os.path.join(VIDEOS_DIR, validated_name)

    # Ensure source video exists
    validate_video_file_exists(validated_name)

    file_basename, file_extension = os.path.splitext(validated_name)

    logger.info(f"Extending video '{validated_name}' to {validated_target_duration} seconds")

    # Get original video duration using ffprobe
    # Try format duration first, then stream duration (for raw streams like .h265)
    original_duration = None

    try:
        # First try format duration (works for container formats like mp4, mkv)
        ffprobe_result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                source_video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        duration_str = ffprobe_result.stdout.strip()
        if duration_str and duration_str != "N/A":
            original_duration = float(duration_str)
            logger.debug(f"Original video duration from format: {original_duration}s")
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.debug(f"Could not get format duration: {e}")

    # If format duration failed, try stream duration (for raw streams)
    if original_duration is None:
        try:
            ffprobe_result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    source_video_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            duration_str = ffprobe_result.stdout.strip()
            if duration_str and duration_str != "N/A":
                original_duration = float(duration_str)
                logger.debug(f"Original video duration from stream: {original_duration}s")
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.debug(f"Could not get stream duration: {e}")

    # If both methods failed, calculate from frame count and fps
    if original_duration is None:
        try:
            # Get frame count
            frame_result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    source_video_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            frame_count = int(frame_result.stdout.strip())

            # Get frame rate
            fps_result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=r_frame_rate",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    source_video_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            fps_str = fps_result.stdout.strip()
            # Parse frame rate (e.g., "30/1" or "30")
            if "/" in fps_str:
                num, denom = fps_str.split("/")
                fps = float(num) / float(denom)
            else:
                fps = float(fps_str)

            original_duration = frame_count / fps
            logger.debug(f"Calculated video duration from {frame_count} frames at {fps} fps: {original_duration}s")
        except (subprocess.CalledProcessError, ValueError, ZeroDivisionError) as e:
            error_message = f"Failed to get video duration using all methods: {e}"
            logger.error(error_message)
            raise RuntimeError(error_message)

    if original_duration is None or original_duration <= 0:
        error_message = "Failed to determine video duration: invalid or zero duration"
        logger.error(error_message)
        raise RuntimeError(error_message)

    # Calculate how many loops needed
    if original_duration >= validated_target_duration:
        logger.info(
            f"Original video ({original_duration}s) is longer than or equal to target "
            f"({validated_target_duration}s). Trimming instead."
        )
        return trim_video(validated_name, validated_target_duration, keep_original)

    # Calculate number of loops needed
    num_loops = int(validated_target_duration / original_duration) + 1
    logger.info(f"Will loop video {num_loops} times to reach target duration")

    # Backup original if keep_original is True
    if keep_original:
        debug_dir = os.path.join(VIDEOS_DIR, "debug")
        Path(debug_dir).mkdir(parents=True, exist_ok=True, mode=0o750)
        backup_video_path = os.path.join(debug_dir, f"{file_basename}_before_extend{file_extension}")
        shutil.copy2(source_video_path, backup_video_path)
        logger.info(f"Backed up original video to debug folder: {backup_video_path}")

    # Create temporary output file (will replace original after processing)
    output_video_path = os.path.join(VIDEOS_DIR, f"{file_basename}_extended_temp{file_extension}")

    # Build FFmpeg pipeline to loop video
    # concat demuxer requires creating a file list
    concat_file_path = os.path.join(VIDEOS_DIR, f"{file_basename}_concat.txt")

    try:
        # Create concat file with repeated entries
        with open(concat_file_path, "w", encoding="utf-8") as f:
            for _ in range(num_loops):
                # Use relative path for concat file
                f.write(f"file '{validated_name}'\n")

        logger.debug(f"Created concat file: {concat_file_path}")

        # Build FFmpeg command to concatenate and trim
        ffmpeg_pipeline = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file_path,
            "-t",
            str(validated_target_duration),  # Trim to exact target duration
            "-c",
            "copy",  # Copy codec without re-encoding when possible
            "-y",  # Overwrite output file
            output_video_path,
        ]

        logger.debug(f"FFmpeg pipeline: {' '.join(ffmpeg_pipeline)}")

        # Execute FFmpeg pipeline
        subprocess.run(ffmpeg_pipeline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        logger.debug("FFmpeg extension completed successfully.")

        # Replace original file with extended version
        shutil.move(output_video_path, source_video_path)
        logger.debug(f"Replaced original video with extended version: {source_video_path}")

    except subprocess.CalledProcessError as e:
        error_message = (
            f"Extending video with FFmpeg failed with exit code {e.returncode}\n"
            f"Command: {' '.join(ffmpeg_pipeline)}\n"
            f"Stdout:\n{e.stdout}\n"
            f"Stderr:\n{e.stderr}\n"
        )
        logger.error(error_message)
        raise RuntimeError(error_message)
    finally:
        # Clean up concat file
        if os.path.exists(concat_file_path):
            os.remove(concat_file_path)
            logger.debug(f"Removed concat file: {concat_file_path}")

    # Build return config - use original filename since we replaced it
    config = {
        "name": validated_name,  # Return original name (file was replaced in-place)
        "path": source_video_path,
        "target_duration": validated_target_duration,
        "original_duration": original_duration,
        "num_loops": num_loops,
    }

    logger.info(f"Successfully extended video: {validated_name}")
    return config


def main():
    """Main entry point for video utilities."""
    parser = argparse.ArgumentParser(description="Video Utilities")

    # Common arguments
    parser.add_argument("--video-name", required=True, help="Video file name")

    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Convert video command
    convert_parser = subparsers.add_parser("convert", help="Convert video with specified transformations")
    convert_parser.add_argument("--width", type=int, default=None, help="Target video width (optional)")
    convert_parser.add_argument("--height", type=int, default=None, help="Target video height (optional)")
    convert_parser.add_argument("--fps", type=int, default=None, help="Target video FPS (optional)")
    convert_parser.add_argument("--codec", type=str, default=None, help="Target video codec: h264 or h265 (optional)")
    convert_parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original video in debug/ subfolder (default: delete to optimize storage)",
    )

    # Trim video command
    trim_parser = subparsers.add_parser("trim", help="Trim video")
    trim_parser.add_argument("--duration", type=int, default=10, help="Video duration in seconds")
    trim_parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original video in debug/ subfolder (default: delete to optimize storage)",
    )

    # Extend video command
    extend_parser = subparsers.add_parser("extend", help="Extend video duration by looping")
    extend_parser.add_argument(
        "--target-duration",
        type=int,
        required=True,
        help="Target duration in seconds (video will be looped to reach this duration)",
    )
    extend_parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original video in debug/ subfolder (default: delete to optimize storage)",
    )

    args = parser.parse_args()

    try:
        sanitized_video_name = sanitize_path(args.video_name, VIDEOS_DIR)

        if args.command == "convert":
            # Pass parameters as-is to convert_video, which will validate only provided ones
            keep_original = args.keep_original

            convert_video(
                sanitized_video_name,
                width=args.width,
                height=args.height,
                fps=args.fps,
                codec=args.codec,
                keep_original=keep_original,
            )
            logger.info(f"Successfully converted video: {sanitized_video_name}")
        elif args.command == "trim":
            validated_duration = validate_duration(args.duration)
            keep_original = args.keep_original

            trim_video(sanitized_video_name, validated_duration, keep_original)
            logger.info(f"Successfully trimmed video: {sanitized_video_name}")
        elif args.command == "extend":
            validated_target_duration = validate_duration(args.target_duration)
            keep_original = args.keep_original

            extend_video(sanitized_video_name, validated_target_duration, keep_original)
            logger.info(f"Successfully extended video: {sanitized_video_name}")
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
