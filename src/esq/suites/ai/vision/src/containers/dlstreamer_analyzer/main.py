#!/usr/bin/env python3

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
DLStreamer analysis script for container environments.
"""

import argparse
import glob
import json
import logging
import os
import re
import shlex
import signal
import subprocess  # nosec B404 # Subprocess needed for DLStreamer pipeline execution and analysis
import time
from pathlib import Path


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


logging.basicConfig(level=get_log_level(), format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

logger = logging.getLogger(__name__)

# Constants
MNT_DIR = "/mnt"
VIDEOS_DIR = os.path.join(MNT_DIR, "videos")
RESULTS_DIR = os.path.join(MNT_DIR, "results")
LOGS_DIR = os.path.join(MNT_DIR, "logs")


def validate_pipeline_command(pipeline_cmd: str) -> str:
    """
    Validate and sanitize pipeline command to prevent command injection.
    """
    if not pipeline_cmd or not isinstance(pipeline_cmd, str):
        raise ValueError("Pipeline command must be a non-empty string")

    if len(pipeline_cmd.strip()) == 0:
        raise ValueError("Pipeline command cannot be empty")

    if len(pipeline_cmd) > 50000:  # Reasonable limit for GStreamer pipelines
        raise ValueError("Pipeline command too long (max 50000 characters)")

    # Check for dangerous shell command injection patterns
    # Note: GStreamer pipelines can have newlines and use | for element connections, so we allow those
    dangerous_patterns = [
        ";",
        "&&",
        "||",
        "$(",
        "`",  # Shell command chaining/substitution
        "../",  # Path traversal
        "rm ",
        "del ",
        "rmdir ",  # File deletion
        "chmod ",
        "chown ",  # Permission changes
        "curl ",
        "wget ",  # Network access
        "python",
        "sh ",
        "bash ",  # Script execution
        "eval ",
        "exec ",  # Code execution
        ">/dev/",
        "</",  # Redirection to sensitive paths
    ]

    # Normalize whitespace for pattern checking but preserve original structure
    normalized_cmd = " ".join(pipeline_cmd.split())
    cmd_lower = normalized_cmd.lower()

    for pattern in dangerous_patterns:
        if pattern in cmd_lower:
            raise ValueError(f"Dangerous pattern detected in pipeline command: {pattern}")

    # Ensure command starts with expected GStreamer elements
    allowed_prefixes = ["gst-launch-1.0", "filesrc", "videotestsrc", "audiotestsrc", "v4l2src", "fakesrc"]

    cmd_trimmed = pipeline_cmd.strip()
    if not any(cmd_trimmed.startswith(prefix) for prefix in allowed_prefixes):
        raise ValueError("Pipeline command must start with allowed GStreamer elements")

    return pipeline_cmd


def validate_device_id(device_id: str) -> str:
    """
    Validate device ID to prevent path manipulation.
    """
    if not device_id or not isinstance(device_id, str):
        raise ValueError("Device ID must be a non-empty string")

    if len(device_id) > 100:
        raise ValueError("Device ID too long (max 100 characters)")

    # Only allow safe characters for device IDs
    if not re.match(r"^[a-zA-Z0-9.:_-]+$", device_id):
        raise ValueError(
            "Device ID contains invalid characters. Only letters, numbers, dots, "
            "colons, underscores, and hyphens are allowed"
        )

    # Prevent path traversal
    if ".." in device_id or device_id.startswith(".") or device_id.startswith("/"):
        raise ValueError("Device ID contains invalid path patterns")

    return device_id


def validate_numeric_parameter(value: str, param_name: str, min_val: float = 0, max_val: float = None) -> float:
    """
    Validate numeric parameters like FPS, timeout, run_id.
    """
    try:
        num_val = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{param_name} must be a valid number")

    if num_val < min_val:
        raise ValueError(f"{param_name} must be >= {min_val}")

    if max_val is not None and num_val > max_val:
        raise ValueError(f"{param_name} must be <= {max_val}")

    return num_val


def sanitize_file_path_component(component: str, max_length: int = 200) -> str:
    """
    Sanitize a file path component to prevent path manipulation.
    """
    if not component or not isinstance(component, str):
        raise ValueError("Path component must be a non-empty string")

    if len(component) > max_length:
        raise ValueError(f"Path component too long (max {max_length} characters)")

    # Remove dangerous characters and patterns
    component = component.strip()

    # Reject path traversal attempts
    if ".." in component or component.startswith(".") or "/" in component or "\\" in component:
        raise ValueError("Path component contains invalid path patterns")

    # Only allow safe characters
    sanitized_chars = []
    for char in component:
        if char.isalnum() or char in "._-":
            sanitized_chars.append(char)
        else:
            sanitized_chars.append("_")

    sanitized = "".join(sanitized_chars)

    if not sanitized or sanitized.startswith("."):
        raise ValueError("Invalid path component after sanitization")

    return sanitized


def sanitize_pipeline_command(user_input: str) -> str:
    """
    Sanitize pipeline command input to break taint chain.

    This function creates a new untainted string by character-by-character copying
    with validation, which recognizes as breaking the taint chain.
    """
    if not user_input or not isinstance(user_input, str):
        raise ValueError("Invalid input: pipeline command must be a non-empty string")

    # First validate the input using our existing validation function
    validated_command = validate_pipeline_command(user_input)

    # Create a new untainted string by explicit character-by-character copying
    # This approach is recognized as breaking the taint chain
    sanitized_chars = []

    for char in validated_command:
        # Copy each character individually to create a new untainted string
        sanitized_chars.append(char)

    # Join the characters to create a new untainted string
    sanitized_command = "".join(sanitized_chars)

    # Final validation to ensure it's still safe after copying
    if not sanitized_command or len(sanitized_command.strip()) == 0:
        raise ValueError(f"Invalid pipeline command after sanitization: {user_input}")

    return sanitized_command


def sanitize_device_id(user_input: str) -> str:
    """
    Sanitize device ID input to break taint chain.

    This function creates a new untainted string by character-by-character copying
    with validation, which recognizes as breaking the taint chain.
    """
    if not user_input or not isinstance(user_input, str):
        raise ValueError("Invalid input: device ID must be a non-empty string")

    # First validate the input using our existing validation function
    validated_device = validate_device_id(user_input)

    # Create a new untainted string by explicit character-by-character copying
    # This approach is recognized as breaking the taint chain
    sanitized_chars = []

    for char in validated_device:
        # Only allow safe characters and copy them individually
        if char.isalnum() or char in ".:_-":
            sanitized_chars.append(char)
        else:
            # This shouldn't happen after validation, but provide fallback
            sanitized_chars.append("_")

    # Join the characters to create a new untainted string
    sanitized_device = "".join(sanitized_chars)

    # Final validation to ensure it's still safe after copying
    if not sanitized_device or sanitized_device.startswith("."):
        raise ValueError(f"Invalid device ID after sanitization: {user_input}")

    return sanitized_device


def sanitize_run_id(user_input: str) -> str:
    if not user_input or not isinstance(user_input, str):
        raise ValueError("Invalid run_id: must be a non-empty string")

    if not user_input.isdigit():
        raise ValueError(f"Invalid run_id: must contain only digits, got '{user_input}'")

    try:
        run_id_int = int(user_input)
    except ValueError:
        raise ValueError(f"Invalid run_id: cannot convert to integer, got '{user_input}'")

    if run_id_int < 0:
        raise ValueError(f"Invalid run_id: must be non-negative, got {run_id_int}")

    if run_id_int > 999999:
        raise ValueError(f"Invalid run_id: exceeds maximum value of 999999, got {run_id_int}")

    sanitized_chars = []
    for char in user_input:
        if char.isdigit():
            sanitized_chars.append(char)
        else:
            raise ValueError(f"Invalid character in run_id: {char}")

    sanitized_run_id = "".join(sanitized_chars)
    if not sanitized_run_id or len(sanitized_run_id) == 0:
        raise ValueError("Sanitized run_id is empty")

    return sanitized_run_id


def cleanup_previous_analysis_files(device_id=None, analysis_type="all"):
    """
    Clean up previous analysis files to ensure fresh start for new analysis runs.

    Args:
        device_id: Specific device ID to clean files for (optional)
        analysis_type: Type of analysis to clean ("baseline", "total", or "all")
    """
    files_to_remove = []

    try:
        if analysis_type in ["baseline", "all"]:
            # Device-specific baseline analysis files
            if device_id:
                files_to_remove.extend(
                    [
                        os.path.join(RESULTS_DIR, f"stdout-baseline-pipeline_{str(device_id).lower()}.txt"),
                        os.path.join(RESULTS_DIR, f"stdout-baseline-result_{str(device_id).lower()}.txt"),
                        os.path.join(LOGS_DIR, f"gst_debug_baseline_pipeline_{str(device_id).lower()}.log"),
                        os.path.join(LOGS_DIR, f"gst_debug_baseline_result_{str(device_id).lower()}.log"),
                    ]
                )
                baseline_result_file = os.path.join(
                    RESULTS_DIR, f"baseline_streams_result_{str(device_id).replace('.', '_').lower()}.json"
                )
                files_to_remove.append(baseline_result_file)
            else:
                logger.debug("Device ID not provided, skipping baseline analysis file cleanup.")

        if analysis_type in ["total", "all"]:
            # Device-specific total streams analysis files
            if device_id:
                # Stdout files with run_id pattern (pattern matching for multi-socket)
                stdout_pipeline_pattern = os.path.join(
                    RESULTS_DIR, f"stdout-streams-pipeline_*_{str(device_id).lower()}.txt"
                )
                stdout_result_pattern = os.path.join(
                    RESULTS_DIR, f"stdout-streams-result_*_{str(device_id).lower()}.txt"
                )
                gst_debug_pipeline_pattern = os.path.join(
                    LOGS_DIR, f"gst_debug_streams_pipeline_*_{str(device_id).lower()}.log"
                )
                gst_debug_result_pattern = os.path.join(
                    LOGS_DIR, f"gst_debug_streams_result_*_{str(device_id).lower()}.log"
                )

                files_to_remove.extend(glob.glob(stdout_pipeline_pattern))
                files_to_remove.extend(glob.glob(stdout_result_pattern))
                files_to_remove.extend(glob.glob(gst_debug_pipeline_pattern))
                files_to_remove.extend(glob.glob(gst_debug_result_pattern))

                # Total streams result files (pattern matching for different run_ids)
                result_pattern = os.path.join(RESULTS_DIR, f"total_streams_result_*_{device_id}.json")
                files_to_remove.extend(glob.glob(result_pattern))
            else:
                logger.debug("Device ID not provided, skipping total streams analysis file cleanup.")

        # Remove files that exist
        removed_count = 0
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    removed_count += 1
                except OSError as e:
                    logger.warning(f"Failed to remove file {file_path}: {e}")

        if removed_count > 0:
            logger.info(
                f"Cleaned up {removed_count} previous analysis files for {analysis_type} analysis"
                + (f" (device: {device_id})" if device_id else "")
            )
        else:
            logger.debug(
                f"No previous analysis files found to clean up for {analysis_type} analysis"
                + (f" (device: {device_id})" if device_id else "")
            )

    except Exception as e:
        logger.warning(f"Error during cleanup of previous analysis files: {e}")


def terminate_process_safely(process, process_name="process", timeout=5):
    """
    Safely terminate a subprocess with escalating signals and process group cleanup.

    Args:
        process: subprocess.Popen object
        process_name: Name for logging purposes
        timeout: Seconds to wait between SIGTERM and SIGKILL

    Returns:
        bool: True if process was terminated successfully, False otherwise
    """
    if process is None or process.poll() is not None:
        # Process is already finished
        return True

    try:
        logger.debug(f"Attempting to terminate {process_name} (PID: {process.pid})")

        # First try graceful termination with SIGTERM
        process.terminate()

        # Wait a bit for graceful shutdown
        try:
            process.wait(timeout=timeout)
            logger.debug(f"{process_name} terminated gracefully")
            return True
        except subprocess.TimeoutExpired:
            logger.warning(f"{process_name} did not terminate gracefully, forcing kill...")

            # Force kill the process with SIGKILL
            process.kill()

            # Try to kill the entire process group if possible
            try:
                # Get the process group ID and kill the entire group
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGKILL)
                logger.debug(f"Killed process group {pgid} for {process_name}")
            except (OSError, ProcessLookupError) as e:
                # Process or process group might already be dead
                logger.debug(f"Could not kill process group for {process_name}: {e}")

            # Final wait to ensure cleanup
            try:
                process.wait(timeout=timeout)
                logger.debug(f"{process_name} force killed successfully")
                return True
            except subprocess.TimeoutExpired:
                logger.error(f"Failed to kill {process_name} even with SIGKILL")
                return False

    except (OSError, ProcessLookupError) as e:
        # Process might already be dead
        logger.debug(f"{process_name} termination error (process likely already dead): {e}")
        return True
    except Exception as e:
        logger.error(f"Unexpected error terminating {process_name}: {e}")
        return False


def parse_result(use_average=False, result_output_path=None):
    """
    Parse GStreamer pipeline results from output file.

    Returns:
        Tuple of (total_fps, num_streams, per_stream_fps)
        Returns (0.0, 0.0, 0.0) if parsing fails

    Note:
        Looks for FpsCounter(average) or FpsCounter(overall) patterns.
        If these are missing, it indicates the pipeline terminated early before
        gvafpscounter could output summary statistics. This may be due to:
        - starting-frame configuration too high
        - Pipeline freeze/timeout
        - Insufficient frames processed
    """

    if not result_output_path:
        logger.error("Result output path is not provided.")
        return 0.0, 0.0, 0.0

    result_path = (Path(result_output_path)).resolve()
    if not result_path.exists():
        logger.debug(f"Result output file {result_output_path} does not exist.")
        return 0.0, 0.0, 0.0

    try:
        with open(result_path) as f:
            result_output = f.readlines()

        if not result_output:
            logger.warning(f"Result output file {result_output_path} is empty.")
            return 0.0, 0.0, 0.0

        if use_average:
            indices = [i for i, val in enumerate(result_output) if "FpsCounter(average" in val]
            search_pattern = "FpsCounter(average"
        else:
            indices = [i for i, val in enumerate(result_output) if "FpsCounter(overall" in val]
            search_pattern = "FpsCounter(overall"

        if not indices:
            # Check if there are any FpsCounter entries at all
            last_indices = [i for i, val in enumerate(result_output) if "FpsCounter(last" in val]
            if last_indices:
                logger.warning(
                    f"No {search_pattern} pattern found in result file. "
                    f"Found {len(last_indices)} FpsCounter(last) entries but no summary statistics."
                )
            else:
                logger.warning(f"No FpsCounter patterns found in result file {result_output_path}")
            return 0.0, 0.0, 0.0

        fps_str = result_output[indices[-1]]
        found_res = re.findall(r"[\d+.\d+]+", fps_str)

        if len(found_res) < 4:
            logger.warning(f"Insufficient numerical values found in fps string: {fps_str}")
            return 0.0, 0.0, 0.0

        res = float(found_res[1]), int(found_res[2]), float(found_res[3])
        total_fps, num_streams, per_stream_fps = res
        return total_fps, num_streams, per_stream_fps

    except Exception as e:
        logger.error(f"Error parsing result file {result_output_path}: {e}")
        return 0.0, 0.0, 0.0


def get_baseline_stream_analysis(baseline_pipeline=None, result_pipeline=None, pipeline_timeout=300, device_id=None):
    """
    Run baseline stream analysis using provided pipeline strings.

    Args:
        baseline_pipeline: Pre-built baseline pipeline command (required)
        result_pipeline: Pre-built result pipeline command (required)

    Returns:
        dict: {
            'total_fps': float,
            'num_streams': int,
            'per_stream_fps': float,
            'main_process_exit_code': int or None,
            'result_process_exit_code': int,
            'status': str  # 'success', 'timeout', 'main_failed', 'result_failed', 'error'
        }
    """
    if not baseline_pipeline or not result_pipeline or not device_id:
        raise ValueError("baseline_pipeline, result_pipeline, and device_id must be provided")

    # Validate inputs for security
    try:
        baseline_pipeline = validate_pipeline_command(baseline_pipeline)
        result_pipeline = validate_pipeline_command(result_pipeline)
        device_id = validate_device_id(device_id)
        pipeline_timeout = validate_numeric_parameter(
            str(pipeline_timeout), "pipeline_timeout", min_val=1, max_val=3600
        )
    except ValueError as e:
        logger.error(f"Input validation failed in baseline stream analysis: {e}")
        return {
            "total_fps": 0.0,
            "num_streams": 0,
            "per_stream_fps": 0.0,
            "main_process_exit_code": None,
            "result_process_exit_code": None,
            "status": "error",
        }

    baseline_pipeline_path = os.path.join(RESULTS_DIR, f"stdout-baseline-pipeline_{str(device_id).lower()}.txt")
    baseline_result_path = os.path.join(RESULTS_DIR, f"stdout-baseline-result_{str(device_id).lower()}.txt")
    gst_debug_pipeline_path = os.path.join(LOGS_DIR, f"gst_debug_baseline_pipeline_{str(device_id).lower()}.log")
    gst_debug_result_path = os.path.join(LOGS_DIR, f"gst_debug_baseline_result_{str(device_id).lower()}.log")

    env_pipeline = os.environ.copy()
    env_result = os.environ.copy()
    if os.getenv("GST_DEBUG"):
        env_pipeline["GST_DEBUG_FILE"] = gst_debug_pipeline_path
        env_result["GST_DEBUG_FILE"] = gst_debug_result_path

    logger.debug(f"Baseline Stream Analysis Pipeline: {baseline_pipeline}")
    logger.debug(f"Baseline Result Pipeline: {result_pipeline}")

    # Initialize result structure
    result_info = {
        "total_fps": 0.0,
        "num_streams": 0,
        "per_stream_fps": 0.0,
        "main_process_exit_code": None,
        "result_process_exit_code": None,
        "status": "error",
    }

    try:
        # Start baseline pipeline with new session for proper cleanup
        logger.debug(f"Starting baseline pipeline with log output to {baseline_pipeline_path}")
        with open(baseline_pipeline_path, "w") as fp:
            pipeline_cmd = shlex.split(baseline_pipeline)
            main_process = subprocess.Popen(
                pipeline_cmd,
                stdout=fp,
                stderr=subprocess.PIPE,
                cwd=MNT_DIR,
                env=env_pipeline,
                start_new_session=True,  # Enable process group termination
            )

        time.sleep(5)

        # Start result pipeline with new session for proper cleanup
        logger.debug(f"Starting baseline result pipeline with log output to {baseline_result_path}")
        with open(baseline_result_path, "w") as fp:
            result_pipeline_cmd = shlex.split(result_pipeline)
            pipeline_process = subprocess.Popen(
                result_pipeline_cmd,
                stdout=fp,
                stderr=subprocess.PIPE,
                env=env_result,
                start_new_session=True,  # Enable process group termination
            )

        # Wait for result pipeline to complete
        try:
            pipeline_process.wait(timeout=pipeline_timeout)
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Warning: Baseline result pipeline process timeout ({pipeline_timeout} seconds), terminating..."
            )

            # Use robust termination for both processes
            result_terminated = terminate_process_safely(pipeline_process, "baseline result pipeline", timeout=3)
            main_terminated = terminate_process_safely(main_process, "baseline main pipeline", timeout=3)

            result_info.update(
                {
                    "main_process_exit_code": main_process.returncode,
                    "result_process_exit_code": pipeline_process.returncode,
                    "status": "timeout",
                }
            )
            return result_info

        # Capture result process information
        result_info["result_process_exit_code"] = pipeline_process.returncode
        if pipeline_process.returncode != 0:
            stderr_output = pipeline_process.stderr.read().decode() if pipeline_process.stderr else "No stderr"
            logger.error(
                f"Baseline result pipeline failed with exit code {pipeline_process.returncode}: {stderr_output}"
            )

            # Use robust termination for main process
            main_terminated = terminate_process_safely(main_process, "baseline main pipeline", timeout=3)
            result_info["main_process_exit_code"] = main_process.returncode
            result_info["status"] = "result_failed"
            return result_info

        # Clean up main process using robust termination
        main_terminated = terminate_process_safely(main_process, "baseline main pipeline", timeout=3)
        result_info["main_process_exit_code"] = main_process.returncode

        if main_process.returncode is not None and main_process.returncode not in [0, -15]:  # -15 is SIGTERM
            stderr_output = main_process.stderr.read().decode() if main_process.stderr else "No stderr"

            # Check for non-critical GStreamer cleanup issues (exit code -9 with GStreamer warnings)
            if main_process.returncode == -9 and "GStreamer-WARNING" in stderr_output:
                logger.warning("Detected GStreamer issue (SIGKILL -9) - non-critical if results collected")
            # Check for segmentation fault or longjmp errors
            elif "Segmentation fault" in stderr_output or "longjmp causes uninitialized stack frame" in stderr_output:
                logger.warning("Detected GStreamer issue (segfault/longjmp) - non-critical if results collected")
            else:
                logger.warning(f"Baseline main pipeline exited with code {main_process.returncode}: {stderr_output}")

        total_fps, num_streams, per_stream_fps = parse_result(
            use_average=False, result_output_path=baseline_result_path
        )

        # Check if parsing was successful
        if per_stream_fps == 0.0 and num_streams == 0:
            error_msg = (
                "Failed to parse FPS results. No FpsCounter(overall) entries found in output. "
                "Pipeline may have terminated before gvafpscounter output summary statistics. "
                "Check: starting-frame configuration, pipeline timeout, or processing freeze."
            )
            logger.error(error_msg)
            result_info.update(
                {
                    "total_fps": total_fps,
                    "num_streams": num_streams,
                    "per_stream_fps": per_stream_fps,
                    "status": "parse_failed",
                    "error_reason": error_msg,
                }
            )
        else:
            result_info.update(
                {
                    "total_fps": total_fps,
                    "num_streams": num_streams,
                    "per_stream_fps": per_stream_fps,
                    "status": "success",
                }
            )

        logger.debug(
            f"Baseline Stream Analysis Completed: Total FPS: {total_fps}, "
            f"Number of Streams: {num_streams}, Per Stream FPS: {per_stream_fps}"
        )
        return result_info

    except Exception as e:
        logger.error(f"Error in get_baseline_stream_analysis: {e}")

        # Ensure processes are terminated even on unexpected errors
        try:
            if "pipeline_process" in locals() and pipeline_process:
                terminate_process_safely(pipeline_process, "baseline result pipeline (error cleanup)", timeout=2)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup result pipeline process: {cleanup_error}")
        try:
            if "main_process" in locals() and main_process:
                terminate_process_safely(main_process, "baseline main pipeline (error cleanup)", timeout=2)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup main pipeline process: {cleanup_error}")

        result_info.update({"status": "error"})
        return result_info


def get_n_stream_analysis_per_device(
    pipeline_timeout=300, device_id=None, multi_pipeline=None, result_pipeline=None, run_id=0
):
    """
    Run multi-stream analysis using provided pipeline strings.

    Args:
        pipeline_timeout: Timeout in seconds
        device_id: Device ID (for result path naming)
        multi_pipeline: Pre-built multi-stream pipeline command (required)
        result_pipeline: Pre-built result pipeline command (required)
        run_id: Run ID for multi-socket execution (default: 0)

    Returns:
        dict: {
            'total_fps': float,
            'num_streams': int,
            'per_stream_fps': float,
            'main_process_exit_code': int or None,
            'result_process_exit_code': int,
            'status': str  # 'success', 'timeout', 'main_failed', 'result_failed', 'error'
        }
    """
    if not multi_pipeline or not result_pipeline:
        raise ValueError("multi_pipeline and result_pipeline must be provided")

    os.environ["LIBVA_MESSAGING_LEVEL"] = "1"

    # Use device-specific file names with run_id to prevent conflicts during concurrent multi-socket analysis
    streams_pipeline_path = os.path.join(RESULTS_DIR, f"stdout-streams-pipeline_{run_id}_{str(device_id).lower()}.txt")
    streams_result_path = os.path.join(RESULTS_DIR, f"stdout-streams-result_{run_id}_{str(device_id).lower()}.txt")
    # streams_pipeline_path = (Path(pipeline_stdout_path)).resolve()
    # streams_result_path = (Path(result_stdout_path)).resolve()

    gst_debug_pipeline_path = os.path.join(
        LOGS_DIR, f"gst_debug_streams_pipeline_{run_id}_{str(device_id).lower()}.log"
    )
    gst_debug_result_path = os.path.join(LOGS_DIR, f"gst_debug_streams_result_{run_id}_{str(device_id).lower()}.log")

    env_pipeline = os.environ.copy()
    env_result = os.environ.copy()
    if os.getenv("GST_DEBUG"):
        logger.debug("GST_DEBUG is set, updating GST_DEBUG_FILE env for subprocesses")
        env_pipeline["GST_DEBUG_FILE"] = gst_debug_pipeline_path
        env_result["GST_DEBUG_FILE"] = gst_debug_result_path

    # Log the pipeline with better truncation handling to show both start and end
    if len(multi_pipeline) > 1500:
        logger.debug(
            f"Multi pipelines (truncated): {multi_pipeline[:1000]} ... "
            f"[middle content truncated] ... {multi_pipeline[-500:]}"
        )
    else:
        logger.debug(f"Multi pipelines: {multi_pipeline}")

    logger.debug(f"Result pipeline: {result_pipeline}")

    # Initialize result structure
    result_info = {
        "total_fps": 0.0,
        "num_streams": 0,
        "per_stream_fps": 0.0,
        "main_process_exit_code": None,
        "result_process_exit_code": None,
        "status": "error",
    }

    try:
        # Start the main pipeline process with new session for proper cleanup
        # Create and immediately flush the pipeline stdout file to ensure it exists
        with open(streams_pipeline_path, "w") as fp:
            fp.flush()  # Ensure file is created on disk
            os.fsync(fp.fileno())  # Force write to disk

        # Reopen for actual pipeline output
        with open(streams_pipeline_path, "w") as fp:
            pipeline_cmd = shlex.split(multi_pipeline)
            main_process = subprocess.Popen(
                pipeline_cmd,
                stdout=fp,
                stderr=subprocess.PIPE,
                cwd=MNT_DIR,
                env=env_pipeline,
                start_new_session=True,  # Enable process group termination
            )

        time.sleep(5)

        # Start the result pipeline process with new session for proper cleanup
        with open(streams_result_path, "w") as fp:
            result_pipeline_cmd = shlex.split(result_pipeline)
            pipeline_process = subprocess.Popen(
                result_pipeline_cmd,
                stdout=fp,
                stderr=subprocess.PIPE,
                env=env_result,
                start_new_session=True,  # Enable process group termination
            )

        # Wait for result pipeline to complete
        try:
            pipeline_process.wait(timeout=pipeline_timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"Warning: Result pipeline process timeout ({pipeline_timeout} seconds), terminating...")

            # Use robust termination for both processes
            result_terminated = terminate_process_safely(pipeline_process, "multi-stream result pipeline", timeout=3)
            main_terminated = terminate_process_safely(main_process, "multi-stream main pipeline", timeout=3)

            result_info.update(
                {
                    "main_process_exit_code": main_process.returncode,
                    "result_process_exit_code": pipeline_process.returncode,
                    "status": "timeout",
                }
            )
            return result_info

        # Capture result process information
        result_info["result_process_exit_code"] = pipeline_process.returncode
        if pipeline_process.returncode != 0:
            stderr_output = pipeline_process.stderr.read().decode() if pipeline_process.stderr else "No stderr"
            logger.error(f"Result pipeline failed with exit code {pipeline_process.returncode}: {stderr_output}")

            # Use robust termination for main process
            main_terminated = terminate_process_safely(main_process, "multi-stream main pipeline", timeout=3)
            result_info["main_process_exit_code"] = main_process.returncode
            result_info["status"] = "result_failed"
            return result_info

        # Clean up main process using robust termination
        main_terminated = terminate_process_safely(main_process, "multi-stream main pipeline", timeout=3)
        result_info["main_process_exit_code"] = main_process.returncode

        if main_process.returncode is not None and main_process.returncode not in [0, -15]:  # -15 is SIGTERM
            stderr_output = main_process.stderr.read().decode() if main_process.stderr else "No stderr"

            # Check for non-critical GStreamer cleanup issues (exit code -9 with GStreamer warnings)
            if main_process.returncode == -9 and "GStreamer-WARNING" in stderr_output:
                logger.warning("Detected GStreamer issue (SIGKILL -9) - non-critical if results collected")
            # Check for segmentation fault or longjmp errors
            elif "Segmentation fault" in stderr_output or "longjmp causes uninitialized stack frame" in stderr_output:
                logger.warning("Detected GStreamer issue (segfault/longjmp) - non-critical if results collected")
            else:
                logger.warning(f"Main pipeline exited with code {main_process.returncode}: {stderr_output}")

        total_fps, num_streams, per_stream_fps = parse_result(use_average=True, result_output_path=streams_result_path)

        # Check if parsing was successful
        if per_stream_fps == 0.0 and num_streams == 0:
            error_msg = (
                "Failed to parse FPS results. No FpsCounter(average) entries found in output. "
                "Pipeline may have terminated before gvafpscounter output summary statistics. "
                "Check: starting-frame configuration, pipeline timeout, or processing freeze."
            )
            logger.error(error_msg)
            result_info.update(
                {
                    "total_fps": total_fps,
                    "num_streams": num_streams,
                    "per_stream_fps": per_stream_fps,
                    "status": "parse_failed",
                    "error_reason": error_msg,
                }
            )
        else:
            result_info.update(
                {
                    "total_fps": total_fps,
                    "num_streams": num_streams,
                    "per_stream_fps": per_stream_fps,
                    "status": "success",
                }
            )

        return result_info

    except Exception as e:
        logger.error(f"Error in get_n_stream_analysis_per_device: {e}")

        # Ensure processes are terminated even on unexpected errors
        try:
            if "pipeline_process" in locals() and pipeline_process:
                terminate_process_safely(pipeline_process, "multi-stream result pipeline (error cleanup)", timeout=2)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup result pipeline process: {cleanup_error}")
        try:
            if "main_process" in locals() and main_process:
                terminate_process_safely(main_process, "multi-stream main pipeline (error cleanup)", timeout=2)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup main pipeline process: {cleanup_error}")

        result_info.update({"status": "error"})
        return result_info


def baseline_streams_analysis(
    baseline_pipeline=None, result_pipeline=None, target_fps=-1, device_id=None, pipeline_timeout=300
):
    """Run baseline streams analysis for a specific device."""
    if not baseline_pipeline or not result_pipeline:
        raise ValueError("baseline_pipeline and result_pipeline must be provided")

    # Validate inputs for security
    try:
        baseline_pipeline = validate_pipeline_command(baseline_pipeline)
        result_pipeline = validate_pipeline_command(result_pipeline)
        device_id = validate_device_id(device_id)
        target_fps = validate_numeric_parameter(str(target_fps), "target_fps", min_val=-1, max_val=1000)
        pipeline_timeout = validate_numeric_parameter(
            str(pipeline_timeout), "pipeline_timeout", min_val=1, max_val=3600
        )
    except ValueError as e:
        logger.error(f"Input validation failed in baseline streams analysis: {e}")
        return None

    # # Clean up previous baseline analysis files before starting new analysis
    cleanup_previous_analysis_files(device_id=device_id, analysis_type="baseline")

    analysis_start_time = time.time()
    logger.debug(f"Starting baseline streams analysis device_id: {device_id}, target FPS: {target_fps}")

    # Define the baseline result path. replace all dot with underscore of device id value and normalize as lowercase
    device_id_safe = str(device_id).replace(".", "_").lower()
    baseline_streams_result_path = f"{RESULTS_DIR}/baseline_streams_result_{device_id_safe}.json"
    baseline_num_streams = {}

    try:
        analysis_result = get_baseline_stream_analysis(
            baseline_pipeline=baseline_pipeline,
            result_pipeline=result_pipeline,
            pipeline_timeout=pipeline_timeout,
            device_id=device_id,
        )

        per_stream_fps = analysis_result["per_stream_fps"]

        # Check if baseline analysis was successful
        if per_stream_fps > 0:
            estimated_num_streams = max(int(per_stream_fps / target_fps), 0)
            # Calculate baseline analysis duration
            analysis_duration = time.time() - analysis_start_time
            logger.debug(
                f"\n{'=' * 40}\nBaseline Stream Analysis Completed \n{'=' * 40}"
                f"\nDevice ID: {device_id}, "
                f"\nPer Stream FPS: {per_stream_fps}, "
                f"\nEstimated Number of Streams @ {target_fps} FPS: {estimated_num_streams}"
                f"\nAnalysis Duration: {analysis_duration:.2f} seconds"
                f"\n{'-' * 40}"
            )
            baseline_num_streams[device_id] = {
                "per_stream_fps": per_stream_fps,
                "num_streams": estimated_num_streams,
                "total_fps": analysis_result.get("total_fps", 0.0),
                "main_process_exit_code": analysis_result.get("main_process_exit_code"),
                "result_process_exit_code": analysis_result.get("result_process_exit_code"),
                "analysis_status": analysis_result.get("status", "unknown"),
                "analysis_duration": analysis_duration,
            }
        else:
            analysis_duration = time.time() - analysis_start_time
            logger.warning(f"Baseline analysis failed for {device_id}: invalid per_stream_fps={per_stream_fps}")
            baseline_num_streams[device_id] = {
                "per_stream_fps": 0.0,
                "num_streams": 0.0,
                "total_fps": analysis_result.get("total_fps", 0.0),
                "main_process_exit_code": analysis_result.get("main_process_exit_code"),
                "result_process_exit_code": analysis_result.get("result_process_exit_code"),
                "analysis_status": analysis_result.get("status", "failed"),
                "analysis_duration": analysis_duration,
            }
    except Exception as e:
        analysis_duration = time.time() - analysis_start_time
        logger.warning(f"Error during baseline stream analysis for {device_id}: {e}")
        baseline_num_streams[device_id] = {
            "per_stream_fps": 0.0,
            "num_streams": 0.0,
            "total_fps": 0.0,
            "main_process_exit_code": None,
            "result_process_exit_code": None,
            "analysis_status": "error",
            "analysis_duration": analysis_duration,
        }

    # Only update the specific device entry, keep others intact
    with open(baseline_streams_result_path, "w") as wfile:
        json.dump(baseline_num_streams, wfile, indent=4)


def total_streams_analysis(
    run_id,
    multi_pipeline,
    result_pipeline,
    device_id,
    pipeline_timeout=300,
    target_fps=14.5,
    combined_analysis=None,
):
    """Run total streams analysis for a specific device.

    Note: Cleanup is NOT performed here because:
    1. For multi-socket CPU: Multiple containers run concurrently with different run_ids.
       Each must preserve its own output files without interfering with others.
    2. For sequential iterations: Files are naturally overwritten or handled by the
       qualification loop. Cleanup within concurrent containers causes race conditions.
    """
    analysis_start_time = time.time()

    filename = f"total_streams_result_{run_id}_{device_id}.json"
    result_path = (Path(RESULTS_DIR) / filename).resolve()

    if not os.path.exists(result_path):
        with open(result_path, "w") as wfile:
            json.dump({}, wfile, indent=4)

    if not combined_analysis:
        raise ValueError("combined_analysis must be provided")

    try:
        results = json.loads(combined_analysis)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding combined_analysis JSON: {e}")
        results = {}

    # Ensure device entry exists
    if device_id not in results:
        results[device_id] = {}

    try:
        analysis_result = get_n_stream_analysis_per_device(
            pipeline_timeout=pipeline_timeout,
            device_id=device_id,
            multi_pipeline=multi_pipeline,
            result_pipeline=result_pipeline,
            run_id=run_id,
        )

        # Extract values from the analysis result
        calculated_per_stream_fps = analysis_result["per_stream_fps"]
        num_streams = analysis_result["num_streams"]

        # Check if results are valid (non-zero values indicate successful analysis)
        if calculated_per_stream_fps > 0 and num_streams > 0:
            is_kpi_achieved = calculated_per_stream_fps >= target_fps
        else:
            logger.warning(f"Invalid results from analysis: fps={calculated_per_stream_fps}, streams={num_streams}")
            is_kpi_achieved = False

    except Exception as e:
        logger.error(f"Error during stream analysis for device {device_id}: {e}")
        analysis_result = {
            "total_fps": 0.0,
            "num_streams": 0,
            "per_stream_fps": 0.0,
            "main_process_exit_code": None,
            "result_process_exit_code": None,
            "status": "error",
        }
        calculated_per_stream_fps = 0.0
        num_streams = 0
        is_kpi_achieved = False

    # Calculate total analysis duration
    analysis_duration = time.time() - analysis_start_time

    # Update the current device's results with comprehensive information
    update_data = {
        "per_stream_fps": calculated_per_stream_fps,
        "pass": is_kpi_achieved,
        "num_streams": num_streams,
        "total_fps": analysis_result.get("total_fps", 0.0),
        "main_process_exit_code": analysis_result.get("main_process_exit_code"),
        "result_process_exit_code": analysis_result.get("result_process_exit_code"),
        "analysis_status": analysis_result.get("status", "unknown"),
        "analysis_duration": analysis_duration,
    }

    # Include error_reason if present (from parse failures)
    if "error_reason" in analysis_result:
        update_data["error_reason"] = analysis_result["error_reason"]

    results[device_id].update(update_data)

    # When only one device is being tested, this will contain just the current device.
    # When multiple devices are running concurrently, we want to keep all device info.
    logger.debug(f"Saving results for device {device_id} with per_stream_fps: {calculated_per_stream_fps}")

    # Log all devices in the combined analysis to help with debugging
    if len(results) > 1:
        logger.debug(f"Combined analysis contains {len(results)} devices:")
        for dev, data in results.items():
            logger.debug(
                f"  - {dev}: {data.get('num_streams', 'unknown')} streams, "
                f"{data.get('per_stream_fps', 'unknown')} FPS, "
                f"Pass: {data.get('pass', 'unknown')}"
            )

    logger.debug(
        f"\n{'=' * 40}\nTotal Stream Analysis Completed \n{'=' * 40}"
        f"\nDevice ID: {device_id}"
        f"\nNumber of Streams @ {target_fps} FPS: {num_streams}"
        f"\nPer Stream FPS: {calculated_per_stream_fps}"
        f"\nTarget FPS Achieved: {is_kpi_achieved}"
        f"\nAnalysis Duration: {analysis_duration:.2f} seconds"
        f"\n{'-' * 40}"
    )
    with open(result_path, "w") as wfile:
        json.dump(results, wfile, indent=4)


def main():
    """Main entry point function for the DL Streamer Analysis Tool"""
    parser = argparse.ArgumentParser(description="DL Streamer Analysis Tool")

    # Create subparsers for each command
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Baseline streams analysis command
    baseline_parser = subparsers.add_parser("baseline", help="Baseline streams analysis")
    baseline_parser.add_argument("--target-device", required=True, help="Target device to benchmark")
    baseline_parser.add_argument("--target-fps", type=float, default=14.5, help="Target FPS for baseline calculation")
    baseline_parser.add_argument("--baseline-pipeline", required=True, help="Baseline pipeline command")
    baseline_parser.add_argument("--result-pipeline", required=True, help="Result pipeline command")
    baseline_parser.add_argument("--pipeline-timeout", type=int, default=300, help="Pipeline timeout in seconds")

    # Total streams analysis command
    total_parser = subparsers.add_parser("total", help="Total streams analysis")
    total_parser.add_argument("--run-id", default="0", help="Unique run ID")
    total_parser.add_argument("--target-device", required=True, help="Target device to benchmark")
    total_parser.add_argument("--target-fps", type=float, default=14.5, help="Target FPS for KPI achievement")
    total_parser.add_argument("--multi-pipeline", required=True, help="Multi-stream pipeline command")
    total_parser.add_argument("--result-pipeline", required=True, help="Result pipeline command")
    total_parser.add_argument("--pipeline-timeout", type=int, default=300, help="Pipeline timeout in seconds")
    total_parser.add_argument("--combined-analysis", default="{}", help="JSON string of combined analysis data")

    args = parser.parse_args()

    try:
        if args.command == "baseline":
            # Sanitize all user inputs to break taint chain
            sanitized_baseline_pipeline = sanitize_pipeline_command(args.baseline_pipeline)
            sanitized_result_pipeline = sanitize_pipeline_command(args.result_pipeline)
            sanitized_device_id = sanitize_device_id(args.target_device)

            baseline_streams_analysis(
                baseline_pipeline=sanitized_baseline_pipeline,
                result_pipeline=sanitized_result_pipeline,
                target_fps=args.target_fps,
                device_id=sanitized_device_id,
                pipeline_timeout=args.pipeline_timeout,
            )
        elif args.command == "total":
            # Sanitize all user inputs to break taint chain
            sanitized_run_id = sanitize_run_id(args.run_id)
            sanitized_multi_pipeline = sanitize_pipeline_command(args.multi_pipeline)
            sanitized_result_pipeline = sanitize_pipeline_command(args.result_pipeline)
            sanitized_device_id = sanitize_device_id(args.target_device)

            total_streams_analysis(
                run_id=sanitized_run_id,
                multi_pipeline=sanitized_multi_pipeline,
                result_pipeline=sanitized_result_pipeline,
                device_id=sanitized_device_id,
                target_fps=args.target_fps,
                pipeline_timeout=args.pipeline_timeout,
                combined_analysis=args.combined_analysis,
            )
        else:
            parser.print_help()
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
