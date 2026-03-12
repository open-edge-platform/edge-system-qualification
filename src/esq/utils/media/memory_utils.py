# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Memory monitoring utilities for media benchmarks.

This module provides functions for monitoring system memory usage and
calculating memory-based stream limits to prevent OOM errors on
memory-constrained platforms (especially iGPU systems that share RAM).
"""

import logging
from importlib.util import find_spec
from typing import Any, Optional, Tuple

# Try to import psutil for memory monitoring
psutil: Any = None
if find_spec("psutil") is not None:
    import importlib

    psutil = importlib.import_module("psutil")
    PSUTIL_AVAILABLE = True
else:
    PSUTIL_AVAILABLE = False

# Memory-based stream limits (streams per GB of available RAM)
# These are conservative estimates to prevent OOM:
# - Each stream uses ~200-400MB for decode + inference
# - iGPU shares system RAM, so needs more conservative limits
STREAMS_PER_GB_IGPU = 1.5  # Conservative for shared memory iGPU
STREAMS_PER_GB_DGPU = 3.0  # dGPU has dedicated VRAM, less system RAM pressure
MIN_AVAILABLE_MEMORY_GB = 4.0  # Minimum available memory to continue scaling
MEMORY_SAFETY_MARGIN = 0.8  # Use only 80% of available memory for streams
DEFAULT_MAX_STREAMS_UNLIMITED = 100  # Default max when memory info unavailable
CONCURRENT_MODE_STREAM_FACTOR = 0.6  # Additional cap for GPU+NPU concurrent VA modes

def get_memory_info() -> Tuple[float, float, float]:
    """
    Get system memory information.

    Returns:
        Tuple of (total_gb, available_gb, used_percent)
        Returns (0, 0, 0) if psutil is not available.
    """
    if not PSUTIL_AVAILABLE:
        return 0.0, 0.0, 0.0

    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        used_percent = mem.percent
        return total_gb, available_gb, used_percent
    except Exception:
        return 0.0, 0.0, 0.0


def get_memory_based_max_streams(
    is_igpu: bool = True,
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Calculate maximum streams based on available system memory.

    This function helps prevent OOM by limiting stream count based on
    available RAM. iGPU platforms are more memory-constrained since
    the GPU shares system RAM.

    Args:
        is_igpu: True if running on iGPU (shared memory), False for dGPU.
        logger: Optional logger for debug output.

    Returns:
        Maximum recommended stream count based on available memory.
        Returns DEFAULT_MAX_STREAMS_UNLIMITED if memory info unavailable.
    """
    if not PSUTIL_AVAILABLE:
        return DEFAULT_MAX_STREAMS_UNLIMITED

    total_gb, available_gb, used_percent = get_memory_info()

    if total_gb == 0:
        return DEFAULT_MAX_STREAMS_UNLIMITED

    # Select streams-per-GB ratio based on GPU type
    streams_per_gb = STREAMS_PER_GB_IGPU if is_igpu else STREAMS_PER_GB_DGPU

    # Calculate max streams based on available memory with safety margin
    usable_memory_gb = available_gb * MEMORY_SAFETY_MARGIN
    max_streams = int(usable_memory_gb * streams_per_gb)

    # Ensure at least 1 stream
    max_streams = max(1, max_streams)

    if logger:
        gpu_type = "iGPU" if is_igpu else "dGPU"
        logger.info(
            f"[MEMORY] System RAM: {total_gb:.1f}GB total, {available_gb:.1f}GB available ({used_percent:.1f}% used)"
        )
        logger.info(
            f"[MEMORY] {gpu_type} memory-based limit: {max_streams} streams "
            f"(using {streams_per_gb} streams/GB, {MEMORY_SAFETY_MARGIN * 100:.0f}% safety margin)"
        )

    return max_streams


def get_concurrent_mode_max_streams(
    base_max_streams: int,
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Apply an additional conservative cap for concurrent GPU+NPU modes.

    Concurrent modes run two pipelines at once and have higher memory pressure
    than split/single-device modes, so this function reduces the base stream
    limit returned by get_memory_based_max_streams().

    Args:
        base_max_streams: Base stream limit from memory-based calculation.
        logger: Optional logger for debug output.

    Returns:
        Conservative stream cap for concurrent execution.
    """
    if base_max_streams <= 0:
        return 1

    concurrent_cap = max(2, int(base_max_streams * CONCURRENT_MODE_STREAM_FACTOR))
    concurrent_cap = min(concurrent_cap, base_max_streams)

    if logger:
        logger.info(
            f"[MEMORY] Concurrent mode cap: {concurrent_cap} streams "
            f"(base={base_max_streams}, factor={CONCURRENT_MODE_STREAM_FACTOR})"
        )

    return concurrent_cap


def check_available_memory(
    min_available_gb: float = MIN_AVAILABLE_MEMORY_GB,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Check if there's sufficient available memory to continue scaling.

    Args:
        min_available_gb: Minimum available memory in GB required.
        logger: Optional logger for warnings.

    Returns:
        True if sufficient memory available, False otherwise.
    """
    if not PSUTIL_AVAILABLE:
        return True  # Assume OK if we can't check

    _, available_gb, used_percent = get_memory_info()

    if available_gb < min_available_gb:
        if logger:
            logger.warning(
                f"[MEMORY] Low memory warning: {available_gb:.1f}GB available "
                f"(minimum: {min_available_gb}GB, {used_percent:.1f}% used)"
            )
        return False

    return True


def log_memory_status(logger: logging.Logger, prefix: str = "") -> None:
    """
    Log current memory status for monitoring.

    Args:
        logger: Logger to use for output.
        prefix: Optional prefix for log message.
    """
    if not PSUTIL_AVAILABLE:
        logger.debug(f"{prefix}[MEMORY] psutil not available, cannot monitor memory")
        return

    total_gb, available_gb, used_percent = get_memory_info()
    logger.info(
        f"{prefix}[MEMORY] RAM: {available_gb:.1f}GB available / {total_gb:.1f}GB total ({used_percent:.1f}% used)"
    )
