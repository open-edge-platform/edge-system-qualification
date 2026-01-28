# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
FPS-Guided Adaptive Search Algorithm for optimal stream count discovery.

This module implements an intelligent search algorithm that uses real-time FPS
measurements to efficiently find the maximum number of concurrent streams that
can meet a target FPS threshold. The algorithm significantly reduces iterations
compared to linear or simple binary search approaches.

Key Features:
- Phase 1: Exponential probe using FPS headroom calculation
- Phase 2: Binary search refinement for precise boundary finding
- Reference-based initialization for faster convergence
- Configurable safety factors and jump parameters
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, List

# Default configuration values
DEFAULT_SAFETY_FACTOR = 0.8
DEFAULT_MIN_JUMP = 2
DEFAULT_MIN_START = 5  # Skip GPU warmup zone
DEFAULT_MAX_STREAMS_UNLIMITED = 100  # Default max when -1 (unlimited) is specified

@dataclass
class AdaptiveSearchConfig:
    """Configuration for FPS-Guided Adaptive Search algorithm."""

    safety_factor: float = DEFAULT_SAFETY_FACTOR
    """Jump multiplier (0.0-1.0). Lower values are more conservative."""

    min_jump: int = DEFAULT_MIN_JUMP
    """Minimum jump size to avoid tiny increments near the limit."""

    min_start: int = DEFAULT_MIN_START
    """Minimum starting stream count to skip GPU warmup zone."""

    enable_confirmation: bool = True
    """Enable confirmation check for failures in low stream counts."""

    confirmation_threshold: int = 15
    """Stream count below which confirmation check is triggered."""

    confirmation_offset: int = 3
    """Number of streams to jump for confirmation check."""


@dataclass
class SearchMetadata:
    """Metadata from adaptive search execution for logging and debugging."""

    iterations: int = 0
    """Total number of pipeline executions."""

    phase1_iterations: int = 0
    """Number of iterations in exponential probe phase."""

    phase2_iterations: int = 0
    """Number of iterations in binary refinement phase."""

    search_path: List[Dict] = field(default_factory=list)
    """Detailed log of each search step."""

    total_time: float = 0.0
    """Total search time in seconds."""

    initial_count: int = 1
    """Starting stream count used."""

    optimal_count: int = 0
    """Final optimal stream count found."""

    final_fps: float = 0.0
    """FPS achieved at optimal stream count."""


def calculate_adaptive_jump(
    current: int,
    headroom: float,
    safety_factor: float,
    min_jump: int,
) -> int:
    """
    Calculate jump size based on FPS headroom.

    The algorithm uses different strategies based on the headroom ratio:
    - headroom >= 3.0: Aggressive jump (GPU has lots of capacity)
    - headroom >= 1.5: Moderate growth
    - headroom >= 1.2: Conservative growth
    - headroom < 1.2: Minimal step (near the limit)

    Args:
        current: Current stream count being tested.
        headroom: Ratio of measured FPS to target FPS (FPS / target).
        safety_factor: Jump multiplier (0.0-1.0) for conservative estimation.
        min_jump: Minimum jump size.

    Returns:
        Calculated jump size (always >= min_jump).
    """
    if headroom >= 3.0:
        # Lots of headroom - jump aggressively
        # Theory: if 1 stream gives 60 FPS (3x headroom over 20 FPS target),
        # we can likely handle ~3 streams
        theoretical_max = int(current * headroom)
        jump = int((theoretical_max - current) * safety_factor)

    elif headroom >= 1.5:
        # Moderate headroom - grow by 50-75% of current
        jump = int(current * (headroom - 1) * safety_factor)

    elif headroom >= 1.2:
        # Small headroom - grow by 10-30%
        jump = max(min_jump, int(current * 0.2))

    else:
        # Very close to limit - single step
        jump = min_jump

    return max(jump, min_jump)


def get_initial_stream_count(
    max_streams: int,
    min_start: int = DEFAULT_MIN_START,
    reference_streams: Optional[int] = None,
    previous_model_streams: Optional[int] = None,
) -> int:
    """
    Determine smart starting point based on available reference data.

    Priority order:
    1. Previous model result (same test, lighter model completed) - use 50%
    2. Reference data (known good value for this platform) - use 75%
    3. Minimum start value (skip GPU warmup zone)

    Args:
        max_streams: Maximum possible stream count.
        min_start: Minimum starting point to skip warmup zone.
        reference_streams: Reference stream count from platform data.
        previous_model_streams: Result from previous model in same test.

    Returns:
        Starting stream count for search.
    """
    # Handle unlimited mode (max_streams <= 0)
    effective_max = max_streams if max_streams > 0 else DEFAULT_MAX_STREAMS_UNLIMITED

    if previous_model_streams is not None and previous_model_streams > 0:
        # Heavier model = fewer streams, start at 50% of lighter model result
        initial = max(min_start, int(previous_model_streams * 0.5))
        return min(initial, effective_max)

    if reference_streams is not None and reference_streams > 0:
        # Start at 75% of reference (account for platform variance)
        initial = max(min_start, int(reference_streams * 0.75))
        return min(initial, effective_max)

    # Default: use minimum start value
    return min(min_start, effective_max)


def fps_guided_adaptive_search(
    run_pipeline_func: Callable[[int], Tuple[float, int]],
    max_streams: int,
    target_fps: float,
    initial_count: int = 1,
    config: Optional[AdaptiveSearchConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, SearchMetadata]:
    """
    FPS-Guided Adaptive Search Algorithm.

    Efficiently finds the maximum stream count meeting target FPS using a
    two-phase approach:

    Phase 1 (Exponential Probe):
        - Start at initial_count
        - Measure FPS and calculate headroom (FPS / target)
        - Jump forward based on headroom: more headroom = bigger jump
        - Continue until FPS drops below target or hit max_streams

    Phase 2 (Binary Refinement):
        - Use standard binary search between last passing and first failing
        - Narrow down to exact optimal stream count

    Args:
        run_pipeline_func: Function that runs pipeline with N streams.
            Signature: (stream_count: int) -> Tuple[avg_fps: float, status: int]
            Returns (fps, 0) on success, (fps, 1) on failure.
        max_streams: Maximum possible stream count (e.g., compose grid size).
        target_fps: Minimum acceptable FPS threshold.
        initial_count: Starting stream count (default 1, can use reference).
        config: Search configuration. Uses defaults if None.
        logger: Optional logger for debug output.

    Returns:
        Tuple of (optimal_stream_count, search_metadata).
        optimal_stream_count is the maximum streams meeting target_fps.
    """
    if config is None:
        config = AdaptiveSearchConfig()

    if logger is None:
        logger = logging.getLogger(__name__)

    # Handle unlimited mode (max_streams <= 0)
    effective_max = max_streams if max_streams > 0 else DEFAULT_MAX_STREAMS_UNLIMITED
    if max_streams <= 0:
        logger.info(
            f"[ADAPTIVE] Unlimited mode detected (max_streams={max_streams}), "
            f"using default max={effective_max}"
        )

    # Initialize metadata
    metadata = SearchMetadata(initial_count=initial_count)
    start_time = time.time()

    # Initialize search bounds
    low = 0  # Last known good stream count
    high = effective_max + 1  # First known bad stream count
    current = max(initial_count, config.min_start)
    best_fps = 0.0

    logger.info(
        f"[ADAPTIVE] Starting search: target={target_fps} FPS, max={effective_max}, "
        f"initial={current}, safety={config.safety_factor}"
    )

    # =========================================
    # PHASE 1: Exponential Probe with FPS Guide
    # =========================================

    while current <= effective_max:
        metadata.iterations += 1
        metadata.phase1_iterations += 1

        fps, status = run_pipeline_func(current)

        step_info = {
            "phase": 1,
            "streams": current,
            "fps": fps,
            "status": status,
            "passed": fps >= target_fps and status == 0,
        }
        metadata.search_path.append(step_info)

        headroom = fps / target_fps if target_fps > 0 else 0
        logger.info(
            f"[ADAPTIVE] Phase 1: {current} streams -> {fps:.1f} FPS "
            f"(target: {target_fps}, headroom: {headroom:.2f}x, status: {status})"
        )

        if status != 0 or fps < target_fps:
            # Failed - but check if this might be a warmup issue on dGPU
            if (
                config.enable_confirmation
                and current < config.confirmation_threshold
                and fps > 0
            ):
                # Confirmation check: try a few streams higher to verify
                # GPU might perform better with more streams (batching efficiency)
                confirm_count = current + config.confirmation_offset
                if confirm_count <= effective_max:
                    logger.info(
                        f"[ADAPTIVE] Confirmation check: trying {confirm_count} streams "
                        f"(current {current} may be in GPU warmup zone)"
                    )
                    confirm_fps, confirm_status = run_pipeline_func(confirm_count)
                    metadata.iterations += 1
                    metadata.phase1_iterations += 1
                    metadata.search_path.append({
                        "phase": 1,
                        "streams": confirm_count,
                        "fps": confirm_fps,
                        "status": confirm_status,
                        "passed": confirm_fps >= target_fps and confirm_status == 0,
                        "type": "confirmation",
                    })

                    if confirm_fps >= target_fps and confirm_status == 0:
                        # Confirmation passed - we were in warmup zone
                        logger.info(
                            f"[ADAPTIVE] Confirmation passed: {confirm_count} streams "
                            f"-> {confirm_fps:.1f} FPS. Continuing from here."
                        )
                        low = confirm_count
                        best_fps = confirm_fps
                        current = confirm_count
                        # Continue with next jump
                        headroom = confirm_fps / target_fps
                        jump = calculate_adaptive_jump(
                            current, headroom, config.safety_factor, config.min_jump
                        )
                        current = min(current + jump, effective_max)
                        continue

            # Set upper bound and move to Phase 2
            high = current
            break
        else:
            # Passed - update lower bound
            low = current
            best_fps = fps

            # Check if we've hit maximum
            if current >= effective_max:
                logger.info(f"[ADAPTIVE] Reached max streams ({effective_max}), stopping")
                break

            # Calculate FPS-guided jump
            jump = calculate_adaptive_jump(
                current, headroom, config.safety_factor, config.min_jump
            )
            next_count = min(current + jump, effective_max)

            logger.debug(
                f"[ADAPTIVE] Headroom {headroom:.2f}x -> Jump {jump} "
                f"({current} -> {next_count})"
            )

            if next_count == current:
                # Can't jump anymore, try one more
                next_count = current + 1
                if next_count > effective_max:
                    break

            current = next_count

    # =========================================
    # PHASE 2: Binary Search Refinement
    # =========================================

    logger.info(
        f"[ADAPTIVE] Phase 2: Binary refinement between {low} and {high}"
    )

    while high - low > 1:
        metadata.iterations += 1
        metadata.phase2_iterations += 1

        mid = (low + high) // 2
        fps, status = run_pipeline_func(mid)

        step_info = {
            "phase": 2,
            "streams": mid,
            "fps": fps,
            "status": status,
            "passed": fps >= target_fps and status == 0,
        }
        metadata.search_path.append(step_info)

        logger.info(
            f"[ADAPTIVE] Phase 2: {mid} streams -> {fps:.1f} FPS "
            f"(binary search: low={low}, high={high})"
        )

        if fps >= target_fps and status == 0:
            low = mid
            best_fps = fps
        else:
            high = mid

    # Finalize metadata
    metadata.total_time = time.time() - start_time
    metadata.optimal_count = low
    metadata.final_fps = best_fps

    logger.info(
        f"[ADAPTIVE] Search complete: {low} streams @ {best_fps:.1f} FPS in "
        f"{metadata.iterations} iterations "
        f"(Phase1: {metadata.phase1_iterations}, Phase2: {metadata.phase2_iterations}, "
        f"Time: {metadata.total_time:.1f}s)"
    )

    return low, metadata


def run_adaptive_search(
    run_pipeline_func: Callable[[int], Tuple[float, int]],
    max_streams: int,
    target_fps: float,
    reference_streams: Optional[int] = None,
    previous_model_streams: Optional[int] = None,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    min_jump: int = DEFAULT_MIN_JUMP,
    min_start: int = DEFAULT_MIN_START,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, SearchMetadata]:
    """
    Convenience wrapper for FPS-Guided Adaptive Search.

    Combines initial stream count calculation with the search algorithm.

    Args:
        run_pipeline_func: Function that runs pipeline with N streams.
        max_streams: Maximum possible stream count.
        target_fps: Minimum acceptable FPS threshold.
        reference_streams: Reference stream count from platform data.
        previous_model_streams: Result from previous model in same test.
        safety_factor: Jump multiplier (0.0-1.0).
        min_jump: Minimum jump size.
        min_start: Minimum starting point to skip warmup zone.
        logger: Optional logger for debug output.

    Returns:
        Tuple of (optimal_stream_count, search_metadata).
    """
    config = AdaptiveSearchConfig(
        safety_factor=safety_factor,
        min_jump=min_jump,
        min_start=min_start,
    )

    initial_count = get_initial_stream_count(
        max_streams=max_streams,
        min_start=min_start,
        reference_streams=reference_streams,
        previous_model_streams=previous_model_streams,
    )

    return fps_guided_adaptive_search(
        run_pipeline_func=run_pipeline_func,
        max_streams=max_streams,
        target_fps=target_fps,
        initial_count=initial_count,
        config=config,
        logger=logger,
    )
