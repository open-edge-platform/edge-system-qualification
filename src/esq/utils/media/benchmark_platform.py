# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark platform identification and matching utilities for media test suites.

Provides functions for identifying system platform characteristics (CPU model, memory)
and matching against reference benchmark platforms with configurable scoring algorithms.
Used by media and AI test suites for platform-aware reference benchmark selection.
"""

import logging
import re
from typing import Optional, Tuple

from sysagent.utils.system.hardware import collect_cpu_info, collect_memory_info

logger = logging.getLogger(__name__)


def get_platform_identifier() -> Tuple[str, int]:
    """
    Get system platform identifier for reference benchmark matching.

    Returns:
        Tuple of (cpu_model, memory_gb) where:
        - cpu_model: Simplified CPU model string (e.g., "i7-1360P", "MTL 165H", "N97", "Xeon(R) Gold 6430")
        - memory_gb: Total system memory in GB (rounded)
    """
    try:
        # Collect CPU and memory information using existing framework utilities
        cpu_info = collect_cpu_info()
        memory_info = collect_memory_info()

        # Extract CPU brand
        cpu_brand = cpu_info.get("brand", "Unknown")

        # Simplify CPU model name for matching
        # Examples:
        #   "Intel(R) Core(TM) i7-1360P @ 2.20GHz" -> "i7-1360P"
        #   "Intel(R) Core(TM) Ultra 7 165H" -> "MTL 165H"
        #   "Intel(R) N97 @ 2.00GHz" -> "N97"
        #   "Intel(R) Xeon(R) Gold 6430 @ 2.10GHz" -> "Xeon(R) Gold 6430"
        cpu_model = "Unknown"

        # Try to extract processor model using regex patterns
        # Pattern 1: Core i7-xxxx, i5-xxxx, etc.
        match = re.search(r"(?:Core\(TM\)|Core\(R\))\s+(i\d+-\w+)", cpu_brand)
        if match:
            cpu_model = match.group(1)
        else:
            # Pattern 2: Ultra X XXXH (Meteor Lake naming)
            match = re.search(r"Ultra\s+\d+\s+(\d+H)", cpu_brand)
            if match:
                cpu_model = f"MTL {match.group(1)}"
            else:
                # Pattern 3: N-series (N97, N100, etc.)
                match = re.search(r"(N\d+)", cpu_brand)
                if match:
                    cpu_model = match.group(1)
                else:
                    # Pattern 4: Xeon processors
                    match = re.search(r"(Xeon\(R\)\s+(?:Gold|Silver|Bronze|Platinum)\s+\w+)", cpu_brand)
                    if match:
                        cpu_model = match.group(1)
                    else:
                        # Pattern 5: Atom processors
                        match = re.search(r"Atom\(TM\)\s+(\w+)", cpu_brand)
                        if match:
                            cpu_model = f"Atom {match.group(1)}"
                        else:
                            # Pattern 6: Celeron/Pentium
                            match = re.search(r"(Celeron|Pentium)\(R\)\s+(\w+)", cpu_brand)
                            if match:
                                cpu_model = f"{match.group(1)} {match.group(2)}"
                            else:
                                # Fallback: use full brand
                                cpu_model = cpu_brand

        # Get total memory in GB (rounded)
        total_memory_bytes = memory_info.get("total", 0)
        memory_gb = round(total_memory_bytes / (1024**3))  # Convert bytes to GB

        logger.debug(f"System platform identified: CPU={cpu_model}, Memory={memory_gb}GB")

        return (cpu_model, memory_gb)

    except Exception as e:
        logger.warning(f"Failed to get system platform identifier: {e}")
        return ("Unknown", 0)


def match_platform(
    system_cpu: str,
    system_memory_gb: int,
    ref_platform: str,
    device: Optional[str] = None,
    system_vdbox: Optional[int] = None,
    ref_vdbox: Optional[int] = None,
) -> int:
    """
    Calculate match score between system platform and reference platform.

    Performs CPU family matching with memory proximity scoring, plus optional
    VD box validation for GPU devices (secondary scoring boost).

    Args:
        system_cpu: System CPU model (e.g., "i7-1360P")
        system_memory_gb: System memory in GB (e.g., 16)
        ref_platform: Reference platform string (e.g., "i7-1360P (16G Mem)", "Arc A380")
        device: Device being tested (e.g., "GPU.0", "CPU") - optional for VD validation
        system_vdbox: System VD box count - optional for GPU topology validation
        ref_vdbox: Reference VD box count - optional for GPU topology validation

    Returns:
        Match score (higher is better):
        Base scoring:
        - 100: Exact match (CPU model and memory)
        - 50: Exact CPU match, memory close (within 25%)
        - 10: Exact CPU match, memory different
        - 8: Same CPU family (e.g., i5 to i5), memory match
        - 6: Same CPU family, memory close (within 25%)
        - 3: Same CPU family, memory different
        - 5: Partial CPU match (substring)
        - 0: No match

        VD box validation bonus (only for GPU devices):
        - +25: VD box count matches (confirms GPU topology)
        - -10: VD box count mismatches (different GPU generation)
        - 0: VD box info not available (no penalty)
    """
    if not ref_platform or system_cpu == "Unknown":
        return 0

    # Extract CPU and memory from reference platform
    # Format examples: "i7-1360P (16G Mem)", "MTL 165H (32G Mem)", "Arc A380"
    ref_cpu = ref_platform
    ref_memory_gb = 0

    # Try to extract memory from reference platform
    memory_match = re.search(r"\((\d+)G\s+Mem\)", ref_platform)
    if memory_match:
        ref_memory_gb = int(memory_match.group(1))
        # Extract CPU part (before memory)
        ref_cpu = ref_platform.split("(")[0].strip()

    # Extract CPU family for family-level matching
    # Examples: "i5-12400" -> "i5", "i7-1360P" -> "i7", "N97" -> "N", "Xeon(R) Gold 6430" -> "Xeon"
    system_family = None
    ref_family = None
    system_family_fallback = None  # For i9->i7 fallback

    # Pattern 1: Core i5/i7/i9 series
    match = re.search(r"(i\d+)", system_cpu, re.IGNORECASE)
    if match:
        system_family = match.group(1).lower()
        # i9 can fall back to i7 if no i9 references exist in CSV
        if system_family == "i9":
            system_family_fallback = "i7"
    else:
        # Pattern 2: N-series (N97, N100, etc.)
        match = re.search(r"^(N)\d+", system_cpu, re.IGNORECASE)
        if match:
            system_family = "n"
        else:
            # Pattern 3: Xeon series
            if "xeon" in system_cpu.lower():
                system_family = "xeon"

    # Extract reference CPU family
    match = re.search(r"(i\d+)", ref_cpu, re.IGNORECASE)
    if match:
        ref_family = match.group(1).lower()
    else:
        match = re.search(r"^(N)\d+", ref_cpu, re.IGNORECASE)
        if match:
            ref_family = "n"
        else:
            if "xeon" in ref_cpu.lower():
                ref_family = "xeon"

    # Calculate match score
    score = 0

    # Priority 1: Exact CPU model match
    if system_cpu.lower() == ref_cpu.lower():
        if ref_memory_gb > 0 and system_memory_gb > 0:
            memory_diff_percent = abs(system_memory_gb - ref_memory_gb) / ref_memory_gb
            if memory_diff_percent < 0.01:  # Exact memory match
                score = 100
            elif memory_diff_percent < 0.25:  # Memory within 25%
                score = 50
            else:
                score = 10
        else:
            score = 10  # CPU match but no memory info

    # Priority 2: CPU family match (e.g., i5 to i5)
    elif system_family and ref_family and system_family == ref_family:
        if ref_memory_gb > 0 and system_memory_gb > 0:
            memory_diff_percent = abs(system_memory_gb - ref_memory_gb) / ref_memory_gb
            if memory_diff_percent < 0.01:  # Exact memory match
                score = 8
            elif memory_diff_percent < 0.25:  # Memory within 25%
                score = 6
            else:
                score = 3
        else:
            score = 3  # Family match but no memory info

    # Priority 3: Fallback family match (e.g., i9 -> i7 when no i9 references exist)
    elif system_family_fallback and ref_family and system_family_fallback == ref_family:
        score = 2  # Fallback family match (lower priority than exact family)

        # Boost score if memory also matches
        if ref_memory_gb > 0 and system_memory_gb > 0:
            memory_diff_percent = abs(system_memory_gb - ref_memory_gb) / ref_memory_gb
            if memory_diff_percent < 0.01:  # Exact memory match
                score = 7
            elif memory_diff_percent < 0.25:  # Memory within 25%
                score = 5

    # Fallback: substring match
    elif system_cpu.lower() in ref_cpu.lower() or ref_cpu.lower() in system_cpu.lower():
        score = 1  # Lowest priority - substring match
    else:
        score = 0  # No match

    # SECONDARY VALIDATION: VD box topology check (only for GPU devices)
    if device and "GPU" in device and system_vdbox is not None and ref_vdbox is not None:
        if system_vdbox == ref_vdbox:
            score += 25  # Bonus for matching GPU topology
            logger.debug(f"VD box match bonus: {system_vdbox} VD boxes confirmed (+25)")
        else:
            score -= 10  # Penalty for mismatched GPU topology
            logger.warning(
                f"VD box mismatch: system={system_vdbox}, reference={ref_vdbox} (-10). "
                f"System may have different GPU generation than reference."
            )

    return score
