# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CPU utilities sub-package.

Provides a unified namespace for all CPU-related utilities.  Importing from
``sysagent.utils.system.cpu`` is fully backward-compatible with the previous
flat ``cpu.py`` module.

Sub-modules
-----------
generation
    Intel CPU generation and segment detection (large CPUID map, brand
    string parsing).  This was the original ``cpu.py``.

topology
    Hybrid core-type detection (P-core / E-core / LP E-core) via sysfs
    ``core_type``, ``lscpu``, or L2+L3 cache-sharing heuristics.
    Also provides ``parse_cpu_list`` / ``compact_cpu_list`` helpers.

rdt
    Intel RDT cache partition configuration reader via MSR registers
    (``rdmsr``).  Reads per-CPU CLOS assignments and L3/L2 CAT bitmasks
    without requiring the resctrl filesystem to be mounted.

cpuidle
    CPU idle state (C-state) data from the Linux cpuidle sysfs tree.
    ``collect_cpuidle_info()`` returns per-CPU per-state rows (with
    ``idle_time_pct``) and an aggregate summary dict; integrated into
    ``collect_cpu_info()`` so C-state data appears in the hardware JSON.

info
    Main ``collect_cpu_info()`` orchestrator plus socket-count and
    feature-flag helpers.  Depends on all other sub-modules.

Planned additions
-----------------
cache
    CPU cache topology (level, size, associativity, sharing) — to be
    populated from ``/sys/devices/system/cpu/cpuN/cache/``.
"""

# ── generation (original cpu.py content) ──────────────────────────────────────
# ── cpuidle ──────────────────────────────────────────────────────────────────
# ── cpufreq ──────────────────────────────────────────────────────────────────
from .cpufreq import collect_cpufreq_info
from .cpuidle import collect_cpuidle_info, derive_cpuidle_summary
from .generation import (
    CPU_GENERATION_MAP,
    SEGMENT_PATTERNS,
    compare_generations,
    detect_cpu_generation_and_segment,
    is_generation_supported,
    match_cpu_generations,
    normalize_generation_string,
)

# ── info ───────────────────────────────────────────────────────────────────────
from .info import collect_cpu_info

# ── rdt ────────────────────────────────────────────────────────────────────────
from .rdt import collect_cache_partition_info

# ── topology ───────────────────────────────────────────────────────────────────
from .topology import (
    collect_cpu_core_types,
    compact_cpu_list,
    parse_cpu_list,
)

__all__ = [
    # generation
    "CPU_GENERATION_MAP",
    "SEGMENT_PATTERNS",
    "compare_generations",
    "detect_cpu_generation_and_segment",
    "is_generation_supported",
    "match_cpu_generations",
    "normalize_generation_string",
    # topology
    "collect_cpu_core_types",
    "compact_cpu_list",
    "parse_cpu_list",
    # rdt
    "collect_rdt_info",
    # cpuidle
    "collect_cpuidle_info",
    "derive_cpuidle_summary",
    # cpufreq
    "collect_cpufreq_info",
    # info
    "collect_cpu_info",
]
