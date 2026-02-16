# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CPU generation and segment detection for Intel processors.

This module provides utilities to detect Intel CPU generation (Series 1, 2, 3, etc.)
and vertical segment (server, desktop, mobile, embedded, workstation) based on CPU
family, model, stepping, and brand string.

Important:
- Vertical segments represent Intel's market categories (server, desktop, mobile, etc.)
- These are NOT the same as profile tiers (entry, mainstream, efficiency_optimized, etc.)
- Profile tier compatibility should be determined by system capabilities, not CPU segment alone

Intel Naming Convention Updates (2024-2025):
- Xeon: Changed from "Xeon Scalable" to "Xeon 6" for 6th gen and beyond (April 2024)
  - Sierra Forest (E-cores): "Xeon 6" not "6th Gen Xeon Scalable"
  - Granite Rapids (P-cores): "Xeon 6" not "6th Gen Xeon Scalable"
  - Sapphire Rapids HBM: "Xeon Max Series" (still 4th Gen Xeon Scalable)
- Core: Two distinct product lines in Series 2
  - "Intel Core" (Series 2): Raptor Lake-based (Core 7/5 - 250U, 220U, etc.)
  - "Intel Core Ultra" (Series 2): Arrow Lake/Lunar Lake (Core Ultra 7/5 - 265U, 235U, etc.)

Data Sources:
- Intel CPU model numbers: InstLatx64 CPUID database (https://github.com/InstLatx64/InstLatx64)
- Wikipedia: Intel processor documentation
  - https://en.wikipedia.org/wiki/List_of_Intel_CPU_microarchitectures
  - https://en.wikipedia.org/wiki/Xeon (Xeon 6 branding change April 2024)
  - https://en.wikipedia.org/wiki/Sierra_Forest
  - https://en.wikipedia.org/wiki/Granite_Rapids
- Model numbers in decimal format (InstLatx64 uses hexadecimal)
  Example: 0xB0650 (hex) = 181 (decimal) = Arrow Lake-U
"""

import logging
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# Intel CPU generation mapping based on family, model, and stepping
# Format: (family, model, stepping_range): (codename, generation, segment_hint)
# generation: Actual Intel generation string (e.g., "Core Ultra (Series 2)", "12th Gen Core")
# segment_hint: Market segment hint (server, desktop, mobile, embedded, entry)
CPU_GENERATION_MAP = {
    # Panther Lake (PTL) - Core Ultra Series 3 - Mobile only (H-series and low-power variants)
    # Models: Core Ultra X9 388H/386H, X7 368H/358H, 7 366H/356H, 5 338H/336H (H-series)
    #         Core Ultra 7 365/355, 5 335/332/325/322 (low-power variants)
    (6, 204, range(0, 10)): ("Panther Lake", "Core Ultra (Series 3)", "mobile"),
    # Lunar Lake (LNL) - Core Ultra Series 2
    # InstLatx64: 0xB06D1 = 182 decimal
    (6, 182, range(0, 10)): ("Lunar Lake", "Core Ultra (Series 2)", "mobile"),
    # Arrow Lake-S (ARL-S) - Core Ultra Series 2
    # InstLatx64: 0xC0662 = 198 decimal
    (6, 198, range(0, 10)): ("Arrow Lake-S", "Core Ultra (Series 2)", "desktop"),
    # Arrow Lake-H (ARL-H) - Core Ultra Series 2
    (6, 197, range(0, 10)): ("Arrow Lake-H", "Core Ultra (Series 2)", "mobile"),
    # Arrow Lake-U (ARL-U) - Core Ultra Series 2
    # InstLatx64: GenuineIntel00B0650_ArrowLakeU_03_CPUID.txt (0xB0650 = 181 decimal)
    (6, 181, range(0, 10)): ("Arrow Lake-U", "Core Ultra (Series 2)", "mobile"),
    # Bartlett Lake-S (BTL-S) - 14th Gen Core
    (6, 183, range(4, 10)): ("Bartlett Lake-S", "14th Gen Core", "desktop"),
    # Meteor Lake (MTL-H/U) - Core Ultra Series 1
    # InstLatx64: 0xA06A4 = 170 decimal (covers both MTL-H and MTL-U)
    (6, 170, range(0, 10)): ("Meteor Lake-H", "Core Ultra (Series 1)", "mobile"),
    (6, 172, range(0, 10)): ("Meteor Lake-U", "Core Ultra (Series 1)", "mobile"),
    # Raptor Lake (RPL) - 13th Gen Core AND Raptor Lake Refresh (RPL-S Refresh) - 14th Gen Core
    # IMPORTANT: Model 183 is shared between 13th gen (RPL) and 14th gen (RPL Refresh)
    # Differentiation is done by brand string detection in _detect_generation_from_brand()
    # - 13th gen: i[3579]-13xxx pattern (e.g., i9-13900K)
    # - 14th gen: i[3579]-14xxx pattern (e.g., i9-14900T, i7-14700K)
    # Default to 13th gen here; brand string detection will override to 14th gen if needed
    (6, 183, range(0, 4)): ("Raptor Lake-S", "13th Gen Core", "desktop"),
    (6, 191, range(0, 10)): ("Raptor Lake-P", "13th Gen Core", "mobile"),
    (6, 186, range(0, 10)): ("Raptor Lake-HX", "13th Gen Core", "mobile"),
    # Alder Lake (ADL) - 12th Gen Core
    (6, 151, range(0, 10)): ("Alder Lake-S", "12th Gen Core", "desktop"),
    (6, 154, range(0, 10)): ("Alder Lake-P", "12th Gen Core", "mobile"),
    (6, 190, range(0, 10)): ("Alder Lake-N", "Intel N-series", "mobile"),  # Alder Lake-N / Twin Lake
    # Tiger Lake (TGL) - 11th Gen Core
    (6, 140, range(0, 10)): ("Tiger Lake", "11th Gen Core", "mobile"),
    (6, 141, range(0, 10)): ("Tiger Lake-H", "11th Gen Core", "mobile"),
    # Rocket Lake (RKL) - 11th Gen Core
    (6, 167, range(0, 10)): ("Rocket Lake-S", "11th Gen Core", "desktop"),
    # Ice Lake (ICL) - 10th Gen Core
    (6, 125, range(0, 10)): ("Ice Lake-U", "10th Gen Core", "mobile"),
    (6, 126, range(0, 10)): ("Ice Lake-Y", "10th Gen Core", "mobile"),
    # Comet Lake (CML) - 10th Gen Core
    (6, 165, range(0, 10)): ("Comet Lake-S", "10th Gen Core", "desktop"),
    (6, 166, range(0, 10)): ("Comet Lake-U", "10th Gen Core", "mobile"),
    # Coffee Lake (CFL) - 8th/9th Gen Core
    (6, 142, range(0, 10)): ("Coffee Lake-U", "8th Gen Core", "mobile"),
    (6, 158, range(0, 10)): ("Coffee Lake-S", "8th Gen Core", "desktop"),
    (6, 159, range(0, 10)): ("Coffee Lake-R", "9th Gen Core", "desktop"),  # Coffee Lake Refresh
    # Kaby Lake (KBL) - 7th Gen Core
    (6, 142, range(10, 20)): ("Kaby Lake-U", "7th Gen Core", "mobile"),  # Amber Lake, Whiskey Lake variants
    (6, 158, range(10, 20)): ("Kaby Lake-S", "7th Gen Core", "desktop"),
    (6, 78, range(0, 10)): ("Kaby Lake-G", "7th Gen Core", "mobile"),
    # Skylake (SKL) - 6th Gen Core
    (6, 78, range(10, 20)): ("Skylake-U", "6th Gen Core", "mobile"),
    (6, 94, range(0, 10)): ("Skylake-S", "6th Gen Core", "desktop"),
    (6, 85, range(0, 10)): ("Skylake-H", "6th Gen Core", "mobile"),
    # Broadwell (BDW) - 5th Gen Core
    (6, 61, range(0, 10)): ("Broadwell-U", "5th Gen Core", "mobile"),
    (6, 71, range(0, 10)): ("Broadwell-H", "5th Gen Core", "mobile"),
    (6, 86, range(0, 10)): ("Broadwell-DE", "5th Gen Core", "server"),
    (6, 87, range(0, 10)): ("Broadwell-DT", "5th Gen Core", "desktop"),
    # Haswell (HSW) - 4th Gen Core
    (6, 60, range(0, 10)): ("Haswell", "4th Gen Core", "desktop"),
    (6, 69, range(0, 10)): ("Haswell-MB", "4th Gen Core", "mobile"),
    (6, 70, range(0, 10)): ("Haswell-DT", "4th Gen Core", "desktop"),  # Devil's Canyon
    (6, 63, range(0, 10)): ("Haswell-E", "4th Gen Core", "desktop"),
    # Ivy Bridge (IVB) - 3rd Gen Core
    (6, 58, range(0, 10)): ("Ivy Bridge", "3rd Gen Core", "desktop"),
    (6, 62, range(0, 10)): ("Ivy Bridge-E", "3rd Gen Core", "desktop"),
    # Sandy Bridge (SNB) - 2nd Gen Core
    (6, 42, range(0, 10)): ("Sandy Bridge", "2nd Gen Core", "mobile"),
    (6, 45, range(0, 10)): ("Sandy Bridge", "2nd Gen Core", "desktop"),
    # Westmere (WSM) - 2nd Gen Core (32nm shrink of Nehalem)
    (6, 37, range(0, 10)): ("Westmere", "2nd Gen Core", "mobile"),  # Arrandale
    (6, 44, range(0, 10)): ("Westmere", "2nd Gen Core", "desktop"),  # Gulftown
    # Nehalem (NHM) - 1st Gen Core
    (6, 30, range(0, 10)): ("Nehalem", "1st Gen Core", "desktop"),  # Lynnfield, Bloomfield
    (6, 46, range(0, 10)): ("Nehalem-EX", "1st Gen Core", "server"),
    # Core 2 (Penryn) - Pre-i7/i5/i3 era (2007-2008)
    (6, 23, range(0, 10)): ("Penryn", "Core 2", "mobile"),
    (6, 29, range(0, 10)): ("Penryn", "Core 2", "desktop"),
    # Core 2 (Core) - Original Core microarchitecture (2006-2007)
    (6, 15, range(0, 10)): ("Conroe", "Core 2", "desktop"),  # Core 2 Duo/Quad
    (6, 22, range(0, 10)): ("Merom", "Core 2", "mobile"),  # Core 2 Duo mobile
    # Core (Yonah) - Enhanced Pentium M (2006)
    (6, 14, range(0, 10)): ("Yonah", "Core (Yonah)", "mobile"),  # Core Solo/Duo
    # Pentium Era - Pre-Core processors
    # Note: These are legacy processors; detection is basic codename-based
    # Pentium 4 (NetBurst): Family 15
    # Pentium M, Pentium III, Pentium II: Family 6 (various models)
    # For simplicity, any Family 15 or older Family 6 models not matched above
    # will be detected as generic "Pentium" by fallback logic
    # Intel Atom X-series (Embedded/IoT) - newer Amston Lake x7000 series
    # Note: Amston Lake (Gracemont) launched Q2 2024 - supports qualification
    # Model number TBD - will be added when CPUID data becomes available
    # (6, xxx, range(0, 10)): ("Amston Lake", "Atom x7000", "embedded"),
    # Intel Atom X-series (Embedded/IoT) - older unsupported series
    (6, 96, range(0, 10)): ("Elkhart Lake", "Atom x6000", "embedded"),
    (6, 92, range(0, 10)): ("Apollo Lake", "Atom x5000", "embedded"),
    (6, 76, range(0, 10)): ("Cherry Trail", "Atom (Cherry Trail)", "embedded"),
    (6, 55, range(0, 10)): ("Bay Trail", "Atom (Bay Trail)", "embedded"),
    # Intel Atom Z-series (Mobile/Tablet) - all unsupported legacy series
    (6, 74, range(0, 10)): ("Clover Trail", "Atom Z-series", "mobile"),
    (6, 77, range(0, 10)): ("Merrifield", "Atom Z-series", "mobile"),
    (6, 90, range(0, 10)): ("Moorefield", "Atom Z-series", "mobile"),
    # Xeon 6 - Model 173 shared by both Sierra Forest and Granite Rapids
    # Distinguished by suffix: E=Sierra Forest (E-cores), P=Granite Rapids (P-cores)
    # Sierra Forest (SRF) - Xeon 6 E-cores (launched June 2024)
    # Official name: "Intel Xeon 6" (6700E series: 6766E, 6740E, 6731E, 6710E, etc.)
    (6, 173, range(0, 10)): ("Sierra Forest", "Xeon 6", "server"),  # E-core default
    # Granite Rapids (GNR) - Xeon 6 P-cores (launched Sept 2024)
    # Official name: "Intel Xeon 6" (6900P/6700P/6500P series: 6980P, 6787P, etc.)
    # Note: Codename override handled in _detect_generation() based on P suffix
    # Xeon Emerald Rapids (EMR)
    (6, 207, range(0, 10)): ("Emerald Rapids", "5th Gen Xeon Scalable", "server"),
    # Xeon Sapphire Rapids (SPR) - Model 143 shared across variants:
    # - SPR-SP (Xeon Scalable server): Standard "Xeon Gold/Platinum/Silver" branding
    # - SPR-WS (Xeon W workstation): "Xeon w3/w5/w7/w9" pattern (detect_cpu_generation_and_segment)
    # - SPR-HBM (Xeon Max server): "Xeon Max" brand pattern (_detect_generation_from_brand)
    # Default classification for model 143 is server Xeon Scalable unless brand indicates otherwise
    (6, 143, range(8, 20)): ("Sapphire Rapids", "4th Gen Xeon Scalable", "server"),
    # Xeon Ice Lake-SP
    (6, 106, range(0, 10)): ("Ice Lake-SP", "3rd Gen Xeon Scalable", "server"),
}


def _get_cli_unsupported_generations():
    """
    Try to import CLI-specific UNSUPPORTED_GENERATIONS list.

    This allows ESQ (or other CLI packages) to define their own unsupported lists
    that override the default sysagent behavior. The function attempts to import
    from the CLI-specific run command module.

    Returns:
        List of unsupported generations if CLI-specific list exists, None otherwise
    """
    try:
        # Try to get current CLI context
        from sysagent.utils.config.config_loader import get_cli_aware_project_name

        cli_name = get_cli_aware_project_name().lower()

        # Skip import if running generic sysagent
        if cli_name == "sysagent":
            return None

        # Try to import CLI-specific unsupported list
        # Example: for "esq" CLI, import from esq.utils.cli.commands.run
        module_path = f"{cli_name}.utils.cli.commands.run"
        module = __import__(module_path, fromlist=["UNSUPPORTED_GENERATIONS"])

        if hasattr(module, "UNSUPPORTED_GENERATIONS"):
            unsupported_gens = getattr(module, "UNSUPPORTED_GENERATIONS")
            logger.debug(
                f"Using CLI-specific unsupported generations list from {module_path} ({len(unsupported_gens)} entries)"
            )
            return unsupported_gens

    except (ImportError, AttributeError) as e:
        logger.debug(f"CLI-specific unsupported list not available: {e}")
    except Exception as e:
        logger.debug(f"Error importing CLI-specific unsupported list: {e}")

    return None


def compare_generations(gen1: str, gen2: str) -> int:
    """
    Compare two Intel processor generations.

    Handles different naming schemes:
    - Core: "1st Gen Core" through "14th Gen Core"
    - Core Ultra: "Core Ultra (Series 1)", "Core Ultra (Series 2)", "Core Ultra (Series 3)"
    - Pre-Core i: "Core 2", "Core (Yonah)", "Pentium"
    - Xeon: "3rd Gen Xeon Scalable", "4th Gen Xeon Scalable", "Xeon 6"
    - Atom/Embedded: "Atom x6000", "Atom x7000", etc. or codenames
    - N-series: "Intel N-series"

    Timeline order (oldest to newest):
    Pentium < Core (Yonah) < Core 2 < 1st Gen Core < ... < 14th Gen Core <
    Core Ultra (Series 1) < Core Ultra (Series 2) < Core Ultra (Series 3)

    Args:
        gen1: First generation string
        gen2: Second generation string

    Returns:
        -1 if gen1 < gen2, 0 if equal, 1 if gen1 > gen2
        Returns 0 if comparison not possible (different families or unknown format)
    """
    # Normalize strings
    g1 = gen1.strip()
    g2 = gen2.strip()

    if g1 == g2:
        return 0

    # Core Series comparison (handles both "Core Ultra (Series X)" and "Core (Series X)")
    # Matches:
    # - "Core Ultra (Series 2)" - Arrow Lake/Lunar Lake (desktop/mobile)
    # - "Core (Series 2)" - Raptor Lake-based Core 7/5/3 (RPL-U/H Re-refresh mobile) + Bartlett Lake-S (embedded)
    # - "Core (Series 1)" - Raptor Lake-U Refresh (mobile only)
    #
    # IMPORTANT: "Core (Series X)" and "Xth Gen Core" are DIFFERENT product lines:
    # - "Core (Series 2)" uses new branding introduced in Q1 2025, targets mobile/embedded
    # - "14th Gen Core" uses traditional branding, targets desktop/workstation
    # They should NOT be treated as equivalent generations.
    #
    # Comparison strategy:
    # - Within "Core (Series X)" family: compare by series number
    # - "Core (Series X)" vs "Xth Gen Core": Cannot compare (different product lines)
    #   - Return 0 to indicate incomparable, let caller handle support decision
    series1_match = re.search(r"Core(?:\s+Ultra)?\s*\(Series\s+(\d+)\)", g1)
    series2_match = re.search(r"Core(?:\s+Ultra)?\s*\(Series\s+(\d+)\)", g2)

    # Traditional Core generation comparison ("8th Gen", "12th Gen", etc.)
    core_gen1_match = re.search(r"(\d+)(?:th|st|nd|rd)\s+Gen\s+Core", g1)
    core_gen2_match = re.search(r"(\d+)(?:th|st|nd|rd)\s+Gen\s+Core", g2)

    # Both are Core Series format
    if series1_match and series2_match:
        s1 = int(series1_match.group(1))
        s2 = int(series2_match.group(1))
        # When series numbers are equal, check if one is Ultra
        if s1 == s2:
            # Within same series number, Core Ultra > Core (non-Ultra)
            ultra1 = "Core Ultra" in g1
            ultra2 = "Core Ultra" in g2
            if ultra1 and not ultra2:
                return 1  # Core Ultra > Core
            elif not ultra1 and ultra2:
                return -1  # Core < Core Ultra
            else:
                return 0  # Both same (both Ultra or both non-Ultra)
        return -1 if s1 < s2 else 1

    # Cross-comparison: Core (Series X) vs traditional "Xth Gen Core"
    # These are different product lines introduced at different times
    # Cannot reliably compare - return 0 (incomparable)
    if series1_match and core_gen2_match:
        logger.debug(
            f"Cannot compare Core Series vs traditional Gen Core: '{g1}' vs '{g2}' "
            f"(different product lines - Series branding targets mobile/embedded, Gen branding targets desktop)"
        )
        return 0

    if core_gen1_match and series2_match:
        logger.debug(
            f"Cannot compare traditional Gen Core vs Core Series: '{g1}' vs '{g2}' "
            f"(different product lines - Gen branding targets desktop, Series branding targets mobile/embedded)"
        )
        return 0

    # Both are traditional Core generation format
    if core_gen1_match and core_gen2_match:
        gen1_num = int(core_gen1_match.group(1))
        gen2_num = int(core_gen2_match.group(1))
        return -1 if gen1_num < gen2_num else (1 if gen1_num > gen2_num else 0)

    # Pre-Core i comparison (Core 2, Core (Yonah), Pentium)
    # Timeline: Pentium < Core (Yonah) < Core 2 < 1st Gen Core
    # Assign virtual generation numbers for comparison:
    # Pentium = -3, Core (Yonah) = -2, Core 2 = -1, 1st Gen Core = 1
    pre_core_order = {
        "pentium": -3,
        "core (yonah)": -2,
        "core 2": -1,
    }

    g1_lower = g1.lower()
    g2_lower = g2.lower()

    # Check if either is a pre-Core i generation
    g1_pre = pre_core_order.get(g1_lower)
    g2_pre = pre_core_order.get(g2_lower)

    # Both are pre-Core i
    if g1_pre is not None and g2_pre is not None:
        return -1 if g1_pre < g2_pre else (1 if g1_pre > g2_pre else 0)

    # One is pre-Core i, other is numbered Gen Core
    if g1_pre is not None and core_gen2_match:
        # Pre-Core i vs numbered Gen Core: Pre-Core i is always older
        return -1
    if core_gen1_match and g2_pre is not None:
        # Numbered Gen Core vs pre-Core i: numbered Gen Core is always newer
        return 1

    # One is pre-Core i, other is Core Series/Core Ultra
    if g1_pre is not None and (series2_match or "Core Ultra" in g2):
        # Pre-Core i vs Core Series/Ultra: Pre-Core i is always older
        return -1
    if (series1_match or "Core Ultra" in g1) and g2_pre is not None:
        # Core Series/Ultra vs pre-Core i: Core Series/Ultra is always newer
        return 1

    # Xeon Scalable generation comparison (legacy naming, pre-2024)
    xeon1_match = re.search(r"(\d+)(?:th|st|nd|rd)\s+Gen\s+Xeon\s+Scalable", g1)
    xeon2_match = re.search(r"(\d+)(?:th|st|nd|rd)\s+Gen\s+Xeon\s+Scalable", g2)
    if xeon1_match and xeon2_match:
        xeon1_num = int(xeon1_match.group(1))
        xeon2_num = int(xeon2_match.group(1))
        return -1 if xeon1_num < xeon2_num else (1 if xeon1_num > xeon2_num else 0)

    # Xeon 6 vs Xeon Scalable comparison (new 2024 branding)
    # "Xeon 6" is considered equivalent to "6th Gen Xeon Scalable" or newer
    xeon6_1 = "xeon 6" in g1.lower()
    xeon6_2 = "xeon 6" in g2.lower()
    if xeon6_1 and xeon2_match:
        # Xeon 6 (6th gen) vs older Xeon Scalable
        return 1 if int(xeon2_match.group(1)) < 6 else 0
    if xeon1_match and xeon6_2:
        # Older Xeon Scalable vs Xeon 6 (6th gen)
        return -1 if int(xeon1_match.group(1)) < 6 else 0
    if xeon6_1 and xeon6_2:
        # Both are Xeon 6
        return 0

    # Atom series comparison ("Atom x6000", "Atom x7000", etc.)
    atom1_match = re.search(r"Atom x(\d+)", g1)
    atom2_match = re.search(r"Atom x(\d+)", g2)
    if atom1_match and atom2_match:
        atom1_num = int(atom1_match.group(1))
        atom2_num = int(atom2_match.group(1))
        return -1 if atom1_num < atom2_num else (1 if atom1_num > atom2_num else 0)

    # Cross-family comparison: Core Ultra is newer than traditional Core
    if "Core Ultra" in g1 and "Gen Core" in g2 and "Core Ultra" not in g2:
        return 1  # Core Ultra > traditional Core
    if "Gen Core" in g1 and "Core Ultra" not in g1 and "Core Ultra" in g2:
        return -1  # traditional Core < Core Ultra

    # Cannot compare - different families or unknown formats
    logger.debug(f"Cannot compare generations: '{g1}' vs '{g2}' (different families or unknown format)")
    return 0


def is_generation_supported(
    cpu_generation: str,
    supported_generations: list = None,
    unsupported_generations: list = None,
    product_collection: str = None,
    segment: str = None,
    codename: str = None,
) -> bool:
    """
    Check if a CPU generation is supported for qualification.

    Supports four validation formats in unsupported_generations list:
    1. String: "Core Ultra (Series 1)" - blocks all Series 1 processors
    2. Tuple (gen, collection): ("4th Gen", "Xeon Scalable") - blocks specific collection only
    3. Tuple (gen, collection, segment): ("4th Gen", "Xeon Scalable", "server") - most specific
    4. Dict: {"codename": "Tiger Lake-W", "product_collection": "Workstation"} - codename-based

    Dict format supports any combination of fields:
    - {"codename": "X"} - match by codename only
    - {"codename": "X", "product_collection": "Y"} - match both
    - {"generation": "X", "codename": "Y"} - match both
    - {"codename": "X", "product_collection": "Y", "segment": "Z"} - match all

    Logic:
    - If unsupported_generations is provided: generation is supported if NOT in that list
      (new platforms are automatically supported - only need to update when dropping support)
    - If supported_generations is provided (legacy): generation is supported if in that list
      or newer than minimum (requires updates for each new platform)
    - If neither is provided: assume supported (for backward compatibility)

    Args:
        cpu_generation: CPU generation string (e.g., "Core Ultra (Series 2)")
        supported_generations: List of supported generation strings (legacy approach)
        unsupported_generations: List of unsupported entries (strings, tuples, or dicts)
        product_collection: Product collection string (e.g., "Xeon Scalable", "Xeon Workstation")
        segment: Segment hint string (e.g., "server", "desktop", "mobile", "workstation")
        codename: CPU codename (e.g., "Tiger Lake-W", "Sapphire Rapids-WS")

    Returns:
        True if CPU generation is supported, False otherwise

    Examples:
        >>> is_generation_supported("4th Gen", unsupported_generations=["4th Gen Xeon Scalable"])
        True  # Different generation strings

        >>> is_generation_supported(
        ...     "4th Gen",
        ...     product_collection="Xeon Scalable",
        ...     unsupported_generations=[("4th Gen", "Xeon Scalable")]
        ... )
        False  # Exact match on (generation, product_collection)

        >>> is_generation_supported(
        ...     codename="Tiger Lake-W",
        ...     product_collection="Workstation",
        ...     unsupported_generations=[{"codename": "Tiger Lake-W"}]
        ... )
        False  # Codename match - blocked
    """
    if not cpu_generation:
        return False

    # Priority 1: Check unsupported list (recommended - new platforms auto-supported)
    if unsupported_generations:
        logger.debug(
            f"Checking if CPU generation '{cpu_generation}' is unsupported "
            f"(codename={codename}, product_collection={product_collection}, segment={segment})"
        )
        for unsupported_entry in unsupported_generations:
            # Handle string, tuple, and dict formats
            if isinstance(unsupported_entry, dict):
                # Dict format: {"codename": "X", "product_collection": "Y", ...}
                # Check all fields specified in the dict
                all_match = True

                # Check codename if specified
                if "codename" in unsupported_entry:
                    if not codename or codename != unsupported_entry["codename"]:
                        all_match = False

                # Check generation if specified
                if "generation" in unsupported_entry and all_match:
                    if cpu_generation != unsupported_entry["generation"]:
                        all_match = False

                # Check product_collection if specified
                if "product_collection" in unsupported_entry and all_match:
                    if not product_collection or product_collection != unsupported_entry["product_collection"]:
                        all_match = False

                # Check segment if specified
                if "segment" in unsupported_entry and all_match:
                    if not segment or segment != unsupported_entry["segment"]:
                        all_match = False

                # If all specified fields match, not supported
                if all_match:
                    return False

            elif isinstance(unsupported_entry, tuple):
                # Tuple format: (generation, product_collection) or (generation, product_collection, segment)
                if len(unsupported_entry) == 2:
                    unsupported_gen, unsupported_collection = unsupported_entry
                    unsupported_segment = None
                elif len(unsupported_entry) == 3:
                    unsupported_gen, unsupported_collection, unsupported_segment = unsupported_entry
                else:
                    # Invalid tuple length - skip
                    continue

                # Check if generation matches
                if cpu_generation == unsupported_gen:
                    # Check product collection if provided
                    if product_collection and product_collection == unsupported_collection:
                        # Check segment if tuple has 3 elements
                        if unsupported_segment is None:
                            # No segment specified in tuple - match on generation + collection only
                            return False
                        elif segment and segment == unsupported_segment:
                            # All three match - not supported
                            return False
                    elif not product_collection:
                        # No product_collection provided - can't validate tuple entries
                        # Skip this entry and continue checking
                        continue
            else:
                # String format: direct generation string match
                unsupported_gen = unsupported_entry

                # Direct match in unsupported list
                if cpu_generation == unsupported_gen:
                    return False

        # Check if CPU is older than any unsupported generation in the same family
        cpu_family = _get_generation_family(cpu_generation)

        for unsupported_entry in unsupported_generations:
            # Only do family comparison for string entries, not tuples or dicts
            if isinstance(unsupported_entry, (tuple, dict)):
                continue

            unsupported_gen = unsupported_entry
            unsupported_family = _get_generation_family(unsupported_gen)

            # Only compare if same family
            if cpu_family == unsupported_family:
                comparison = compare_generations(cpu_generation, unsupported_gen)
                # If comparison returns 0 (can't compare - different sub-families), treat as supported
                # If comparison < 0 (CPU is older), not supported
                # If comparison > 0 (CPU is newer), supported
                if comparison < 0:
                    return False

        # Not in unsupported list and not older than any unsupported - supported!
        logger.debug(
            f"CPU generation '{cpu_generation}' is supported: "
            f"not found in unsupported list and not older than any unsupported generation. "
            f"(codename={codename}, product_collection={product_collection}, segment={segment})"
        )
        return True

    # Priority 2: Check supported list (legacy approach - requires updates for new platforms)
    if supported_generations:
        # Direct match
        if cpu_generation in supported_generations:
            return True

        # For cross-family comparison, require explicit listing
        # Only check if CPU is newer within the same family
        cpu_family = _get_generation_family(cpu_generation)

        for supported_gen in supported_generations:
            supported_family = _get_generation_family(supported_gen)

            # Only compare if same family
            if cpu_family == supported_family:
                comparison = compare_generations(cpu_generation, supported_gen)
                # Only accept if CPU is genuinely newer (> 0)
                if comparison > 0:
                    return True

        return False

    # No restrictions defined - assume supported (backward compatibility)
    return True


def _get_generation_family(generation: str) -> str:
    """
    Extract the processor family from a generation string.

    Args:
        generation: Generation string (e.g., "Core Ultra (Series 2)", "12th Gen Core")

    Returns:
        Family identifier (e.g., "core", "xeon", "atom")

    Note:
        - Traditional Core (e.g., "12th Gen Core") and Core Ultra are considered
          the same "core" family since Core Ultra is the evolution of Core
        - "Core (Series 2)" (Raptor Lake) and "Core Ultra (Series 2)" are also
          considered the same "core" family despite different naming
        - "Xeon 6" and "Xeon Scalable" are both "xeon" family
    """
    gen_lower = generation.lower()

    # Core family includes traditional Core, Core Ultra, and Core Series
    if "core" in gen_lower:  # Matches "Core Ultra", "Gen Core", "Core (Series X)"
        return "core"
    # Xeon family includes Xeon Scalable, Xeon 6, and Xeon Max
    elif "xeon" in gen_lower:  # Matches "Xeon Scalable", "Xeon 6", "Xeon Max"
        return "xeon"
    elif "atom" in gen_lower:
        return "atom"
    else:
        return "unknown"


# Intel Vertical Segment Classification
# These are Intel's market segments: server, desktop, mobile, embedded, workstation
# NOT to be confused with profile tiers (entry, mainstream, etc.)
SEGMENT_PATTERNS = {
    "server": {
        "brand_patterns": [
            r"Intel\(R\) Xeon.*Scalable",
            r"Intel\(R\) Xeon.*Gold",
            r"Intel\(R\) Xeon.*Platinum",
            r"Intel\(R\) Xeon.*Silver",
            r"Intel\(R\) Xeon.*Bronze",
            r"Intel\(R\) Xeon\(R\) [0-9]{4}[EP]",  # Xeon 6 series (e.g., 6787P, 6766E)
        ],
        "model_patterns": [173, 207, 143, 106],  # GNR, EMR, SPR, ICL-SP
        "core_count_min": 8,
    },
    "desktop": {
        "brand_patterns": [
            r"Intel\(R\) Core\(TM\) Ultra [0-9]+ *\d{3}[KF]$",  # Core Ultra Desktop (MUST have K/F suffix)
            r"Intel\(R\) Core\(TM\) i[3-9]-\d+[KFST]?$",  # Core i-series Desktop
            r"Intel\(R\) Core\(TM\) [0-9]+-\d+[KF]?$",  # Core 9/7/5/3 Desktop
        ],
        "model_patterns": [198, 183, 151, 167, 165, 158],  # ARL-S, BTL-S, ADL-S, RKL-S, CML-S, CFL-S
        "suffix_patterns": ["K", "KF", "F", "T"],  # Desktop suffixes
    },
    "mobile": {
        "brand_patterns": [
            r"Intel\(R\) Core\(TM\) Ultra.*[UHP]$",  # Core Ultra Mobile with suffix
            # Core Ultra Mobile without suffix (e.g., Panther Lake 365, 355)
            r"Intel\(R\) Core\(TM\) Ultra [X0-9]+ \d{3}$",
            r"Intel\(R\) Core\(TM\) i[3-9]-\d+[UHPG]",  # Core i-series Mobile
            r"Intel\(R\) Core\(TM\) [0-9]+-\d+[UHP]$",  # Core 9/7/5/3 Mobile
            r"Intel\(R\).*(?:Processor|Core\(TM\)).*\bN\d{2,3}\b",  # N-series (Alder Lake-N / Twin Lake-N)
        ],
        "model_patterns": [
            204,  # PTL - Panther Lake (Core Ultra Series 3) - All mobile
            197,  # ARL-H - Arrow Lake-H
            181,  # ARL-U - Arrow Lake-U
            182,  # LNL - Lunar Lake
            170,  # MTL-H - Meteor Lake-H
            172,  # MTL-U - Meteor Lake-U
            154,  # ADL-P - Alder Lake-P
            191,  # RPL-P - Raptor Lake-P
            186,  # RPL-HX - Raptor Lake-HX
            166,  # CML-U - Comet Lake-U
            125,  # ICL-U - Ice Lake-U
            126,  # ICL-Y - Ice Lake-Y
            142,  # Multiple: CFL-U, KBL-U
            190,  # N-series (Alder Lake-N / Twin Lake-N)
        ],
        "suffix_patterns": ["H", "U", "P", "HX", "HK", "G1", "G4", "G7"],  # Mobile suffixes
    },
    "embedded": {
        "brand_patterns": [
            r"Intel\(R\) Atom",
            r"Intel\(R\) Celeron.*[NJ]\d+",
            r"Intel\(R\) Pentium.*[NJ]\d+",
            r"Intel\(R\) Core\(TM\) [3579] \d+[A-Z]*E\b",  # Core Series 2 Embedded
        ],
        "model_patterns": [96, 183],  # EHL, BTL-S (Bartlett Lake Embedded)
        # Embedded suffixes: E=Embedded, TE=IoT, MRE/RE=Reliability Enhanced
        "suffix_patterns": ["E", "TE", "MRE", "RE"],
        "core_count_max": 32,  # High-performance embedded (Bartlett Lake has 24 cores)
    },
    "workstation": {
        "brand_patterns": [
            r"(?i)Intel\(R\) Xeon.*\sw[-0-9]",  # Matches both "W-" and "w7" (case-insensitive)
            r"Intel\(R\) Core\(TM\).*Extreme",
        ],
        "model_patterns": [],  # Xeon W uses similar models to desktop/server
        "suffix_patterns": ["X", "XE"],  # Extreme/Workstation suffixes
    },
}


def _detect_generation_from_brand(brand: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Fallback: Try to detect generation from brand string when model number is unmapped.

    This helps with newer CPUs not yet in CPU_GENERATION_MAP.

    Important: Intel has two distinct "Series 2" product lines:
    - "Intel Core (Series 2)": Raptor Lake-based (Core 7/5, not Ultra)
    - "Intel Core Ultra (Series 2)": Arrow Lake/Lunar Lake-based

    Args:
        brand: CPU brand string

    Returns:
        Tuple of (codename, generation, product_collection, segment_hint) or (None, None, None, None) if cannot detect
    """
    brand_lower = brand.lower()
    is_n_series_brand = re.search(r"\bN\d{2,3}\b", brand, re.IGNORECASE) is not None

    # Detect Core Ultra Series 3, 2, 1 (has "ultra" keyword)
    # Intel naming pattern: Core Ultra [X/9/7/5] [3-digit model number][suffix]
    # Series determined by model number:
    # - 3XX (300-399) = Series 3 (Panther Lake)
    # - 2XX (200-299) = Series 2 (Arrow Lake)
    # - 1XX (100-199) = Series 1 (Meteor Lake)
    if "ultra" in brand_lower:
        # Extract model number and suffix from brand string
        # Pattern matches: "Ultra X9 388H", "Ultra 9 285K", "Ultra 7 165H"
        model_match = re.search(r"ultra\s+[X]?[0-9]+\s+(\d{3})([A-Z]*)", brand, re.IGNORECASE)
        if model_match:
            model_number = int(model_match.group(1))
            suffix = model_match.group(2)  # H, U, K, HX, etc.

            # Determine series based on model number
            if 300 <= model_number <= 399:  # Series 3 (Panther Lake)
                # Panther Lake: All models are mobile only
                codename = "Panther Lake"
                generation = "Core Ultra (Series 3)"
                product_collection = "Core Ultra"
                segment = "mobile"
                logger.debug(f"Detected Core Ultra Series 3 (Panther Lake) from model: {model_number}{suffix}")
                return codename, generation, product_collection, segment

            elif 200 <= model_number <= 299:  # Series 2 (Arrow Lake)
                # Arrow Lake has different codenames based on suffix
                generation = "Core Ultra (Series 2)"
                product_collection = "Core"

                # Determine codename and segment from suffix
                if suffix in ["K", "KF", "T"]:  # Desktop variants
                    codename = "Arrow Lake-S"
                    segment = "desktop"
                elif suffix in ["H", "HX"]:  # High-power mobile
                    codename = "Arrow Lake-H"
                    segment = "mobile"
                elif suffix == "U":  # Low-power mobile
                    codename = "Arrow Lake-U"
                    segment = "mobile"
                else:
                    # Unknown suffix, use generic codename
                    codename = "Arrow Lake"
                    segment = "mobile"  # Most Arrow Lake are mobile

                logger.debug(f"Detected Core Ultra Series 2 (Arrow Lake) from model: {model_number}{suffix}")
                return codename, generation, product_collection, segment

            elif 100 <= model_number <= 199:  # Series 1 (Meteor Lake)
                # Meteor Lake has different codenames based on suffix
                generation = "Core Ultra (Series 1)"
                product_collection = "Core"

                # Determine codename from suffix
                if suffix in ["H", "HX"]:  # High-power mobile
                    codename = "Meteor Lake-H"
                elif suffix == "U":  # Low-power mobile
                    codename = "Meteor Lake-U"
                else:
                    # Unknown suffix, use generic codename
                    codename = "Meteor Lake"

                segment = "mobile"  # All Meteor Lake are mobile
                logger.debug(f"Detected Core Ultra Series 1 (Meteor Lake) from model: {model_number}{suffix}")
                return codename, generation, product_collection, segment

        # Fallback: Core Ultra detected but couldn't parse model number
        # Check for explicit series mentions
        if "series 3" in brand_lower or "series3" in brand_lower:
            return "Unknown (Series 3)", "Core Ultra (Series 3)", "Core Ultra", "unknown"
        elif "series 2" in brand_lower or "series2" in brand_lower:
            return "Unknown (Series 2)", "Core Ultra (Series 2)", "Core", "unknown"
        elif "series 1" in brand_lower or "series1" in brand_lower:
            return "Unknown (Series 1)", "Core Ultra (Series 1)", "Core", "unknown"
        else:
            # Cannot determine series, return generic
            logger.debug("Detected 'Core Ultra' but cannot determine series from brand string")
            return "Unknown (Core Ultra)", "Core Ultra", "Core Ultra", "unknown"

    # Detect Core Series 2 (Raptor Lake-based, NO "ultra" keyword)
    # These are distinct from Core Ultra Series 2
    # Key differentiator: "Core 7" vs "Core i7" (no "i" prefix)
    # Examples: Core 7 250U, Core 5 220U (NOT Core i7, Core i5)
    if "ultra" not in brand_lower and not is_n_series_brand:
        # Check for Core 3/5/7/9 pattern WITHOUT "i" prefix
        # Pattern: "core" ... whitespace ... digit (3/5/7/9) ... whitespace ... NOT "i"
        # Matches: "Core(TM) 7 250U" but NOT "Core(TM) i7-13700K"
        if re.search(r"core.*?\s+[3579]\s+(?!i)", brand_lower):
            # Extract model number and suffix to determine codename and segment
            # Pattern: "Core(TM) 7 251TE" -> 251, "TE" (handles TM, R symbols)
            model_match = re.search(r"core.*?[3579]\s+(\d{2,3})([A-Z]*)", brand, re.IGNORECASE)
            if model_match:
                model_number = int(model_match.group(1))
                suffix = model_match.group(2) if model_match.group(2) else ""

                # Bartlett Lake-S: 25X series (embedded processors)
                # Based on Intel ARK: Core 7 251TE, Core 7 251E
                if 250 <= model_number <= 259:
                    codename = "Bartlett Lake-S"
                    # Detect embedded segment from TE/E suffixes
                    segment = "embedded" if suffix in ["TE", "E"] else ("mobile" if suffix in ["U", "H"] else "unknown")
                    return codename, "Core (Series 2)", "Core", segment

                # Other Core Series 2 processors (e.g., Raptor Lake-U Re-refresh)
                # May have model numbers outside 25X range
                # Examples: Core 7 250U, Core 5 220U (mobile)
                codename = "Unknown (Core Series 2)"
                segment = "mobile" if suffix in ["U", "H"] else ("embedded" if suffix in ["TE", "E"] else "unknown")
                return codename, "Core (Series 2)", "Core", segment

            # Fallback if no model number found
            return "Unknown (Core Series 2)", "Core (Series 2)", "Core", "unknown"

    # Detect Intel Atom processors
    # Support for multiple Atom families with proper product collection and segment assignment

    # Atom x7000 series - TWO different codenames in same x7000 series:
    # 1. Alder Lake-N (2022): x7xxxE models (x7425E, x7213E, x7211E) - UNSUPPORTED
    # 2. Amston Lake (2024): x7xxxC, x7xxxRE, x7xxxFE models (x7203C, x7809C, x7433FE) - SUPPORTED

    # Alder Lake-N: Atom x7000 series with E suffix (2022, Gracemont) - Embedded segment
    # Wikipedia: Listed under "Alder Lake-N" section, 12th Gen processors
    # Examples: x7425E, x7213E, x7211E
    # Pattern: "atom" followed by "x7" then digits ending with "e" (lowercase)
    if re.search(r"atom.*?x7\d{2,3}e\b", brand_lower):
        return "Alder Lake-N", "Atom x7000", "Atom", "embedded"

    # Amston Lake: Atom x7000 series with C/RE/FE suffixes (2024, Gracemont) - Embedded segment
    # Examples: x7203C, x7211RE, x7405C, x7433RE, x7809C, x7835RE
    # Pattern: "atom" followed by "x7" then digits (but not ending with "e")
    if re.search(r"atom.*?x7\d{3,4}(c|re|fe)", brand_lower):
        return "Amston Lake", "Atom x7000", "Atom", "embedded"

    # Atom x6000 series (Elkhart Lake, 2021) - Embedded segment
    # Examples: x6413E, x6414RE, x6425RE, x6427FE
    # Pattern: "atom" followed by "x6" then 3-4 digits
    if re.search(r"atom.*?x6\d{3,4}", brand_lower):
        return "Elkhart Lake", "Atom x6000", "Atom", "embedded"

    # Atom x5000 series (Apollo Lake, 2016) - Embedded segment
    # Examples: x5-E3930, x5-E3940, x7-E3950 (note: includes x7-E series)
    # Pattern: "atom" followed by "x5-e" or "x7-e" then 4 digits (hyphenated format, lowercase)
    if re.search(r"atom.*?x[57]-e\d{4}", brand_lower):
        return "Apollo Lake", "Atom x5000", "Atom", "embedded"

    # Atom C-series (Denverton, 2017) - SERVER segment (not embedded!)
    # Examples: C3308, C3336, C3338R, C3508, C3538, C3558, C3558R, C3708, C3750, C3758, C3758R
    #           C3808, C3830, C3850, C3858, C3950, C3955, C3958
    # Pattern: "atom" followed by "c" then 4 digits (lowercase)
    # CRITICAL: Wikipedia confirms this is SERVER segment, not embedded
    if re.search(r"atom.*?c\d{4}", brand_lower):
        return "Denverton", "Atom (Silvermont)", "Atom", "server"

    # Atom Z-series (Bay Trail, 2013-2015) - Mobile segment
    # Examples: Z3680, Z3735D, Z3735E, Z3735F, Z3735G, Z3736G, Z3740, Z3740D, Z3745,
    #           Z3770, Z3775, Z3785, Z3795
    # Pattern: "atom" followed by "z" then 4 digits (lowercase)
    if re.search(r"atom.*?z\d{4}", brand_lower):
        return "Bay Trail", "Atom (Bay Trail)", "Atom", "mobile"

    # Detect Intel Celeron J-series (Desktop Low Power)
    # Examples: J1750, J1800, J1850, J1900 (Bay Trail-D) through J6412, J6413 (Elkhart Lake)
    # Pattern: "celeron" followed by "j" then 4 digits (lowercase)
    if re.search(r"celeron.*?j\d{4}", brand_lower):
        # Map by model number to codename
        model_match = re.search(r"[jJ](\d{4})", brand)
        if model_match:
            model_num = int(model_match.group(1))
            if 1000 <= model_num < 2000:
                return "Bay Trail-D", "Celeron J-series", "Celeron", "desktop"
            elif 3000 <= model_num < 3200:
                return "Braswell", "Celeron J-series", "Celeron", "desktop"
            elif 3200 <= model_num < 4000:
                return "Apollo Lake", "Celeron J-series", "Celeron", "desktop"
            elif 4000 <= model_num < 4100:
                return "Gemini Lake", "Celeron J-series", "Celeron", "desktop"
            elif 4100 <= model_num < 5000:
                return "Gemini Lake Refresh", "Celeron J-series", "Celeron", "desktop"
            elif 6000 <= model_num < 7000:
                return "Elkhart Lake", "Celeron J-series", "Celeron", "desktop"
        # Fallback if no model number found
        return "Unknown (Celeron J-series)", "Celeron J-series", "Celeron", "desktop"

    # Detect Intel Celeron G-series (Desktop Standard Power)
    # Examples: G440, G465, G470 (Sandy Bridge) through G6900, G6900E (Alder Lake)
    # Pattern: "celeron" followed by "g" then 4 digits (lowercase)
    if re.search(r"celeron.*?g\d{4}", brand_lower):
        # Map by model number to codename
        model_match = re.search(r"[gG](\d{4})", brand)
        if model_match:
            model_num = int(model_match.group(1))
            if 400 <= model_num < 600:
                return "Sandy Bridge", "Celeron G-series", "Celeron", "desktop"
            elif 1600 <= model_num < 1700:
                return "Ivy Bridge", "Celeron G-series", "Celeron", "desktop"
            elif 1800 <= model_num < 1900:
                return "Haswell", "Celeron G-series", "Celeron", "desktop"
            elif 3900 <= model_num < 4000:
                # G39xx can be either Skylake or Kaby Lake (both use similar model numbers)
                return "Skylake", "Celeron G-series", "Celeron", "desktop"
            elif 4900 <= model_num < 5000:
                return "Coffee Lake", "Celeron G-series", "Celeron", "desktop"
            elif 5900 <= model_num < 6000:
                return "Comet Lake", "Celeron G-series", "Celeron", "desktop"
            elif 6900 <= model_num < 7000:
                return "Alder Lake", "Celeron G-series", "Celeron", "desktop"
        # Fallback if no model number found
        return "Unknown (Celeron G-series)", "Celeron G-series", "Celeron", "desktop"

    # Detect Intel Celeron B-series (Mobile Sandy Bridge)
    # Examples: B710, B720, B730, B800, B810, B815, B820, B830, B840
    # Pattern: "celeron" followed by "b" then 3 digits (lowercase)
    if re.search(r"celeron.*?b\d{3}\b", brand_lower):
        return "Sandy Bridge", "Celeron B-series", "Celeron", "mobile"

    # Detect Intel Celeron N-series (Traditional 4-digit N-series, 2013-2021)
    # Examples: N2805-N2910 (Bay Trail-M), N3050 (Braswell), N3350, N3450 (Apollo Lake),
    #           N4000, N4100 (Gemini Lake), N4020, N4120 (Gemini Lake Refresh),
    #           N4500, N4505, N5100, N5105 (Jasper Lake), N6210 (Elkhart Lake)
    # Pattern: "celeron" followed by "n" then 4 digits (lowercase)
    # NOTE: This is different from user's 2-3 digit N-series (N305, N300, N200, etc.)
    if re.search(r"celeron.*?n\d{4}", brand_lower):
        model_match = re.search(r"[nN](\d{4})", brand)
        if model_match:
            model_num = int(model_match.group(1))
            if 2000 <= model_num < 3000:
                return "Bay Trail-M", "Celeron N-series", "Celeron", "mobile"
            elif model_num == 3050:
                return "Braswell", "Celeron N-series", "Celeron", "mobile"
            elif 3350 <= model_num <= 3450:
                return "Apollo Lake", "Celeron N-series", "Celeron", "mobile"
            elif 4000 <= model_num <= 4100:
                return "Gemini Lake", "Celeron N-series", "Celeron", "mobile"
            elif 4020 <= model_num <= 4120:
                return "Gemini Lake Refresh", "Celeron N-series", "Celeron", "mobile"
            elif 4500 <= model_num <= 5105:
                return "Jasper Lake", "Celeron N-series", "Celeron", "mobile"
            elif model_num == 6210:
                return "Elkhart Lake", "Celeron N-series", "Celeron", "mobile"
        # Fallback if no model number found or unrecognized
        return "Unknown (Celeron N-series)", "Celeron N-series", "Celeron", "mobile"

    # Detect Intel Pentium Silver J-series (Desktop Low Power)
    # Examples: J5005 (Gemini Lake), J5040 (Gemini Lake Refresh)
    # Pattern: "pentium" + "silver" + "j" + 4 digits (lowercase)
    if re.search(r"pentium.*?silver.*?j\d{4}", brand_lower):
        model_match = re.search(r"[jJ](\d{4})", brand)
        if model_match:
            model_num = int(model_match.group(1))
            if 5000 <= model_num < 5100:
                # J5000-J5099: Gemini Lake or Gemini Lake Refresh
                # J5005: Gemini Lake (December 2017)
                # J5040: Gemini Lake Refresh (November 2019)
                if model_num <= 5010:
                    return "Gemini Lake", "Pentium Silver J-series", "Pentium", "desktop"
                else:
                    return "Gemini Lake Refresh", "Pentium Silver J-series", "Pentium", "desktop"
        return "Gemini Lake", "Pentium Silver J-series", "Pentium", "desktop"

    # Detect Intel Pentium Silver N-series (Mobile Ultra-Low Power - Jasper Lake)
    # Examples: N6000 (Jasper Lake), N6005 (Jasper Lake)
    # Pattern: "pentium" + "silver" + "n" + 4 digits (lowercase)
    if re.search(r"pentium.*?silver.*?n\d{4}", brand_lower):
        return "Jasper Lake", "Pentium Silver N-series", "Pentium", "mobile"

    # Detect Intel Pentium Gold G-series (Desktop Standard Power - Alder Lake)
    # Examples: G7400 (Alder Lake)
    # Pattern: "pentium" + "gold" + "g" + 4 digits (lowercase)
    if re.search(r"pentium.*?gold.*?g\d{4}", brand_lower):
        return "Alder Lake", "Pentium Gold G-series", "Pentium", "desktop"

    # Detect Intel N-series (User's newer processors + Traditional Celeron N-series)
    # CRITICAL: Three distinct families to handle!
    #
    # 1. User's N-series (2-3 digits, 2023-2026+):
    #    - Alder Lake-N (2023): i3-N305, i3-N300, N305, N300, N200, N100, N97, N50 → UNSUPPORTED
    #    - Twin Lake (2026): N355, N350, N250, N150 → SUPPORTED
    #
    # 2. Traditional Celeron N-series (4 digits, 2013-2021):
    #    - Bay Trail-M: N2805, N2806, N2807, N2810, N2815, N2830, N2840, N2910
    #    - Braswell: N3050
    #    - Apollo Lake: N3350, N3450
    #    - Gemini Lake: N4000, N4100
    #    - Gemini Lake Refresh: N4020, N4120
    #    - Jasper Lake: N4500, N4505, N5100, N5105
    #    - Elkhart Lake: N6210
    #
    # Examples: "Intel(R) Processor N100", "Intel(R) Core(TM) 3 N355", "Intel(R) Core(TM) i3-N300"
    if is_n_series_brand:
        # Step 1: Check for i3-N prefix (Alder Lake-N only)
        # Examples: i3-N305, i3-N300
        if re.search(r"i3-N\d{2,3}", brand):
            return "Alder Lake-N", "Intel N-series", "N-series", "mobile"

        # Step 2: Extract model number from brand string (case-insensitive)
        # Pattern: "N" followed by 2-4 digits
        model_match = re.search(r"\b[nN](\d{2,4})\b", brand)
        if model_match:
            model_str = model_match.group(1)
            model_num = int(model_str)

            # Step 3a: User's 2-3 digit N-series (Alder Lake-N vs Twin Lake)
            if len(model_str) <= 3:  # 2-3 digits: N50, N97, N100, N200, N300, N305, N355, N350, N250, N150
                # Twin Lake models (2026): N355, N350, N250, N150
                if model_num in [355, 350, 250, 150]:
                    return "Twin Lake", "Intel N-series", "N-series", "mobile"
                else:
                    # Alder Lake-N models (2023): N305, N300, N200, N100, N97, N50
                    return "Alder Lake-N", "Intel N-series", "N-series", "mobile"

            # Step 3b: Traditional Celeron 4-digit N-series (2013-2021)
            elif len(model_str) == 4:  # 4 digits: N2xxx through N6xxx
                # Map by model number ranges to codename
                if 2000 <= model_num < 3000:
                    return "Bay Trail-M", "Celeron N-series", "Celeron", "mobile"
                elif model_num == 3050:
                    return "Braswell", "Celeron N-series", "Celeron", "mobile"
                elif 3350 <= model_num <= 3450:
                    return "Apollo Lake", "Celeron N-series", "Celeron", "mobile"
                elif 4000 <= model_num <= 4100:
                    return "Gemini Lake", "Celeron N-series", "Celeron", "mobile"
                elif 4020 <= model_num <= 4120:
                    return "Gemini Lake Refresh", "Celeron N-series", "Celeron", "mobile"
                elif 4500 <= model_num <= 5105:
                    return "Jasper Lake", "Celeron N-series", "Celeron", "mobile"
                elif model_num == 6210:
                    return "Elkhart Lake", "Celeron N-series", "Celeron", "mobile"

        # Fallback if model number not found or unrecognized
        return "Unknown N-series", "Intel N-series", "N-series", "mobile"

    # Detect traditional Core i-series generations (14th, 13th, 12th gen)
    # IMPORTANT: 14th gen (Raptor Lake Refresh) shares CPUID model 183 with 13th gen
    # Must detect by brand string pattern: i[3579]-14xxx (e.g., i9-14900T, i7-14700K)
    if re.search(r"core.*i[3579]-14\d{3}", brand_lower):  # 14xxx: i3/i5/i7/i9-14000 series
        return "Raptor Lake-S Refresh", "14th Gen Core", "Core", "unknown"
    elif re.search(r"core.*i[3579]-1[5-9]\d{3}", brand_lower):  # 15xxx, 16xxx, etc. (future)
        gen_match = re.search(r"i[3579]-(1[5-9])\d{3}", brand_lower)
        if gen_match:
            gen_num = gen_match.group(1)
            return f"Unknown ({gen_num}th Gen)", f"{gen_num}th Gen Core", "Core", "unknown"
        return "Unknown (15th+ Gen)", "15th Gen Core", "Core", "unknown"
    elif re.search(r"core.*i[3579]-13\d{3}", brand_lower):  # 13xxx: i3/i5/i7/i9-13000 series
        return "Raptor Lake-S", "13th Gen Core", "Core", "unknown"
    elif re.search(r"core.*i[3-9]-1[2]\d{3}", brand_lower):  # 12xxx
        return "Unknown (12th Gen)", "12th Gen Core", "Core", "unknown"
    elif re.search(r"core.*i[3-9]-1[0-1]\d{3}", brand_lower):  # 10xxx, 11xxx
        gen_match = re.search(r"i[3-9]-1([0-1])\d{3}", brand_lower)
        if gen_match:
            gen_num = gen_match.group(1)
            return f"Unknown ({gen_num}th Gen)", f"{gen_num}th Gen Core", "Core", "unknown"
        return "Unknown (10th/11th Gen)", "10th Gen Core", "Core", "unknown"

    # Detect Xeon processors with new 2024 branding
    if "xeon" in brand_lower:
        # Xeon W (workstation) - check FIRST before Xeon Scalable (shares same model 143)
        # Pattern detection for different Xeon W series:
        # - Sapphire Rapids-WS (2023+): lowercase "w" + tier (w3, w5, w7, w9)
        #   Example: "Intel(R) Xeon(R) w7-3565X"
        # - Older Xeon W (2017-2021): uppercase "W-" + model number
        #   Examples: "Xeon W-3375" (Ice Lake-W), "Xeon W-2295" (Cascade Lake-W)

        # Sapphire Rapids-WS (new lowercase "w" pattern)
        if re.search(r"xeon.*\sw[0-9]-", brand_lower):
            return "Sapphire Rapids-WS", "4th Gen", "Xeon Workstation", "workstation"

        # Older Xeon W series (uppercase "W-" pattern)
        # These return generation-agnostic "Xeon W" to be handled by product_collection
        elif re.search(r"xeon.*\sW-", brand):  # Use original brand (not lower) to detect uppercase W
            # Try to identify specific generation from model number pattern
            if re.search(r"W-11\d{3}", brand):  # W-11xxx = Tiger Lake-W
                return "Tiger Lake-W", "Xeon W", "Workstation", "workstation"
            elif re.search(r"W-1[0-9]{3}", brand):  # W-1xxx = Rocket/Comet Lake-W
                return "Rocket/Comet Lake-W", "Xeon W", "Workstation", "workstation"
            elif re.search(r"W-3[0-9]{3}", brand):  # W-3xxx = Ice Lake-W or Cascade Lake-W
                return "Ice/Cascade Lake-W", "Xeon W", "Workstation", "workstation"
            elif re.search(r"W-2[0-9]{3}", brand):  # W-2xxx = Cascade Lake-W or Skylake-W
                return "Cascade/Skylake-W", "Xeon W", "Workstation", "workstation"
            else:
                return "Unknown Xeon W", "Xeon W", "Workstation", "workstation"

        # Xeon Scalable (Gold/Silver/Platinum/Bronze) - CHECK FIRST before Xeon 6
        # Traditional Xeon Scalable naming uses metal tiers + model numbers
        # Examples: "Xeon Gold 6548Y+", "Xeon Silver 4410Y", "Xeon Platinum 8380"
        # CRITICAL: Must check BEFORE Xeon 6 pattern to avoid false matches
        if re.search(r"(gold|silver|platinum|bronze)", brand_lower):
            # Extract model number (4 digits) from brand string
            model_match = re.search(r"(gold|silver|platinum|bronze)\s+(\d{4})", brand_lower)
            if model_match:
                model_number = int(model_match.group(2))

                # Determine generation and codename based on model number ranges
                # 5th Gen: Emerald Rapids-SP (55XX, 65XX, 85XX range)
                # 4th Gen: Sapphire Rapids-SP (44XX, 54XX, 64XX, 84XX range)
                # 3rd Gen: Ice Lake-SP (43XX, 53XX, 63XX, 83XX range)

                if 5500 <= model_number <= 5599 or 6500 <= model_number <= 6599 or 8500 <= model_number <= 8999:
                    # 5th Gen Xeon Scalable (Emerald Rapids-SP)
                    return "Emerald Rapids-SP", "5th Gen", "Xeon Scalable", "server"
                elif (
                    4400 <= model_number <= 4499
                    or 5400 <= model_number <= 5499
                    or 6400 <= model_number <= 6499
                    or 8400 <= model_number <= 8499
                ):
                    # 4th Gen Xeon Scalable (Sapphire Rapids-SP)
                    return "Sapphire Rapids-SP", "4th Gen", "Xeon Scalable", "server"
                elif (
                    4200 <= model_number <= 4399
                    or 5200 <= model_number <= 5399
                    or 6200 <= model_number <= 6399
                    or 8200 <= model_number <= 8399
                ):
                    # 3rd Gen Xeon Scalable (Ice Lake-SP)
                    return "Ice Lake-SP", "3rd Gen", "Xeon Scalable", "server"
                elif (
                    4100 <= model_number <= 4199
                    or 5100 <= model_number <= 5199
                    or 6100 <= model_number <= 6199
                    or 8100 <= model_number <= 8199
                ):
                    # 2nd Gen Xeon Scalable (Cascade Lake-SP)
                    return "Cascade Lake-SP", "2nd Gen", "Xeon Scalable", "server"
                else:
                    # Unknown generation
                    return "Unknown", "Unknown", "Xeon Scalable", "server"

        # Xeon Scalable (NO tier keyword) - Fallback for processors like "Xeon 8580"
        # Some systems may report Xeon Scalable processors without Gold/Silver/Platinum/Bronze prefix
        # Examples: "Xeon 8580" (should be Platinum 8580), "Xeon 8571N"
        # CRITICAL: Must check BEFORE Xeon 6 and exclude 6XXX models (which are Xeon 6, not Scalable)
        # Pattern: "xeon" + optional "(r)" + space + 4-digit starting with 4, 5, or 8 (NOT 6)
        elif re.search(r"xeon.*?\s+[458]\d{3}", brand_lower):
            # Extract model number (4 digits) from brand string
            model_match = re.search(r"xeon.*?\s+([458]\d{3})", brand_lower)
            if model_match:
                model_number = int(model_match.group(1))

                # Use same generation detection logic as tier-based check
                if 5500 <= model_number <= 5599 or 8500 <= model_number <= 8999:
                    # 5th Gen Xeon Scalable (Emerald Rapids-SP)
                    return "Emerald Rapids-SP", "5th Gen", "Xeon Scalable", "server"
                elif 4400 <= model_number <= 4499 or 5400 <= model_number <= 5499 or 8400 <= model_number <= 8499:
                    # 4th Gen Xeon Scalable (Sapphire Rapids-SP)
                    return "Sapphire Rapids-SP", "4th Gen", "Xeon Scalable", "server"
                elif 4200 <= model_number <= 4399 or 5200 <= model_number <= 5399 or 8200 <= model_number <= 8399:
                    # 3rd Gen Xeon Scalable (Ice Lake-SP)
                    return "Ice Lake-SP", "3rd Gen", "Xeon Scalable", "server"
                elif 4100 <= model_number <= 4199 or 5100 <= model_number <= 5199 or 8100 <= model_number <= 8199:
                    # 2nd Gen Xeon Scalable (Cascade Lake-SP)
                    return "Cascade Lake-SP", "2nd Gen", "Xeon Scalable", "server"
                else:
                    # Unknown generation
                    return "Unknown", "Unknown", "Xeon Scalable", "server"

        # Xeon Max Series (Sapphire Rapids with HBM)
        # Examples: "Xeon Max 9460", "Xeon Max 9468"
        # Note: Product collection is "Xeon Scalable" not "Xeon Max" per test expectations
        elif "max" in brand_lower:
            return "Sapphire Rapids-HBM", "4th Gen", "Xeon Scalable", "server"

        # Xeon 6 (2024 branding) - replaces "6th Gen Xeon Scalable"
        # Xeon 6 has TWO segments: Server AND Workstation
        # Model numbering pattern:
        # - 6XXXP (P suffix): Server segment, P-cores (Granite Rapids)
        # - 6XXXE (E suffix): Server segment, E-cores (Sierra Forest)
        # - 678X, 677X: Workstation segment (Granite Rapids-WS)
        # CRITICAL: This is different from "Xeon W" which is a separate legacy product collection
        # Pattern: "Xeon" + 4-digit number starting with 6, NO metal tier keyword
        # Pattern allows for (R) trademark symbol between Xeon and model number
        elif re.search(r"xeon.*?\s+6\d{3}", brand_lower):
            # Extract model number to determine segment
            # Pattern: "Xeon" followed by optional "(R)" then 4-digit model number starting with 6
            # Examples: "Xeon(R) 6787P" (server), "Xeon(R) 6780" (workstation), "Xeon(R) 6766E" (server)
            model_match = re.search(r"xeon.*?(\d{4}\w?)", brand_lower)
            if model_match:
                model_number = model_match.group(1)
                # Workstation models: 6780, 6781, etc. (starts with 678 or 677, no P/E suffix)
                # Server models: 6787P, 6730P (P suffix), 6766E (E suffix)
                if (model_number.startswith("678") or model_number.startswith("677")) and not (
                    model_number.endswith("p") or model_number.endswith("e")
                ):
                    # Xeon 6 Workstation segment (e.g., 6780, 6781)
                    logger.debug(f"Detected Xeon 6 Workstation segment from model: {model_number}")
                    return "Granite Rapids-WS", "Xeon 6", "Xeon 6", "workstation"
                elif model_number.endswith("p"):
                    # P-core server processors (Granite Rapids)
                    logger.debug(f"Detected Xeon 6 P-core Server (Granite Rapids) from model: {model_number}")
                    return "Granite Rapids", "Xeon 6", "Xeon 6", "server"
                elif model_number.endswith("e"):
                    # E-core server processors (Sierra Forest)
                    logger.debug(f"Detected Xeon 6 E-core Server (Sierra Forest) from model: {model_number}")
                    return "Sierra Forest", "Xeon 6", "Xeon 6", "server"
                else:
                    # Unknown variant, default to Granite Rapids (most common)
                    logger.debug(f"Detected Xeon 6 Server (default to Granite Rapids) from model: {model_number}")
                    return "Granite Rapids", "Xeon 6", "Xeon 6", "server"
            # Fallback: if no model number detected, assume Granite Rapids server
            return "Granite Rapids", "Xeon 6", "Xeon 6", "server"

        # Traditional Xeon Scalable with explicit "scalable" keyword (legacy branding, pre-2024)
        elif "scalable" in brand_lower:
            # Try to extract generation number
            xeon_gen_match = re.search(r"(\d+)(?:th|st|nd|rd)\s+gen", brand_lower)
            if xeon_gen_match:
                gen_num = xeon_gen_match.group(1)
                return f"Unknown ({gen_num}th Gen Xeon)", f"{gen_num}th Gen", "Xeon Scalable", "server"

    return None, None, None, None


def detect_cpu_generation_and_segment(
    family: int, model: int, stepping: int, brand: str, core_count: int
) -> Dict[str, str]:
    """
    Detect Intel CPU generation, product collection, and vertical segment.

    This function provides comprehensive CPU classification including:
    - Generation: Intel generation (e.g., "4th Gen", "Core Ultra (Series 2)")
    - Product Collection: Product line within generation (e.g., "Xeon Scalable", "Xeon Workstation")
    - Codename: Architecture codename (e.g., "Sapphire Rapids", "Arrow Lake-S")
    - Segment: Market segment (e.g., "server", "desktop", "mobile", "workstation")

    Product Collection Examples:
    - Sapphire Rapids: "Xeon Scalable" (server), "Xeon Workstation" (workstation), "Xeon Max" (HBM)
    - 4th Gen: "Core", "Core Ultra"

    Args:
        family: CPU family number (e.g., 6)
        model: CPU model number (e.g., 198)
        stepping: CPU stepping number (e.g., 2)
        brand: CPU brand string (e.g., "Intel(R) Core(TM) Ultra 9 285K")
        core_count: Physical core count

    Returns:
        Dict containing:
        - codename: CPU codename (e.g., "Arrow Lake-S", "Sapphire Rapids-WS")
        - generation: Intel generation string (e.g., "4th Gen", "Core Ultra (Series 2)")
        - product_collection: Product line (e.g., "Xeon Scalable", "Xeon Workstation", "Core Ultra")
        - segment: Market segment hint (e.g., "desktop", "server", "mobile", "embedded", "workstation")
        - is_supported: Deprecated - use is_generation_supported() for validation
    """
    result = {
        "codename": "Unknown",
        "generation": "Unknown",
        "product_collection": "Unknown",
        "segment": "unknown",
        "is_supported": False,
    }

    # CRITICAL: Check for Xeon W first (shares models with Xeon Scalable)
    # Xeon W workstation processors must be distinguished by brand string
    # Two patterns to detect:
    # 1. Sapphire Rapids-WS: lowercase "w[digit]-" (e.g., "Xeon w7-3565X")
    # 2. Legacy Xeon W: uppercase "W-" (e.g., "Xeon W-3375")
    xeon_w_detected = False  # Track if we detected Xeon W explicitly
    product_collection = None  # Initialize product collection
    brand_lower = brand.lower()

    if "xeon" in brand_lower:
        # Check for Sapphire Rapids WS pattern (lowercase w[digit]-)
        if re.search(r"\sw[0-9]-", brand_lower):
            # This is Xeon W (workstation), NOT Xeon Scalable (server)
            # Sapphire Rapids WS uses "w[0-9]-" pattern (w3, w5, w7, w9)
            codename = "Sapphire Rapids-WS"
            generation = "4th Gen"  # Generation only (not "4th Gen Xeon Workstation")
            product_collection = "Xeon Workstation"  # Product collection separate
            segment_hint = "workstation"
            xeon_w_detected = True
            logger.debug(
                f"Detected Xeon W workstation processor (Sapphire Rapids-WS) from brand string: "
                f"codename={codename}, generation={generation}, product_collection={product_collection}"
            )
        # Check for legacy Xeon W pattern (uppercase W-) - use original brand string
        elif re.search(r"Xeon.*\sW-", brand):  # Note: using 'brand' not 'brand_lower' to detect uppercase W
            # Legacy Xeon W (pre-Sapphire Rapids): Skylake-W, Cascade Lake-W, Ice Lake-W, Rocket Lake-W, Tiger Lake-W
            # Get specific codename and generation from brand pattern
            if re.search(r"W-11\d{3}", brand):
                codename = "Tiger Lake-W"
                generation = "Tiger Lake Xeon W"
            elif re.search(r"W-1[0-9]{3}", brand):
                codename = "Rocket/Comet Lake-W"
                generation = "Rocket/Comet Lake Xeon W"
            elif re.search(r"W-3[0-9]{3}", brand):
                # W-3xxx series: Ice Lake-W (W-33xx) or Cascade Lake-W (W-32xx)
                codename = "Ice/Cascade Lake-W"
                generation = "Ice/Cascade Lake Xeon W"
            elif re.search(r"W-2[0-9]{3}", brand):
                # W-2xxx series: Cascade Lake-W (W-22xx) or Skylake-W (W-21xx)
                codename = "Cascade/Skylake-W"
                generation = "Cascade/Skylake Xeon W"
            else:
                codename = "Unknown Xeon W"
                generation = "Legacy Xeon W"

            product_collection = "Workstation"  # Product collection
            segment_hint = "workstation"
            xeon_w_detected = True
            logger.debug(
                f"Detected legacy Xeon W workstation processor from brand string: "
                f"codename={codename}, generation={generation}, product_collection={product_collection}"
            )

    # If Xeon W wasn't detected, use standard detection from family/model/stepping
    if not xeon_w_detected:
        # Standard detection from family/model/stepping
        # Pass brand string for Xeon 6 E/P core distinction (model 173)
        codename, generation, product_collection, segment_hint = _detect_generation(family, model, stepping, brand)

    # If not found in mapping, try fallback detection from brand string
    if not codename:
        codename, generation, product_collection, segment_hint = _detect_generation_from_brand(brand)
        if codename:
            logger.debug(
                f"Detected generation from brand string (model not in map): "
                f"codename={codename}, generation={generation}, "
                f"product_collection={product_collection}, segment={segment_hint}"
            )

    # CRITICAL: Check brand string overrides
    brand_lower = brand.lower()

    # Override 1: Check for 14th Gen Core (Raptor Lake Refresh)
    # 14th gen shares CPUID model 183 with 13th gen (Raptor Lake)
    # Differentiate by brand string: i[3579]-14xxx pattern (e.g., i9-14900T, i7-14700K)
    if codename and "core" in brand_lower and "ultra" not in brand_lower:
        if re.search(r"i[3579]-14\d{3}", brand_lower):
            # This is 14th Gen (Raptor Lake Refresh)
            logger.debug(
                f"Detected 14th Gen Core (Raptor Lake Refresh) from brand string - overriding generation: "
                f"original={generation} -> 14th Gen Core, brand={brand}"
            )
            generation = "14th Gen Core"
            if codename == "Raptor Lake-S":
                codename = "Raptor Lake-S Refresh"
            elif codename == "Raptor Lake-P":
                codename = "Raptor Lake-P Refresh"
            elif codename == "Raptor Lake-HX":
                codename = "Raptor Lake-HX Refresh"

    # Override 2: Check for Core Series 2 (non-Ultra)
    # Intel Core Series 2 (non-Ultra) uses Raptor Lake/Bartlett Lake architecture but different branding
    # Key differentiator: "Core 7" vs "Core i7" (no "i" prefix)
    # Examples: Core 7 251TE, Core 7 251E, Core 7 250U, Core 5 220U (not Core i7, Core i5)
    # These share CPUID model numbers with 13th/14th Gen but need separate generation string
    # Note: Segment will be detected by suffix (E/TE=embedded, U/P=mobile) via _detect_segment()
    if codename and "ultra" not in brand_lower:
        # Check for Core 3/5/7/9 pattern WITHOUT "i" prefix (not i3/i5/i7/i9)
        # Pattern: "core" ... whitespace ... digit (3/5/7/9) ... whitespace ... NOT "i"
        # Matches: "Core(TM) 7 251TE" or "Core(TM) 7 250U" but NOT "Core(TM) i7-13700K"
        is_n_series_brand = re.search(r"\bN\d{2,3}\b", brand, re.IGNORECASE) is not None
        if re.search(r"core.*?\s+[3579]\s+(?!i)", brand_lower) and not is_n_series_brand:
            # This is Core Series 2 (new branding without "i")
            logger.debug(
                f"Detected Core Series 2 (non-Ultra) from brand string - overriding generation: "
                f"original={generation} -> Core (Series 2), brand={brand}"
            )
            generation = "Core (Series 2)"
            # Segment will be determined by _detect_segment() based on suffix
            # Keep original codename but update it to reflect "Refresh"
            if "Raptor Lake" in codename:
                codename = codename.replace("Raptor Lake", "Raptor Lake Refresh")

    if codename:
        result["codename"] = codename
        result["generation"] = generation
        result["product_collection"] = product_collection or "Unknown"

        # Determine if CPU is supported using generation comparison
        # Try to use CLI-specific unsupported list (e.g., from ESQ) if available,
        # otherwise fall back to sysagent's default supported list
        unsupported_gens = _get_cli_unsupported_generations()
        if unsupported_gens is not None:
            # Use unsupported list approach (CLI-specific)
            result["is_supported"] = is_generation_supported(
                generation,
                unsupported_generations=unsupported_gens,
                product_collection=product_collection,
                segment=result.get("segment"),
                codename=codename,
            )
        else:
            # Fall back to sysagent's default supported list
            supported_gens = ["Core Ultra (Series 2)", "Core Ultra (Series 3)", "4th Gen", "5th Gen", "Xeon 6"]
            result["is_supported"] = is_generation_supported(generation, supported_gens)

    # Detect vertical segment from brand string and characteristics
    # Special handling for Xeon W: check for embedded suffixes
    if xeon_w_detected:
        # Check if this is an embedded variant (RE, MRE, E suffixes)
        # Examples: W-11865MRE, W-1390E, W-3375RE
        if re.search(r"W-\d+[A-Z]*(?:RE|MRE|E)\b", brand):
            result["segment"] = "embedded"
            logger.debug(f"Detected embedded Xeon W processor (RE/MRE/E suffix): segment=embedded, brand={brand}")
        else:
            # Standard workstation segment for non-embedded Xeon W
            result["segment"] = "workstation"
    else:
        # Standard segment detection for all other CPUs
        # PRIORITY: Use segment_hint from special handling (e.g., Xeon 6 678X workstation)
        # Otherwise, use _detect_segment() for generic detection
        if segment_hint:
            result["segment"] = segment_hint
            logger.debug(f"Using segment hint from special handling: segment={segment_hint}, brand={brand}")
        else:
            segment = _detect_segment(brand, model, core_count)
            if segment:
                result["segment"] = segment

    logger.debug(f"CPU Detection: {result}")
    return result


def _extract_product_collection_from_generation(generation: str) -> str:
    """
    Extract product collection from generation string in CPU_GENERATION_MAP.

    This function parses the generation strings to determine the product collection.
    It's needed because CPU_GENERATION_MAP stores combined information in the generation field.
    This is NOT backward compatibility - it's proper data extraction from the MAP structure.

    Args:
        generation: Generation string from CPU_GENERATION_MAP (e.g., "Core Ultra (Series 2)", "4th Gen Xeon Scalable")

    Returns:
        Product collection name
    """
    if not generation or generation == "Unknown":
        return "Unknown"

    gen_lower = generation.lower()

    # Xeon product collections
    if "xeon" in gen_lower:
        if "workstation" in gen_lower:
            return "Xeon Workstation"
        elif "scalable" in gen_lower:
            return "Xeon Scalable"
        elif "max" in gen_lower:
            return "Xeon Max"
        elif "xeon 6" in gen_lower or "xeon6" in gen_lower:
            return "Xeon 6"
        else:
            return "Xeon"

    # Core product collections
    if "core" in gen_lower:
        if "ultra" in gen_lower:
            return "Core Ultra"
        else:
            return "Core"

    # Atom product collections
    if "atom" in gen_lower:
        if "x7000" in gen_lower:
            return "Atom x7000"
        elif "x6000" in gen_lower:
            return "Atom x6000"
        elif "x5000" in gen_lower:
            return "Atom x5000"
        elif "z-series" in gen_lower:
            return "Atom Z-series"
        else:
            return "Atom"

    # N-series
    if "n-series" in gen_lower:
        return "N-series"

    return "Unknown"


def _detect_generation(
    family: int, model: int, stepping: int, brand: str = ""
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Detect CPU generation from family, model, and stepping.

    Args:
        family: CPU family number
        model: CPU model number
        stepping: CPU stepping number
        brand: CPU brand string (optional, used for Xeon 6 E/P core distinction)

    Returns:
        Tuple of (codename, generation, product_collection, segment_hint) or (None, None, None, None) if not found
    """
    for (fam, mod, step_range), (codename, generation, segment_hint) in CPU_GENERATION_MAP.items():
        if family == fam and model == mod and stepping in step_range:
            # Special handling: N-series model 190 includes both Alder Lake-N and Twin Lake
            if family == 6 and model == 190 and brand:
                if re.search(r"\bN(355|350|250|150)\b", brand, re.IGNORECASE):
                    codename = "Twin Lake"
                else:
                    codename = "Alder Lake-N"
            # Special handling: Xeon 6 model 173 shared by Sierra Forest and Granite Rapids
            # Distinguish by suffix in brand string:
            #   E = Sierra Forest, P = Granite Rapids, X/no-suffix = Granite Rapids WS
            # CRITICAL: Xeon 6 has TWO segments: Server (6XXXP, 6XXXE) AND Workstation (678X)
            # This is different from "Xeon W" which is a separate legacy product collection
            if family == 6 and model == 173 and brand:
                # First check for P/E suffixes (server segment) - these take priority
                # Check if this is a Xeon 6 processor with P suffix (Granite Rapids Server)
                # Examples: 6787P (64 cores), 6730P (32 cores), 6761P (64 cores)
                if re.search(r"Xeon\(R\) \d{4}P\b", brand, re.IGNORECASE):
                    codename = "Granite Rapids"
                    logger.debug(f"Detected Granite Rapids (P-core) Server from brand: {brand}")
                # E suffix processors (Sierra Forest) use the default from CPU_GENERATION_MAP
                elif re.search(r"Xeon\(R\) \d{4}E\b", brand, re.IGNORECASE):
                    logger.debug(f"Detected Sierra Forest (E-core) from brand: {brand}")
                # Finally check for workstation models (678X - no P/E suffix)
                # Examples: 6780 (48 cores, 2.4GHz, workstation segment)
                # Pattern: "Xeon(R) 678[0-9]" without P or E suffix at the end
                elif re.search(r"Xeon\(R\) 678\d(?![PE])", brand, re.IGNORECASE):
                    codename = "Granite Rapids-WS"
                    segment_hint = "workstation"
                    logger.debug(f"Detected Granite Rapids Workstation from brand: {brand}, model 678X")

            # Extract product collection from generation string
            product_collection = _extract_product_collection_from_generation(generation)
            return codename, generation, product_collection, segment_hint

    # Log unknown CPU for future mapping updates
    logger.debug(
        f"Unknown CPU model detected - please report to project maintainer: "
        f"family={family} (0x{family:X}), model={model} (0x{model:X}), stepping={stepping}"
    )
    logger.info(
        "To help improve CPU detection, please report this CPU model to the project maintainer "
        "with your CPU brand string and model information."
    )
    return None, None, None, None


def _detect_segment(brand: str, model: int, core_count: int) -> Optional[str]:
    """
    Detect Intel vertical segment from brand string and CPU characteristics.

    Intel vertical segments represent market categories:
    - server: Xeon Scalable processors for data centers
    - desktop: Core processors for desktop PCs (suffixes: K, KF, F, T)
    - mobile: Core processors for laptops/tablets (suffixes: H, U, P, HX)
    - embedded: IoT/industrial processors (suffixes: E, TE, RE, MRE)
    - workstation: Xeon W and Core Extreme for workstations

    Segment Detection Priority:
    1. Suffix-based detection (highest priority - most specific)
    2. Brand pattern matching
    3. Model + core count characteristics

    Args:
        brand: CPU brand string
        model: CPU model number
        core_count: Physical core count

    Returns:
        Segment name or None if not detected
    """
    # PRIORITY 1: Check suffix patterns first (most specific indicator)
    # EXCEPTION: Skip suffix detection for Xeon processors
    # Xeon 6 uses P/E to indicate core type (P-cores/E-cores), not segment
    # For Core processors, P/E indicate segment (Performance mobile/Embedded)
    is_xeon = "Xeon" in brand

    if not is_xeon:
        # Extract suffix from brand string - check for 1-3 character suffix at end
        # Examples: "251TE" -> "TE", "285K" -> "K", "13900" -> None
        suffix_search = re.search(r"\s(\d+)([A-Z]{1,3})\b", brand)
        if suffix_search:
            suffix = suffix_search.group(2)  # Get suffix (e.g., "TE", "K", "E")

            # Check suffix against all segment patterns
            for segment, patterns in SEGMENT_PATTERNS.items():
                if "suffix_patterns" in patterns and suffix in patterns["suffix_patterns"]:
                    logger.debug(f"Segment detected by suffix '{suffix}': {segment}")
                    return segment

    # PRIORITY 2: Check brand patterns and model/core combinations
    for segment, patterns in SEGMENT_PATTERNS.items():
        # Check brand patterns
        brand_match = any(re.search(pattern, brand, re.IGNORECASE) for pattern in patterns.get("brand_patterns", []))

        # Check model patterns
        model_match = model in patterns.get("model_patterns", [])

        # Check core count constraints
        core_match = True
        if "core_count_max" in patterns:
            core_match = core_count <= patterns["core_count_max"]
        elif "core_count_min" in patterns:
            core_match = core_count >= patterns["core_count_min"]

        # Segment is matched if:
        # - Brand matches (strong signal)
        # - OR model + core count match (hardware characteristics)
        if brand_match or (model_match and core_match):
            return segment

    # Fallback logic based on characteristics
    if core_count >= 16:
        return "server"  # High core count suggests server
    elif core_count >= 8:
        return "desktop"  # Medium core count suggests desktop
    elif core_count <= 4:
        return "embedded"  # Low core count suggests embedded
    else:
        return "mobile"  # Default for middle ground


def normalize_generation_string(generation: str) -> str:
    """
    Normalize CPU generation string for consistent comparison.

    This utility function provides consistent normalization of Intel CPU generation
    strings, making them suitable for comparison and matching operations.

    Handles variations like:
    - "Core Ultra Series 3" vs "Core Ultra (Series 3)" -> "core ultra series 3"
    - "12th Gen Core" vs "12th Gen" -> "12th gen core"
    - "Xeon 6" vs "Xeon6" -> "xeon 6"
    - "5th Gen Xeon Scalable" -> "5th gen xeon"
    - "Xeon-W" vs "Xeon W" -> "xeon-w" or "xeon w"

    Args:
        generation: Generation string to normalize (from detect_cpu_generation_and_segment())
                   Examples: "Core Ultra (Series 2)", "14th Gen Core", "Xeon 6"

    Returns:
        Normalized generation string for comparison (lowercase, standardized format)

    Example:
        >>> normalize_generation_string("Core Ultra (Series 2)")
        "core ultra series 2"
        >>> normalize_generation_string("12th Gen Core")
        "12th gen core"
        >>> normalize_generation_string("Xeon 6")
        "xeon 6"
    """
    if not generation:
        return ""

    # Convert to lowercase for case-insensitive comparison
    gen_lower = generation.lower()

    # Normalize "Core Ultra Series X" format
    # Handles: "core ultra (series 2)" -> "core ultra series 2"
    if "core ultra" in gen_lower and "series" in gen_lower:
        import re

        match = re.search(r"series\s*(\d+)", gen_lower)
        if match:
            return f"core ultra series {match.group(1)}"

    # Normalize "Xth Gen Core" format
    # Handles: "12th gen" -> "12th gen core"
    if "gen core" in gen_lower:
        import re

        match = re.search(r"(\d+)(?:st|nd|rd|th)\s*gen", gen_lower)
        if match:
            return f"{match.group(1)}th gen core"

    # Normalize "Xeon 6" format
    # Handles: "xeon 6" vs "xeon6"
    if "xeon" in gen_lower:
        import re

        # Match "Xeon X" patterns (Xeon 6, Xeon 5, etc.)
        match = re.search(r"xeon\s*(\d+)", gen_lower)
        if match:
            return f"xeon {match.group(1)}"
        # Handle "Xth Gen Xeon Scalable" -> "Xth gen xeon"
        match = re.search(r"(\d+)(?:st|nd|rd|th)\s*gen\s*xeon", gen_lower)
        if match:
            return f"{match.group(1)}th gen xeon"

    return gen_lower


def match_cpu_generations(system_generation: str, target_sku: str) -> bool:
    """
    Check if a target CPU SKU matches the system's CPU generation.

    This is a general-purpose matching function that can be used for:
    - Reference data filtering
    - CPU compatibility checking
    - Generation-based filtering in any context

    Supports ALL Intel processor families:
    - Core Ultra Series (1, 2, 3, etc.)
    - Core i-series (14th, 13th, 12th Gen, etc.)
    - Core Series (non-i, e.g., Core 7 250U)
    - Xeon 6 (new 2024+ naming: Xeon 6XXX)
    - Xeon Scalable (legacy: Xth Gen Xeon Scalable)
    - Xeon W (workstation: Xeon-W or Xeon w-series)
    - Intel Atom (x7000, x6000, x5000, N-series, etc.)
    - Intel N-series (Alder Lake-N, Twin Lake)
    - Pentium, Celeron (all generations)

    The function normalizes both system and target strings for robust comparison,
    handling variations in formatting, capitalization, and naming conventions.

    Args:
        system_generation: System CPU generation string from detect_cpu_generation_and_segment()
                          Examples: "Core Ultra (Series 2)", "14th Gen Core", "Xeon 6"
        target_sku: Target CPU SKU to match against
                   Examples: "Intel Core Ultra 7 255H", "Intel Xeon 6730P", "Intel Atom x7-E3950"

    Returns:
        True if the target SKU matches the system generation, False otherwise

    Example:
        >>> match_cpu_generations("Core Ultra (Series 2)", "Intel Core Ultra 7 255H")
        True
        >>> match_cpu_generations("Xeon 6", "Intel Xeon 6730P")
        True
        >>> match_cpu_generations("14th Gen Core", "Intel Core i7-14700K")
        True
    """
    if not system_generation or not target_sku:
        return False

    # Normalize both for comparison
    sys_gen_norm = normalize_generation_string(system_generation)
    target_lower = target_sku.lower()

    # === CORE ULTRA SERIES MATCHING ===
    # Supports: Core Ultra Series 1, 2, 3, 4, etc.
    if "core ultra series" in sys_gen_norm:
        import re

        sys_match = re.search(r"series\s*(\d+)", sys_gen_norm)
        if sys_match:
            sys_series = sys_match.group(1)

            # Check if target explicitly mentions "Series X"
            target_match = re.search(r"series\s*(\d+)", target_lower)
            if target_match:
                target_series = target_match.group(1)
                return sys_series == target_series

            # Check for processor model numbers that belong to specific series
            # Core Ultra Series 2: 2xxH/K/U (Lunar Lake, Arrow Lake)
            # Core Ultra Series 3: 3xxH (Panther Lake)
            # Core Ultra Series 4+: 4xxH/5xxH/etc. (future)
            if "core ultra" in target_lower:
                # Extract model number (e.g., "255H" -> 255, "285K" -> 285, "358H" -> 358)
                model_match = re.search(r"core ultra\s*(?:\d+\s+)?(\d{3})[hkupe]", target_lower)
                if model_match:
                    model_num = int(model_match.group(1))
                    # Series N: Nxx models (N00-N99)
                    target_series_from_model = str(model_num // 100)
                    return sys_series == target_series_from_model

    # === TRADITIONAL CORE i-SERIES MATCHING ===
    # Supports: 14th Gen Core, 13th Gen Core, 12th Gen Core, etc.
    if "gen core" in sys_gen_norm and "xeon" not in sys_gen_norm:
        import re

        sys_match = re.search(r"(\d+)(?:st|nd|rd|th)\s*gen", sys_gen_norm)
        if sys_match:
            sys_gen_num = sys_match.group(1)
            # Match target with same generation number
            target_match = re.search(r"(\d+)(?:st|nd|rd|th)\s*gen", target_lower)
            if target_match:
                target_gen_num = target_match.group(1)
                return sys_gen_num == target_gen_num

            # Also match processor model numbers (e.g., i7-14700K)
            # 14th Gen: i7-14xxx, 13th Gen: i7-13xxx, etc.
            target_model_match = re.search(r"i[3579]-(\d{2})\d{3}", target_lower)
            if target_model_match:
                target_gen_from_model = target_model_match.group(1)
                return sys_gen_num == target_gen_from_model

    # === CORE SERIES (NON-i) MATCHING ===
    # Supports: Core 7 250U, Core 5 220U (Series 2)
    if "core (series" in sys_gen_norm or "core series" in sys_gen_norm:
        import re

        sys_match = re.search(r"series\s*(\d+)", sys_gen_norm)
        if sys_match:
            # Match target with "Core 7" or "Core 5" pattern (without "i")
            if re.search(r"core\s+[3579]\s+", target_lower) and "core i" not in target_lower:
                return True

    # === XEON 6 MATCHING (2024+ NAMING) ===
    # Supports: Xeon 6XXX (Xeon 6730P, 6761P, 6787P, etc.)
    if "xeon" in sys_gen_norm:
        import re

        # Pattern 1: "Xeon 6" -> matches "Xeon 6XXX" models
        if "xeon 6" in sys_gen_norm or "xeon6" in sys_gen_norm:
            # Match device SKUs with model numbers like 6730P, 6761P, 6787P
            # Pattern: "Xeon" + space + 4-digit model starting with 6
            target_match = re.search(r"xeon\s+6\d{3}[ep]?", target_lower)
            if target_match:
                return True

        # Pattern 2: "Xth Gen Xeon Scalable" -> matches "Xth Gen Xeon" in target
        # Supports: 5th Gen Xeon Scalable, 4th Gen Xeon Scalable, etc.
        sys_gen_match = re.search(r"(\d+)(?:st|nd|rd|th)\s*gen\s*xeon", sys_gen_norm)
        if sys_gen_match:
            sys_gen_num = sys_gen_match.group(1)
            # Match target with same generation number
            target_gen_match = re.search(r"(\d+)(?:st|nd|rd|th)\s*gen\s*xeon", target_lower)
            if target_gen_match:
                target_gen_num = target_gen_match.group(1)
                return sys_gen_num == target_gen_num

            # Also match by Xeon model number inference
            # Example: "Xeon" + number -> extract first digit as generation hint
            target_model_match = re.search(r"xeon\s+(\d)", target_lower)
            if target_model_match:
                target_gen_hint = target_model_match.group(1)
                # This is a weak match - only use if generation hints align
                if sys_gen_num == target_gen_hint:
                    return True

        # Pattern 3: "Xeon-W" or "Xeon w-series" (workstation)
        # IMPORTANT: Xeon-W is a separate product line from Xeon Scalable
        # Must match exactly - no fallback to generic Xeon
        if "xeon-w" in sys_gen_norm or "xeon w" in sys_gen_norm:
            if "xeon-w" in target_lower or "xeon w" in target_lower:
                return True
            # Don't fallback to generic Xeon match for Xeon-W
            return False

    # === INTEL ATOM MATCHING ===
    # Supports: Atom x7000, x6000, x5000, N-series, Z-series, etc.
    if "atom" in sys_gen_norm:
        import re

        # Check if target is an Atom processor
        if "atom" in target_lower:
            # Try to match series (x7000, x6000, x5000, N-series, Z-series)
            sys_series_match = re.search(r"atom\s*(x\d{4}|[nz]-series)", sys_gen_norm)
            target_series_match = re.search(r"atom\s*(x\d{4}|x[357]-|[nz]\d{3})", target_lower)

            if sys_series_match and target_series_match:
                sys_series = sys_series_match.group(1)
                target_series = target_series_match.group(1)
                # Match if series align (e.g., "x7000" -> "x7-")
                if sys_series[0] == target_series[0]:  # First character match (x, n, z)
                    return True
            elif "atom" in target_lower:
                # Generic Atom match if series not specified
                return True

    # === INTEL N-SERIES MATCHING ===
    # Supports: Intel N100, N200, Intel Core i3-N305, etc.
    if "n-series" in sys_gen_norm:
        # Match any target with N-series pattern
        import re

        if re.search(r"\bn\d{3}\b", target_lower) or re.search(r"i[357]-n\d{3}", target_lower):
            return True

    # === PENTIUM / CELERON MATCHING ===
    # Supports: All Pentium and Celeron generations
    if "pentium" in sys_gen_norm:
        if "pentium" in target_lower:
            return True

    if "celeron" in sys_gen_norm:
        if "celeron" in target_lower:
            return True

    # === FALLBACK: GENERIC NAME MATCHING ===
    # If no specific pattern matched, try generic word-based matching
    # This catches edge cases and future processor families
    import re

    # Extract key processor family words from system generation
    sys_words = set(re.findall(r"\b[a-z]{4,}\b", sys_gen_norm))
    target_words = set(re.findall(r"\b[a-z]{4,}\b", target_lower))

    # Remove common words that don't indicate processor family
    common_words = {"intel", "processor", "series", "core", "ultra", "generation", "with"}
    sys_words -= common_words
    target_words -= common_words

    # If significant overlap in processor family words, consider it a match
    if sys_words and target_words and len(sys_words & target_words) >= 1:
        return True

    return False
