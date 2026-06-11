# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Consolidated CPU test — data collection and optional qualification.

Provides a single test function, ``test_cpu``, that covers CPU identification,
generation detection, core counts, and clock frequencies in one configurable
test.  All metrics are always collected; the active validation is driven
entirely by profile parameters via ``kpi_refs``.

CPU compatibility qualification modes
--------------------------------------
Allowlist modes (may be combined — CPU passes when ANY one matches):

- processor_numbers : SKU-level allowlist — CPU passes when its brand string
  contains one of the listed processor number tokens (case-insensitive,
  word-boundary aware).  E.g. ``["155H", "165H", "285K"]``.

- exact_generations : strict allowlist — no age-forward comparison.  Each
  entry may be a string, a dict (with generation, codename, segment,
  product_collection), or a 2/3-element list/tuple.

- supported_generations : minimum-and-above allowlist — CPU passes when its
  generation matches or is newer than any entry within the same product family.

Combining allowlists (OR logic):
  When more than one allowlist is set, the CPU passes if it satisfies *any*
  of them.  Example: accept all Core Ultra Series 2 mobile CPUs *plus* a
  fixed list of specific processor numbers from other generations.

Denylist mode (exclusive — cannot be combined with allowlists):

- unsupported_generations : CPU passes when its generation is NOT in the list
  and is not older than any listed entry.

When none of the above is supplied, ``cpu_compatibility`` defaults to 1
(audit / data-collection mode).
"""

import logging
import re
from typing import Any, List

import allure
from sysagent.utils.core import Metrics, Result
from sysagent.utils.system import SystemInfoCache
from sysagent.utils.system.cpu import is_generation_supported

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_cpu_info():
    """Return the cpu dict from the system info cache."""
    return SystemInfoCache().get_hardware_info().get("cpu", {})


def _normalize_generation_list(raw_list: List[Any]) -> List[Any]:
    """Convert YAML-parsed list entries for use with is_generation_supported.

    YAML sequences become Python lists but is_generation_supported expects
    tuples for (generation, product_collection[, segment]) forms.  This
    helper converts list entries to tuples while leaving strings and dicts
    unchanged.
    """
    return [tuple(e) if isinstance(e, list) else e for e in raw_list]


def _extract_processor_number(cpu_brand: str) -> str:
    """Extract the processor model number from the CPU brand string.

    Strips trademark markers and returns the last space-separated token,
    which for Intel processors corresponds to the processor number
    (e.g. ``"285K"``, ``"155H"``, ``"6226R"``, ``"i9-14900K"``).

    Args:
        cpu_brand: Full CPU brand string from the system info cache.

    Returns:
        Processor number string, or ``"Unknown"`` when the brand is empty.
    """
    cleaned = cpu_brand.replace("(R)", "").replace("(TM)", "").strip()
    parts = cleaned.split()
    return parts[-1] if parts else "Unknown"


def _is_processor_number_match(cpu_brand: str, processor_numbers: List[str]) -> bool:
    """Return True when the brand string contains any listed processor number.

    Matching is case-insensitive and word-boundary aware so that ``"155H"``
    does not accidentally match ``"1550H"`` or ``"2155H"``.

    Args:
        cpu_brand: CPU brand string from the system info cache.
        processor_numbers: Processor number tokens to search for.

    Returns:
        True if any token matches, False otherwise.
    """
    brand_lower = cpu_brand.lower()
    for pn in processor_numbers:
        if not pn or not isinstance(pn, str):
            continue
        pattern = r"(?<![\w-])" + re.escape(pn.lower()) + r"(?![\w-])"
        if re.search(pattern, brand_lower):
            return True
    return False


def _is_exact_match(cpu_info: dict, exact_list: List[Any]) -> bool:
    """Return True when the CPU exactly matches any entry in exact_list.

    Unlike the age-forward comparison used by supported_generations, this
    performs direct matching only — no neighbouring generation is implicitly
    accepted.

    Supported entry formats:
        - str           : matches generation string directly
        - dict          : all specified keys (generation, codename,
                          product_collection, segment) must match
        - list / tuple  : (generation,), (generation, product_collection),
                          or (generation, product_collection, segment)

    Args:
        cpu_info: CPU info dict from SystemInfoCache (hardware["cpu"]).
        exact_list: Normalised allowlist of entries.

    Returns:
        True if any entry matches, False otherwise.
    """
    gen_info = cpu_info.get("generation_info", {})
    cpu_gen = gen_info.get("generation", "Unknown")
    codename = gen_info.get("codename")
    prod_coll = gen_info.get("product_collection")
    segment = gen_info.get("segment")

    for entry in exact_list:
        if isinstance(entry, str):
            if cpu_gen == entry:
                return True
        elif isinstance(entry, dict):
            if not entry:
                continue
            ok = True
            if "generation" in entry and cpu_gen != entry["generation"]:
                ok = False
            if ok and "codename" in entry and codename != entry["codename"]:
                ok = False
            if ok and "product_collection" in entry and prod_coll != entry["product_collection"]:
                ok = False
            if ok and "segment" in entry and segment != entry["segment"]:
                ok = False
            if ok:
                return True
        elif isinstance(entry, tuple) and len(entry) >= 1:
            if cpu_gen != entry[0]:
                continue
            if len(entry) >= 2 and prod_coll != entry[1]:
                continue
            if len(entry) >= 3 and segment != entry[2]:
                continue
            return True
    return False


# ---------------------------------------------------------------------------
# test_cpu
# ---------------------------------------------------------------------------


@allure.title("CPU")
@allure.description("Collects CPU related metrics")
def test_cpu(
    request,
    configs,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
):
    """Collect CPU data and optionally enforce generation or SKU qualification.

    All CPU metrics are always collected.  KPI validation is controlled by the
    ``kpi_refs`` parameter in the active profile.

    Configurable profile parameters
    --------------------------------
    processor_numbers : list, optional (highest priority)
        SKU-level allowlist (e.g. ``["155H", "165H", "285K"]``).  Matched as
        word-boundary tokens in the brand string (case-insensitive).

    exact_generations : list, optional
        Strict allowlist — no age-forward comparison.  Each entry may be a
        string, dict (with generation, codename, segment, product_collection),
        or a 2/3-element list/tuple.

    supported_generations : list, optional
        Minimum-and-above allowlist.  CPU passes if its generation matches or
        is newer than an entry within the same product family.

    unsupported_generations : list, optional
        Denylist.  CPU passes if its generation is not in the list and is not
        older than any listed entry.

    display_name : str, optional
        Human-readable test name shown in reports.

    test_id : str, optional
        Identifier used in test reports.

    timeout : int, optional
        Maximum test duration in seconds (default 60).

    Metrics reported
    ----------------
    cpu_compatibility      : 1 (pass) or 0 (fail) for the CPU compatibility check.
    physical_cores        : Physical (non-hyperthreaded) core count.
    logical_cores         : Logical core count (includes hyperthreading).
    sockets               : Number of physical CPU sockets.
    max_frequency_mhz     : Maximum reported CPU frequency in MHz.
    min_frequency_mhz     : Minimum (base) CPU frequency in MHz.

    Metadata always captured
    ------------------------
    processor_number: Last token of the CPU brand string (e.g. "285K", "155H").
    cpu_generation, codename, product_collection, segment: from generation_info.
    """
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    logger.info(f"Starting CPU test: {test_display_name}")

    # Step 1: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Step 2: Collect CPU data
    cpu_info = _get_cpu_info()
    generation_info = cpu_info.get("generation_info", {})

    cpu_brand = cpu_info.get("brand", "Unknown")
    architecture = cpu_info.get("architecture", "Unknown")
    cpu_generation = generation_info.get("generation", "Unknown")
    codename = generation_info.get("codename", "Unknown")
    product_collection = generation_info.get("product_collection", "Unknown")
    segment = generation_info.get("segment", "Unknown")
    processor_number = _extract_processor_number(cpu_brand)

    physical_cores = int(cpu_info.get("count", 0) or 0)
    logical_cores = int(cpu_info.get("logical_count", 0) or 0)
    sockets = int(cpu_info.get("sockets", 1) or 1)

    frequency = cpu_info.get("frequency", {}) or {}
    max_frequency_mhz = float(frequency.get("max") or 0.0)
    min_frequency_mhz = float(frequency.get("min") or 0.0)

    logger.info(f"CPU: {cpu_brand} | Processor number: {processor_number}")
    logger.info(f"Generation: {cpu_generation} | Codename: {codename} | Segment: {segment}")
    logger.info(
        f"Cores: {physical_cores}P / {logical_cores}L ({sockets} socket(s)) | "
        f"Freq: max={max_frequency_mhz:.0f} MHz, min={min_frequency_mhz:.0f} MHz"
    )

    # Step 3: CPU compatibility check.
    #
    # Allowlists (processor_numbers, exact_generations, supported_generations) are
    # OR-combined: a CPU passes when it satisfies ANY one of the configured lists.
    # This allows a profile to accept, for example, an entire generation of mobile
    # CPUs plus a handful of specific desktop processor numbers in one test entry.
    #
    # The denylist (unsupported_generations) is mutually exclusive with allowlists
    # because its semantics ("not in list") cannot be meaningfully OR'd with them.
    pn_list = configs.get("processor_numbers", [])
    exact_generations = _normalize_generation_list(configs.get("exact_generations", []))
    supported_generations = _normalize_generation_list(configs.get("supported_generations", []))
    unsupported_generations = _normalize_generation_list(configs.get("unsupported_generations", []))

    allowlist_results = []
    active_modes = []

    if pn_list:
        allowlist_results.append(_is_processor_number_match(cpu_brand, pn_list))
        active_modes.append("processor_numbers")

    if exact_generations:
        allowlist_results.append(_is_exact_match(cpu_info, exact_generations))
        active_modes.append("exact_generations")

    if supported_generations:
        allowlist_results.append(
            is_generation_supported(
                cpu_generation=cpu_generation,
                supported_generations=supported_generations,
                product_collection=product_collection,
                segment=segment,
                codename=codename,
            )
        )
        active_modes.append("supported_generations")

    if allowlist_results:
        # OR: pass when any allowlist criterion is satisfied.
        is_supported = any(allowlist_results)
        check_mode = "+".join(active_modes)
    elif unsupported_generations:
        is_supported = is_generation_supported(
            cpu_generation=cpu_generation,
            unsupported_generations=unsupported_generations,
            product_collection=product_collection,
            segment=segment,
            codename=codename,
        )
        check_mode = "unsupported_generations"
    else:
        is_supported = True
        check_mode = "audit"

    cpu_compatibility_value = 1 if is_supported else 0
    logger.info(
        f"CPU compatibility check [{check_mode}]: {'PASS' if is_supported else 'FAIL'} "
        f"(cpu_compatibility={cpu_compatibility_value})"
    )

    # Step 4: Determine which metrics are key for this test run.
    # Priority: kpi_refs (qualification profiles) > key_metrics (suite profiles).
    # Only the explicitly requested metrics are included in the result; all others
    # are omitted to keep reports concise when multiple CPU tests run together.
    kpi_refs_set = set(configs.get("kpi_refs", []))
    key_metrics_set = set(configs.get("key_metrics", []))
    active_key_metrics = kpi_refs_set | key_metrics_set

    # Full pool — only entries whose name is in active_key_metrics are kept.
    _all_metrics = {
        "cpu_compatibility": Metrics(unit=None, value=cpu_compatibility_value, is_key_metric=True),
        "physical_cores": Metrics(unit="cores", value=physical_cores, is_key_metric=True),
        "logical_cores": Metrics(unit="cores", value=logical_cores, is_key_metric=True),
        "sockets": Metrics(unit="sockets", value=sockets, is_key_metric=True),
        "max_frequency_mhz": Metrics(unit="MHz", value=max_frequency_mhz, is_key_metric=True),
        "min_frequency_mhz": Metrics(unit="MHz", value=min_frequency_mhz, is_key_metric=True),
    }
    filtered_metrics = {k: v for k, v in _all_metrics.items() if k in active_key_metrics}

    # Step 5: Build result
    results = Result(
        name=test_name,
        parameters={
            "Test ID": test_id,
            "CPU Brand": cpu_brand,
            "Processor Number": processor_number,
            "Architecture": architecture,
            "Generation": cpu_generation,
            "Codename": codename,
            "Product Collection": product_collection,
            "Segment": segment,
            "Sockets": sockets,
            "Check Mode": check_mode,
        },
        metrics=filtered_metrics,
        metadata={
            "processor_number": processor_number,
            "cpu_generation": cpu_generation,
            "codename": codename,
            "product_collection": product_collection,
            "segment": segment,
        },
    )

    # Step 6: KPI validation (only active when kpi_refs is set in profile)
    validate_test_results(
        results=results,
        configs=configs,
        get_kpi_config=get_kpi_config,
        test_name=test_name,
    )

    # Step 7: Summarise
    summarize_test_results(
        results=results,
        configs=configs,
        get_kpi_config=get_kpi_config,
        test_name=test_name,
    )
