# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System Memory Health Test.

Checks the health of installed RAM and helps identify faulty memory modules
while the OS is running. Unlike bootloader tools such as MemTest86 (which run
before the OS and can cover 100% of physical RAM), an in-OS test cannot touch
memory already reserved by the kernel and userspace. This suite therefore
combines complementary, non-destructive signals so a faulty module can still be
spotted at runtime:

  * EDAC / RAS ECC error counters (``/sys/devices/system/edac``) passively scan
    hardware-detected correctable (CE) and uncorrectable (UE) errors per memory
    controller and per DIMM. On ECC systems this is the strongest signal for
    pinpointing a specific faulty module by its label/location.
  * ``memtester`` actively writes and verifies bit patterns over the memory it
    can allocate from userspace. Coverage is limited to allocatable pages
    (OS-reserved memory is excluded) but it exercises real cells and flags
    read-back mismatches.

This suite is data-collection oriented: it records what it finds and never
fails a run based on discovered errors. Only an interrupt or an unexpected
runtime error terminates the test as a failure. The raw command/probe output is
attached to the Allure report during the execution phase.
"""

import logging
import os
import re
import resource
import time
from pathlib import Path

import allure
import pytest
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result, check_command_available, run_command

logger = logging.getLogger(__name__)

# Allow-list of supported probe types. The profile selects one per test via the
# ``check_type`` param; anything else is rejected to avoid dispatching on
# unvalidated input.
_VALID_CHECK_TYPES = ("edac", "memtester")

_EDAC_BASE = "/sys/devices/system/edac/mc"

# Per-suite environment overrides for memtester parameters. Named after this
# test file ("memory_health") so they never collide with other suite knobs. Export these
# at runtime to retune without editing profiles, e.g.
#   ENV_SUITE_MEMORY_HEALTH_MEMTESTER_SIZE_MB=1024  esq run --profile ...
#   ENV_SUITE_MEMORY_HEALTH_MEMTESTER_ITERATIONS=2  esq run --profile ...
_MEMTESTER_SIZE_MB_ENV_VAR = "ENV_SUITE_MEMORY_HEALTH_MEMTESTER_SIZE_MB"
_MEMTESTER_ITERATIONS_ENV_VAR = "ENV_SUITE_MEMORY_HEALTH_MEMTESTER_ITERATIONS"
# Sentinel values accepted in profile ``memtester_size_mb`` to enable dynamic
# sizing. When one of these is detected the test reads ``MemAvailable`` from
# /proc/meminfo and allocates all available memory minus a safety reserve.
# ``ENV_SUITE_MEMORY_HEALTH_MEMTESTER_SIZE_MB`` always wins over dynamic mode.
_MEMTESTER_SIZE_DYNAMIC_SENTINELS = frozenset(("auto", "dynamic", "available", "all"))


def _read_int(path: Path) -> int | None:
    """Read a single integer from a sysfs file, or None if unavailable."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _read_str(path: Path) -> str:
    """Read a trimmed string from a sysfs file, or '' if unavailable."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _attach_text(content: str, name: str) -> None:
    """Attach plain-text probe output to the Allure report if non-empty."""
    if not content:
        return
    try:
        allure.attach(content, name=name, attachment_type=allure.attachment_type.TEXT)
    except Exception as error:  # pragma: no cover - report backend failure
        logger.warning(f"Failed to attach '{name}': {error}")


def _write_and_return(output_dir: str, filename: str, content: str) -> str:
    """Persist probe output to the results directory (best effort)."""
    if not content:
        return ""
    file_path = os.path.join(output_dir, filename)
    try:
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as error:
        logger.warning(f"Failed to write '{file_path}': {error}")
        return ""
    return file_path


def _collect_edac() -> tuple[dict[str, int], list[str], str]:
    """
    Read EDAC ECC error counters from sysfs.

    Returns:
        Tuple of (summary, faulty_dimms, text_dump) where summary carries the
        aggregate counters, faulty_dimms lists labels/locations of DIMMs with a
        non-zero error count, and text_dump is a human-readable report for the
        Allure attachment.
    """
    base = Path(_EDAC_BASE)
    lines: list[str] = []
    faulty: list[str] = []
    summary = {
        "available": 0,
        "memory_controllers": 0,
        "total_correctable_errors": 0,
        "total_uncorrectable_errors": 0,
        "dimms_total": 0,
        "dimms_with_errors": 0,
    }

    if not base.exists():
        lines.append("EDAC sysfs not present at /sys/devices/system/edac/mc.")
        lines.append("The platform may lack ECC memory or the EDAC driver is not loaded.")
        return summary, faulty, "\n".join(lines)

    summary["available"] = 1
    controllers = sorted(p for p in base.glob("mc*") if p.is_dir())
    summary["memory_controllers"] = len(controllers)

    lines.append("EDAC Memory Error Report")
    lines.append("=" * 60)

    for mc in controllers:
        mc_ce = _read_int(mc / "ce_count")
        mc_ue = _read_int(mc / "ue_count")
        mc_type = _read_str(mc / "mc_name")
        summary["total_correctable_errors"] += mc_ce or 0
        summary["total_uncorrectable_errors"] += mc_ue or 0

        lines.append("")
        lines.append(f"[{mc.name}] {mc_type or 'unknown controller'}")
        lines.append(f"  correctable (CE)   : {mc_ce if mc_ce is not None else 'n/a'}")
        lines.append(f"  uncorrectable (UE) : {mc_ue if mc_ue is not None else 'n/a'}")

        dimms = sorted(p for p in mc.glob("dimm*") if p.is_dir())
        # Legacy kernels expose per-rank csrow* nodes instead of dimm* nodes.
        if not dimms:
            dimms = sorted(p for p in mc.glob("csrow*") if p.is_dir())

        for dimm in dimms:
            summary["dimms_total"] += 1
            label = _read_str(dimm / "dimm_label")
            location = _read_str(dimm / "dimm_location")
            mem_type = _read_str(dimm / "dimm_mem_type")
            size = _read_str(dimm / "size")
            dce = _read_int(dimm / "dimm_ce_count")
            due = _read_int(dimm / "dimm_ue_count")
            if dce is None and due is None:
                # csrow* layout: sum the per-channel counters.
                dce = sum((_read_int(f) or 0) for f in dimm.glob("ch*_ce_count")) or None
                label = label or dimm.name

            errors = (dce or 0) + (due or 0)
            if errors > 0:
                summary["dimms_with_errors"] += 1
                tag = label or location or f"{mc.name}/{dimm.name}"
                faulty.append(tag)

            descriptor = label or location or dimm.name
            lines.append(
                f"    {dimm.name}: label='{descriptor}' type={mem_type or 'n/a'} "
                f"size={size or 'n/a'} CE={dce if dce is not None else 0} "
                f"UE={due if due is not None else 0}"
            )

    lines.append("")
    lines.append("-" * 60)
    lines.append(
        f"Totals: controllers={summary['memory_controllers']} "
        f"dimms={summary['dimms_total']} "
        f"CE={summary['total_correctable_errors']} "
        f"UE={summary['total_uncorrectable_errors']} "
        f"faulty_dimms={summary['dimms_with_errors']}"
    )

    # RAS daemon, when installed, aggregates the same errors by DIMM label and
    # is a valuable cross-check for identifying a failing module.
    if check_command_available("ras-mc-ctl"):
        for args, header in (
            (["ras-mc-ctl", "--summary"], "ras-mc-ctl --summary"),
            (["ras-mc-ctl", "--error-count"], "ras-mc-ctl --error-count"),
        ):
            probe = run_command(args, timeout=15)
            if probe and probe.stdout:
                lines.append("")
                lines.append(f"# {header}")
                lines.append(probe.stdout.strip())

    return summary, faulty, "\n".join(lines)


def _run_memtester(size_mb: int, iterations: int, timeout: int) -> tuple[dict[str, int], str]:
    """
    Run memtester over a userspace-allocated region and parse the outcome.

    Returns:
        Tuple of (summary, output) with error counts and the raw command output.
    """
    summary = {
        "available": 0,
        "tested_mb": 0,
        "iterations": iterations,
        "errors": -1,
        "passed": 0,
        "test_duration_seconds": -1.0,
        "testing_rate_mb_per_sec": -1.0,
    }

    if not check_command_available("memtester"):
        note = "memtester is not installed on the host.\nRun: sudo scripts/system-setup.sh"
        return summary, note

    summary["available"] = 1
    command = ["memtester", f"{size_mb}M", str(iterations)]
    _t_start = time.monotonic()
    # stream_output=True prints each line to the logger in real-time so the
    # terminal shows memtester progress as it runs. stderr is merged into
    # stdout in pipe mode, so combined holds the full output for parsing.
    probe = run_command(command, timeout=timeout, stream_output=True)
    _t_elapsed = time.monotonic() - _t_start
    # In pipe/stream mode stderr is merged into stdout; probe.stderr is empty.
    stdout = probe.stdout or ""
    combined = f"$ {' '.join(command)}\n\n{stdout}".strip()

    # Every subtest that fails prints a line containing "FAILURE"; a clean run
    # reports each check as "ok". Count failures as the health signal.
    failure_count = len(re.findall(r"FAILURE", stdout, flags=re.IGNORECASE))
    # Allocation errors appear in the merged stdout when using stream mode.
    allocation_failed = "too many bytes" in stdout.lower() or "cannot allocate" in stdout.lower()
    unexpected_error = probe.returncode != 0 and failure_count == 0 and not allocation_failed

    if allocation_failed:
        # memtester ran but could not lock the requested region.
        summary["tested_mb"] = 0
        summary["errors"] = -1
        summary["passed"] = 0
        summary["fail_reason"] = "allocation_failed"
    elif unexpected_error:
        # Tool exited non-zero for an unknown reason; nothing was actually tested.
        summary["tested_mb"] = 0
        summary["errors"] = -1
        summary["passed"] = 0
        summary["fail_reason"] = f"unexpected_exit_code_{probe.returncode}"
    else:
        summary["tested_mb"] = size_mb
        summary["errors"] = failure_count
        summary["passed"] = 1 if (probe.returncode == 0 and failure_count == 0) else 0
        summary["fail_reason"] = ""
        summary["test_duration_seconds"] = round(_t_elapsed, 2)
        # testing_rate_mb_per_sec reflects how fast memtester scanned the
        # allocated region — this is NOT a memory bandwidth measurement.
        if _t_elapsed > 0:
            summary["testing_rate_mb_per_sec"] = round(size_mb / _t_elapsed, 2)

    return summary, combined


def _safe_int(value, default: int) -> int:
    """Parse an integer safely with a fallback, clamped to at least 1."""
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return default


def _get_mlock_limit_mb() -> int:
    """Return the process locked-memory limit (RLIMIT_MEMLOCK) in MB.

    ``memtester`` calls ``mlock()`` on every page it allocates.  If the process
    limit is lower than the requested size the kernel returns ENOMEM and
    memtester falls into an extremely slow self-reduction loop (4 KB per step).

    Returns 0 when the limit is ``RLIM_INFINITY`` (unlimited) or cannot be read,
    meaning the caller should not apply a cap.
    """
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if soft == resource.RLIM_INFINITY:
            return 0  # unlimited — no cap needed
        return max(0, soft // (1024 * 1024))
    except (OSError, AttributeError):
        return 0


def _get_available_memory_mb(reserve_mb: int, reserve_percent: float) -> int:
    """Return a safe memtester allocation size (MB) based on current free memory.

    Reads ``MemAvailable`` from ``/proc/meminfo``, which accounts for reclaimable
    page-cache and kernel data structures — it is more accurate than ``MemFree``
    for estimating how much memory a new allocation can safely consume without
    forcing the system to swap.

    A safety reserve is subtracted to leave headroom for the OS and other running
    processes. The reserve applied is the *larger* of ``reserve_mb`` and
    ``reserve_percent``% of the available memory, so both a fixed floor and a
    proportional buffer are honoured.

    The result is additionally capped at the process's locked-memory limit
    (``ulimit -l`` / ``RLIMIT_MEMLOCK``), because ``memtester`` requires every
    allocated page to be ``mlock``-ed.  Requesting more than this limit causes
    memtester to emit ``too many pages, reducing...`` and stall in an extremely
    slow self-reduction loop (4 KB steps from hundreds of GB).
    To test the full machine RAM, raise the limit first::

        ulimit -l unlimited

    Returns 0 when ``/proc/meminfo`` is unreadable so callers can fall back.
    """
    available_kb = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
                    break
    except (OSError, ValueError, IndexError):
        logger.warning("Could not read /proc/meminfo; dynamic memtester size unavailable")
        return 0

    if available_kb <= 0:
        return 0

    available_mb = available_kb // 1024
    percent_reserve_mb = int(available_mb * reserve_percent / 100.0)
    actual_reserve_mb = max(reserve_mb, percent_reserve_mb)
    safe_mb = max(1, available_mb - actual_reserve_mb)

    # Cap at the locked-memory limit so memtester's mlock() call succeeds.
    mlock_limit_mb = _get_mlock_limit_mb()
    mlock_capped = mlock_limit_mb > 0 and safe_mb > mlock_limit_mb
    if mlock_capped:
        logger.warning(
            f"memtester allocation capped from {safe_mb} MB to mlock limit "
            f"{mlock_limit_mb} MB (ulimit -l). "
            "To test more memory run: ulimit -l unlimited"
        )
        safe_mb = mlock_limit_mb

    logger.info(
        f"Dynamic memtester sizing: {available_mb} MB available, "
        f"reserve={actual_reserve_mb} MB (fixed={reserve_mb} MB, "
        f"{reserve_percent:.0f}%={percent_reserve_mb} MB)"
        + (f", mlock cap={mlock_limit_mb} MB" if mlock_capped else "")
        + f" -> allocating {safe_mb} MB"
    )
    return safe_mb


def _resolve_memtester_size_mb(configs: dict) -> int:
    """Resolve the memtester region size (MB), honoring env override and dynamic mode.

    Priority order:

    1. ``ENV_SUITE_MEMORY_HEALTH_MEMTESTER_SIZE_MB`` env var — always wins, including dynamic mode:
         - A sentinel value (``auto``, ``dynamic``, ``available``, ``all``) activates
           dynamic sizing even when the profile specifies a static integer.
         - An integer value pins the allocation to that many MB.
         - Any other non-integer value is ignored with a warning and falls through.
    2. Dynamic detection — active when profile ``memtester_size_mb`` is one of
       the sentinel strings above or is ≤ 0.
       Reads ``MemAvailable`` from ``/proc/meminfo`` and subtracts a safety
       reserve. Reserve is controlled by two optional profile params:

         ``memtester_dynamic_reserve_mb``      — fixed floor (default 512 MB)
         ``memtester_dynamic_reserve_percent`` — proportional buffer (default 10 %)

       The larger of the two is applied.  Falls back to 256 MB if
       ``/proc/meminfo`` is unavailable.
    3. Static numeric value from profile ``memtester_size_mb``.
    4. 256 MB built-in default.
    """

    def _run_dynamic(source: str) -> int:
        """Detect available memory and return a safe allocation size."""
        reserve_mb = max(_safe_int(configs.get("memtester_dynamic_reserve_mb", 512), 512), 0)
        try:
            reserve_pct = float(configs.get("memtester_dynamic_reserve_percent", 10.0))
        except (TypeError, ValueError):
            reserve_pct = 10.0
        dynamic_mb = _get_available_memory_mb(reserve_mb=reserve_mb, reserve_percent=reserve_pct)
        if dynamic_mb > 0:
            logger.info(f"Dynamic memtester size ({source}): {dynamic_mb} MB")
            return dynamic_mb
        logger.warning(f"Dynamic memory detection failed ({source}); using 256 MB fallback")
        return 256

    # --- 1. Env var always wins ---
    raw_override = os.environ.get(_MEMTESTER_SIZE_MB_ENV_VAR)
    if raw_override is not None:
        stripped = str(raw_override).strip()
        # Sentinel strings in the env var activate dynamic mode, overriding
        # a static profile value without requiring a profile edit.
        if stripped.lower() in _MEMTESTER_SIZE_DYNAMIC_SENTINELS:
            logger.info(f"Dynamic mode requested via {_MEMTESTER_SIZE_MB_ENV_VAR}={raw_override!r}")
            return _run_dynamic(source=_MEMTESTER_SIZE_MB_ENV_VAR)
        try:
            override = max(int(stripped), 1)
            logger.info(f"memtester size override from {_MEMTESTER_SIZE_MB_ENV_VAR}: {override}MB")
            return override
        except (TypeError, ValueError):
            logger.warning(
                f"Ignoring non-integer {_MEMTESTER_SIZE_MB_ENV_VAR}={raw_override!r}; "
                "falling back to profile/dynamic value"
            )

    # --- 2. Dynamic mode (profile-level sentinel or non-positive integer) ---
    profile_value = configs.get("memtester_size_mb", 256)
    is_dynamic = (
        isinstance(profile_value, str) and profile_value.strip().lower() in _MEMTESTER_SIZE_DYNAMIC_SENTINELS
    ) or (isinstance(profile_value, (int, float)) and int(profile_value) <= 0)

    if is_dynamic:
        return _run_dynamic(source="profile memtester_size_mb")

    # --- 3 & 4. Static numeric value or built-in default ---
    return max(_safe_int(profile_value, 256), 1)


def _resolve_memtester_iterations(configs: dict) -> int:
    """Resolve the memtester iteration count, honoring the per-suite env override.

    Priority: ``ENV_SUITE_MEMORY_HEALTH_MEMTESTER_ITERATIONS`` env var > profile
    ``memtester_iterations`` > 1 default. Non-integer overrides are ignored
    with a warning. The resolved value is clamped to at least 1.
    """
    default_iter = max(_safe_int(configs.get("memtester_iterations", 1), 1), 1)

    raw_override = os.environ.get(_MEMTESTER_ITERATIONS_ENV_VAR)
    if raw_override is None:
        return default_iter

    try:
        override = max(int(str(raw_override).strip()), 1)
    except (TypeError, ValueError):
        logger.warning(f"Ignoring non-integer {_MEMTESTER_ITERATIONS_ENV_VAR}={raw_override!r}; using {default_iter}")
        return default_iter

    logger.info(f"memtester iterations override from {_MEMTESTER_ITERATIONS_ENV_VAR}: {override}")
    return override


@allure.title("System Memory Health")
def test_memory_health(
    request,
    configs,
    cached_result,
    cache_result,
    execute_test_with_cache,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
):
    """
    Check RAM health and surface potentially faulty modules at runtime.

    The probe is selected by the ``check_type`` param (``edac`` or
    ``memtester``). Each probe extracts metrics (including a key metric) and
    attaches its raw output to the Allure report during execution. As a
    data-collection test it does not fail on discovered errors; only interrupts
    and unexpected runtime errors surface as failures.
    """
    # Step 1: Extract parameters
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    description = configs.get("description")
    if description:
        allure.dynamic.description(description)

    check_type = str(configs.get("check_type", "edac")).strip().lower()
    if check_type not in _VALID_CHECK_TYPES:
        pytest.fail(f"Unsupported check_type '{check_type}'. Expected one of: {', '.join(_VALID_CHECK_TYPES)}")

    memtester_size_mb = _resolve_memtester_size_mb(configs)
    memtester_iterations = _resolve_memtester_iterations(configs)
    # Propagate the resolved values (env var or dynamic) back into configs so the
    # cache key reflects the parameters that were actually used.  This prevents
    # a run with ENV_SUITE_MEMORY_HEALTH_MEMTESTER_SIZE_MB=auto from getting a cache hit from a
    # previous run that used a different profile-static size (mirrors the same
    # pattern used in test_stress_ng: configs["stress_duration_seconds"] = duration).
    configs["memtester_size_mb"] = memtester_size_mb
    configs["memtester_iterations"] = memtester_iterations
    timeout = _safe_int(configs.get("timeout", 300), 300)

    logger.info(f"Starting System Memory Health test: {test_display_name} (check_type={check_type})")

    # Step 2: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Step 3: Setup directories with path sanitization (break taint chain)
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    core_data_resolved = str(Path(core_data_dir_tainted).resolve())
    core_data_dir = "".join(char for char in core_data_resolved)

    expected_base = Path(os.getcwd()).resolve()
    if not Path(core_data_dir).resolve().is_relative_to(expected_base):
        core_data_dir = os.path.join(os.getcwd(), "esq_data")

    data_dir = os.path.join(core_data_dir, "data", "suites", "system", "memory")
    results_dir = os.path.join(data_dir, "results", test_id)
    results_resolved = str(Path(results_dir).resolve())
    results_dir = "".join(char for char in results_resolved)

    os.makedirs(results_dir, mode=0o770, exist_ok=True)
    ensure_dir_permissions(results_dir, uid=os.getuid(), gid=os.getgid(), mode=0o770)

    # Step 3.5: Pre-flight — fail immediately if the required tool is absent.
    # EDAC is a kernel/sysfs interface (no external tool needed).
    # memtester is an external binary that must be installed.
    if check_type == "memtester" and not check_command_available("memtester"):
        pytest.fail("memtester is not installed; run: sudo scripts/system-setup.sh")

    # State for clean termination on interrupt/error.
    result = None
    test_interrupted = False
    test_failed = False
    failure_message = ""
    is_qualification = configs.get("labels", {}).get("type") == "qualification"

    def _collect_memory_health() -> Result:
        """Run the selected probe, attach its output, and build the result."""
        metrics: dict[str, Metrics] = {}
        extended: dict[str, object] = {"memory_output_dir": results_dir, "check_type": check_type}

        if check_type == "edac":
            with allure.step("EDAC / RAS ECC error scan"):
                summary, faulty, dump = _collect_edac()
                _attach_text(dump, f"{test_id}_edac_report.txt")
                _write_and_return(results_dir, "edac_report.txt", dump)

            available = bool(summary["available"])
            metrics = {
                "edac_available": Metrics(value=available, is_key_metric=False),
                "memory_controllers": Metrics(unit="count", value=summary["memory_controllers"], is_key_metric=False),
                "correctable_errors": Metrics(
                    unit="errors", value=summary["total_correctable_errors"], is_key_metric=False
                ),
                # UE is the strongest faulty-module signal: 0 means healthy.
                "uncorrectable_errors": Metrics(
                    unit="errors", value=summary["total_uncorrectable_errors"], is_key_metric=True
                ),
                "dimms_with_errors": Metrics(unit="count", value=summary["dimms_with_errors"], is_key_metric=False),
            }
            extended["faulty_dimms"] = faulty
            if not available:
                message = "EDAC not available (no ECC support or driver not loaded)"
            elif summary["dimms_with_errors"] > 0:
                message = f"ECC errors detected on {summary['dimms_with_errors']} module(s): {', '.join(faulty)}"
            else:
                message = (
                    f"No ECC errors across {summary['dimms_total']} module(s) on "
                    f"{summary['memory_controllers']} controller(s)"
                )

        elif check_type == "memtester":
            with allure.step(f"memtester active pattern test ({memtester_size_mb}MB x {memtester_iterations})"):
                summary, output = _run_memtester(memtester_size_mb, memtester_iterations, timeout)
                # Output is streamed to the terminal in real-time; skip the
                # Allure attachment to avoid garbled control characters from
                # memtester's progress output. Raw output is still saved to disk.
                _write_and_return(results_dir, "memtester.txt", output)

            available = bool(summary["available"])
            # tested_mb > 0 means memtester actually ran and completed the region.
            # errors == -1 means it did not complete (allocation or unexpected exit).
            ran = summary["tested_mb"] > 0
            metrics = {
                "tested_mb": Metrics(unit="MB", value=summary["tested_mb"], is_key_metric=False),
                "iterations": Metrics(unit="count", value=summary["iterations"], is_key_metric=False),
                # Error count is the key signal: 0 means the tested region is clean.
                # -1 means the test did not complete (allocation failure or tool error).
                "memory_errors": Metrics(unit="errors", value=summary["errors"], is_key_metric=True),
                "memtester_passed": Metrics(value=bool(summary["passed"]), is_key_metric=False),
                # How long memtester ran (-1 when it did not complete the test region).
                "test_duration_seconds": Metrics(
                    unit="seconds", value=summary["test_duration_seconds"], is_key_metric=False
                ),
                # How fast memtester scanned the allocated region. This is the
                # testing throughput, NOT a memory bandwidth measurement.
                "testing_rate_mb_per_sec": Metrics(
                    unit="MB/s", value=summary["testing_rate_mb_per_sec"], is_key_metric=False
                ),
            }
            fail_reason = summary.get("fail_reason", "")
            if not ran and fail_reason == "allocation_failed":
                message = f"memtester could not allocate {memtester_size_mb}MB; reduce memtester_size_mb or run as root"
            elif not ran and fail_reason:
                message = (
                    f"memtester exited unexpectedly ({fail_reason.replace('_', ' ')}); "
                    "check stderr output in the attached log"
                )
            elif summary["errors"] > 0:
                message = f"memtester found {summary['errors']} error(s) in {summary['tested_mb']}MB"
            else:
                _dur = summary["test_duration_seconds"]
                _rate = summary["testing_rate_mb_per_sec"]
                message = (
                    f"memtester passed: {summary['tested_mb']}MB verified over "
                    f"{summary['iterations']} iteration(s) in {_dur:.1f}s "
                    f"(testing rate: {_rate:.2f} MB/s)"
                )
            # Status: False when the tool could not complete the test run.
            memtester_status = ran

        logger.info(message)

        # memtester status depends on whether the tool actually ran to completion.
        # EDAC degrades gracefully and is always True.
        result_status = memtester_status if check_type == "memtester" else True

        return Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": result_status},
            extended_metadata={**extended, "message": message},
            metrics=metrics,
        )

    # Step 4: Execute probe with caching, surfacing interrupts/errors cleanly
    try:
        result = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=_collect_memory_health,
            test_name=test_name,
            configs=configs,
        )
    except KeyboardInterrupt:
        failure_message = "Interrupt detected during memory health test execution"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during memory health test execution: {e!s}"
        logger.error(failure_message, exc_info=True)

    # Ensure a result object always exists so the test terminates cleanly even
    # when interrupted before the probe produced a result.
    if result is None:
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={
                "memory_output_dir": results_dir,
                "message": failure_message or "Memory health test did not complete",
            },
            metrics={},
        )

    # When memtester itself failed to complete the test run (timeout, allocation
    # failure, unexpected exit), the result status is False and the test must
    # also fail.  This is distinct from discovering memory errors during a
    # successful run, which is data-collection behaviour (no pytest failure).
    if check_type == "memtester" and not test_interrupted and not test_failed:
        if not result.metadata.get("status", True):
            test_failed = True
            failure_message = result.extended_metadata.get(
                "message", "memtester did not complete the test successfully"
            )
            logger.error(f"memtester execution failed: {failure_message}")

    # Step 5: Always validate and summarize so the result is recorded
    try:
        validate_test_results(
            test_name=test_name,
            results=result,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as validation_error:
        logger.error(f"Validation failed: {validation_error}")

    try:
        summarize_test_results(
            results=result,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as summary_error:
        logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)

    # Step 6: Cache results — only when the test completed without interruption
    # or unexpected failure.  Caching an interrupted/failed placeholder result
    # (status=False, empty metrics) would cause subsequent runs to find it in
    # the cache and report a false "passed" status without re-executing the test.
    if not test_interrupted and not test_failed:
        cache_result(result)

    logger.info(f"Memory health test completed: {test_display_name}")

    # Terminate cleanly: surface interrupts/errors as a proper test outcome.
    if test_interrupted:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
