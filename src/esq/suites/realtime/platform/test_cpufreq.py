# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Realtime Platform — RT Core CPU Frequency Scaling Governor Check.

Verifies that the CPU frequency scaling governor on isolated RT cores is set
to ``performance`` (or another expected governor), preventing the kernel from
dynamically scaling CPU frequency during RT workloads.

Frequency scaling introduces determinism hazards on RT cores:

- The CPU may be at a low frequency when the RT task wakes, requiring a P-state
  ramp-up that adds 10–100 µs of unexpected latency before the thread can execute
  at full speed.
- Interrupt delivery latency spikes coincide with hardware P-state transitions.
- ``schedutil`` and ``ondemand`` governors cause bursty frequency changes that are
  correlated with load rather than RT deadlines.

Keeping RT cores on the ``performance`` governor (or using ``intel_pstate=disable``
with ``cpufreq.default_governor=performance``) ensures the CPU is always at its
maximum allowed frequency when an RT thread is scheduled, eliminating this jitter
source.

Requires ``isolcpus=<cpus>`` in the kernel boot parameters; skips when absent.
RT cores are read from ``isolcpus``; ``rt_cpu_ids`` in the profile restricts the
check to a subset (useful when only some isolated cores run RT tasks).

Key metric: ``rt_cores_cpufreq_performance``
    1 = all nominated RT cores use the performance scaling governor.
    0 = at least one RT core uses a different governor.

Secondary metrics collected per RT core (reported in extended metadata):
    ``scaling_governor``, ``scaling_max_freq_khz``, ``scaling_min_freq_khz``,
    ``cpuinfo_max_freq_khz``, ``energy_perf_bias`` (when available).

Development override: set ``ENV_SUITE_CPUFREQ_RT_CPU_IDS=<cpu-list>``
(e.g. ``2,3``) to bypass the ``isolcpus`` requirement and check specific CPUs
directly without modifying the kernel boot parameters.
"""

import csv
import io
import logging
import os
from pathlib import Path

import allure
import pytest
from sysagent.utils.core import Metrics, Result
from sysagent.utils.system.cpu.cpufreq import collect_cpufreq_info
from sysagent.utils.system.cpu.topology import parse_cpu_list

logger = logging.getLogger(__name__)

# Default expected governor — the ``performance`` governor disables P-state
# transitions entirely and keeps the CPU at its maximum scaling_max_freq.
_DEFAULT_EXPECTED_GOVERNOR = "performance"

# EPP string → EPB-equivalent integer mapping (modern intel_pstate HWP mode).
# Mirrors the mapping in test_epb.py; only the four strings the kernel actually
# returns on read are listed ("default" is write-only, never returned).
_EPP_TO_EPB: dict[str, int] = {
    "performance": 0,
    "balance_performance": 4,
    "balance_power": 8,
    "power": 15,
}

# CSV column order for the RT-cores frequency report attachment.
_CSV_FIELDS = [
    "cpu",
    "scaling_governor",
    "scaling_max_freq_khz",
    "scaling_min_freq_khz",
    "scaling_cur_freq_khz",
    "cpuinfo_max_freq_khz",
    "energy_perf_bias",
    "cpufreq_performance",
]

# Per-suite environment override — bypasses the ``isolcpus`` requirement for
# development and CI where the kernel may not have RT isolation configured.
# Example: ENV_SUITE_CPUFREQ_RT_CPU_IDS=2,3 esq run -p profile.suite.realtime.platform
_RT_CPU_IDS_ENV_VAR = "ENV_SUITE_CPUFREQ_RT_CPU_IDS"


# ---------------------------------------------------------------------------
# Global EPP / EPB helpers
# ---------------------------------------------------------------------------


def _resolve_energy_perf_pref(cpufreq_info: dict) -> int:
    """Derive the global energy/performance preference integer from cpufreq_info.

    Priority order (EPP first — the modern interface on Intel P-state HWP):

    1. ``global_epp_policy`` — EPP string from Intel P-state HWP mode
       (``policy0/energy_performance_preference``), converted to an EPB-
       equivalent integer via ``_EPP_TO_EPB``.
    2. ``global_energy_perf_bias`` — legacy EPB integer read from
       ``cpu0/power/energy_perf_bias``.

    Returns -1 when neither interface is available (VM, non-Intel CPU, or
    kernel without intel_epb / intel_pstate drivers).
    """
    epp = cpufreq_info.get("global_epp_policy")
    if epp and epp in _EPP_TO_EPB:
        return _EPP_TO_EPB[epp]
    epb = cpufreq_info.get("global_energy_perf_bias")
    if epb is not None:
        return int(epb)
    return -1


# ---------------------------------------------------------------------------
# RT CPU auto-detection helpers
# ---------------------------------------------------------------------------

_CMDLINE_PATH = Path("/proc/cmdline")


def _parse_isolcpus() -> list[int]:
    """Parse the ``isolcpus`` kernel boot parameter from ``/proc/cmdline``.

    Handles both the simple form (``isolcpus=2,3``) and the extended form
    with type specifiers (``isolcpus=domain:managed_irq,2-3``).  Type
    specifiers contain letters and are filtered out; only numeric CPU-range
    tokens (e.g. ``2``, ``0-3``) are parsed.

    Returns a sorted list of isolated CPU indices, or an empty list when
    the parameter is absent or the file is unreadable.
    """
    try:
        cmdline = _CMDLINE_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return []

    for token in cmdline.split():
        if not token.startswith("isolcpus="):
            continue
        value = token[len("isolcpus=") :]
        # Type specifiers (e.g. "domain", "managed_irq", "nohz") contain
        # only letters/underscores; CPU list parts start with a digit.
        cpu_parts = [p for p in value.split(",") if p and p[0].isdigit()]
        if not cpu_parts:
            return []
        try:
            return sorted(parse_cpu_list(",".join(cpu_parts)))
        except Exception:
            # Inline fallback when topology import is unavailable.
            cpus: list[int] = []
            for part in cpu_parts:
                if "-" in part:
                    try:
                        lo, hi = part.split("-", 1)
                        cpus.extend(range(int(lo), int(hi) + 1))
                    except ValueError:
                        pass
                else:
                    try:
                        cpus.append(int(part))
                    except ValueError:
                        pass
            return sorted(set(cpus))
    return []


def _resolve_rt_cpu_ids(configs: dict) -> tuple[list[int] | None, str]:
    """Resolve which logical CPUs to validate.

    Priority order:

    1. ``ENV_SUITE_CPUFREQ_RT_CPU_IDS`` env var — bypasses isolcpus requirement
       (for development / CI without a real RT kernel configuration).
    2. ``isolcpus`` kernel boot parameter — required when env var is not set;
       returns ``(None, reason)`` so the caller can skip the test.
    3. ``rt_cpu_ids`` profile parameter — optional subset of isolated CPUs.

    Returns ``(cpu_ids, source)``; *source* is included in the log and report.
    """
    # Development/CI override — bypass isolcpus requirement.
    env_val = os.environ.get(_RT_CPU_IDS_ENV_VAR, "").strip()
    if env_val:
        try:
            ids = sorted(parse_cpu_list(env_val))
            if ids:
                return ids, f"{_RT_CPU_IDS_ENV_VAR}={env_val} (env override)"
        except Exception:
            logger.warning("Invalid %s value %r — ignored", _RT_CPU_IDS_ENV_VAR, env_val)

    isolated = _parse_isolcpus()
    if not isolated:
        return None, "isolcpus not found in /proc/cmdline"

    # Optional profile override: restrict to a specific subset of isolated cores.
    rt_cpu_ids_raw = configs.get("rt_cpu_ids", [])
    if rt_cpu_ids_raw:
        try:
            ids = [int(c) for c in rt_cpu_ids_raw]
            if ids:
                return ids, f"rt_cpu_ids profile parameter: {ids} (isolcpus={isolated})"
        except (TypeError, ValueError):
            logger.warning(
                "Invalid rt_cpu_ids in profile (%r) — falling back to all isolcpus",
                rt_cpu_ids_raw,
            )

    return isolated, f"all isolcpus: {isolated}"


# ---------------------------------------------------------------------------
# Check logic
# ---------------------------------------------------------------------------


def _check_rt_core_cpufreq(
    cpufreq_info: dict,
    rt_cpu_ids: list[int],
    expected_governor: str,
    source: str = "",
) -> tuple[int, str, list[dict]]:
    """Check whether RT CPUs use the expected CPU frequency scaling governor.

    Args:
        cpufreq_info:      Dict returned by ``collect_cpufreq_info()``.
        rt_cpu_ids:        Nominated RT CPU indices to validate.
        expected_governor: The governor all RT cores must use (e.g. ``"performance"``).
        source:            Human-readable description of how rt_cpu_ids were resolved.

    Returns:
        ``(rt_cores_cpufreq_performance, report_text, core_rows)``

        ``rt_cores_cpufreq_performance`` is 1 when every nominated RT core uses
        the expected governor, 0 when any core differs.

        ``core_rows`` is a list of per-core dicts ready for CSV export.
    """
    per_cpu = cpufreq_info.get("per_cpu", {})

    global_epp = cpufreq_info.get("global_epp_policy")
    global_epb = cpufreq_info.get("global_energy_perf_bias")
    epb_int = _resolve_energy_perf_pref(cpufreq_info)
    if global_epp is not None:
        epb_source = f"EPP={global_epp!r} ({epb_int})"
    elif global_epb is not None:
        epb_source = f"EPB={global_epb} (legacy sysfs)"
    else:
        epb_source = "n/a"

    lines: list[str] = [
        "RT Core CPU Frequency Scaling Governor Check",
        "=" * 54,
        f"RT CPUs checked        : {sorted(rt_cpu_ids)}",
    ]
    if source:
        lines.append(f"CPU source             : {source}")
    lines += [
        f"Expected governor      : {expected_governor}",
        f"Global energy perf pref: {epb_source}",
        "",
    ]

    if not cpufreq_info.get("available", False):
        lines += [
            "WARNING: cpufreq sysfs is not available on this system.",
            "  The cpufreq driver may not be loaded, or the kernel was built",
            "  without CONFIG_CPU_FREQ.  Cannot verify scaling governor.",
            "",
            "rt_cores_cpufreq_performance = 0",
        ]
        return 0, "\n".join(lines), []

    violations: list[int] = []
    core_rows: list[dict] = []

    for cpu_id in sorted(rt_cpu_ids):
        entry = per_cpu.get(str(cpu_id))
        if entry is None:
            lines.append(f"CPU {cpu_id}  — no cpufreq data (CPU offline or driver not loaded)")
            violations.append(cpu_id)
            core_rows.append(
                {
                    "cpu": cpu_id,
                    "scaling_governor": None,
                    "scaling_max_freq_khz": None,
                    "scaling_min_freq_khz": None,
                    "scaling_cur_freq_khz": None,
                    "cpuinfo_max_freq_khz": None,
                    "energy_perf_bias": "n/a",
                    "cpufreq_performance": 0,
                }
            )
            continue

        governor = entry.get("scaling_governor") or "unknown"
        max_freq = entry.get("scaling_max_freq_khz")
        min_freq = entry.get("scaling_min_freq_khz")
        cur_freq = entry.get("scaling_cur_freq_khz")
        hw_max = entry.get("cpuinfo_max_freq_khz")
        epb = entry.get("energy_perf_bias")

        ok = governor == expected_governor
        if not ok:
            violations.append(cpu_id)

        verdict = "[OK]" if ok else f"MISMATCH — expected '{expected_governor}'"

        max_mhz = f"{max_freq // 1000} MHz" if max_freq else "?"
        min_mhz = f"{min_freq // 1000} MHz" if min_freq else "?"
        epb_str = str(epb) if epb is not None else "n/a"

        lines.append(
            f"CPU {cpu_id:<3d}  governor={governor:<16s}  "
            f"scaling_max={max_mhz:<12s}  scaling_min={min_mhz}  "
            f"epb={epb_str}  {verdict}"
        )

        core_rows.append(
            {
                "cpu": cpu_id,
                "scaling_governor": governor,
                "scaling_max_freq_khz": max_freq,
                "scaling_min_freq_khz": min_freq,
                "scaling_cur_freq_khz": cur_freq,
                "cpuinfo_max_freq_khz": hw_max,
                "energy_perf_bias": epb if epb is not None else "n/a",
                "cpufreq_performance": 1 if ok else 0,
            }
        )

    all_ok = len(violations) == 0
    lines += [
        "",
        "-" * 54,
        f"rt_cores_cpufreq_performance = {1 if all_ok else 0}",
    ]
    if violations:
        lines.append(
            f"Violations ({len(violations)} core(s) not using '{expected_governor}' governor): CPUs {violations}"
        )

    return (1 if all_ok else 0), "\n".join(lines), core_rows


# ---------------------------------------------------------------------------
# Allure attachment helpers
# ---------------------------------------------------------------------------


def _attach_csv(content: str, name: str) -> None:
    """Attach CSV content to the Allure report; silently skips when empty."""
    if not content:
        return
    try:
        allure.attach(content, name=name, attachment_type=allure.attachment_type.CSV)
    except Exception as exc:
        logger.debug("Failed to attach CSV %s: %s", name, exc)


def _attach_text(content: str, name: str) -> None:
    """Attach plain-text content to the Allure report; silently skips when empty."""
    if not content:
        return
    try:
        allure.attach(content, name=name, attachment_type=allure.attachment_type.TEXT)
    except Exception as exc:
        logger.debug("Failed to attach text %s: %s", name, exc)


def _rows_to_csv(rows: list[dict]) -> str:
    """Serialise core rows to CSV."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test: Global energy / performance preference check
# ---------------------------------------------------------------------------


@allure.title("Energy Performance Preference")
def test_energy_perf_pref_check(
    request,
    configs,
    cached_result,
    cache_result,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """
    Read the global CPU energy/performance preference via cpufreq sysfs.

    Uses ``collect_cpufreq_info()`` — the same shared utility as the governor
    check — so no additional sysfs reads are required.

    Priority order (EPP first — the modern interface on newer Intel platforms):
    1. EPP string from ``intel_pstate`` HWP mode
       (``policy0/energy_performance_preference``) converted to an integer
       via the standard four-value mapping.
    2. Legacy EPB integer from ``cpu0/power/energy_perf_bias`` (fallback).

    Key metric: ``energy_perf_pref``
        0  = performance (maximum CPU performance, no power-saving).
        15 = power (maximum power-saving).
        -1 = unavailable (neither EPP nor EPB sysfs present).

    Does not require ``isolcpus`` — works on any platform.
    """
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info("Starting Global Energy/Performance Bias Check: %s", test_display_name)

    validate_system_requirements_from_configs(configs)

    result = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    def _run_epp_check():
        cpufreq = collect_cpufreq_info()
        epp_val = _resolve_energy_perf_pref(cpufreq)

        epp = cpufreq.get("global_epp_policy")
        epb_raw = cpufreq.get("global_energy_perf_bias")
        if epp is not None and epp in _EPP_TO_EPB:
            source = f"EPP ({epp!r} \u2192 {epp_val})"
        elif epb_raw is not None:
            source = f"legacy EPB ({epb_raw})"
        else:
            source = "unavailable"

        report_text = "\n".join(
            [
                "Energy Performance Preference",
                "=" * 40,
                f"EPP (modern intel_pstate HWP): {epp if epp is not None else 'n/a'}",
                f"EPB (legacy sysfs fallback)  : {epb_raw if epb_raw is not None else 'n/a'}",
                f"Resolved value               : {epp_val}  (0=performance, 15=power-saving, -1=unavailable)",
                f"Source                       : {source}",
            ]
        )

        logger.info(
            "Energy performance preference: epp_val=%d  source=%s",
            epp_val,
            source,
        )

        return Result(
            name=f"{test_id} - {test_display_name}",
            metrics={
                "energy_perf_pref": Metrics(
                    unit=None,
                    value=epp_val,
                    is_key_metric=True,
                ),
            },
            extended_metadata={
                "epp_report": report_text,
                "global_epp_policy": epp,
                "global_energy_perf_bias": epb_raw,
            },
            metadata={"status": epp_val != -1},
        )

    try:
        result = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=_run_epp_check,
            test_name=test_name,
            configs=configs,
        )
    except KeyboardInterrupt:
        failure_message = "Interrupt detected during Energy Performance Preference check"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as exc:
        test_failed = True
        failure_message = f"Unexpected error during Energy Performance Preference check: {exc}"
        logger.error(failure_message, exc_info=True)

    if result is None:
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "Energy performance preference check did not complete"},
            metrics={},
        )

    # Re-attach from extended_metadata — recreated on both fresh runs and cache hits.
    if result and result.extended_metadata:
        report = result.extended_metadata.get("epp_report")
        if report:
            _attach_text(report, "epp_report.txt")

    try:
        validate_test_results(
            test_name=test_name,
            results=result,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as validation_error:
        logger.error("Validation failed: %s", validation_error)

    try:
        summarize_test_results(
            results=result,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception:
        logger.exception("Test result summarization failed")

    logger.info("Energy Performance Preference Check completed: %s", test_display_name)

    if test_interrupted:
        if configs.get("labels", {}).get("type") == "qualification":
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)


# ---------------------------------------------------------------------------
# Test: RT core CPU frequency scaling governor check
# ---------------------------------------------------------------------------


@allure.title("RT Core CPU Frequency Scaling Governor")
def test_cpufreq_rt_check(
    request,
    configs,
    cached_result,
    cache_result,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """
    Verify that the CPU frequency scaling governor on isolated RT cores is set
    to ``performance`` (or the configured expected governor).

    Frequency scaling causes P-state transitions that add 10–100 µs of
    latency before an RT thread can execute at full CPU speed.  Pinning RT
    cores to the ``performance`` governor eliminates this jitter source.

    Skips when ``isolcpus`` is absent from the kernel boot parameters.
    Key metric: ``rt_cores_cpufreq_performance`` (1 = all RT cores use the
    performance governor, 0 = at least one core differs).
    """
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info("Starting RT Core CPU Frequency Scaling Governor Check: %s", test_display_name)

    validate_system_requirements_from_configs(configs)

    # Resolve RT CPU IDs — requires isolcpus kernel parameter.
    # rt_cpu_ids profile param narrows the check to a specific subset.
    rt_cpu_ids, cpu_source = _resolve_rt_cpu_ids(configs)
    if rt_cpu_ids is None:
        pytest.skip("RT cpufreq check skipped: isolcpus is not configured in kernel boot parameters")

    expected_governor = configs.get("rt_cpufreq_expected_governor", _DEFAULT_EXPECTED_GOVERNOR)

    logger.info(
        "RT core cpufreq check: resolved rt_cpu_ids=%s via %s  expected_governor=%s",
        rt_cpu_ids,
        cpu_source,
        expected_governor,
    )

    result = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    def _run_rt_check():
        cpufreq = collect_cpufreq_info()
        governor_ok, report_text, core_rows = _check_rt_core_cpufreq(
            cpufreq_info=cpufreq,
            rt_cpu_ids=rt_cpu_ids,
            expected_governor=expected_governor,
            source=cpu_source,
        )
        rt_csv = _rows_to_csv(core_rows) if core_rows else ""

        logger.info(
            "RT core cpufreq check: rt_cpu_ids=%s  source=%s  expected=%s  rt_cores_cpufreq_performance=%d",
            rt_cpu_ids,
            cpu_source,
            expected_governor,
            governor_ok,
        )

        return Result(
            name=f"{test_id} - {test_display_name}",
            metrics={
                "rt_cores_cpufreq_performance": Metrics(
                    unit=None,
                    value=int(governor_ok),
                    is_key_metric=True,
                ),
            },
            extended_metadata={
                "rt_cpufreq_report": report_text,
                "rt_cpufreq_csv": rt_csv,
                "rt_cpufreq_global_governors": cpufreq.get("global_scaling_governors", []),
                "rt_cpufreq_global_epp_policy": cpufreq.get("global_epp_policy"),
                "rt_cpufreq_global_energy_perf_bias": cpufreq.get("global_energy_perf_bias"),
            },
            metadata={"status": True},
        )

    try:
        result = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=_run_rt_check,
            test_name=test_name,
            configs=configs,
        )
    except KeyboardInterrupt:
        failure_message = "Interrupt detected during RT Core CPU Frequency Scaling check"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as exc:
        test_failed = True
        failure_message = f"Unexpected error during RT Core CPU Frequency Scaling check: {exc}"
        logger.error(failure_message, exc_info=True)

    if result is None:
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "RT core cpufreq check did not complete"},
            metrics={},
        )

    # Re-attach from extended_metadata — recreated on both fresh runs and cache hits.
    if result and result.extended_metadata:
        report = result.extended_metadata.get("rt_cpufreq_report")
        if report:
            _attach_text(report, "rt_cpufreq_report.txt")
        rt_csv = result.extended_metadata.get("rt_cpufreq_csv")
        if rt_csv:
            _attach_csv(rt_csv, "rt_cpufreq_cores.csv")

    try:
        validate_test_results(
            test_name=test_name,
            results=result,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as validation_error:
        logger.error("Validation failed: %s", validation_error)

    try:
        summarize_test_results(
            results=result,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception:
        logger.exception("Test result summarization failed")

    logger.info("RT Core CPU Frequency Scaling Governor Check completed: %s", test_display_name)

    if test_interrupted:
        if configs.get("labels", {}).get("type") == "qualification":
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
