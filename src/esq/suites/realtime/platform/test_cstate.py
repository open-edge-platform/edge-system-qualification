# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Realtime Platform — RT Core C-State Disabled Check.

Verifies that CPU C-states with non-zero exit latency are disabled on the
nominated RT CPU cores.  C-states introduce scheduling jitter: waking from a
sleep state takes non-deterministic time, delaying RT threads after every
interrupt.  Keeping RT cores in C0/POLL eliminates this jitter source.

Requires ``isolcpus=<cpus>`` in the kernel boot parameters; skips when absent.
RT cores are read from ``isolcpus``; ``rt_cpu_ids`` restricts the check to a
subset (e.g. when more cores are isolated than used for RT tasks).

Key metric: ``rt_cores_cstate_disabled``
    1 = all C-states above the latency threshold disabled (jitter source eliminated).
    0 = one or more C-states still active on RT cores.

Development override: set ``ENV_SUITE_CSTATE_RT_CPU_IDS=<cpu-list>`` (e.g. ``2,3``)
to bypass the ``isolcpus`` requirement and check specific CPUs directly.
"""

import csv
import io
import logging
import os
from pathlib import Path

import allure
import pytest
from sysagent.utils.core import Metrics, Result
from sysagent.utils.system.cpu.cpuidle import collect_cpuidle_info
from sysagent.utils.system.cpu.topology import parse_cpu_list

logger = logging.getLogger(__name__)

# CSV column order for the RT-cores subset attachment.
# state_name is excluded from CSV (kept internally for the report table only).
# idle_time_pct is provided by the shared cpuidle module.
_CSV_FIELDS = ["cpu", "state_id", "disabled", "latency_us", "usage", "time_us", "idle_time_pct"]

# Per-suite environment override — bypasses the isolcpus requirement for
# development and CI.  Named after this test file ("cstate") per convention.
# Example: ENV_SUITE_CSTATE_RT_CPU_IDS=2,3 esq run -p profile.suite.realtime.platform
_RT_CPU_IDS_ENV_VAR = "ENV_SUITE_CSTATE_RT_CPU_IDS"


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
            # Inline fallback if topology import failed
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
    1. ``ENV_SUITE_CSTATE_RT_CPU_IDS`` env var — bypasses isolcpus requirement
       (for development / CI without a real RT kernel configuration).
    2. ``isolcpus`` kernel boot parameter — required when env var is not set;
       returns ``(None, reason)`` so the caller can skip the test.
    3. ``rt_cpu_ids`` profile parameter — optional subset of the isolated CPUs.

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
            logger.warning("Invalid rt_cpu_ids in profile (%r) — falling back to all isolcpus", rt_cpu_ids_raw)

    return isolated, f"all isolcpus: {isolated}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rows_to_csv(rows: list[dict]) -> str:
    """Serialise state rows to CSV; state_name excluded via extrasaction='ignore'."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# RT core check
# ---------------------------------------------------------------------------


def _check_rt_core_cstates(
    states: list[dict],
    rt_cpu_ids: list[int],
    max_latency_us: int,
    source: str = "",
) -> tuple[int, str]:
    """Check whether RT CPUs have all C-states above ``max_latency_us`` disabled.

    Args:
        states: Enriched state rows from ``collect_cpuidle_info()["states"]``.
        rt_cpu_ids: Nominated RT CPU indices to validate.
        max_latency_us: States with latency strictly above this must be disabled.
        source: Human-readable string describing how rt_cpu_ids were resolved
                (included in the report header for traceability).

    Returns:
        (rt_cores_cstate_disabled, report_text)
        rt_cores_cstate_disabled is 1 when every qualifying C-state is
        disabled on every nominated RT CPU, 0 when any remain enabled.
    """
    rt_set = set(rt_cpu_ids)
    rt_rows = [r for r in states if r["cpu"] in rt_set]

    lines: list[str] = [
        "RT Core C-State Disabled Check",
        "=" * 50,
        f"RT CPUs checked        : {sorted(rt_cpu_ids)}",
        f"CPU source             : {source}" if source else "",
        f"Max permitted latency  : {max_latency_us} µs",
        f"  (C-states with latency > {max_latency_us} µs must be disabled)",
        "",
    ]
    # Remove blank placeholder lines (empty source string produces one)
    lines = [l for l in lines if l != ""] + [""]

    if not rt_rows:
        lines.append("WARNING: No cpuidle data found for the specified RT CPUs.")
        lines += ["", "rt_cores_cstate_disabled = 0"]
        return 0, "\n".join(lines)

    violations: list[dict] = []
    for cpu_id in sorted(rt_cpu_ids):
        cpu_rows = sorted([r for r in rt_rows if r["cpu"] == cpu_id], key=lambda r: r["state_id"])
        lines.append(f"CPU {cpu_id}")
        for r in cpu_rows:
            if r["latency_us"] <= max_latency_us:
                verdict = "ok (at or below threshold)"
            elif r["disabled"]:
                verdict = "disabled [OK]"
            else:
                verdict = "ENABLED — should be disabled"
                violations.append(r)
            lines.append(
                f"  state{r['state_id']} {r['state_name']:<8s}  "
                f"latency={r['latency_us']:>5d} µs  "
                f"disabled={r['disabled']}  {verdict}"
            )
        lines.append("")

    all_ok = len(violations) == 0
    lines += ["-" * 50, f"rt_cores_cstate_disabled = {1 if all_ok else 0}"]
    if violations:
        lines.append(f"Violations ({len(violations)} state(s) still enabled on RT core(s)):")
        for v in violations:
            lines.append(f"  CPU{v['cpu']}  {v['state_name']}  latency={v['latency_us']} µs")

    return (1 if all_ok else 0), "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Test: RT core C-state disabled check
# ---------------------------------------------------------------------------


@allure.title("RT Core C-State Disabled")
def test_cstate_rt_check(
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
    Verify that CPU C-states with non-zero exit latency are disabled on
    the nominated RT CPU cores.

    C-states cause scheduling jitter: waking from a sleep state takes
    non-deterministic time, delaying the RT thread after every interrupt.
    Keeping RT cores in C0/POLL eliminates this jitter source.

    Skips when ``isolcpus`` is absent from the kernel boot parameters.
    Key metric: ``rt_cores_cstate_disabled`` (1 = jitter source eliminated,
    0 = C-states still active on RT cores).
    """
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info("Starting RT Core C-State Disabled Check: %s", test_display_name)

    validate_system_requirements_from_configs(configs)

    # Resolve RT CPU IDs — requires isolcpus kernel parameter.
    # rt_cpu_ids profile param narrows the check to a specific subset.
    rt_cpu_ids, cpu_source = _resolve_rt_cpu_ids(configs)
    if rt_cpu_ids is None:
        pytest.skip("RT C-state check skipped: isolcpus is not configured in kernel boot parameters")

    max_latency_us = int(configs.get("rt_cstate_max_latency_us", 0))

    logger.info(
        "RT core C-state check: resolved rt_cpu_ids=%s via %s",
        rt_cpu_ids,
        cpu_source,
    )

    result = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    def _run_rt_check():
        cpuidle = collect_cpuidle_info()
        states = cpuidle["states"]
        disabled_val, report_text = _check_rt_core_cstates(states, rt_cpu_ids, max_latency_us, source=cpu_source)

        # CSV for the RT cores only — subset of the full per-CPU per-state data.
        rt_rows = [r for r in states if r["cpu"] in set(rt_cpu_ids)]
        rt_csv = _rows_to_csv(rt_rows) if rt_rows else ""

        logger.info(
            "RT core C-state check: rt_cpu_ids=%s  source=%s  max_latency_us=%d  rt_cores_cstate_disabled=%d",
            rt_cpu_ids,
            cpu_source,
            max_latency_us,
            disabled_val,
        )

        return Result(
            name=f"{test_id} - {test_display_name}",
            metrics={
                "rt_cores_cstate_disabled": Metrics(
                    unit=None,
                    value=int(disabled_val),
                    is_key_metric=True,
                ),
            },
            extended_metadata={
                "rt_cstate_report": report_text,
                "rt_cstate_csv": rt_csv,
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
        failure_message = "Interrupt detected during RT Core C-State check"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as exc:
        test_failed = True
        failure_message = f"Unexpected error during RT Core C-State check: {exc}"
        logger.error(failure_message, exc_info=True)

    if result is None:
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "RT core C-state check did not complete"},
            metrics={},
        )

    # Re-attach from extended_metadata — recreated on both fresh runs and cache hits.
    if result and result.extended_metadata:
        report = result.extended_metadata.get("rt_cstate_report")
        if report:
            _attach_text(report, "rt_cstate_report.txt")
        rt_csv = result.extended_metadata.get("rt_cstate_csv")
        if rt_csv:
            _attach_csv(rt_csv, "rt_cstate_cores.csv")

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

    logger.info("RT Core C-State Disabled Check completed: %s", test_display_name)

    if test_interrupted:
        if configs.get("labels", {}).get("type") == "qualification":
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
