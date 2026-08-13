# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CPU idle state (C-state) information collector.

Reads the Linux cpuidle sysfs tree for every logical CPU and returns
structured data suitable for inclusion in the hardware info JSON and
for use by real-time qualification tests.

sysfs source::

    /sys/devices/system/cpu/cpu<N>/cpuidle/state<M>/
        name     — state identifier (POLL, C1, C1E, C3, C6, …)
        disable  — 1 if the governor has disabled this state, 0 if enabled
        latency  — exit latency in µs (0 for the POLL busy-wait pseudo-state)
        usage    — number of times the state was entered (cumulative since boot)
        time     — total µs spent in the state (cumulative since boot, kernel-measured)

``idle_time_pct``
    Fraction of cumulative idle time (since boot) spent in this state, per CPU.
    Denominator = sum of all ``time`` values for that CPU (total kernel-measured
    idle time), NOT total wall-clock time.  This answers "when idle, where does
    the CPU sleep?" rather than turbostat's "what fraction of wall time is C-state X?".
    turbostat reads MSR hardware residency counters and divides by elapsed TSC
    ticks; our source is sysfs cumulative counters — semantically different.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CPU_BASE = Path("/sys/devices/system/cpu")


# ---------------------------------------------------------------------------
# sysfs helpers
# ---------------------------------------------------------------------------


def _read_int(path: Path) -> int | None:
    """Read a single integer from a sysfs file; None on any error."""
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_str(path: Path) -> str:
    """Read a trimmed string from a sysfs file; empty string on any error."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Raw collection
# ---------------------------------------------------------------------------


def _collect_state_rows() -> list[dict]:
    """Read cpuidle sysfs for all logical CPUs; return raw row dicts.

    Each row contains: cpu, state_id, state_name, disabled, latency_us,
    usage, time_us.  CPUs without a cpuidle directory are silently skipped.
    """
    rows: list[dict] = []
    try:
        cpu_dirs = sorted(
            (p for p in _CPU_BASE.iterdir() if p.name.startswith("cpu") and p.name[3:].isdigit()),
            key=lambda p: int(p.name[3:]),
        )
    except OSError as exc:
        logger.warning("Cannot enumerate CPUs under %s: %s", _CPU_BASE, exc)
        return rows

    for cpu_dir in cpu_dirs:
        cpu_id = int(cpu_dir.name[3:])
        cpuidle_dir = cpu_dir / "cpuidle"
        if not cpuidle_dir.exists():
            continue
        try:
            state_dirs = sorted(
                (p for p in cpuidle_dir.iterdir() if p.name.startswith("state") and p.name[5:].isdigit()),
                key=lambda p: int(p.name[5:]),
            )
        except OSError:
            continue
        for state_dir in state_dirs:
            state_id = int(state_dir.name[5:])
            rows.append(
                {
                    "cpu": cpu_id,
                    "state_id": state_id,
                    "state_name": _read_str(state_dir / "name"),
                    "disabled": _read_int(state_dir / "disable") or 0,
                    "latency_us": _read_int(state_dir / "latency") or 0,
                    "usage": _read_int(state_dir / "usage") or 0,
                    "time_us": _read_int(state_dir / "time") or 0,
                }
            )
    return rows


def _add_idle_time_pct(rows: list[dict]) -> list[dict]:
    """Enrich rows with ``idle_time_pct`` — fraction of per-CPU idle time.

    Denominator is sum of all ``time_us`` values for that CPU (total
    kernel-measured idle time).  Rounded to one decimal place.
    Rows where total idle time = 0 receive 0.0.
    """
    by_cpu: dict[int, list[dict]] = {}
    for r in rows:
        by_cpu.setdefault(r["cpu"], []).append(r)

    enriched: list[dict] = []
    for cpu_id in sorted(by_cpu):
        cpu_rows = by_cpu[cpu_id]
        total_time = sum(r["time_us"] for r in cpu_rows)
        for r in cpu_rows:
            enriched.append(
                {
                    **r,
                    "idle_time_pct": round(r["time_us"] / total_time * 100, 1) if total_time > 0 else 0.0,
                }
            )
    return enriched


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------


def derive_cpuidle_summary(states: list[dict]) -> dict:
    """Compute aggregate C-state metrics from enriched state rows.

    Deep C-states are those with exit latency > 0 µs (excludes the always-active
    POLL pseudo-state).

    Returns:
        cpus_with_idle              — CPUs exposing cpuidle entries
        cstates_deep_total          — total (cpu, state) deep-state pairs
        cstates_deep_disabled_count — deep pairs with disabled flag set
        cstates_deep_enabled_count  — deep pairs still enabled
        deepest_enabled_latency_us  — highest exit latency among enabled deep states
    """
    if not states:
        return {
            "cpus_with_idle": 0,
            "cstates_deep_total": 0,
            "cstates_deep_disabled_count": 0,
            "cstates_deep_enabled_count": 0,
            "deepest_enabled_latency_us": 0,
        }
    deep = [r for r in states if r["latency_us"] > 0]
    deep_disabled = [r for r in deep if r["disabled"]]
    deep_enabled = [r for r in deep if not r["disabled"]]
    return {
        "cpus_with_idle": len({r["cpu"] for r in states}),
        "cstates_deep_total": len(deep),
        "cstates_deep_disabled_count": len(deep_disabled),
        "cstates_deep_enabled_count": len(deep_enabled),
        "deepest_enabled_latency_us": max((r["latency_us"] for r in deep_enabled), default=0),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_cpuidle_info() -> dict:
    """Collect CPU idle state (C-state) data from the cpuidle sysfs tree.

    Returns a dict suitable for embedding in the hardware info JSON:

    .. code-block:: json

        {
          "states": [
            {"cpu": 0, "state_id": 0, "state_name": "POLL", "disabled": 0,
             "latency_us": 0, "usage": 1234567, "time_us": 98765, "idle_time_pct": 0.1},
            ...
          ],
          "summary": {
            "cpus_with_idle": 24,
            "cstates_deep_total": 72,
            "cstates_deep_disabled_count": 0,
            "cstates_deep_enabled_count": 72,
            "deepest_enabled_latency_us": 1048
          }
        }

    ``states`` rows include ``idle_time_pct`` (fraction of that CPU's cumulative
    idle time spent in each state).  See module docstring for interpretation notes.

    Returns an empty structure when the cpuidle sysfs tree is unavailable
    (containers, certain VM configurations).
    """
    rows = _collect_state_rows()
    if not rows:
        logger.debug("No cpuidle sysfs data found — returning empty cpuidle info")
        return {"states": [], "summary": derive_cpuidle_summary([])}

    enriched = _add_idle_time_pct(rows)
    summary = derive_cpuidle_summary(enriched)

    logger.debug(
        "cpuidle: cpus=%d  deep_total=%d  disabled=%d  enabled=%d  deepest_enabled=%d µs",
        summary["cpus_with_idle"],
        summary["cstates_deep_total"],
        summary["cstates_deep_disabled_count"],
        summary["cstates_deep_enabled_count"],
        summary["deepest_enabled_latency_us"],
    )

    return {"states": enriched, "summary": summary}
