# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for RT performance tests.

Provides input sanitization, duration and CPU-affinity helpers, RT binary
resolution, timer-migration reads, and Allure attachment helpers.  These are
independent of any specific benchmark tool and can be imported by future RT
performance tests (e.g. rtla, rt-tests variants).
"""

import logging
import os
from pathlib import Path

import allure

logger = logging.getLogger(__name__)

# Per-suite environment override for the cyclictest run duration.
# Named after the test file so it never collides with other suites.
DURATION_ENV_VAR = "ENV_SUITE_CYCLICTEST_DURATION"

# Kernel sysctl for timer migration (0 = disabled = RT-safe, 1 = enabled)
TIMER_MIGRATION_PATH = Path("/proc/sys/kernel/timer_migration")


# ---------------------------------------------------------------------------
# RT binary resolution
# ---------------------------------------------------------------------------


def get_session_rt_dir() -> str:
    """Return the user-specific session directory for RT binaries (``/run/user/<UID>/esq``)."""
    return f"/run/user/{os.getuid()}/esq"


def resolve_rt_binary(name: str) -> str | None:
    """Return the session RT binary path for *name*, or None when not present.

    Session copies carry the required file capabilities.  Never falls back to
    the system binary, which lacks them.
    """
    session_path = os.path.join(get_session_rt_dir(), name)
    if os.path.isfile(session_path) and os.access(session_path, os.X_OK):
        logger.debug("Using session RT binary: %s", session_path)
        return session_path
    logger.debug("Session RT binary not found: %s", session_path)
    return None


# ---------------------------------------------------------------------------
# Input validation helpers (break taint chains; no shell=True anywhere)
# ---------------------------------------------------------------------------


def safe_int(value, default: int) -> int:
    """Parse integer safely with fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sanitize_affinity(affinity_raw) -> str:
    """Break taint chain: accepts a list/tuple of CPU indices (e.g. ``[2, 3]``
    from YAML) or a range/comma string (e.g. ``"2-3"``, ``"0,2,4"``).  Only
    digits and separators are copied into the result.

    Returns an empty string when *affinity_raw* is ``None``, empty, or
    contains no valid CPU digits — callers treat ``""`` as "no affinity
    restriction".
    """
    if affinity_raw is None:
        return ""
    if isinstance(affinity_raw, (list, tuple)):
        parts = []
        for item in affinity_raw:
            digits = "".join(ch for ch in str(item) if ch.isdigit())
            if digits:
                parts.append(digits)
        return ",".join(parts)
    safe = ""
    for ch in str(affinity_raw):
        if ch.isdigit() or ch in ",-":
            safe += ch
    return safe


def sanitize_duration(duration_raw) -> str:
    """Break taint chain: copy only digits and a single trailing unit character
    (h/m/s) from the raw duration value. Rejects anything else."""
    raw = str(duration_raw).strip().lower()
    digits = ""
    for ch in raw:
        if ch.isdigit():
            digits += ch
    if not digits:
        return "24h"
    unit_ch = raw[-1] if raw and raw[-1] in "hms" else "h"
    return digits + unit_ch


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------


def parse_duration_to_seconds(duration_str: str) -> int:
    """Convert a duration string (e.g. '24h', '30m', '60s', '3600') to seconds."""
    s = str(duration_str).strip().lower()
    if s.endswith("h"):
        return max(safe_int(s[:-1], 1), 1) * 3600
    if s.endswith("m"):
        return max(safe_int(s[:-1], 1), 1) * 60
    if s.endswith("s"):
        return max(safe_int(s[:-1], 1), 1)
    return max(safe_int(s, 86400), 1)


def duration_to_metric(duration_seconds: int) -> tuple[float, str]:
    """Return a human-readable (value, unit) pair for reporting.

    Selects hours (h) for ≥ 1 h, minutes (min) for ≥ 1 min, else seconds (s).
    """
    if duration_seconds >= 3600:
        return round(duration_seconds / 3600, 2), "h"
    if duration_seconds >= 60:
        return round(duration_seconds / 60, 2), "min"
    return float(duration_seconds), "s"


def resolve_cyclic_duration(configs: dict) -> str:
    """Resolve the cyclictest duration, honouring the per-suite env override.

    Priority: ``ENV_SUITE_CYCLICTEST_DURATION`` env var > profile
    ``cyclic_duration`` param > ``"24h"`` default.  Invalid overrides are
    ignored with a warning.
    """
    default_raw = str(configs.get("cyclic_duration", "24h"))
    default_duration = sanitize_duration(default_raw)

    raw_override = os.environ.get(DURATION_ENV_VAR)
    if raw_override is None:
        return default_duration

    sanitized = sanitize_duration(raw_override)
    if not sanitized or sanitized == "h":
        logger.warning(
            "Ignoring non-parseable %s=%r; using profile default %s",
            DURATION_ENV_VAR,
            raw_override,
            default_duration,
        )
        return default_duration

    logger.info("Duration override from %s: %s", DURATION_ENV_VAR, sanitized)
    return sanitized


# ---------------------------------------------------------------------------
# CPU affinity helpers
# ---------------------------------------------------------------------------


def parse_core_set(affinity: str) -> set:
    """Return the set of integer CPU indices encoded in an affinity string."""
    cores: set = set()
    for part in str(affinity).split(","):
        part = part.strip()
        if "-" in part:
            halves = part.split("-", 1)
            try:
                start, end = int(halves[0]), int(halves[1])
                cores.update(range(start, end + 1))
            except (ValueError, IndexError) as exc:
                logger.debug("Ignoring malformed affinity range %r: %s", part, exc)
        elif part.isdigit():
            cores.add(int(part))
    return cores


def count_cores_in_affinity(affinity: str) -> int:
    """Count CPU cores encoded in an affinity string like '0-1', '2-3', '0,2,4'."""
    return max(len(parse_core_set(affinity)), 1)


def build_stress_affinity_all_except(cyclic_affinity: str) -> str:
    """Return an affinity string covering all CPUs except the cyclic ones."""
    total = os.cpu_count() or 4
    cyclic_cores = parse_core_set(cyclic_affinity)
    stress_cores = [i for i in range(total) if i not in cyclic_cores]
    if not stress_cores:
        logger.warning("All CPUs are assigned to cyclictest; stress will run on core 0 as fallback")
        return "0"
    return ",".join(str(c) for c in stress_cores)


# ---------------------------------------------------------------------------
# Timer migration
# ---------------------------------------------------------------------------


def read_timer_migration() -> int | None:
    """Read ``/proc/sys/kernel/timer_migration``.

    Returns 0 (disabled = RT-safe), 1 (enabled), or None when absent.
    """
    try:
        return int(TIMER_MIGRATION_PATH.read_text().strip())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Allure attachment helpers
# ---------------------------------------------------------------------------


def attach_json_file(file_path: str, attachment_name: str) -> None:
    """Attach an existing JSON file to the Allure report."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        if content.strip():
            allure.attach(content, name=attachment_name, attachment_type=allure.attachment_type.JSON)
    except OSError as exc:
        logger.debug("Could not attach %s to Allure report: %s", file_path, exc)


# ---------------------------------------------------------------------------
# CPU isolation detection
# ---------------------------------------------------------------------------


def check_cpu_isolation(cyclic_cores: set[int]) -> tuple[bool, set[int]]:
    """Parse ``/proc/cmdline`` for ``isolcpus`` and intersect with *cyclic_cores*.

    Returns ``(isolcpus_configured, isolated_cyclic_cores)``.  When
    ``isolcpus_configured`` is True but ``isolated_cyclic_cores`` is empty,
    the target cores are not isolated and the test should be skipped.
    """
    try:
        cmdline = Path("/proc/cmdline").read_text(encoding="utf-8").strip()
    except OSError:
        return False, set()

    isolcpus_set: set[int] = set()
    isolcpus_configured = False

    for token in cmdline.split():
        if not token.startswith("isolcpus="):
            continue
        isolcpus_configured = True
        value = token[len("isolcpus=") :]
        # Type specifiers (e.g. "domain", "managed_irq", "nohz") contain only
        # letters/underscores; CPU-range tokens start with a digit.
        cpu_parts = [p for p in value.split(",") if p and p[0].isdigit()]
        for part in cpu_parts:
            try:
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    isolcpus_set.update(range(int(lo), int(hi) + 1))
                else:
                    isolcpus_set.add(int(part))
            except ValueError:
                pass
        break  # Only one isolcpus token is expected in cmdline

    return isolcpus_configured, cyclic_cores & isolcpus_set
