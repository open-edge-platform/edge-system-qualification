# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CPU frequency scaling information collector.

Reads the Linux cpufreq sysfs tree for every logical CPU and returns
structured data suitable for inclusion in the hardware info JSON and
for use by real-time qualification tests.

sysfs sources::

    /sys/devices/system/cpu/cpu<N>/cpufreq/
        scaling_governor    — active frequency scaling policy
        scaling_max_freq    — current upper frequency limit in kHz (governor/policy-set)
        scaling_min_freq    — current lower frequency limit in kHz
        scaling_cur_freq    — instantaneous current frequency in kHz
        cpuinfo_max_freq    — hardware-reported maximum frequency in kHz
        cpuinfo_min_freq    — hardware-reported minimum frequency in kHz

    /sys/devices/system/cpu/cpu<N>/power/
        energy_perf_bias    — per-core EPB hint (0 = performance … 15 = power-saving)

    /sys/devices/system/cpu/cpufreq/policy<N>/
        energy_performance_preference  — EPP string (Intel P-state HWP mode)

``scaling_max_freq`` and ``scaling_min_freq`` reflect the *operating* limits set
by the current governor (or ``cpupower`` / ``intel_pstate`` knobs).  They may be
lower than ``cpuinfo_max_freq`` when the system is power-constrained or when
``schedutil`` / ``powersave`` is in use — important context for RT certification.

``global_scaling_governors`` is the sorted de-duplicated list of governors
currently active across all CPUs.  On a correctly tuned RT system all CPUs (or
at least all isolated RT cores) should show ``["performance"]``.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CPU_BASE = Path("/sys/devices/system/cpu")
_CPUFREQ_POLICY_BASE = _CPU_BASE / "cpufreq"


# ---------------------------------------------------------------------------
# sysfs helpers
# ---------------------------------------------------------------------------


def _read_int(path: Path) -> int | None:
    """Read a single integer from a sysfs file; None on any error."""
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_str(path: Path) -> str | None:
    """Read a trimmed string from a sysfs file; None on OSError."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Raw collection
# ---------------------------------------------------------------------------


def _collect_per_cpu() -> dict[int, dict]:
    """Read cpufreq sysfs entries for every logical CPU.

    CPUs whose ``cpufreq`` subdirectory does not exist are silently skipped
    (common for offline CPUs or when the cpufreq driver is not loaded).

    Returns a dict keyed by integer CPU index.
    """
    per_cpu: dict[int, dict] = {}
    try:
        cpu_dirs = sorted(
            (p for p in _CPU_BASE.iterdir() if p.name.startswith("cpu") and p.name[3:].isdigit()),
            key=lambda p: int(p.name[3:]),
        )
    except OSError as exc:
        logger.warning("Cannot enumerate CPUs under %s: %s", _CPU_BASE, exc)
        return per_cpu

    for cpu_dir in cpu_dirs:
        cpu_id = int(cpu_dir.name[3:])
        freq_dir = cpu_dir / "cpufreq"
        if not freq_dir.exists():
            continue

        entry: dict = {
            "scaling_governor": _read_str(freq_dir / "scaling_governor"),
            "scaling_max_freq_khz": _read_int(freq_dir / "scaling_max_freq"),
            "scaling_min_freq_khz": _read_int(freq_dir / "scaling_min_freq"),
            "scaling_cur_freq_khz": _read_int(freq_dir / "scaling_cur_freq"),
            "cpuinfo_max_freq_khz": _read_int(freq_dir / "cpuinfo_max_freq"),
            "cpuinfo_min_freq_khz": _read_int(freq_dir / "cpuinfo_min_freq"),
        }

        # Per-core EPB — present on Intel CPUs with the acpi-cpufreq or
        # intel_pstate driver; absent on other architectures.
        epb = _read_int(cpu_dir / "power" / "energy_perf_bias")
        if epb is not None:
            entry["energy_perf_bias"] = epb

        per_cpu[cpu_id] = entry

    return per_cpu


def _collect_global_epp() -> str | None:
    """Read the EPP policy string from the first available cpufreq policy.

    Intel P-state HWP mode exposes an
    ``energy_performance_preference`` file under each policy directory.
    Typical values: ``performance``, ``balance_performance``,
    ``balance_power``, ``power``.

    Returns None when the file is absent (non-HWP system, or when
    intel_pstate is not the active driver).
    """
    # policy0 is representative; all policies share the same EPP on most
    # single-package Intel systems.
    return _read_str(_CPUFREQ_POLICY_BASE / "policy0" / "energy_performance_preference")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_cpufreq_info() -> dict:
    """Collect CPU frequency scaling information from the cpufreq sysfs tree.

    Returns a dict suitable for embedding in the hardware info JSON::

        {
          "available": true,
          "cpu_count": 24,
          "global_scaling_governors": ["performance"],
          "global_epp_policy": "performance",
          "global_energy_perf_bias": 0,
          "per_cpu": {
            "0": {
              "scaling_governor":    "performance",
              "scaling_max_freq_khz": 5700000,
              "scaling_min_freq_khz":  400000,
              "scaling_cur_freq_khz": 3600000,
              "cpuinfo_max_freq_khz": 5700000,
              "cpuinfo_min_freq_khz":  400000,
              "energy_perf_bias": 0
            },
            ...
          }
        }

    Returns ``{"available": false}`` when the cpufreq sysfs tree is not
    accessible (driver not loaded, VM without direct hardware exposure, etc.).
    """
    per_cpu = _collect_per_cpu()
    if not per_cpu:
        logger.debug("cpufreq sysfs not available — skipping cpufreq collection")
        return {"available": False}

    global_epp = _collect_global_epp()

    # Global EPB from cpu0 — representative on single-package systems where
    # all cores share the same power domain configuration.
    global_epb: int | None = per_cpu.get(0, {}).get("energy_perf_bias")

    unique_governors = sorted({v["scaling_governor"] for v in per_cpu.values() if v.get("scaling_governor")})

    return {
        "available": True,
        "cpu_count": len(per_cpu),
        "global_scaling_governors": unique_governors,
        "global_epp_policy": global_epp,
        "global_energy_perf_bias": global_epb,
        "per_cpu": {str(cpu_id): entry for cpu_id, entry in sorted(per_cpu.items())},
    }
