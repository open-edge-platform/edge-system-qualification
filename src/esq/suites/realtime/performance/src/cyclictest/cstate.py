# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""C-state optimization helpers for RT performance tests.

Provides access verification, disable, restore, and report formatting for
CPU C-state management via the cpuidle sysfs interface.
"""

import logging
import os
from pathlib import Path

from sysagent.utils.core.process import get_executor

logger = logging.getLogger(__name__)

# Base path for cpuidle sysfs (C-state control)
CPU_BASE_PATH = Path("/sys/devices/system/cpu")


# ---------------------------------------------------------------------------
# Session tee (cap_sys_admin) binary resolution
# ---------------------------------------------------------------------------


def _find_session_tee() -> str | None:
    """Return path to the session tee binary with cap_sys_admin, or None."""
    session = f"/run/user/{os.getuid()}/esq/tee"
    if Path(session).is_file() and os.access(session, os.X_OK):
        return session
    return None


# ---------------------------------------------------------------------------
# Sysfs write helper
# ---------------------------------------------------------------------------


def write_cstate_disable(disable_file: Path, value: str) -> None:
    """Write *value* to a cpuidle disable sysfs file.

    Tries a direct write first; falls back to the session tee binary when
    the direct write fails.  Raises ``OSError`` when both paths fail.
    """
    try:
        disable_file.write_text(f"{value}\n")
        return
    except OSError:
        pass

    tee = _find_session_tee()
    if tee is None:
        raise OSError(
            f"Cannot write to {disable_file}: direct write failed and session tee "
            f"(cap_sys_admin) not found. Run system-setup-rt.sh Kernel Tuning module."
        )

    result = get_executor().run(
        command=[tee, str(disable_file)],
        input_data=value + "\n",
        capture_output=True,
        timeout=5,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise OSError(f"Session tee failed writing {disable_file}: {stderr}")


# ---------------------------------------------------------------------------
# Prerequisite check
# ---------------------------------------------------------------------------


def check_cstate_write_access(cpu_ids: list[int]) -> tuple[bool, str]:
    """Probe whether cpuidle state disable sysfs files are writable on *cpu_ids*.

    Performs a real no-op write rather than relying on ``os.access()``.
    Returns ``(writable, skip_reason)``.
    """
    for cid in cpu_ids:
        cpuidle_dir = CPU_BASE_PATH / f"cpu{cid}" / "cpuidle"
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
            if int(state_dir.name[5:]) == 0:
                continue  # POLL state — skip
            disable_file = state_dir / "disable"
            if disable_file.exists():
                try:
                    current = disable_file.read_text().strip()
                    write_cstate_disable(disable_file, current)
                    return True, ""
                except OSError:
                    return (
                        False,
                        (
                            f"C-state disable sysfs files are not writable (e.g. {disable_file}). "
                            f"Refer to the installation guide to run system-setup-rt.sh "
                            f"(Kernel Tuning module)."
                        ),
                    )
    # No cpuidle dirs found — nothing to disable; treat as writable.
    return True, ""


# ---------------------------------------------------------------------------
# Save / disable / restore
# ---------------------------------------------------------------------------


def save_rt_cstate_config(cpu_ids: list[int]) -> dict[int, dict[int, int]]:
    """Read current disable flag for every cpuidle state on the specified CPUs.

    Returns ``{cpu_id: {state_id: current_disable_value}}``.
    """
    saved: dict[int, dict[int, int]] = {}
    for cid in cpu_ids:
        cpuidle_dir = CPU_BASE_PATH / f"cpu{cid}" / "cpuidle"
        if not cpuidle_dir.exists():
            continue
        saved[cid] = {}
        try:
            state_dirs = sorted(
                (p for p in cpuidle_dir.iterdir() if p.name.startswith("state") and p.name[5:].isdigit()),
                key=lambda p: int(p.name[5:]),
            )
        except OSError:
            continue
        for state_dir in state_dirs:
            state_id = int(state_dir.name[5:])
            try:
                saved[cid][state_id] = int((state_dir / "disable").read_text().strip())
            except (OSError, ValueError):
                saved[cid][state_id] = 0
    return saved


def disable_rt_cstates(
    cpu_ids: list[int],
    max_latency_us: int = 0,
) -> tuple[dict[int, dict[int, int]], list[str], list[str]]:
    """Disable C-states with exit latency > ``max_latency_us`` on the specified CPUs.

    Saves current disable values before any writes so they can be restored
    unconditionally via :func:`restore_rt_cstates`.

    Returns ``(saved_config, applied_changes, write_failures)``.
    *applied_changes* lists states that were successfully disabled.
    *write_failures* lists states where the write was attempted but failed.
    """
    saved = save_rt_cstate_config(cpu_ids)
    applied: list[str] = []
    write_failures: list[str] = []

    for cid in cpu_ids:
        cpuidle_dir = CPU_BASE_PATH / f"cpu{cid}" / "cpuidle"
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
            try:
                latency = int((state_dir / "latency").read_text().strip())
            except (OSError, ValueError):
                continue
            if latency <= max_latency_us:
                continue
            if saved.get(cid, {}).get(state_id, 0) == 1:
                continue  # Already disabled
            disable_file = state_dir / "disable"
            try:
                write_cstate_disable(disable_file, "1")
                state_name = ""
                try:
                    state_name = (state_dir / "name").read_text().strip()
                except OSError:
                    pass
                applied.append(f"CPU {cid} state{state_id} ({state_name}, latency={latency}µs)")
                logger.debug(
                    "C-state disabled: cpu%d state%d (%s, latency=%dµs)",
                    cid,
                    state_id,
                    state_name,
                    latency,
                )
            except OSError as exc:
                write_failures.append(f"CPU {cid} state{state_id} (latency={latency}µs): {exc}")
                logger.warning("Failed to disable C-state cpu%d state%d: %s", cid, state_id, exc)

    return saved, applied, write_failures


def restore_rt_cstates(saved_config: dict[int, dict[int, int]]) -> None:
    """Restore cpuidle disable values to pre-test state.

    Called unconditionally from ``finally`` blocks — errors are logged but
    never raised so the test outcome is not obscured.
    """
    try:
        for cid, states in saved_config.items():
            cpuidle_dir = CPU_BASE_PATH / f"cpu{cid}" / "cpuidle"
            if not cpuidle_dir.exists():
                continue
            for state_id, orig_val in states.items():
                disable_file = cpuidle_dir / f"state{state_id}" / "disable"
                if not disable_file.exists():
                    continue
                try:
                    current = int(disable_file.read_text().strip())
                    if current != orig_val:
                        write_cstate_disable(disable_file, str(orig_val))
                        logger.debug("C-state restored: cpu%d state%d -> %d", cid, state_id, orig_val)
                except OSError as exc:
                    logger.warning("Failed to restore C-state cpu%d state%d: %s", cid, state_id, exc)
        logger.info("RT C-state configuration restored to pre-test baseline")
    except Exception as exc:
        logger.error("C-state restore error (may be partially restored): %s", exc)


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------


def format_cstate_opt_report(
    saved_config: dict[int, dict[int, int]],
    applied_changes: list[str],
    cpu_ids: list[int],
    write_failures: list[str] | None = None,
) -> str:
    """Format a human-readable C-state optimization report for the Allure attachment."""
    lines = [
        "RT C-State Optimization — cyclictest Pre-Run",
        "=" * 60,
        f"RT CPUs : {sorted(cpu_ids)}",
        "",
        "Baseline (before test):",
    ]
    for cid in sorted(cpu_ids):
        states = saved_config.get(cid, {})
        if states:
            parts = [f"state{sid}={'disabled' if v else 'enabled'}" for sid, v in sorted(states.items()) if sid > 0]
            lines.append(f"  CPU {cid}: {', '.join(parts) if parts else 'no deep states'}")
    lines.append("")
    if applied_changes:
        lines.append(f"Changes applied ({len(applied_changes)}):")
        for ch in applied_changes:
            lines.append(f"  disabled: {ch}")
    elif not write_failures:
        lines.append("No changes needed (all qualifying C-states were already disabled).")
    if write_failures:
        lines.append(f"Write failures ({len(write_failures)}) — run system-setup-rt.sh Kernel Tuning module:")
        for f in write_failures:
            lines.append(f"  FAILED: {f}")
    lines += [
        "",
        "Restore: original disable values restored unconditionally after test.",
    ]
    return "\n".join(lines)


def read_cstate_disabled_passive(rt_cpu_ids: list[int]) -> int:
    """Return 1 if all non-zero-latency C-states are disabled on *rt_cpu_ids*, 0 otherwise.

    Read-only observation; performs no sysfs writes.
    """
    try:
        from sysagent.utils.system.cpu.cpuidle import collect_cpuidle_info

        info = collect_cpuidle_info()
        states = info.get("states", [])
        if not states:
            return 0
        rt_set = set(rt_cpu_ids)
        rt_rows = [r for r in states if r.get("cpu") in rt_set]
        if not rt_rows:
            return 0
        violations = [r for r in rt_rows if r.get("latency_us", 0) > 0 and not r.get("disabled", False)]
        return 1 if not violations else 0
    except Exception:  # noqa: BLE001 — any failure yields a safe 0
        return 0
