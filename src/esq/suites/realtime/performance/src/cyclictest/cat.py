# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Intel L3 CAT cache partition helpers for RT performance tests.

Provides prerequisite detection, setup, restore, and report formatting for
Intel Cache Allocation Technology (CAT) L3 cache partitioning.  Gives RT
threads a dedicated set of L3 cache ways, reducing cache contention with
stress/OS workloads.

This module is intentionally tool-agnostic so it can be reused by any RT
performance test (e.g. future rtla or rt-tests variants).

Requires ``system-setup-rt.sh`` MSR Tools module (rdmsr/wrmsr session
binaries with ``cap_sys_rawio`` in ``/run/user/<UID>/esq/``).

MSR addresses (Intel SDM Vol. 3B):
  IA32_PQR_ASSOC   (0xC8F) — per-CPU CLOS ID in bits[63:32]
  IA32_L3CA_MASK_0 (0xC90) — package-global L3 bitmask for CLOS 0
"""

import logging
import os
import shutil
from pathlib import Path

from sysagent.utils.core import run_command

from .utils import safe_int

logger = logging.getLogger(__name__)

# MSR addresses (Intel SDM Vol. 3B)
MSR_PQR_ASSOC = 0xC8F  # IA32_PQR_ASSOC  — per-CPU CLOS ID in bits[63:32]
MSR_L3_MASK_BASE = 0xC90  # IA32_L3CA_MASK_0 — package-global L3 bitmask for CLOS 0


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------


def _find_rdmsr() -> str | None:
    """Return the rdmsr binary: session copy first (cap_sys_rawio), then PATH."""
    session = f"/run/user/{os.getuid()}/esq/rdmsr"
    if Path(session).is_file() and os.access(session, os.X_OK):
        return session
    return shutil.which("rdmsr")


def _find_wrmsr() -> str | None:
    """Return the wrmsr binary: session copy first (cap_sys_rawio), then PATH."""
    session = f"/run/user/{os.getuid()}/esq/wrmsr"
    if Path(session).is_file() and os.access(session, os.X_OK):
        return session
    return shutil.which("wrmsr")


# ---------------------------------------------------------------------------
# Low-level MSR read / write
# ---------------------------------------------------------------------------


def read_msr_val(rdmsr_bin: str, cpu: int, msr: int) -> int | None:
    """Read a 64-bit MSR via ``rdmsr -p <cpu> <addr>``. Returns None on failure."""
    proc = run_command([rdmsr_bin, "-p", str(cpu), hex(msr)], timeout=5)
    if proc and proc.returncode == 0:
        raw = proc.stdout.strip()
        if raw:
            try:
                return int(raw, 16)
            except ValueError:
                pass
    return None


def write_msr_val(wrmsr_bin: str, cpu: int, msr: int, value: int) -> bool:
    """Write a 64-bit MSR via ``wrmsr -p <cpu> <addr> <value>``. Returns True on success."""
    proc = run_command([wrmsr_bin, "-p", str(cpu), hex(msr), hex(value)], timeout=5)
    return proc is not None and proc.returncode == 0


# ---------------------------------------------------------------------------
# Prerequisite detection
# ---------------------------------------------------------------------------


def check_cat_prerequisites() -> tuple[bool, str, dict]:
    """Check whether L3 CAT partition setup is possible on this platform.

    Returns:
        (available, skip_reason, context)

        *context* keys when available is True: ``rdmsr``, ``wrmsr``,
        ``ref_cpu``, ``clos0_mask``, ``total_ways``.
    """
    rdmsr = _find_rdmsr()
    if rdmsr is None:
        return (
            False,
            (
                "CAT optimization requires the rdmsr binary with CAP_SYS_RAWIO. "
                "Refer to the installation guide to run system-setup-rt.sh "
                "(MSR Tools module)."
            ),
            {},
        )

    wrmsr = _find_wrmsr()
    if wrmsr is None:
        return (
            False,
            (
                "CAT optimization requires the wrmsr binary with CAP_SYS_RAWIO. "
                "Refer to the installation guide to run system-setup-rt.sh "
                "(MSR Tools module)."
            ),
            {},
        )

    ref_cpu = 0
    msr_found = False
    for cid in range(min(os.cpu_count() or 1, 4)):
        if Path(f"/dev/cpu/{cid}/msr").exists():
            ref_cpu = cid
            msr_found = True
            break
    if not msr_found:
        return (
            False,
            (
                "MSR devices (/dev/cpu/N/msr) not found — the msr kernel module may not be "
                "loaded. Refer to the installation guide to run system-setup-rt.sh."
            ),
            {},
        )

    clos0_mask = read_msr_val(rdmsr, ref_cpu, MSR_L3_MASK_BASE)
    if not clos0_mask:
        return (
            False,
            (
                "L3 CAT MSR (IA32_L3_QOS_MASK_0 0xC90) is not accessible or returned 0 — "
                "L3 CAT may not be supported on this platform. "
                "Refer to the installation guide to run system-setup-rt.sh."
            ),
            {},
        )

    total_ways = bin(clos0_mask).count("1")
    if total_ways < 2:
        return (
            False,
            f"L3 CLOS 0 bitmask 0x{clos0_mask:x} has fewer than 2 ways; cannot create a useful cache partition.",
            {},
        )

    return (
        True,
        "",
        {
            "rdmsr": rdmsr,
            "wrmsr": wrmsr,
            "ref_cpu": ref_cpu,
            "clos0_mask": clos0_mask,
            "total_ways": total_ways,
        },
    )


# ---------------------------------------------------------------------------
# Partition setup and restore
# ---------------------------------------------------------------------------


def setup_cat_partition(
    rt_cpu_ids: list[int],
    configs: dict,
    prereq_ctx: dict,
) -> tuple[dict, dict]:
    """Partition L3 cache to give RT cores dedicated upper ways.

    RT cores are assigned to ``cat_rt_clos`` (default: 1) with the upper
    ``cat_rt_ways`` L3 ways; all other CPUs remain in CLOS 0 with the lower
    ways.  The original CLOS masks and per-RT-CPU IA32_PQR_ASSOC values are
    captured as baseline before any writes.

    Returns:
        (baseline_state, partition_info)

    Raises:
        RuntimeError: when a required MSR write fails.
    """
    rdmsr: str = prereq_ctx["rdmsr"]
    wrmsr: str = prereq_ctx["wrmsr"]
    ref_cpu: int = prereq_ctx["ref_cpu"]
    clos0_mask: int = prereq_ctx["clos0_mask"]
    total_ways: int = prereq_ctx["total_ways"]

    rt_clos = max(1, min(safe_int(configs.get("cat_rt_clos", 1), 1), 127))
    default_rt_ways = (total_ways + 1) // 2
    rt_ways = max(
        1,
        min(
            safe_int(configs.get("cat_rt_ways", default_rt_ways), default_rt_ways),
            total_ways - 1,
        ),
    )
    os_ways = total_ways - rt_ways

    # Contiguous bitmasks: RT gets upper bits, OS gets lower bits
    rt_mask = ((1 << rt_ways) - 1) << os_ways
    os_mask = (1 << os_ways) - 1

    # Capture baseline before any writes
    clos_rt_mask_orig = read_msr_val(rdmsr, ref_cpu, MSR_L3_MASK_BASE + rt_clos)
    rt_cpu_pqr_orig: dict[int, int | None] = {cid: read_msr_val(rdmsr, cid, MSR_PQR_ASSOC) for cid in rt_cpu_ids}

    baseline_state: dict = {
        "rdmsr": rdmsr,
        "wrmsr": wrmsr,
        "ref_cpu": ref_cpu,
        "rt_clos": rt_clos,
        "clos0_mask": clos0_mask,
        "clos_rt_mask_orig": clos_rt_mask_orig,
        "rt_cpu_pqr_orig": rt_cpu_pqr_orig,
    }

    # Apply partition
    if not write_msr_val(wrmsr, ref_cpu, MSR_L3_MASK_BASE, os_mask):
        raise RuntimeError(f"Failed to write CLOS 0 L3 mask 0x{os_mask:x} to MSR 0x{MSR_L3_MASK_BASE:x}")

    if not write_msr_val(wrmsr, ref_cpu, MSR_L3_MASK_BASE + rt_clos, rt_mask):
        write_msr_val(wrmsr, ref_cpu, MSR_L3_MASK_BASE, clos0_mask)  # best-effort restore
        raise RuntimeError(
            f"Failed to write CLOS {rt_clos} L3 mask 0x{rt_mask:x} to MSR 0x{MSR_L3_MASK_BASE + rt_clos:x}"
        )

    # Assign each RT CPU to rt_clos, preserving existing RMID in bits[9:0]
    failed: list[int] = []
    for cid in rt_cpu_ids:
        cur_pqr = rt_cpu_pqr_orig.get(cid)
        rmid = (cur_pqr & 0x3FF) if cur_pqr is not None else 0
        if not write_msr_val(wrmsr, cid, MSR_PQR_ASSOC, (rt_clos << 32) | rmid):
            failed.append(cid)

    if failed:
        logger.warning("CAT setup: failed to assign CLOS %d to CPUs %s", rt_clos, failed)

    partition_info: dict = {
        "rt_clos": rt_clos,
        "rt_cpu_ids": rt_cpu_ids,
        "total_l3_ways": total_ways,
        "rt_ways": rt_ways,
        "os_ways": os_ways,
        "rt_mask": f"0x{rt_mask:x}",
        "os_mask": f"0x{os_mask:x}",
        "clos0_original_mask": f"0x{clos0_mask:x}",
        "clos_rt_original_mask": (f"0x{clos_rt_mask_orig:x}" if clos_rt_mask_orig is not None else "default"),
        "failed_cpu_assignments": failed,
    }

    logger.info(
        "CAT partition applied: CLOS 0=0x%x (%d ways), CLOS %d=0x%x (%d ways), RT CPUs=%s",
        os_mask,
        os_ways,
        rt_clos,
        rt_mask,
        rt_ways,
        rt_cpu_ids,
    )
    return baseline_state, partition_info


def restore_cat_partition(baseline_state: dict) -> None:
    """Restore L3 CAT configuration to the pre-test baseline.

    Called unconditionally from ``finally`` blocks — errors are logged but
    never raised so the test outcome is not obscured.
    """
    try:
        wrmsr: str | None = baseline_state.get("wrmsr")
        rdmsr: str | None = baseline_state.get("rdmsr")
        ref_cpu: int = baseline_state.get("ref_cpu", 0)
        rt_clos: int = baseline_state.get("rt_clos", 1)
        clos0_mask: int | None = baseline_state.get("clos0_mask")
        clos_rt_mask_orig: int | None = baseline_state.get("clos_rt_mask_orig")
        rt_cpu_pqr_orig: dict = baseline_state.get("rt_cpu_pqr_orig", {})

        if wrmsr is None:
            logger.warning("CAT restore: wrmsr not available — configuration may not be restored")
            return

        for cid, orig_pqr in rt_cpu_pqr_orig.items():
            if orig_pqr is not None:
                restore_val = orig_pqr
            else:
                cur = read_msr_val(rdmsr, int(cid), MSR_PQR_ASSOC) if rdmsr else None
                restore_val = (cur & 0x3FF) if cur is not None else 0
            if not write_msr_val(wrmsr, int(cid), MSR_PQR_ASSOC, restore_val):
                logger.warning("CAT restore: failed to restore CLOS assignment for CPU %d", cid)

        if clos0_mask is not None:
            if not write_msr_val(wrmsr, ref_cpu, MSR_L3_MASK_BASE, clos0_mask):
                logger.warning("CAT restore: failed to restore CLOS 0 mask to 0x%x", clos0_mask)

        restore_rt_mask = clos_rt_mask_orig if clos_rt_mask_orig is not None else clos0_mask
        if restore_rt_mask is not None:
            if not write_msr_val(wrmsr, ref_cpu, MSR_L3_MASK_BASE + rt_clos, restore_rt_mask):
                logger.warning("CAT restore: failed to restore CLOS %d mask", rt_clos)

        logger.info("CAT partition restored to pre-test baseline")

    except Exception as exc:
        logger.error("CAT restore error (configuration may be partially restored): %s", exc)


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------


def format_cat_report(baseline_state: dict, partition_info: dict) -> str:
    """Format a human-readable CAT partition report for the Allure attachment."""
    rt_clos = partition_info.get("rt_clos", 1)
    total_ways = partition_info.get("total_l3_ways", "?")
    rt_ways = partition_info.get("rt_ways", "?")
    os_ways = partition_info.get("os_ways", "?")
    rt_mask = partition_info.get("rt_mask", "?")
    os_mask = partition_info.get("os_mask", "?")
    clos0_orig = partition_info.get("clos0_original_mask", "?")
    clos_rt_orig = partition_info.get("clos_rt_original_mask", "default")
    rt_cpus = partition_info.get("rt_cpu_ids", [])
    failed = partition_info.get("failed_cpu_assignments", [])

    lines = [
        "CAT L3 Cache Partition — cyclictest Optimization",
        "=" * 60,
        f"Total L3 ways       : {total_ways}",
        "",
        "Baseline (before test):",
        f"  CLOS 0 mask       : {clos0_orig}  (all ways — shared)",
        f"  CLOS {rt_clos} mask       : {clos_rt_orig}",
        "",
        "Partition applied:",
        f"  CLOS 0 (OS/stress): {os_mask}  ({os_ways} ways)",
        f"  CLOS {rt_clos} (RT cores) : {rt_mask}  ({rt_ways} ways)",
        f"  RT CPUs assigned  : {rt_cpus} → CLOS {rt_clos}",
    ]
    if failed:
        lines.append(f"  WARNING: Failed to assign CPUs {failed} to CLOS {rt_clos}")
    lines += [
        "",
        "Restore: masks and CPU CLOS assignments restored after test.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Passive CAT observation (metric collection, no MSR writes)
# ---------------------------------------------------------------------------


def read_cat_partitioned_passive() -> int:
    """Return 1 if Intel L3 CAT partitioning is active, 0 otherwise.

    This is a **read-only** observation used to record whether L3 cache
    partitioning was in effect at test execution time (e.g. set up by a
    prior ``setup_cat_partition()`` call or by the operator).  It performs
    no MSR writes.

    Returns:
        ``1`` when any CPU is assigned to a non-zero CLOS, or when the CLOS 0
        mask differs from the full bit-mask (implying a reduced allocation),
        ``0`` otherwise or on any error.
    """
    try:
        from sysagent.utils.system.cpu.rdt import collect_cache_partition_info

        info = collect_cache_partition_info()
        if not info.get("msr_accessible", False):
            return 0
        cpu_clos: dict = info.get("cpu_clos", {})
        clos_masks: dict = info.get("clos_masks", {})
        if not cpu_clos or not clos_masks:
            return 0
        active_clos_ids = set(cpu_clos.values())
        if any(clos != 0 for clos in active_clos_ids):
            return 1
        clos0_masks = clos_masks.get("0", {})
        if "l3" in clos0_masks:
            l3_mask = clos0_masks["l3"]
            if isinstance(l3_mask, int) and l3_mask > 0:
                max_mask = max(
                    (m["l3"] for m in clos_masks.values() if isinstance(m.get("l3"), int) and m["l3"] > 0),
                    default=l3_mask,
                )
                full_mask = (1 << max_mask.bit_length()) - 1
                if l3_mask != full_mask:
                    return 1
        return 0
    except Exception:  # noqa: BLE001 — any failure yields a safe 0
        return 0
