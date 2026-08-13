# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Intel RDT cache partition configuration reader.

Reads the current per-CPU CLOS (Class of Service) assignments and the
per-CLOS L3 / L2 cache bitmasks directly from MSR registers via the
``rdmsr`` command (``msr-tools`` package).  Does NOT require the resctrl
filesystem to be mounted.

MSR addresses (Intel SDM Vol. 3B; confirmed against intel-cmt-cat source):
  IA32_PQR_ASSOC   0x00C8F  per logical CPU:
                             bits[63:32] = CLOS ID (COS_ID)
                             bits[31:10] = Reserved
                             bits[9:0]   = RMID
  IA32_L3_MASK_n   0x00C90 + n   L3 CAT capacity bitmask for CLOS n
                                 (valid range: 0xC90–0xD0F, up to 128 CLOS)
  IA32_L2_MASK_n   0x00D10 + n   L2 CAT capacity bitmask for CLOS n

Requirements
------------
- ``msr-tools`` installed: ``sudo apt install msr-tools``
- ``msr`` kernel module loaded: ``modprobe msr``
- MSR read permission: root, ``cap_sys_rawio`` file-cap on rdmsr binary,
  or the session rdmsr copy created by ``system-setup-rt.sh``
  MSR Tools module.

The L3 and L2 mask MSRs are package-global (same value on all logical CPUs
within one socket).  Only one reference CPU per package is read for masks.
Per-CPU CLOS assignments are read individually for all online CPUs.
"""

import logging
import os
import shutil

from sysagent.utils.core.process import run_command

logger = logging.getLogger(__name__)

# ── MSR addresses (Intel SDM / intel-cmt-cat cpu_registers.h) ─────────────────
_MSR_PQR_ASSOC = 0xC8F  # IA32_PQR_ASSOC — CLOS[63:32] + RMID[9:0]
_MSR_L3_MASK_BASE = 0xC90  # IA32_L3CA_MASK_0 … IA32_L3CA_MASK_127 (0xD0F)
_MSR_L2_MASK_BASE = 0xD10  # IA32_L2CA_MASK_0 …
_MSR_L3_MASK_MAX = 0xD0F  # inclusive upper bound for L3 mask MSRs
_MAX_CLOS_PROBE = 128  # probe at most this many CLOS slots per level


def _find_rdmsr() -> str | None:
    """Return the path to the ``rdmsr`` binary, or ``None`` if not found.

    Search order:
    1. Session-scoped copy at ``/run/user/<UID>/esq/rdmsr`` — created by
       ``system-setup-rt.sh`` MSR Tools module with
       ``cap_sys_rawio,cap_dac_read_search+ep``.
       Preferred because it works for the current user without root.
    2. System PATH via :func:`shutil.which` — works when running as root or
       when the system rdmsr binary has been given ``cap_sys_rawio``.
    """
    uid = os.getuid()
    session_rdmsr = f"/run/user/{uid}/esq/rdmsr"
    if os.path.isfile(session_rdmsr) and os.access(session_rdmsr, os.X_OK):
        return session_rdmsr
    return shutil.which("rdmsr")


def _read_msr(rdmsr_bin: str, cpu: int, msr: int) -> int | None:
    """Read one MSR from a specific logical CPU via ``rdmsr -p <cpu> <addr>``.

    Returns the 64-bit integer value on success, or ``None`` on any failure
    (binary not found, permission denied, MSR not supported on this CPU, etc.).
    """
    try:
        result = run_command([rdmsr_bin, "-p", str(cpu), hex(msr)], timeout=5)
        if result.returncode == 0:
            raw = result.stdout.strip()
            if raw:
                return int(raw, 16)
    except (ValueError, OSError, FileNotFoundError):
        pass
    return None


def _enumerate_cpu_ids() -> list[int]:
    """Return sorted list of online logical CPU IDs from sysfs."""
    cpu_base = "/sys/devices/system/cpu"
    try:
        return sorted(
            int(e.name[3:]) for e in os.scandir(cpu_base) if e.name.startswith("cpu") and e.name[3:].isdigit()
        )
    except Exception:
        return list(range(os.cpu_count() or 1))


def _read_level_masks(
    rdmsr_bin: str,
    ref_cpu: int,
    msr_base: int,
) -> dict[str, str]:
    """Read all CLOS bitmasks for one cache level (L3 or L2).

    Probes MSRs starting at ``msr_base`` for CLOS 0, 1, 2, … until the
    first failure (non-zero rdmsr exit code, indicating the address is
    architecturally invalid on this CPU).

    Returns a dict: ``str(clos_id)`` → hex-string mask (e.g. ``"0xfff"``).
    An empty dict is returned when the first probe fails (level not supported).
    """
    masks: dict[str, str] = {}
    max_addr = _MSR_L3_MASK_MAX if msr_base == _MSR_L3_MASK_BASE else msr_base + _MAX_CLOS_PROBE - 1

    for clos in range(_MAX_CLOS_PROBE):
        msr_addr = msr_base + clos
        if msr_addr > max_addr:
            break
        val = _read_msr(rdmsr_bin, ref_cpu, msr_addr)
        if val is None:
            break  # MSR unsupported or permission error — stop probing
        masks[str(clos)] = hex(val)

    return masks


def collect_cache_partition_info(cpu_ids: list[int] | None = None) -> dict:
    """Read current Intel CAT cache-partition configuration via MSR registers.

    Uses ``rdmsr`` from ``msr-tools`` to read:

    * ``IA32_PQR_ASSOC`` (0xC8F) from every logical CPU — reports which
      CLOS each CPU is currently assigned to.
    * ``IA32_L3_MASK_n`` (0xC90 + n) — L3 CAT capacity bitmask per CLOS.
    * ``IA32_L2_MASK_n`` (0xD10 + n) — L2 CAT capacity bitmask per CLOS
      (skipped when L2 CAT is not supported on this CPU).

    Parameters
    ----------
    cpu_ids:
        Explicit list of logical CPU numbers to query for CLOS assignments.
        When *None*, all online CPUs under ``/sys/devices/system/cpu/cpuN/``
        are enumerated automatically.

    Returns
    -------
    dict
        ``rdmsr_available``
            ``True`` when the ``rdmsr`` binary was found in PATH.
        ``msr_accessible``
            ``True`` when at least one MSR was successfully read.
        ``cpu_clos``
            Mapping of ``str(cpu_id)`` → CLOS integer for each logical CPU.
            Empty when MSR access failed.
        ``clos_masks``
            Mapping of ``str(clos_id)`` → per-level mask dict.
            Each entry has ``"l3"`` (hex string, always present when L3 CAT
            is supported) and optionally ``"l2"`` (hex string, when L2 CAT
            is supported).  Only CLOS IDs readable without error are included.
        ``note``
            Optional human-readable guidance string when access was not
            possible.
    """
    result: dict = {
        "rdmsr_available": False,
        "msr_accessible": False,
        "cpu_clos": {},
        "clos_masks": {},
    }

    rdmsr_bin = _find_rdmsr()
    if rdmsr_bin is None:
        result["note"] = (
            "rdmsr not found. Install msr-tools: sudo apt install msr-tools. "
            "Also ensure the msr kernel module is loaded: modprobe msr"
        )
        return result

    result["rdmsr_available"] = True

    # Enumerate logical CPU IDs
    if cpu_ids is None:
        cpu_ids = _enumerate_cpu_ids()

    # ── Per-CPU CLOS assignment via IA32_PQR_ASSOC (0xC8F) ───────────────────
    # bits[63:32] = CLOS (COS_ID); bits[9:0] = RMID (ignored here)
    any_read = False
    for cid in cpu_ids:
        val = _read_msr(rdmsr_bin, cid, _MSR_PQR_ASSOC)
        if val is not None:
            clos = (val >> 32) & 0xFFFF_FFFF
            result["cpu_clos"][str(cid)] = clos
            any_read = True
        else:
            logger.debug(f"rdmsr: could not read IA32_PQR_ASSOC for cpu{cid}")

    if not any_read:
        result["note"] = (
            "MSR read failed. Ensure the msr kernel module is loaded "
            "(modprobe msr) and that rdmsr has the required permissions "
            "(root or cap_sys_rawio file capability on the rdmsr binary). "
            "See system-setup-rt.sh MSR Tools module for session-scoped setup."
        )
        return result

    result["msr_accessible"] = True

    # Use the first (lowest) online CPU as the reference for package-wide masks.
    ref_cpu = cpu_ids[0]

    # ── L3 CAT masks (0xC90 + n) ─────────────────────────────────────────────
    l3_masks = _read_level_masks(rdmsr_bin, ref_cpu, _MSR_L3_MASK_BASE)
    if not l3_masks:
        logger.debug("rdmsr: L3 CAT mask MSRs not readable — L3 CAT not supported or no access")

    # ── L2 CAT masks (0xD10 + n) ─────────────────────────────────────────────
    l2_masks = _read_level_masks(rdmsr_bin, ref_cpu, _MSR_L2_MASK_BASE)
    if not l2_masks:
        logger.debug("rdmsr: L2 CAT mask MSRs not readable — L2 CAT not supported or no access")

    # ── Combine into clos_masks ───────────────────────────────────────────────
    all_clos = sorted(set(map(int, l3_masks)) | set(map(int, l2_masks)))
    for clos in all_clos:
        entry: dict[str, str] = {}
        if str(clos) in l3_masks:
            entry["l3"] = l3_masks[str(clos)]
        if str(clos) in l2_masks:
            entry["l2"] = l2_masks[str(clos)]
        if entry:
            result["clos_masks"][str(clos)] = entry

    return result
