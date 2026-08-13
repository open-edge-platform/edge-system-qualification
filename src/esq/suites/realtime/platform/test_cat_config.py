# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Realtime Platform — Intel L3/L2 CAT Partition Configuration.

Detects whether Intel Cache Allocation Technology (CAT) cache partitioning
is currently active by reading MSR registers via rdmsr/wrmsr:

  Stage 1 — MSR read probe (IA32_TSC_ADJUST 0x3B): confirms MSR access.
  Stage 2 — L3 CAT write probe (IA32_L3_QOS_MASK_0 0xC90): round-trip
    write verifies CAT registers are programmable (prerequisite for TCC).
  Stage 3 — Partition snapshot (IA32_PQR_ASSOC 0xC8F per CPU,
    IA32_L3_MASK_n/IA32_L2_MASK_n per CLOS): detects active partitioning.

Key metric ``cat_partitioned`` is 1 when any non-default configuration is
detected (CPU in CLOS != 0, or any restricted bitmask); 0 on stock systems.

The test skips when MSR access prerequisites are not met. Both Allure
reports (cat_access_probe.txt, cat_config_report.txt) are stored in
extended_metadata and re-attached on every run including cache hits.

Refer to the installation guide to run system-setup-rt.sh.
"""

import logging
import os
import shutil
import struct
from pathlib import Path
from typing import Any

import allure
import pytest
from sysagent.utils.core import Metrics, Result, run_command
from sysagent.utils.system.cpu.rdt import collect_cache_partition_info

logger = logging.getLogger(__name__)

_MSR_IA32_TSC_ADJUST = 0x3B
_MSR_L3_QOS_MASK_0 = 0xC90


# ── Session binary helpers ─────────────────────────────────────────────────────────


def _find_rdmsr() -> str | None:
    """Find rdmsr: session copy first (cap_sys_rawio,cap_dac_read_search), then PATH."""
    session = f"/run/user/{os.getuid()}/esq/rdmsr"
    if Path(session).is_file() and os.access(session, os.X_OK):
        return session
    return shutil.which("rdmsr")


def _find_wrmsr() -> str | None:
    """Find wrmsr: session copy first (cap_sys_rawio,cap_dac_override), then PATH."""
    session = f"/run/user/{os.getuid()}/esq/wrmsr"
    if Path(session).is_file() and os.access(session, os.X_OK):
        return session
    return shutil.which("wrmsr")


def _find_msr_device() -> tuple[str | None, int]:
    """Return the first accessible /dev/cpu/N/msr device and its CPU index."""
    for cid in range(min(os.cpu_count() or 1, 4)):
        candidate = f"/dev/cpu/{cid}/msr"
        if Path(candidate).exists():
            return candidate, cid
    return None, 0


# ── Stage 1 + 2: MSR access probe ───────────────────────────────────────────────


def _check_cat_access(msr_dev: str, cpu_id: int) -> tuple[bool, bool, dict]:
    """
    Two-stage MSR probe.

    Stage 1: read IA32_TSC_ADJUST (0x3B) — confirms MSR read access.
    Stage 2: round-trip write IA32_L3_QOS_MASK_0 (0xC90) — confirms L3 CAT
      registers are programmable.  Restores the original value unconditionally.

    Tries direct /dev/cpu/N/msr access first; falls back to session rdmsr/wrmsr
    binaries when the process does not have direct permission.

    Returns:
        (msr_readable, msr_writable, probe_details)
        probe_details keys: orig, probe, readback, write_verified (Stage 2 only)
    """
    msr_readable = False
    _direct_read_ok = False

    # ── Stage 1 ───────────────────────────────────────────────────────────────────
    try:
        with open(msr_dev, "rb") as f:
            os.pread(f.fileno(), 8, _MSR_IA32_TSC_ADJUST)
        msr_readable = True
        _direct_read_ok = True
    except PermissionError:
        rdmsr_bin = _find_rdmsr()
        if rdmsr_bin:
            proc = run_command([rdmsr_bin, "-p", str(cpu_id), hex(_MSR_IA32_TSC_ADJUST)], timeout=5)
            if proc and proc.returncode == 0 and proc.stdout.strip():
                msr_readable = True
                logger.debug(f"Stage 1 via rdmsr: IA32_TSC_ADJUST={proc.stdout.strip()}")
        if not msr_readable:
            logger.debug(f"MSR read denied on {msr_dev} and rdmsr fallback also failed")
    except OSError as e:
        logger.debug(f"MSR read error on {msr_dev}: {e}")

    if not msr_readable:
        return False, False, {}

    # ── Stage 2 ───────────────────────────────────────────────────────────────────
    msr_writable = False
    probe_details: dict = {}

    if _direct_read_ok:
        try:
            with open(msr_dev, "rb") as f:
                raw = os.pread(f.fileno(), 8, _MSR_L3_QOS_MASK_0)
            orig = struct.unpack("<Q", raw)[0]
            top_bit = (1 << (orig.bit_length() - 1)) if orig > 0 else 0
            probe = orig ^ top_bit
            if probe == 0:
                probe = 3 if orig == 1 else orig | (orig >> 1)

            with open(msr_dev, "r+b") as f:
                fd = f.fileno()
                try:
                    os.pwrite(fd, struct.pack("<Q", probe), _MSR_L3_QOS_MASK_0)
                    readback = struct.unpack("<Q", os.pread(fd, 8, _MSR_L3_QOS_MASK_0))[0]
                    msr_writable = readback == probe
                    probe_details = {"orig": orig, "probe": probe, "readback": readback, "write_verified": msr_writable}
                    logger.debug(
                        f"L3 CAT probe on {msr_dev}: orig={orig:#x} probe={probe:#x} "
                        f"readback={readback:#x} ({'MATCH' if msr_writable else 'MISMATCH'})"
                    )
                finally:
                    try:
                        os.pwrite(fd, struct.pack("<Q", orig), _MSR_L3_QOS_MASK_0)
                    except OSError as exc:
                        logger.debug("Could not restore L3 CAT MSR on %s: %s", msr_dev, exc)
        except PermissionError:
            logger.debug(f"L3 CAT MSR write denied on {msr_dev} (direct path)")
        except OSError as e:
            logger.debug(f"L3 CAT MSR (0xC90) inaccessible on {msr_dev}: {e}")
    else:
        rdmsr_bin = _find_rdmsr()
        wrmsr_bin = _find_wrmsr()
        if rdmsr_bin and wrmsr_bin:
            try:
                proc = run_command([rdmsr_bin, "-p", str(cpu_id), hex(_MSR_L3_QOS_MASK_0)], timeout=5)
                if proc and proc.returncode == 0 and proc.stdout.strip():
                    orig = int(proc.stdout.strip(), 16)
                    top_bit = (1 << (orig.bit_length() - 1)) if orig > 0 else 0
                    probe = orig ^ top_bit
                    if probe == 0:
                        probe = 3 if orig == 1 else orig | (orig >> 1)

                    proc = run_command([wrmsr_bin, "-p", str(cpu_id), hex(_MSR_L3_QOS_MASK_0), hex(probe)], timeout=5)
                    if proc and proc.returncode == 0:
                        proc = run_command([rdmsr_bin, "-p", str(cpu_id), hex(_MSR_L3_QOS_MASK_0)], timeout=5)
                        if proc and proc.returncode == 0 and proc.stdout.strip():
                            readback = int(proc.stdout.strip(), 16)
                            msr_writable = readback == probe
                            probe_details = {
                                "orig": orig,
                                "probe": probe,
                                "readback": readback,
                                "write_verified": msr_writable,
                            }
                            logger.debug(
                                f"L3 CAT probe via wrmsr: orig={orig:#x} probe={probe:#x} "
                                f"readback={readback:#x} ({'MATCH' if msr_writable else 'MISMATCH'})"
                            )
                    run_command([wrmsr_bin, "-p", str(cpu_id), hex(_MSR_L3_QOS_MASK_0), hex(orig)], timeout=5)
            except (ValueError, TypeError) as exc:
                logger.debug(f"wrmsr probe parse error: {exc}")
        elif rdmsr_bin:
            logger.debug("wrmsr not available; Stage 2 write verification skipped")
        else:
            logger.debug("Neither rdmsr nor wrmsr available for Stage 2")

    return msr_readable, msr_writable, probe_details


# ── Stage 3: CAT partition configuration ──────────────────────────────────────


def _read_cat_state() -> dict[str, Any]:
    """
    Read current Intel CAT partition state from MSR registers via rdmsr.

    Returns a dict with keys: rdmsr_available, msr_accessible, cat_partitioned,
    cat_l3_ways, cat_active_clos, active_clos_ids, cpu_clos, clos_masks, note.
    """
    info = collect_cache_partition_info()

    rdmsr_available: bool = info.get("rdmsr_available", False)
    msr_accessible: bool = info.get("msr_accessible", False)
    cpu_clos: dict = info.get("cpu_clos", {})
    clos_masks: dict = info.get("clos_masks", {})
    note: str = info.get("note", "")

    active_clos_ids = sorted(set(cpu_clos.values())) if cpu_clos else []
    cat_partitioned = False
    cat_l3_ways = 0

    if msr_accessible and clos_masks:
        if any(clos != 0 for clos in active_clos_ids):
            cat_partitioned = True

        clos0_masks = clos_masks.get("0", {})
        if "l3" in clos0_masks:
            l3_mask_val = int(clos0_masks["l3"], 16)
            cat_l3_ways = l3_mask_val.bit_count()

            if not cat_partitioned:
                ref_mask = l3_mask_val
                for masks in clos_masks.values():
                    if "l3" in masks and int(masks["l3"], 16) != ref_mask:
                        cat_partitioned = True
                        break

    return {
        "rdmsr_available": rdmsr_available,
        "msr_accessible": msr_accessible,
        "cat_partitioned": cat_partitioned,
        "cat_l3_ways": cat_l3_ways,
        "cat_active_clos": len(active_clos_ids),
        "active_clos_ids": active_clos_ids,
        "cpu_clos": cpu_clos,
        "clos_masks": clos_masks,
        "note": note,
    }


# ── Allure report formatters ─────────────────────────────────────────────────────


def _format_access_probe_report(msr_dev: str, msr_readable: bool, msr_writable: bool, probe: dict) -> str:
    """Format the Stage 1/2 probe details for cat_access_probe.txt."""
    stage1 = "PASSED" if msr_readable else "FAILED (permission denied)"

    if probe.get("orig") is not None:
        orig, prb, rb = probe["orig"], probe["probe"], probe["readback"]
        stage2 = (
            f"  Original CLOS 0:  {orig:#018x}  ({orig.bit_count()} ways)\n"
            f"  Probe mask:       {prb:#018x}  ({prb.bit_count()} ways)\n"
            f"  Readback:         {rb:#018x}\n"
            f"  Verified:         {'PASSED' if probe.get('write_verified') else 'FAILED'}"
        )
    elif msr_readable:
        wrmsr_avail = _find_wrmsr() is not None
        stage2 = (
            "  Skipped (no L3 CAT hardware — register returned EIO)"
            if not msr_writable and wrmsr_avail
            else "  Skipped (wrmsr binary not available)"
        )
    else:
        stage2 = "  Skipped (Stage 1 MSR read failed)"

    return (
        f"L3 CAT Register Access Probe — IA32_L3_QOS_MASK_0 (0xC90)\n"
        f"{'=' * 67}\n"
        f"Device:  {msr_dev}\n\n"
        f"Stage 1 — IA32_TSC_ADJUST (0x3B): {stage1}\n\n"
        f"Stage 2 — IA32_L3_QOS_MASK_0 (0xC90):\n{stage2}\n\n"
        f"Result: cat_writable={1 if msr_writable else 0}\n"
    )


def _format_config_report(state: dict[str, Any]) -> str:
    """Format the Stage 3 partition snapshot for cat_config_report.txt."""
    lines: list[str] = [
        "Intel CAT Cache Partition Configuration",
        "=" * 50,
        f"rdmsr available : {'yes' if state['rdmsr_available'] else 'no'}",
        f"MSR accessible  : {'yes' if state['msr_accessible'] else 'no'}",
    ]

    if state.get("note"):
        lines.append(f"Note            : {state['note']}")

    if not state["msr_accessible"]:
        lines += ["", "MSR access unavailable for partition snapshot."]
        return "\n".join(lines)

    partitioned_str = (
        "YES — non-default CAT configuration detected" if state["cat_partitioned"] else "no (default configuration)"
    )
    lines += [
        f"L3 cache ways   : {state['cat_l3_ways']}",
        f"Partitioned     : {partitioned_str}",
        f"Active CLOS IDs : {state['cat_active_clos']} in use {state['active_clos_ids']}",
        "",
        "CLOS assignments per CPU",
        "-" * 30,
    ]

    clos_groups: dict[int, list[int]] = {}
    for cpu_str, clos_id in state["cpu_clos"].items():
        clos_groups.setdefault(clos_id, []).append(int(cpu_str))

    for clos_id in sorted(clos_groups):
        cpus = sorted(clos_groups[clos_id])
        cpu_str = ",".join(str(c) for c in cpus) if len(cpus) <= 8 else f"{cpus[0]}-{cpus[-1]} ({len(cpus)} CPUs)"
        lines.append(f"  CLOS {clos_id:3d}  CPUs: {cpu_str}")

    lines += ["", "Per-CLOS bitmasks", "-" * 30]
    for clos_id_str, masks in sorted(state["clos_masks"].items(), key=lambda x: int(x[0])):
        parts: list[str] = []
        if "l3" in masks:
            ways = int(masks["l3"], 16).bit_count()
            parts.append(f"L3={masks['l3']} ({ways} ways)")
        if "l2" in masks:
            ways = int(masks["l2"], 16).bit_count()
            parts.append(f"L2={masks['l2']} ({ways} ways)")
        lines.append(f"  CLOS {int(clos_id_str):3d}  {',  '.join(parts)}")

    lines += ["", f"cat_partitioned = {1 if state['cat_partitioned'] else 0}"]
    return "\n".join(lines)


# ── Main test function ──────────────────────────────────────────────────────────────


@allure.title("L3/L2 CAT Configuration")
def test_cat_config(
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
    Detect whether Intel L3/L2 CAT cache partitioning is currently active.

    Key metric ``cat_partitioned``: 1 if any non-default CLOS assignment or
    restricted bitmask is detected, 0 on a stock/unconfigured system.

    Skips when MSR access prerequisites are not met (msr module not loaded
    or no rdmsr binary available).
    """
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info(f"Starting L3/L2 CAT Configuration: {test_display_name}")

    validate_system_requirements_from_configs(configs)

    # ── Prerequisite check — skip early if MSR access is not available ──────────
    msr_dev, cpu_id = _find_msr_device()
    if msr_dev is None:
        pytest.skip(
            "MSR devices (/dev/cpu/N/msr) not found — the msr kernel module may not be loaded. "
            "Refer to the installation guide to run system-setup-rt.sh."
        )

    rdmsr_bin = _find_rdmsr()
    can_direct_read = os.access(msr_dev, os.R_OK)
    if rdmsr_bin is None and not can_direct_read:
        pytest.skip(
            f"MSR read access not available on {msr_dev} — no rdmsr binary with CAP_SYS_RAWIO "
            "and no direct read permission. "
            "Refer to the installation guide to run system-setup-rt.sh."
        )
    # ──────────────────────────────────────────────────────────────────────────

    result = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    def _run_detection():
        # Stage 1 + Stage 2: access probe
        msr_readable, msr_writable, probe = _check_cat_access(msr_dev, cpu_id)

        cat_write_val = 1 if msr_writable else 0
        logger.info(f"CAT access: readable={msr_readable}, writable={msr_writable}, device={msr_dev}")

        # Stage 3: partition configuration snapshot
        state = _read_cat_state()
        logger.info(
            f"CAT config: partitioned={state['cat_partitioned']}, "
            f"l3_ways={state['cat_l3_ways']}, active_clos={state['cat_active_clos']}"
        )

        # Build Allure reports (stored in extended_metadata for cache re-attach)
        probe_report = _format_access_probe_report(msr_dev, msr_readable, msr_writable, probe)
        config_report = _format_config_report(state)

        return Result(
            name=f"{test_id} - {test_display_name}",
            metrics={
                "cat_partitioned": Metrics(unit=None, value=1 if state["cat_partitioned"] else 0, is_key_metric=True),
                "cat_writable": Metrics(unit=None, value=cat_write_val, is_key_metric=False),
                "cat_l3_ways": Metrics(unit="ways", value=state["cat_l3_ways"], is_key_metric=False),
                "cat_active_clos": Metrics(unit=None, value=state["cat_active_clos"], is_key_metric=False),
                "cat_readable": Metrics(unit=None, value=1 if msr_readable else 0, is_key_metric=False),
            },
            extended_metadata={
                "cat_access_probe": probe_report,
                "cat_config_report": config_report,
            },
            metadata={"status": True},
        )

    try:
        result = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=_run_detection,
            test_name=test_name,
            configs=configs,
        )
    except KeyboardInterrupt:
        failure_message = "Interrupt detected during L3/L2 CAT Configuration"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during L3/L2 CAT Configuration: {e}"
        logger.exception(failure_message)

    if result is None:
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "CAT configuration check did not complete"},
            metrics={},
        )

    # Always attach Allure reports — works for both fresh runs and cache hits.
    if result and result.extended_metadata:
        probe_txt = result.extended_metadata.get("cat_access_probe")
        if probe_txt:
            allure.attach(probe_txt, name="cat_access_probe.txt", attachment_type=allure.attachment_type.TEXT)
        config_txt = result.extended_metadata.get("cat_config_report")
        if config_txt:
            allure.attach(config_txt, name="cat_config_report.txt", attachment_type=allure.attachment_type.TEXT)

    try:
        validate_test_results(
            test_name=test_name,
            results=result,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as validation_error:  # broad catch: validation errors must not mask test results
        logger.error(f"Validation failed: {validation_error}")

    try:
        summarize_test_results(
            results=result,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception:  # broad catch: summarization errors must not mask test results
        logger.exception("Test result summarization failed")

    logger.info(f"L3/L2 CAT Configuration completed: {test_display_name}")

    if test_interrupted:
        if configs.get("labels", {}).get("type") == "qualification":
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
