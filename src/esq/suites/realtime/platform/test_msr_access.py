# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Realtime Platform — MSR Read/Write Access Verification.

Verifies that the MSR kernel driver is loaded and CAP_SYS_RAWIO is available
via a two-stage probe:

  Stage 1 — Read IA32_TSC_ADJUST (0x3B): always present on Intel processors.
    Confirms CAP_SYS_RAWIO and the MSR driver are available.

  Stage 2 — Round-trip write IA32_L3_QOS_MASK_0 (0xC90): the L3 Cache
    Allocation Technology (CAT) Class of Service 0 mask register. Removes the
    highest set bit (e.g. 0xFFF → 0x7FF), writes, reads back to verify the
    change, then restores the original. Proves L3 CAT registers are
    programmable — the requirement for TCC cache partitioning.
    Returns EIO when the CPU lacks L3 CAT hardware, mapping to
    msr_write_capable=0.

When CAP_SYS_RAWIO is not held by the calling process, the probe retries via
the session Python binary (``/dev/shm/esq/python3-msr``) placed by
``system-setup-advanced.sh`` with ``cap_sys_rawio+ep``. If neither path
succeeds, ``msr_write_capable=-1`` (indeterminate) and the test fails with an
actionable message.

Refer to the installation guide: run ``system-setup-advanced.sh`` to enable MSR access.
"""

import logging
import os
import struct
from pathlib import Path
from typing import Optional, Tuple

import allure
import pytest
from sysagent.utils.core import Metrics, Result, run_command

logger = logging.getLogger(__name__)

_MSR_IA32_TSC_ADJUST = 0x3B
_MSR_L3_QOS_MASK_0 = 0xC90
_SESSION_MSR_PYTHON = "/dev/shm/esq/python3-msr"

# Inline stdlib-only probe script (no virtualenv needed).
# Passed via -c to the session python binary.
# sys.argv[1] = MSR device path.
# Output:
#   Line 1: read_ok write_ok | read_ok write_fail | read_ok no_cat |
#            permission_denied | error
#   Line 2 (write_ok / write_fail only): orig=<hex> probe=<hex> readback=<hex>
_MSR_PROBE_SCRIPT = (
    "import os, struct, sys\n"
    "msr_dev = sys.argv[1]\n"
    "try:\n"
    "    with open(msr_dev, 'rb') as f:\n"
    "        os.pread(f.fileno(), 8, 0x3B)\n"
    "except PermissionError:\n"
    "    print('permission_denied'); sys.exit(0)\n"
    "except OSError:\n"
    "    print('error'); sys.exit(0)\n"
    "try:\n"
    "    with open(msr_dev, 'rb') as f:\n"
    "        raw = os.pread(f.fileno(), 8, 0xC90)\n"
    "    val = struct.unpack('<Q', raw)[0]\n"
    "    top_bit = (1 << (val.bit_length() - 1)) if val > 0 else 0\n"
    "    probe = val ^ top_bit\n"
    "    if probe == 0:\n"
    "        probe = 3 if val == 1 else val | (val >> 1)\n"
    "    with open(msr_dev, 'r+b') as f:\n"
    "        fd = f.fileno()\n"
    "        try:\n"
    "            os.pwrite(fd, struct.pack('<Q', probe), 0xC90)\n"
    "            rb = struct.unpack('<Q', os.pread(fd, 8, 0xC90))[0]\n"
    "            print('read_ok write_ok' if rb == probe else 'read_ok write_fail')\n"
    "            print(f'orig={val:#x} probe={probe:#x} readback={rb:#x}')\n"
    "        finally:\n"
    "            try: os.pwrite(fd, struct.pack('<Q', val), 0xC90)\n"
    "            except OSError: pass\n"
    "except OSError:\n"
    "    print('read_ok no_cat')\n"
)


def _probe_via_session_python(msr_dev: str) -> Optional[Tuple[bool, bool, dict]]:
    """
    Retry MSR probe via the session binary (``cap_sys_rawio+ep``).

    Returns ``(msr_readable, msr_writable, probe_details)`` or ``None`` when
    the session binary is absent or the probe fails.
    """
    if not Path(_SESSION_MSR_PYTHON).exists():
        logger.debug(f"Session python3-msr not found at {_SESSION_MSR_PYTHON}")
        return None

    proc = run_command([_SESSION_MSR_PYTHON, "-c", _MSR_PROBE_SCRIPT, msr_dev], timeout=10)
    if not (proc and proc.returncode == 0 and proc.stdout):
        logger.debug(f"Session python3-msr probe failed on {msr_dev}")
        return None

    lines = proc.stdout.strip().splitlines()
    status = lines[0] if lines else ""
    details: dict = {}
    if len(lines) > 1 and lines[1].startswith("orig="):
        try:
            for part in lines[1].split():
                k, v = part.split("=", 1)
                details[k] = int(v, 16)
        except (ValueError, IndexError):
            pass

    if "read_ok write_ok" in status:
        details["write_verified"] = True
        logger.debug(f"Session python3-msr: L3 CAT write verified on {msr_dev}")
        return True, True, details
    if "read_ok no_cat" in status:
        logger.debug(f"Session python3-msr: MSR readable, no L3 CAT (EIO) on {msr_dev}")
        return True, False, {}
    if "read_ok" in status:
        details["write_verified"] = False
        return True, False, details
    logger.debug(f"Session python3-msr unexpected output: {proc.stdout.strip()!r}")
    return None


def _check_msr_access() -> Tuple[bool, bool, Optional[str], dict]:
    """
    Two-stage MSR probe: Stage 1 confirms CAP_SYS_RAWIO + MSR driver,
    Stage 2 verifies L3 CAT registers are writable.

    Returns:
        (msr_readable, msr_writable, msr_device_path, probe_details).
        probe_details contains orig/probe/readback/write_verified when
        Stage 2 ran, empty dict otherwise.
    """
    msr_dev: Optional[str] = None
    for cpu_id in range(min(os.cpu_count() or 1, 4)):
        candidate = f"/dev/cpu/{cpu_id}/msr"
        if Path(candidate).exists():
            msr_dev = candidate
            break

    if msr_dev is None:
        logger.debug("No /dev/cpu/N/msr device found (msr kernel module not loaded?)")
        return False, False, None, {}

    # Stage 1: read IA32_TSC_ADJUST (always present, never written).
    try:
        with open(msr_dev, "rb") as f:
            os.pread(f.fileno(), 8, _MSR_IA32_TSC_ADJUST)
    except PermissionError:
        logger.debug(f"MSR read denied on {msr_dev}; retrying via session python3-msr")
        r = _probe_via_session_python(msr_dev)
        if r is not None:
            return r[0], r[1], msr_dev, r[2]
        return False, False, msr_dev, {}
    except OSError as e:
        logger.debug(f"MSR read error on {msr_dev}: {e}")
        return False, False, msr_dev, {}

    # Stage 2: L3 CAT write verification — IA32_L3_QOS_MASK_0 (0xC90).
    msr_writable = False
    probe_details: dict = {}
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
                probe_details = {
                    "orig": orig,
                    "probe": probe,
                    "readback": readback,
                    "write_verified": msr_writable,
                }
                logger.debug(
                    f"L3 CAT probe on {msr_dev}: orig={orig:#x} probe={probe:#x} "
                    f"readback={readback:#x} ({'MATCH' if msr_writable else 'MISMATCH'})"
                )
            finally:
                try:
                    os.pwrite(fd, struct.pack("<Q", orig), _MSR_L3_QOS_MASK_0)
                except OSError:
                    pass
    except PermissionError:
        logger.debug(f"L3 CAT MSR write denied on {msr_dev}")
    except OSError as e:
        logger.debug(f"L3 CAT MSR (0xC90) not accessible on {msr_dev}: {e}")

    return True, msr_writable, msr_dev, probe_details


@allure.title("MSR Write Access")
def test_msr_access(
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
    Verify MSR read access (Stage 1) and L3 CAT write access (Stage 2).

    Reports ``msr_write_capable``: 1=L3 CAT writable, 0=read-only or no L3
    CAT hardware, -1=indeterminate (CAP_SYS_RAWIO denied). A value of -1
    always causes test failure with an actionable message referencing the
    installation guide to run ``system-setup-advanced.sh``.
    """
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info(f"Starting MSR Write Access: {test_display_name}")

    validate_system_requirements_from_configs(configs)

    result = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    def _run_detection():
        msr_readable, msr_writable, msr_dev, probe = _check_msr_access()

        permission_denied = not msr_readable and msr_dev is not None
        if permission_denied:
            msr_write_val = -1
            logger.warning(
                f"MSR access INDETERMINATE on {msr_dev} (CAP_SYS_RAWIO denied). "
                "Refer to the installation guide to run system-setup-advanced.sh."
            )
        else:
            msr_write_val = 1 if msr_writable else 0
            logger.info(f"MSR: readable={msr_readable}, writable={msr_writable}, device={msr_dev or 'not found'}")

        # Build probe report attachment.
        dev_str = msr_dev or "not found"
        stage1 = (
            "PASSED"
            if msr_readable
            else (
                "FAILED (permission denied — refer to the installation guide to run system-setup-advanced.sh)"
                if msr_dev
                else "FAILED (msr module not loaded?)"
            )
        )
        if probe.get("orig") is not None:
            orig, prb, rb = probe["orig"], probe["probe"], probe["readback"]
            stage2 = (
                f"  Original CLOS 0:  {orig:#018x}  ({bin(orig).count('1')} ways)\n"
                f"  Probe mask:       {prb:#018x}  ({bin(prb).count('1')} ways)\n"
                f"  Readback:         {rb:#018x}\n"
                f"  Verified:         {'PASSED' if probe.get('write_verified') else 'FAILED'}"
            )
        elif msr_readable:
            stage2 = "  Skipped (no L3 CAT hardware — EIO)" if not msr_writable else "  Not run"
        else:
            stage2 = "  Skipped (no MSR access)"

        report = (
            f"MSR Probe — IA32_L3_QOS_MASK_0 (0xC90): L3 CAT Write Verification\n"
            f"{'=' * 67}\n"
            f"Device:  {dev_str}\n\n"
            f"Stage 1 — IA32_TSC_ADJUST (0x3B): {stage1}\n\n"
            f"Stage 2 — IA32_L3_QOS_MASK_0 (0xC90):\n{stage2}\n\n"
            f"Result: msr_write_capable={msr_write_val}\n"
        )
        allure.attach(report, name="msr_cat_probe.txt", attachment_type=allure.attachment_type.TEXT)

        return Result(
            name=f"{test_id} - {test_display_name}",
            extended_metadata={"msr_device": dev_str},
            metrics={
                "msr_write_capable": Metrics(unit=None, value=msr_write_val, is_key_metric=True),
                "msr_readable": Metrics(unit=None, value=1 if msr_readable else 0, is_key_metric=False),
            },
            metadata={"status": msr_writable},
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
        failure_message = "Interrupt detected during MSR Write Access"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during MSR Write Access: {e}"
        logger.error(failure_message, exc_info=True)

    if result is None:
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "MSR access check did not complete"},
            metrics={},
        )

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

    # msr_write_capable=-1 always fails: indeterminate result requires user action.
    if not test_failed and result and result.metrics:
        msr_metric = result.metrics.get("msr_write_capable")
        if msr_metric is not None and msr_metric.value == -1:
            msr_dev = (result.extended_metadata or {}).get("msr_device", "unknown")
            test_failed = True
            failure_message = (
                f"MSR access denied on {msr_dev} (CAP_SYS_RAWIO required). "
                "Refer to the installation guide to run system-setup-advanced.sh."
            )

    logger.info(f"MSR Write Access completed: {test_display_name}")

    if test_interrupted:
        if configs.get("labels", {}).get("type") == "qualification":
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
