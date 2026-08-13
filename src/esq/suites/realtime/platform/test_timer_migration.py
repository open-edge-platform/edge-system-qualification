# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Realtime Platform — Kernel Timer Migration Check.

Linux timer migration moves timer callbacks from one CPU to another to
consolidate timers on fewer cores and allow idle CPUs to enter deep sleep
states (power management optimisation).  On real-time systems this behaviour
is undesirable: a timer migrated onto an isolated RT core adds unexpected
interrupt overhead and introduces jitter, breaking the latency guarantees that
isolation is designed to provide.

The kernel timer-migration sysctl is controlled via:

    /proc/sys/kernel/timer_migration

Values:
    0 — migration disabled; timers stay on the CPU they were queued on.
    1 — migration enabled (default on most distributions).

Disable it for RT workloads:

    echo 0 > /proc/sys/kernel/timer_migration

Or persistently via ``/etc/sysctl.d/99-realtime.conf``:

    kernel.timer_migration = 0

Key metric: ``timer_migration_disabled``
    1 = migration disabled (value == 0) — RT-safe configuration.
    0 = migration enabled (value == 1) or sysctl unavailable.
"""

import logging
from pathlib import Path

import allure
import pytest
from sysagent.utils.core import Metrics, Result

logger = logging.getLogger(__name__)

_TIMER_MIGRATION_PATH = Path("/proc/sys/kernel/timer_migration")


def _read_timer_migration() -> int | None:
    """Read the current ``timer_migration`` value from the proc sysctl.

    Returns:
        0 — migration disabled (RT-safe).
        1 — migration enabled (default).
        None — file is absent or unreadable (non-SMP kernel or no access).
    """
    try:
        return int(_TIMER_MIGRATION_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


@allure.title("Timer Migration")
def test_timer_migration(
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
    Check whether kernel timer migration is disabled on this system.

    Timer migration can move timer callbacks onto isolated RT cores, introducing
    interrupt-driven jitter that degrades real-time latency consistency.
    Disabling it (``/proc/sys/kernel/timer_migration = 0``) keeps timers on
    their origin CPU and prevents interference with RT threads.

    Key metric: ``timer_migration_disabled`` (1 = disabled = RT-safe, 0 = enabled).
    Supporting metric: ``timer_migration`` (raw sysctl value; -1 when unavailable).
    """
    # ================================================================
    # STEP 1: Extract Parameters
    # ================================================================
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    is_qualification = configs.get("labels", {}).get("type") == "qualification"

    logger.info(f"Starting Timer Migration check: {test_display_name}")

    # ================================================================
    # STEP 2: Validate System Requirements
    # ================================================================
    validate_system_requirements_from_configs(configs)

    results = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    try:
        # ============================================================
        # STEP 3: Prepare Assets/Dependencies
        # (no external assets — skipped)
        # ============================================================

        # ============================================================
        # STEP 4: Execute Test Logic (with caching)
        # ============================================================
        def _run_detection():
            value = _read_timer_migration()
            disabled = 1 if value == 0 else 0

            if value is None:
                logger.info(
                    "timer_migration: /proc/sys/kernel/timer_migration not readable "
                    "(non-SMP kernel or restricted access)"
                )
            else:
                state = "disabled (RT-safe)" if value == 0 else f"enabled ({value}) — may cause RT jitter"
                logger.info(f"timer_migration: {value} — {state}")

            return Result(
                name=f"{test_id} - {test_display_name}",
                metrics={
                    "timer_migration_disabled": Metrics(unit=None, value=disabled, is_key_metric=True),
                },
                metadata={"status": True},
            )

        results = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=_run_detection,
            test_name=test_name,
            configs=configs,
        )

    except KeyboardInterrupt:
        failure_message = "Interrupt detected during Timer Migration check"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during Timer Migration check: {e}"
        logger.error(failure_message, exc_info=True)

    if results is None:
        results = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "Timer migration detection did not complete"},
            metrics={},
        )

    # ================================================================
    # STEP 5: Validate Results Against KPIs (qualification only)
    # ================================================================
    try:
        validate_test_results(
            test_name=test_name,
            results=results,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as validation_error:
        logger.error(f"KPI validation failed: {validation_error}")

    # ================================================================
    # STEP 6: Summarize Results
    # ================================================================
    try:
        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as summary_error:
        logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)

    logger.info(f"Timer Migration check completed: {test_display_name}")

    if test_interrupted:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
