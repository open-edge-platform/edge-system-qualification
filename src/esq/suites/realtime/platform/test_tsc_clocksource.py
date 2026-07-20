# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Realtime Platform — TSC Active Clocksource Detection.

Reports ``tsc_clocksource_active=1`` when the kernel has selected the Time
Stamp Counter (TSC) as the active clocksource. When TCC mode is enabled and
the platform is configured for deterministic timing the kernel promotes TSC
as the current clocksource. Available clocksource names are captured as an
informational metric.
"""

import logging
from pathlib import Path
from typing import List, Optional

import allure
import pytest
from sysagent.utils.core import Metrics, Result

logger = logging.getLogger(__name__)

_CLOCKSOURCE_DIR = Path("/sys/devices/system/clocksource/clocksource0")


def _get_clocksource_info() -> dict:
    """Return current and available clocksource names from sysfs."""
    current: Optional[str] = None
    available: List[str] = []

    try:
        p = _CLOCKSOURCE_DIR / "current_clocksource"
        if p.exists():
            current = p.read_text(encoding="utf-8").strip()
    except (OSError, IOError) as e:
        logger.debug(f"Could not read current_clocksource: {e}")

    try:
        p = _CLOCKSOURCE_DIR / "available_clocksource"
        if p.exists():
            available = p.read_text(encoding="utf-8").strip().split()
    except (OSError, IOError) as e:
        logger.debug(f"Could not read available_clocksource: {e}")

    return {
        "current": current,
        "available": available,
        "tsc_active": current == "tsc",
    }


@allure.title("TSC Active Clocksource")
def test_tsc_clocksource(
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
    Verify TSC is the active kernel clocksource.

    Reports ``tsc_clocksource_active`` (1/0). TSC as the active clocksource
    indicates the platform is configured for deterministic timing, a key
    signal of TCC mode enabled in BIOS.
    """
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info(f"Starting TSC Clocksource Detection: {test_display_name}")

    validate_system_requirements_from_configs(configs)

    result = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    def _run_detection():
        cs = _get_clocksource_info()

        logger.info(
            f"Clocksource: current={cs['current']!r}, tsc_active={cs['tsc_active']}, available={cs['available']}"
        )

        return Result(
            name=f"{test_id} - {test_display_name}",
            extended_metadata={
                "current_clocksource": cs["current"] or "unknown",
                "available_clocksources": cs["available"],
            },
            metrics={
                "tsc_clocksource_active": Metrics(unit=None, value=1 if cs["tsc_active"] else 0, is_key_metric=True),
            },
            metadata={"status": cs["tsc_active"]},
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
        failure_message = "Interrupt detected during TSC Clocksource Detection"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during TSC Clocksource Detection: {e}"
        logger.error(failure_message, exc_info=True)

    if result is None:
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "TSC clocksource detection did not complete"},
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

    logger.info(f"TSC Clocksource Detection completed: {test_display_name}")

    if test_interrupted:
        if configs.get("labels", {}).get("type") == "qualification":
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
