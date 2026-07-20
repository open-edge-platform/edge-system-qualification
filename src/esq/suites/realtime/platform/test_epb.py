# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Realtime Platform — Energy Performance Bias / Preference Detection.

Reads the CPU energy/performance preference as an integer in the range 0–15
(0=maximum performance, 15=maximum power saving). Two interfaces are probed
in order; neither is platform-specific — the correct interface is selected
automatically based on what the kernel exposes:

  1. Legacy EPB sysfs — ``/sys/devices/system/cpu/cpu0/power/energy_perf_bias``
     Integer 0–15. Exposed by the ``intel_epb`` driver when the CPU reports
     CPUID.06H:ECX[3]=1 (``epb`` flag in ``/proc/cpuinfo``).

  2. Modern EPP sysfs — ``/sys/devices/system/cpu/cpufreq/policy0/
     energy_performance_preference``
     Exposed by ``intel_pstate`` when HWP with ``hwp_epp`` is enabled.
     Returns one of four strings on read: ``performance``→0,
     ``balance_performance``→4, ``balance_power``→8, ``power``→15.
     Numeric EPP values (0–255) are linearly scaled to 0–15.
     (``default`` is write-only and is never returned on read.)

Reports -1 when neither interface is available.
"""

import logging
from pathlib import Path
from typing import Optional

import allure
import pytest
from sysagent.utils.core import Metrics, Result

logger = logging.getLogger(__name__)

_EPB_PATH = Path("/sys/devices/system/cpu/cpu0/power/energy_perf_bias")
_EPP_PATH = Path("/sys/devices/system/cpu/cpufreq/policy0/energy_performance_preference")

# EPP string → EPB-equivalent integer mapping.
# Only the four strings the kernel actually returns on read are included.
# "default" is write-only (resets EPP to firmware default; never returned).
_EPP_TO_EPB = {
    "performance": 0,
    "balance_performance": 4,
    "balance_power": 8,
    "power": 15,
}


def _read_energy_perf_bias() -> Optional[int]:
    """
    Read the energy/performance preference from the first available interface.

    Tries legacy EPB sysfs first, then modern EPP sysfs. Returns an integer
    0–15, or None when neither interface is available.
    """
    # Interface 1: legacy EPB sysfs.
    try:
        return int(_EPB_PATH.read_text(encoding="utf-8").strip())
    except (OSError, IOError, ValueError):
        pass

    # Interface 2: modern EPP sysfs.
    try:
        epp = _EPP_PATH.read_text(encoding="utf-8").strip().lower()
        if epp in _EPP_TO_EPB:
            return _EPP_TO_EPB[epp]
        return round(int(epp) * 15 / 255)
    except (OSError, IOError, ValueError):
        pass

    return None


@allure.title("Energy Performance Bias")
def test_epb(
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
    Read the CPU energy/performance preference (EPB or EPP).

    Reports ``energy_perf_bias`` (0–15: 0=performance, 15=power-saving, -1=unavailable).
    Probes legacy EPB sysfs first, then modern EPP sysfs as fallback.
    No platform-specific logic — the correct interface is selected automatically.
    """
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info(f"Starting Energy Performance Bias: {test_display_name}")

    validate_system_requirements_from_configs(configs)

    result = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    def _run_detection():
        value = _read_energy_perf_bias()
        epb_val = value if value is not None else -1

        logger.info(f"Energy performance bias: value={epb_val}")

        return Result(
            name=f"{test_id} - {test_display_name}",
            metrics={
                "energy_perf_bias": Metrics(unit=None, value=epb_val, is_key_metric=True),
            },
            metadata={"status": value is not None},
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
        failure_message = "Interrupt detected during Energy Performance Bias"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during Energy Performance Bias: {e}"
        logger.error(failure_message, exc_info=True)

    if result is None:
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "EPB detection did not complete"},
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

    logger.info(f"Energy Performance Bias completed: {test_display_name}")

    if test_interrupted:
        if configs.get("labels", {}).get("type") == "qualification":
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
