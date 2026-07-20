# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Realtime Platform — TCC TSC CPU Flag Detection.

Reports ``tcc_capable=1`` when the CPU exposes both ``constant_tsc`` and
``nonstop_tsc`` flags (invariant TSC), the hardware prerequisite for TCC
deterministic timing.
"""

import logging
from typing import List, Tuple

import allure
import pytest
from sysagent.utils.core import Metrics, Result
from sysagent.utils.system import SystemInfoCache

logger = logging.getLogger(__name__)


def _get_cpu_flags() -> List[str]:
    """Return lower-cased CPU feature flags from the system info cache."""
    cpu_info = SystemInfoCache().get_hardware_info().get("cpu", {})
    flags = cpu_info.get("flags", []) or []
    if isinstance(flags, list):
        return [str(flag).lower() for flag in flags]
    return []


def detect_tcc() -> Tuple[bool, dict]:
    """
    Detect TCC capability via invariant TSC CPU flags.

    ``tcc_capable`` is True when both ``constant_tsc`` and ``nonstop_tsc``
    are present, confirming invariant TSC hardware support.
    """
    flags = _get_cpu_flags()
    constant_tsc = "constant_tsc" in flags
    nonstop_tsc = "nonstop_tsc" in flags
    return constant_tsc and nonstop_tsc, {"constant_tsc": constant_tsc, "nonstop_tsc": nonstop_tsc}


@allure.title("TCC Detection")
def test_tcc(
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
    Verify TCC-capable hardware via invariant TSC CPU flags.

    Reports ``tcc_capable`` (1/0) as the key metric. A platform is considered
    capable when both ``constant_tsc`` and ``nonstop_tsc`` CPU flags are
    present (invariant TSC), the hardware prerequisite for TCC deterministic
    timing. ACPI table presence and resctrl availability are informational.
    """
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info(f"Starting TCC Detection: {test_display_name}")

    validate_system_requirements_from_configs(configs)

    result = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    def _run_detection():
        is_tcc_capable, details = detect_tcc()

        logger.info(f"TCC flags: constant_tsc={details['constant_tsc']}, nonstop_tsc={details['nonstop_tsc']}")

        return Result(
            name=f"{test_id} - {test_display_name}",
            metrics={
                "tcc_capable": Metrics(unit=None, value=1 if is_tcc_capable else 0, is_key_metric=True),
                "constant_tsc": Metrics(unit=None, value=1 if details["constant_tsc"] else 0, is_key_metric=False),
                "nonstop_tsc": Metrics(unit=None, value=1 if details["nonstop_tsc"] else 0, is_key_metric=False),
            },
            metadata={"status": is_tcc_capable},
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
        failure_message = "Interrupt detected during TCC Detection"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during TCC Detection: {e}"
        logger.error(failure_message, exc_info=True)

    if result is None:
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "TCC detection did not complete"},
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

    logger.info(f"TCC Detection completed: {test_display_name}")

    if test_interrupted:
        if configs.get("labels", {}).get("type") == "qualification":
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
