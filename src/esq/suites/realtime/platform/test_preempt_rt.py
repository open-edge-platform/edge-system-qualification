# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Realtime Platform — PREEMPT-RT Kernel Check.

Detects whether the running Linux kernel provides full real-time preemption
(``PREEMPT_RT``), which is a prerequisite for deterministic robotics and
motion-control workloads.

PREEMPT_RT vs. lowlatency — a critical distinction
===================================================
Linux ships several preemption models. Only ``PREEMPT_RT`` qualifies as a
hard-real-time kernel:

* **PREEMPT_NONE** (``CONFIG_PREEMPT_NONE=y``) — server default. Preemption
  only at explicit scheduler yield points.
* **PREEMPT_VOLUNTARY** (``CONFIG_PREEMPT_VOLUNTARY=y``) — adds voluntary
  preemption points. Common on server distributions.
* **PREEMPT** / full preemption (``CONFIG_PREEMPT=y``) — the Ubuntu
  *lowlatency* kernel. Preempts anywhere safe in the kernel but spinlocks
  still spin and interrupt handlers still run in hard-IRQ context. Reduces
  average latency but provides **no bounded worst-case guarantee**. This is
  **not** a real-time kernel.
* **PREEMPT_RT** (``CONFIG_PREEMPT_RT=y``) — the *realtime* kernel. Every
  spinlock becomes a sleeping mutex; interrupt handlers run as preemptible
  threads. Provides hard real-time latency guarantees (typically < 100 µs
  worst-case on modern hardware). **Required for robotics motion control.**

Detection signal
================
Kernel build configuration — ``CONFIG_PREEMPT_RT=y`` in
``/boot/config-<release>``. This is the authoritative compile-time record of
the kernel preemption model, written by the distribution build system and
present on all mainstream Linux distributions (Debian, Ubuntu, Fedora, RHEL,
openSUSE). It is the most reliable signal because:

* ``/sys/kernel/realtime`` was never merged into mainline Linux — the core
  kernel maintainers explicitly chose not to land that sysfs patch, so it
  cannot be relied upon.
* ``/proc/config.gz`` requires ``CONFIG_IKCONFIG_PROC=y`` to be compiled in,
  which many distribution and custom kernels disable.

Naming conventions (``uname -r`` suffix, ``uname -v`` marker) are
intentionally excluded: they are user-customizable and can produce false
positives on kernels not compiled with ``CONFIG_PREEMPT_RT``.

The kernel preemption model (``none`` / ``voluntary`` / ``full`` / ``rt``) is
also captured as a parameter for cross-checking and reporting.
"""

import logging
import os
import re
from typing import Optional, Tuple

import allure
import pytest
from sysagent.utils.core import Metrics, Result, run_command

logger = logging.getLogger(__name__)


def _read_text_file(path: str) -> Optional[str]:
    """Read a small text file, returning None if unavailable."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except (IOError, OSError):
        return None


def _get_kernel_release() -> str:
    """Return the running kernel release (``uname -r``)."""
    result = run_command(["uname", "-r"], timeout=10)
    if result and result.returncode == 0 and result.stdout:
        return result.stdout.strip()
    return ""


def _read_kernel_config(kernel_release: str) -> Optional[str]:
    """Return the kernel build configuration text from /boot/config-<release>.

    This is the authoritative compile-time record written by the distribution
    build system. Present on all mainstream distributions (Debian, Ubuntu,
    Fedora, RHEL, openSUSE). Returns None when the file is absent.
    """
    # Sanitize kernel release for safe path construction (defence in depth).
    safe_release = re.sub(r"[^A-Za-z0-9._+\-]", "", kernel_release or "")
    if not safe_release:
        return None
    config_path = os.path.join("/boot", f"config-{safe_release}")
    return _read_text_file(config_path)


def _detect_preemption_model(kernel_config: Optional[str]) -> str:
    """Derive the configured kernel preemption model from build config."""
    if not kernel_config:
        return "unknown"
    if re.search(r"^CONFIG_PREEMPT_RT=y", kernel_config, re.MULTILINE):
        return "rt"
    if re.search(r"^CONFIG_PREEMPT(_LL)?=y", kernel_config, re.MULTILINE):
        return "full"
    if re.search(r"^CONFIG_PREEMPT_VOLUNTARY=y", kernel_config, re.MULTILINE):
        return "voluntary"
    if re.search(r"^CONFIG_PREEMPT_NONE=y", kernel_config, re.MULTILINE):
        return "none"
    return "unknown"


def detect_preempt_rt() -> Tuple[bool, dict]:
    """
    Detect PREEMPT-RT support on the running kernel.

    Reads ``CONFIG_PREEMPT_RT=y`` from ``/boot/config-<release>``, the
    authoritative compile-time record present on all mainstream distributions.
    Naming conventions (release suffix, version string marker) are excluded —
    they are user-customizable and can produce false positives.

    Returns:
        Tuple of (is_rt, details) where details captures the detection result,
        kernel release, and inferred preemption model.
    """
    kernel_release = _get_kernel_release()
    kernel_config = _read_kernel_config(kernel_release)

    config_rt = bool(kernel_config and re.search(r"^CONFIG_PREEMPT_RT=y", kernel_config, re.MULTILINE))
    preemption_model = _detect_preemption_model(kernel_config)

    details = {
        "kernel_release": kernel_release or "unknown",
        "preemption_model": preemption_model,
    }
    return config_rt, details


@allure.title("PREEMPT-RT Kernel Check")
def test_preempt_rt(
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
    Verify the running kernel provides full real-time preemption (PREEMPT_RT).

    Real-time robotics workloads require a PREEMPT_RT kernel to guarantee
    bounded scheduling latency. This test reports ``preempt_rt_enabled``
    (1 = PREEMPT_RT detected, 0 = not detected) as its key metric, detected
    via ``CONFIG_PREEMPT_RT=y`` in ``/boot/config-<release>``.
    """
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info(f"Starting PREEMPT-RT Kernel Check: {test_display_name}")

    # Step 1: Validate system requirements
    validate_system_requirements_from_configs(configs)

    is_qualification = configs.get("labels", {}).get("type") == "qualification"
    result = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    def _run_detection():
        is_rt, details = detect_preempt_rt()

        logger.info(
            f"PREEMPT-RT check: {'DETECTED' if is_rt else 'NOT DETECTED'} "
            f"(model={details['preemption_model']}, kernel={details['kernel_release']})"
        )

        return Result(
            name=f"{test_id} - {test_display_name}",
            metrics={
                # Authoritative signal: CONFIG_PREEMPT_RT=y in /boot/config-<release>.
                "preempt_rt_enabled": Metrics(unit=None, value=1 if is_rt else 0, is_key_metric=True),
                # Kernel preemption model: "rt" | "full" | "voluntary" | "none" | "unknown".
                "preemption_model": Metrics(unit=None, value=details["preemption_model"], is_key_metric=False),
            },
            metadata={"status": is_rt},
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
        failure_message = "Interrupt detected during PREEMPT-RT Kernel Check"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during PREEMPT-RT Kernel Check: {e}"
        logger.error(failure_message, exc_info=True)

    if result is None:
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "PREEMPT-RT check did not complete"},
            metrics={},
        )

    # Step 2: KPI validation (only active when kpi_refs is set in profile)
    try:
        validate_test_results(
            test_name=test_name,
            results=result,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as validation_error:
        logger.error(f"Validation failed: {validation_error}")

    # Step 3: Summarize (always runs)
    try:
        summarize_test_results(
            results=result,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as summary_error:
        logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)

    # Caching is handled by execute_test_with_cache: cached only when
    # preempt_rt_enabled=1 (status=True). A non-RT result is not cached so
    # the test re-runs automatically after a kernel upgrade.

    logger.info(f"PREEMPT-RT Kernel Check completed: {test_display_name}")

    # Surface interrupts/errors as a proper pytest outcome.
    if test_interrupted:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
