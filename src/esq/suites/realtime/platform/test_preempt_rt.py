# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Realtime Platform — PREEMPT-RT Kernel Check.

Detects whether the running Linux kernel provides full real-time preemption
(``PREEMPT_RT``) and, optionally, whether required kernel boot parameters are
present in ``/proc/cmdline``.

Detection uses ``CONFIG_PREEMPT_RT=y`` in ``/boot/config-<release>`` — the
authoritative compile-time record on all mainstream distributions. The Ubuntu*
*lowlatency* kernel (``CONFIG_PREEMPT=y``) is **not** a real-time kernel: it
reduces average latency but gives no bounded worst-case guarantee. The kernel
preemption model (``none`` / ``voluntary`` / ``full`` / ``rt``) is captured as
a supporting metric.

Key metric: ``preempt_rt`` (1 = ready, 0 = not ready). Without
``kernel_boot_params`` in the profile it equals ``preempt_rt_kernel``. With
``kernel_boot_params``, both a PREEMPT_RT kernel and full boot parameter
compliance are required.
"""

import csv
import io
import logging
import os
import re

import allure
import pytest
from sysagent.utils.core import Metrics, Result, run_command

logger = logging.getLogger(__name__)


def _read_text_file(path: str) -> str | None:
    """Read a small text file, returning None if unavailable."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return None


def _read_cmdline() -> str | None:
    """Read the kernel boot command line from /proc/cmdline."""
    return _read_text_file("/proc/cmdline")


def _check_boot_params(cmdline: str | None, required_params: list[str]) -> dict:
    """Check which required boot parameters are present in the kernel cmdline.

    Each entry in ``required_params`` is matched as an exact space-delimited
    token in ``/proc/cmdline`` (e.g., ``"clocksource=tsc"``, ``"nosoftlockup"``).

    Args:
        cmdline: Content of ``/proc/cmdline``, or None if unavailable.
        required_params: List of required boot parameter strings.

    Returns:
        dict with keys:
        - ``compliant``: True if all required params are present.
        - ``present``: list of params found in cmdline.
        - ``missing``: list of params not found in cmdline.
        - ``cmdline_available``: True if ``/proc/cmdline`` was readable.
    """
    if not cmdline:
        return {
            "compliant": False,
            "present": [],
            "missing": list(required_params),
            "cmdline_available": False,
        }

    cmdline_tokens = set(cmdline.strip().split())

    present = []
    missing = []
    for param in required_params:
        if param in cmdline_tokens:
            present.append(param)
        else:
            missing.append(param)

    return {
        "compliant": len(missing) == 0,
        "present": present,
        "missing": missing,
        "cmdline_available": True,
    }


def _get_kernel_release() -> str:
    """Return the running kernel release (``uname -r``)."""
    result = run_command(["uname", "-r"], timeout=10)
    if result and result.returncode == 0 and result.stdout:
        return result.stdout.strip()
    return ""


def _read_kernel_config(kernel_release: str) -> str | None:
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


def _detect_preemption_model(kernel_config: str | None) -> str:
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


def detect_preempt_rt(required_boot_params: list[str] | None = None) -> tuple[bool, dict]:
    """
    Detect PREEMPT-RT support on the running kernel.

    Reads ``CONFIG_PREEMPT_RT=y`` from ``/boot/config-<release>``, the
    authoritative compile-time record present on all mainstream distributions.
    Naming conventions (release suffix, version string marker) are excluded —
    they are user-customizable and can produce false positives.

    When ``required_boot_params`` is provided, also checks ``/proc/cmdline``
    for the presence of each required parameter as a space-delimited token.

    Args:
        required_boot_params: Optional list of kernel boot parameter strings
            to verify are set (e.g., ``["clocksource=tsc", "nosoftlockup"]``).
            When ``None``, boot parameter checking is skipped.

    Returns:
        Tuple of (is_rt, details) where details captures the detection result,
        kernel release, inferred preemption model, and (when requested) boot
        parameter compliance.
    """
    kernel_release = _get_kernel_release()
    kernel_config = _read_kernel_config(kernel_release)

    config_rt = bool(kernel_config and re.search(r"^CONFIG_PREEMPT_RT=y", kernel_config, re.MULTILINE))
    preemption_model = _detect_preemption_model(kernel_config)

    details = {
        "kernel_release": kernel_release or "unknown",
        "preemption_model": preemption_model,
    }

    if required_boot_params is not None:
        cmdline = _read_cmdline()
        details["boot_params"] = _check_boot_params(cmdline, required_boot_params)

    return config_rt, details


def _attach_boot_params_csv(results: Result) -> None:
    """Attach a CSV summary of kernel boot parameter check results to the Allure report.

    Only generates the attachment when ``boot_params_checked`` is present in
    ``results.extended_metadata`` (i.e., when ``kernel_boot_params`` was
    configured for the test).
    """
    boot_params_checked = results.extended_metadata.get("boot_params_checked")
    if not boot_params_checked:
        return

    present_set = set(results.extended_metadata.get("boot_params_present", []))

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["#", "boot_param", "status"])
    for idx, param in enumerate(boot_params_checked, start=1):
        writer.writerow([idx, param, "present" if param in present_set else "missing"])

    allure.attach(
        csv_buf.getvalue(),
        name="kernel_boot_params.csv",
        attachment_type=allure.attachment_type.CSV,
    )


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

    Key metric: ``preempt_rt`` (1 = ready, 0 = not ready). Without
    ``kernel_boot_params`` it equals ``preempt_rt_kernel``; with
    ``kernel_boot_params`` it requires both a PREEMPT_RT kernel and full
    boot parameter compliance.
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

    # Qualification tests fail the run on interrupt; data-collection tests
    # surface it as a runtime error instead. This flag drives that decision.
    is_qualification = configs.get("labels", {}).get("type") == "qualification"

    logger.info(f"Starting PREEMPT-RT Kernel Check: {test_display_name}")

    # ================================================================
    # STEP 2: Validate System Requirements
    # ================================================================
    validate_system_requirements_from_configs(configs)

    # Outcome tracking — initialized before the try block so post-run
    # blocks can always reference them.
    results = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    try:
        # ============================================================
        # STEP 3: Prepare Assets/Dependencies
        # (no external assets required — skipped)
        # ============================================================

        # ============================================================
        # STEP 4: Execute Test Logic (with caching)
        # ============================================================
        def _run_detection():
            required_boot_params = configs.get("kernel_boot_params")
            is_rt, details = detect_preempt_rt(required_boot_params=required_boot_params)

            logger.info(
                f"PREEMPT-RT check: {'DETECTED' if is_rt else 'NOT DETECTED'} "
                f"(model={details['preemption_model']}, kernel={details['kernel_release']})"
            )

            # Compute boot params compliance when configured; None means not checked.
            boot_params_compliant: bool | None = None
            if "boot_params" in details:
                bp = details["boot_params"]
                boot_params_compliant = bp["compliant"]
                if bp["missing"]:
                    logger.warning(
                        f"Missing kernel boot params "
                        f"({len(bp['missing'])}/{len(required_boot_params)}): {bp['missing']}"
                    )
                logger.info(
                    f"Kernel boot params: {len(bp['present'])}/{len(required_boot_params)} present, "
                    f"compliant={boot_params_compliant}"
                )

            # Single key metric: PREEMPT-RT ready when RT kernel is confirmed
            # and (when configured) all required boot parameters are present.
            preempt_rt = is_rt and (boot_params_compliant if boot_params_compliant is not None else True)

            metrics = {
                # Single key metric — covers both check modes.
                "preempt_rt": Metrics(unit=None, value=1 if preempt_rt else 0, is_key_metric=True),
                # Supporting metrics for detailed reporting.
                # CONFIG_PREEMPT_RT=y present in /boot/config-<release>.
                "preempt_rt_kernel": Metrics(unit=None, value=1 if is_rt else 0, is_key_metric=False),
                # Kernel preemption model: "rt" | "full" | "voluntary" | "none" | "unknown".
                "preemption_model": Metrics(unit=None, value=details["preemption_model"], is_key_metric=False),
            }

            if boot_params_compliant is not None:
                metrics["kernel_boot_params"] = Metrics(
                    unit=None, value=1 if boot_params_compliant else 0, is_key_metric=False
                )

            # Store boot params check details for CSV attachment generation.
            # Persisted in extended_metadata so cached results carry the data.
            extended_metadata: dict = {}
            if "boot_params" in details:
                bp = details["boot_params"]
                extended_metadata["boot_params_checked"] = list(required_boot_params) if required_boot_params else []
                extended_metadata["boot_params_present"] = bp["present"]
                extended_metadata["boot_params_missing"] = bp["missing"]
                extended_metadata["boot_params_cmdline_available"] = bp["cmdline_available"]

            return Result(
                name=f"{test_id} - {test_display_name}",
                metrics=metrics,
                metadata={"status": preempt_rt},
                extended_metadata=extended_metadata,
            )

        results = execute_test_with_cache(
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

    # Ensure a result object always exists so validation and summarization
    # can record the outcome even when execution was interrupted.
    if results is None:
        results = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "PREEMPT-RT check did not complete"},
            metrics={},
        )

    # Generate CSV attachment from results — works for both fresh execution and
    # cached results because boot params data lives in extended_metadata.
    _attach_boot_params_csv(results)

    # ================================================================
    # STEP 5: Validate Results Against KPIs (optional)
    # Only enforced when kpi_refs are configured for the test.
    # ================================================================
    try:
        validate_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as validation_error:
        logger.error(f"Validation failed: {validation_error}")

    # ================================================================
    # STEP 6: Generate Summary (always runs)
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

    # Caching is handled by execute_test_with_cache: cached only when
    # preempt_rt=1 (status=True). A non-ready result is not cached so
    # the test re-runs automatically after a kernel upgrade.

    logger.info(f"PREEMPT-RT Kernel Check completed: {test_display_name}")

    # ================================================================
    # STEP 7: Surface the outcome
    # Report interrupts and failures as proper pytest results instead of
    # leaving a broken/errored status behind.
    # ================================================================
    if test_interrupted:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
