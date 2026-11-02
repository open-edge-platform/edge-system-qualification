# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Readiness check orchestration.

Runs the subset of configured dependencies that are ``functional_check``
installs (e.g. OpenVINO device compile/inference smoke tests, outbound
network reachability) BEFORE test execution begins. This is deliberately
separate from the presence-only dependency check that runs for every CLI
command (``check_and_report_dependencies``) - functional checks execute real
(if lightweight) workloads and are only relevant right before tests run.

Rendering is intentionally minimal:
- Without ``-v``/``-d``: a single-line spinner (see ``sysagent.utils.spinner``)
  showing only the check currently in flight (e.g. "Pre-run readiness check: verifying
  network connectivity - attempt 2/3"). Each check's own duration is indeterminate (e.g. a 30s OpenVINO
  device smoke test vs. a <1s network check), hence a spinner instead of a
  static label or percentage bar. Falls back to plain log lines if ``tqdm``
  is unavailable.
- With ``-v``/``-d``: one info/debug log line per check, since a spinner
  would otherwise interleave with/obscure verbose/debug log output.

Whether a failed check blocks execution is driven by each dependency's
``required`` flag in the dependencies.yml configuration - not hardcoded here.
"""

import logging
import os
import shutil
import textwrap

from sysagent.utils.spinner import Spinner

from .functional_checks import split_mark
from .manager import get_dependency_manager
from .schema import DependencyCheckResult, DependencyStatus

logger = logging.getLogger(__name__)

# Statuses that count as "passed" for gating purposes.
_PASSING_STATUSES = (DependencyStatus.INSTALLED, DependencyStatus.MANUAL_REQUIRED, DependencyStatus.FUNCTIONAL)

# Network reachability gates every download the CLI performs (tools, GitHub
# dependencies, PCI IDs, models), so it runs before the rest of the bootstrap.
NETWORK_CHECK_NAME = "network_connectivity"

# Checks that already passed in this process, so a later run_prerun_checks() call
# (e.g. the full pre-test pass after the early network-only pass) doesn't repeat them.
_completed_checks: dict[str, DependencyCheckResult] = {}

# Outcome of every check run in this process (passed or not), in a JSON-serializable
# shape, for embedding in the hardware info (see ``get_prerun_report``).
_prerun_report: dict[str, dict] = {}

# Human-readable labels for known readiness checks - the raw dependency
# config key (e.g. "network_connectivity") isn't self-explanatory on its own.
# Falls back to a space-separated version of the key for any other check.
_CHECK_DISPLAY_NAMES = {
    "network_connectivity": "network connectivity",
    "openvino_devices": "OpenVINO devices",
    "docker_runtime": "Docker runtime",
}


def _display_name(name: str) -> str:
    """Human-readable label for a check name, e.g. "OpenVINO devices"."""
    return _CHECK_DISPLAY_NAMES.get(name, name.replace("_", " "))


# Escape hatch for debugging/troubleshooting only (e.g. isolating whether a readiness
# check itself is the problem): skips every readiness check. NOT an officially supported
# option or configuration - only mentioned in the developer guide's advanced section.
SKIP_ENV = "CORE_SKIP_PRERUN_CHECKS"
_skip_warned = False

# Names this stage on every spinner/log line, so it is identifiable among the other
# stages (downloads, system info collection, tests, ...) when running without -v/-d.
_STAGE_LABEL = "Pre-run readiness check"

# Plain-text (no ANSI, so they stay readable in the log file too) markers
# making a check's outcome scannable at a glance instead of only legible by
# reading its (easily missed) status word.
_STATUS_MARKERS = {
    logging.INFO: "[ OK ]",
    logging.WARNING: "[WARN]",
    logging.ERROR: "[FAIL]",
}


def run_prerun_checks(
    verbose: bool = False,
    debug: bool = False,
    active_flags: set[str] | None = None,
    names: list[str] | None = None,
) -> tuple[bool, dict[str, DependencyCheckResult]]:
    """Run the configured functional readiness checks relevant to this run.

    Args:
        verbose: Render plain log lines instead of a status spinner.
        debug: Same as verbose for rendering purposes.
        active_flags: Profile requirement flags (e.g. {"docker_required"}) known to
            be needed by the profile(s)/test(s) about to run. Checks gated behind a
            ``requires_flag`` not in this set are skipped (e.g. skip the Docker
            runtime check when nothing being run actually needs Docker). Pass None
            (the default) to run every configured check - e.g. when the profile(s)
            about to run aren't known ahead of time.
        names: Restrict the run to these checks (e.g. only network connectivity,
            before any download starts). None runs every applicable check. Checks
            that already passed earlier in this process are never repeated.

    Returns:
        (ok, results) - ``ok`` is False only when a *required* check failed;
        optional check failures are logged as warnings but do not block.
    """
    if _skip_requested():
        return True, {}

    manager = get_dependency_manager()
    check_names = manager.get_functional_dependency_names(active_flags)
    if names is not None:
        check_names = [n for n in check_names if n in names]
    check_names = [n for n in check_names if n not in _completed_checks]

    if active_flags is not None and names is None:
        skipped_names = [
            n
            for n in manager.get_functional_dependency_names(None)
            if n not in check_names and n not in _completed_checks
        ]
        for name in skipped_names:
            logger.debug("Skipping readiness check '%s' - not required by the profile(s)/test(s) being run", name)

    if not check_names:
        logger.debug("No readiness checks configured - skipping")
        return True, {}

    results: dict[str, DependencyCheckResult] = {}
    ok = True
    failed_display_names: list[str] = []

    with Spinner(enabled=not (verbose or debug)) as spinner:
        if not spinner.active:
            logger.info("Running %d readiness check(s)...", len(check_names))

        for name in check_names:
            dependency = manager.config.dependencies[name]
            display = _display_name(name)
            label = f"{_STAGE_LABEL}: verifying {display}"
            logger.info("%s...", label)

            with spinner.spin(label):
                result = manager.check_dependency(name, active_flags)
            results[name] = result

            passed = result.status in _PASSING_STATUSES
            if passed:
                _completed_checks[name] = result
            _prerun_report[name] = {
                "status": result.status.value,
                "required": dependency.required,
                "details": [_structured_detail(line, passed) for line in _detail_items(result)],
            }
            if not passed and dependency.required:
                ok = False
                failed_display_names.append(display)

            log_level = logging.INFO if passed else (logging.ERROR if dependency.required else logging.WARNING)
            marker = _STATUS_MARKERS[log_level]
            logger.log(log_level, "%s %s: %s", marker, _STAGE_LABEL, display)
            for line in _format_check_detail_lines(result):
                logger.log(log_level, "  %s", line)

    if ok:
        logger.info("Readiness checks passed")
    else:
        logger.error("Required readiness check(s) failed: %s - see above for details", ", ".join(failed_display_names))

    return ok, results


def _skip_requested() -> bool:
    """True when ``CORE_SKIP_PRERUN_CHECKS`` is set to a truthy value (warns once per process)."""
    global _skip_warned
    if os.environ.get(SKIP_ENV, "").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    if not _skip_warned:
        _skip_warned = True
        logger.warning("%s is set - skipping pre-run readiness checks (debug/troubleshooting only)", SKIP_ENV)
    return True


def _detail_items(result: DependencyCheckResult) -> list[str]:
    return result.details.split("\n") if result.details else [result.message]


def _structured_detail(line: str, default_ok: bool) -> dict:
    """``{"ok", "message"}`` for one sub-component line, without its display glyph."""
    ok, text = split_mark(line)
    return {"ok": default_ok if ok is None else ok, "message": text}


def reset_prerun_state() -> None:
    """Forget readiness results from earlier in this process.

    Readiness is never reused across runs: call this at the start of a run so every
    check executes again. Within a run, checks that already passed (e.g. the early
    network-only pass) are still skipped by the later full pass.
    """
    _completed_checks.clear()
    _prerun_report.clear()


def get_prerun_report() -> dict[str, dict]:
    """Outcome of every readiness check run so far in this process.

    Maps check name to ``{status, required, details}``, where ``details`` is a list of
    ``{ok, message}`` per sub-component (host, device, ...). Empty when no
    readiness check ran (e.g. commands other than ``run``).
    """
    return {name: dict(entry) for name, entry in _prerun_report.items()}


def _format_check_detail_lines(result: DependencyCheckResult) -> list[str]:
    """Split a check result into one terminal-width-wrapped line per sub-component.

    Checks spanning multiple sub-components (e.g. several OpenVINO devices,
    or several network hosts) would otherwise render as a single, very long
    semicolon-joined line. This instead gives each sub-component its own
    line, wrapped to fit the current terminal width.
    """
    items = _detail_items(result)
    width = max(shutil.get_terminal_size(fallback=(100, 24)).columns - 6, 40)

    lines = []
    for item in items:
        wrapped = textwrap.wrap(item, width=width) or [item]
        # Sub-component lines already start with a ✓/✗ glyph (see functional_checks._mark);
        # align any wrapped continuation lines under the glyph instead of re-marking them.
        lines.append(wrapped[0])
        lines.extend(f"  {cont}" for cont in wrapped[1:])
    return lines
