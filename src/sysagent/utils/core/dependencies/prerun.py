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
- Without ``-v``/``-d``: a single-line spinner showing only the check
  currently in flight (name plus its position, e.g. "(1/2)") - no
  percentage bar or counts, since those render as a fixed-width bar that
  stretches to the terminal width regardless of content. Each check's own
  duration is indeterminate (e.g. a 30s OpenVINO device smoke test vs. a
  <1s network check), hence the spinner instead of a static label. Falls
  back to plain log lines if ``tqdm`` is unavailable.
- With ``-v``/``-d``: one info/debug log line per check, since a progress
  bar would otherwise interleave with/obscure verbose/debug log output.

Whether a failed check blocks execution is driven by each dependency's
``required`` flag in the dependencies.yml configuration - not hardcoded here.
"""

import itertools
import logging
import shutil
import sys
import textwrap
import threading
from contextlib import contextmanager, nullcontext

from .manager import get_dependency_manager
from .schema import DependencyCheckResult, DependencyStatus

logger = logging.getLogger(__name__)

# Statuses that count as "passed" for gating purposes.
_PASSING_STATUSES = (DependencyStatus.INSTALLED, DependencyStatus.MANUAL_REQUIRED, DependencyStatus.FUNCTIONAL)

# Animation frames for the in-flight check spinner. These denser/"filled"
# braille cells read as a bolder spinner than the thin single-dot variant,
# especially at small terminal font sizes.
_SPINNER_FRAMES = ["\u28fe", "\u28fd", "\u28fb", "\u28bf", "\u287f", "\u28df", "\u28ef", "\u28f7"]
_SPINNER_INTERVAL = 0.1

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


# Plain-text (no ANSI, so they stay readable in the log file too) markers
# making a check's outcome scannable at a glance instead of only legible by
# reading its (easily missed) status word.
_STATUS_MARKERS = {
    logging.INFO: "[ OK ]",
    logging.WARNING: "[WARN]",
    logging.ERROR: "[FAIL]",
}


def run_prerun_checks(
    verbose: bool = False, debug: bool = False, active_flags: set[str] | None = None
) -> tuple[bool, dict[str, DependencyCheckResult]]:
    """Run the configured functional readiness checks relevant to this run.

    Args:
        verbose: Render plain log lines instead of a progress bar.
        debug: Same as verbose for rendering purposes.
        active_flags: Profile requirement flags (e.g. {"docker_required"}) known to
            be needed by the profile(s)/test(s) about to run. Checks gated behind a
            ``requires_flag`` not in this set are skipped (e.g. skip the Docker
            runtime check when nothing being run actually needs Docker). Pass None
            (the default) to run every configured check - e.g. when the profile(s)
            about to run aren't known ahead of time.

    Returns:
        (ok, results) - ``ok`` is False only when a *required* check failed;
        optional check failures are logged as warnings but do not block.
    """
    manager = get_dependency_manager()
    check_names = manager.get_functional_dependency_names(active_flags)

    if active_flags is not None:
        skipped_names = [n for n in manager.get_functional_dependency_names(None) if n not in check_names]
        for name in skipped_names:
            logger.debug("Skipping readiness check '%s' - not required by the profile(s)/test(s) being run", name)

    if not check_names:
        logger.debug("No readiness checks configured - skipping")
        return True, {}

    use_progress_bar = not (verbose or debug)
    progress = _create_progress_bar(len(check_names)) if use_progress_bar else None

    if progress is None:
        logger.info("Running %d readiness check(s)...", len(check_names))

    results: dict[str, DependencyCheckResult] = {}
    ok = True
    failed_display_names: list[str] = []
    total = len(check_names)

    with _logging_redirect_context(progress):
        for i, name in enumerate(check_names, 1):
            dependency = manager.config.dependencies[name]
            display = _display_name(name)
            label = f"Verifying {display} ({i}/{total})"
            logger.info("%s...", label)

            stop_spinner = _start_spinner(progress, label) if progress is not None else None
            try:
                result = manager.check_dependency(name)
            finally:
                if stop_spinner is not None:
                    stop_spinner()
            results[name] = result

            passed = result.status in _PASSING_STATUSES
            if not passed and dependency.required:
                ok = False
                failed_display_names.append(display)

            log_level = logging.INFO if passed else (logging.ERROR if dependency.required else logging.WARNING)
            marker = _STATUS_MARKERS[log_level]
            logger.log(
                log_level,
                "%s %s%s (%d/%d): %s",
                marker,
                display[:1].upper(),
                display[1:],
                i,
                total,
                result.status.value,
            )
            for line in _format_check_detail_lines(result):
                logger.log(log_level, "  %s", line)

            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    if ok:
        logger.info("Readiness checks passed")
    else:
        logger.error("Required readiness check(s) failed: %s - see above for details", ", ".join(failed_display_names))

    return ok, results


def _logging_redirect_context(progress):
    """Context manager that safely interleaves log output with an active tqdm bar.

    Without this, a log line emitted mid-check gets written at the bar's
    current (mid-redraw, no-newline) cursor position instead of its own
    line, so the two visually run together on one line. No-op when there is
    no bar to protect (verbose/debug mode, or tqdm unavailable).

    Deliberately NOT ``tqdm.contrib.logging.logging_redirect_tqdm()``: it
    replaces each console handler with a brand-new handler that copies the
    original's formatter/stream but not its ``level``, silently undoing the
    WARNING/INFO console-level filtering ``configure_logging`` sets up for
    non-verbose/verbose mode (DEBUG/INFO lines would leak to the console).
    This instead swaps only ``.stream`` on the existing handler object, so
    its level (and everything else) is untouched.
    """
    if progress is None:
        return nullcontext()
    try:
        import tqdm as _tqdm_module  # noqa: F401 - just validating availability
    except ImportError:
        return nullcontext()
    return _redirect_console_handlers_to_tqdm()


class _TqdmWriteStream:
    """Proxies a logging handler's stream so writes go through ``tqdm.write()``."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text: str) -> None:
        from tqdm import tqdm

        text = text.rstrip("\n")
        if text:
            tqdm.write(text, file=self._stream)

    def flush(self) -> None:
        self._stream.flush()


@contextmanager
def _redirect_console_handlers_to_tqdm():
    console_handlers = [
        h
        for h in logging.root.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    originals = {h: h.stream for h in console_handlers}
    for h in console_handlers:
        h.stream = _TqdmWriteStream(h.stream)
    try:
        yield
    finally:
        for h, stream in originals.items():
            h.stream = stream


def _format_check_detail_lines(result: DependencyCheckResult) -> list[str]:
    """Split a check result into one terminal-width-wrapped line per sub-component.

    Checks spanning multiple sub-components (e.g. several OpenVINO devices,
    or several network hosts) would otherwise render as a single, very long
    semicolon-joined line. This instead gives each sub-component its own
    line, wrapped to fit the current terminal width.
    """
    items = result.details.split("\n") if result.details else [result.message]
    width = max(shutil.get_terminal_size(fallback=(100, 24)).columns - 6, 40)

    lines = []
    for item in items:
        wrapped = textwrap.wrap(item, width=width) or [item]
        # Sub-component lines already start with a ✓/✗ glyph (see functional_checks._mark);
        # align any wrapped continuation lines under the glyph instead of re-marking them.
        lines.append(wrapped[0])
        lines.extend(f"  {cont}" for cont in wrapped[1:])
    return lines


def _start_spinner(progress, label: str):
    """Animate ``progress``'s description with a spinner while a check is in flight.

    Each check's own duration is indeterminate, so without this the status
    line would sit frozen on ``label`` - indistinguishable from a hang - for
    however long that check takes. Runs in a daemon thread so it never
    blocks interpreter exit.

    Returns a zero-arg callable that stops the animation.
    """
    stop_event = threading.Event()

    def _animate():
        for frame in itertools.cycle(_SPINNER_FRAMES):
            if stop_event.is_set():
                return
            progress.set_description_str(f"{frame} {label}")
            stop_event.wait(_SPINNER_INTERVAL)

    thread = threading.Thread(target=_animate, daemon=True)
    thread.start()

    def _stop():
        stop_event.set()
        thread.join(timeout=1)

    return _stop


def _create_progress_bar(total: int):
    """Create a minimal tqdm-based status line, or ``None`` if tqdm is unavailable.

    ``bar_format="{desc}"`` renders only the (spinner-animated) description -
    no percentage bar, fraction counts, or rate stats - so the line stays as
    short as its content instead of stretching to the terminal width.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        return None

    try:
        return tqdm(total=total, bar_format="{desc}", leave=False, file=sys.stderr, dynamic_ncols=True)
    except Exception as exc:  # noqa: BLE001 - progress bar is purely cosmetic
        logger.debug("Could not create prerun check progress bar: %s", exc)
        return None
