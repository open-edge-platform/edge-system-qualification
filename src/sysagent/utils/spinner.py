# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Single-line status spinner for long-running CLI steps.

Used when the CLI runs without ``-v``/``-d`` so a slow step (readiness checks,
dependency downloads) visibly shows activity - indistinguishable from a hang
otherwise. Only the label and an animated spinner glyph are rendered: no
percentage bar, counts, or rate stats, so the line stays as short as its content.

Log records emitted while the spinner is active are written above it (via
``tqdm.write``) instead of being interleaved with the in-place redraw. Falls
back to a no-op when ``tqdm`` is unavailable, or when ``enabled`` is False
(verbose/debug mode, where plain log lines are shown instead).
"""

import itertools
import logging
import sys
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Denser/"filled" braille cells read as a bolder spinner than the thin single-dot
# variant, especially at small terminal font sizes.
# The missing dot travels top-left -> top-right -> bottom-right -> bottom-left: clockwise.
_SPINNER_FRAMES = ["⣾", "⣷", "⣯", "⣟", "⡿", "⢿", "⣻", "⣽"]
_SPINNER_INTERVAL = 0.1


def console_is_quiet() -> bool:
    """True when the console only shows warnings (i.e. no ``-v``/``-d``).

    For code with no verbose/debug flag in scope (e.g. lazily-collected system
    info): in quiet mode a spinner is the only sign of activity, while in
    verbose/debug mode the log lines are. False when there is no console handler.
    """
    console_levels = [
        h.level
        for h in logging.root.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    return bool(console_levels) and all(level >= logging.WARNING for level in console_levels)


# Extra text appended to the active spinner's label (e.g. "attempt 2/3"), so long-running
# steps can report their progress without the spinner's owner having to plumb a callback
# through. A single module-level slot is enough: only one step is spinning at a time.
_label_suffix = ""


def set_spinner_suffix(text: str = "") -> None:
    """Append `` - <text>`` to the active spinner's label; ``""`` clears it. No-op if none is active."""
    global _label_suffix
    _label_suffix = f" - {text}" if text else ""


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
    """Route console log handlers through ``tqdm.write`` while a spinner is active.

    Deliberately NOT ``tqdm.contrib.logging.logging_redirect_tqdm()``: it replaces
    each console handler with a brand-new handler that copies the original's
    formatter/stream but not its ``level``, silently undoing the WARNING/INFO
    console-level filtering ``configure_logging`` sets up for non-verbose/verbose
    mode (DEBUG/INFO lines would leak to the console). This instead swaps only
    ``.stream`` on the existing handler object, so its level is untouched.
    """
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


class Spinner:
    """Context manager owning the status line; use ``spin(label)`` per step.

    Example::

        with Spinner(enabled=not verbose) as spinner:
            for name in names:
                with spinner.spin(f"Verifying {name}"):
                    run_check(name)
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._line = None
        self._redirect = None

    @property
    def active(self) -> bool:
        """True when a spinner is actually being rendered."""
        return self._line is not None

    def __enter__(self) -> "Spinner":
        if self._enabled:
            self._line = self._create_line()
            if self._line is not None:
                self._redirect = _redirect_console_handlers_to_tqdm()
                self._redirect.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._redirect is not None:
            self._redirect.__exit__(*exc_info)
            self._redirect = None
        if self._line is not None:
            self._line.close()
            self._line = None

    @contextmanager
    def spin(self, label: str):
        """Animate ``label`` with a spinner for the duration of the block."""
        if self._line is None:
            yield
            return

        stop_event = threading.Event()
        set_spinner_suffix("")

        def _animate():
            for frame in itertools.cycle(_SPINNER_FRAMES):
                if stop_event.is_set():
                    return
                self._line.set_description_str(f"{frame} {label}{_label_suffix}")
                stop_event.wait(_SPINNER_INTERVAL)

        # Daemon thread so it never blocks interpreter exit.
        thread = threading.Thread(target=_animate, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop_event.set()
            thread.join(timeout=1)
            set_spinner_suffix("")

    @staticmethod
    def _create_line():
        """Create the tqdm-backed status line, or ``None`` if unavailable.

        ``bar_format="{desc}"`` renders only the (spinner-animated) description, so
        the line stays as short as its content instead of stretching to the
        terminal width.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            return None

        try:
            return tqdm(bar_format="{desc}", leave=False, file=sys.stderr, dynamic_ncols=True)
        except Exception as exc:  # noqa: BLE001 - the spinner is purely cosmetic
            logger.debug("Could not create status spinner: %s", exc)
            return None


@contextmanager
def status_spinner(label: str, enabled: bool = True):
    """Show a one-step spinner with ``label`` while the block runs."""
    with Spinner(enabled=enabled) as spinner, spinner.spin(label):
        yield
