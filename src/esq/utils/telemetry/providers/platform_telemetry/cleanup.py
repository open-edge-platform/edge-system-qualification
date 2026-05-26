# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Process-wide cleanup orchestrator for platform_telemetry.

Guarantees that **every** runtime artefact this package starts (qmmd
daemon, in-process collector threads) is reaped before the Python
interpreter exits, regardless of how the test session terminates:

* Normal completion (pytest finishes) — handled by
  ``TelemetryCollector.stop()`` calling ``module.close()``.
* Uncaught exception — handled by ``atexit``.
* User Ctrl+C (SIGINT) — handled by the signal hook below.
* External SIGTERM (CI runner kills the job) — same signal hook.

The hook is registered once per process on first import.  Cleanup
callbacks register themselves with :func:`register` and run in LIFO
order so resources are torn down in the reverse order they were
created (typical safe ordering for nested daemons).
"""

from __future__ import annotations

import atexit
import logging
import signal
import threading
from typing import Callable, List

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_callbacks: List[Callable[[], None]] = []
_installed = False
_running = False
_prev_sigint = None
_prev_sigterm = None


def register(cb: Callable[[], None]) -> None:
    """Register ``cb`` to run during process-wide cleanup.

    Callbacks must be idempotent and exception-safe; raised exceptions
    are logged and swallowed so one bad callback cannot block the rest.
    Duplicate registrations are coalesced (same callable registered twice
    is only invoked once).
    """
    global _installed
    with _lock:
        if cb not in _callbacks:
            _callbacks.append(cb)
        if not _installed:
            _install_hooks()
            _installed = True


def _run_callbacks() -> None:
    """Invoke registered callbacks in LIFO order, exception-safe."""
    global _running
    with _lock:
        if _running:
            return  # re-entrancy guard (e.g. SIGINT during atexit)
        _running = True
        cbs = list(reversed(_callbacks))
        _callbacks.clear()
    for cb in cbs:
        try:
            cb()
        except Exception as exc:  # pragma: no cover - best-effort
            logger.debug("platform_telemetry cleanup callback failed: %s", exc)


def _signal_handler(signum, frame):
    """Run cleanup, restore the previous handler, then re-raise the signal.

    Re-raising preserves the original behaviour (e.g. pytest's own
    KeyboardInterrupt handling for SIGINT) so we don't swallow the user's
    Ctrl+C \u2014 we only insert our cleanup ahead of it.
    """
    logger.info("platform_telemetry: received signal %s, running cleanup", signum)
    _run_callbacks()

    # Restore + re-raise so default / previous behaviour applies.
    prev = _prev_sigint if signum == signal.SIGINT else _prev_sigterm
    try:
        signal.signal(signum, prev if callable(prev) else signal.SIG_DFL)
    except (ValueError, OSError):
        pass
    try:
        signal.raise_signal(signum)
    except Exception:  # pragma: no cover
        # raise_signal not available on very old pythons; fall back to default exit.
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)


def _install_hooks() -> None:
    """Wire atexit + SIGINT/SIGTERM handlers exactly once."""
    global _prev_sigint, _prev_sigterm
    atexit.register(_run_callbacks)
    try:
        _prev_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _signal_handler)
    except (ValueError, OSError):
        # signal() may fail when running off the main thread (e.g. inside
        # some pytest plugins or async runners).  atexit alone still gives
        # us coverage for the normal-exit path.
        _prev_sigint = None
    try:
        _prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _signal_handler)
    except (ValueError, OSError):
        _prev_sigterm = None
