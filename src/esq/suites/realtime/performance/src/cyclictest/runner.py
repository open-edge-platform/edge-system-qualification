# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Process execution for cyclictest with concurrent stress-ng and a progress bar.

Manages the lifecycle of both processes and ensures they are cleaned up
unconditionally, regardless of how execution exits (normal completion,
interrupt, or exception).
"""

import logging
import threading
import time

from sysagent.utils.core import start_process

logger = logging.getLogger(__name__)


def _run_progress_thread(duration_seconds: int, stop_event: threading.Event) -> None:
    """Show a tqdm time-based progress bar written directly to /dev/tty.

    Using ``/dev/tty`` (the controlling terminal) rather than ``sys.stderr``
    ensures the bar never ends up in captured stderr, Allure attachments, or
    log files. Silently skipped when no controlling terminal is available
    (CI, non-interactive sessions, etc.).
    """
    try:
        from tqdm import tqdm
    except ImportError as exc:
        logger.debug("Progress bar skipped (tqdm not installed): %s", exc)
        return

    try:
        tty = open("/dev/tty", "w")
    except OSError as exc:
        logger.debug("Progress bar skipped (no controlling terminal): %s", exc)
        return

    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}, {rate_fmt}]"
    try:
        with tqdm(
            total=duration_seconds,
            desc="cyclictest",
            unit="s",
            file=tty,
            leave=False,
            bar_format=bar_format,
            dynamic_ncols=True,
        ) as pbar:
            last_n = 0
            start = time.monotonic()
            while not stop_event.is_set():
                elapsed = int(time.monotonic() - start)
                new_n = min(elapsed, duration_seconds)
                delta = new_n - last_n
                if delta > 0:
                    pbar.update(delta)
                    last_n = new_n
                if last_n >= duration_seconds:
                    break
                stop_event.wait(timeout=1)
            remaining = duration_seconds - last_n
            if remaining > 0:
                pbar.update(remaining)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Progress bar error: %s", exc)
    finally:
        tty.close()


def run_cyclictest_with_stress(
    cyclic_command: list[str],
    stress_command: list[str] | None,
    duration_seconds: int,
    timeout: int,
) -> dict:
    """Launch cyclictest with optional concurrent stress-ng and a tqdm progress bar.

    Returns a dict with keys: ``returncode``, ``success``, ``stderr``,
    ``interrupted``.  All processes are cleaned up unconditionally in the
    finally block.
    """
    stress_handle = None
    cyclic_handle = None
    stop_event = threading.Event()

    try:
        if stress_command:
            stress_handle = start_process(stress_command)
            logger.info(
                "stress-ng started (PID=%s): %s",
                stress_handle.pid,
                " ".join(stress_command),
            )

        cyclic_handle = start_process(cyclic_command)
        logger.info(
            "cyclictest started (PID=%s): %s",
            cyclic_handle.pid,
            " ".join(cyclic_command),
        )

        progress_thread = threading.Thread(
            target=_run_progress_thread,
            args=(duration_seconds, stop_event),
            daemon=True,
        )
        progress_thread.start()

        interrupted = False
        try:
            rc = cyclic_handle.wait(timeout=timeout)
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Interrupt received; sending SIGTERM to cyclictest for graceful shutdown")
            rc = -1

        if rc == -1:
            if not interrupted:
                logger.warning("cyclictest exceeded timeout (%ss); sending SIGTERM", timeout)
            cyclic_handle.terminate()
            # Give cyclictest up to 10 s to flush --json= output after SIGTERM
            try:
                cyclic_handle.wait(timeout=10)
            except Exception:
                pass

        stop_event.set()
        progress_thread.join(timeout=5)

        cyclic_stderr = cyclic_handle.read_stderr()
        returncode = cyclic_handle.returncode if cyclic_handle.returncode is not None else -1
        logger.info("cyclictest finished (returncode=%s, interrupted=%s)", returncode, interrupted)

        return {
            "returncode": returncode,
            "success": returncode == 0 and not interrupted,
            "stderr": cyclic_stderr,
            "interrupted": interrupted,
        }

    finally:
        stop_event.set()
        if cyclic_handle is not None:
            cyclic_handle.terminate()
        if stress_handle is not None:
            stress_handle.terminate()
        logger.debug("Process cleanup complete")
