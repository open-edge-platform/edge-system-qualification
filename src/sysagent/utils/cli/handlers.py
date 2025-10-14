# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Signal handlers and interrupt management for CLI operations.

Provides centralized signal handling for graceful shutdown and
interrupt management across all CLI commands.
"""

import logging
import signal

from sysagent.utils.core import shared_state

logger = logging.getLogger(__name__)


def handle_interrupt(sig, frame):
    """
    Global signal handler for various interrupts.
    Sets the global flag and re-raises appropriate exception.
    """

    logger.debug(f"handle_interrupt called with signal: {sig}, frame: {frame}")
    # Signal name mapping for better logging
    signal_names = {
        signal.SIGINT: "SIGINT (Keyboard Interrupt)",
        signal.SIGTERM: "SIGTERM (Termination)",
        signal.SIGQUIT: "SIGQUIT (Quit)",
        signal.SIGHUP: "SIGHUP (Hangup)",
    }

    signal_name = signal_names.get(sig, f"Signal {sig}")

    # Update the shared state directly using the shared_state module
    shared_state.INTERRUPT_OCCURRED = True
    shared_state.INTERRUPT_SIGNAL = sig
    shared_state.INTERRUPT_SIGNAL_NAME = signal_name

    logger.warning(f"Interrupt detected: {signal_name}. Stopping all test executions.")

    # Re-raise appropriate exception based on signal type
    if sig == signal.SIGINT:
        raise KeyboardInterrupt()
    elif sig == signal.SIGTERM:
        raise SystemExit(f"Terminated by {signal_name}")
    else:
        raise RuntimeError(f"Interrupted by {signal_name}")
