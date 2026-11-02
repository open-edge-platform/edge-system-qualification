# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Retry utilities for robust download operations.

Shared by the downloaders (tool/dependency archives, files, Node.js, PCI IDs, model and
container image pulls, installers) so they all handle intermittent network failures the same way: a few
attempts with a fixed delay between them, no retry for non-transient HTTP 4xx errors.
"""

import functools
import logging
import os
import time
from collections.abc import Callable
from typing import Any

import requests

from sysagent.utils.spinner import set_spinner_suffix

logger = logging.getLogger(__name__)

# Defaults shared by every downloader and by the network readiness check.
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_INITIAL_DELAY = 2.0

# Environment override for the number of attempts (1 = no retry), applied to the
# downloaders and the network readiness check alike. For testing different retry
# behavior without editing source/config.
RETRY_ATTEMPTS_ENV = "CORE_NETWORK_RETRY_ATTEMPTS"

_warned_invalid_values: set[str] = set()


def get_retry_attempts(default: int | None = None) -> int:
    """Resolve the number of attempts: env override, else ``default``, else the built-in default.

    ``CORE_NETWORK_RETRY_ATTEMPTS`` wins when set to a positive integer; an invalid
    value (non-integer, < 1) is logged once as a warning and ignored.
    """
    raw = os.environ.get(RETRY_ATTEMPTS_ENV, "").strip()
    if raw:
        try:
            attempts = int(raw)
            if attempts >= 1:
                return attempts
        except ValueError:
            pass
        if raw not in _warned_invalid_values:
            _warned_invalid_values.add(raw)
            logger.warning("Ignoring invalid %s=%r (expected a positive integer)", RETRY_ATTEMPTS_ENV, raw)
    return default if default is not None else DEFAULT_MAX_ATTEMPTS


# Network errors that are considered transient and worth retrying
RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    TimeoutError,
    ConnectionError,
    # Include generic OSError for network-related issues
    OSError,
)


def with_retry(
    max_attempts: int | None = None,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    backoff_factor: float = 1.0,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
    cleanup_on_failure: Callable[[Any], None] | None = None,
) -> Callable:
    """
    Decorator to retry a function on transient network failures.

    Waits a fixed ``initial_delay`` between retries by default (``backoff_factor=1.0``)
    to handle intermittent network issues; use a factor > 1 for exponential backoff.
    The current attempt is shown on the status spinner, if one is active.

    Args:
        max_attempts: Maximum number of attempts (including initial try). Defaults to 5;
            the ``CORE_NETWORK_RETRY_ATTEMPTS`` environment variable overrides it.
        initial_delay: Delay in seconds before the first retry
        backoff_factor: Multiplier for the delay after each retry (1.0 = fixed delay)
        retryable_exceptions: Tuple of exception types to retry on
        cleanup_on_failure: Optional cleanup function called with function args on failure

    Returns:
        Decorated function with retry logic

    Example:
        >>> @with_retry(max_attempts=5, initial_delay=2.0)
        ... def download_file(url, dest):
        ...     # Download logic here
        ...     pass
    """
    if retryable_exceptions is None:
        retryable_exceptions = RETRYABLE_EXCEPTIONS

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return run_with_retries(*args, **kwargs)
            finally:
                set_spinner_suffix("")

        def run_with_retries(*args, **kwargs):
            max_attempts_resolved = get_retry_attempts(max_attempts)
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts_resolved + 1):
                # Only surface the attempt once a retry happens: the first try is just the
                # normal operation (often a cache hit, with nothing to download).
                if attempt > 1:
                    set_spinner_suffix(f"download retry {attempt}/{max_attempts_resolved}")
                try:
                    # Attempt the operation
                    return func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e

                    # Don't retry HTTP errors that aren't transient (4xx client errors)
                    if isinstance(e, requests.exceptions.HTTPError):
                        if hasattr(e, "response") and e.response is not None:
                            status_code = e.response.status_code
                            # 4xx errors (except 429 Rate Limit) are not retryable
                            if 400 <= status_code < 500 and status_code != 429:
                                logger.error(f"{func.__name__} failed with HTTP {status_code} (non-retryable): {e}")
                                raise

                    if attempt < max_attempts_resolved:
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts_resolved} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        # Final attempt failed
                        logger.error(f"{func.__name__} failed after {max_attempts_resolved} attempts: {e}")

                except Exception as e:
                    # Non-retryable exception - fail immediately
                    logger.error(f"{func.__name__} failed with non-retryable error: {e}")
                    if cleanup_on_failure:
                        try:
                            cleanup_on_failure(*args, **kwargs)
                        except Exception as cleanup_error:
                            logger.error(f"Cleanup failed: {cleanup_error}")
                    raise

            # All retries exhausted
            if cleanup_on_failure:
                try:
                    cleanup_on_failure(*args, **kwargs)
                except Exception as cleanup_error:
                    logger.error(f"Cleanup failed: {cleanup_error}")

            # Re-raise the last exception
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"{func.__name__} failed after {max_attempts_resolved} attempts")

        return wrapper

    return decorator


# Output fragments of CLI tools (docker, compose, ...) that indicate a transient network or
# registry problem worth retrying, as opposed to a permanent one (unknown image, bad
# credentials, malformed compose file) that a retry cannot fix.
TRANSIENT_ERROR_MARKERS = (
    "tls handshake timeout",
    "i/o timeout",
    "timeout",
    "timed out",
    "context deadline exceeded",
    "temporary failure",
    "connection reset",
    "connection refused",
    "unexpected eof",
    "eof",
    "no route to host",
    "network is unreachable",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "too many requests",
)


def is_transient_error_text(error_text: str | None) -> bool:
    """True if a tool's error output looks like a transient network/registry failure."""
    lowered = str(error_text or "").lower()
    return any(marker in lowered for marker in TRANSIENT_ERROR_MARKERS)


def run_command_with_retry(
    command: list[str],
    retries: int | None = None,
    delay: float = DEFAULT_INITIAL_DELAY,
    backoff: float = 1.0,
    description: str = "Command",
    **run_kwargs: Any,
):
    """Run ``command`` (see ``sysagent.utils.core.run_command``), retrying transient failures.

    Only failures whose output looks transient (``is_transient_error_text``) are retried; a
    timeout or a permanent failure is returned immediately. The attempt count follows
    ``CORE_NETWORK_RETRY_ATTEMPTS`` (else ``retries``, else 3) with a fixed ``delay``
    unless ``backoff`` > 1.

    Returns:
        The last ``CommandResult``, so callers keep their existing failure handling.
    """
    from sysagent.utils.core import run_command

    attempts = get_retry_attempts(retries)
    result = None
    try:
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                set_spinner_suffix(f"download retry {attempt}/{attempts}")
            result = run_command(command, **run_kwargs)
            if result.success or result.timed_out:
                return result

            error_text = result.stderr or result.stdout
            if attempt == attempts or not is_transient_error_text(error_text):
                return result

            logger.warning(
                "%s attempt %d/%d failed (transient network error); retrying in %.1fs. %s",
                description,
                attempt,
                attempts,
                delay,
                str(error_text).strip().splitlines()[-1] if str(error_text).strip() else "",
            )
            time.sleep(delay)
            delay *= backoff
        return result
    finally:
        set_spinner_suffix("")
