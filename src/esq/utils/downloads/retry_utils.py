# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Retry utilities for robust download operations.

This module provides retry decorators and utilities to handle intermittent
network failures during download operations.
"""

import functools
import logging
import time
from typing import Any, Callable, Optional, Tuple, Type

import requests

logger = logging.getLogger(__name__)

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
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    cleanup_on_failure: Optional[Callable[[Any], None]] = None,
) -> Callable:
    """
    Decorator to retry a function on transient network failures.

    Implements exponential backoff between retries to handle intermittent
    network issues gracefully.

    Args:
        max_attempts: Maximum number of attempts (including initial try)
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry (exponential backoff)
        retryable_exceptions: Tuple of exception types to retry on
        cleanup_on_failure: Optional cleanup function called with function args on failure

    Returns:
        Decorated function with retry logic

    Example:
        >>> @with_retry(max_attempts=3, initial_delay=2.0)
        ... def download_file(url, dest):
        ...     # Download logic here
        ...     pass
    """
    if retryable_exceptions is None:
        retryable_exceptions = RETRYABLE_EXCEPTIONS

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
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

                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        # Final attempt failed
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")

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
                raise RuntimeError(f"{func.__name__} failed after {max_attempts} attempts")

        return wrapper

    return decorator


def retry_download(
    download_func: Callable,
    *args,
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    cleanup_func: Optional[Callable] = None,
    **kwargs,
) -> Any:
    """
    Retry a download function with exponential backoff.

    This is a functional interface alternative to the @with_retry decorator,
    useful for dynamic retry logic or wrapping existing functions.

    Args:
        download_func: Function to retry
        *args: Positional arguments to pass to download_func
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds before first retry
        cleanup_func: Optional cleanup function called on failure
        **kwargs: Keyword arguments to pass to download_func

    Returns:
        Result from download_func

    Raises:
        Exception: Last exception raised by download_func after all retries

    Example:
        >>> result = retry_download(
        ...     requests.get,
        ...     "https://example.com/file.zip",
        ...     max_attempts=3,
        ...     stream=True
        ... )
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            return download_func(*args, **kwargs)

        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e

            # Don't retry non-transient HTTP errors
            if isinstance(e, requests.exceptions.HTTPError):
                if hasattr(e, "response") and e.response is not None:
                    status_code = e.response.status_code
                    if 400 <= status_code < 500 and status_code != 429:
                        logger.error(f"Download failed with HTTP {status_code} (non-retryable): {e}")
                        raise

            if attempt < max_attempts:
                logger.warning(f"Download attempt {attempt}/{max_attempts} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= initial_delay  # Exponential backoff
            else:
                logger.error(f"Download failed after {max_attempts} attempts: {e}")

    # All retries exhausted
    if cleanup_func:
        try:
            cleanup_func()
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed: {cleanup_error}")

    if last_exception:
        raise last_exception
    else:
        raise RuntimeError(f"Download failed after {max_attempts} attempts")
