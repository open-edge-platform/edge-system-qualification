# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Shared runner for the model/dataset download CLIs (``hf``, ``modelscope``).

Both CLIs resume partially downloaded files, so a failed run is retried with the shared
retry policy (``CORE_NETWORK_RETRY_ATTEMPTS``, default 3). Only a non-zero exit is
retried: a timeout (up to ``timeout`` seconds per attempt) is not, since repeating it
would multiply an already long wait.
"""

from sysagent.utils.core import run_command
from sysagent.utils.infrastructure.retry import with_retry


class DownloadCommandError(RuntimeError):
    """A download CLI exited non-zero (network failure, missing repo, auth, ...)."""


@with_retry(retryable_exceptions=(DownloadCommandError,))
def run_download_command(cmd: list[str], timeout: float, description: str) -> None:
    """Run a download CLI, streaming its output.

    Raises:
        TimeoutError: If the command timed out (not retried).
        DownloadCommandError: If the command exited non-zero (retried).
    """
    result = run_command(command=cmd, timeout=timeout, stream_output=True)

    if result.timed_out:
        raise TimeoutError(f"{description} timed out after {timeout}s")

    if result.returncode != 0:
        error_msg = f"{description} failed with return code {result.returncode}"
        if result.stderr:
            error_msg += f": {result.stderr}"
        raise DownloadCommandError(error_msg)
