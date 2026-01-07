# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Minimal subprocess execution utilities for container environments.

This module provides a lightweight alternative to the full sysagent.utils.core.process
module for use in Docker containers. It contains only the essential functionality
needed for media benchmarking without the full framework overhead.

For development/testing outside containers, use the full sysagent.utils.core.process module.
"""

import subprocess  # nosec B404
import time
from typing import Dict, List, Optional, Union


class ProcessResult:
    """
    Container for subprocess execution results.

    Provides a simplified version of sysagent.utils.core.process.ProcessResult
    with only the fields used by media benchmarks.
    """

    def __init__(
        self,
        returncode: int,
        stdout: str = "",
        stderr: str = "",
        command: List[str] = None,
        execution_time: float = 0.0,
        timed_out: bool = False,
    ):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.command = command or []
        self.execution_time = execution_time
        self.timed_out = timed_out

    @property
    def success(self) -> bool:
        """Check if the command executed successfully."""
        return self.returncode == 0 and not self.timed_out

    @property
    def failed(self) -> bool:
        """Check if the command failed."""
        return not self.success

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"ProcessResult(status={status}, returncode={self.returncode}, time={self.execution_time:.2f}s)"


def run_command(
    command: Union[str, List[str]],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    check: bool = False,
    capture_output: bool = True,
) -> ProcessResult:
    """
    Execute a command securely.

    This is a simplified version of sysagent.utils.core.process.run_command()
    designed for container environments. It provides basic subprocess execution
    without the full security framework.

    Args:
        command: Command to execute (string or list of arguments)
        cwd: Working directory for command execution
        env: Environment variables (merged with os.environ if provided)
        timeout: Execution timeout in seconds
        check: If True, raise exception on non-zero exit code
        capture_output: If True, capture stdout and stderr

    Returns:
        ProcessResult: Object containing execution results

    Raises:
        subprocess.CalledProcessError: If check=True and command returns non-zero
        subprocess.TimeoutExpired: If command exceeds timeout
    """
    import os
    import shlex

    start_time = time.time()

    # Prepare command
    if isinstance(command, str):
        # Parse shell command safely
        cmd_list = shlex.split(command)
    elif isinstance(command, list):
        cmd_list = [str(arg) for arg in command]
    else:
        raise ValueError(f"Invalid command type: {type(command)}")

    # Prepare environment
    if env:
        full_env = os.environ.copy()
        full_env.update(env)
    else:
        full_env = None

    # Execute command
    try:
        if capture_output:
            result = subprocess.run(
                cmd_list,
                cwd=cwd,
                env=full_env,
                timeout=timeout,
                check=check,
                capture_output=True,
                text=True,
            )
            execution_time = time.time() - start_time
            return ProcessResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                command=cmd_list,
                execution_time=execution_time,
            )
        else:
            result = subprocess.run(
                cmd_list,
                cwd=cwd,
                env=full_env,
                timeout=timeout,
                check=check,
            )
            execution_time = time.time() - start_time
            return ProcessResult(
                returncode=result.returncode,
                command=cmd_list,
                execution_time=execution_time,
            )

    except subprocess.TimeoutExpired as e:
        execution_time = time.time() - start_time
        result = ProcessResult(
            returncode=-1,
            stdout=getattr(e, "stdout", "") or "",
            stderr=getattr(e, "stderr", "") or "",
            command=cmd_list,
            execution_time=execution_time,
            timed_out=True,
        )
        if check:
            raise
        return result

    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        result = ProcessResult(
            returncode=e.returncode,
            stdout=getattr(e, "stdout", "") or "",
            stderr=getattr(e, "stderr", "") or "",
            command=cmd_list,
            execution_time=execution_time,
        )
        if check:
            raise
        return result

    except Exception as e:
        execution_time = time.time() - start_time
        result = ProcessResult(
            returncode=-1,
            stderr=str(e),
            command=cmd_list,
            execution_time=execution_time,
        )
        if check:
            raise
        return result


def secure_popen(
    command: Union[str, List[str]],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    stdout=None,
    stderr=None,
    stdin=None,
    text: bool = True,
    **kwargs,
) -> subprocess.Popen:
    """
    Create a Popen process securely for long-running commands.

    This function provides a secure wrapper around subprocess.Popen for cases
    where direct process control is needed (monitoring, background tasks, etc.).
    All commands are validated and executed in list format to prevent injection.

    Args:
        command: Command to execute (string or list of arguments)
        cwd: Working directory for command execution
        env: Environment variables (merged with os.environ if provided)
        stdout: stdout handling (e.g., subprocess.PIPE, file object)
        stderr: stderr handling (e.g., subprocess.PIPE, subprocess.DEVNULL)
        stdin: stdin handling (e.g., subprocess.DEVNULL)
        text: If True, decode output as text
        **kwargs: Additional arguments passed to subprocess.Popen

    Returns:
        subprocess.Popen: Process object for monitoring/control
    """
    import shlex

    # Prepare command - always use list format for security
    if isinstance(command, str):
        cmd_list = shlex.split(command)
    elif isinstance(command, list):
        cmd_list = [str(arg) for arg in command]
    else:
        raise ValueError(f"Invalid command type: {type(command)}")

    # Prepare environment
    if env:
        full_env = os.environ.copy()
        full_env.update(env)
    else:
        full_env = None

    # Create Popen process with validated inputs
    return subprocess.Popen(
        cmd_list, cwd=cwd, env=full_env, stdout=stdout, stderr=stderr, stdin=stdin, text=text, **kwargs
    )
