# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Secure subprocess execution utilities.

This module provides a centralized, secure way to execute subprocess commands
throughout the application. It implements security best practices including:
- Command validation and sanitization
- Secure environment handling
- Standardized error handling
- Logging and monitoring
- Timeout management
- Resource management

All subprocess usage across the application should use this module instead
of direct subprocess calls to ensure consistency and security.
"""

import logging
import os
import shlex
import shutil
import signal
import subprocess  # nosec B404 # For secure process execution API
import threading
import time
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessResult:
    """
    Container for subprocess execution results with enhanced metadata.
    """

    def __init__(
        self,
        returncode: int,
        stdout: str = "",
        stderr: str = "",
        command: list[str] = None,
        execution_time: float = 0.0,
        pid: int | None = None,
        timed_out: bool = False,
    ):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.command = command or []
        self.execution_time = execution_time
        self.pid = pid
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


class ProcessExecutionMode(Enum):
    """Execution modes for subprocess operations."""

    CAPTURE = "capture"  # Capture stdout/stderr
    PIPE = "pipe"  # Real-time streaming
    BACKGROUND = "background"  # Fire and forget
    INTERACTIVE = "interactive"  # Interactive session


class ProcessSecurityConfig:
    """Security configuration for subprocess execution."""

    def __init__(
        self,
        allowed_commands: list[str] | None = None,
        blocked_commands: list[str] | None = None,
        allowed_paths: list[str] | None = None,
        max_execution_time: float = 300.0,  # 5 minutes default
        max_memory_mb: int | None = None,
        sanitize_environment: bool = True,
        allow_shell: bool = False,
        log_commands: bool = True,
    ):
        self.allowed_commands = set(allowed_commands or [])
        self.blocked_commands = set(blocked_commands or [])
        self.allowed_paths = [Path(p) for p in (allowed_paths or [])]
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.sanitize_environment = sanitize_environment
        self.allow_shell = allow_shell
        self.log_commands = log_commands


class SecureProcessExecutor:
    """
    Secure subprocess executor with comprehensive security controls.

    This class provides a secure interface for executing subprocess commands
    with built-in security controls, logging, and error handling.
    """

    def __init__(self, security_config: ProcessSecurityConfig | None = None):
        """
        Initialize the secure process executor.

        Args:
            security_config: Security configuration for subprocess execution
        """
        self.security_config = security_config or ProcessSecurityConfig()
        self._active_processes: dict[int, subprocess.Popen] = {}
        self._process_lock = threading.Lock()

    def run(
        self,
        command: str | list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
        input_data: str | None = None,
        mode: ProcessExecutionMode = ProcessExecutionMode.CAPTURE,
    ) -> ProcessResult:
        """
        Execute a command securely with comprehensive error handling.

        Args:
            command: Command to execute (string or list)
            cwd: Working directory for command execution
            env: Environment variables (will be sanitized)
            timeout: Maximum execution time in seconds
            check: Whether to raise exception on non-zero exit codes
            capture_output: Whether to capture stdout/stderr
            text: Whether to decode output as text
            input_data: Input data to send to the process
            mode: Execution mode for the process

        Returns:
            ProcessResult: Execution results with metadata

        Raises:
            SecurityError: If command violates security policy
            subprocess.TimeoutExpired: If command times out
            subprocess.CalledProcessError: If command fails and check=True
        """
        start_time = time.time()

        # Validate and prepare command
        cmd_list = self._prepare_command(command)
        self._validate_security(cmd_list, cwd)

        # Prepare environment
        safe_env = self._prepare_environment(env)

        # Set timeout
        effective_timeout = timeout or self.security_config.max_execution_time

        # Log command execution
        if self.security_config.log_commands:
            logger.debug(f"Executing command: {' '.join(cmd_list)} (cwd={cwd}, timeout={effective_timeout})")

        try:
            if mode == ProcessExecutionMode.BACKGROUND:
                logger.debug("Running command in background mode")
                return self._run_background(cmd_list, cwd, safe_env)
            elif mode == ProcessExecutionMode.PIPE:
                logger.debug("Running command with real-time output streaming")
                return self._run_with_pipe(cmd_list, cwd, safe_env, effective_timeout)
            else:
                logger.debug("Running command in standard capture mode")
                return self._run_standard(
                    cmd_list, cwd, safe_env, effective_timeout, capture_output, text, input_data, check
                )

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.warning(f"Command timed out after {execution_time:.2f}s: {' '.join(cmd_list)}")
            result = ProcessResult(
                returncode=-1,
                stderr=f"Command timed out after {effective_timeout}s",
                command=cmd_list,
                execution_time=execution_time,
                timed_out=True,
            )
            if check:
                raise
            return result

        except subprocess.CalledProcessError as e:
            execution_time = time.time() - start_time
            logger.error(f"Command failed with exit code {e.returncode}: {' '.join(cmd_list)}")
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
            logger.error(f"Unexpected error executing command: {e}")
            result = ProcessResult(returncode=-1, stderr=str(e), command=cmd_list, execution_time=execution_time)
            if check:
                raise
            return result

    def _prepare_command(self, command: str | list[str]) -> list[str]:
        """Prepare and validate command format."""
        if isinstance(command, str):
            if self.security_config.allow_shell:
                # Only allow shell commands if explicitly permitted
                return ["/bin/bash", "-c", command]
            else:
                # Parse shell command safely
                try:
                    return shlex.split(command)
                except ValueError as e:
                    raise SecurityError(f"Invalid command format: {e}")
        elif isinstance(command, list):
            return [str(arg) for arg in command]
        else:
            raise SecurityError(f"Invalid command type: {type(command)}")

    def _validate_security(self, cmd_list: list[str], cwd: str | None) -> None:
        """Validate command against security policy."""
        if not cmd_list:
            raise SecurityError("Empty command not allowed")

        command_name = os.path.basename(cmd_list[0])

        # Check blocked commands
        if command_name in self.security_config.blocked_commands:
            raise SecurityError(f"Command '{command_name}' is blocked by security policy")

        # Check allowed commands (if whitelist is configured)
        if self.security_config.allowed_commands and command_name not in self.security_config.allowed_commands:
            raise SecurityError(f"Command '{command_name}' is not in allowed commands list")

        # Validate working directory
        if cwd and self.security_config.allowed_paths:
            cwd_path = Path(cwd).resolve()
            allowed = any(cwd_path.is_relative_to(allowed_path) for allowed_path in self.security_config.allowed_paths)
            if not allowed:
                raise SecurityError(f"Working directory '{cwd}' is not in allowed paths")

        # Check for dangerous patterns
        dangerous_patterns = ["rm -rf /", "sudo", "su -", "> /dev/", "curl |", "wget |"]
        command_str = " ".join(cmd_list)
        for pattern in dangerous_patterns:
            if pattern in command_str.lower():
                logger.warning(f"Potentially dangerous command detected: {pattern}")

    def _prepare_environment(self, env: dict[str, str] | None) -> dict[str, str]:
        """
        Prepare environment variables for subprocess execution.
        Args:
            env: User-provided environment variables
        """
        # Start with a copy of the current environment to preserve all settings
        safe_env = os.environ.copy()

        # Always ensure PYTHONUNBUFFERED is set to prevent output buffering
        safe_env.setdefault("PYTHONUNBUFFERED", "1")

        # Add/override with user-provided environment variables
        if env:
            # Validate environment variables
            for key, value in env.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise SecurityError(f"Invalid environment variable: {key}={value}")
                safe_env[key] = value

        return safe_env

    def _run_standard(
        self,
        cmd_list: list[str],
        cwd: str | None,
        env: dict[str, str],
        timeout: float,
        capture_output: bool,
        text: bool,
        input_data: str | None,
        check: bool,
    ) -> ProcessResult:
        """Execute command with standard subprocess.run."""
        start_time = time.time()

        result = subprocess.run(
            cmd_list,
            cwd=cwd,
            env=env,
            timeout=timeout,
            capture_output=capture_output,
            text=text,
            input=input_data,
            check=check,
        )

        execution_time = time.time() - start_time

        return ProcessResult(
            returncode=result.returncode,
            stdout=result.stdout if capture_output else "",
            stderr=result.stderr if capture_output else "",
            command=cmd_list,
            execution_time=execution_time,
        )

    def _run_background(self, cmd_list: list[str], cwd: str | None, env: dict[str, str]) -> ProcessResult:
        """Execute command in background mode."""
        start_time = time.time()

        process = subprocess.Popen(
            cmd_list, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL
        )

        with self._process_lock:
            self._active_processes[process.pid] = process

        execution_time = time.time() - start_time

        return ProcessResult(
            returncode=0,  # Background process assumed successful at start
            command=cmd_list,
            execution_time=execution_time,
            pid=process.pid,
        )

    def _run_with_pipe(
        self, cmd_list: list[str], cwd: str | None, env: dict[str, str], timeout: float
    ) -> ProcessResult:
        """Execute command with real-time output streaming to console and logging."""
        import select

        start_time = time.time()
        stdout_lines = []
        stderr_lines = []
        last_timeout_check = start_time

        process = subprocess.Popen(
            cmd_list,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for unified output
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        with self._process_lock:
            self._active_processes[process.pid] = process

        # Use poll() instead of select() to support file descriptors > 1023.
        # select() has a hard FD_SETSIZE=1024 limit on Linux and raises
        # "filedescriptor out of range in select()" when fd numbers are exhausted
        # after many tests open Docker connections, log streams, etc.
        # poll() has no such restriction.
        poller = select.poll()
        if process.stdout:
            poller.register(process.stdout, select.POLLIN)

        try:
            # Read output in real-time using poll for non-blocking I/O
            while True:
                # Check if process has finished
                if process.poll() is not None:
                    break

                # Check timeout every iteration (not just after output)
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.debug(f"Timeout detected after {elapsed:.2f}s (limit: {timeout}s)")
                    logger.debug(f"Sending SIGTERM to process {process.pid}")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                        logger.debug(f"Process {process.pid} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        # Process didn't respond to terminate within 2 seconds
                        # Forcefully kill it to prevent background execution
                        logger.warning(f"Process {process.pid} didn't respond to SIGTERM after 2s, sending SIGKILL")
                        process.kill()
                        process.wait()  # Wait for kill to complete
                        logger.debug(f"Process {process.pid} killed forcefully")
                    raise subprocess.TimeoutExpired(cmd_list, timeout)

                # Log timeout check progress every 5 seconds
                current_time = time.time()
                if current_time - last_timeout_check >= 5.0:
                    last_timeout_check = current_time

                # Use poll to check if data is available (100ms timeout in milliseconds)
                # poll() works with any fd value, unlike select() which fails above fd 1023
                if process.stdout:
                    events = poller.poll(100)
                    if events:
                        line = process.stdout.readline()
                        if line:
                            line_stripped = line.rstrip()
                            stdout_lines.append(line_stripped)
                            logger.info(line_stripped)

            # Get any remaining output
            stdout, stderr = process.communicate()
            if stdout:
                remaining_lines = stdout.splitlines()
                for line in remaining_lines:
                    if line:
                        stdout_lines.append(line)
                        logger.info(line)
            if stderr:
                stderr_lines.extend(stderr.splitlines())

        finally:
            if process.stdout:
                try:
                    poller.unregister(process.stdout)
                except Exception:
                    pass
            with self._process_lock:
                self._active_processes.pop(process.pid, None)

        execution_time = time.time() - start_time

        return ProcessResult(
            returncode=process.returncode,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
            command=cmd_list,
            execution_time=execution_time,
            pid=process.pid,
        )

    def terminate_all_processes(self) -> None:
        """Terminate all active background processes."""
        with self._process_lock:
            for pid, process in list(self._active_processes.items()):
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
                finally:
                    self._active_processes.pop(pid, None)

    def get_active_processes(self) -> list[int]:
        """Get list of active process PIDs."""
        with self._process_lock:
            return list(self._active_processes.keys())

    def start_process(
        self,
        command: str | list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stdout=None,
        stderr=subprocess.PIPE,
    ) -> "ProcessHandle":
        """Start a process in a new session and return a :class:`ProcessHandle`.

        Unlike :meth:`run`, this method returns **immediately** — the process
        runs concurrently.  The caller is responsible for lifecycle management
        via the returned handle.

        The process is started with ``start_new_session=True`` so it receives
        its own process group ID.  :meth:`ProcessHandle.terminate` and
        :meth:`ProcessHandle.kill` therefore affect all child processes in that
        group, enabling clean shutdown of multi-worker tools like stress-ng.

        Args:
            command: Command and arguments.  Never pass user-controlled strings
                     directly; validate inputs at system boundaries first.
            cwd:     Working directory for the process.
            env:     Extra environment variables merged with ``os.environ``.
            stdout:  Stdout destination.  ``None`` (default) redirects to
                     ``DEVNULL``; pass an open file object to capture output;
                     pass ``subprocess.PIPE`` to read via the handle.
            stderr:  Stderr destination.  Defaults to ``subprocess.PIPE`` so
                     error output can be retrieved via
                     :meth:`ProcessHandle.read_stderr`.

        Returns:
            :class:`ProcessHandle` for the running process.
        """
        cmd_list = self._prepare_command(command)
        self._validate_security(cmd_list, cwd)
        safe_env = self._prepare_environment(env)

        effective_stdout = subprocess.DEVNULL if stdout is None else stdout

        proc = subprocess.Popen(  # nosec B603 # validated above
            cmd_list,
            cwd=cwd,
            env=safe_env,
            stdout=effective_stdout,
            stderr=stderr,
            start_new_session=True,
        )

        with self._process_lock:
            self._active_processes[proc.pid] = proc

        if self.security_config.log_commands:
            logger.debug("Started process (PID=%s): %s", proc.pid, " ".join(cmd_list))

        return ProcessHandle(proc, cmd_list)


class SecurityError(Exception):
    """Exception raised for security policy violations."""


class ProcessHandle:
    """Handle for a long-running process started via :func:`start_process`.

    Processes are launched in a new session (``start_new_session=True``),
    creating their own process group.  :meth:`terminate` and :meth:`kill`
    therefore send the signal to the **entire process group**, ensuring all
    child workers (e.g. stress-ng workers or cyclictest threads) are cleaned
    up together.

    Usage example::

        handle = start_process(["cyclictest", "-a2-3", "-t2", "-p99", "-D5s"],
                               stderr=subprocess.PIPE)
        try:
            handle.wait(timeout=30)
        finally:
            handle.terminate()
        stderr = handle.read_stderr()
    """

    def __init__(self, proc: subprocess.Popen, command: list[str]) -> None:
        self._proc = proc
        self.command = command
        self.pid: int = proc.pid

    def poll(self) -> int | None:
        """Return the exit code if the process has finished, else ``None``."""
        return self._proc.poll()

    @property
    def returncode(self) -> int | None:
        """Exit code of the process, or ``None`` if still running."""
        return self._proc.returncode

    def wait(self, timeout: float | None = None) -> int:
        """Block until the process exits.

        Args:
            timeout: Maximum seconds to wait.  Returns ``-1`` on timeout
                     (process is still running — call :meth:`terminate` to
                     stop it).

        Returns:
            The process exit code, or ``-1`` if the timeout elapsed.
        """
        try:
            return self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return -1

    def terminate(self, wait_secs: float = 10.0) -> None:
        """Send ``SIGTERM`` to the process group and wait for exit.

        Falls back to ``SIGKILL`` if the process does not exit within
        *wait_secs* seconds.
        """
        if self._proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return
        try:
            self._proc.wait(timeout=wait_secs)
        except subprocess.TimeoutExpired:
            self.kill()

    def kill(self) -> None:
        """Send ``SIGKILL`` to the process group immediately."""
        if self._proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            self._proc.wait()
        except (ProcessLookupError, OSError):
            pass

    def read_stderr(self) -> str:
        """Read all remaining stderr output as a decoded string.

        Safe to call after the process has exited.  Returns an empty string
        if stderr was not captured (e.g. redirected to ``DEVNULL``).
        """
        if self._proc.stderr is None:
            return ""
        try:
            data = self._proc.stderr.read()
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="replace")
            return str(data) if data else ""
        except Exception:
            return ""

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        """Wait for process completion and return ``(stdout, stderr)`` strings.

        Sends ``SIGKILL`` and drains output on timeout so the caller is never
        left with a zombie process.
        """
        try:
            out, err = self._proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            out, err = self._proc.communicate()

        def _decode(b) -> str:
            if isinstance(b, bytes):
                return b.decode("utf-8", errors="replace")
            return str(b) if b else ""

        return _decode(out), _decode(err)


# Global secure executor instance
_global_executor = None


def get_executor(security_config: ProcessSecurityConfig | None = None) -> SecureProcessExecutor:
    """
    Get the global secure process executor instance.

    Args:
        security_config: Security configuration (only used for first call)

    Returns:
        SecureProcessExecutor: Global executor instance
    """
    global _global_executor
    if _global_executor is None:
        _global_executor = SecureProcessExecutor(security_config)
    return _global_executor


# Convenience functions for common use cases


def run_command(
    command: str | list[str],
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    check: bool = False,
    capture_output: bool = True,
    stream_output: bool = False,
) -> ProcessResult:
    """
    Execute a command securely with default settings.

    This is the primary function that should be used throughout the application
    for subprocess execution instead of direct subprocess calls.

    Args:
        command: Command to execute
        cwd: Working directory
        env: Environment variables
        timeout: Execution timeout
        check: Raise exception on failure
        capture_output: Capture stdout/stderr
        stream_output: Stream output in real-time to console (implies capture_output=True)

    Returns:
        ProcessResult: Execution results
    """
    executor = get_executor()
    mode = ProcessExecutionMode.PIPE if stream_output else ProcessExecutionMode.CAPTURE
    return executor.run(
        command=command,
        cwd=cwd,
        env=env,
        timeout=timeout,
        check=check,
        capture_output=capture_output,
        mode=mode,
    )


def run_command_with_output(
    command: str | list[str],
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> tuple[int, str, str]:
    """
    Execute a command and return exit code, stdout, and stderr.

    This function provides backward compatibility with existing code patterns.

    Args:
        command: Command to execute
        cwd: Working directory
        env: Environment variables
        timeout: Execution timeout

    Returns:
        Tuple[int, str, str]: (returncode, stdout, stderr)
    """
    result = run_command(command, cwd, env, timeout, capture_output=True)
    return result.returncode, result.stdout, result.stderr


def check_command_available(command: str, timeout: float = 5.0) -> bool:
    """
    Check if a command is available on the system.

    Args:
        command: Command name to check
        timeout: Timeout for the check

    Returns:
        bool: True if command is available
    """
    # shutil.which() is a reliable PATH lookup with no exit-code ambiguity.
    # Running the command with --version is not reliable — many tools (e.g.
    # memtester) exit non-zero for --version even when correctly installed.
    if shutil.which(command) is not None:
        return True
    # Fallback: subprocess which covers edge cases where Python's PATH differs
    # from the shell's runtime PATH (e.g. /usr/sbin not in os.environ PATH).
    try:
        result = run_command(["which", command], timeout=timeout, capture_output=True)
        return result.success
    except Exception:
        return False


def run_git_command(cmd: list[str], cwd: str | None = None, check: bool = True, timeout: float = 30.0) -> ProcessResult:
    """
    Execute a git command securely.

    Args:
        cmd: Git command and arguments (without 'git' prefix)
        cwd: Working directory
        check: Raise exception on failure
        timeout: Execution timeout

    Returns:
        ProcessResult: Execution results
    """
    git_cmd = ["git"] + cmd
    return run_command(git_cmd, cwd=cwd, check=check, timeout=timeout)


def configure_security(
    allowed_commands: list[str] | None = None,
    blocked_commands: list[str] | None = None,
    max_execution_time: float = 300.0,
    allow_shell: bool = False,
) -> None:
    """
    Configure global security settings for subprocess execution.

    Args:
        allowed_commands: Whitelist of allowed commands
        blocked_commands: Blacklist of blocked commands
        max_execution_time: Maximum execution time
        allow_shell: Whether to allow shell commands
    """
    global _global_executor
    security_config = ProcessSecurityConfig(
        allowed_commands=allowed_commands,
        blocked_commands=blocked_commands,
        max_execution_time=max_execution_time,
        allow_shell=allow_shell,
    )
    _global_executor = SecureProcessExecutor(security_config)


# Cleanup function for graceful shutdown
def cleanup_processes() -> None:
    """Clean up all active processes."""
    global _global_executor
    if _global_executor:
        _global_executor.terminate_all_processes()


def start_process(
    command: str | list[str],
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    stdout=None,
    stderr=subprocess.PIPE,
) -> "ProcessHandle":
    """Start a long-running process and return a handle for lifecycle management.

    This is the preferred way to launch background processes from test code
    instead of calling ``subprocess.Popen`` directly.  The process runs in a
    new session (own process group) so :meth:`~ProcessHandle.terminate` and
    :meth:`~ProcessHandle.kill` clean up all child workers correctly.

    Unlike :func:`run_command` (which blocks), this function returns
    immediately.  Call :meth:`ProcessHandle.wait` to block for completion, or
    :meth:`ProcessHandle.terminate` to stop the process early.

    Args:
        command: Command and arguments list.  Do **not** pass user-controlled
                 strings without validation; sanitize at system boundaries.
        cwd:     Optional working directory.
        env:     Extra environment variables merged with ``os.environ``.
        stdout:  Stdout destination.  ``None`` (default) → ``DEVNULL``; pass
                 an open file object to capture output to a file; pass
                 ``subprocess.PIPE`` to read via :meth:`ProcessHandle.communicate`.
        stderr:  Stderr destination.  Defaults to ``subprocess.PIPE`` so
                 error output is available via
                 :meth:`ProcessHandle.read_stderr` after completion.

    Returns:
        :class:`ProcessHandle` wrapping the running process.

    Example::

        # Run cyclictest to a file and stress-ng concurrently
        with open(output_path, "w") as out_f:
            cyclic = start_process(cyclic_cmd, stdout=out_f)
        stress = start_process(stress_cmd)
        try:
            cyclic.wait(timeout=3600)
        finally:
            cyclic.terminate()
            stress.terminate()
        stderr = cyclic.read_stderr()
    """
    return get_executor().start_process(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
