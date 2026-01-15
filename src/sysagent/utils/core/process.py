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
import subprocess  # nosec B404 # For secure process execution API
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
        command: List[str] = None,
        execution_time: float = 0.0,
        pid: Optional[int] = None,
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
        allowed_commands: Optional[List[str]] = None,
        blocked_commands: Optional[List[str]] = None,
        allowed_paths: Optional[List[str]] = None,
        max_execution_time: float = 300.0,  # 5 minutes default
        max_memory_mb: Optional[int] = None,
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

    def __init__(self, security_config: Optional[ProcessSecurityConfig] = None):
        """
        Initialize the secure process executor.

        Args:
            security_config: Security configuration for subprocess execution
        """
        self.security_config = security_config or ProcessSecurityConfig()
        self._active_processes: Dict[int, subprocess.Popen] = {}
        self._process_lock = threading.Lock()

    def run(
        self,
        command: Union[str, List[str]],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
        input_data: Optional[str] = None,
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

    def _prepare_command(self, command: Union[str, List[str]]) -> List[str]:
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

    def _validate_security(self, cmd_list: List[str], cwd: Optional[str]) -> None:
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

    def _prepare_environment(self, env: Optional[Dict[str, str]]) -> Dict[str, str]:
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
        cmd_list: List[str],
        cwd: Optional[str],
        env: Dict[str, str],
        timeout: float,
        capture_output: bool,
        text: bool,
        input_data: Optional[str],
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

    def _run_background(self, cmd_list: List[str], cwd: Optional[str], env: Dict[str, str]) -> ProcessResult:
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
        self, cmd_list: List[str], cwd: Optional[str], env: Dict[str, str], timeout: float
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

        try:
            # Read output in real-time using select for non-blocking I/O
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

                # Use select to check if data is available (non-blocking with 0.1s timeout)
                # This allows us to check the timeout more frequently
                if process.stdout:
                    ready, _, _ = select.select([process.stdout], [], [], 0.1)
                    if ready:
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

    def get_active_processes(self) -> List[int]:
        """Get list of active process PIDs."""
        with self._process_lock:
            return list(self._active_processes.keys())


class SecurityError(Exception):
    """Exception raised for security policy violations."""

    pass


# Global secure executor instance
_global_executor = None


def get_executor(security_config: Optional[ProcessSecurityConfig] = None) -> SecureProcessExecutor:
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
    command: Union[str, List[str]],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
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
    command: Union[str, List[str]],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> Tuple[int, str, str]:
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
    try:
        result = run_command([command, "--version"], timeout=timeout, capture_output=True)
        return result.success
    except Exception:
        try:
            result = run_command(["which", command], timeout=timeout, capture_output=True)
            return result.success
        except Exception:
            return False


def run_git_command(
    cmd: List[str], cwd: Optional[str] = None, check: bool = True, timeout: float = 30.0
) -> ProcessResult:
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
    allowed_commands: Optional[List[str]] = None,
    blocked_commands: Optional[List[str]] = None,
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
