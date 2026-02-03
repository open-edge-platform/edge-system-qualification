# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Minimal subprocess execution utilities for container environments.

This module provides a lightweight alternative to the full sysagent.utils.core.process
module for use in Docker containers. It contains only the essential functionality
needed for media benchmarking without the full framework overhead.

For development/testing outside containers, use the full sysagent.utils.core.process module.
"""

import subprocess  # nosec B404 # For container environment command execution
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
    import os
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


# =============================================================================
# X11 Display Detection Utilities
# =============================================================================
# These functions are used by both media and proxy test suites for handling
# display output with X11 (xvimagesink/compositor) in Docker containers.


def detect_display_settings(logger=None) -> tuple:
    """
    Auto-detect X11 display settings from the system.

    This function checks for a working X11 display by:
    1. Checking the DISPLAY environment variable
    2. Detecting X11 sockets in /tmp/.X11-unix/
    3. Determining if display output should be enabled

    This is useful for both media and proxy test suites that use xvimagesink
    or compositor elements which require X11 display access.

    Args:
        logger: Optional logger instance. If None, uses print statements.

    Returns:
        tuple: (display_string, display_available)
            - display_string: The DISPLAY value (e.g., ":1", ":0")
            - display_available: True if a valid X11 display was detected

    Example:
        >>> host_display, display_available = detect_display_settings()
        >>> if display_available:
        ...     environment["DISPLAY"] = host_display
    """
    import os
    from pathlib import Path

    def log_info(msg):
        if logger:
            logger.info(msg)
        else:
            print(f"[INFO] {msg}")

    def log_warning(msg):
        if logger:
            logger.warning(msg)
        else:
            print(f"[WARNING] {msg}")

    # First check environment variable
    host_display = os.environ.get("DISPLAY", "")
    display_available = False

    if host_display:
        # DISPLAY is set, verify the socket exists
        display_num = host_display.lstrip(":").split(".")[0]
        socket_path = Path(f"/tmp/.X11-unix/X{display_num}")
        if socket_path.exists():
            display_available = True
            log_info(f"Using DISPLAY from environment: {host_display}")
        else:
            log_warning(f"DISPLAY={host_display} but socket {socket_path} not found. Display output may fail.")
    else:
        # Try to detect display from X11 sockets
        x11_socket_dir = Path("/tmp/.X11-unix")
        if x11_socket_dir.exists():
            x11_sockets = list(x11_socket_dir.glob("X*"))
            if x11_sockets:
                # Get valid socket numbers, preferring lower numbers
                socket_nums = []
                for s in x11_sockets:
                    try:
                        num = int(s.name[1:])
                        socket_nums.append(num)
                    except ValueError:
                        continue

                if socket_nums:
                    detected_num = min(socket_nums)
                    host_display = f":{detected_num}"
                    display_available = True
                    log_info(f"DISPLAY not set, auto-detected from X11 socket: {host_display}")

    if not host_display:
        host_display = ":0"
        log_warning(
            "No X11 display detected. Display output will be disabled. "
            "Set DISPLAY environment variable or connect a monitor."
        )

    return host_display, display_available


def get_x11_volumes(host_display: str, logger=None) -> dict:
    """
    Get Docker volume mappings for X11 display access.

    Creates volume mappings for:
    1. X11 socket directory (/tmp/.X11-unix)
    2. .Xauthority file for secure X11 access

    Args:
        host_display: The X11 display string (e.g., ":0", ":1")
        logger: Optional logger instance.

    Returns:
        dict: Volume mappings in Docker SDK format
            {"/path/on/host": {"bind": "/path/in/container", "mode": "ro/rw"}}

    Example:
        >>> volumes = get_x11_volumes(":1")
        >>> # Returns {"/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "rw"}, ...}
    """
    import os
    from pathlib import Path

    def log_debug(msg):
        if logger:
            logger.debug(msg)

    def log_warning(msg):
        if logger:
            logger.warning(msg)
        else:
            print(f"[WARNING] {msg}")

    volumes = {}

    # Mount X11 socket directory
    x11_socket_dir = Path("/tmp/.X11-unix")
    if x11_socket_dir.exists():
        volumes[str(x11_socket_dir)] = {"bind": "/tmp/.X11-unix", "mode": "rw"}
        log_debug(f"Mounted X11 socket for display output: {x11_socket_dir}")

    # Mount .Xauthority for secure X11 access (avoids need for 'xhost +')
    xauthority_path = os.environ.get("XAUTHORITY", os.path.expanduser("~/.Xauthority"))
    if Path(xauthority_path).exists():
        volumes[xauthority_path] = {"bind": "/root/.Xauthority", "mode": "ro"}
        log_debug(f"Mounted .Xauthority for secure X11 access: {xauthority_path}")
    else:
        log_warning(
            f".Xauthority not found at {xauthority_path}. X11 access may require 'xhost +local:root' (less secure)."
        )

    return volumes


def get_x11_environment(host_display: str, display_enabled: bool = True) -> dict:
    """
    Get environment variables for X11 display access in containers.

    Args:
        host_display: The X11 display string (e.g., ":0", ":1")
        display_enabled: Whether display output is enabled

    Returns:
        dict: Environment variables for container
            {"DISPLAY": ":1", "XAUTHORITY": "/root/.Xauthority"}
            Returns empty dict if display_enabled is False

    Example:
        >>> env = get_x11_environment(":1", display_enabled=True)
        >>> # Returns {"DISPLAY": ":1", "XAUTHORITY": "/root/.Xauthority"}
    """
    if not display_enabled:
        return {}

    return {
        "DISPLAY": host_display,
        "XAUTHORITY": "/root/.Xauthority",
    }


def determine_display_output(config_display_output, display_available: bool, logger=None) -> bool:
    """
    Determine if display output should be enabled based on config and availability.

    This implements auto-fallback logic:
    - If display_output=0: always use fakesink (no display)
    - If display_output=1: try to use display, but auto-fallback to fakesink if unavailable
    - If not set (None): auto-detect based on display availability

    This ensures CI runners and headless environments work even with display_output=1.

    Args:
        config_display_output: Value from config (0, 1, or None)
        display_available: Whether a valid X11 display was detected
        logger: Optional logger instance

    Returns:
        bool: True if display output should be enabled, False otherwise

    Example:
        >>> display_output = determine_display_output(1, display_available=True)
        >>> # Returns True
        >>> display_output = determine_display_output(1, display_available=False)
        >>> # Returns False (auto-fallback to headless mode)
    """

    def log_info(msg):
        if logger:
            logger.info(msg)
        else:
            print(f"[INFO] {msg}")

    def log_warning(msg):
        if logger:
            logger.warning(msg)
        else:
            print(f"[WARNING] {msg}")

    if config_display_output is not None:
        config_value = int(config_display_output)
        if config_value == 0:
            # Explicit disable - always use fakesink
            log_info("display_output=0: Using fakesink (display disabled by config)")
            return False
        elif config_value == 1:
            # Display requested - check if available
            if display_available:
                log_info("display_output=1: Display available, enabling display output")
                return True
            else:
                # AUTO-FALLBACK: display requested but not available
                log_warning(
                    "display_output=1 requested but no display detected. "
                    "Auto-fallback to fakesink (headless mode). "
                    "Set DISPLAY environment variable for display output."
                )
                return False
    else:
        # No explicit config - use display if available
        if display_available:
            log_info("No display_output config, display detected - enabling display output")
            return True
        else:
            log_info("No display_output config, no display detected - using fakesink")
            return False
