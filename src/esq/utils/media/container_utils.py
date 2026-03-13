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
    2. A Docker-compatible X authority file at /tmp/.docker.xauth

    The xauth file is generated via:
        xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -

    The wildcard hostname format (ffff prefix via sed) allows the cookie to be
    used by any hostname inside the container without needing the exact host
    machine name.  Critically, the file is placed in /tmp (world-accessible),
    so non-root container users (e.g. dlstreamer) can read it — unlike
    ~/.Xauthority which is only readable by the owning user.

    Args:
        host_display: The X11 display string (e.g., ":0", ":1")
        logger: Optional logger instance.

    Returns:
        dict: Volume mappings in Docker SDK format
            {"/path/on/host": {"bind": "/path/in/container", "mode": "rw/ro"}}

    Example:
        >>> volumes = get_x11_volumes(":1")
        >>> # Returns {"/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "rw"},
        >>> #          "/tmp/.docker.xauth": {"bind": "/tmp/.docker.xauth", "mode": "rw"}}
    """
    import os
    import subprocess  # nosec B404 # For xauth cookie generation pipeline
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

    # Generate a Docker-compatible xauth file in /tmp so non-root container
    # users can read it (avoids the /root/.Xauthority accessibility problem).
    xauth_file = "/tmp/.docker.xauth"
    try:
        # Remove stale file from a previous run
        if os.path.exists(xauth_file):
            os.remove(xauth_file)
        Path(xauth_file).touch(mode=0o660)

        # Sanitize display value — allow only alphanumeric, colon, dot, dash
        sanitized_display = "".join(c for c in host_display if c.isalnum() or c in ":.-")
        # Sanitize xauth file path — allow only safe filesystem characters
        sanitized_xauth = "".join(c for c in xauth_file if c.isalnum() or c in "/._-")

        # Chain: xauth nlist | sed (wildcard hostname) | xauth nmerge
        p1 = p2 = p3 = None
        try:
            p1 = subprocess.Popen(
                ["xauth", "nlist", sanitized_display],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            p2 = subprocess.Popen(
                ["sed", "-e", "s/^..../ffff/"],
                stdin=p1.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if p1.stdout:
                p1.stdout.close()
            p3 = subprocess.Popen(
                ["xauth", "-f", sanitized_xauth, "nmerge", "-"],
                stdin=p2.stdout,
                stderr=subprocess.PIPE,
                text=True,
            )
            if p2.stdout:
                p2.stdout.close()

            _, stderr = p3.communicate(timeout=10)
            if p3.returncode != 0:
                log_warning(f"xauth nmerge returned {p3.returncode}: {stderr.strip()}")
            else:
                file_size = os.path.getsize(xauth_file)
                if file_size == 0:
                    log_warning("xauth file is empty — X11 display authorization may fail inside container")
                else:
                    log_debug(f"Docker xauth file created ({file_size} bytes): {xauth_file}")
        except subprocess.TimeoutExpired:
            log_warning("xauth pipeline timed out after 10 s")
            for proc in [p3, p2, p1]:
                try:
                    if proc is not None and proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=1)
                except Exception:
                    pass
        except FileNotFoundError:
            log_warning("xauth binary not found — X11 auth cookie will not be generated")
        except Exception as xauth_err:
            log_warning(f"xauth pipeline failed: {xauth_err}")

        os.chmod(xauth_file, 0o660)

    except Exception as e:
        log_warning(f"Failed to create Docker X authority file: {e}")

    if os.path.exists(xauth_file):
        # Mount at the same /tmp path so the XAUTHORITY env var works for any user
        volumes[xauth_file] = {"bind": xauth_file, "mode": "rw"}
        log_debug(f"Mounted Docker xauth file: {xauth_file}")

    return volumes


def get_x11_environment(host_display: str, display_enabled: bool = True) -> dict:
    """
    Get environment variables for X11 display access in containers.

    XAUTHORITY is set to /tmp/.docker.xauth — the Docker-compatible cookie file
    generated by get_x11_volumes().  Unlike /root/.Xauthority this path is
    readable by any container user (e.g. dlstreamer running as non-root).

    Args:
        host_display: The X11 display string (e.g., ":0", ":1")
        display_enabled: Whether display output is enabled

    Returns:
        dict: Environment variables for container
            {"DISPLAY": ":1", "XAUTHORITY": "/tmp/.docker.xauth"}
            Returns empty dict if display_enabled is False

    Example:
        >>> env = get_x11_environment(":1", display_enabled=True)
        >>> # Returns {"DISPLAY": ":1", "XAUTHORITY": "/tmp/.docker.xauth"}
    """
    if not display_enabled:
        return {}

    return {
        "DISPLAY": host_display,
        "XAUTHORITY": "/tmp/.docker.xauth",
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
