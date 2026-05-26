# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime installer/starter for the ``qmmd`` (qmassa metrics daemon).

``qmmd`` is the authoritative source of Intel GPU engine utilization,
frequency, power and temperature on Linux.  ``QmassaCollector`` scrapes
its Prometheus endpoint (http://127.0.0.1:9000/metrics) and the
platform_telemetry stack treats qmassa as the **primary** GPU telemetry
source — the in-process collectors only fall back to direct DRM sysfs +
``xpu-smi`` reads when qmmd cannot be brought up.

Per ESQ FW guidance this module mirrors the npu-monitor-tool fetcher:

* All runtime artefacts live under
  ``$CORE_DATA_DIR/data/system/platform_telemetry/qmassa/`` so they are
  co-located with other suite downloads and cleaned up by ``esq clean``.
* The install is performed **once** at the system level (not per suite)
  and cached across runs.
* Failure is non-fatal: when qmmd cannot be installed or started the
  caller transparently falls back to sysfs + xpu-smi.

Public API:

* :func:`ensure_qmmd_running` — guarantee a reachable Prometheus endpoint
  on the configured port, installing and/or launching ``qmmd`` as
  needed.  Returns the URL string on success, ``None`` otherwise.
* :func:`stop_qmmd` — terminate any qmmd subprocess this module started.
"""

from __future__ import annotations

import http.client
import logging
import os
import shutil
import subprocess  # nosec B404 # For runtime cargo install + qmmd daemon lifecycle
import time
import urllib.parse
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Per-process handle to a qmmd subprocess started by ensure_qmmd_running.
_qmmd_proc: Optional[subprocess.Popen] = None

_DEFAULT_PORT = 9000
_RUNTIME_SUBDIR = ("system", "platform_telemetry", "qmassa")


def _resolve_data_root() -> Path:
    """Same resolution rules as :mod:`npu_monitor_fetcher`."""
    override = os.environ.get("ESQ_PLATFORM_TELEMETRY_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()

    core_data_dir = os.environ.get("CORE_DATA_DIR")
    if core_data_dir:
        return Path(core_data_dir).expanduser().resolve() / "data"

    return (Path.cwd() / "esq_data" / "data").resolve()


def _install_root() -> Path:
    return _resolve_data_root().joinpath(*_RUNTIME_SUBDIR)


def _bin_dir() -> Path:
    return _install_root() / "bin"


def _cargo_target_dir() -> Path:
    return _install_root() / "cargo-target"


# ── qmmd discovery ───────────────────────────────────────────────────────────


def _find_existing_qmmd() -> Optional[Path]:
    """Locate a usable ``qmmd`` binary without installing.

    Search order: ESQ-managed install root → user's cargo bin → PATH.
    """
    candidates: list[Path] = [
        _bin_dir() / "qmmd",
        Path.home() / ".cargo" / "bin" / "qmmd",
    ]
    on_path = shutil.which("qmmd")
    if on_path:
        candidates.append(Path(on_path))

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


# ── port reachability ────────────────────────────────────────────────────────


def _qmmd_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/metrics"


def _is_reachable(port: int, timeout_s: float = 1.0) -> bool:
    try:
        url = _qmmd_url(port)
        parsed = urllib.parse.urlparse(url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_s)
        try:
            conn.request("GET", parsed.path or "/")
            return conn.getresponse().status == 200
        finally:
            conn.close()
    except Exception:
        return False


# ── cargo install ────────────────────────────────────────────────────────────


_RUSTUP_URL = "https://sh.rustup.rs"


def _resolve_cargo() -> Optional[Path]:
    """Locate a usable ``cargo`` binary on PATH or under ``~/.cargo/bin``."""
    on_path = shutil.which("cargo")
    if on_path:
        return Path(on_path)
    candidate = Path.home() / ".cargo" / "bin" / "cargo"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    return None


def _install_rustup(timeout_s: float = 600.0) -> Optional[Path]:
    """Bootstrap a per-user rustup toolchain and return the cargo path.

    Uses the official ``sh.rustup.rs`` installer with ``-y --default-toolchain
    stable --profile minimal`` so we get the smallest viable toolchain.
    Installation is unprivileged (writes to ``~/.cargo`` / ``~/.rustup``)
    so no sudo is required.  Failure is non-fatal — caller falls back to
    sysfs telemetry.
    """
    curl = shutil.which("curl")
    if curl is None:
        logger.info(
            "[platform_telemetry] cannot bootstrap rustup: 'curl' not found on PATH"
        )
        return None

    logger.info(
        "[platform_telemetry] cargo not found; bootstrapping rustup (per-user, no sudo) ..."
    )
    try:
        # Pipe rustup-init.sh into sh -s -- -y for non-interactive install.
        dl = subprocess.run(
            [curl, "--proto", "=https", "--tlsv1.2", "-sSf", _RUSTUP_URL],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if dl.returncode != 0 or not dl.stdout:
            logger.info(
                "[platform_telemetry] rustup download failed (rc=%d)", dl.returncode
            )
            return None
        proc = subprocess.run(
            ["sh", "-s", "--", "-y", "--default-toolchain", "stable", "--profile", "minimal", "--no-modify-path"],
            input=dl.stdout,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.info("[platform_telemetry] rustup install failed: %s", exc)
        return None

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace").strip().splitlines()[-3:]
        logger.info(
            "[platform_telemetry] rustup install exited %d: %s",
            proc.returncode,
            " | ".join(tail),
        )
        return None

    cargo = _resolve_cargo()
    if cargo is None:
        logger.info("[platform_telemetry] rustup completed but cargo binary not found")
        return None
    logger.info("[platform_telemetry] rustup ready, cargo at %s", cargo)
    return cargo


def _install_via_cargo(timeout_s: float, install_cargo_if_missing: bool = True) -> Optional[Path]:
    """Run ``cargo install --locked qmmd`` into our managed install root."""
    cargo = _resolve_cargo()
    if cargo is None and install_cargo_if_missing:
        cargo = _install_rustup()
    if cargo is None:
        logger.info(
            "[platform_telemetry] qmmd not installed and 'cargo' is not available; "
            "skipping install (will fall back to sysfs + xpu-smi)."
        )
        return None

    install_root = _install_root()
    target_dir = _cargo_target_dir()
    install_root.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CARGO_INSTALL_ROOT"] = str(install_root)
    env["CARGO_TARGET_DIR"] = str(target_dir)

    logger.info(
        "[platform_telemetry] installing qmmd into %s (this may take several minutes) ...",
        install_root,
    )
    try:
        proc = subprocess.run(
            [str(cargo), "install", "--locked", "--force", "qmmd"],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.info("[platform_telemetry] qmmd cargo install failed: %s", exc)
        return None

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        logger.info(
            "[platform_telemetry] qmmd cargo install exited %d: %s",
            proc.returncode,
            " | ".join(tail),
        )
        return None

    binary = _bin_dir() / "qmmd"
    if not binary.is_file():
        logger.info("[platform_telemetry] qmmd cargo install completed but binary missing at %s", binary)
        return None
    logger.info("[platform_telemetry] qmmd installed at %s", binary)
    return binary


# ── daemon start ─────────────────────────────────────────────────────────────

# Port this module spawned qmmd on (set by _start_daemon, read by stop_qmmd
# so we can issue a precise pkill match for the qmmd command line).
_qmmd_port: Optional[int] = None


def _start_daemon(binary: Path, port: int, log_dir: Path, ready_timeout_s: float) -> bool:
    """Spawn ``qmmd`` as a background subprocess and wait for readiness."""
    global _qmmd_proc, _qmmd_port

    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "qmmd.stdout.log"
    stderr_log = log_dir / "qmmd.stderr.log"

    # qmmd needs raw access to /sys/kernel/debug and DRM perf. Those
    # permissions are granted up front by the ESQ pre-requisites script
    # (system-setup.sh — perf_event_paranoid, debugfs ACL, render/video
    # group membership). We never escalate in-process; if qmmd cannot
    # start unprivileged the caller falls back to sysfs.
    cmd: list[str] = [str(binary), "--port", str(port)]

    try:
        # New session so we can kill the whole process group on stop_qmmd().
        _qmmd_proc = subprocess.Popen(
            cmd,
            stdout=stdout_log.open("ab"),
            stderr=stderr_log.open("ab"),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        _qmmd_port = port
    except OSError as exc:
        logger.info("[platform_telemetry] failed to spawn qmmd: %s", exc)
        _qmmd_proc = None
        return False

    deadline = time.monotonic() + ready_timeout_s
    while time.monotonic() < deadline:
        if _qmmd_proc.poll() is not None:
            logger.info(
                "[platform_telemetry] qmmd exited prematurely (rc=%s); see %s",
                _qmmd_proc.returncode,
                stderr_log,
            )
            _qmmd_proc = None
            return False
        if _is_reachable(port):
            logger.info(
                "[platform_telemetry] qmmd daemon is up on %s (pid=%d)",
                _qmmd_url(port),
                _qmmd_proc.pid,
            )
            # Hard-kill / Ctrl+C safety net: even if the provider's close()
            # path is bypassed (uncaught exception, SIGINT, SIGTERM) the
            # cleanup orchestrator still invokes stop_qmmd() at exit.
            try:
                from esq.utils.telemetry.providers.platform_telemetry.cleanup import register

                register(stop_qmmd)
            except Exception:
                pass
            return True
        time.sleep(0.25)

    logger.info(
        "[platform_telemetry] qmmd did not become reachable on %s within %.1fs",
        _qmmd_url(port),
        ready_timeout_s,
    )
    stop_qmmd()
    return False


def stop_qmmd() -> None:
    """Terminate any qmmd subprocess started by this module.

    qmmd is spawned unprivileged by ``_start_daemon`` (the ESQ
    pre-requisites script grants the runtime user the access it needs),
    so an in-process ``pkill -f`` matched on the exact ``qmmd --port
    <port>`` command line is sufficient. No sudo escalation occurs here.
    """
    global _qmmd_proc, _qmmd_port
    proc = _qmmd_proc
    port = _qmmd_port
    _qmmd_proc = None
    _qmmd_port = None

    def _run(args: list[str], timeout: float = 3.0) -> None:
        try:
            subprocess.run(
                args, check=False, capture_output=True, timeout=timeout
            )
        except (subprocess.SubprocessError, OSError):
            pass

    # 1. Targeted pkill matching the actual qmmd command line. Safe by
    #    construction — ``--port <port>`` is unique to our spawned
    #    daemon, so this can never affect unrelated processes.
    if port is not None:
        _run(["pkill", "-TERM", "-f", f"qmmd --port {port}"])
        time.sleep(0.5)

        # 2. SIGKILL escalation for anything still hanging around.
        _run(["pkill", "-KILL", "-f", f"qmmd --port {port}"])

    # Final wait on the original Popen so we don't leak a zombie child.
    if proc is not None and proc.poll() is None:
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


# ── public entry point ───────────────────────────────────────────────────────


def ensure_qmmd_running(
    port: int = _DEFAULT_PORT,
    install_timeout_s: float = 600.0,
    start_timeout_s: float = 6.0,
    install_if_missing: bool = True,
    install_cargo_if_missing: bool = True,
) -> Optional[str]:
    """Ensure qmmd is reachable on ``port`` and return its scrape URL.

    Returns ``None`` (and logs at INFO) when neither an existing endpoint
    nor a freshly-launched daemon can be made available; callers are
    expected to fall back to direct sysfs reads in that case.
    """
    # 1. Already running?  Cheapest path; nothing to do.
    if _is_reachable(port):
        logger.debug("[platform_telemetry] qmmd already reachable on %s", _qmmd_url(port))
        return _qmmd_url(port)

    # 2. Find an installed binary.
    binary = _find_existing_qmmd()
    if binary is None and install_if_missing:
        binary = _install_via_cargo(
            timeout_s=install_timeout_s,
            install_cargo_if_missing=install_cargo_if_missing,
        )
    if binary is None:
        return None

    # 3. Launch the daemon and wait for readiness.
    log_dir = _install_root() / "logs"
    if not _start_daemon(binary, port=port, log_dir=log_dir, ready_timeout_s=start_timeout_s):
        return None
    return _qmmd_url(port)
