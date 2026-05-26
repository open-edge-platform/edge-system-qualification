# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight detection of third-party tooling used by platform_telemetry.

The in-process collectors degrade gracefully when their preferred data
source is missing — for example ``QmassaCollector`` falls back to direct
DRM sysfs reads when ``qmmd`` is not running on :9000, and
``NpuCollector`` reports the NPU as "unavailable" when the ``intel_vpu``
driver is not loaded.  This module captures a single up-front snapshot
of the detected capabilities so the snapshot can be logged at module
start and emitted into the telemetry summary metadata for diagnostics.

Detected capabilities (Linux only):

* ``psutil``                – mandatory Python dependency for CpuCollector.
* ``rapl``                  – ``/sys/.../powercap/intel-rapl`` package energy.
* ``coretemp``              – ``/sys/class/hwmon`` package temperature input.
* ``vmstat``                – ``/proc/vmstat`` for CPU bandwidth proxy.
* ``intel_pmt``             – ``/sys/class/intel_pmt`` (NPU PMT registers).
* ``intel_vpu``             – ``/sys/bus/pci/drivers/intel_vpu`` (NPU driver).
* ``drm_card``              – at least one ``/sys/class/drm/card*`` device.
* ``qmmd``                  – HTTP ``GET /metrics`` reachable on the qmmd port.
* ``xpu_smi``               – ``xpu-smi`` binary on PATH.
* ``intel_gpu_top``         – binary present on PATH (informational only).

Note: ``intel_gpu_top`` is **not** consumed by the in-process collectors
today (qmassa already uses the same kernel PMU underneath), but is
detected so future enhancements can opt-in.
"""

from __future__ import annotations

import http.client
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_DEFAULT_QMMD_URL = "http://127.0.0.1:9000/metrics"

# Bound the QMMD_URL surface to a strict allow-list. qmmd is a host-local
# Prometheus exporter; only loopback addresses are valid targets. Any
# user-supplied override that fails this check is rejected and the safe
# default is used instead.
_ALLOWED_QMMD_SCHEMES = frozenset({"http", "https"})
_ALLOWED_QMMD_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_ALLOWED_QMMD_PORT_RANGE = (1, 65535)


# Canonical default components matching ``_DEFAULT_QMMD_URL``.
_DEFAULT_QMMD_COMPONENTS = ("http", "127.0.0.1", 9000, "/metrics")

# Allow-list of canonical hosts. Any user-supplied host that resolves into
# this set is replaced by the corresponding constant so the returned host
# string never carries bytes that originated from the untrusted input.
_CANONICAL_HOST_BY_INPUT = {
    "127.0.0.1": "127.0.0.1",
    "::1": "[::1]",
    "localhost": "localhost",
}
# Allow-list of paths the Prometheus exporter actually serves.
_ALLOWED_QMMD_PATHS = frozenset({"/", "/metrics"})


def _sanitize_qmmd_components(candidate: str) -> tuple[str, str, int, str]:
    """Parse ``candidate`` and return ``(scheme, host, port, path)`` drawn
    only from fixed allow-lists.

    The returned values are guaranteed to be one of the constants in
    ``_ALLOWED_QMMD_SCHEMES`` / ``_CANONICAL_HOST_BY_INPUT`` /
    ``_ALLOWED_QMMD_PATHS`` plus a port inside
    ``_ALLOWED_QMMD_PORT_RANGE``. Any rejected input falls back to
    ``_DEFAULT_QMMD_COMPONENTS``. Together with ``_qmmd_reachable``
    (which takes the already-validated tuple), this ensures the network
    call site never sees data derived from the untrusted ``candidate``
    string.
    """
    try:
        parsed = urllib.parse.urlparse(candidate)
    except (TypeError, ValueError):
        return _DEFAULT_QMMD_COMPONENTS

    if parsed.scheme not in _ALLOWED_QMMD_SCHEMES:
        return _DEFAULT_QMMD_COMPONENTS
    # Re-emit from a literal so the returned string is not derived from
    # the parsed (tainted) attribute.
    scheme = "https" if parsed.scheme == "https" else "http"

    host_key = (parsed.hostname or "").lower()
    canonical_host = _CANONICAL_HOST_BY_INPUT.get(host_key)
    if canonical_host is None:
        return _DEFAULT_QMMD_COMPONENTS

    try:
        port_value = parsed.port if parsed.port is not None else 9000
    except ValueError:
        return _DEFAULT_QMMD_COMPONENTS
    lo, hi = _ALLOWED_QMMD_PORT_RANGE
    if not (lo <= int(port_value) <= hi):
        return _DEFAULT_QMMD_COMPONENTS
    canonical_port = int(port_value)

    raw_path = parsed.path or "/metrics"
    canonical_path = raw_path if raw_path in _ALLOWED_QMMD_PATHS else "/metrics"

    return scheme, canonical_host, canonical_port, canonical_path


def _format_qmmd_url(scheme: str, host: str, port: int, path: str) -> str:
    """Render a display URL from already-validated components."""
    return f"{scheme}://{host}:{port}{path}"


def _sanitize_qmmd_url(candidate: str) -> str:
    """Backward-compatible wrapper that returns the canonical URL string."""
    return _format_qmmd_url(*_sanitize_qmmd_components(candidate))


def _command_exists(name: str) -> bool:
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return True
    return False


def _qmmd_reachable(port: int, timeout_s: float = 1.5) -> bool:
    """Check whether the host-local qmmd Prometheus endpoint responds 200.

    Takes only an integer port, which is re-validated inline against the
    documented TCP range. Every other argument to
    ``http.client.HTTPConnection`` is a module-local string literal
    (``"127.0.0.1"``, ``"/metrics"``) — no value derived from any
    user-supplied URL string reaches the network call.
    """
    try:
        port_int = int(port)
    except (TypeError, ValueError):
        return False
    if not 1 <= port_int <= 65535:
        return False

    # All non-port arguments are local string literals; the only variable
    # is the freshly bounded integer port above.
    conn = http.client.HTTPConnection("127.0.0.1", port_int, timeout=timeout_s)
    try:
        conn.request("GET", "/metrics")
        return conn.getresponse().status == 200
    except Exception:
        return False
    finally:
        conn.close()


def _has_rapl() -> bool:
    for root in (
        "/sys/devices/virtual/powercap/intel-rapl",
        "/sys/devices/virtual/powercap/intel-rapl-mmio",
        "/host_powercap/intel-rapl",
        "/host_powercap/intel-rapl-mmio",
    ):
        if Path(root).exists():
            return True
    return False


def _has_coretemp() -> bool:
    hwmon_root = Path("/sys/class/hwmon")
    if not hwmon_root.exists():
        return False
    for hw in hwmon_root.glob("hwmon*"):
        try:
            name = (hw / "name").read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if name in {"coretemp", "k10temp", "zenpower", "cpu_thermal", "soc_thermal"}:
            return True
    return False


def _has_drm_card() -> bool:
    drm_root = Path("/sys/class/drm")
    if not drm_root.exists():
        return False
    for entry in drm_root.iterdir():
        if entry.name.startswith("card") and entry.name[4:].isdigit():
            return True
    return False


def _has_intel_vpu() -> bool:
    return Path("/sys/bus/pci/drivers/intel_vpu").exists()


def _has_intel_pmt() -> bool:
    return Path("/sys/class/intel_pmt").exists()


def _has_psutil() -> bool:
    try:
        import psutil  # noqa: F401
        return True
    except Exception:
        return False


def detect(qmmd_url: str | None = None) -> Dict[str, Any]:
    """Return a dict snapshot of detected third-party telemetry capabilities."""
    raw_url = qmmd_url or os.environ.get("QMMD_URL", _DEFAULT_QMMD_URL)
    # Validate the candidate URL once for display purposes; the network
    # call below uses only the integer port (re-cast inline inside
    # ``_qmmd_reachable``) and module-local string literals for everything
    # else, so no data derived from ``raw_url`` reaches HTTPConnection.
    scheme, host, port, path = _sanitize_qmmd_components(raw_url)
    canonical_url = _format_qmmd_url(scheme, host, port, path)
    if canonical_url != raw_url:
        logger.warning(
            "[platform_telemetry] rejected non-loopback QMMD_URL=%r; using default",
            raw_url,
        )
    snapshot = {
        "psutil": _has_psutil(),
        "rapl": _has_rapl(),
        "coretemp": _has_coretemp(),
        "vmstat": Path("/proc/vmstat").exists(),
        "drm_card": _has_drm_card(),
        "intel_vpu": _has_intel_vpu(),
        "intel_pmt": _has_intel_pmt(),
        "qmmd": _qmmd_reachable(port),
        "qmmd_url": canonical_url,
        "xpu_smi": _command_exists("xpu-smi"),
        "intel_gpu_top": _command_exists("intel_gpu_top"),
        "git": _command_exists("git"),
    }
    return snapshot


def log_snapshot(snapshot: Dict[str, Any]) -> None:
    """Emit a single-line summary of detected capabilities at INFO level."""
    flags = []
    for key in (
        "psutil",
        "rapl",
        "coretemp",
        "vmstat",
        "drm_card",
        "intel_vpu",
        "intel_pmt",
        "qmmd",
        "xpu_smi",
        "intel_gpu_top",
        "git",
    ):
        flag = "+" if snapshot.get(key) else "-"
        flags.append(f"{flag}{key}")
    logger.info("[platform_telemetry] capabilities: %s", " ".join(flags))
