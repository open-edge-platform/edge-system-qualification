# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Docker lifecycle management for the Metrics Manager container.

Pulls and runs the **official** ``intel/metrics-manager`` image from Docker
Hub (never builds from source) as a process-wide singleton:

* Started lazily on first use by any test needing ``module_group: external``
  telemetry.
* Reused across all subsequent tests in the same ``esq run`` process
  (health-probed, not restarted per test — a fresh container+health-wait per
  test would be far too slow for a function-scoped telemetry fixture).
* Stopped exactly once at process exit (normal completion, Ctrl+C, SIGTERM)
  via the existing process-wide cleanup registry in
  ``esq.utils.telemetry.providers.platform_telemetry.cleanup``, to avoid
  leaving residual containers behind.

Mirrors the install/start/stop shape of the legacy ``qmmd_installer`` module
it replaces, but manages a Docker container instead of a locally-built Rust
binary.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import requests
from sysagent.utils.infrastructure import DockerClient

from .configs import get_metrics_manager_configs

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_docker_client: Optional[DockerClient] = None
_base_url: Optional[str] = None
_cleanup_registered = False

_CONTAINER_PORTS = {
    "9090/tcp": "api",
    "9273/tcp": "telegraf_prometheus",
    "8186/tcp": "telegraf_http",
}


def _resolve_render_gid(docker_cfg: Dict[str, Any]) -> int:
    """Return the host 'render' group GID, falling back to the configured default.

    Mirrors the ``grp.getgrnam("render")`` pattern already used elsewhere in
    ESQ (e.g. ``esq/suites/ai/audio/src/whisper/container.py``) for granting
    GPU device access to containers.
    """
    import grp

    group_name = str(docker_cfg.get("render_group_name", "render"))
    fallback = int(docker_cfg.get("render_group_gid_fallback", 109))
    try:
        return grp.getgrnam(group_name).gr_gid
    except KeyError:
        logger.warning(
            "'%s' group not found on this host; falling back to GID %s for the Metrics Manager container",
            group_name,
            fallback,
        )
        return fallback


def _health_url(cfg: Dict[str, Any]) -> str:
    port = cfg["docker"]["ports"]["api"]
    path = cfg["api"]["health_path"]
    return f"http://127.0.0.1:{port}{path}"


def _host_port_mappings(docker_cfg: Dict[str, Any]) -> Dict[str, tuple[str, int]]:
    """Bind the fixed container ports to configured host ports on loopback only."""
    host_ports = docker_cfg["ports"]
    return {
        container_port: ("127.0.0.1", int(host_ports[config_key]))
        for container_port, config_key in _CONTAINER_PORTS.items()
    }


def _latest_metrics_url(cfg: Dict[str, Any]) -> str:
    port = cfg["docker"]["ports"]["api"]
    path = cfg["api"]["latest_metrics_path"]
    return f"http://127.0.0.1:{port}{path}"


def _capabilities_url(cfg: Dict[str, Any]) -> str:
    port = cfg["docker"]["ports"]["api"]
    path = cfg["api"]["capabilities_path"]
    return f"http://127.0.0.1:{port}{path}?profile=expanded"


def latest_metrics_url() -> str:
    """Return the current ``/api/v1/metrics/latest`` URL (for the provider)."""
    return _latest_metrics_url(get_metrics_manager_configs())


def capabilities_url() -> str:
    """Return the expanded device-capabilities URL."""
    return _capabilities_url(get_metrics_manager_configs())


def _is_healthy(url: str, timeout_s: float = 2.0) -> bool:
    try:
        response = requests.get(url, timeout=timeout_s)
        return response.status_code == 200
    except Exception:
        return False


def is_running() -> bool:
    """Return True if a healthy Metrics Manager endpoint is already known."""
    return _base_url is not None and _is_healthy(_health_url(get_metrics_manager_configs()))


def ensure_metrics_manager_running() -> Optional[str]:
    """Idempotently ensure the Metrics Manager container is running.

    Returns the base URL (e.g. ``http://127.0.0.1:9090``) once the
    container reports healthy, or ``None`` if it could not be pulled,
    started, or become healthy. The external adapter will report the
    unavailable Metrics Manager source and collect only supplemental gap fields.
    """
    global _docker_client, _base_url, _cleanup_registered

    cfg = get_metrics_manager_configs()
    docker_cfg = cfg["docker"]
    health_url = _health_url(cfg)

    with _lock:
        if _base_url is not None and _is_healthy(health_url):
            return _base_url

        try:
            client = _docker_client or DockerClient(timeout=int(docker_cfg.get("pull", {}).get("timeout_s", 600)))
            _docker_client = client
        except Exception as exc:
            logger.warning("Docker is not available; Metrics Manager cannot start: %s", exc)
            return None

        name = str(docker_cfg["container_name"])
        image = f"{docker_cfg['image']}:{docker_cfg['tag']}"

        # Reuse an already-running (e.g. previous-run leftover) container
        # if it is healthy, instead of restarting it.
        try:
            if client.is_container_running(name) and _is_healthy(health_url):
                _base_url = f"http://127.0.0.1:{docker_cfg['ports']['api']}"
                _register_cleanup()
                return _base_url
        except Exception:
            pass

        try:
            client.pull_image(image)
        except Exception as exc:
            logger.warning("Failed to pull Metrics Manager image '%s': %s", image, exc)
            return None

        render_gid = _resolve_render_gid(docker_cfg)
        volumes = {
            str(entry["host"]): {"bind": str(entry["container"]), "mode": str(entry.get("mode", "ro"))}
            for entry in docker_cfg.get("volumes", [])
        }
        ports = _host_port_mappings(docker_cfg)

        # Remove any stale/unhealthy container left from a previous run
        # before starting a fresh one.
        try:
            client.cleanup_container(name)
        except Exception:
            pass

        try:
            client.run_container(
                name=name,
                image=image,
                environment=dict(docker_cfg.get("environment", {})),
                volumes=volumes,
                devices=list(docker_cfg.get("devices", [])),
                group_add=[render_gid],
                privileged=bool(docker_cfg.get("privileged", True)),
                cap_add=list(docker_cfg.get("cap_add") or []),
                pid_mode=docker_cfg.get("pid_mode"),
                security_opt=list(docker_cfg.get("security_opt") or []),
                ports=ports,
                mode="server",
                remove=False,
                attach_logs=False,
            )
        except Exception as exc:
            logger.warning("Failed to start Metrics Manager container '%s': %s", name, exc)
            return None

        health_cfg = docker_cfg.get("health", {})
        timeout_s = float(health_cfg.get("timeout_s", 60))
        poll_s = float(health_cfg.get("poll_interval_s", 1.0))
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if _is_healthy(health_url, timeout_s=2.0):
                _base_url = f"http://127.0.0.1:{docker_cfg['ports']['api']}"
                _register_cleanup()
                logger.info("Metrics Manager container '%s' is healthy at %s", name, _base_url)
                return _base_url
            time.sleep(poll_s)

        logger.warning(
            "Metrics Manager container '%s' did not become healthy within %.0fs; tearing it down",
            name,
            timeout_s,
        )
        try:
            client.cleanup_container(name)
        except Exception:
            pass
        return None


def _register_cleanup() -> None:
    global _cleanup_registered
    if _cleanup_registered:
        return
    try:
        # Shared, generic atexit/SIGINT/SIGTERM-safe cleanup registry; not
        # specific to the legacy qmmd path despite its package location.
        from esq.utils.telemetry.providers.platform_telemetry.cleanup import register

        register(stop_metrics_manager)
        _cleanup_registered = True
    except Exception:
        pass


def stop_metrics_manager() -> None:
    """Stop and remove the Metrics Manager container (idempotent, safe to call
    even if it was never started)."""
    global _base_url

    with _lock:
        if _docker_client is None:
            _base_url = None
            return
        cfg = get_metrics_manager_configs()
        name = str(cfg["docker"]["container_name"])
        try:
            _docker_client.cleanup_container(name)
        except Exception as exc:
            logger.debug("Metrics Manager container cleanup raised: %s", exc)
        _base_url = None
