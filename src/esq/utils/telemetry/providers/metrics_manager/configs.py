# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration loader for the Metrics Manager integration.

Reads ``configs/core/metrics_manager.yml`` using the same
``discover_entrypoint_paths`` lookup used by the rest of ESQ/sysagent so
extension packages can override it. A built-in default is used if the file
is missing, so the provider degrades gracefully rather than raising.
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG_RELATIVE_PATH = os.path.join("core", "metrics_manager.yml")

_DEFAULT_CONFIG: dict[str, Any] = {
    "docker": {
        "image": "intel/metrics-manager",
        "tag": "2026.2.0",
        "container_name": "esq-metrics-manager",
        "ports": {
            "api": 19090,
            "telegraf_prometheus": 19273,
            "telegraf_http": 18186,
        },
        "environment": {
            "HARDWARE_TELEMETRY_ENABLED": "true",
            "METRICS_RETENTION_SECONDS": "300",
            "MAX_METRICS_BATCH_SIZE": "1000",
            "LOG_LEVEL": "INFO",
        },
        "volumes": [
            {"host": "/sys", "container": "/sys", "mode": "ro"},
            {"host": "/run", "container": "/run", "mode": "ro"},
        ],
        "devices": ["/dev/dri:/dev/dri"],
        "render_group_name": "render",
        "render_group_gid_fallback": 109,
        "privileged": False,
        "pid_mode": None,
        "cap_add": ["PERFMON"],
        "security_opt": ["no-new-privileges:true"],
        "health": {
            "endpoint": "/health",
            "timeout_s": 60,
            "poll_interval_s": 1.0,
        },
        "pull": {"timeout_s": 600},
    },
    "api": {
        "latest_metrics_path": "/api/v1/metrics/latest",
        "health_path": "/health",
    },
}

_lock = threading.Lock()
_cached_configs: dict[str, Any] | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_from_disk() -> dict[str, Any]:
    try:
        from sysagent.utils.config.config_loader import discover_entrypoint_paths

        for base_path in discover_entrypoint_paths("configs"):
            config_path = os.path.join(str(base_path), _CONFIG_RELATIVE_PATH)
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as handle:
                        data = yaml.safe_load(handle) or {}
                    logger.debug("Loaded Metrics Manager config from: %s", config_path)
                    return data
                except Exception as exc:
                    logger.warning("Failed to parse Metrics Manager config '%s': %s", config_path, exc)
    except Exception as exc:
        logger.debug("Metrics Manager config discovery unavailable: %s", exc)
    return {}


def get_metrics_manager_configs(refresh: bool = False) -> dict[str, Any]:
    """Return the merged Metrics Manager config (disk config over defaults)."""
    global _cached_configs
    with _lock:
        if _cached_configs is not None and not refresh:
            return copy.deepcopy(_cached_configs)

        on_disk = _load_from_disk()
        merged = _deep_merge(copy.deepcopy(_DEFAULT_CONFIG), on_disk)
        _cached_configs = merged
        return copy.deepcopy(merged)
