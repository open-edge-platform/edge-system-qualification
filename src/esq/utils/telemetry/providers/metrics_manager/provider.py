# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Metrics Manager-backed provider for the ``platform_telemetry`` module.

Primary backend for ``module_group: external`` telemetry: pulls/runs the
official ``intel/metrics-manager`` Docker Hub image (see
:mod:`esq.utils.telemetry.providers.metrics_manager.container`) and reads its
``/api/v1/metrics/latest`` REST endpoint instead of collecting GPU/NPU/CPU
metrics in-process.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from ..base import BaseFrameworkTelemetryProvider
from .container import (
    capabilities_url,
    ensure_metrics_manager_running,
    latest_metrics_url,
)
from .field_map import flatten_latest_metrics

logger = logging.getLogger(__name__)


class MetricsManagerProvider(BaseFrameworkTelemetryProvider):
    """Reads GPU/NPU/CPU telemetry from a Metrics Manager Docker container."""

    name = "metrics_manager"

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        self._base_url: str | None = None
        self._last_error: str = ""
        self._device_inventory: list[dict[str, Any]] | None = None
        self._capabilities_error: str = ""

    def _ensure_started(self) -> bool:
        if self._base_url:
            return True
        self._base_url = ensure_metrics_manager_running()
        return self._base_url is not None

    def health(self) -> bool:
        return self._ensure_started()

    def get_device_inventory(self) -> list[dict[str, Any]]:
        if not self._ensure_started():
            return []
        if self._device_inventory is not None:
            return self._device_inventory or []

        try:
            response = requests.get(capabilities_url(), timeout=3.0)
            response.raise_for_status()
            payload = response.json()
            devices = payload.get("devices") if isinstance(payload, dict) else None
            if not isinstance(devices, list):
                raise TypeError("Metrics Manager capabilities response has no devices list")
            self._device_inventory = [
                device for device in devices if isinstance(device, dict)
            ]
            self._capabilities_error = ""
        except (requests.RequestException, TypeError, ValueError) as exc:
            self._capabilities_error = str(exc)
            logger.warning("[%s] failed to fetch expanded device capabilities: %s", self.name, exc)
            return []
        return self._device_inventory or []

    def collect_sample(self) -> dict[str, float]:
        if not self._ensure_started():
            return {}
        try:
            response = requests.get(latest_metrics_url(), timeout=3.0)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("Metrics Manager latest response is not an object")
        except (requests.RequestException, TypeError, ValueError) as exc:
            self._last_error = str(exc)
            logger.debug("[%s] failed to fetch latest metrics: %s", self.name, exc)
            # A fetch failure doesn't necessarily mean the container died;
            # force a re-probe (cheap health GET) on the next call.
            self._base_url = None
            return {}
        return flatten_latest_metrics(
            payload.get("metrics", {}),
            devices=self.get_device_inventory(),
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self._base_url,
            "last_error": self._last_error,
            "capabilities_error": self._capabilities_error,
            "device_count": len(self._device_inventory or []),
            "health": bool(self._base_url),
        }

    def close(self) -> None:
        # The Metrics Manager container is a process-wide singleton reused
        # across every test in this run; it is intentionally NOT stopped
        # here (a per-test start/stop would be far too slow for a Docker
        # container). It is stopped exactly once at process exit via
        # container.stop_metrics_manager(), registered with the shared
        # cleanup registry on first successful start.
        return
