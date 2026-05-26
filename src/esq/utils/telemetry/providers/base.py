# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Provider interface for framework-level telemetry backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseFrameworkTelemetryProvider(ABC):
    """Minimal provider interface used by platform_telemetry."""

    name: str = "base"

    def __init__(self, options: Dict[str, Any]) -> None:
        self.options = options

    @abstractmethod
    def health(self) -> bool:
        """Return provider health state."""

    @abstractmethod
    def collect_sample(self) -> Dict[str, Any]:
        """Return one normalized sample as metric_name -> numeric_value."""

    def get_status(self) -> Dict[str, Any]:
        """Return provider status details for telemetry summary metadata."""
        return {"name": self.name, "health": self.health()}

    def close(self) -> None:
        """Release any external resources (subprocesses, compose stacks, sockets).

        Default no-op.  Providers that spawn daemons or start docker compose
        stacks MUST override this and reap whatever they started, but only
        what they started themselves — never tear down resources that were
        already running before the provider was constructed.
        """
        return
