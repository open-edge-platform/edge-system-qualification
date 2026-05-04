# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Common service interface contracts for multi-service suites."""

from dataclasses import dataclass, field
from typing import Dict, Protocol


@dataclass
class ServiceSpec:
    """Canonical service specification shared across timeseries suites."""

    name: str
    kind: str
    compose_service_name: str
    default_group: str = "core"
    env: Dict[str, str] = field(default_factory=dict)


class ServiceInterface(Protocol):
    """Behavior contract for service adapters (future runtime-specific adapters)."""

    def start(self) -> None:
        """Start the service and return when startup request is issued."""

    def stop(self) -> None:
        """Stop the service and clean up temporary resources."""

    def healthcheck(self) -> bool:
        """Return True when the service is healthy and ready to use."""
