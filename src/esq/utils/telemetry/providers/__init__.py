# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Provider registry for framework-level telemetry backends."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseFrameworkTelemetryProvider
from .platform_telemetry.provider import PlatformTelemetryProvider

_PROVIDERS = {
    PlatformTelemetryProvider.name: PlatformTelemetryProvider,
    # Backward-compat alias: profile YAMLs and historical result schemas
    # may still reference the legacy provider id "telecollect_collector".
    # Resolve them to the same in-process PlatformTelemetryProvider.
    "telecollect_collector": PlatformTelemetryProvider,
}


def get_provider(name: str, options: Dict[str, Any]) -> Optional[BaseFrameworkTelemetryProvider]:
    provider_class = _PROVIDERS.get(name)
    if provider_class is None:
        return None
    return provider_class(options)


__all__ = ["BaseFrameworkTelemetryProvider", "get_provider"]
