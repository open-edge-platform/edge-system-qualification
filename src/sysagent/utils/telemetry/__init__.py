# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Modular telemetry framework for sysagent.

Provides background telemetry collection driven by profile YAML configuration.
Supports extensible module registration so downstream packages (e.g., esq)
can contribute additional telemetry modules without modifying this package.

Core modules registered here (sysagent):
- cpu_freq     — CPU frequency (MHz) via psutil
- cpu_usage    — CPU utilisation (%) via psutil
- cpu_temp     — CPU package and peak core temperature (°C) via coretemp driver
- memory_usage — RAM usage (GiB / %) via psutil

ESQ package registers additional modules via the ``sysagent_telemetry`` entry
point (``esq.utils.telemetry.modules``).  The collector calls
``get_telemetry_modules()`` from that module to register them explicitly:
- package_power — Intel® CPU package power via RAPL sysfs
- gpu_temp      — Intel® GPU package and VRAM temperatures (xe / i915)
- gpu_freq      — Intel® GPU operating frequency per GT (xe / i915)
- gpu_power     — Intel® GPU power in Watts (xe / i915)
- gpu_usage     — Intel® GPU engine utilization % (xe Arc) / RC6 residency (i915)
- npu_usage     — Intel® NPU busy utilization % and device memory (intel_vpu)
- npu_freq      — Intel® NPU operating frequency (intel_vpu)

Usage example in profile YAML::

    telemetry:
      enabled: true
      interval: 10
      modules:
        - name: cpu_freq
          enabled: true
          thresholds:
            current_mhz:
              warning: 4500
        - name: cpu_usage
          enabled: true
        - name: memory_usage
          enabled: true
        - name: package_power   # registered by esq package
          enabled: true
        - name: gpu_usage       # registered by esq package
          enabled: true
"""

# Register core modules on import (idempotent)
from sysagent.utils.telemetry import modules as _core_modules  # noqa: F401
from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetryConfig, TelemetrySample
from sysagent.utils.telemetry.collector import TelemetryCollector
from sysagent.utils.telemetry.registry import get as get_telemetry_module
from sysagent.utils.telemetry.registry import list_modules
from sysagent.utils.telemetry.registry import register as register_telemetry_module

__all__ = [
    "BaseTelemetryModule",
    "TelemetryConfig",
    "TelemetrySample",
    "TelemetryCollector",
    "register_telemetry_module",
    "get_telemetry_module",
    "list_modules",
]
