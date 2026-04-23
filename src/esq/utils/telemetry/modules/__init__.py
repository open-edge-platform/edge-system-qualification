# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
ESQ-specific telemetry modules package.

Provides all ESQ telemetry module classes and registers them into the sysagent
telemetry registry.

The preferred registration path is the function-based API:
    ``get_telemetry_modules()`` returns a ``{name: class}`` dict and the
    collector calls it explicitly after loading the entry point module.  This
    is immune to ``sys.modules`` caching of partially-initialised modules.

As a fallback, registration also fires at module import time (matching the
sysagent core-module pattern), so direct ``import esq.utils.telemetry.modules``
still works.

Modules registered:
    package_power — Intel® RAPL-based CPU package power measurement.
    gpu_temp      — Intel® GPU package and VRAM temperatures (xe / i915).
    gpu_freq      — Intel® GPU actual operating frequency per GT (xe / i915).
    gpu_power     — Intel® GPU power in Watts via hwmon energy counters (xe / i915).
    gpu_usage     — Intel® GPU utilization: per-engine PMU percentages for xe
                    (Arc) GPUs; GT-level RC6/C6 residency for i915 GPUs.
    npu_usage     — Intel® NPU busy utilisation (%) and device memory (MB) via
                    the intel_vpu driver sysfs interface.
    npu_freq      — Intel® NPU current operating frequency (MHz) via the
                    intel_vpu driver sysfs interface.
"""

from typing import Dict, Type

from sysagent.utils.telemetry.base import BaseTelemetryModule
from sysagent.utils.telemetry.registry import register as _register


def get_telemetry_modules() -> Dict[str, Type[BaseTelemetryModule]]:
    """
    Return all ESQ telemetry module classes as ``{name: class}``.

    Uses lazy imports inside the function so an import failure in one module
    file does not prevent the remaining modules from loading.  The sysagent
    collector calls this function explicitly after loading the entry point
    module, which is more robust than relying on ``importlib.import_module``
    side-effects that can be skipped when the module is already cached in
    ``sys.modules``.
    """
    modules: Dict[str, Type[BaseTelemetryModule]] = {}

    try:
        from esq.utils.telemetry.modules.power import PackagePowerModule

        modules["package_power"] = PackagePowerModule
    except Exception:
        pass

    try:
        from esq.utils.telemetry.modules.gpu_temp import GpuTempModule

        modules["gpu_temp"] = GpuTempModule
    except Exception:
        pass

    try:
        from esq.utils.telemetry.modules.gpu_freq import GpuFreqModule

        modules["gpu_freq"] = GpuFreqModule
    except Exception:
        pass

    try:
        from esq.utils.telemetry.modules.gpu_power import GpuPowerModule

        modules["gpu_power"] = GpuPowerModule
    except Exception:
        pass

    try:
        from esq.utils.telemetry.modules.gpu_usage import GpuUsageModule

        modules["gpu_usage"] = GpuUsageModule
    except Exception:
        pass

    try:
        from esq.utils.telemetry.modules.npu_usage import NpuUsageModule

        modules["npu_usage"] = NpuUsageModule
    except Exception:
        pass

    try:
        from esq.utils.telemetry.modules.npu_freq import NpuFreqModule

        modules["npu_freq"] = NpuFreqModule
    except Exception:
        pass

    return modules


# Also register at module import time so that a plain
# ``import esq.utils.telemetry.modules`` still populates the registry
# (matches the sysagent core-module pattern).
for _name, _cls in get_telemetry_modules().items():
    _register(_name, _cls)

__all__ = ["get_telemetry_modules"]
