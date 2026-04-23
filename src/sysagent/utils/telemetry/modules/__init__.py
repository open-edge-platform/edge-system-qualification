# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Core telemetry modules package.

Importing this package automatically registers the built-in sysagent
telemetry modules (cpu_freq, cpu_usage, cpu_temp, memory_usage) into the
global registry.
"""

from sysagent.utils.telemetry import registry
from sysagent.utils.telemetry.modules.cpu_freq import CpuFreqModule
from sysagent.utils.telemetry.modules.cpu_temp import CpuTempModule
from sysagent.utils.telemetry.modules.cpu_usage import CpuUsageModule
from sysagent.utils.telemetry.modules.memory_usage import MemoryUsageModule

# Register core modules
registry.register("cpu_freq", CpuFreqModule)
registry.register("cpu_usage", CpuUsageModule)
registry.register("cpu_temp", CpuTempModule)
registry.register("memory_usage", MemoryUsageModule)

__all__ = ["CpuFreqModule", "CpuUsageModule", "CpuTempModule", "MemoryUsageModule"]
