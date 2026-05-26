# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""In-process collector stack for the framework-level system telemetry path.

This package hosts the third-party-tool-backed collector implementations
(qmassa/qmmd, xpu-smi, intel_pmt, RAPL/vmstat) that drive the
``platform_telemetry`` module when the operator selects the external module
group from a profile.  It is fully decoupled from the sysfs-based
``modules/{cpu,gpu,npu}_*`` telemetry stack — both groups can coexist and
profiles select between them via ``CORE_TELEMETRY_MODULE_GROUP`` /
``module_group``.
"""

from .models import MetricSample
from .base import BaseCollector

__all__ = ["MetricSample", "BaseCollector"]
