# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Metrics Manager backed framework-level telemetry provider.

This package implements the primary backend for the ``platform_telemetry``
module (``module_group: external``): instead of collecting GPU/NPU/CPU
metrics in-process, it pulls/runs the official Open Edge Platform
"Metrics Manager" container (``intel/metrics-manager``, published on Docker
Hub) and reads its REST API.
"""

from .provider import MetricsManagerProvider

__all__ = ["MetricsManagerProvider"]
