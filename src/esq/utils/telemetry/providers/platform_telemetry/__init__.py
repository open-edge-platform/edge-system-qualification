# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Supplemental external telemetry collectors for Metrics Manager gaps."""

from .base import BaseCollector
from .models import MetricSample

__all__ = ["MetricSample", "BaseCollector"]
