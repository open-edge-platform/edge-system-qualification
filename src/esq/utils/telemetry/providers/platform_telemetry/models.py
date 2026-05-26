# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Data model for samples produced by platform_telemetry collectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MetricSample:
    """One numeric reading from an in-process collector.

    Mirrors the telecollect ``MetricSample`` shape so collector code can be
    vendored in with only import path changes.
    """

    timestamp_utc: str
    collector: str
    device: str
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
