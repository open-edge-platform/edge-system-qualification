# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared service models for timeseries and other multi-service ESQ suites."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ComposeProjectConfig:
    """Configuration for a docker compose project used by test suites."""

    compose_file: str
    project_name: str
    working_dir: str


@dataclass
class ServiceStartResult:
    """Result from starting one or more services."""

    ok: bool
    services: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""


@dataclass
class TimeseriesServiceConfig:
    """Common service-level config for timeseries style deployments."""

    num_streams: int = 1
    num_data_points: int = 1000
    environment: Dict[str, str] = field(default_factory=dict)
    container_images: Optional[List[str]] = None
