# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Common service orchestration utilities for ESQ suites."""

from .catalog import TIMESERIES_COMMON_SERVICES
from .compose import ComposeServiceOrchestrator
from .dockerhub_app import DockerHubTimeseriesAppManager
from .influx_reader import compute_processed_points_latency_from_influx, compute_throughput_latency_from_influx
from .interfaces import ServiceInterface, ServiceSpec
from .models import ComposeProjectConfig, ServiceStartResult, TimeseriesServiceConfig
from .reference_app import ReferenceTimeseriesAppManager
from .report_gen import (
    append_performance_row,
    ensure_timeseries_report_paths,
    generate_performance_graphs,
    generate_presentation_csv,
    _get_current_system_cpu,
)

__all__ = [
    "ComposeProjectConfig",
    "ComposeServiceOrchestrator",
    "DockerHubTimeseriesAppManager",
    "ServiceInterface",
    "ServiceStartResult",
    "ServiceSpec",
    "TIMESERIES_COMMON_SERVICES",
    "TimeseriesServiceConfig",
    "ReferenceTimeseriesAppManager",
    "compute_processed_points_latency_from_influx",
    "compute_throughput_latency_from_influx",
    "append_performance_row",
    "ensure_timeseries_report_paths",
    "generate_performance_graphs",
    "generate_presentation_csv",
]
