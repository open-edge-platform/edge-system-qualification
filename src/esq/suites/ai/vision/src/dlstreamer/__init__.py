# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer test modular utilities."""

from .analysis import parse_device_result_file
from .concurrent import (
    get_cpu_socket_numa_info,
    run_benchmark_container,
    run_concurrent_analysis,
)
from .preparation import (
    prepare_assets,
    prepare_baseline,
    prepare_estimate_num_streams_for_device,
)
from .qualification import qualify_device
from .results import (
    finalize_device_metrics,
    process_device_results,
    update_device_pipeline_info,
    update_final_results_metadata,
    validate_final_streams_results,
)
from .utils import (
    cleanup_stale_containers,
    cleanup_thread_pool,
    sort_devices_by_priority,
    update_device_metrics,
)

__all__ = [
    # Utilities
    "cleanup_stale_containers",
    "cleanup_thread_pool",
    "sort_devices_by_priority",
    "update_device_metrics",
    # Preparation
    "prepare_assets",
    "prepare_baseline",
    "prepare_estimate_num_streams_for_device",
    # Concurrent and benchmarking
    "get_cpu_socket_numa_info",
    "run_benchmark_container",
    "run_concurrent_analysis",
    # Qualification
    "qualify_device",
    # Analysis
    "parse_device_result_file",
    # Results
    "finalize_device_metrics",
    "process_device_results",
    "update_device_pipeline_info",
    "update_final_results_metadata",
    "validate_final_streams_results",
]
