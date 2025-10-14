# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer test modular utilities."""

from .utils import (
    cleanup_stale_containers,
    cleanup_thread_pool,
    sort_devices_by_priority,
    update_device_metrics
)

from .preparation import (
    prepare_assets,
    prepare_baseline,
    prepare_estimate_num_streams_for_device
)

from .qualification import (
    get_cpu_socket_numa_info,
    run_benchmark_container,
    run_concurrent_analysis,
    qualify_device
)

from .analysis import (
    parse_device_result_file
)

__all__ = [
    # Utilities
    'cleanup_stale_containers',
    'cleanup_thread_pool', 
    'sort_devices_by_priority',
    'update_device_metrics',
    
    # Preparation
    'prepare_assets',
    'prepare_baseline',
    'prepare_estimate_num_streams_for_device',
    
    # Qualification and benchmarking
    'get_cpu_socket_numa_info',
    'run_benchmark_container',
    'run_concurrent_analysis',
    'qualify_device',
    
    # Analysis
    'parse_device_result_file'
]
