# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Text Generation qualification functions for device testing."""

import logging
from typing import Dict, Any, Optional
from sysagent.utils.core import Result, Metrics

logger = logging.getLogger(__name__)


def qualify_device(
    device_id: str,
    throughput: float,
    min_throughput_threshold: float = 0.1
) -> bool:
    """
    Determine if a device qualifies based on performance metrics.
    
    Args:
        device_id: Device identifier
        throughput: Measured throughput in tokens/sec
        min_throughput_threshold: Minimum throughput to qualify
        
    Returns:
        True if device qualifies, False otherwise
    """
    logger.info(f"Qualifying device {device_id} with throughput: {throughput} tokens/sec")
    
    if throughput < 0:
        logger.warning(f"Device {device_id} failed - negative throughput")
        return False
        
    if throughput < min_throughput_threshold:
        logger.warning(f"Device {device_id} failed - throughput {throughput} below threshold {min_throughput_threshold}")
        return False
        
    logger.info(f"Device {device_id} qualified with throughput: {throughput} tokens/sec")
    return True


def validate_device_results(
    results: list,
    min_success_rate: float = 0.5
) -> Dict[str, Any]:
    """
    Validate results from device testing.
    
    Args:
        results: List of Result objects from device tests
        min_success_rate: Minimum success rate required
        
    Returns:
        Dict with validation results
    """
    logger.info("Validating device test results...")
    
    total_devices = len(results)
    successful_devices = 0
    qualified_devices = {}
    
    for result in results:
        device_id = result.metadata.get("device_id", "unknown")
        
        # Check if device test was successful
        if result.metadata.get("status", True) and not result.metadata.get("error"):
            # Extract throughput
            throughput = -1.0
            for metric_name, metric in result.metrics.items():
                if "throughput" in metric_name:
                    throughput = metric.value
                    break
                    
            # Qualify device
            if qualify_device(device_id, throughput):
                successful_devices += 1
                qualified_devices[device_id] = {
                    "throughput": throughput,
                    "qualified": True,
                    "status": "success"
                }
            else:
                qualified_devices[device_id] = {
                    "throughput": throughput,
                    "qualified": False,
                    "status": "failed_qualification"
                }
        else:
            error = result.metadata.get("error", "Unknown error")
            qualified_devices[device_id] = {
                "throughput": -1.0,
                "qualified": False,
                "status": "failed_execution",
                "error": error
            }
    
    success_rate = successful_devices / total_devices if total_devices > 0 else 0.0
    overall_success = success_rate >= min_success_rate
    
    validation_results = {
        "total_devices": total_devices,
        "successful_devices": successful_devices,
        "success_rate": success_rate,
        "min_success_rate": min_success_rate,
        "overall_success": overall_success,
        "qualified_devices": qualified_devices
    }
    
    logger.info(f"Validation results: {successful_devices}/{total_devices} devices successful ({success_rate:.2%})")
    
    return validation_results
