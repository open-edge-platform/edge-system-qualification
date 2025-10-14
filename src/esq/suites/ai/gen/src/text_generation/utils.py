# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Text Generation utility functions for device management and cleanup."""

import logging
from typing import Dict, Any, List
from sysagent.utils.infrastructure import DockerClient

logger = logging.getLogger(__name__)


def cleanup_stale_containers(docker_client: DockerClient, container_prefix: str) -> None:
    """
    Clean up stale containers with the given prefix.
    
    Args:
        docker_client: Docker client instance
        container_prefix: Container name prefix to match
    """
    try:
        logger.info(f"Cleaning up stale containers with prefix: {container_prefix}")
        
        containers = docker_client.client.containers.list(all=True)
        cleaned_count = 0
        
        for container in containers:
            if container.name.startswith(container_prefix):
                try:
                    logger.debug(f"Removing stale container: {container.name}")
                    
                    # Stop container if it's running
                    if container.status == "running":
                        container.stop(timeout=10)
                    
                    # Remove container
                    container.remove()
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to clean up container {container.name}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale containers")
        else:
            logger.debug("No stale containers found")
            
    except Exception as e:
        logger.error(f"Failed to cleanup stale containers: {e}")


def sort_devices_by_priority(device_list: List[str]) -> List[str]:
    """
    Sort devices by priority for testing order.
    
    Priority order: GPU (discrete) -> GPU (integrated) -> NPU -> CPU
    
    Args:
        device_list: List of device IDs
        
    Returns:
        Sorted list of device IDs
    """
    def get_device_priority(device_id: str) -> int:
        """Get device priority (lower number = higher priority)."""
        device_lower = device_id.lower()
        
        # Discrete GPU (highest priority)
        if device_lower.startswith("gpu") and not device_lower.endswith(".0"):
            return 1
            
        # Integrated GPU
        if device_lower.startswith("gpu") and device_lower.endswith(".0"):
            return 2
            
        # NPU
        if device_lower.startswith("npu"):
            return 3
            
        # CPU (lowest priority)
        if device_lower.startswith("cpu"):
            return 4
            
        # Unknown devices (very low priority)
        return 5
    
    sorted_devices = sorted(device_list, key=get_device_priority)
    logger.info(f"Device testing order: {sorted_devices}")
    
    return sorted_devices


def validate_device_availability(device_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that devices are available and properly configured.
    
    Args:
        device_dict: Dictionary of device information
        
    Returns:
        Dict with validation results
    """
    logger.info("Validating device availability...")
    
    validation_results = {
        "total_devices": len(device_dict),
        "available_devices": [],
        "unavailable_devices": [],
        "device_info": {}
    }
    
    for device_id, device_info in device_dict.items():
        try:
            # Basic validation
            if not device_info or not isinstance(device_info, dict):
                logger.warning(f"Device {device_id} has invalid info")
                validation_results["unavailable_devices"].append(device_id)
                continue
                
            device_type = device_info.get("device_type", "unknown")
            full_name = device_info.get("full_name", "unknown")
            
            # Check if device type is supported
            supported_types = ["cpu", "gpu", "npu"]
            if device_type.lower() not in supported_types:
                logger.warning(f"Device {device_id} has unsupported type: {device_type}")
                validation_results["unavailable_devices"].append(device_id)
                continue
            
            # Device is available
            validation_results["available_devices"].append(device_id)
            validation_results["device_info"][device_id] = {
                "type": device_type,
                "name": full_name,
                "status": "available"
            }
            
            logger.debug(f"Device {device_id} ({device_type}): {full_name} - Available")
            
        except Exception as e:
            logger.error(f"Error validating device {device_id}: {e}")
            validation_results["unavailable_devices"].append(device_id)
            validation_results["device_info"][device_id] = {
                "status": "error",
                "error": str(e)
            }
    
    available_count = len(validation_results["available_devices"])
    total_count = validation_results["total_devices"]
    
    logger.info(f"Device validation: {available_count}/{total_count} devices available")
    
    return validation_results


def generate_test_summary(
    results: List[Any],
    device_list: List[str],
    validation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate comprehensive test summary.
    
    Args:
        results: List of test results
        device_list: List of tested devices
        validation_results: Device validation results
        
    Returns:
        Dict with test summary
    """
    logger.info("Generating test summary...")
    
    summary = {
        "test_overview": {
            "total_devices": len(device_list),
            "devices_tested": len([r for r in results if r.metadata.get("device_id")]),
            "successful_tests": len([r for r in results if not r.metadata.get("error")]),
            "failed_tests": len([r for r in results if r.metadata.get("error")])
        },
        "device_results": {},
        "performance_metrics": {
            "total_throughput": 0.0,
            "max_throughput": 0.0,
            "min_throughput": float('inf'),
            "avg_throughput": 0.0
        },
        "validation_summary": validation_results
    }
    
    throughputs = []
    
    for result in results:
        device_id = result.metadata.get("device_id", "unknown")
        
        # Extract performance metrics
        throughput = -1.0
        for metric_name, metric in result.metrics.items():
            if "throughput" in metric_name:
                throughput = metric.value
                break
        
        # Device result summary
        device_result = {
            "device_id": device_id,
            "throughput": throughput,
            "status": "success" if not result.metadata.get("error") else "failed",
            "error": result.metadata.get("error")
        }
        
        summary["device_results"][device_id] = device_result
        
        # Collect valid throughputs for statistics
        if throughput > 0:
            throughputs.append(throughput)
    
    # Calculate performance statistics
    if throughputs:
        summary["performance_metrics"]["total_throughput"] = sum(throughputs)
        summary["performance_metrics"]["max_throughput"] = max(throughputs)
        summary["performance_metrics"]["min_throughput"] = min(throughputs)
        summary["performance_metrics"]["avg_throughput"] = sum(throughputs) / len(throughputs)
    else:
        summary["performance_metrics"]["min_throughput"] = 0.0
    
    logger.info(f"Test summary generated for {len(device_list)} devices")
    
    return summary
