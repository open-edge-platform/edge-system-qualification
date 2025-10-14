# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Text Generation test execution and result processing functions."""

import json
import logging
import os
from typing import Dict, Any, Optional

from sysagent.utils.core import Result, Metrics, get_metric_name_for_device
from sysagent.utils.infrastructure import DockerClient
from .container import run_ovms_server_container

logger = logging.getLogger(__name__)


def run_device_test(
    docker_client: DockerClient,
    device_id: str,
    docker_image_tag: str,
    docker_container_prefix: str,
    data_dir: str,
    container_mnt_dir: str,
    model_id: str,
    model_precision: str,
    test_num_prompts: int,
    test_request_rate: int,
    test_max_concurrent_requests: int,
    server_timeout: int,
    benchmark_timeout: int,
    dataset_path: str,
    metrics: Optional[Dict[str, Metrics]] = None
) -> Result:
    """
    Execute Text Generation test for a single device.
    
    Args:
        docker_client: Docker client instance
        device_id: Device ID to test
        docker_image_tag: Docker image tag
        docker_container_prefix: Container name prefix
        data_dir: Host data directory
        container_mnt_dir: Container mount directory
        model_id: Model identifier
        model_precision: Model precision
        test_num_prompts: Number of prompts to test
        test_request_rate: Request rate
        test_max_concurrent_requests: Max concurrent requests
        server_timeout: Server startup timeout
        benchmark_timeout: Benchmark timeout
        dataset_path: Path to dataset file
        metrics: Default metrics for the device
        
    Returns:
        Result object with test outcomes
    """
    logger.info(f"Running Text Generation test on device: {device_id}")
    
    # Initialize metrics if not provided
    if metrics is None:
        metric_name = get_metric_name_for_device(device_id, prefix="throughput")
        metrics = {metric_name: Metrics(unit="tokens/sec", value=-1.0)}
    
    # Container configuration - use fixed port like original implementation
    ovms_port = 8000  # Fixed port like original
    
    container_info = None
    try:
        # Export model to OpenVINO Model Server format
        logger.info(f"Exporting model {model_id} for {device_id} with {model_precision} "
                   f"to OpenVINO Model Server format")
        from esq.utils.models import export_ovms_model
        
        # Models directory is in data_dir/models as per test configuration
        models_dir = os.path.join(data_dir, "models")
        
        export_status = export_ovms_model(
            model_id_or_path=model_id,
            models_dir=models_dir,
            model_precision=model_precision,
            device_id=device_id,
        )
        
        if not export_status:
            error_message = f"Failed to export model {model_id} for device {device_id} with precision {model_precision}"
            logger.error(error_message)
            raise RuntimeError(error_message)
            
        # Start OVMS server container
        container_info = run_ovms_server_container(
            docker_client=docker_client,
            device_id=device_id,
            docker_image_tag=docker_image_tag,
            docker_container_prefix=docker_container_prefix,
            data_dir=data_dir,
            container_mnt_dir=container_mnt_dir,
            server_timeout=server_timeout,
            model_id=model_id,
            port=ovms_port
        )
        
        logger.info(f"OVMS server started on port {ovms_port} for device {device_id}")
        
        # Run benchmark using the original container approach
        throughput = run_benchmark_performance_test(
            docker_client=docker_client,
            docker_container_prefix=docker_container_prefix,
            port=ovms_port,
            dataset_path=dataset_path,
            data_dir=data_dir,
            model_id=model_id,
            model_precision=model_precision,
            device_id=device_id,
            test_num_prompts=test_num_prompts,
            test_request_rate=test_request_rate,
            test_max_concurrent_requests=test_max_concurrent_requests,
            benchmark_timeout=benchmark_timeout
        )
        
        # Update metrics with results
        metric_name = get_metric_name_for_device(device_id, prefix="throughput")
        if metric_name in metrics:
            metrics[metric_name].value = throughput
            
        logger.info(f"Device {device_id} throughput: {throughput:.2f} tokens/sec")
        
        # Get detailed benchmark metadata
        benchmark_metadata = get_benchmark_metadata(
            data_dir=data_dir,
            model_id=model_id,
            model_precision=model_precision,
            device_id=device_id
        )
        
        # Create result with detailed benchmark metadata
        result = Result(
            parameters={
                "Device ID": device_id,
                "Model ID": model_id,
                "Model Precision": model_precision,
                "Number of Prompts": test_num_prompts,
                "Request Rate": test_request_rate,
                "Max Concurrent Requests": test_max_concurrent_requests,
            },
            metrics=metrics,
            metadata={
                "device_id": device_id,
                "status": True,
                "container_name": container_info.get("name", "") if container_info else "",
                "container_id": container_info.get("container_id", "") if container_info else "",
                **benchmark_metadata
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Test failed for device {device_id}: {e}")
        
        # Update metrics with failure
        metric_name = get_metric_name_for_device(device_id, prefix="throughput")
        if metric_name in metrics:
            metrics[metric_name].value = -1.0
            
        # Create failure result
        result = Result(
            parameters={
                "Device ID": device_id,
                "Model ID": model_id,
                "Error": str(e),
            },
            metrics=metrics,
            metadata={
                "device_id": device_id,
                "status": False,
                "error": str(e),
            }
        )
        
        return result
        
    finally:
        # Clean up container and collect logs for attachment
        if container_info and "container" in container_info:
            try:
                container_name = container_info.get("name", "")
                container = container_info.get("container")
                logger.info(f"Stopping container: {container_name}")
                
                if container:
                    # Stop log streaming first - this will handle log consolidation automatically
                    # when CORE_SUPPRESS_CONTAINER_LOG_ATTACHMENTS is set
                    docker_client.stop_log_streaming(container_name)
                    docker_client.cleanup_container(container_name)
                
            except Exception as e:
                logger.warning(f"Failed to cleanup container: {e}")
                # Fallback cleanup like original
                try:
                    docker_client.cleanup_containers_by_name_pattern(container_name)
                except Exception as cleanup_e:
                    logger.warning(f"Fallback cleanup also failed: {cleanup_e}")
                except Exception as fallback_error:
                    logger.warning(f"Fallback cleanup also failed: {fallback_error}")


def run_benchmark_performance_test(
    docker_client: DockerClient,
    docker_container_prefix: str,
    port: int,
    dataset_path: str,
    data_dir: str,
    model_id: str,
    model_precision: str,
    device_id: str,
    test_num_prompts: int,
    test_request_rate: int,
    test_max_concurrent_requests: int,
    benchmark_timeout: int
) -> float:
    """
    Run performance test using benchmark container (original approach).
    
    Args:
        docker_client: Docker client instance
        docker_container_prefix: Container name prefix
        port: OVMS server port
        dataset_path: Path to dataset file
        data_dir: Data directory path
        model_id: Model identifier
        model_precision: Model precision
        device_id: Device identifier
        test_num_prompts: Number of prompts to test
        test_request_rate: Request rate
        test_max_concurrent_requests: Max concurrent requests
        benchmark_timeout: Benchmark timeout
        
    Returns:
        Throughput in tokens per second
    """
    logger.info(f"Running benchmark performance test on port {port}")
    
    # Import the benchmark container function
    from .container import run_benchmark_container
    
    # Extract dataset filename
    hf_dataset_filename = os.path.basename(dataset_path)
    
    try:
        # Run benchmark container
        benchmark_result = run_benchmark_container(
            docker_client=docker_client,
            docker_container_prefix=docker_container_prefix,
            data_dir=data_dir,
            model_id=model_id,
            model_precision=model_precision,
            device_id=device_id,
            dataset_path=dataset_path,
            hf_dataset_filename=hf_dataset_filename,
            ovms_port=port,
            test_num_prompts=test_num_prompts,
            test_request_rate=test_request_rate,
            test_max_concurrent_requests=test_max_concurrent_requests,
            benchmark_timeout=benchmark_timeout
        )
        
        if not benchmark_result.get("success", False):
            logger.error("Benchmark container execution failed")
            return -1.0
        
        # Process benchmark results (following original implementation)
        results_file = benchmark_result.get("results_file")
        if not results_file or not os.path.exists(results_file):
            logger.error(f"Benchmark results file not found: {results_file}")
            return -1.0
        
        # Load and parse results JSON
        with open(results_file, 'r') as file:
            results_json = json.load(file)
        
        # Extract throughput metrics following original format
        output_throughput = results_json.get('output_throughput', -1.0)
        
        # Round throughput to 2 decimal places like the original implementation
        rounded_throughput = round(output_throughput, 2) if output_throughput > 0 else 0.0
        
        logger.info(f"Benchmark results: Output Throughput: {rounded_throughput} tokens/sec")
        logger.debug(f"Full benchmark results: {json.dumps(results_json, indent=2)}")
        
        return rounded_throughput
        
    except Exception as e:
        logger.error(f"Benchmark performance test failed: {e}")
        return -1.0


def get_benchmark_metadata(
    data_dir: str,
    model_id: str,
    model_precision: str,
    device_id: str
) -> Dict[str, Any]:
    """
    Extract detailed benchmark metadata from results file.
    
    Args:
        data_dir: Data directory path
        model_id: Model identifier
        model_precision: Model precision
        device_id: Device identifier
        
    Returns:
        Dict with detailed benchmark metadata
    """
    metadata = {
        "Total Input Tokens": -1.0,
        "Total Output Tokens": -1.0,
        "Output Throughput": -1.0,
        "Mean TTFT (ms)": -1.0,
        "Mean TPOT (ms)": -1.0,
    }
    
    try:
        # Results file path
        results_dir = os.path.join(data_dir, "results", "text_generation")
        # File should be based on the model basename
        model_basename = os.path.basename(model_id.replace('/', '_'))
        results_file = os.path.join(
            results_dir, 
            f"ovms-{model_basename}-{model_precision}-{device_id}.json"
        )
        
        if not os.path.exists(results_file):
            logger.warning(f"Benchmark results file not found: {results_file}")
            return metadata
            
        # Load and parse results JSON
        with open(results_file, 'r') as file:
            results_json = json.load(file)
        
        # Extract and round all metadata values like the original implementation
        metadata["Total Input Tokens"] = round(results_json.get('total_input_tokens', -1.0), 2)
        metadata["Total Output Tokens"] = round(results_json.get('total_output_tokens', -1.0), 2)  
        metadata["Output Throughput"] = round(results_json.get('output_throughput', -1.0), 2)
        metadata["Mean TTFT (ms)"] = round(results_json.get('mean_ttft_ms', -1.0), 2)
        metadata["Mean TPOT (ms)"] = round(results_json.get('mean_tpot_ms', -1.0), 2)
        
        logger.debug(f"Extracted benchmark metadata: {metadata}")
        
    except Exception as e:
        logger.warning(f"Failed to extract benchmark metadata: {e}")
        
    return metadata


# Note: Inference request functions removed - now using benchmark container approach


def load_test_prompts(dataset_path: str, num_prompts: int) -> list:
    """
    Load test prompts from dataset file.
    
    Args:
        dataset_path: Path to dataset file
        num_prompts: Number of prompts to load
        
    Returns:
        List of prompt strings
    """
    prompts = []
    
    try:
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset file not found: {dataset_path}")
            return prompts
            
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract prompts from ShareGPT format
        if isinstance(data, list):
            for item in data[:num_prompts]:
                if isinstance(item, dict) and "conversations" in item:
                    conversations = item["conversations"]
                    if conversations and isinstance(conversations[0], dict):
                        prompt = conversations[0].get("value", "")
                        if prompt:
                            prompts.append(prompt)
                            
        logger.info(f"Loaded {len(prompts)} prompts from dataset")
        
    except Exception as e:
        logger.warning(f"Failed to load prompts from dataset: {e}")
        
    return prompts


def process_device_results(
    results: list,
    device_list: list,
    final_results: Dict[str, Any]
) -> None:
    """
    Process results from individual device tests.
    
    Args:
        results: List of Result objects from device tests
        device_list: List of device IDs
        final_results: Dictionary to update with final results
    """
    logger.info("Processing device test results...")
    
    total_throughput = 0.0
    successful_devices = 0
    
    for result in results:
        device_id = result.metadata.get("device_id", "unknown")
        
        # Extract throughput metric
        metric_name = get_metric_name_for_device(device_id, prefix="throughput")
        
        if metric_name in result.metrics:
            throughput = result.metrics[metric_name].value
            
            if throughput > 0:
                total_throughput += throughput
                successful_devices += 1
                logger.info(f"Device {device_id}: {throughput:.2f} tokens/sec")
            else:
                logger.warning(f"Device {device_id}: Test failed")
        
        # Update final results with individual device metrics
        final_results["metrics"].update(result.metrics)
    
    # Add summary metrics
    if successful_devices > 0:
        avg_throughput = total_throughput / successful_devices
        final_results["metadata"]["total_throughput"] = round(total_throughput, 2) if total_throughput > 0 else 0.0
        final_results["metadata"]["average_throughput"] = round(avg_throughput, 2) if avg_throughput > 0 else 0.0
        final_results["metadata"]["successful_devices"] = successful_devices
        
        logger.info(f"Total throughput: {total_throughput:.2f} tokens/sec")
        logger.info(f"Average throughput: {avg_throughput:.2f} tokens/sec")
        logger.info(f"Successful devices: {successful_devices}/{len(device_list)}")
    else:
        logger.error("No devices completed successfully")
        final_results["metadata"]["status"] = False
