# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer pipeline utilities and helpers."""

import logging
import re
from uuid import uuid4

from sysagent.utils.system.ov_helper import get_available_devices, get_openvino_device_type

logger = logging.getLogger(__name__)


def get_device_type_key(device_id: str, device_dict=None, has_igpu=None):
    """
    Centralized function to determine device type key for parameter mapping.

    Args:
        device_id: Device ID to use for parameter lookup
        device_dict: Dictionary containing device information (optional)
        has_igpu: Boolean indicating if system has iGPU (optional, will be detected if not provided)

    Returns:
        String type key for parameter mapping with variants for indexed GPUs:
        - "igpu" or "igpu_indexed" (iGPU with dGPU present)
        - "dgpu", "dgpu_primary" (single dGPU), or "dgpu_indexed" (multiple dGPUs)
        - "cpu", "npu", etc.
    """
    # Get device type from device_dict if available
    device_type = None
    if device_dict and device_id in device_dict:
        device_type = device_dict[device_id]["device_type"]

    device_type_key = device_type.lower() if device_type else ""
    device_id_key = device_id.lower()

    # Detect if integrated GPU is present in the system (if not provided)
    if has_igpu is None and "gpu" in device_id_key:
        has_igpu = False
        all_system_devices = get_available_devices()
        for dev_id in all_system_devices:
            if dev_id.upper().startswith("GPU"):
                dev_type = get_openvino_device_type(dev_id)
                if dev_type and "integrated" in dev_type.lower():
                    has_igpu = True
                    break

    # Determine type_key for mapping with indexed variants
    if "gpu" in device_id_key:
        if "integrated" in device_type_key:
            # iGPU: check if indexed (GPU.0) vs non-indexed (GPU)
            if "." in device_id:
                return "igpu_indexed"  # iGPU with dGPU present → GPU.0
            else:
                return "igpu"  # iGPU alone → GPU
        elif "discrete" in device_type_key:
            # dGPU: distinguish primary, indexed, or regular
            if "." in device_id:
                # Indexed dGPU (GPU.0, GPU.1, etc.)
                device_num = int(device_id.split(".")[-1])
                if device_num == 0 and not has_igpu:
                    return "dgpu_indexed"  # First dGPU in multi-dGPU system (no iGPU)
                else:
                    return "dgpu"  # dGPU with iGPU present (GPU.1, GPU.2, etc.)
            else:
                # Non-indexed dGPU (GPU)
                return "dgpu_primary"  # Single dGPU without iGPU
        else:
            return "gpu"
    elif "cpu" in device_id_key:
        return "cpu"
    elif "npu" in device_id_key:
        return "npu"
    else:
        return device_type_key


def build_multi_pipeline_with_devices(pipeline, device_id, num_streams, sink_element=None, fpscounter_elements=None):
    """
    Build a multi-stream pipeline for a specific device with FPS measurement aggregation.

    Architecture:
    - Multi-stream pipeline: Runs the actual workload with user-configured sink element
    - Result pipeline: Separate aggregator that reads FPS data via named pipe

    The result pipeline always uses fakesink (not configurable) because it only aggregates
    FPS measurements from the named pipe, not video processing. However, the sync and async
    properties are extracted from the workload sink element to ensure timing and state change
    consistency between the workload and aggregator pipelines.

    Args:
        pipeline: The base pipeline template string
        device_id: Device ID to use for the pipeline
        num_streams: Number of streams to include in the multi-pipeline
        sink_element: Full GStreamer sink element specification with properties for the WORKLOAD pipeline
                     (e.g., "fakesink sync=true", "fakesink sync=false async=false",
                           "filesink location=/tmp/out.mp4", "ximagesink",
                           "gvawatermark device=CPU ! videoconvert ! autovideosink" for visualization)
                     Both sync and async properties (if present) are extracted and applied to result pipeline.
                     If None, defaults to "fakesink sync=true"
        fpscounter_elements: Additional properties for gvafpscounter element (e.g., "starting-frame=2000")

    Returns:
        Tuple of (multi_pipeline, result_pipeline)
        - multi_pipeline: Workload pipeline with user-configured sink element
        - result_pipeline: FPS aggregator pipeline (always uses fakesink with matching sync property)
    """
    # Set default sink element if not provided
    if sink_element is None:
        sink_element = "fakesink sync=true"

    fpscounter_props = f" {fpscounter_elements}" if fpscounter_elements else ""

    # Extract sync and async properties from sink_element to ensure result pipeline matches
    # These properties must match between workload and result pipelines for timing consistency
    sync_value = "false"
    async_value = None  # None means property not specified (use GStreamer default)

    if "sync=" in sink_element:
        # Extract sync value from sink_element (handles "sync=true" or "sync=false")
        sync_match = re.search(r"sync=(true|false)", sink_element)
        if sync_match:
            sync_value = sync_match.group(1)

    if "async=" in sink_element:
        # Extract async value from sink_element (handles "async=true" or "async=false")
        async_match = re.search(r"async=(true|false)", sink_element)
        if async_match:
            async_value = async_match.group(1)

    inputs = []
    logger.debug(
        f"Building multi-pipeline for device {device_id} with {num_streams} streams using sink: {sink_element}"
    )
    for i in range(num_streams):
        # Generate unique pipeline ID for each stream
        placeholder_pipeline_id = str(uuid4())[:8]
        pipe_id = str(uuid4())[:8]

        # Replace ${PIPELINE_ID} placeholder in the pipeline string
        current_pipeline = pipeline.replace("${PIPELINE_ID}", placeholder_pipeline_id)

        additional_elements = f"! gvafpscounter{fpscounter_props} write-pipe=/tmp/{pipe_id} ! {sink_element}"
        inputs.append(f"{current_pipeline} {additional_elements}")

    pipeline_branches = " ".join(inputs)
    multi_pipeline = f"gst-launch-1.0 -q {pipeline_branches}"

    # Result pipeline is a separate aggregator process that only reads FPS data from named pipe.
    # It MUST use fakesink because it's not processing video frames, only aggregating measurements.
    # However, the sync and async properties MUST match the workload pipeline's sink to ensure:
    # - sync: Timing consistency between write-pipe (workload) and read-pipe (aggregator) for accurate FPS measurements
    # - async: State change behavior consistency to avoid potential pipeline stalls or preroll issues
    fakesink_props = f"sync={sync_value}"
    if async_value is not None:
        fakesink_props += f" async={async_value}"

    result_pipeline = (
        f"gst-launch-1.0 -q gvafpscounter{fpscounter_props} read-pipe=/tmp/{pipe_id} ! fakesink {fakesink_props}"
    )

    # Log the pipeline with better truncation handling to show both start and end
    if len(multi_pipeline) > 1500:
        truncated_pipeline = f"{multi_pipeline[:1000]} ... \n[middle content truncated] ... {multi_pipeline[-500:]}"
        logger.debug(f"Generated multi_pipeline (truncated): {truncated_pipeline}")
    else:
        truncated_pipeline = multi_pipeline
        logger.debug(f"Generated multi_pipeline: {multi_pipeline}")

    # Log result pipeline with extracted properties
    props_info = f"sync={sync_value}"
    if async_value is not None:
        props_info += f", async={async_value}"
    logger.debug(f"Generated result_pipeline ({props_info}): {result_pipeline}")

    return multi_pipeline, result_pipeline


def build_baseline_pipeline(pipeline):
    """
    Build a baseline pipeline for single stream analysis.
    Always uses sync=false for maximum throughput measurement.
    Note: Baseline analysis uses default gvafpscounter (no custom properties)
          to measure raw pipeline performance.

    Args:
        pipeline: The resolved pipeline string

    Returns:
        Tuple of (baseline_pipeline, result_pipeline)
    """
    pipe_id = str(uuid4())[:8]
    placeholder_pipeline_id = str(uuid4())[:8]
    sync_str = "false"

    # Replace ${PIPELINE_ID} placeholder in the pipeline string
    pipeline = pipeline.replace("${PIPELINE_ID}", placeholder_pipeline_id)

    baseline_pipeline = (
        f"gst-launch-1.0 -q {pipeline} ! gvafpscounter write-pipe=/tmp/{pipe_id} ! fakesink sync={sync_str}"
    )
    result_pipeline = f"gst-launch-1.0 -q gvafpscounter read-pipe=/tmp/{pipe_id} ! fakesink sync={sync_str}"

    logger.debug(f"Generated baseline_pipeline: {baseline_pipeline}")
    logger.debug(f"Generated result_pipeline: {result_pipeline}")

    return baseline_pipeline, result_pipeline


def get_sink_element_config(pipeline_params, device_id, device_dict=None):
    """
    Get sink element configuration for multi-stream pipeline from pipeline_params.
    Supports full GStreamer sink element specification with properties.
    Baseline analysis always uses "fakesink sync=false" (hardcoded) for maximum throughput measurement.

    Args:
        pipeline_params: Dictionary of pipeline parameters
        device_id: Device ID to use for parameter lookup
        device_dict: Dictionary containing device information (optional)

    Returns:
        String sink element configuration (e.g., "fakesink sync=true", "filesink location=/tmp/out.mp4", "ximagesink")
    """
    if not pipeline_params:
        # Return default if no params provided
        return "fakesink sync=true"

    # Use centralized type_key determination
    type_key = get_device_type_key(device_id, device_dict)

    # Get sink element configuration
    sink_params = pipeline_params.get("SINK_ELEMENT", {})

    # Get device-specific sink element or use default
    sink_element = sink_params.get(type_key, sink_params.get("default", "fakesink sync=true"))
    logger.debug(f"Sink element for {device_id}: {sink_element}")

    return sink_element


def get_fpscounter_config(pipeline_params, device_id, device_dict=None):
    """
    Get FPS counter element properties configuration from pipeline_params.
    This allows configuring gvafpscounter properties like starting-frame.

    Args:
        pipeline_params: Dictionary of pipeline parameters
        device_id: Device ID to use for parameter lookup
        device_dict: Dictionary containing device information (optional)

    Returns:
        String with fpscounter element properties (e.g., "starting-frame=2000")
    """
    if not pipeline_params:
        # Return empty if no params provided
        return ""

    # Use centralized type_key determination
    type_key = get_device_type_key(device_id, device_dict)

    # Get fpscounter properties configuration
    fpscounter_params = pipeline_params.get("FPSCOUNTER_PROPS", {})

    # Try to get device-specific value, fall back to default
    fpscounter_value = fpscounter_params.get(type_key, fpscounter_params.get("default", ""))

    logger.debug(f"FPS counter properties for {device_id}: {fpscounter_value}")
    return fpscounter_value


def get_pipeline_info(
    device_id: str,
    pipeline: str,
    pipeline_params=None,
    device_dict=None,
    num_streams: int = None,
    is_baseline: bool = False,
):
    """
    Helper function to get pipeline information for a device.

    Args:
        device_id: Device ID
        pipeline: Base pipeline string
        pipeline_params: Dictionary of parameters for different devices/types
        device_dict: Dictionary containing device information (optional)
        num_streams: Number of streams (for multi-pipeline), None for baseline
        is_baseline: Whether this is for baseline or multi-stream analysis

    Returns:
        Dict with pipeline information
    """
    try:
        # Resolve pipeline placeholders
        resolved_pipeline = resolve_pipeline_placeholders(pipeline, pipeline_params, device_id, device_dict)

        if is_baseline or num_streams is None:
            base, result = build_baseline_pipeline(
                pipeline=resolved_pipeline,
            )
            if len(base) > 1500:
                truncated = f"{base[:1000]} ... \n[middle content truncated] ... {base[-500:]}"
            else:
                truncated = base
            return {"baseline_pipeline": truncated, "result_pipeline": result}
        else:
            fpscounter_config = get_fpscounter_config(pipeline_params, device_id, device_dict)
            sink_element = get_sink_element_config(pipeline_params, device_id, device_dict)
            multi, result = build_multi_pipeline_with_devices(
                pipeline=resolved_pipeline,
                device_id=device_id,
                num_streams=num_streams,
                sink_element=sink_element,
                fpscounter_elements=fpscounter_config,
            )
            if len(multi) > 1500:
                truncated = f"{multi[:1000]} ... \n[middle content truncated] ... {multi[-500:]}"
            else:
                truncated = multi
            return {"multi_pipeline": truncated, "result_pipeline": result}
    except Exception as e:
        logger.warning(f"Failed to get pipeline info for {device_id}: {e}")
        return {"baseline_pipeline": "", "multi_pipeline": "", "result_pipeline": ""}


def resolve_pipeline_placeholders(pipeline, pipeline_params=None, device_id="CPU", device_dict=None):
    """
    Resolve placeholders in the pipeline string with actual values.

    Supports flexible parameter structure:
    - Type-based resolution: Uses type_key (cpu, npu, igpu, dgpu, etc.)
    - Property-based overrides: Device-specific overrides using OpenVINO device properties
    - Default fallback: Uses 'default' value if no type/property match

    Args:
        pipeline: The pipeline string with placeholders
        pipeline_params: Dictionary of parameters with structure:
            PARAM_NAME:
              default: "fallback_value"
              cpu: "cpu_value"
              npu: "npu_value"
              igpu: "igpu_value"
              igpu_indexed: "igpu_indexed_value"
              dgpu: "dgpu_value"
              dgpu_primary: "dgpu_primary_value"
              dgpu_indexed: "dgpu_indexed_value"
              overrides:
                DEVICE_PROPERTY_NAME:
                  "property_value": "override_value"
        device_id: Device ID to use for parameter lookup
                  Special value "multi_stage" indicates multi-stage pipeline mode
        device_dict: Dictionary containing device information (optional)

    Returns:
        Resolved pipeline string

    Reserved placeholders:
        - ${DEVICE_ID}: Replaced with the device_id parameter
        - ${RENDER_DEVICE_NUM}: For discrete GPUs, calculated from device ID
        - ${HAS_IGPU}: Boolean indicating if integrated GPU is present in system

    Note: ${PIPELINE_ID} is handled separately in build_multi_pipeline_with_devices
    and build_baseline_pipeline functions, not here.
    """
    if not pipeline_params:
        pipeline_params = {}

    # Handle multi-stage mode
    is_multi_stage = device_id == "multi_stage"

    if is_multi_stage:
        logger.debug("Multi-stage pipeline mode: resolving placeholders for composite pipeline")
        # For multi-stage, use first available GPU to determine type_key variant
        gpu_device_id = None
        if device_dict:
            for dev_id, dev_info in device_dict.items():
                if dev_id.upper().startswith("GPU"):
                    gpu_device_id = dev_id
                    logger.debug(f"Multi-stage mode: using {gpu_device_id} for GPU type resolution")
                    break

        effective_device_id = gpu_device_id if gpu_device_id else "GPU"
    else:
        effective_device_id = device_id

    # Detect if integrated GPU is available in the system
    has_igpu = False
    all_system_devices = get_available_devices()
    for dev_id in all_system_devices:
        if dev_id.upper().startswith("GPU"):
            dev_type = get_openvino_device_type(dev_id)
            if dev_type and "integrated" in dev_type.lower():
                has_igpu = True
                logger.debug(f"Integrated GPU detected: {dev_id}")
                break

    # Get type_key using enhanced function
    type_key = get_device_type_key(effective_device_id, device_dict, has_igpu)
    logger.debug(f"Resolving placeholders for {device_id}: type_key={type_key}, has_igpu={has_igpu}")

    # Get device properties from OpenVINO
    from sysagent.utils.system.ov_helper import get_openvino_device_properties

    device_properties = get_openvino_device_properties(effective_device_id)
    all_properties = device_properties.get("all_properties", {})

    # Compute RENDER_DEVICE_NUM for discrete GPUs
    render_device_num = ""
    if "dgpu" in type_key:
        base_render_num = 128
        device_num = 0
        try:
            if "." in effective_device_id:
                device_num = int(effective_device_id.split(".")[-1])
            render_device_num = str(base_render_num + device_num)
            logger.debug(f"Discrete GPU: device_num={device_num}, render={render_device_num}")
        except Exception as e:
            logger.warning(f"Failed to extract device number from {effective_device_id}: {e}")
            render_device_num = str(base_render_num)

    # Initialize replacements with reserved placeholders
    replacements = {
        "RENDER_DEVICE_NUM": render_device_num,
        "HAS_IGPU": "true" if has_igpu else "false",
    }

    # Resolve all pipeline_params placeholders
    for placeholder, mapping in pipeline_params.items():
        # Skip reserved placeholders
        if placeholder in ("DEVICE_ID", "RENDER_DEVICE_NUM", "HAS_IGPU"):
            continue

        if not isinstance(mapping, dict):
            continue

        value = None
        resolved_type_key = type_key  # Default to global type_key

        # Multi-stage mode: Try to infer device type from placeholder content
        # Check if placeholder has ONLY values for specific device types (e.g., only 'npu' key)
        if is_multi_stage:
            available_type_keys = [k for k in mapping.keys() if k not in ("default", "overrides")]

            # If placeholder has exactly one non-default type key, use it
            if len(available_type_keys) == 1:
                resolved_type_key = available_type_keys[0]
                logger.debug(f"'{placeholder}': single type key found → using '{resolved_type_key}'")

            # If placeholder has multiple type keys, try to match based on global type_key category
            elif len(available_type_keys) > 1:
                # Check if global type_key is in the mapping
                if type_key in available_type_keys:
                    resolved_type_key = type_key
                # Otherwise, try to find matching category (e.g., dgpu matches dgpu_primary, dgpu_indexed)
                else:
                    for available_key in available_type_keys:
                        if "gpu" in type_key and "gpu" in available_key:
                            resolved_type_key = available_key
                            logger.debug(f"'{placeholder}': GPU category match → using '{resolved_type_key}'")
                            break

        # Single-device mode: Check property-based overrides first
        if not is_multi_stage:
            overrides = mapping.get("overrides", {})
            if overrides and all_properties:
                for prop_key, override_map in overrides.items():
                    device_value = all_properties.get(prop_key)
                    if device_value and isinstance(override_map, dict) and device_value in override_map:
                        value = override_map[device_value]
                        logger.debug(f"'{placeholder}': override match {prop_key}='{device_value}' → '{value}'")
                        break

        # Priority 1 (or 2 for single-device): Type-key specific value
        if not value and resolved_type_key in mapping:
            value = mapping[resolved_type_key]
            logger.debug(f"'{placeholder}': type_key '{resolved_type_key}' → '{value}'")

        # Priority 2 (or 3 for single-device): Default value
        if not value and "default" in mapping:
            value = mapping["default"]
            logger.debug(f"'{placeholder}': using default → '{value}'")

        if value:
            replacements[placeholder] = value

    # Resolve nested placeholders in replacement values
    def resolve_nested(val: str, max_depth: int = 10) -> str:
        """
        Resolve nested placeholders in a string with depth limit.

        Args:
            val: String containing placeholders
            max_depth: Maximum nesting depth to prevent infinite loops

        Returns:
            String with resolved placeholders
        """
        import re

        pattern = r"\{([A-Z0-9_]+)\}"
        skip_placeholders = {"PIPELINE_ID", "DEVICE_ID"}  # Handled elsewhere
        depth = 0

        while depth < max_depth:
            match = re.search(pattern, val)
            if not match:
                break

            nested_key = match.group(1)

            if nested_key in skip_placeholders:
                logger.debug(f"Skipping nested placeholder '{nested_key}' (handled elsewhere)")
                val = val.replace(f"{{{nested_key}}}", f"RESERVED_{nested_key}")
                continue

            nested_val = replacements.get(nested_key, "")
            val = val.replace(f"{{{nested_key}}}", nested_val)
            depth += 1

        if depth == max_depth and re.search(pattern, val):
            logger.warning(
                f"Maximum nesting depth ({max_depth}) reached resolving placeholders. "
                "Possible circular references or excessive nesting."
            )

        return val

    # Resolve nested placeholders in all replacement values
    for k in replacements:
        if isinstance(replacements[k], str):
            replacements[k] = resolve_nested(replacements[k])

    # Apply replacements to pipeline string
    for placeholder, value in replacements.items():
        placeholder_str = f"${{{placeholder}}}"
        if placeholder_str in pipeline:
            pipeline = pipeline.replace(placeholder_str, value)

    # Always replace DEVICE_ID
    pipeline = pipeline.replace("${DEVICE_ID}", device_id)

    logger.debug(f"Resolved pipeline: {pipeline}")
    return pipeline
