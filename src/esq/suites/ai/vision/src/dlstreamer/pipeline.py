# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer pipeline utilities and helpers."""

import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


def build_multi_pipeline_with_devices(pipeline, device_id, num_streams, visualize_stream=False, sync_model=True):
    """
    Build a multi-stream pipeline for a specific device.

    Args:
        pipeline: The base pipeline template string
        device_id: Device ID to use for the pipeline
        num_streams: Number of streams to include in the multi-pipeline
        visualize_stream: Whether to visualize the pipeline output
        sync_model: Whether to synchronize the model execution

    Returns:
        Tuple of (multi_pipeline, result_pipeline)
    """
    sync_str = "true" if sync_model else "false"

    if visualize_stream:
        # sync_elements = f"videoconvert ! autovideosink sync={sync_str}"
        # sync_elements = f"gvawatermark device=CPU ! videoconvert ! ximagesink sync={sync_str}"
        sync_elements = f"gvawatermark device=CPU ! videoconvert ! autovideosink sync={sync_str}"
    else:
        sync_elements = f"fakesink sync={sync_str}"

    inputs = []
    logger.debug(f"Building multi-pipeline for device {device_id} with {num_streams} streams.")
    for i in range(num_streams):
        # Generate unique pipeline ID for each stream
        placeholder_pipeline_id = str(uuid4())[:8]
        pipe_id = str(uuid4())[:8]

        # Replace ${PIPELINE_ID} placeholder in the pipeline string
        current_pipeline = pipeline.replace("${PIPELINE_ID}", placeholder_pipeline_id)

        additional_elements = f"! gvafpscounter write-pipe=/tmp/{pipe_id} ! {sync_elements}"
        inputs.append(f"{current_pipeline} {additional_elements}")

    pipeline_branches = " ".join(inputs)
    multi_pipeline = f"gst-launch-1.0 -q {pipeline_branches}"
    result_pipeline = f"gst-launch-1.0 -q gvafpscounter read-pipe=/tmp/{pipe_id} ! fakesink sync=true"

    # Log the pipeline with better truncation handling to show both start and end
    if len(multi_pipeline) > 1500:
        truncated_pipeline = f"{multi_pipeline[:1000]} ... \n[middle content truncated] ... {multi_pipeline[-500:]}"
        logger.debug(f"Generated multi_pipeline (truncated): {truncated_pipeline}")
    else:
        truncated_pipeline = multi_pipeline
        logger.debug(f"Generated multi_pipeline: {multi_pipeline}")
    logger.debug(f"Generated result_pipeline: {result_pipeline}")
    return multi_pipeline, result_pipeline


def build_baseline_pipeline(pipeline, sync_model=False):
    """
    Build a baseline pipeline for single stream analysis.

    Args:
        pipeline: The resolved pipeline string
        sync_model: Whether to use sync=true or sync=false for the fakesink

    Returns:
        Tuple of (baseline_pipeline, result_pipeline)
    """
    pipe_id = str(uuid4())[:8]
    placeholder_pipeline_id = str(uuid4())[:8]
    sync_str = "true" if sync_model else "false"

    # Replace ${PIPELINE_ID} placeholder in the pipeline string
    pipeline = pipeline.replace("${PIPELINE_ID}", placeholder_pipeline_id)

    baseline_pipeline = (
        f"gst-launch-1.0 -q {pipeline} ! gvafpscounter write-pipe=/tmp/{pipe_id} ! fakesink sync={sync_str}"
    )
    result_pipeline = f"gst-launch-1.0 -q gvafpscounter read-pipe=/tmp/{pipe_id} ! fakesink sync=false"

    logger.debug(f"Generated baseline_pipeline: {baseline_pipeline}")
    logger.debug(f"Generated result_pipeline: {result_pipeline}")

    return baseline_pipeline, result_pipeline


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
            # Build baseline pipeline
            base, result = build_baseline_pipeline(pipeline=resolved_pipeline, sync_model=False)
            if len(base) > 1500:
                truncated = f"{base[:1000]} ... \n[middle content truncated] ... {base[-500:]}"
            else:
                truncated = base
            return {"baseline_pipeline": truncated, "result_pipeline": result}
        else:
            # Build multi-stream pipeline
            multi, result = build_multi_pipeline_with_devices(
                pipeline=resolved_pipeline, device_id=device_id, num_streams=num_streams, sync_model=True
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

    Args:
        pipeline: The pipeline string with placeholders
        pipeline_params: Dictionary of parameters for different devices/types
        device_id: Device ID to use for parameter lookup
        device_dict: Dictionary containing device information (optional)

    Returns:
        Resolved pipeline string

    Note:
        Reserved placeholders handled internally:
        - ${DEVICE_ID}: Replaced with the device_id parameter
        - ${RENDER_DEVICE_NUM}: For discrete GPUs, calculated from device ID

        Note that ${PIPELINE_ID} is handled separately in build_multi_pipeline_with_devices
        and build_baseline_pipeline functions, not here.
    """
    if not pipeline_params:
        pipeline_params = {}

    # Get device type from device_dict if available
    device_type = None
    if device_dict and device_id in device_dict:
        device_type = device_dict[device_id]["device_type"]

    device_type_key = device_type.lower() if device_type else ""
    device_id_key = device_id.lower()

    logger.debug(f"Resolving pipeline placeholders for device_id: {device_id}, device_type: {device_type}")

    # Determine type_key for mapping
    if "gpu" in device_id_key:
        if "integrated" in device_type_key:
            type_key = "gpu_integrated"
        elif "discrete" in device_type_key:
            type_key = "gpu_discrete"
        else:
            type_key = "gpu"
    elif "cpu" in device_id_key:
        type_key = "cpu"
    elif "npu" in device_id_key:
        type_key = "npu"
    else:
        type_key = device_type_key

    # Get device properties using OpenVINO
    from sysagent.utils.system.ov_helper import get_openvino_device_properties

    device_properties = get_openvino_device_properties(device_id)
    all_properties = device_properties.get("all_properties", {})

    # Compute RENDER_DEVICE_NUM for discrete GPUs
    render_device_num = ""
    if type_key == "gpu_discrete":
        base_render_num = int(128)
        # Extract device number from device_id (e.g. GPU.1 -> 1)
        try:
            device_num = int(device_id.split(".")[-1])
            render_device_num = str(base_render_num + device_num)
        except Exception:
            render_device_num = str(base_render_num)

    # Note: PIPELINE_ID is now handled in build_multi_pipeline_with_devices and build_baseline_pipeline
    replacements = {"RENDER_DEVICE_NUM": render_device_num}

    # First pass: resolve all top-level placeholders
    for placeholder, mapping in pipeline_params.items():
        if placeholder in ("DEVICE_ID", "RENDER_DEVICE_NUM"):
            continue  # handled separately

        value = None
        if isinstance(mapping, dict):
            # Check for property-based overrides
            overrides = mapping.get("overrides", {})
            if overrides and all_properties:
                for prop_key, override_map in overrides.items():
                    device_value = all_properties.get(prop_key)
                    if device_value and device_value in override_map:
                        value = override_map[device_value]
                        break
            # If no override, use type_key or default
            if not value:
                value = mapping.get(type_key) or mapping.get("default")
        if value:
            replacements[placeholder] = value

    # Second pass: resolve nested placeholders in replacement values
    def resolve_nested(val: str, max_depth: int = 10) -> str:
        """
        Resolve nested placeholders in a string with depth limit to prevent infinite loops.

        Args:
            val: The string containing placeholders
            max_depth: Maximum depth of nested placeholders to resolve

        Returns:
            String with resolved placeholders
        """
        import re

        pattern = r"\{([A-Z0-9_]+)\}"
        # Skip placeholders that are handled elsewhere to avoid wasting iterations
        skip_placeholders = {"PIPELINE_ID", "DEVICE_ID"}
        depth = 0

        while depth < max_depth:
            match = re.search(pattern, val)
            if not match:
                break

            nested_key = match.group(1)

            # Skip reserved placeholders that are handled elsewhere
            if nested_key in skip_placeholders:
                # Replace with empty to avoid infinite loop but log the skip
                logger.debug(f"Skipping nested placeholder '{nested_key}' as it's handled elsewhere")
                val = val.replace(f"{{{nested_key}}}", f"RESERVED_{nested_key}")
                continue

            nested_val = replacements.get(nested_key, "")
            val = val.replace(f"{{{nested_key}}}", nested_val)
            depth += 1

        if depth == max_depth and re.search(pattern, val):
            logger.warning(
                f"Maximum nesting depth ({max_depth}) reached when resolving placeholders. "
                "There might be circular references or too deeply nested placeholders."
            )
        return val

    # Only resolve nested placeholders if they're actually in the original string
    # to avoid unnecessary processing
    for k in replacements:
        if isinstance(replacements[k], str):
            replacements[k] = resolve_nested(replacements[k])

    # Only replace placeholders that actually exist in the pipeline string
    for placeholder, value in replacements.items():
        if f"${{{placeholder}}}" in pipeline:
            pipeline = pipeline.replace(f"${{{placeholder}}}", value)

    # Always replace DEVICE_ID since it's a required placeholder
    pipeline = pipeline.replace("${DEVICE_ID}", device_id)
    logger.debug(f"Resolved pipeline: {pipeline}")

    return pipeline
