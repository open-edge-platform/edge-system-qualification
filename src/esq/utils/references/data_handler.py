# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Reference Data Handler Module.

Provides utilities for processing verified reference data from profile configurations:
- Filtering by CPU generation and GPU models to show only relevant data
- Converting to CSV format for Allure attachments
- Adding to test results as extended metadata

ARCHITECTURAL NOTE:
This module previously contained CPU generation matching logic that has been
refactored to src/sysagent/utils/system/cpu.py for reusability across the entire
codebase. The core matching functions are now:

1. normalize_generation_string() - Normalizes CPU generation strings
2. match_cpu_generations() - Matches CPU generations comprehensively

These shared functions support ALL Intel processor families:
- Core Ultra Series (1, 2, 3, etc.)
- Core i-series (14th, 13th, 12th Gen, etc.)
- Core Series (non-i, e.g., Core 7 250U)
- Xeon 6 (new 2024+ naming)
- Xeon Scalable (legacy naming)
- Xeon W (workstation)
- Intel Atom (x7000, x6000, x5000, N-series, etc.)
- Intel N-series (Alder Lake-N, Twin Lake)
- Pentium, Celeron (all generations)

This architecture ensures:
- No code duplication between modules
- Future processor generations automatically supported
- Consistent matching logic across different use cases
- Easy maintenance and testing

For GPU matching, this module focuses on Intel GPUs only:
- Intel Arc Graphics (A/B/C-series)
- Intel Iris Xe Graphics
- Intel UHD/HD Graphics
- Intel Xe Graphics (any generation)
"""

import csv
import io
import logging
from typing import Dict, List

import allure
from sysagent.utils.core import Result
from sysagent.utils.system import SystemInfoCache
from sysagent.utils.system.cpu import detect_cpu_generation_and_segment, match_cpu_generations

logger = logging.getLogger(__name__)


def _get_system_cpu_generation() -> str:
    """
    Get the current system's CPU generation string.

    Returns:
        CPU generation string (e.g., "Core Ultra (Series 2)", "12th Gen Core")
        or empty string if detection fails
    """
    try:
        system_info = SystemInfoCache()
        hardware_info = system_info.get_hardware_info()
        cpu_info = hardware_info.get("cpu", {})

        # Get CPU detection parameters
        family = cpu_info.get("family", 0)
        model = cpu_info.get("model", 0)
        stepping = cpu_info.get("stepping", 0)
        brand = cpu_info.get("brand", "")
        core_count = cpu_info.get("cores", 0)

        # Detect generation
        detection_result = detect_cpu_generation_and_segment(
            family=family, model=model, stepping=stepping, brand=brand, core_count=core_count
        )

        generation = detection_result.get("generation", "")
        logger.debug(f"Detected system CPU generation: {generation}")
        return generation

    except Exception as e:
        logger.warning(f"Failed to detect system CPU generation: {e}")
        return ""


def _get_system_gpu_models() -> List[str]:
    """
    Get the current system's GPU model names.

    Returns:
        List of GPU model names found in the system (e.g., ["Intel Arc B60", "Intel Arc A770"])
    """
    try:
        system_info = SystemInfoCache()
        hardware_info = system_info.get_hardware_info()
        gpu_info = hardware_info.get("gpu", {})
        devices = gpu_info.get("devices", [])

        gpu_models = []
        for device in devices:
            # Try different fields for GPU name
            full_name = device.get("full_name", "")
            canonical_name = device.get("canonical_name", "")
            openvino_info = device.get("openvino", {})
            openvino_name = openvino_info.get("full_device_name", "")

            # Use the first available name
            gpu_name = full_name or canonical_name or openvino_name

            if gpu_name:
                # Normalize GPU name for matching
                gpu_name_normalized = gpu_name.strip()
                if gpu_name_normalized and gpu_name_normalized not in gpu_models:
                    gpu_models.append(gpu_name_normalized)
                    logger.debug(f"Found GPU: {gpu_name_normalized}")

        logger.debug(f"Detected {len(gpu_models)} GPU model(s): {gpu_models}")
        return gpu_models

    except Exception as e:
        logger.warning(f"Failed to detect system GPU models: {e}")
        return []


def _normalize_generation(generation: str) -> str:
    """
    Normalize generation string for comparison.

    NOTE: This is a deprecated wrapper around the shared normalize_generation_string()
    function in cpu.py. New code should import and use:
        from sysagent.utils.system.cpu import normalize_generation_string

    Args:
        generation: Generation string to normalize

    Returns:
        Normalized generation string for comparison
    """
    from sysagent.utils.system.cpu import normalize_generation_string

    return normalize_generation_string(generation)


def _normalize_device_sku(device_sku: str) -> str:
    """
    Normalize device SKU for matching.

    Handles:
    - GPU name variations (e.g., "Arc B60" vs "Intel Arc B60" vs "DG2 [Arc B60]")
    - Case insensitivity
    - Extra whitespace

    Args:
        device_sku: Device SKU string to normalize

    Returns:
        Normalized device SKU for matching
    """
    if not device_sku:
        return ""

    # Convert to lowercase and strip whitespace
    sku_lower = device_sku.lower().strip()

    # Remove common prefixes
    prefixes_to_remove = ["intel ", "intel® ", "intel(r) "]
    for prefix in prefixes_to_remove:
        if sku_lower.startswith(prefix):
            sku_lower = sku_lower[len(prefix) :]

    # Remove bracket content for GPU names (e.g., "dg2 [arc b60]" -> "arc b60")
    import re

    bracket_match = re.search(r"\[([^\]]+)\]", sku_lower)
    if bracket_match:
        sku_lower = bracket_match.group(1).strip()

    # Remove special characters and extra spaces
    sku_lower = re.sub(r"[™®]", "", sku_lower)
    sku_lower = re.sub(r"\s+", " ", sku_lower).strip()

    return sku_lower


def _match_generation(system_generation: str, entry_sku: str) -> bool:
    """
    Check if a reference data entry matches the system's CPU generation.

    NOTE: This is a wrapper around the shared match_cpu_generations() function
    in cpu.py. The actual matching logic has been moved to cpu.py for reusability
    across the entire codebase.

    New code should import and use:
        from sysagent.utils.system.cpu import match_cpu_generations

    Args:
        system_generation: System CPU generation string from detect_cpu_generation_and_segment()
                          Examples: "Core Ultra (Series 2)", "14th Gen Core", "Xeon 6"
        entry_sku: Device SKU from reference data entry
                   Examples: "Intel Core Ultra 7 255H", "Intel Xeon 6730P"

    Returns:
        True if the entry matches the system generation, False otherwise
    """
    return match_cpu_generations(system_generation, entry_sku)


def _match_model(model_filter: str, entry_model: str) -> bool:
    """
    Check if an entry's AI model matches the filter model.

    Handles model name variations:
    - "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" matches "DeepSeek-R1-Distill-Qwen-1.5B"
    - "microsoft/Phi-4-mini-reasoning" matches "Phi-4-mini-reasoning"
    - "Qwen/Qwen3-32B" matches "Qwen3-32B"
    - Case insensitive matching
    - Handles precision suffixes (INT4, INT8, etc.)

    Args:
        model_filter: Model ID from test case (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        entry_model: Model name from reference data (e.g., "DeepSeek-R1-Distill-Qwen-7B")

    Returns:
        True if models match, False otherwise
    """
    if not model_filter or not entry_model:
        return False

    # Normalize both strings
    filter_lower = model_filter.lower()
    entry_lower = entry_model.lower()

    # Remove organization prefix from filter (e.g., "deepseek-ai/" -> "")
    if "/" in filter_lower:
        filter_lower = filter_lower.split("/", 1)[1]

    # Remove precision suffixes from entry model (INT4, INT8, etc.)
    import re

    entry_lower = re.sub(r"\s+(int\d+|fp\d+)$", "", entry_lower)

    # Check for exact match or substring match
    return filter_lower == entry_lower or filter_lower in entry_lower or entry_lower in filter_lower


def _match_gpu(system_gpu_models: List[str], entry_sku: str) -> bool:
    """
    Check if a reference data entry matches any of the system's Intel GPU models.

    This function focuses on Intel GPUs only:
    - Intel Arc Graphics (A-series, B-series, C-series+): Arc A770, Arc B60, Arc B580, etc.
    - Intel Iris Xe Graphics: Iris Xe, Iris Plus
    - Intel UHD Graphics: UHD 770, UHD 730, UHD Graphics
    - Intel HD Graphics: HD 630, HD 620, HD Graphics
    - Intel Xe Graphics (any generation)

    Matching strategies:
    1. Exact normalized string match
    2. Substring containment (handles "Arc B60" vs "Intel Arc B60")
    3. Model number/series extraction (e.g., "B60", "A770")
    4. Intel GPU family keyword matching (Arc, Iris, UHD, HD, Xe)

    Args:
        system_gpu_models: List of Intel GPU model names from system
                          Examples: ["Intel Arc B60", "Intel Iris Xe Graphics"]
        entry_sku: Device SKU from reference data entry
                   Examples: "Intel Arc B60", "Arc A770", "Iris Xe"

    Returns:
        True if the entry matches any system Intel GPU, False otherwise
    """
    if not system_gpu_models or not entry_sku:
        return False

    entry_normalized = _normalize_device_sku(entry_sku)

    # Skip if entry is not an Intel GPU
    if not any(keyword in entry_normalized for keyword in ["arc", "iris", "uhd", "hd graphics", "xe"]):
        return False

    for gpu_model in system_gpu_models:
        gpu_normalized = _normalize_device_sku(gpu_model)

        # Strategy 1: Exact match
        if gpu_normalized == entry_normalized:
            logger.debug(f"Intel GPU exact match: '{entry_sku}' == '{gpu_model}'")
            return True

        # Strategy 2: Substring containment
        # Handles "Arc B60" vs "Intel Arc B60" vs "DG2 [Arc B60]"
        if gpu_normalized in entry_normalized or entry_normalized in gpu_normalized:
            logger.debug(f"Intel GPU substring match: '{entry_sku}' matches '{gpu_model}'")
            return True

        # Strategy 3: Intel Arc model number extraction
        # Extract Arc model identifiers: "A770", "B60", "B580", etc.
        import re

        gpu_arc_model = re.search(r"\barc\s+([a-z]\d{2,3})\b", gpu_normalized)
        entry_arc_model = re.search(r"\barc\s+([a-z]\d{2,3})\b", entry_normalized)

        if gpu_arc_model and entry_arc_model:
            if gpu_arc_model.group(1) == entry_arc_model.group(1):
                logger.debug(
                    f"Intel Arc model match: '{entry_sku}' matches '{gpu_model}' (Arc {gpu_arc_model.group(1).upper()})"
                )
                return True

        # Strategy 4: Intel GPU family keyword matching
        # Check for common Intel GPU families: Arc, Iris, UHD, HD, Xe
        gpu_keywords = set(re.findall(r"\b(arc|iris|uhd|hd|xe)\b", gpu_normalized))
        entry_keywords = set(re.findall(r"\b(arc|iris|uhd|hd|xe)\b", entry_normalized))

        if gpu_keywords and entry_keywords:
            overlap = gpu_keywords & entry_keywords
            if overlap:
                logger.debug(f"Intel GPU family match: '{entry_sku}' matches '{gpu_model}' (family: {overlap})")
                return True

    return False


def filter_reference_data_by_generation(reference_data: Dict, model_filter: str = None) -> List[Dict]:
    """
    Filter reference data to include only entries matching system CPU generation,
    GPU models, and optionally AI model.

    For dGPU entries: Matches against system GPU models
    For CPU/iGPU/NPU entries: Matches against CPU generation
    For Gen AI: Optionally filters by model name if model_filter is provided

    Reference data should be in dict format with 'columns' and 'data' keys:
    {
        "columns": {"device_sku": "Device SKU", ...},
        "data": [{"device_sku": "...", ...}, ...]
    }

    Args:
        reference_data: Dict with 'columns' and 'data' keys
        model_filter: Optional AI model name/ID to filter by (e.g., for Gen AI tests)

    Returns:
        List of filtered data entries
    """
    if not reference_data:
        logger.debug("No reference data provided")
        return []

    # Extract data entries
    data_entries = reference_data.get("data", [])
    if not data_entries:
        logger.warning("No data entries found in reference data")
        return []

    # Get system CPU generation and GPU models
    system_generation = _get_system_cpu_generation()
    system_gpus = _get_system_gpu_models()

    if not system_generation and not system_gpus:
        logger.warning("Could not detect system generation or GPU models; returning all entries")
        return data_entries

    logger.info(f"Filtering reference data by generation: {system_generation}, GPUs: {system_gpus}")

    # Filter entries by device type and optionally by model
    filtered_entries = []
    for entry in data_entries:
        device_sku = entry.get("device_sku", "")
        entry_type = entry.get("type", "").lower()

        # If model_filter is provided, check if entry matches the model
        if model_filter:
            entry_model = entry.get("model", "")
            if not _match_model(model_filter, entry_model):
                logger.debug(f"Skipping entry - model mismatch: {entry_model} != {model_filter}")
                continue

        should_include = False

        # For dGPU entries, match against system GPUs
        if entry_type == "dgpu":
            if _match_gpu(system_gpus, device_sku):
                should_include = True
                logger.debug(f"Including dGPU entry: {device_sku} (matches system GPU)")
        else:
            # For CPU/iGPU/NPU entries, match against CPU generation
            if _match_generation(system_generation, device_sku):
                should_include = True
                logger.debug(f"Including entry: {device_sku} (matches system generation)")

        if should_include:
            filtered_entries.append(entry)

    logger.info(f"Filtered {len(filtered_entries)} entries out of {len(data_entries)}")
    return filtered_entries


def convert_reference_data_to_csv(reference_data: Dict) -> str:
    """
    Convert reference data to CSV format string with human-readable column headers.

    Args:
        reference_data: Dict with 'columns' and 'data' keys

    Returns:
        CSV formatted string
    """
    if not reference_data:
        return ""

    data_entries = reference_data.get("data", [])
    if not data_entries:
        return ""

    # Get column mappings and field names
    column_mapping = reference_data.get("columns", {})
    fieldnames = list(data_entries[0].keys())

    # Create headers using column mapping
    headers = [column_mapping.get(field, field) for field in fieldnames]

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header row
    writer.writerow(headers)

    # Write data rows
    for entry in data_entries:
        row = [entry.get(field, "") for field in fieldnames]
        writer.writerow(row)

    csv_content = output.getvalue()
    output.close()

    return csv_content


def attach_reference_data_to_allure(
    reference_data: Dict,
    attachment_name: str = "Verified Reference Data",
    filter_by_generation: bool = True,
    model_filter: str = None,
) -> None:
    """
    Attach reference data as CSV to Allure report.

    Args:
        reference_data: Dict with 'columns' and 'data' keys
        attachment_name: Name for the Allure attachment
        filter_by_generation: Whether to filter by system generation (default: True)
        model_filter: Optional AI model name/ID to filter by (e.g., for Gen AI tests)
    """
    if not reference_data:
        logger.debug("No reference data to attach to Allure")
        return

    # Filter by generation and/or model if requested
    data_to_attach = reference_data
    if filter_by_generation:
        filtered_entries = filter_reference_data_by_generation(reference_data, model_filter=model_filter)
        if not filtered_entries:
            logger.warning("No reference data matches system generation/model; skipping Allure attachment")
            return

        # Preserve column mappings with filtered data
        data_to_attach = {"columns": reference_data.get("columns", {}), "data": filtered_entries}

    # Convert to CSV
    csv_content = convert_reference_data_to_csv(data_to_attach)

    # Attach to Allure
    try:
        allure.attach(csv_content, name=attachment_name, attachment_type=allure.attachment_type.CSV)

        entry_count = len(filtered_entries) if filter_by_generation else len(reference_data.get("data", []))
        logger.info(f"Attached reference data to Allure: {attachment_name} ({entry_count} entries)")
    except Exception as e:
        logger.error(f"Failed to attach reference data to Allure: {e}")


def add_reference_data_to_result(
    result: Result,
    reference_data: Dict,
    data_key: str = "verified_reference_data",
    filter_by_generation: bool = True,
    model_filter: str = None,
) -> None:
    """
    Add reference data to Result extended_metadata.

    Args:
        result: Result object to update
        reference_data: Dict with 'columns' and 'data' keys
        data_key: Key name in extended_metadata (default: "verified_reference_data")
        filter_by_generation: Whether to filter by system generation (default: True)
        model_filter: Optional AI model name/ID to filter by (e.g., for Gen AI tests)
    """
    if not reference_data:
        logger.debug("No reference data to add to result")
        return

    # Filter by generation and/or model if requested
    if filter_by_generation:
        filtered_entries = filter_reference_data_by_generation(reference_data, model_filter=model_filter)
        data_to_add = {"columns": reference_data.get("columns", {}), "data": filtered_entries}
    else:
        data_to_add = reference_data

    # Add to extended_metadata
    result.extended_metadata[data_key] = data_to_add

    entry_count = len(filtered_entries) if filter_by_generation else len(reference_data.get("data", []))
    logger.info(f"Added {entry_count} reference data entries to result.extended_metadata['{data_key}']")
