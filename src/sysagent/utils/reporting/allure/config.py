# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Allure configuration utilities.

This module contains functions for managing Allure report configuration,
including report naming, versioning, and filename generation with system information.
"""

import logging
import os
import re
import shutil
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _is_placeholder_value(value: str) -> bool:
    """
    Check if a DMI value is a placeholder or meaningless string.

    Args:
        value: DMI string to check

    Returns:
        bool: True if value is a placeholder/meaningless
    """
    if not value or not isinstance(value, str):
        return True

    # Convert to lowercase for case-insensitive comparison
    value_lower = value.lower().strip()

    # List of placeholder patterns
    placeholders = [
        "to be filled by o.e.m.",
        "to be filled by o.e.m",
        "to be filled by oem",
        "system product name",
        "system version",
        "default string",
        "sku",
        "sku number",
        "family",
        "unknown",
        "not specified",
        "n/a",
        "none",
    ]

    # Check exact matches
    if value_lower in placeholders:
        return True

    # Check if value is only dots (e.g., "............", "....................")
    if value.strip(".") == "":
        return True

    # Check if value is only asterisks
    if value.strip("*") == "":
        return True

    return False


def update_allure_config(
    allure_config_path: str,
    allure_repo_dir: str,
    report_name: Optional[str] = None,
    report_version: Optional[str] = None,
) -> str:
    """
    Copy and update Allure configuration file with custom settings.

    Args:
        allure_config_path: Path to the source allure configuration file
        allure_repo_dir: Directory where Allure repository is located
        report_name: Custom name for the Allure report
        report_version: Version string to use for allureVersion in config

    Returns:
        str: Path to the copied and updated configuration file

    Raises:
        Exception: If copying or modifying configuration fails
    """
    # Copy the configuration file to the project directory
    project_config_path = os.path.join(allure_repo_dir, "allurerc.mjs")
    logger.debug(f"Copying Allure config from {allure_config_path} to {project_config_path}")

    try:
        shutil.copy2(allure_config_path, project_config_path)

        # Read the config file for modifications
        with open(project_config_path, "r") as f:
            config_content = f.read()

        config_modified = False

        # Update the report name in the config file if specified
        if report_name:
            logger.debug(f"Setting custom report name: {report_name}")
            if "name:" in config_content:
                # Replace existing name
                config_content = re.sub(r'name:\s*"[^"]*"', f'name: "{report_name}"', config_content)
            else:
                # Add name before plugins if it doesn't exist
                config_content = config_content.replace("plugins:", f'name: "{report_name}",\n  plugins:')
            config_modified = True

        # Update the allureVersion if report_version is provided
        if report_version:
            logger.debug(f"Setting allureVersion to specified version: {report_version}")
            # Replace allureVersion in the awesome plugin options
            allure_version_pattern = r'(allureVersion:\s*)"[^"]*"'
            if re.search(allure_version_pattern, config_content):
                config_content = re.sub(allure_version_pattern, rf'\1"{report_version}"', config_content)
                config_modified = True
            else:
                logger.warning("Could not find allureVersion field in config file to update")

        # Write the modified config file back if any changes were made
        if config_modified:
            with open(project_config_path, "w") as f:
                f.write(config_content)

        return project_config_path

    except Exception as e:
        logger.error(f"Failed to copy or modify Allure configuration: {e}")
        raise


def generate_final_report_filename(app_name: str, system_info: Dict) -> str:
    """
    Generate a comprehensive filename for the final Allure report.

    Format: <appname>_report_<system_and_productname>_<cpu_brand>
    _<discrete_gpus>_<timestamp>.html

    Args:
        app_name: Name of the application
        system_info: Dictionary containing system information

    Returns:
        str: Generated filename
    """
    timestamp = _generate_short_timestamp()

    # Construct final filename with format
    filename_parts = [f"{app_name}_report"]

    # Add system and product information (only if both are valid)
    vendor = system_info.get("vendor", "").strip()
    product = system_info.get("product", "").strip()

    # Only include vendor/product if they have meaningful values
    if vendor and product:
        system_product = f"{vendor}_{product}"
        filename_parts.append(system_product)
    elif vendor:
        filename_parts.append(vendor)
    elif product:
        filename_parts.append(product)

    # Add CPU brand
    if system_info.get("cpu_brand"):
        filename_parts.append(system_info["cpu_brand"])

    # Add discrete GPUs
    if system_info.get("discrete_gpus"):
        # Join multiple discrete GPUs with underscore
        gpus_part = "_".join(system_info["discrete_gpus"])
        filename_parts.append(gpus_part)

    # Add timestamp at the end
    filename_parts.append(timestamp)

    final_filename = "_".join(filename_parts) + ".html"
    return final_filename.lower()


def get_comprehensive_system_info_for_filename() -> Dict:
    """
    Get comprehensive system vendor, product, CPU brand,
    and discrete GPU information for filename generation.

    Returns:
        dict: Dictionary with 'vendor', 'product', 'cpu_brand', and 'discrete_gpus' keys
    """
    try:
        logger.debug("Gathering comprehensive system information for filename")
        from sysagent.utils.config import setup_data_dir
        from sysagent.utils.system.cache import SystemInfoCache

        data_dir = setup_data_dir()
        cache_dir = os.path.join(data_dir, "cache")
        system_info_cache = SystemInfoCache(cache_dir)

        hw_info = system_info_cache.get_hardware_info()
        if not hw_info:
            return {}

        system_info = {}

        # Get DMI information (updated for current hardware structure)
        if "dmi" in hw_info:
            dmi_info = hw_info["dmi"]

            # Get vendor from system section
            if "system" in dmi_info and dmi_info["system"].get("vendor"):
                vendor_raw = dmi_info["system"]["vendor"]
                if not _is_placeholder_value(vendor_raw):
                    # Preserve "Intel" in vendor name - use preserve_intel=True
                    system_info["vendor"] = normalize_filename_component(vendor_raw, preserve_intel=True)

            # Get product name from system section (updated field name)
            if "system" in dmi_info and dmi_info["system"].get("product_name"):
                product_raw = dmi_info["system"]["product_name"]
                if not _is_placeholder_value(product_raw):
                    system_info["product"] = normalize_filename_component(product_raw, preserve_intel=False)

        # Get CPU brand information
        if "cpu" in hw_info:
            cpu_info = hw_info["cpu"]
            cpu_brand = cpu_info.get("brand_raw") or cpu_info.get("brand", "") or cpu_info.get("model", "")
            if cpu_brand:
                # Normalize CPU brand for filename
                system_info["cpu_brand"] = normalize_cpu_brand(cpu_brand)

        # Get discrete GPU information (updated for flattened OpenVINO structure)
        discrete_gpus = []
        if "gpu" in hw_info and "devices" in hw_info["gpu"]:
            gpu_devices = hw_info["gpu"]["devices"]
            if gpu_devices:
                # Look for all discrete GPUs with flattened OpenVINO structure
                for gpu in gpu_devices:
                    is_discrete = gpu.get("is_discrete", False)
                    openvino = gpu.get("openvino", {})

                    # Use is_discrete field first, fallback to OpenVINO device_type
                    if is_discrete or openvino.get("device_type") == "Type.DISCRETE":
                        # Use OpenVINO full_device_name if available
                        dgpu_name = (
                            openvino.get("full_device_name")
                            or gpu.get("canonical_name", "")
                            or gpu.get("device_name", "")
                            or openvino.get("device_name", "")
                        )
                        if dgpu_name:
                            normalized_gpu = normalize_intel_gpu_name(dgpu_name)
                            if normalized_gpu and normalized_gpu != "unknown":
                                discrete_gpus.append(normalized_gpu)

        if discrete_gpus:
            system_info["discrete_gpus"] = discrete_gpus

        logger.debug(f"Comprehensive system info for filename: {system_info}")

        return system_info

    except Exception as e:
        logger.debug(f"Could not get comprehensive system info for filename: {e}")
        return {}


def cleanup_old_final_reports(report_dir: str, debug: bool = False) -> None:
    """
    Clean up old final report files to keep only the latest version.
    Removes files matching the pattern *_report_*.html except index.html

    Args:
        report_dir: Directory containing the report files
        debug: Whether to show debug level logs
    """
    try:
        if not os.path.exists(report_dir):
            return

        # Find all final report files (but not index.html)
        report_pattern = "*_report_*.html"
        import glob

        old_reports = glob.glob(os.path.join(report_dir, report_pattern))

        # Remove old final reports
        for old_report in old_reports:
            try:
                os.remove(old_report)
                if debug:
                    logger.debug(f"Removed old final report: {os.path.basename(old_report)}")
            except Exception as e:
                logger.warning(f"Failed to remove old report {old_report}: {e}")

        if old_reports and debug:
            logger.debug(f"Cleaned up {len(old_reports)} old final report(s)")

    except Exception as e:
        logger.warning(f"Failed to cleanup old final reports: {e}")


def normalize_cpu_brand(cpu_brand: str) -> str:
    """
    Normalize CPU brand for filename, making it generic and flexible for any CPU.
    Excludes "Intel" prefix since reports are always for Intel products.

    Args:
        cpu_brand: Original CPU brand string

    Returns:
        str: Normalized CPU brand
    """
    if not cpu_brand:
        return "unknown"

    # Remove common unnecessary parts and Intel branding
    cpu_clean = re.sub(r"\(R\)|\(TM\)|\(C\)", "", cpu_brand)
    cpu_clean = re.sub(r"\bCorporation\b|\bIntel\b", "", cpu_clean, flags=re.IGNORECASE)
    cpu_clean = re.sub(r"\bCPU\b|\bProcessor\b", "", cpu_clean, flags=re.IGNORECASE)

    # Remove extra whitespace
    cpu_clean = " ".join(cpu_clean.split())

    # Extract generic CPU information (avoid being too specific)
    # Look for general patterns like "Core", numbers, or model identifiers
    # Pattern to separately capture base number and suffix
    core_pattern = (
        r"(Core|Xeon|Atom|Pentium|Celeron)\s*"
        r"(?:(Ultra|i\d+|Pro|Max))?\s*"
        r"(\d+)(?:\s+(\w+))?(?:-(\d+\w*))?"
    )
    core_match = re.search(core_pattern, cpu_clean, re.IGNORECASE)
    if core_match:
        series = core_match.group(1).lower()
        intermediate = core_match.group(2).lower() if core_match.group(2) else None
        base_number = core_match.group(3)
        suffix = core_match.group(4).lower() if core_match.group(4) else None
        hyphen_part = core_match.group(5).lower() if core_match.group(5) else None

        # Build the model part with proper underscore separation
        model_parts = [base_number]
        if suffix:
            model_parts.append(suffix)
        if hyphen_part:
            model_parts.append(hyphen_part)

        model = "_".join(model_parts)

        if intermediate:
            cpu_clean = f"{series}_{intermediate}_{model}"
        else:
            cpu_clean = f"{series}_{model}"
    else:
        # For other patterns, apply general normalization
        cpu_clean = normalize_filename_component(cpu_clean)

    # Remove any remaining "intel" references and apply length limits
    cpu_clean = re.sub(r"\bintel\b", "", cpu_clean, flags=re.IGNORECASE)
    cpu_clean = cpu_clean.strip("_")

    # Limit length for CPU names specifically
    if len(cpu_clean) > 25:
        cpu_clean = cpu_clean[:25]

    return cpu_clean if cpu_clean else "unknown"


def normalize_filename_component(component: str, preserve_intel: bool = False) -> str:
    """
    Normalize a filename component by removing problematic characters and spaces.
    Optionally excludes "Intel" references for non-vendor components.

    Args:
        component: Original component string
        preserve_intel: If True, keep "Intel" in the string (for vendor names)

    Returns:
        str: Normalized component
    """
    if not component:
        return "unknown"

    # Remove Intel branding only if not preserving it (e.g., for vendor names)
    if not preserve_intel:
        cleaned = re.sub(r"\bIntel\b", "", component, flags=re.IGNORECASE)
    else:
        cleaned = component

    # Replace spaces and problematic characters with underscores
    cleaned = re.sub(r"[^\w\-.]", "_", cleaned)
    # Remove multiple consecutive underscores
    cleaned = re.sub(r"_+", "_", cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip("_")
    # Limit length to avoid overly long filenames
    cleaned = cleaned[:20] if len(cleaned) > 20 else cleaned

    return cleaned.lower() if cleaned else "unknown"


def normalize_intel_gpu_name(gpu_name: str) -> str:
    """
    Normalize GPU name for filename, generic enough to support any discrete GPU model.
    Excludes "Intel" prefix since reports are always for Intel products.

    Args:
        gpu_name: Original GPU name

    Returns:
        str: Normalized GPU name
    """
    if not gpu_name:
        return "unknown"

    # Remove common unnecessary parts and Intel branding
    gpu_clean = re.sub(
        r"\b(Corporation|Corp|Inc|Ltd|Limited|Intel)\b",
        "",
        gpu_name,
        flags=re.IGNORECASE,
    )
    gpu_clean = re.sub(r"\(R\)|\(TM\)|\(C\)", "", gpu_clean)
    gpu_clean = re.sub(r"\(dGPU\)|\(iGPU\)|\bGraphics\b", "", gpu_clean, flags=re.IGNORECASE)

    # Remove extra whitespace
    gpu_clean = " ".join(gpu_clean.split())

    # Extract content from brackets first (e.g., "DG2 [Arc A770]" -> "Arc A770")
    bracket_match = re.search(r"\[([^\]]+)\]", gpu_clean)
    if bracket_match:
        gpu_clean = bracket_match.group(1).strip()

    # Extract generic GPU information (flexible for any GPU model)
    # Look for common GPU patterns: Arc, Iris, Xe, UHD, HD, etc.
    # Note: Exclude "Arc" prefix for brevity (e.g., B580 instead of arc_b580)
    gpu_patterns = [
        (r"Arc\s*(?:Pro\s*)?([A-Z]?\d+\w*)", r"\1"),  # Arc B580 / Arc Pro B60 -> b580 / b60
        (r"(Iris).*?(Xe|Pro).*?(\d+\w*)?", r"\1_\2"),  # Iris Xe -> iris_xe
        (r"(UHD|HD).*?(\d+\w*)", r"\1_\2"),  # UHD Graphics 770 -> uhd_770
        (r"(Xe).*?(\w+)", r"\1_\2"),  # Xe variants -> xe_variant
    ]

    for pattern, replacement in gpu_patterns:
        match = re.search(pattern, gpu_clean, re.IGNORECASE)
        if match:
            # Apply the replacement pattern
            gpu_clean = re.sub(pattern, replacement, gpu_clean, flags=re.IGNORECASE).lower()
            break
    else:
        # If no specific pattern matches, apply general normalization
        gpu_clean = normalize_filename_component(gpu_clean)

    # Remove any remaining "intel" references and clean up
    gpu_clean = re.sub(r"\bintel\b", "", gpu_clean, flags=re.IGNORECASE)
    gpu_clean = gpu_clean.strip("_")

    # Limit length for GPU names specifically
    if len(gpu_clean) > 20:
        gpu_clean = gpu_clean[:20]

    return gpu_clean if gpu_clean else "unknown"


def _generate_short_timestamp() -> str:
    """
    Generate a short timestamp in format YYMMDD_HHMM (without seconds).

    Returns:
        str: Short timestamp string
    """
    return datetime.now().strftime("%y%m%d_%H%M")
