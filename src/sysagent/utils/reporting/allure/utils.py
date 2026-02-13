# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Allure utility functions.

This module contains miscellaneous utility functions for Allure reporting,
including title updates and other helper functions.
"""

import logging
import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def update_allure_title_with_metrics():
    """
    Update the Allure title to include result metric values.
    Usage:
        update_allure_title_with_metrics
        (configs, results, metrics=["latency", "throughput"])
    """
    import allure

    def _update(configs: Dict[str, Any], results: Any, metrics: Optional[List[str]] = None) -> None:
        """
        Update Allure test title with metric information.

        Args:
            configs: Test configuration dictionary
            results: Test results object with metrics attribute
            metrics: List of metric names to include in title
        """
        test_id = configs.get("test_id", "T0000")
        display_name = configs.get("display_name", configs.get("name", "Test"))
        metrics = metrics or list(results.metrics.keys())
        metric_strs = []

        for metric in metrics:
            metric_data = results.metrics.get(metric)
            if metric_data and hasattr(metric_data, "value"):
                value = metric_data.value
                unit = metric_data.unit or ""
                metric_strs.append(f"{metric}: {value}{(' ' + unit) if unit else ''}")

        if metric_strs:
            new_title = f"{test_id} - {display_name} | " + ", ".join(metric_strs)
        else:
            new_title = f"{test_id} - {display_name}"

        allure.dynamic.title(new_title)

    return _update


def _generate_final_report_copy(report_dir: str, debug: bool = False) -> str:
    """
    Generate a timestamped final copy of the Allure report.
    Format: <appname>_report_<system_and_productname>_<normalize_cpu_brand>
    _<list_of_discrete_gpus_if_available>_<timestamp>.html

    Args:
        report_dir: Directory containing the generated Allure report
        debug: Whether to show debug level logs

    Returns:
        str: Path to the final report file, or None if generation failed
    """
    try:
        # Check if the standard report exists
        source_report = os.path.join(report_dir, "index.html")
        if not os.path.exists(source_report):
            logger.error(f"Source report not found: {source_report}")
            return None

        # Clean up old final reports first to keep only the latest
        _cleanup_old_final_reports(report_dir, debug)

        # Generate components for filename
        app_name = _get_app_name()
        timestamp = _generate_short_timestamp()
        system_info = _get_comprehensive_system_info_for_filename()

        logger.debug(f"System info for filename: {system_info}")
        logger.debug(f"App name for filename: {app_name}")
        logger.debug(f"Timestamp for filename: {timestamp}")

        # Construct final filename with format:
        # <appname>_report_<system_and_productname>_<cpu_brand>_<discrete_gpus>
        # _<timestamp>.html
        filename_parts = [f"{app_name}_report"]

        # Add system and product information
        if system_info.get("vendor") and system_info.get("product"):
            system_product = f"{system_info['vendor']}_{system_info['product']}"
            filename_parts.append(system_product)
        elif system_info.get("vendor"):
            filename_parts.append(system_info["vendor"])
        elif system_info.get("product"):
            filename_parts.append(system_info["product"])

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
        final_filename = final_filename.lower()

        final_report_path = os.path.join(report_dir, final_filename)

        # Copy the report file
        shutil.copy2(source_report, final_report_path)

        logger.debug(f"Final report copy created: {final_report_path}")
        return final_report_path

    except Exception as e:
        logger.error(f"Failed to generate final report copy: {e}")
        if debug:
            logger.error(f"Error details: {e}", exc_info=True)
        return None


def _get_app_name() -> str:
    """
    Get the application name for filename prefix.

    Returns:
        str: Application name or 'app' as fallback
    """
    try:
        from sysagent.utils.config import get_dist_name

        app_name = get_dist_name()
        if app_name:
            # Clean up app name for filename (remove problematic characters)
            cleaned = re.sub(r"[^\w\-.]", "_", app_name)
            cleaned = re.sub(r"_+", "_", cleaned)
            cleaned = cleaned.strip("_")
            return cleaned or "app"
        return "app"
    except Exception:
        return "app"


def _generate_short_timestamp() -> str:
    """
    Generate a short timestamp in format YYMMDD_HHMM (without seconds).

    Returns:
        str: Short timestamp string
    """
    now = datetime.now()
    return now.strftime("%y%m%d_%H%M")


def _get_comprehensive_system_info_for_filename() -> dict:
    """
    Get comprehensive system information for filename generation.

    Returns:
        dict: System information dictionary with normalized values
    """
    try:
        from sysagent.utils.config import setup_data_dir
        from sysagent.utils.system import SystemInfoCache

        data_dir = setup_data_dir()
        cache_dir = os.path.join(data_dir, "cache")
        system_cache = SystemInfoCache(cache_dir)

        # Get system information
        hardware_info = system_cache.get_hardware_info()
        system_info = hardware_info.get("system", {})
        cpu_info = hardware_info.get("cpu", {})
        gpu_info = hardware_info.get("gpu", {})

        result = {}

        # Get and normalize system vendor
        vendor = system_info.get("vendor", "").strip()
        if vendor:
            result["vendor"] = _normalize_filename_component(vendor)

        # Get and normalize product name
        product = system_info.get("product", "").strip()
        if product:
            result["product"] = _normalize_filename_component(product)

        # Get and normalize CPU brand
        cpu_brand = cpu_info.get("brand", "").strip()
        if cpu_brand:
            result["cpu_brand"] = _normalize_cpu_brand(cpu_brand)

        # Get discrete GPUs
        discrete_gpus = []
        if isinstance(gpu_info, list):
            for gpu in gpu_info:
                if gpu.get("discrete", False):
                    gpu_name = gpu.get("name", "").strip()
                    if gpu_name:
                        normalized_gpu = _normalize_intel_gpu_name(gpu_name)
                        if normalized_gpu not in discrete_gpus:
                            discrete_gpus.append(normalized_gpu)

        if discrete_gpus:
            result["discrete_gpus"] = discrete_gpus

        return result

    except Exception as e:
        logger.debug(f"Error getting system info for filename: {e}")
        return {}


def _cleanup_old_final_reports(report_dir: str, debug: bool = False) -> None:
    """
    Clean up old final report copies, keeping only the most recent one.

    Args:
        report_dir: Directory containing Allure reports
        debug: Whether to show debug level logs
    """
    try:
        if not os.path.exists(report_dir):
            return

        # Find all final report files (those with timestamps)
        final_reports = []
        for filename in os.listdir(report_dir):
            if filename.endswith(".html") and filename != "index.html":
                # Check if it looks like a final report (has timestamp pattern)
                if re.search(r"_\d{6}_\d{4}\.html$", filename):
                    filepath = os.path.join(report_dir, filename)
                    final_reports.append((filepath, os.path.getmtime(filepath)))

        # Sort by modification time (newest first)
        final_reports.sort(key=lambda x: x[1], reverse=True)

        # Keep only the 3 most recent final reports, remove the rest
        for filepath, _ in final_reports[3:]:
            try:
                os.remove(filepath)
                if debug:
                    logger.debug(f"Removed old final report: {os.path.basename(filepath)}")
            except OSError as e:
                if debug:
                    logger.debug(f"Could not remove old final report {filepath}: {e}")

    except Exception as e:
        if debug:
            logger.debug(f"Error cleaning up old final reports: {e}")


def _normalize_cpu_brand(cpu_brand: str) -> str:
    """
    Normalize CPU brand name for filename use.

    Args:
        cpu_brand: Raw CPU brand string

    Returns:
        str: Normalized CPU brand string
    """
    if not cpu_brand:
        return "unknown_cpu"

    # Convert to lowercase for consistent processing
    brand = cpu_brand.lower().strip()

    # Intel processors
    if "intel" in brand:
        # Extract Intel processor family and model
        if "core" in brand:
            # Look for patterns like "i3", "i5", "i7", "i9"
            core_match = re.search(r"core\s*i(\d)", brand)
            if core_match:
                core_num = core_match.group(1)

                # Look for generation indicators
                gen_patterns = [
                    (r"(\d+)th\s+gen", lambda m: f"gen{m.group(1)}"),
                    (
                        r"(\d{4,5})[a-z]*",
                        lambda m: f"gen{str(m.group(1))[0:2]}" if len(m.group(1)) >= 4 else f"gen{m.group(1)[0]}",
                    ),
                ]

                generation = ""
                for pattern, formatter in gen_patterns:
                    gen_match = re.search(pattern, brand)
                    if gen_match:
                        generation = formatter(gen_match)
                        break

                if generation:
                    return f"intel_i{core_num}_{generation}"
                else:
                    return f"intel_i{core_num}"
            else:
                return "intel_core"
        elif "xeon" in brand:
            return "intel_xeon"
        elif "atom" in brand:
            return "intel_atom"
        elif "celeron" in brand:
            return "intel_celeron"
        elif "pentium" in brand:
            return "intel_pentium"
        else:
            return "intel"

    # AMD processors
    elif "amd" in brand:
        if "ryzen" in brand:
            # Look for Ryzen series
            ryzen_match = re.search(r"ryzen\s*(\d+)", brand)
            if ryzen_match:
                series = ryzen_match.group(1)
                return f"amd_ryzen{series}"
            else:
                return "amd_ryzen"
        elif "epyc" in brand:
            return "amd_epyc"
        elif "threadripper" in brand:
            return "amd_threadripper"
        elif "athlon" in brand:
            return "amd_athlon"
        else:
            return "amd"

    # ARM processors
    elif any(arm in brand for arm in ["arm", "aarch64", "cortex"]):
        return "arm"

    # Fallback: clean up the brand name
    else:
        return _normalize_filename_component(brand) or "unknown_cpu"


def _normalize_filename_component(component: str) -> str:
    """
    Normalize a filename component by removing/replacing problematic characters.

    Args:
        component: Raw component string

    Returns:
        str: Normalized component string safe for filenames
    """
    if not component:
        return ""

    # Convert to lowercase and strip whitespace
    normalized = component.lower().strip()

    # Replace problematic characters with underscores
    # Keep only alphanumeric, hyphens, and dots
    normalized = re.sub(r"[^\w\-.]", "_", normalized)

    # Replace multiple consecutive underscores with a single underscore
    normalized = re.sub(r"_+", "_", normalized)

    # Remove leading/trailing underscores
    normalized = normalized.strip("_")

    # Limit length to reasonable size for filenames
    if len(normalized) > 50:
        normalized = normalized[:50].rstrip("_")

    return normalized


def _normalize_intel_gpu_name(gpu_name: str) -> str:
    """
    Normalize Intel GPU name for filename use.

    Args:
        gpu_name: Raw GPU name string

    Returns:
        str: Normalized Intel GPU name string
    """
    if not gpu_name:
        return "unknown_gpu"

    # Convert to lowercase for consistent processing
    name = gpu_name.lower().strip()

    # Intel GPU patterns
    if "intel" in name:
        # Intel Arc series (exclude "Arc" prefix for brevity)
        arc_match = re.search(r"arc\s*([a-z]?)(\d+)", name)
        if arc_match:
            series = arc_match.group(1) or ""
            model = arc_match.group(2)
            return f"{series}{model}"  # Just model ID (e.g., b580, a770)

        # Intel Iris series
        if "iris" in name:
            iris_match = re.search(r"iris\s*(xe|plus|pro)?\s*(\d+)?", name)
            if iris_match:
                variant = iris_match.group(1) or ""
                model = iris_match.group(2) or ""
                return f"intel_iris{variant}{model}".replace(" ", "")
            else:
                return "intel_iris"

        # Intel UHD/HD Graphics
        uhd_match = re.search(r"(uhd|hd)\s*graphics\s*(\d+)?", name)
        if uhd_match:
            graphics_type = uhd_match.group(1)
            model = uhd_match.group(2) or ""
            return f"intel_{graphics_type}{model}"

        # Generic Intel graphics
        if "graphics" in name:
            return "intel_graphics"

        return "intel_gpu"

    # For non-Intel GPUs, use generic normalization
    return _normalize_filename_component(gpu_name) or "unknown_gpu"
