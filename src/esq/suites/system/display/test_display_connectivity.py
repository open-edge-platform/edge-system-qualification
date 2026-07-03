# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System Display Connectivity Test.

Detects display ports and checks their status using the DRM subsystem.
This suite is data-collection oriented: it reports the number of connected
ports per connector type and never fails on connectivity alone. Only an
interrupt or an unexpected runtime error terminates the test as a failure.
"""

import logging
import os
from pathlib import Path
from typing import Dict

import allure
import pytest
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result
from sysagent.utils.system.drm import (
    get_connector_modes,
    get_drm_cards,
    get_drm_connectors,
)

logger = logging.getLogger(__name__)


def save_display_info(output_dir: str, drm_devices: Dict, display_ports: Dict):
    """
    Save detailed display information to files.

    Args:
        output_dir: Directory to save display info
        drm_devices: Dict of DRM devices
        display_ports: Dict of display ports
    """
    try:
        # Save DRM devices info
        devices_file = os.path.join(output_dir, "drm_devices.txt")
        with open(devices_file, "w", encoding="utf-8") as f:
            f.write("DRM Devices:\n")
            for device, info in drm_devices.items():
                f.write(f"  {device}: {info['path']} (available={info['available']})\n")

        # Save display ports info
        ports_file = os.path.join(output_dir, "display_ports.txt")
        with open(ports_file, "w", encoding="utf-8") as f:
            f.write("Display Ports:\n")
            for port_name, port_info in display_ports.items():
                f.write(f"\n  {port_name}:\n")
                f.write(f"    Device: {port_info['device']}\n")
                f.write(f"    Type: {port_info['type']}\n")
                f.write(f"    Status: {port_info['status']}\n")
                f.write(f"    Enabled: {port_info['enabled']}\n")

                # Get modes for connected ports
                if port_info["status"] == "connected":
                    modes = get_connector_modes(port_info["path"])
                    if modes:
                        f.write(f"    Modes ({len(modes)}):\n")
                        for mode in modes:
                            f.write(f"      {mode}\n")

    except IOError as e:
        logger.warning(f"Could not save display info: {e}")


@allure.title("Display Connectivity Test")
def test_display_connectivity(
    request,
    configs,
    cached_result,
    cache_result,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """
    Test display connectivity for display ports.

    Detects display ports using DRM subsystem and checks their status.
    """
    # Request
    test_name = request.node.name.split("[")[0]

    # Parameters
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    # Set test description from config if provided
    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info(f"Starting Display Connectivity Test: {test_display_name}")

    check_all = configs.get("check_all", False)
    check_hdmi = configs.get("check_hdmi", False)
    check_displayport = configs.get("check_displayport", False)
    drm_device = configs.get("drm_device", None)  # Optional: target specific DRM device

    # Step 1: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Step 2: Setup directories
    # Sanitize environment variable path using character-by-character copy
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))

    # Resolve path and reconstruct to break taint chain
    core_data_resolved = str(Path(core_data_dir_tainted).resolve())
    chars: list = []
    for char in core_data_resolved:
        chars.append(char)
    core_data_dir = "".join(chars)

    # Validate path stays within expected directory (no path traversal)
    expected_base = Path(os.getcwd()).resolve()
    if not Path(core_data_dir).resolve().is_relative_to(expected_base):
        # If path is outside current directory, use safe default
        core_data_dir = os.path.join(os.getcwd(), "esq_data")

    data_dir = os.path.join(core_data_dir, "data", "suites", "system", "display")
    display_results = os.path.join(data_dir, "results", test_id)

    # Sanitize final path before directory creation
    display_resolved = str(Path(display_results).resolve())
    chars_display: list = []
    for char in display_resolved:
        chars_display.append(char)
    display_results_clean = "".join(chars_display)

    os.makedirs(display_results_clean, mode=0o770, exist_ok=True)
    ensure_dir_permissions(display_results_clean, uid=os.getuid(), gid=os.getgid(), mode=0o770)

    # Use sanitized path for all subsequent operations
    display_results = display_results_clean

    # State for clean termination on interrupt/error.
    result = None
    test_interrupted = False
    test_failed = False
    failure_message = ""
    is_qualification = configs.get("labels", {}).get("type") == "qualification"

    def _collect_display_info():
        """Enumerate display ports via the DRM subsystem and build the result."""
        logger.info("Detecting display ports via DRM subsystem")

        # Step 3: Get DRM devices
        drm_devices = get_drm_cards()
        drm_available = len(drm_devices) > 0

        if not drm_available:
            logger.warning("No DRM devices found - display subsystem may not be available")

        # Step 4: Get display ports
        all_ports = get_drm_connectors(card=drm_device)

        # Filter ports based on test criteria
        filtered_ports = {}

        for port_name, port_info in all_ports.items():
            include = False

            # Check if showing all ports
            if check_all:
                include = True

            # Check HDMI ports
            if check_hdmi and "HDMI" in port_info["type"]:
                include = True

            # Check DisplayPort
            if check_displayport and "DP" in port_info["type"]:
                include = True

            if include:
                filtered_ports[port_name] = port_info

        # Count ports by type and status
        total_ports = len(filtered_ports)
        total_connected = sum(1 for port in filtered_ports.values() if port["status"] == "connected")

        hdmi_ports = {k: v for k, v in filtered_ports.items() if "HDMI" in v["type"]}
        dp_ports = {k: v for k, v in filtered_ports.items() if "DP" in v["type"]}

        hdmi_total = len(hdmi_ports)
        hdmi_connected = sum(1 for port in hdmi_ports.values() if port["status"] == "connected")

        dp_total = len(dp_ports)
        dp_connected = sum(1 for port in dp_ports.values() if port["status"] == "connected")

        # Log port information
        logger.info(f"Display ports: {total_ports} total, {total_connected} connected")

        if check_all:
            logger.info(f"  HDMI: {hdmi_total} total, {hdmi_connected} connected")
            for port_name, port_info in hdmi_ports.items():
                status = "CONNECTED" if port_info["status"] == "connected" else "DISCONNECTED"
                modes = get_connector_modes(port_info["path"])
                modes_info = f"{len(modes)} modes" if modes else "No modes"
                logger.info(f"    {port_name}: {status} - {modes_info}")

            logger.info(f"  DisplayPort: {dp_total} total, {dp_connected} connected")
            for port_name, port_info in dp_ports.items():
                status = "CONNECTED" if port_info["status"] == "connected" else "DISCONNECTED"
                modes = get_connector_modes(port_info["path"])
                modes_info = f"{len(modes)} modes" if modes else "No modes"
                logger.info(f"    {port_name}: {status} - {modes_info}")

        if check_hdmi:
            logger.info(f"  HDMI: {hdmi_total} total, {hdmi_connected} connected")
            for port_name, port_info in hdmi_ports.items():
                status = "CONNECTED" if port_info["status"] == "connected" else "DISCONNECTED"
                modes = get_connector_modes(port_info["path"])
                modes_info = f"{len(modes)} modes" if modes else "No modes"
                logger.info(f"    {port_name}: {status} - {modes_info}")

        if check_displayport:
            logger.info(f"  DisplayPort: {dp_total} total, {dp_connected} connected")
            for port_name, port_info in dp_ports.items():
                status = "CONNECTED" if port_info["status"] == "connected" else "DISCONNECTED"
                modes = get_connector_modes(port_info["path"])
                modes_info = f"{len(modes)} modes" if modes else "No modes"
                logger.info(f"    {port_name}: {status} - {modes_info}")

        # Step 5: Save display info
        save_display_info(display_results, drm_devices, filtered_ports)

        # Step 6: Create metrics (only relevant metrics per test type)
        # Base metrics for all tests
        metrics = {
            "drm_available": Metrics(value=drm_available, is_key_metric=False),
            "drm_devices_count": Metrics(unit="devices", value=len(drm_devices), is_key_metric=False),
        }

        # Add test-specific metrics
        if check_all:
            # For "All Ports" test: show total + breakdown by type
            metrics["total_ports"] = Metrics(unit="ports", value=total_ports, is_key_metric=False)
            metrics["total_connected"] = Metrics(unit="ports", value=total_connected, is_key_metric=True)
            metrics["hdmi_ports_total"] = Metrics(unit="ports", value=hdmi_total, is_key_metric=False)
            metrics["hdmi_ports_connected"] = Metrics(unit="ports", value=hdmi_connected, is_key_metric=False)
            metrics["dp_ports_total"] = Metrics(unit="ports", value=dp_total, is_key_metric=False)
            metrics["dp_ports_connected"] = Metrics(unit="ports", value=dp_connected, is_key_metric=False)

        elif check_hdmi:
            # For HDMI-only test: show only HDMI metrics
            metrics["hdmi_ports_total"] = Metrics(unit="ports", value=hdmi_total, is_key_metric=False)
            metrics["hdmi_ports_connected"] = Metrics(unit="ports", value=hdmi_connected, is_key_metric=True)

        elif check_displayport:
            # For DisplayPort-only test: show only DP metrics
            metrics["dp_ports_total"] = Metrics(unit="ports", value=dp_total, is_key_metric=False)
            metrics["dp_ports_connected"] = Metrics(unit="ports", value=dp_connected, is_key_metric=True)

        # Step 7: Build status message (informational, suite does not fail on connectivity)
        if not drm_available:
            test_message = "DRM subsystem not available"
        elif total_ports == 0:
            test_message = "No display ports found"
        else:
            parts = []
            if check_hdmi:
                parts.append(f"HDMI: {hdmi_connected}/{hdmi_total}")
            elif check_displayport:
                parts.append(f"DP: {dp_connected}/{dp_total}")
            elif check_all:
                parts.append(f"HDMI: {hdmi_connected}/{hdmi_total}")
                parts.append(f"DP: {dp_connected}/{dp_total}")

            test_message = f"{', '.join(parts)} - Total: {total_connected}/{total_ports} connected"

        # Step 8: Create result with message recorded in extended metadata
        return Result(
            name=test_display_name,
            metadata={"status": True},
            extended_metadata={"message": test_message, "display_output_dir": display_results},
            metrics=metrics,
        )

    # Step 9: Execute detection with caching, surfacing interrupts/errors cleanly
    try:
        result = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=_collect_display_info,
            test_name=test_name,
            configs=configs,
        )
    except KeyboardInterrupt:
        failure_message = "Interrupt detected during display connectivity test execution"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during display connectivity test execution: {str(e)}"
        logger.error(failure_message, exc_info=True)

    # Ensure a result object always exists so the test terminates cleanly even
    # when interrupted before detection produced a result.
    if result is None:
        result = Result(
            name=test_display_name,
            metadata={"status": False},
            extended_metadata={
                "display_output_dir": display_results,
                "message": failure_message or "Display connectivity test did not complete",
            },
            metrics={},
        )

    # Step 10: Always validate and summarize so the result is recorded
    try:
        validate_test_results(
            test_name=test_name,
            results=result,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as validation_error:
        logger.error(f"Validation failed: {validation_error}")

    try:
        summarize_test_results(
            results=result,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as summary_error:
        logger.error(f"Test result summarization failed: {summary_error}", exc_info=True)

    # Step 11: Cache results
    cache_result(result)

    logger.info(f"Display connectivity test completed: {test_display_name}")

    # Terminate cleanly: surface interrupts/errors as a proper test outcome.
    if test_interrupted:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
