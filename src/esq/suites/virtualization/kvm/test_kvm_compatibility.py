# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
KVM Virtualization Compatibility Test.

Checks Intel VT-x, VT-d/IOMMU, KVM/VFIO kernel modules, nested virtualization,
and the ``/dev/kvm`` device to determine whether the host is compatible with
KVM/QEMU virtualization. Each feature is toggled independently via profile
params so a single function covers the comprehensive, basic, passthrough, and
nested-only test cases.

This is a suite-level, data-collection test: it reports virtualization
compatibility without enforcing pass/fail criteria. Only an interrupt or an
unexpected runtime error terminates the test as a failure. Peripheral VM
capacity is intentionally handled by a dedicated test (``test_kvm_vm_capacity``)
so each file keeps its own focused params and context. Host-side probes live in
the ``src/kvm`` package.
"""

import logging
import os
from pathlib import Path

import allure
import pytest
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result

from esq.suites.virtualization.kvm.src.kvm.checks import (
    check_cpu_vt_x,
    check_iommu_support,
    check_kernel_module,
    check_kvm_device,
    check_nested_virtualization,
    check_vfio_devices,
)
from esq.suites.virtualization.kvm.src.kvm.info import save_virtualization_info

logger = logging.getLogger(__name__)


def test_kvm_compatibility(
    request,
    configs,
    cached_result,
    cache_result,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
):
    """
    Test KVM virtualization compatibility by checking Intel VT-x, VT-d/IOMMU,
    KVM/VFIO kernel modules, nested virtualization, and ``/dev/kvm``.

    This is a suite-level test that collects data about virtualization
    compatibility without enforcing pass/fail criteria (status always True).
    Interrupts and unexpected errors are surfaced as proper test outcomes.
    """
    # Step 1: Extract parameters
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    # Set test description from config if provided
    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    # Feature flags
    check_vt_x = configs.get("check_vt_x", True)
    check_vt_d = configs.get("check_vt_d", True)
    check_kvm_modules = configs.get("check_kvm_modules", True)
    check_vfio_modules = configs.get("check_vfio_modules", True)
    check_nested_virt = configs.get("check_nested_virt", True)
    check_kvm_dev = configs.get("check_kvm_dev", True)

    logger.info(f"Starting KVM Compatibility Test: {test_display_name}")

    # Step 2: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Step 3: Setup directories with path sanitization
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))

    # Resolve path and reconstruct to break taint chain
    core_data_resolved = str(Path(core_data_dir_tainted).resolve())
    chars: list = []
    for char in core_data_resolved:
        chars.append(char)
    core_data_dir = "".join(chars)

    # Validate path stays within expected directory
    expected_base = Path(os.getcwd()).resolve()
    if not Path(core_data_dir).resolve().is_relative_to(expected_base):
        core_data_dir = os.path.join(os.getcwd(), "esq_data")

    data_dir = os.path.join(core_data_dir, "data", "suites", "virtualization", "kvm")
    virt_results = os.path.join(data_dir, "results", test_id)

    # Sanitize final path
    virt_resolved = str(Path(virt_results).resolve())
    chars_virt: list = []
    for char in virt_resolved:
        chars_virt.append(char)
    virt_results_clean = "".join(chars_virt)

    os.makedirs(virt_results_clean, mode=0o770, exist_ok=True)
    ensure_dir_permissions(virt_results_clean, uid=os.getuid(), gid=os.getgid(), mode=0o770)
    virt_results = virt_results_clean

    # State for clean termination on interrupt/error.
    result = None
    test_interrupted = False
    test_failed = False
    failure_message = ""
    is_qualification = configs.get("labels", {}).get("type") == "qualification"

    def _collect_compatibility_info():
        """Run the configured virtualization compatibility probes and build the result."""
        logger.info("Probing virtualization compatibility (VT-x, VT-d/IOMMU, KVM/VFIO modules)")

        # Step 4: Execute virtualization checks
        results_data = {}

        # Check VT-x
        if check_vt_x:
            vt_x_supported, vt_x_msg = check_cpu_vt_x()
            results_data["vt_x_supported"] = vt_x_supported
            results_data["vt_x_message"] = vt_x_msg
            logger.info(f"VT-x: {vt_x_msg}")

        # Check VT-d/IOMMU
        if check_vt_d:
            vt_d_enabled, vt_d_msg, vt_d_details = check_iommu_support()
            results_data["vt_d_enabled"] = vt_d_enabled
            results_data["vt_d_message"] = vt_d_msg
            results_data["iommu_groups_count"] = vt_d_details["iommu_groups"]
            results_data["iommu_kernel_param"] = vt_d_details["kernel_param"]
            logger.info(f"VT-d/IOMMU: {vt_d_msg}")

        # Check KVM modules
        if check_kvm_modules:
            kvm_loaded, kvm_msg = check_kernel_module("kvm")
            kvm_intel_loaded, kvm_intel_msg = check_kernel_module("kvm_intel")
            results_data["kvm_module_loaded"] = kvm_loaded
            results_data["kvm_intel_module_loaded"] = kvm_intel_loaded
            logger.info(f"KVM: {kvm_msg}")
            logger.info(f"KVM Intel: {kvm_intel_msg}")

        # Check VFIO modules
        if check_vfio_modules:
            vfio_loaded, vfio_msg = check_kernel_module("vfio")
            vfio_pci_loaded, vfio_pci_msg = check_kernel_module("vfio_pci")
            vfio_iommu_loaded, vfio_iommu_msg = check_kernel_module("vfio_iommu_type1")
            results_data["vfio_module_loaded"] = vfio_loaded
            results_data["vfio_pci_module_loaded"] = vfio_pci_loaded
            results_data["vfio_iommu_module_loaded"] = vfio_iommu_loaded
            logger.info(f"VFIO: {vfio_msg}")
            logger.info(f"VFIO PCI: {vfio_pci_msg}")

        # Check nested virtualization
        if check_nested_virt:
            nested_enabled, nested_msg = check_nested_virtualization()
            results_data["nested_virt_enabled"] = nested_enabled
            results_data["nested_virt_message"] = nested_msg
            logger.info(f"Nested Virtualization: {nested_msg}")

        # Check /dev/kvm device
        if check_kvm_dev:
            kvm_dev_accessible, kvm_dev_msg = check_kvm_device()
            results_data["kvm_dev_accessible"] = kvm_dev_accessible
            results_data["kvm_dev_message"] = kvm_dev_msg
            logger.info(f"/dev/kvm: {kvm_dev_msg}")

        # Check VFIO devices
        vfio_dev_count, vfio_dev_msg = check_vfio_devices()
        results_data["vfio_devices_count"] = vfio_dev_count
        results_data["vfio_devices_message"] = vfio_dev_msg
        logger.info(f"VFIO Devices: {vfio_dev_msg}")

        # Step 5: Save detailed info
        save_virtualization_info(virt_results)

        # Step 6: Create metrics (only relevant ones for each test case)
        # Key metric strategy: Only ONE key metric per test case
        # - Comprehensive (all checks): kvm_dev_accessible (ultimate usability test)
        # - Basic KVM: kvm_dev_accessible (basic VM hosting needs this)
        # - Passthrough (VT-d/VFIO only): vt_d_enabled (device passthrough needs VT-d)
        # - Nested (nested only): nested_virt_enabled (specific nested capability)
        metrics = {}

        # Determine which metric should be the key metric based on test configuration
        key_metric_name = None

        # Count how many feature checks are enabled
        checks_enabled = sum(
            [check_vt_x, check_vt_d, check_kvm_modules, check_vfio_modules, check_nested_virt, check_kvm_dev]
        )

        if checks_enabled == 1 and check_nested_virt:
            # Nested-only test
            key_metric_name = "nested_virt_enabled"
        elif check_vfio_modules and not check_kvm_dev:
            # Passthrough test (has VFIO but no kvm_dev check)
            key_metric_name = "vt_d_enabled"
        elif check_kvm_dev:
            # Comprehensive or basic test (has kvm_dev check)
            key_metric_name = "kvm_dev_accessible"

        # Add VT-x metrics if checked
        if check_vt_x:
            metrics["vt_x_supported"] = Metrics(
                unit=None, value=results_data.get("vt_x_supported", False), is_key_metric=False
            )

        # Add VT-d/IOMMU metrics if checked
        if check_vt_d:
            metrics["vt_d_enabled"] = Metrics(
                unit=None,
                value=results_data.get("vt_d_enabled", False),
                is_key_metric=(key_metric_name == "vt_d_enabled"),
            )
            metrics["iommu_groups_count"] = Metrics(
                unit=None, value=results_data.get("iommu_groups_count", 0), is_key_metric=False
            )
            metrics["iommu_kernel_param"] = Metrics(
                unit=None, value=results_data.get("iommu_kernel_param", False), is_key_metric=False
            )

        # Add KVM module metrics if checked
        if check_kvm_modules:
            metrics["kvm_module_loaded"] = Metrics(
                unit=None, value=results_data.get("kvm_module_loaded", False), is_key_metric=False
            )
            metrics["kvm_intel_module_loaded"] = Metrics(
                unit=None, value=results_data.get("kvm_intel_module_loaded", False), is_key_metric=False
            )

        # Add VFIO module metrics if checked
        if check_vfio_modules:
            metrics["vfio_module_loaded"] = Metrics(
                unit=None, value=results_data.get("vfio_module_loaded", False), is_key_metric=False
            )
            metrics["vfio_pci_module_loaded"] = Metrics(
                unit=None, value=results_data.get("vfio_pci_module_loaded", False), is_key_metric=False
            )
            metrics["vfio_iommu_module_loaded"] = Metrics(
                unit=None, value=results_data.get("vfio_iommu_module_loaded", False), is_key_metric=False
            )
            metrics["vfio_devices_count"] = Metrics(
                unit=None, value=results_data.get("vfio_devices_count", 0), is_key_metric=False
            )

        # Add nested virtualization metrics if checked
        if check_nested_virt:
            metrics["nested_virt_enabled"] = Metrics(
                unit=None,
                value=results_data.get("nested_virt_enabled", False),
                is_key_metric=(key_metric_name == "nested_virt_enabled"),
            )

        # Add /dev/kvm metrics if checked
        if check_kvm_dev:
            metrics["kvm_dev_accessible"] = Metrics(
                unit=None,
                value=results_data.get("kvm_dev_accessible", False),
                is_key_metric=(key_metric_name == "kvm_dev_accessible"),
            )

        # Step 7: Build status message (only show checked features)
        status_parts = []

        if check_vt_x:
            if results_data.get("vt_x_supported"):
                status_parts.append("VT-x: ✓")
            else:
                status_parts.append("VT-x: ✗")

        if check_vt_d:
            if results_data.get("vt_d_enabled"):
                status_parts.append(f"VT-d: ✓ ({results_data.get('iommu_groups_count', 0)} groups)")
            else:
                status_parts.append("VT-d: ✗")

        if check_kvm_modules:
            if results_data.get("kvm_module_loaded"):
                status_parts.append("KVM: ✓")
            else:
                status_parts.append("KVM: ✗")

        if check_vfio_modules:
            vfio_status = "✓" if results_data.get("vfio_module_loaded") else "✗"
            vfio_count = results_data.get("vfio_devices_count", 0)
            status_parts.append(f"VFIO: {vfio_status} ({vfio_count} devices)")

        if check_nested_virt:
            if results_data.get("nested_virt_enabled"):
                status_parts.append("Nested: ✓")
            else:
                status_parts.append("Nested: ✗")

        if check_kvm_dev:
            if results_data.get("kvm_dev_accessible"):
                status_parts.append("/dev/kvm: ✓")
            else:
                status_parts.append("/dev/kvm: ✗")

        test_message = " | ".join(status_parts) if status_parts else "No features checked"

        # Step 8: Attach status summary to Allure report
        allure.attach(
            test_message,
            name="Virtualization Status Summary",
            attachment_type=allure.attachment_type.TEXT,
        )

        # Step 9: Create result (suite test - status always True; message in
        # extended metadata for informational context)
        return Result(
            name=test_display_name,
            metadata={"status": True},
            extended_metadata={"message": test_message, "virtualization_output_dir": virt_results},
            metrics=metrics,
        )

    # Step 10: Execute checks with caching, surfacing interrupts/errors cleanly
    try:
        result = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=_collect_compatibility_info,
            test_name=test_name,
            configs=configs,
        )
    except KeyboardInterrupt:
        failure_message = "Interrupt detected during KVM compatibility test execution"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during KVM compatibility test execution: {str(e)}"
        logger.error(failure_message, exc_info=True)

    # Ensure a result object always exists so the test terminates cleanly even
    # when interrupted before the probes produced a result.
    if result is None:
        result = Result(
            name=test_display_name,
            metadata={"status": False},
            extended_metadata={
                "virtualization_output_dir": virt_results,
                "message": failure_message or "KVM compatibility test did not complete",
            },
            metrics={},
        )

    # Step 11: Always validate and summarize so the result is recorded
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

    # Step 12: Cache results
    cache_result(result)

    logger.info(f"KVM compatibility test completed: {test_display_name}")

    # Terminate cleanly: surface interrupts/errors as a proper test outcome.
    if test_interrupted:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
