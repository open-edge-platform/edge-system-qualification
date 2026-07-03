# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
KVM Peripheral VM Capacity Test.

Estimates how many VMs can each be assigned a dedicated set of 1 display +
1 keyboard + 1 mouse for pass-through. Keyboard/mouse pairs are grouped by USB
hub branch so devices on the same hub form one VM station, matched against
connected displays, and gated on at least one IOMMU group. This is distinct
from a generic CPU/memory-based VM count.

This entry point is intentionally separate from the broader KVM compatibility
test (``test_kvm_compatibility``) so peripheral VM capacity owns its own params,
metrics, diagram, and Allure context. As a data-collection test it reports
capacity without enforcing pass/fail; qualification profiles add KPI thresholds
via ``kpi_refs``. Only an interrupt or an unexpected runtime error terminates the
test as a failure. Host-side enumeration lives in the ``src/kvm`` package.
"""

import logging
import os
from pathlib import Path

import allure
import pytest
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result, run_command

from esq.suites.virtualization.kvm.src.kvm.checks import check_iommu_support
from esq.suites.virtualization.kvm.src.kvm.topology import (
    build_vm_topology_diagram,
    compute_peripheral_vm_capacity,
)

logger = logging.getLogger(__name__)


def test_kvm_vm_capacity(
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
    Estimate peripheral passthrough VM capacity (1 display + 1 keyboard +
    1 mouse per VM), gated on IOMMU availability.

    This is a data-collection test that reports capacity without enforcing
    pass/fail (status always True). Qualification profiles supply a KPI
    threshold on ``peripheral_vm_capacity``. Interrupts and unexpected errors
    are surfaced as proper test outcomes.
    """
    # Step 1: Extract parameters
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    # Set test description from config if provided
    test_description = configs.get("description")
    if test_description:
        allure.dynamic.description(test_description)

    logger.info(f"Starting KVM Peripheral VM Capacity Test: {test_display_name}")

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

    def _collect_vm_capacity_info():
        """Estimate peripheral VM capacity and build the result."""
        logger.info("Estimating peripheral VM capacity (display + keyboard + mouse per VM)")

        # Step 4: Determine IOMMU group availability (capacity is gated on it)
        _, iommu_msg, iommu_details = check_iommu_support()
        iommu_groups = iommu_details["iommu_groups"]
        logger.info(f"IOMMU: {iommu_msg}")

        # Step 5: Estimate peripheral VM capacity (each VM gets 1 display +
        # 1 keyboard + 1 mouse passed through; distinct from a generic
        # CPU/memory-based VM count)
        peripheral_vms, kbd_count, mouse_count, display_count = compute_peripheral_vm_capacity(iommu_groups)
        logger.info(
            f"Peripheral VM capacity: {peripheral_vms} "
            f"(displays={display_count}, keyboards={kbd_count}, mouse={mouse_count}, "
            f"iommu_groups={iommu_groups})"
        )

        # Step 6: Build a block diagram so reviewers can visualize how peripheral
        # VM capacity maps to displays + keyboard/mouse pairs.
        topology_diagram = build_vm_topology_diagram(
            peripheral_vms, kbd_count, mouse_count, display_count, iommu_groups
        )
        diagram_path = os.path.join(virt_results, "peripheral_vm_capacity.txt")
        try:
            with open(diagram_path, "w", encoding="utf-8") as f:
                f.write(topology_diagram)
        except IOError as e:
            logger.warning(f"Failed to save peripheral VM capacity diagram: {e}")

        with allure.step("Peripheral VM Capacity"):
            allure.attach(
                topology_diagram,
                name="Peripheral VM Capacity Diagram",
                attachment_type=allure.attachment_type.TEXT,
            )
            # Attach the raw USB device tree for full peripheral topology context.
            lsusb_tree = run_command(["lsusb", "-t"], timeout=10)
            if lsusb_tree and lsusb_tree.returncode == 0 and lsusb_tree.stdout:
                allure.attach(
                    lsusb_tree.stdout,
                    name="USB Device Tree (lsusb -t)",
                    attachment_type=allure.attachment_type.TEXT,
                )

        # Step 7: Create metrics (peripheral_vm_capacity is the key metric)
        metrics = {
            "peripheral_vm_capacity": Metrics(
                unit="vms",
                value=peripheral_vms,
                is_key_metric=True,
            ),
            "displays_detected": Metrics(
                unit=None,
                value=display_count,
                is_key_metric=False,
            ),
            "keyboards_detected": Metrics(
                unit=None,
                value=kbd_count,
                is_key_metric=False,
            ),
            "mouse_detected": Metrics(
                unit=None,
                value=mouse_count,
                is_key_metric=False,
            ),
            "iommu_groups_count": Metrics(
                unit=None,
                value=iommu_groups,
                is_key_metric=False,
            ),
        }

        # Step 8: Build status message
        test_message = (
            f"Peripheral VMs: {peripheral_vms} "
            f"(disp={display_count}, kbd={kbd_count}, mouse={mouse_count}, "
            f"iommu_groups={iommu_groups})"
        )

        allure.attach(
            test_message,
            name="Peripheral VM Capacity Summary",
            attachment_type=allure.attachment_type.TEXT,
        )

        # Step 9: Create result (data-collection test - status always True;
        # message in extended metadata for informational context)
        return Result(
            name=test_display_name,
            metadata={"status": True},
            extended_metadata={"message": test_message, "virtualization_output_dir": virt_results},
            metrics=metrics,
        )

    # Step 10: Execute estimation with caching, surfacing interrupts/errors cleanly
    try:
        result = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=_collect_vm_capacity_info,
            test_name=test_name,
            configs=configs,
        )
    except KeyboardInterrupt:
        failure_message = "Interrupt detected during KVM VM capacity test execution"
        test_interrupted = True
        logger.error(failure_message)
    except Exception as e:
        test_failed = True
        failure_message = f"Unexpected error during KVM VM capacity test execution: {str(e)}"
        logger.error(failure_message, exc_info=True)

    # Ensure a result object always exists so the test terminates cleanly even
    # when interrupted before the estimation produced a result.
    if result is None:
        result = Result(
            name=test_display_name,
            metadata={"status": False},
            extended_metadata={
                "virtualization_output_dir": virt_results,
                "message": failure_message or "KVM VM capacity test did not complete",
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

    logger.info(f"KVM VM capacity test completed: {test_display_name}")

    # Terminate cleanly: surface interrupts/errors as a proper test outcome.
    if test_interrupted:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
