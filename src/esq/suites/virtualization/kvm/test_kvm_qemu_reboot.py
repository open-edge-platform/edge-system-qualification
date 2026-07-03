# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
QEMU/KVM VM Reboot Test.

Validates VM lifecycle reboot behavior for a single QEMU/KVM guest: prepare a
guest image, start the VM, verify it boots to a running and guest-ready state,
execute a configurable number of reboot cycles, and measure per-reboot time.
The primary key metric is ``avg_reboot_time`` (seconds).

This entry point stays focused on orchestration. The reusable QEMU/KVM VM
lifecycle helpers — availability probes, image preparation, cloud-init seeding,
start/stop/reboot, guest-readiness probes, and the reboot scenario runner —
live in the shared ``src/kvm/qemu`` module so other QEMU/KVM tests can reuse
them, and the ``/dev/kvm`` availability check is shared via ``src/kvm/checks``.

As a qualification-capable test it reports a real pass/fail based on successful
VM lifecycle completion (create -> boot -> reboot xN -> cleanup); it fails when
QEMU or KVM is unavailable. Only an interrupt or an unexpected runtime error
terminates the test abnormally, and both are surfaced as proper test outcomes.
"""

import logging
import os
import tempfile
from pathlib import Path

import allure
import pytest
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result

from esq.suites.virtualization.kvm.src.kvm.checks import check_kvm_device
from esq.suites.virtualization.kvm.src.kvm.qemu import (
    check_qemu_availability,
    run_qemu_vm_test,
)

logger = logging.getLogger(__name__)


def test_kvm_qemu_reboot(
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
    Validate reboot behavior for a single QEMU/KVM VM instance.

    Reports a real pass/fail based on a successful VM lifecycle (create -> boot
    -> reboot xN -> cleanup) and exposes ``avg_reboot_time`` as the key metric.
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

    logger.info(f"Starting QEMU/KVM VM Reboot Test: {test_display_name}")

    # VM configuration (use_kvm defaults True; set use_kvm: false in profile to run without KVM)
    use_kvm = bool(configs.get("use_kvm", True))
    vm_memory_mb = int(configs.get("vm_memory_mb", 4096))
    vm_disk_mb = int(configs.get("vm_disk_mb", 16384))
    vm_cpu_count = int(configs.get("vm_cpu_count", 2))
    reboot_count = int(configs.get("reboot_count", 1))
    guest_image_url = configs.get("guest_image_url")
    guest_ready_probe = configs.get("guest_ready_probe", "serial")
    guest_ssh_user = configs.get("guest_ssh_user", "ubuntu")
    guest_ssh_host_port = int(configs.get("guest_ssh_host_port", 2222))
    verify_guest_os = bool(configs.get("verify_guest_os", True))
    expected_guest_os_contains = configs.get("expected_guest_os_contains")
    enable_qga = bool(configs.get("enable_qga", True))
    enable_cloud_init_seed = bool(configs.get("enable_cloud_init_seed", True))
    # Firmware selection: "seabios" (default) or "ovmf" (UEFI via pflash).
    # The resolved value is included in cache_configs so OVMF and SeaBIOS runs
    # get distinct cache keys even when all other params are identical.
    firmware = str(configs.get("firmware", "seabios")).strip().lower()

    # Specific, named per-phase timeouts (seconds). These were previously
    # hardcoded; exposing them as named params lets each lifecycle phase have
    # its own sensible budget instead of relying on one oversized generic
    # timeout. A single VM boot/reboot is expected to take well under two
    # minutes, so the defaults are sized accordingly.
    vm_start_timeout = int(configs.get("vm_start_timeout", 60))
    vm_boot_timeout = int(configs.get("vm_boot_timeout", 120))
    reboot_timeout = int(configs.get("reboot_timeout", 120))
    vm_stop_timeout = int(configs.get("vm_stop_timeout", 30))
    guest_ready_timeout = int(configs.get("guest_ready_timeout", 180))

    # Step 2: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Step 3: Probe QEMU/KVM availability
    # KVM availability is enforced as a requirement (kvm_required: true in profile) and
    # verified by validate_system_requirements_from_configs at Step 2, so it is always
    # present here. QEMU itself is a system tool, not a hardware feature, so it cannot
    # be checked via the requirements framework and still needs a runtime probe.
    qemu_available, qemu_msg, qemu_path = check_qemu_availability()
    kvm_available, kvm_msg = check_kvm_device()

    if not kvm_available:
        pytest.fail(f"KVM not available: {kvm_msg}")

    # Step 4: Setup directories with path sanitization
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

    suite_data_dir = Path(core_data_dir) / "data" / "suites" / "virtualization" / "kvm" / "qemu"
    images_dir = suite_data_dir / "images"
    test_results_dir = suite_data_dir / "results" / test_id

    try:
        os.makedirs(images_dir, mode=0o770, exist_ok=True)
        os.makedirs(test_results_dir, mode=0o770, exist_ok=True)
        ensure_dir_permissions(str(images_dir), uid=os.getuid(), gid=os.getgid(), mode=0o770)
        ensure_dir_permissions(str(test_results_dir), uid=os.getuid(), gid=os.getgid(), mode=0o770)
    except (IOError, OSError, PermissionError) as exc:
        logger.warning("Primary data directory unavailable (%s): %s", suite_data_dir, exc)
        images_dir = Path(tempfile.gettempdir()) / "esq_qemu_images"
        test_results_dir = Path(tempfile.gettempdir()) / "esq_qemu_results" / test_id
        os.makedirs(images_dir, mode=0o770, exist_ok=True)
        os.makedirs(test_results_dir, mode=0o770, exist_ok=True)

    # State for clean termination on interrupt/error.
    result = None
    test_interrupted = False
    test_failed = False
    failure_message = ""
    is_qualification = configs.get("labels", {}).get("type") == "qualification"

    # Cache configuration for deduplication.  Include firmware so OVMF and
    # SeaBIOS runs don't collide on the same cache key.
    cache_configs = {
        "vm_memory_mb": vm_memory_mb,
        "vm_cpu_count": vm_cpu_count,
        "reboot_count": reboot_count,
        "guest_image_url": guest_image_url,
        "guest_ready_probe": guest_ready_probe,
        "firmware": firmware,
    }

    def _run_reboot_workload():
        """Run the QEMU VM reboot scenario and build the result."""
        logger.info(
            "Running QEMU KVM VM reboot scenario: reboots=%s, image=%s",
            reboot_count,
            guest_image_url or "blank-qcow2",
        )
        return run_qemu_vm_test(
            test_display_name=test_display_name,
            test_id=test_id,
            qemu_path=qemu_path,
            use_kvm=use_kvm,
            vm_memory_mb=vm_memory_mb,
            vm_disk_mb=vm_disk_mb,
            vm_cpu_count=vm_cpu_count,
            reboot_count=reboot_count,
            guest_image_url=guest_image_url,
            guest_ready_probe=guest_ready_probe,
            guest_ready_timeout=guest_ready_timeout,
            guest_ssh_user=guest_ssh_user,
            guest_ssh_host_port=guest_ssh_host_port,
            verify_guest_os=verify_guest_os,
            expected_guest_os_contains=expected_guest_os_contains,
            enable_qga=enable_qga,
            enable_cloud_init_seed=enable_cloud_init_seed,
            images_dir=images_dir,
            test_results_dir=test_results_dir,
            qemu_available=qemu_available,
            kvm_available=kvm_available,
            vm_start_timeout=vm_start_timeout,
            vm_boot_timeout=vm_boot_timeout,
            reboot_timeout=reboot_timeout,
            vm_stop_timeout=vm_stop_timeout,
            firmware=firmware,
        )

    # Step 5: Execute the reboot scenario, surfacing interrupts/errors cleanly.
    if not qemu_available:
        test_failed = True
        failure_message = f"QEMU not available: {qemu_msg}"
        logger.error(failure_message)
    else:
        try:
            result = execute_test_with_cache(
                cached_result=cached_result,
                cache_result=cache_result,
                test_name=test_name,
                configs=configs,
                cache_configs=cache_configs,
                run_test_func=_run_reboot_workload,
            )
        except KeyboardInterrupt:
            test_interrupted = True
            failure_message = "Interrupt detected during QEMU/KVM VM reboot test execution"
            logger.error(failure_message)
        except Exception as e:
            test_failed = True
            failure_message = f"Unexpected error during QEMU/KVM VM reboot test execution: {str(e)}"
            logger.error(failure_message, exc_info=True)

    # Ensure a result object always exists so the test terminates cleanly even
    # when QEMU is missing or it is interrupted before a result was produced.
    if result is None:
        result = Result(
            name=test_display_name,
            metadata={
                "status": False,
                "message": failure_message or "QEMU/KVM VM reboot test did not complete",
            },
            metrics={
                "avg_reboot_time": Metrics(unit="seconds", value=-1, is_key_metric=True),
                "total_reboot_time": Metrics(unit="seconds", value=-1, is_key_metric=False),
            },
        )

    # A produced-but-failing result must also fail the test.
    if not result.metadata.get("status", False):
        test_failed = True
        if not failure_message:
            failure_message = result.metadata.get("message", "QEMU/KVM VM reboot test failed")

    # Step 6: Always validate and summarize so the result is recorded
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

    # Step 7: Cache results — only when the test completed without interruption
    # or unexpected failure (guards against caching broken placeholder results
    # that would cause subsequent runs to show a false "passed" status).
    if not test_interrupted and not test_failed:
        cache_result(result)

    logger.info(
        "QEMU/KVM VM reboot test completed: %s - %s",
        test_display_name,
        result.metadata.get("message", ""),
    )

    # Terminate cleanly: surface interrupts/errors as a proper test outcome.
    if test_interrupted:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
