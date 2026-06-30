# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
QEMU/KVM VM reboot test.

This suite validates VM lifecycle operations focused on reboot behavior:
create image, start VM, validate boot, execute reboot cycles, and cleanup.
"""

import json
import logging
import os
import re
import shutil
import socket
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Dict, Optional

import allure
import pytest
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result, run_command
from sysagent.utils.infrastructure import download_file

logger = logging.getLogger(__name__)


def get_scenario_key_metric(configs: Dict) -> tuple[str, str]:
    """Return a stable scenario label for report key matrix display."""
    use_kvm = configs.get("use_kvm", True)
    return "scenario_mode", "kvm" if use_kvm else "tcg"


def check_qemu_availability() -> tuple[bool, str, Optional[str]]:
    """Check if qemu-system-x86_64 is installed and executable."""
    result = run_command(["which", "qemu-system-x86_64"], timeout=5)
    if result and result.returncode == 0 and result.stdout:
        qemu_path = result.stdout.strip()
        if os.access(qemu_path, os.X_OK):
            version_result = run_command([qemu_path, "--version"], timeout=5)
            if version_result and version_result.returncode == 0 and version_result.stdout:
                version = version_result.stdout.strip().split("\n")[0]
                return True, f"QEMU available: {version}", qemu_path
            return True, f"QEMU available at {qemu_path}", qemu_path
        return False, f"QEMU found but not executable: {qemu_path}", None
    return False, "QEMU not found (qemu-system-x86_64 not in PATH)", None


def check_kvm_available() -> tuple[bool, str]:
    """Check if /dev/kvm exists and is accessible."""
    kvm_dev = "/dev/kvm"
    if not os.path.exists(kvm_dev):
        return False, "/dev/kvm not found - KVM not available"
    if not os.access(kvm_dev, os.R_OK | os.W_OK):
        return False, "/dev/kvm not accessible - check permissions"
    return True, "/dev/kvm available"


def create_test_vm_image(output_dir: str, size_mb: int = 100) -> Optional[str]:
    """Create a minimal qcow2 image for reboot validation."""
    image_path = os.path.join(output_dir, "test_vm.qcow2")
    result = run_command(["qemu-img", "create", "-f", "qcow2", image_path, f"{size_mb}M"], timeout=30)
    if result and result.returncode == 0 and os.path.exists(image_path):
        # Validate that the image exists and is non-empty before test execution continues.
        if os.path.getsize(image_path) > 0:
            logger.info("Created VM image: %s (%sMB)", image_path, size_mb)
            return image_path
        logger.error("VM image exists but is empty: %s", image_path)
        return None
    logger.error("Failed to create VM image: %s", result.stderr if result else "unknown error")
    return None


def _safe_image_name_from_url(image_url: str) -> str:
    """Build a safe local filename for a downloaded guest image URL."""
    parsed = urllib.parse.urlparse(image_url)
    name = os.path.basename(parsed.path) or "guest-image.qcow2"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def download_guest_image(image_url: str, output_dir: str) -> Optional[str]:
    """Download a guest OS image and validate it exists and is non-empty."""
    parsed_url = urllib.parse.urlparse(image_url)
    if parsed_url.scheme not in ("http", "https"):
        logger.error("Invalid URL scheme '%s'. Only http and https are allowed: %s", parsed_url.scheme, image_url)
        return None
    
    local_name = _safe_image_name_from_url(image_url)
    image_path = os.path.join(output_dir, local_name)

    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        logger.info("Using cached guest image: %s", image_path)
        return image_path

    try:
        logger.info("Downloading guest image from: %s", image_url)
        download_file(url=image_url, target_path=image_path)
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            logger.info("Downloaded guest image: %s", image_path)
            return image_path
        logger.error("Downloaded guest image is missing or empty: %s", image_path)
        return None
    except (RuntimeError, IOError, OSError) as exc:
        logger.error("Failed to download guest image from %s: %s", image_url, exc)
        return None


def prepare_vm_image(output_dir: str, size_mb: int, guest_image_url: Optional[str]) -> Optional[str]:
    """Prepare VM image from URL when provided, otherwise create a blank qcow2 image."""
    if guest_image_url:
        return download_guest_image(guest_image_url, output_dir)
    return create_test_vm_image(output_dir, size_mb)


def command_available(command_name: str) -> bool:
    """Check whether a command exists in PATH."""
    result = run_command(["which", command_name], timeout=5)
    return bool(result and result.returncode == 0 and result.stdout)


def ensure_ssh_keypair(key_path: str) -> tuple[bool, str]:
    """Ensure SSH key pair exists; create one if missing."""
    pub_key_path = f"{key_path}.pub"
    if os.path.exists(key_path) and os.path.exists(pub_key_path):
        return True, pub_key_path

    result = run_command(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", key_path], timeout=15)
    if result and result.returncode == 0 and os.path.exists(pub_key_path):
        return True, pub_key_path

    return False, ""


def create_cloud_init_seed(seed_iso_path: str, vm_id: str, ssh_user: str, ssh_pubkey_path: str) -> bool:
    """Create cloud-init seed ISO for cloud images using cloud-localds."""
    if not command_available("cloud-localds"):
        logger.error("cloud-localds is not available")
        return False

    if not os.path.exists(ssh_pubkey_path):
        logger.warning("SSH public key not found for cloud-init seed: %s", ssh_pubkey_path)
        return False

    with open(ssh_pubkey_path, "r", encoding="utf-8") as file:
        ssh_key = file.read().strip()

    temp_dir = tempfile.mkdtemp(prefix=f"qemu-seed-{vm_id}-")
    user_data_path = os.path.join(temp_dir, "user-data")
    meta_data_path = os.path.join(temp_dir, "meta-data")

    user_data = (
        "#cloud-config\n"
        f"users:\n"
        f"  - name: {ssh_user}\n"
        f"    sudo: ALL=(ALL) NOPASSWD:ALL\n"
        "    shell: /bin/bash\n"
        "    ssh_authorized_keys:\n"
        f"      - {ssh_key}\n"
        "ssh_pwauth: false\n"
        "disable_root: true\n"
    )
    meta_data = f"instance-id: {vm_id}\nlocal-hostname: {vm_id}\n"

    with open(user_data_path, "w", encoding="utf-8") as file:
        file.write(user_data)
    with open(meta_data_path, "w", encoding="utf-8") as file:
        file.write(meta_data)

    result = run_command(["cloud-localds", seed_iso_path, user_data_path, meta_data_path], timeout=20)
    shutil.rmtree(temp_dir, ignore_errors=True)

    if result and result.returncode == 0 and os.path.exists(seed_iso_path):
        return True

    logger.warning("Failed to generate cloud-init seed ISO: %s", result.stderr if result else "unknown")
    return False


def detect_image_format(image_path: str) -> str:
    """Detect image format using qemu-img info, fallback to qcow2."""
    result = run_command(["qemu-img", "info", "--output=json", image_path], timeout=10)
    if result and result.returncode == 0 and result.stdout:
        try:
            parsed = json.loads(result.stdout)
            image_format = parsed.get("format")
            if isinstance(image_format, str) and image_format:
                return image_format
        except json.JSONDecodeError:
            pass
    return "qcow2"


def wait_for_serial_prompt(serial_log_path: str, timeout: int = 180, start_position: int = 0) -> tuple[bool, str]:
    """Wait for serial log to indicate guest readiness/login prompt.
    
    Args:
        serial_log_path: Path to the serial log file
        timeout: Maximum time to wait in seconds
        start_position: File position to start reading from (for reboots)
    """
    patterns = [
        r"login:\\s*$",
        r"cloud-init.*finished",
        r"Reached target .*Login",
    ]

    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(serial_log_path):
            try:
                with open(serial_log_path, "r", encoding="utf-8", errors="ignore") as file:
                    file.seek(start_position)
                    content = file.read()
                if any(re.search(pattern, content, flags=re.MULTILINE | re.IGNORECASE) for pattern in patterns):
                    return True, "Serial log readiness pattern matched"
            except (IOError, OSError):
                pass
        time.sleep(2)

    return False, "Timed out waiting for readiness patterns in serial log"


def wait_for_ssh_ready(ssh_user: str, ssh_key_path: str, ssh_port: int, timeout: int = 180) -> tuple[bool, str]:
    """Wait for guest SSH to become reachable and accept key auth."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = run_command(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                "ConnectTimeout=5",
                "-i",
                ssh_key_path,
                "-p",
                str(ssh_port),
                f"{ssh_user}@127.0.0.1",
                "echo ssh-ready",
            ],
            timeout=12,
        )
        if result and result.returncode == 0 and result.stdout and "ssh-ready" in result.stdout:
            return True, "SSH readiness probe succeeded"
        time.sleep(3)

    return False, "Timed out waiting for SSH readiness"


def wait_for_qga_ready(qga_socket_path: str, timeout: int = 120) -> tuple[bool, str]:
    """Best-effort QGA readiness probe using guest-ping on QGA socket."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(qga_socket_path):
            try:
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.settimeout(3)
                client.connect(qga_socket_path)
                client.sendall(b'{"execute":"guest-ping"}\n')
                response = client.recv(4096).decode("utf-8", errors="ignore")
                client.close()
                if "return" in response:
                    return True, "QGA guest-ping succeeded"
            except (OSError, socket.error, socket.timeout):
                pass
        time.sleep(2)

    return False, "Timed out waiting for QGA readiness"


def get_guest_os_release_via_ssh(ssh_user: str, ssh_key_path: str, ssh_port: int) -> tuple[bool, str]:
    """Read PRETTY_NAME from guest /etc/os-release via SSH."""
    result = run_command(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=5",
            "-i",
            ssh_key_path,
            "-p",
            str(ssh_port),
            f"{ssh_user}@127.0.0.1",
            "cat /etc/os-release",
        ],
        timeout=15,
    )

    if not result or result.returncode != 0 or not result.stdout:
        return False, ""

    for line in result.stdout.splitlines():
        if line.startswith("PRETTY_NAME="):
            return True, line.split("=", 1)[1].strip().strip('"')
    return False, ""


def should_require_ssh_for_os_check(guest_ready_probe: str, configs: Dict) -> bool:
    """Whether OS identity check must fail when SSH is unavailable."""
    # Default: strict only when the primary readiness probe is SSH.
    # Serial/QGA modes can run without guest networking and should not hard-fail.
    if "require_ssh_for_os_check" in configs:
        return bool(configs.get("require_ssh_for_os_check"))
    return guest_ready_probe == "ssh"


def get_vm_pid_file(vm_id: str) -> str:
    return f"/tmp/qemu-{vm_id}.pid"


def get_short_tmp_socket_path(prefix: str, vm_id: str) -> str:
    """Return a short UNIX socket path under /tmp (must be <108 bytes)."""
    safe_vm = "".join(ch for ch in vm_id if ch.isalnum() or ch in "-_")
    suffix = safe_vm[-24:] if len(safe_vm) > 24 else safe_vm
    return f"/tmp/{prefix}-{suffix}.sock"


def read_vm_pid(vm_id: str) -> Optional[int]:
    pid_file = get_vm_pid_file(vm_id)
    if not os.path.exists(pid_file):
        return None
    try:
        with open(pid_file, "r", encoding="utf-8") as file:
            pid_text = file.read().strip()
        pid = int(pid_text)
        return pid if pid > 0 else None
    except (IOError, OSError, ValueError):
        return None


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def start_qemu_vm(
    qemu_path: str,
    image_path: str,
    vm_id: str,
    use_kvm: bool = True,
    memory_mb: int = 512,
    cpu_count: int = 1,
    image_format: str = "qcow2",
    serial_log_path: Optional[str] = None,
    ssh_host_port: Optional[int] = None,
    qga_socket_path: Optional[str] = None,
    seed_iso_path: Optional[str] = None,
    timeout: int = 30,
) -> tuple[bool, str, Optional[int]]:
    """Start a headless QEMU VM and return PID."""
    pid_file = get_vm_pid_file(vm_id)
    qemu_args = [
        qemu_path,
        "-name",
        f"test-vm-{vm_id}",
        "-m",
        str(memory_mb),
        "-smp",
        str(cpu_count),
        "-drive",
        f"file={image_path},format={image_format},if=virtio",
        "-display",
        "none",
        "-daemonize",
        "-pidfile",
        pid_file,
        "-monitor",
        f"unix:/tmp/qemu-monitor-{vm_id}.sock,server,nowait",
    ]

    if use_kvm:
        qemu_args.extend(["-enable-kvm", "-cpu", "host"])
    else:
        qemu_args.extend(["-cpu", "qemu64"])

    if serial_log_path:
        qemu_args.extend(["-serial", f"file:{serial_log_path}"])

    if ssh_host_port:
        qemu_args.extend([
            "-netdev",
            f"user,id=net0,hostfwd=tcp::{ssh_host_port}-:22",
            "-device",
            "virtio-net-pci,netdev=net0",
        ])

    if qga_socket_path:
        qemu_args.extend([
            "-device",
            "virtio-serial",
            "-chardev",
            f"socket,path={qga_socket_path},server=on,wait=off,id=qga0",
            "-device",
            "virtserialport,chardev=qga0,name=org.qemu.guest_agent.0",
        ])

    if seed_iso_path and os.path.exists(seed_iso_path):
        qemu_args.extend([
            "-drive",
            f"file={seed_iso_path},format=raw,media=cdrom,readonly=on",
        ])

    result = run_command(qemu_args, timeout=timeout)
    if not result or result.returncode != 0:
        error_text = result.stderr if result and result.stderr else "unknown error"
        logger.error("Failed to start VM: %s", error_text)
        return False, f"Error starting VM: {error_text}", None

    for _ in range(timeout):
        vm_pid = read_vm_pid(vm_id)
        if vm_pid and is_process_running(vm_pid):
            return True, f"VM started successfully (PID: {vm_pid})", vm_pid
        time.sleep(1)

    return False, "VM did not become active before timeout", None


def send_qemu_command(vm_id: str, command: str, timeout: int = 10) -> tuple[bool, str]:
    """Send one command to QEMU monitor socket."""
    monitor_socket = f"/tmp/qemu-monitor-{vm_id}.sock"
    if not os.path.exists(monitor_socket):
        return False, f"Monitor socket not found: {monitor_socket}"

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(monitor_socket)
        _ = sock.recv(4096)
        sock.sendall(f"{command}\n".encode("utf-8"))
        time.sleep(0.5)
        response = sock.recv(4096).decode("utf-8", errors="ignore")
        sock.close()
        return True, response
    except (OSError, socket.error, socket.timeout) as exc:
        logger.error("Failed to send monitor command: %s", exc)
        return False, f"Error: {exc}"


def check_vm_booted(vm_id: str, vm_pid: int, timeout: int = 20) -> tuple[bool, str]:
    """Verify VM process and monitor status reach running state."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not is_process_running(vm_pid):
            return False, f"VM process exited unexpectedly (PID: {vm_pid})"

        success, response = send_qemu_command(vm_id, "info status")
        if success:
            response_lower = response.lower()
            if "running" in response_lower:
                return True, "VM reported running state"
            if "paused" in response_lower:
                return False, "VM is paused"

        time.sleep(1)

    return False, "Timed out waiting for VM running state"


def reboot_vm(
    vm_pid: int,
    vm_id: str,
    timeout: int = 30,
    guest_ready_probe: str = "serial",
    serial_log_path: Optional[str] = None,
    ssh_user: Optional[str] = None,
    ssh_key_path: Optional[str] = None,
    ssh_port: Optional[int] = None,
    qga_socket_path: Optional[str] = None,
    guest_ready_timeout: int = 180,
) -> tuple[bool, str, float]:
    """Reboot a running VM using QEMU monitor and measure reboot time including guest readiness."""
    if not is_process_running(vm_pid):
        return False, "VM not running", 0.0

    # Track serial log position before reboot to only check new content
    serial_log_position = 0
    if guest_ready_probe == "serial" and serial_log_path and os.path.exists(serial_log_path):
        try:
            serial_log_position = os.path.getsize(serial_log_path)
        except (IOError, OSError):
            serial_log_position = 0

    start_time = time.time()
    success, response = send_qemu_command(vm_id, "system_reset")
    if not success:
        return False, f"Failed to send reboot command: {response}", 0.0

    booted, boot_msg = check_vm_booted(vm_id, vm_pid, timeout=timeout)
    if not booted:
        return False, f"VM reboot did not reach running state: {boot_msg}", 0.0

    # Wait for guest to be fully ready after reboot
    readiness_ok = False
    readiness_msg = ""
    if guest_ready_probe == "ssh" and ssh_user and ssh_key_path and ssh_port:
        readiness_ok, readiness_msg = wait_for_ssh_ready(
            ssh_user, ssh_key_path, ssh_port, timeout=guest_ready_timeout
        )
    elif guest_ready_probe == "qga" and qga_socket_path:
        readiness_ok, readiness_msg = wait_for_qga_ready(qga_socket_path, timeout=guest_ready_timeout)
    elif guest_ready_probe == "serial" and serial_log_path:
        readiness_ok, readiness_msg = wait_for_serial_prompt(
            serial_log_path, timeout=guest_ready_timeout, start_position=serial_log_position
        )
    else:
        return False, f"Invalid readiness probe configuration: {guest_ready_probe}", 0.0

    if not readiness_ok:
        return False, f"Guest not ready after reboot: {readiness_msg}", 0.0

    return True, "VM rebooted successfully", time.time() - start_time


def stop_qemu_vm(vm_pid: Optional[int], vm_id: str, timeout: int = 10) -> tuple[bool, str]:
    """Stop VM gracefully, then force terminate if needed."""
    if vm_pid is None:
        return True, "No VM process to stop"

    success, _ = send_qemu_command(vm_id, "quit")
    if success:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not is_process_running(vm_pid):
                return True, "VM stopped gracefully"
            time.sleep(1)

    if is_process_running(vm_pid):
        kill_result = run_command(["kill", "-TERM", str(vm_pid)], timeout=5)
        if kill_result and kill_result.returncode == 0:
            deadline = time.time() + 5
            while time.time() < deadline:
                if not is_process_running(vm_pid):
                    return True, "VM terminated"
                time.sleep(1)

        run_command(["kill", "-KILL", str(vm_pid)], timeout=5)
        return True, "VM killed (forced)"

    return True, "VM stopped"


def cleanup_vm_resources(vm_id: str, image_path: Optional[str] = None, remove_image: bool = True) -> None:
    """Clean monitor socket, pid file, and temporary image."""
    monitor_socket = f"/tmp/qemu-monitor-{vm_id}.sock"
    if os.path.exists(monitor_socket):
        try:
            os.remove(monitor_socket)
        except (IOError, OSError) as exc:
            logger.warning("Failed to remove monitor socket: %s", exc)

    pid_file = get_vm_pid_file(vm_id)
    if os.path.exists(pid_file):
        try:
            os.remove(pid_file)
        except (IOError, OSError) as exc:
            logger.warning("Failed to remove VM pid file: %s", exc)

    if remove_image and image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
        except (IOError, OSError) as exc:
            logger.warning("Failed to remove VM image: %s", exc)


def run_qemu_vm_test(
    test_display_name: str,
    test_id: str,
    qemu_path: str,
    use_kvm: bool,
    vm_memory_mb: int,
    vm_disk_mb: int,
    vm_cpu_count: int,
    reboot_count: int,
    guest_image_url: Optional[str],
    guest_ready_probe: str,
    guest_ready_timeout: int,
    guest_ssh_user: str,
    guest_ssh_host_port: int,
    verify_guest_os: bool,
    expected_guest_os_contains: Optional[str],
    enable_qga: bool,
    enable_cloud_init_seed: bool,
    images_dir: Path,
    test_results_dir: Path,
    scenario_metric_name: str,
    scenario_metric_value: str,
    scenario_reboot_plan: str,
    qemu_available: bool,
    kvm_available: bool,
) -> Result:
    """Execute the QEMU VM reboot test and return results."""
    vm_id = f"{test_id}-{int(time.time())}"
    vm_result = {
        "vm_id": vm_id,
        "created": False,
        "started": False,
        "rebooted": False,
        "ready": False,
        "reboot_times": [],
        "errors": [],
        "readiness_probe": guest_ready_probe,
        "guest_os_pretty_name": None,
    }

    vm_launch_time_seconds = -1.0
    readiness_duration_seconds = -1.0
    ssh_ready_for_os_check = False
    qga_probe_ready = False
    serial_probe_ready = False

    image_path = None
    image_format = "qcow2"
    vm_pid = None
    seed_iso_path = str(test_results_dir / f"{vm_id}-seed.iso")
    serial_log_path = str(test_results_dir / f"{vm_id}-serial.log")
    qga_socket_path = get_short_tmp_socket_path("qga", vm_id) if enable_qga else None
    ssh_key_path = str(images_dir / "qemu_guest_ssh_key")
    ssh_pubkey_path = ""
    cloud_init_attached = False

    try:
        image_path = prepare_vm_image(str(images_dir), vm_disk_mb, guest_image_url)
        if not image_path:
            vm_result["errors"].append("Failed to create VM image")
        else:
            vm_result["created"] = True
            image_format = detect_image_format(image_path)

            if guest_image_url and enable_cloud_init_seed:
                key_ready, ssh_pubkey_path = ensure_ssh_keypair(ssh_key_path)
                if key_ready:
                    cloud_init_attached = create_cloud_init_seed(seed_iso_path, vm_id, guest_ssh_user, ssh_pubkey_path)
                    if not cloud_init_attached:
                        vm_result["errors"].append("Failed to generate cloud-init seed ISO (cloud-localds required)")
                else:
                    vm_result["errors"].append("Failed to create SSH keypair for cloud-init seed")

            startup_start = time.time()
            started, start_msg, vm_pid = start_qemu_vm(
                qemu_path,
                image_path,
                vm_id,
                use_kvm=use_kvm,
                memory_mb=vm_memory_mb,
                cpu_count=vm_cpu_count,
                image_format=image_format,
                serial_log_path=serial_log_path,
                ssh_host_port=guest_ssh_host_port,
                qga_socket_path=qga_socket_path,
                seed_iso_path=seed_iso_path if cloud_init_attached else None,
                timeout=30,
            )
            vm_launch_time_seconds = time.time() - startup_start
            if not started or vm_pid is None:
                vm_result["errors"].append(f"Failed to start VM: {start_msg}")
            else:
                vm_result["started"] = True
                booted, boot_msg = check_vm_booted(vm_id, vm_pid, timeout=60)
                if not booted:
                    vm_result["errors"].append(f"VM boot verification failed: {boot_msg}")
                else:
                    readiness_start = time.time()
                    readiness_ok = False
                    readiness_msg = ""
                    if guest_ready_probe == "ssh":
                        readiness_ok, readiness_msg = wait_for_ssh_ready(
                            guest_ssh_user, ssh_key_path, guest_ssh_host_port, timeout=guest_ready_timeout
                        )
                        ssh_ready_for_os_check = readiness_ok
                    elif guest_ready_probe == "qga":
                        if qga_socket_path:
                            readiness_ok, readiness_msg = wait_for_qga_ready(qga_socket_path, timeout=guest_ready_timeout)
                            qga_probe_ready = readiness_ok
                        else:
                            readiness_ok, readiness_msg = False, "QGA probe requested but qga socket is disabled"
                    else:
                        readiness_ok, readiness_msg = wait_for_serial_prompt(serial_log_path, timeout=guest_ready_timeout)
                        serial_probe_ready = readiness_ok
                    readiness_duration_seconds = time.time() - readiness_start

                    vm_result["ready"] = readiness_ok
                    if not readiness_ok:
                        vm_result["errors"].append(f"Guest readiness probe failed: {readiness_msg}")
                    else:
                        if verify_guest_os:
                            ssh_timeout_for_os = 90 if guest_ready_probe != "ssh" else 30
                            os_ok, _ = wait_for_ssh_ready(
                                guest_ssh_user, ssh_key_path, guest_ssh_host_port, timeout=ssh_timeout_for_os
                            )
                            ssh_ready_for_os_check = os_ok
                            if os_ok:
                                os_info_ok, guest_os = get_guest_os_release_via_ssh(
                                    guest_ssh_user, ssh_key_path, guest_ssh_host_port
                                )
                                if os_info_ok:
                                    vm_result["guest_os_pretty_name"] = guest_os
                                    if expected_guest_os_contains and expected_guest_os_contains.lower() not in guest_os.lower():
                                        vm_result["errors"].append(
                                            f"Guest OS mismatch: expected contains '{expected_guest_os_contains}', got '{guest_os}'"
                                        )
                                else:
                                    vm_result["errors"].append("Failed to read /etc/os-release from guest via SSH")
                            else:
                                vm_result["errors"].append("Guest OS identity check requires SSH readiness")

                    if not vm_result["errors"]:
                        for index in range(reboot_count):
                            reboot_ok, reboot_msg, reboot_time = reboot_vm(
                                vm_pid,
                                vm_id,
                                timeout=30,
                                guest_ready_probe=guest_ready_probe,
                                serial_log_path=serial_log_path,
                                ssh_user=guest_ssh_user,
                                ssh_key_path=ssh_key_path,
                                ssh_port=guest_ssh_host_port,
                                qga_socket_path=qga_socket_path,
                                guest_ready_timeout=guest_ready_timeout,
                            )
                            if reboot_ok:
                                vm_result["reboot_times"].append(reboot_time)
                            else:
                                vm_result["errors"].append(f"Reboot {index + 1} failed: {reboot_msg}")
                                break

                    if len(vm_result["reboot_times"]) == reboot_count:
                        vm_result["rebooted"] = True
    finally:
        if os.path.exists(serial_log_path):
            try:
                with open(serial_log_path, "r", encoding="utf-8", errors="ignore") as file:
                    serial_content = file.read()
                allure.attach(
                    serial_content,
                    name="VM Serial Console Output",
                    attachment_type=allure.attachment_type.TEXT,
                )
            except (IOError, OSError) as exc:
                logger.warning("Failed to attach serial log to allure: %s", exc)
        if vm_pid is not None:
            stop_qemu_vm(vm_pid, vm_id, timeout=10)
        cleanup_vm_resources(vm_id, image_path, remove_image=not bool(guest_image_url))
        if os.path.exists(seed_iso_path):
            try:
                os.remove(seed_iso_path)
            except (IOError, OSError):
                pass
        if qga_socket_path and os.path.exists(qga_socket_path):
            try:
                os.remove(qga_socket_path)
            except (IOError, OSError):
                pass

    vms_total = 1
    vms_created = 1 if vm_result["created"] else 0
    vms_started = 1 if vm_result["started"] else 0
    vms_ready = 1 if vm_result["ready"] else 0
    vms_rebooted = 1 if vm_result["rebooted"] else 0

    reboot_times = vm_result["reboot_times"]
    metrics = {}
    if reboot_times:
        total_reboot_time = sum(reboot_times)
        avg_reboot_time = total_reboot_time / len(reboot_times)
        min_reboot_time = min(reboot_times)
        max_reboot_time = max(reboot_times)
        metrics["total_reboot_time"] = Metrics(unit="seconds", value=round(total_reboot_time, 2), is_key_metric=False)
        metrics["avg_reboot_time"] = Metrics(unit="seconds", value=round(avg_reboot_time, 2), is_key_metric=True)
        metrics["min_reboot_time"] = Metrics(unit="seconds", value=round(min_reboot_time, 2), is_key_metric=False)
        metrics["max_reboot_time"] = Metrics(unit="seconds", value=round(max_reboot_time, 2), is_key_metric=False)
    else:
        metrics["total_reboot_time"] = Metrics(unit="seconds", value=-1, is_key_metric=False)
        metrics["avg_reboot_time"] = Metrics(unit="seconds", value=-1, is_key_metric=True)
        metrics["min_reboot_time"] = Metrics(unit="seconds", value=-1, is_key_metric=False)
        metrics["max_reboot_time"] = Metrics(unit="seconds", value=-1, is_key_metric=False)

    metrics["vm_launch_time"] = Metrics(unit="seconds", value=round(vm_launch_time_seconds, 2), is_key_metric=False)
    metrics["initial_boot_readiness_time"] = Metrics(unit="seconds", value=round(readiness_duration_seconds, 2), is_key_metric=False)
    metrics["ssh_ready_for_os_check"] = Metrics(value=ssh_ready_for_os_check, is_key_metric=False)
    metrics["qga_probe_ready"] = Metrics(value=qga_probe_ready, is_key_metric=False)
    metrics["serial_probe_ready"] = Metrics(value=serial_probe_ready, is_key_metric=False)

    guest_os_value = vm_result["guest_os_pretty_name"] or "Unknown (SSH not available)"
    metrics["guest_os_pretty_name"] = Metrics(value=guest_os_value, is_key_metric=True)

    test_passed = vm_result["created"] and vm_result["started"] and vm_result["ready"] and vm_result["rebooted"]
    test_message = f"VM reboot test: {vms_rebooted}/{vms_total} VM(s) passed."
    if reboot_times:
        test_message += f" Avg reboot time: {sum(reboot_times) / len(reboot_times):.2f}s."

    return Result(
        name=test_display_name,
        metadata={
            "status": test_passed,
            "message": test_message,
            "vm_created_success": vm_result["created"],
            "vm_started_success": vm_result["started"],
            "guest_ready_success": vm_result["ready"],
            "vm_reboot_loop_success": vm_result["rebooted"],
        },
        metrics=metrics,
        extended_metadata={
            "execution_context": {
                "scenario_reboot_plan": scenario_reboot_plan,
                scenario_metric_name: scenario_metric_value,
                "target_reboot_count": reboot_count,
                "qemu_available": qemu_available,
                "kvm_available": kvm_available,
            },
            "execution_summary": {
                "vms_total": vms_total,
                "vms_created": vms_created,
                "vms_started": vms_started,
                "vms_ready": vms_ready,
                "vms_rebooted_successfully": vms_rebooted,
                "vm_created_successfully": vm_result["created"],
            },
            "vm_configuration": {
                "cpu_count": vm_cpu_count,
                "memory_mb": vm_memory_mb,
                "disk_mb": vm_disk_mb,
                "use_kvm": use_kvm,
                "guest_image_url": guest_image_url,
                "guest_ready_probe": guest_ready_probe,
                "guest_ready_timeout": guest_ready_timeout,
                "guest_ssh_user": guest_ssh_user,
                "guest_ssh_host_port": guest_ssh_host_port,
                "verify_guest_os": verify_guest_os,
                "require_ssh_for_os_check": True,
                "expected_guest_os_contains": expected_guest_os_contains,
                "enable_qga": enable_qga,
                "enable_cloud_init_seed": enable_cloud_init_seed,
            },
            "paths": {
                "suite_data_dir": str(images_dir.parent.parent.parent),
                "images_dir": str(images_dir),
                "results_dir": str(test_results_dir),
                "serial_boot_log": str(test_results_dir / f"{vm_id}-serial.log"),
            },
            "reboot_count": reboot_count,
            "vm_result": vm_result,
        },
    )


def test_kvm_qemu(
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
    """Validate reboot behavior for a single QEMU/KVM VM instance."""
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)

    use_kvm = configs.get("use_kvm", True)
    vm_memory_mb = configs.get("vm_memory_mb", 4096)
    vm_disk_mb = configs.get("vm_disk_mb", 16384)
    vm_cpu_count = configs.get("vm_cpu_count", 2)
    reboot_count = configs.get("reboot_count", 1)
    guest_image_url = configs.get("guest_image_url")
    guest_ready_probe = configs.get("guest_ready_probe", "serial")
    guest_ready_timeout = int(configs.get("guest_ready_timeout", 300))
    guest_ssh_user = configs.get("guest_ssh_user", "ubuntu")
    guest_ssh_host_port = int(configs.get("guest_ssh_host_port", 2222))
    verify_guest_os = bool(configs.get("verify_guest_os", True))
    expected_guest_os_contains = configs.get("expected_guest_os_contains")
    enable_qga = bool(configs.get("enable_qga", True))
    enable_cloud_init_seed = bool(configs.get("enable_cloud_init_seed", True))

    validate_system_requirements_from_configs(configs)

    core_data_dir = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    expected_base = Path(os.getcwd()).resolve()
    resolved_core_data = Path(core_data_dir).resolve()
    if not resolved_core_data.is_relative_to(expected_base):
        resolved_core_data = expected_base / "esq_data"

    suite_data_dir = resolved_core_data / "data" / "suites" / "virtualization" / "kvm" / "qemu"
    images_dir = suite_data_dir / "images"
    test_results_dir = suite_data_dir / "results" / test_id

    try:
        os.makedirs(images_dir, mode=0o770, exist_ok=True)
        os.makedirs(test_results_dir, mode=0o770, exist_ok=True)
        ensure_dir_permissions(str(images_dir), uid=os.getuid(), gid=os.getgid(), mode=0o770)
        ensure_dir_permissions(str(test_results_dir), uid=os.getuid(), gid=os.getgid(), mode=0o770)
    except (IOError, OSError, PermissionError) as exc:
        logger.warning("Primary data directory unavailable (%s): %s", suite_data_dir, exc)
        fallback_images_dir = Path(tempfile.gettempdir()) / "esq_qemu_images"
        fallback_dir = Path(tempfile.gettempdir()) / "esq_qemu_results" / test_id
        os.makedirs(fallback_images_dir, mode=0o770, exist_ok=True)
        os.makedirs(fallback_dir, mode=0o770, exist_ok=True)
        images_dir = fallback_images_dir
        test_results_dir = fallback_dir

    qemu_available, qemu_msg, qemu_path = check_qemu_availability()
    kvm_available, kvm_msg = check_kvm_available()

    scenario_metric_name, scenario_metric_value = get_scenario_key_metric(configs)
    scenario_reboot_plan = f"{scenario_metric_value}_reboot_x{reboot_count}"

    if not qemu_available:
        metrics = {
            "avg_reboot_time": Metrics(unit="seconds", value=-1, is_key_metric=True),
            "total_reboot_time": Metrics(unit="seconds", value=-1, is_key_metric=False),
        }
        test_message = f"QEMU not available: {qemu_msg}"
        result = Result(name=test_display_name, metadata={"status": False, "message": test_message}, metrics=metrics)
        validate_test_results(test_name=test_name, results=result, configs=configs, get_kpi_config=get_kpi_config)
        summarize_test_results(results=result, test_name=test_name, configs=configs, get_kpi_config=get_kpi_config)
        cache_result(result)
        pytest.fail(test_message)

    if use_kvm and not kvm_available:
        logger.warning("KVM requested but not available: %s. Falling back to TCG mode.", kvm_msg)
        use_kvm = False

    # Cache configuration for deduplication
    cache_configs = {
        "use_kvm": use_kvm,
        "vm_memory_mb": vm_memory_mb,
        "vm_cpu_count": vm_cpu_count,
        "reboot_count": reboot_count,
        "guest_image_url": guest_image_url,
        "guest_ready_probe": guest_ready_probe,
    }

    # Execute test with cache
    result = execute_test_with_cache(
        cached_result=cached_result,
        cache_result=cache_result,
        test_name=test_name,
        configs=configs,
        cache_configs=cache_configs,
        run_test_func=lambda: run_qemu_vm_test(
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
            scenario_metric_name=scenario_metric_name,
            scenario_metric_value=scenario_metric_value,
            scenario_reboot_plan=scenario_reboot_plan,
            qemu_available=qemu_available,
            kvm_available=kvm_available,
        ),
    )

    # Validate and summarize results
    validate_test_results(test_name=test_name, results=result, configs=configs, get_kpi_config=get_kpi_config)
    summarize_test_results(results=result, test_name=test_name, configs=configs, get_kpi_config=get_kpi_config)

    logger.info("QEMU VM reboot test completed: %s - %s", test_display_name, result.metadata.get("message", ""))
    if not result.metadata.get("status", False):
        pytest.fail(result.metadata.get("message", "Test failed"))
