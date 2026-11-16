# KVM Virtualization

Checks that the system can host KVM virtual machines, estimates how many VMs can get their own display, keyboard, and mouse, and measures how fast a QEMU/KVM VM reboots.

| | |
|---|---|
| **Test suite** | `test_kvm_compatibility`, `test_kvm_vm_capacity`, `test_kvm_qemu_reboot` (`src/esq/suites/virtualization/kvm/`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). ESQ checks the system directly, and the profile selects the checks. |
| **Key metric** | Compatibility: `kvm_dev_accessible` (true/false), or `vt_d_enabled` and `nested_virt_enabled` when only those checks run. Capacity: `peripheral_vm_capacity` (vms). Reboot: `avg_reboot_time` (seconds). |
| **Devices** | Host system (CPU, IOMMU, connected displays and USB input devices) |

| Test | What it verifies |
|------|------------------|
| `test_kvm_compatibility` | VT-x CPU flags, VT-d/IOMMU, `kvm` and `kvm_intel` modules, VFIO modules and devices, nested virtualization, and access to `/dev/kvm`. Collects data only. The test always passes. |
| `test_kvm_vm_capacity` | Number of VMs that can each get one display, one keyboard, and one mouse by passthrough. It is the smaller of the display count and keyboard/mouse pairs, and 0 without IOMMU groups. Collects data only. Qualification profiles set a KPI on it. |
| `test_kvm_qemu_reboot` | One QEMU/KVM VM is created, booted, rebooted `reboot_count` times, and removed. Fails if any step fails. |

## How It Works

1. **Compatibility:** read `/proc/cpuinfo`, the DMAR table and IOMMU groups, `lsmod`, the `kvm_intel` nested parameter, and `/dev/kvm` for each enabled check, and save the details under `esq_data/data/suites/virtualization/kvm/results/<test_id>`.
2. **Capacity:** count IOMMU groups, connected displays, and keyboard/mouse pairs (pairs on one USB hub first), and attach a topology diagram and `lsusb -t`.
3. **Reboot:** download `guest_image_url` (or create a blank qcow2 disk), seed an SSH key with cloud-init, and start a headless VM with the chosen `firmware`.
4. **Reboot:** wait for the guest with `guest_ready_probe`, optionally read `/etc/os-release` over SSH, then reboot the VM and time each cycle until the guest is ready again. Image preparation is not part of the reboot times.

## Parameters

**`test_kvm_compatibility`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `check_vt_x`, `check_vt_d` | `true`, `true` | Check VT-x CPU flags and VT-d/IOMMU |
| `check_kvm_modules`, `check_vfio_modules` | `true`, `true` | Check the `kvm`/`kvm_intel` and `vfio`/`vfio_pci`/`vfio_iommu_type1` modules |
| `check_nested_virt`, `check_kvm_dev` | `true`, `true` | Check nested virtualization and access to `/dev/kvm` |

**`test_kvm_vm_capacity`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kpi_refs`, `kpi_override` | unset | KPI on `peripheral_vm_capacity` (qualification profiles) |

**`test_kvm_qemu_reboot`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `guest_image_url` | unset | Guest cloud image to download. Unset creates a blank qcow2 disk. |
| `firmware` | `seabios` | `seabios` or `ovmf` (UEFI, needs the `ovmf` package) |
| `use_kvm` | `true` | Use KVM acceleration |
| `vm_cpu_count`, `vm_memory_mb`, `vm_disk_mb` | `2`, `4096`, `16384` | VM CPUs, memory, and disk size |
| `reboot_count` | `1` | Reboot cycles |
| `guest_ready_probe` | `serial` | How guest readiness is detected: `serial` (console prompt), `ssh`, or `qga` (QEMU guest agent) |
| `verify_guest_os`, `expected_guest_os_contains` | `true`, unset | Read `/etc/os-release` over SSH and check that it contains the given text |
| `guest_ssh_user`, `guest_ssh_host_port` | `ubuntu`, `2222` | SSH user and forwarded host port |
| `enable_qga`, `enable_cloud_init_seed` | `true`, `true` | Attach the guest agent socket and the cloud-init seed ISO (cloud images only) |
| `vm_start_timeout`, `vm_boot_timeout`, `reboot_timeout`, `vm_stop_timeout`, `guest_ready_timeout` | `60`, `120`, `120`, `30`, `180` | Seconds per lifecycle phase |

Qualification criteria are listed in [Qualifications](../../profiles/qualification.md).

## Profiles

<!-- esq:suite-profiles test_kvm_compatibility test_kvm_vm_capacity test_kvm_qemu_reboot -->

## Run

```bash
esq run --profile profile.horizontal.virtualization.kvm-qemu --filter test_id=KVM-RBT-002
```
