# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""KVM virtualization compatibility helpers.

Splits host-side detection logic out of the pytest entry points so each test
file stays focused on orchestration. Modules:

- ``checks``: VT-x, VT-d/IOMMU, kernel module, nested virtualization, /dev/kvm,
  and VFIO device compatibility probes (used by the compatibility test).
- ``topology``: peripheral/display/USB enumeration, peripheral VM capacity
  estimation, and the ASCII block-diagram builder (used by the VM capacity
  test).
- ``qemu``: reusable QEMU/KVM VM lifecycle helpers (availability probes, image
  preparation, cloud-init seeding, start/stop/reboot, guest-readiness probes,
  and the reboot scenario runner) shared across QEMU/KVM test entry points.
- ``info``: persistence of detailed virtualization diagnostics to disk.
"""
