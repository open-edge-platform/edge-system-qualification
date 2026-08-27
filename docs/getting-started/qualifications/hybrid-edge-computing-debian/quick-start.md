# Quick Start

The Hybrid Edge Computing (HEC) - Debian* qualification measures system, display, memory, network, peripheral, and virtualization readiness for hybrid edge deployments on Debian*-based systems.

Before you begin, confirm your system meets the [Requirements](requirements.md).

## Installation

### 1. System Drivers

@import "../../../includes/quick-start/system-drivers.md"

### 2. Docker Engine

@import "../../../includes/quick-start/docker-engine.md"

### 3. Python Package Manager

@import "../../../includes/quick-start/python-package-manager.md"

### 4. Hardware Virtualization

This qualification runs KVM*/QEMU* virtualization tests, so hardware virtualization must be turned on:

1. Restart the system and enter the BIOS/UEFI setup.
2. Enable **Intel® VT-x** and **Intel® VT-d** (names vary by vendor).
3. Save changes and reboot.

Verify virtualization is enabled:

```bash
egrep -c '(vmx|svm)' /proc/cpuinfo
```

A result greater than `0` confirms virtualization is available.

### 5. System Setup

@import "../../../includes/quick-start/system-setup.md"

@import "../../../includes/quick-start/system-setup-advanced.md"

### 6. Intel® ESQ

@import "../../../includes/quick-start/esq-install.md"

## Qualification

### 1. Run

@import "../../../includes/quick-start/run.md"

Run this qualification directly:

```bash
esq run -t hec-debian
```

### 2. Uninstall

@import "../../../includes/quick-start/uninstall.md"

---

Continue to [Qualification Profiles](../index.md) →
