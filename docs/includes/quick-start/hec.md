## Installation

### 1. System Drivers

@import "../common/quick-start/system-drivers.md"

### 2. Docker Engine

@import "../common/quick-start/docker-engine.md"

### 3. Python Package Manager

@import "../common/quick-start/python-package-manager.md"

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

@import "../common/quick-start/system-setup.md"

@import "../common/quick-start/system-setup-advanced.md"

### 6. Intel® ESQ

@import "../common/quick-start/esq-install.md"

## Usage

### 1. Run

Run the HEC profile for your platform:

```bash
esq run -t hec-debian
```

For HEC EMT, run:

```bash
esq run -t hec-emt
```

@import "../common/quick-start/run.md"

### 2. Uninstall

@import "../common/quick-start/uninstall.md"