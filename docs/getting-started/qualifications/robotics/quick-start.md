# Quick Start

The Robotics qualification measures real-time performance and determinism readiness for robotics workloads.

Before you begin, confirm your system meets the [Requirements](requirements.md).

## Installation

### 1. System Drivers

@import "../../../includes/quick-start/system-drivers.md"

### 2. Docker Engine

@import "../../../includes/quick-start/docker-engine.md"

### 3. Python Package Manager

@import "../../../includes/quick-start/python-package-manager.md"

### 4. Real-Time Kernel

Download and run the real-time Linux setup script, with GRUB and runtime tuning enabled for lower latency:

```bash
mkdir -p ~/.esq/setup/rt-kernel && cd ~/.esq/setup/rt-kernel

wget -qO os_setup_install.sh https://raw.githubusercontent.com/open-edge-platform/edge-ai-suites/2026.1/robotics-ai-suite/docs/embodied/get-started/prerequisites/os_setup_install.sh
wget -qO rt_linux_setup.sh https://raw.githubusercontent.com/open-edge-platform/edge-ai-suites/2026.1/robotics-ai-suite/docs/embodied/get-started/installation/rt_linux_setup.sh
chmod +x os_setup_install.sh rt_linux_setup.sh

sudo -E ./rt_linux_setup.sh --apply-rt-grub-tuning --disable-timer-migration --disable-swap --disable-cstate-cpus 11-11
```

The script configures GRUB* to boot the real-time kernel automatically. After it finishes, reboot the system.

!!! info "Additional Reference"
    For manual installation steps and other runtime tuning options, see the [Real-Time Linux](https://github.com/open-edge-platform/edge-ai-suites/blob/2026.1/robotics-ai-suite/docs/embodied/get-started/installation/rt_linux.md) documentation.

### 5. System Setup

@import "../../../includes/quick-start/system-setup.md"

@import "../../../includes/quick-start/system-setup-rt.md"

### 6. Intel® ESQ

@import "../../../includes/quick-start/esq-install.md"

## Qualification

### 1. Run

@import "../../../includes/quick-start/run.md"

Run this qualification directly:

```bash
esq run -t robotics
```

### 2. Uninstall

@import "../../../includes/quick-start/uninstall.md"

---

Continue to [Qualification Profiles](../index.md) →
