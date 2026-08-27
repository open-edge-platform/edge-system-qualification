# Quick Start

Before you begin, confirm your system meets the [Requirements](requirements.md). Some steps below are optional depending on the profile you run. See [Qualification](qualifications/index.md) for the list of available qualifications and their dedicated quick start guides.

## Installation

### 1. System Drivers

@import "../includes/quick-start/system-drivers.md"

### 2. Docker Engine

@import "../includes/quick-start/docker-engine.md"

### 3. Python Package Manager

@import "../includes/quick-start/python-package-manager.md"

### 4. System Setup

@import "../includes/quick-start/system-setup.md"

@import "../includes/quick-start/system-setup-advanced.md"

@import "../includes/quick-start/system-setup-rt.md"

### 5. Intel® ESQ

@import "../includes/quick-start/esq-install.md"

## Usage

### 1. Run

@import "../includes/quick-start/run.md"

Run `esq run` to start the interactive prompt:

```bash
esq run
```

By default, this command:

1. Prompts you to select exactly one qualification profile to run.
2. If the selected qualification profile has associated vertical profiles, prompts you to include them.
3. Collects metrics and generates a test report.

### 2. Uninstall

@import "../includes/quick-start/uninstall.md"

---

Ready to run a qualification? Continue to [Qualification Profiles](qualifications/index.md) →
