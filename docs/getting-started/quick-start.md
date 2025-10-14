# Quick Start

Install all required dependencies to run all tests and evaluate your edge system.

## Requirements

Before starting, ensure your system meets the following requirements:


### 1. Hardware

Intel® ESQ supports a wide range of Intel® edge systems optimized for various performance and use case requirements.

| **Category** | **CPU** | **Memory** | **Storage** | **Discrete GPU** |
|--------------|------------------------|------------|-------------|------------------|
| **Scalable Performance** | Intel® Xeon® 5 processor | 256–512 GB | 1 TB | Optional |
| **Scalable AI, Graphics & Media** | Intel® processors with Intel® Arc™ Graphics | Min 64 GB | 512 GB | **Required: Alchemist or Battlemage** |
| **Efficiency Optimized AI** | Intel® Core™ Ultra processor Series 1 or Series 2 | Min 32 GB | 512 GB | Optional |
| **Mainstream** | 14th generation Intel® Core™ processors or higher | Min 32 GB | 256 GB | Optional |
| **Entry** | Intel® Core™ processor, Intel® Processor, Intel Atom® | Min 16 GB | 256 GB | Not supported |

### 2. Operating System

Install a supported operating system before proceeding.

| OS | Version | Notes |
|----|---------|-------|
| [**Ubuntu***](https://ubuntu.com/download/desktop) | 24.04 Desktop LTS or newer | Recommended Linux* distribution |
| **Windows*** | Coming soon | Coming soon |

## Installation

### 1. System Drivers

Configure system drivers:

```bash
sudo bash -c "$(wget -qLO - https://raw.githubusercontent.com/intel/edge-developer-kit-reference-scripts/refs/heads/main/main_installer.sh)"
```

!!! info "Additional Reference"
    For detailed information about system drivers, see the [Edge Developer Kit Reference Scripts](https://github.com/intel/edge-developer-kit-reference-scripts) documentation.

### 2. System Dependencies

Install essential system packages:

```bash
sudo apt update && sudo apt install -y curl git
```

### 3. Docker Engine

Install Docker* Engine:

```bash
# Add Docker's official GPG key
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker packages
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Add your user to the docker group:

!!! warning
    Adding your user to the `docker` group grants root-level access. This should only be done on development or test systems.

```bash
sudo usermod -aG docker $USER
```

To activate your new group membership immediately in your current terminal, run:

```bash
newgrp docker
```

This command starts a new shell session with updated group permissions, allowing you to use Docker* without logging out. You can now verify Docker* installation:

```bash
docker ps
```

!!! note
    If you see a list of containers (even if empty), your user is correctly added to the `docker` group. If you get a permission error, ensure you have run `newgrp docker` in your terminal. For persistent access across all sessions, log out and log back in, or reboot your system.

!!! info "Additional Reference"
    For detailed Docker* installation instructions, see the official [Docker* Engine installation documentation](https://docs.docker.com/engine/install).

### 4. Python Package Manager

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to accelerate Python* package management:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
```

!!! info "Additional Reference"
    For detailed `uv` installation instructions, see the official [uv Installation](https://docs.astral.sh/uv/getting-started/installation/) documentation.


### 5. Intel® ESQ

Install Intel® ESQ from GitHub*:

```bash
uv tool install --force --refresh git+https://github.com/open-edge-platform/edge-system-qualification.git@main
```

Verify that ESQ is working correctly:

```bash
esq --version
```

## Quick Start

Run all tests and review the generated test report:

!!! tip "Newer Version"
    Before running a new version of ESQ, run the following command to clean up any previously created `esq_data` folder:
    
    ```bash
    esq clean --all
    ```
    
    This ensures that leftover data from previous ESQ versions does not interfere with the new installation. If you have uninstalled ESQ but the `esq_data` folder still exists, remove it using the above command before running any new ESQ commands. Otherwise, ESQ may not work as expected.

### 1. Run Intel® ESQ

Run all tests to generate a test report:

```bash
esq run
```

This command will:

1. Run all test suites
2. Collect metrics
3. Generate a test report

!!! tip "Verbose Output"
    Use the `--verbose` option to see detailed information while running tests:
    
    ```bash
    esq --verbose run
    ```
    
    This provides real-time feedback on test progress, system information collection, and detailed execution logs.

!!! info "Driver Requirements"
    Intel® GPU and NPU tests require specific drivers. Ensure you have the latest Intel® drivers installed for your hardware configuration.

!!! note "Virtualization"
    Running in virtual machines may impact performance and hardware acceleration capabilities. Bare metal installation is recommended for accurate testing results.

---

Ready to explore all test suites? Continue to the [Test Suites](suites/index.md) →
