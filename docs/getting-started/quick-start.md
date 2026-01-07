# Quick Start

Install all required dependencies to run all tests and evaluate your edge system.

!!! info
    Refer to [Intel® ESQ for Intel® AI Edge Systems](https://www.intel.com/content/www/us/en/developer/articles/guide/esq-for-ai-edge-systems.html) for official main qualification page.


## Requirements

Before starting, ensure your system meets the following requirements:


### 1. Hardware

Intel® ESQ supports a wide range of Intel® edge systems optimized for various performance and use case requirements.

| **Category** | **Processors** | **Specifications** |**Storage** | **Discrete GPU Options** |
|--------------|------------------------|------------------------|-------------|------------------|
| **Scalable Performance Graphics Media** | **Xeon-Based**: <br>Intel® Xeon® 6 Processors,<br>5th Gen Intel® Xeon® Scalable Processors,<br>Intel® Xeon® W Processors<br><br>**Core-Based**: <br>Intel® Core Ultra Series 2,<br> Intel® Core™ Series 2 | **Xeon-Based**: <br>**SKU**: Dual and Single Socket<br>**Minimum System Memory**: 128 GB DDR5 (Dual Channel)<br><br> **Core-Based**:<br> **Minimum System Memory**: 64GB DDR5 (Dual Channel)<br>| **Recommended**: 1TB | **Intel® Arc™ B-Series Graphics,**<br>**Intel® Arc™ Pro B-Series Graphics** |
| **Scalable Performance** | Intel® Xeon® 6 Processors,<br> 5th Gen Intel® Xeon® Scalable Processors, <br>Intel® Xeon® W Processors | **SKU**: Dual and Single Socket<br>**Minimum System Memory**: 128 GB DDR5 (Dual Channnel) <br>| **Recommended**: 1TB | N/A |
| **Efficiency Optimized** | Intel® Core™ Ultra processor Series 2 | **Graphics**: Integrated GPU with 7 Xe-Cores or more <br> **SKU**: Single Socket <br>**Minimum System Memory**: 32 GB DDR5 (Dual Channel)| **Recommended**: 512 GB | N/A |
| **Mainstream** | Intel® Core™ Series 2 | **Minimum System Memory**: 32 GB DDR5 (Dual Channel)| **Recommended**: 512 GB | N/A |
| **Entry** | Intel® Processor for Desktop,<br>Intel® Processor X-series, <br>Intel® Processor N-series | **Minimum System Memory**: 32 GB DDR5 (Dual Channel) | **Recommended**: 512 GB | N/A |

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
sudo bash -c "$(wget -qLO - https://raw.githubusercontent.com/open-edge-platform/edge-developer-kit-reference-scripts/refs/heads/main/main_installer.sh)"
```

!!! info "Additional Reference"
    For detailed information about system drivers, see the [Edge Developer Kit Reference Scripts](https://github.com/open-edge-platform/edge-developer-kit-reference-scripts) documentation.

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


### 5. Platform Power Monitoring

Configure non-root access to RAPL (Running Average Power Limit) powercap files for platform power monitoring:

!!! info "About RAPL Power Monitoring"
    RAPL provides energy consumption data for Intel® processors. This step enables Intel® ESQ to collect power configuration information without requiring root privileges.

Run the automated setup script to configure group-based permissions:

```bash
sudo bash -c "$(wget -qLO - https://raw.githubusercontent.com/open-edge-platform/edge-system-qualification/refs/heads/main/scripts/setup-powercap-permissions.sh)"
```

This setup script:

- **Auto-detects** all powercap energy files on your system
- Creates a `powercap` group for secure, user-specific access
- Adds your user to the `powercap` group
- Configures persistent permissions via `/etc/sysfs.d/powercap.conf` (applied automatically on boot)
- Applies permissions immediately for the current session

!!! note "Security & Persistence"
    This configuration uses group-based permissions (mode 0440, owner root:powercap) to grant read-only access to RAPL powercap files exclusively to users in the `powercap` group. Permissions are automatically reapplied on boot via the sysfs configuration.

!!! tip "Verification"
    Verify access with:
    ```bash
    cat /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
    ```
    If this command prints a number without requiring sudo, the setup was successful.


### 6. Intel® ESQ

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

Run qualification and vertical (recommended) test profiles to generate a test report:

```bash
esq run
```

By default, this command will:

1. Run qualification profiles (always included)
2. Run vertical profiles (unless you choose to skip them at the prompt)
3. Collect metrics and generate a test report

!!! tip "Verbose Output"
    Use the `--verbose` option to see detailed information while running tests:
    
    ```bash
    esq --verbose run
    ```
    
    This provides real-time feedback on test progress, system information collection, and detailed execution logs.

!!! tip "Run All Profile Types"
    Use the `--all` option to run all available profile types (qualification, vertical, and suite):
    
    ```bash
    esq --verbose run --all
    ```

!!! info "Driver Requirements"
    Intel® GPU and NPU tests require specific drivers. Ensure you have the latest Intel® drivers installed for your hardware configuration.

!!! note "Virtualization"
    Running in virtual machines may impact performance and hardware acceleration capabilities. Bare metal installation is recommended for accurate testing results.

---

Ready to explore all test suites? Continue to the [Test Suites](suites/index.md) →
