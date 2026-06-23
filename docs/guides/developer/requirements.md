# System Requirements

The framework provides reusable requirement validation flags. Add them to a profile's `params.requirements` block or to individual test parameters under `requirements`. The test is automatically skipped if any requirement is not met.

---

## CPU

| Flag | Type | Description | Example |
|------|------|-------------|---------|
| `cpu_min_cores` | `int` | Minimum logical CPU cores | `cpu_min_cores: 4` |
| `cpu_xeon_required` | `bool` | Requires Intel® Xeon® processor | `cpu_xeon_required: true` |
| `cpu_core_required` | `bool` | Requires Intel® Core™ processor (includes Intel® Core™ Ultra Desktop) | `cpu_core_required: true` |
| `cpu_ultra_required` | `bool` | Requires Intel® Core™ Ultra processor | `cpu_ultra_required: true` |
| `cpu_ultra_mobile_required` | `bool` | Requires Intel® Core™ Ultra mobile (H/U/V/HX/P suffix) | `cpu_ultra_mobile_required: true` |
| `cpu_entry_required` | `bool` | Requires Intel® entry-level processor (N-series, Intel® Atom® processor) | `cpu_entry_required: true` |
| `cpu_entry_excluded` | `bool` | Excludes entry-level processors (skips test on N-series, Intel® Atom® processor) | `cpu_entry_excluded: true` |

---

## Memory

| Flag | Type | Description | Example |
|------|------|-------------|---------|
| `memory_min_gib` | `float` | Minimum available (free) RAM in GiB | `memory_min_gib: 8.0` |
| `memory_total_min_gib` | `float` | Minimum total installed RAM in GiB | `memory_total_min_gib: 16.0` |

---

## Storage

| Flag | Type | Description | Example |
|------|------|-------------|---------|
| `storage_min_gib` | `float` | Minimum free disk space in GiB | `storage_min_gib: 10.0` |
| `storage_total_min_gib` | `float` | Minimum total disk capacity in GiB | `storage_total_min_gib: 64.0` |

---

## Devices (GPU / NPU)

Device detection uses OpenVINO\*. Only Intel® devices detected by OpenVINO are counted. If a device is present but not detected, the error message suggests installing the relevant drivers.

| Flag | Type | Description | Example |
|------|------|-------------|---------|
| `igpu_required` | `bool` | Requires at least one Intel® integrated GPU | `igpu_required: true` |
| `dgpu_required` | `bool` | Requires at least one Intel® discrete GPU | `dgpu_required: true` |
| `dgpu_min_devices` | `int` | Minimum number of Intel® discrete GPUs | `dgpu_min_devices: 2` |
| `dgpu_max_devices` | `int` | Maximum number of Intel® discrete GPUs | `dgpu_max_devices: 4` |
| `dgpu_min_vram_gib` | `float` | Minimum total VRAM across all discrete GPUs (GiB) | `dgpu_min_vram_gib: 8.0` |
| `dgpu_max_vram_gib` | `float` | Maximum total VRAM across all discrete GPUs (GiB) | `dgpu_max_vram_gib: 48.0` |
| `dgpu_min_vram_per_device_gib` | `float` | Minimum VRAM per discrete GPU (each device must meet this) | `dgpu_min_vram_per_device_gib: 6.0` |
| `dgpu_max_vram_per_device_gib` | `float` | Maximum VRAM per discrete GPU | `dgpu_max_vram_per_device_gib: 24.0` |
| `npu_required` | `bool` | Requires at least one Intel® NPU | `npu_required: true` |
| `npu_min_devices` | `int` | Minimum number of Intel® NPUs | `npu_min_devices: 1` |

---

## Software

| Flag | Type | Description | Example |
|------|------|-------------|---------|
| `os_type` | `list[str]` | Allowed OS types (`"linux"`, `"windows"`) | `os_type: ["linux"]` |
| `docker_required` | `bool` | Requires Docker\* daemon to be available | `docker_required: true` |
| `min_python_version` | `str` | Minimum Python\* version | `min_python_version: "3.10"` |
| `required_system_packages` | `list[str]` | System packages that must be installed | `required_system_packages: ["ffmpeg"]` |
| `required_python_packages` | `list[str]` | Python\* packages that must be installed | `required_python_packages: ["torch"]` |
| `env` | `list[str]` | Environment variables that must be set and non-empty | `env: [HF_TOKEN]` |

---

## Complete Requirements Block Example

```yaml
requirements:
  # CPU
  cpu_min_cores: 8
  cpu_entry_excluded: true      # Skip on entry-level CPUs

  # Memory & Storage
  memory_min_gib: 16.0
  storage_min_gib: 20.0
  storage_total_min_gib: 128.0

  # Devices
  dgpu_required: true
  dgpu_min_vram_per_device_gib: 6.0

  # Software
  os_type: ["linux"]
  docker_required: true
  required_system_packages:
    - ffmpeg

  # Environment variables
  env:
    - HF_TOKEN
    - MY_API_KEY
```

---

## Using Requirements in Tests

Requirements are validated by calling:

```python
validate_system_requirements_from_configs(configs)
```

This will:

1. Check all specified requirements against system capabilities.
2. Skip the test if any requirement is not met (via `pytest.skip`).
3. Provide detailed error messages for failed requirements.
4. Log fix suggestions for common issues.

**Example validation output:**

```
╭─ Validation Failed: my-profile
│  Missing requirements (2):
│  • CPU cores >= 8: Upgrade to a CPU with more cores
│  • Environment variable 'HF_TOKEN' required: Set the environment variable before running: export HF_TOKEN=<value>
╰─
```

---

## Related Pages

- [Profile & Test Config](configuration.md) — Where to place `requirements` blocks in YAML
- [Fixtures Reference](fixtures.md) — `validate_system_requirements_from_configs` fixture details
