# Background Telemetry

Intel® ESQ automatically collects system metrics as a background thread during each test run. No configuration or code changes in the test itself are needed — telemetry is controlled entirely from the profile YAML and results are visible in the Allure report.

---

## How It Works

A lightweight daemon thread polls each enabled telemetry module at a configurable interval (default: 10 seconds). Collection starts when a test begins and stops cleanly when the test finishes, regardless of the outcome. The collected data — time-series samples, averages, and min/max values — is attached to the test result in the Allure report.

---

## Available Modules

### CPU and Memory

| Module | Metrics | Description |
|--------|---------|-------------|
| `cpu_freq` | `current_mhz`, `min_mhz`, `max_mhz` | CPU clock frequency |
| `cpu_usage` | `total_percent` | Total CPU utilization (%) |
| `cpu_temp` | `package_c`, `core_max_c` | CPU package and peak core temperature (°C) via kernel coretemp driver |
| `memory_usage` | `used_percent`, `available_gib`, `used_gib` | System RAM usage |

### Power

| Module | Metrics | Description |
|--------|---------|-------------|
| `package_power` | `package_power_w`, `core_power_w`, `uncore_power_w`, `dram_power_w` | Intel® CPU package power via RAPL powercap interface (requires hardware support and `system-setup.sh`) |

### GPU

| Module | Metrics | Description |
|--------|---------|-------------|
| `gpu_temp` | `gpu_{N}_pkg_c`, `gpu_{N}_vram_c` (per GPU) | Intel® GPU package and VRAM temperature (°C) per detected GPU via hwmon sysfs |
| `gpu_freq` | `gpu_{N}_gt{M}_mhz` (per GPU and GT) | Intel® GPU operating frequency (MHz) per GT via DRM sysfs |
| `gpu_power` | `gpu_{N}_w`, `gpu_{N}_card_w` (per GPU) | Intel® GPU power (W) via hwmon energy counters; `card_w` includes total board power where available |
| `gpu_usage` | `gpu_{N}_render_pct`, `gpu_{N}_compute_pct`, `gpu_{N}_copy_pct`, `gpu_{N}_video_pct`, `gpu_{N}_video_enh_pct` (Arc/xe); `gpu_{N}_gt{M}_pct` (iGPU/i915) | Intel® GPU engine utilization (%) via Linux perf PMU (Arc GPUs) or GT-level RC6/C6 residency (iGPU) |

### NPU

| Module | Metrics | Description |
|--------|---------|-------------|
| `npu_usage` | `npu_{N}_busy_pct`, `npu_{N}_mem_mib` (per NPU) | Intel® NPU busy utilization (%) and allocated device memory (MB) via `intel_vpu` driver sysfs |
| `npu_freq` | `npu_{N}_freq_mhz` (per NPU) | Intel® NPU current operating frequency (MHz) via `intel_vpu` driver sysfs |

> **Note:** GPU and NPU modules automatically skip on systems where the corresponding hardware or driver is not present. The `gpu_usage` module requires `perf_event_paranoid ≤ 0` for Arc GPU engine-level metrics (set by `scripts/system-setup.sh`).

---

## Controlling the Sampling Interval

The default interval is set in each profile (typically 10 seconds). Override it per-run without editing the profile:

```bash
# Via CLI option (highest priority)
esq run --profile <profile> --telemetry-interval 10

# Via environment variable
CORE_TELEMETRY_INTERVAL=10 esq run --profile <profile>
```

The value must be a whole number of seconds, minimum 1.

---
