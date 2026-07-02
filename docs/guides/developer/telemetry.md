# Modular Telemetry

The framework automatically collects system metrics as a background daemon thread during test execution. No test code changes are required — telemetry is enabled entirely through the profile YAML.

---

## Enabling Telemetry

Add a `telemetry` block inside `params` in your profile YAML:

```yaml
params:
  telemetry:
    enabled: true
    interval: 10         # Seconds between samples (integer, minimum 1)
    modules:
      - name: cpu_freq          # CPU frequency (current_mhz, min_mhz, max_mhz)
        enabled: true
      - name: cpu_usage         # CPU utilization (total_percent)
        enabled: true
        thresholds:
          total_percent:
            warning: 95         # Log WARNING if CPU usage exceeds 95%
      - name: memory_usage      # RAM usage (used_percent, available_gib, used_gib)
        enabled: true
        thresholds:
          used_percent:
            warning: 90
      - name: package_power     # Intel® RAPL CPU package power (hardware-dependent)
        enabled: true
      - name: gpu_usage         # Intel® GPU engine utilization (%)
        enabled: true
      - name: gpu_freq          # Intel® GPU operating frequency (MHz)
        enabled: true
      - name: gpu_temp          # Intel® GPU temperature (°C)
        enabled: true
      - name: gpu_power         # Intel® GPU power (W)
        enabled: true
      - name: npu_usage         # Intel® NPU busy utilization (%) and memory (MiB)
        enabled: true
      - name: npu_freq          # Intel® NPU operating frequency (MHz)
        enabled: true
```

To disable telemetry, remove the block or set `enabled: false`.

---

## Per-Module Keys

| Key | Description |
|-----|-------------|
| `metrics` | Restrict collection to a subset (e.g., `metrics: [current_mhz]`) |
| `thresholds` | Log a `WARNING` when a metric exceeds the configured value (does not fail the test) |
| `chart_type` | Chart hint for the report renderer: `line` (default), `area`, or `bar_vertical` |
| `title` / `scales` | Display labels and units for chart axes |

---

## Overriding the Telemetry Interval at Runtime

Priority order (highest to lowest):

1. `--telemetry-interval SECONDS` CLI option
2. `CORE_TELEMETRY_INTERVAL` environment variable
3. `interval` in the profile YAML

The value must be a whole number of seconds (minimum 1).

```bash
esq run --profile profile.suite.ai.gen --telemetry-interval 10
CORE_TELEMETRY_INTERVAL=10 esq run --profile profile.suite.ai.gen
```

---

## Available Modules

| Module | Metrics Collected | Notes |
|--------|-------------------|-------|
| `cpu_freq` | `current_mhz`, `min_mhz`, `max_mhz` | |
| `cpu_usage` | `total_percent` | |
| `cpu_temp` | `package_c`, `core_max_c` | |
| `memory_usage` | `used_percent`, `available_gib`, `used_gib` | |
| `package_power` | `package_power_w`, `core_power_w`, `uncore_power_w`, `dram_power_w` | Requires RAPL; run `scripts/system-setup.sh` |
| `gpu_temp` | `gpu_{N}_pkg_c`, `gpu_{N}_vram_c` (per GPU) | Skipped if no Intel® GPU present |
| `gpu_freq` | `gpu_{N}_gt{M}_mhz` (per GPU and GT) | Skipped if no Intel® GPU present |
| `gpu_power` | `gpu_{N}_w`, `gpu_{N}_card_w` (per GPU) | Skipped if no Intel® GPU present |
| `gpu_usage` | `gpu_{N}_render_pct`, `gpu_{N}_compute_pct`, `gpu_{N}_copy_pct`, `gpu_{N}_video_pct`, `gpu_{N}_video_enh_pct` (Intel® Arc™/xe); `gpu_{N}_gt{M}_pct` (iGPU/i915) | Requires `perf_event_paranoid ≤ 0` for Arc engine metrics; run `scripts/system-setup.sh` |
| `npu_usage` | `npu_{N}_busy_pct`, `npu_{N}_mem_mib` (per NPU) | Requires `intel_vpu` driver |
| `npu_freq` | `npu_{N}_freq_mhz` (per NPU) | Requires `intel_vpu` driver |

---

## Creating a Custom Module

Implement `BaseTelemetryModule` and register it in `src/esq/utils/telemetry/modules/`.

### Step 1 — Create the Module File

```python
# src/esq/utils/telemetry/modules/disk_io.py

import time
from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetrySample


class DiskIoModule(BaseTelemetryModule):
    module_name = "disk_io"

    def is_available(self) -> bool:
        return True  # Check for dependencies here

    def collect_sample(self) -> TelemetrySample:
        raw = {"read_mb_s": 0.0}  # Replace with real readings
        values = self._filter_values(raw)
        sample = TelemetrySample(timestamp=time.time(), values=values)
        self.check_thresholds(values)
        return sample
```

### Step 2 — Register the Module

```python
# src/esq/utils/telemetry/modules/__init__.py

from esq.utils.telemetry.modules.disk_io import DiskIoModule
from sysagent.utils.telemetry.registry import register as _register

_register("disk_io", DiskIoModule)
```

### Step 3 — Enable in a Profile

```yaml
telemetry:
  enabled: true
  interval: 10
  modules:
    - name: disk_io
      enabled: true
```

---

## Telemetry Data in Results

Telemetry samples collected during a test run are automatically stored in:

```python
results.extended_metadata["telemetry"]
```

See [Results & Metrics](results-metrics.md) for details on `extended_metadata`.

---

## Related Pages

- [Results & Metrics](results-metrics.md) — Where telemetry data is stored in `Result`
- [Profile & Test Config](configuration.md) — Adding the `telemetry` block to a profile
