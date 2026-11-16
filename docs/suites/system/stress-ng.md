# System Stress (stress-ng)

Runs `stress-ng` on the host with CPU, memory, and GPU stressors for a set time, checks that the system stays stable, and records the stressors' throughput.

| | |
|---|---|
| **Test suite** | `test_stress_ng` (`src/esq/suites/system/stress/test_stress_ng.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). It wraps `stress-ng`, and the profile sets the stressors, workers, load, and duration. |
| **Key metric** | `cpu_bogo_ops_per_real_time`, or `gpu_bogo_ops_per_real_time` when GPU stress runs (bogo operations per second). Override it with `key_metric_name`. |
| **Devices** | CPU, memory, iGPU |

## How It Works

1. Build the `stress-ng` command from the enabled stressors.
2. Run it for `stress_duration_seconds`.
3. Read the `stress-ng` metrics of each stressor and report the bogo operations per second.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stress_duration_seconds` | `60` | Run time in seconds. The environment variable `ENV_SUITE_STRESS_NG_DURATION_SECONDS` overrides it without editing the profile. |
| `enable_cpu_stress`, `stress_cpu_workers`, `stress_cpu_load` | `true`, `0`, `90` | CPU stressors, workers (`0` = one per CPU), and load per worker in % |
| `enable_memory_stress`, `stress_vm_workers`, `stress_vm_bytes` | `true`, `0`, `100%` | Memory stressors, workers, and memory per worker |
| `enable_gpu_stress`, `stress_gpu_workers`, `stress_gpu_device_index` | `true`, `0`, `-1` | GPU rendering stressors, workers, and GPU (`-1` = default) |
| `stress_gpu_ops`, `stress_gpu_frag`, `stress_gpu_tex_size`, `stress_gpu_upload`, `stress_gpu_xsize`, `stress_gpu_ysize` | `0` | `stress-ng` GPU options. `0` keeps the `stress-ng` default. |
| `gpu_tool`, `gpu_custom_command` | `stress-ng`, unset | GPU stress tool, or a custom GPU stress command |
| `key_metric_name` | unset | Metric to use as the key metric |
| `timeout` | duration plus 120 s | Seconds for the test case |

Qualification criteria are listed in [Qualifications](../../profiles/qualification.md).

## Profiles

<!-- esq:suite-profiles test_stress_ng -->

## Run

```bash
esq run --profile profile.horizontal.system.stress --filter test_id=SNG-CPU-001
```
