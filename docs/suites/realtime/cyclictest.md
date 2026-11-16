# Real-Time Wakeup Latency (cyclictest)

Measures the worst-case wakeup latency of real-time threads with `cyclictest` on isolated CPU cores, while `stress-ng` loads the other cores. Optional cache partitioning and C-state settings show the effect of real-time tuning.

| | |
|---|---|
| **Test suite** | `test_cyclictest` (`src/esq/suites/realtime/performance/test_cyclictest.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). It wraps `cyclictest` and `stress-ng`, and the profile sets the cores, timing, load, and tuning. |
| **Key metric** | `max_latency_us` (µs, lower is better), with `avg_latency_us` and `min_latency_us` |
| **Devices** | CPU |

## How It Works

1. Optionally partition the L3 cache with Intel® Cache Allocation Technology (`cat_enabled`) and disable deep C-states on the real-time cores (`cstate_disable_enabled`). The previous state is saved.
2. Start `stress-ng` on the stress cores.
3. Run `cyclictest` on the real-time cores for `cyclic_duration`.
4. Report the latencies and restore the saved state, also when the test is interrupted.

The optimizations need the modules of `scripts/system-setup-rt.sh`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cyclic_cpu_affinity` | cores 2 and 3 | Cores of the real-time threads |
| `cyclic_main_affinity` | unset | Core of the cyclictest main thread |
| `cyclic_priority` | `99` | `SCHED_FIFO` priority |
| `cyclic_interval_us` | `250` | Timer interval in µs |
| `cyclic_histogram_us` | `700` | Latency histogram range in µs |
| `cyclic_mlock` | `true` | Lock memory to prevent paging |
| `cyclic_duration` | `24h` | Run time, for example `1h` or `24h` |
| `stress_enabled` | `true` | Load the other cores with `stress-ng` |
| `stress_cpu_affinity`, `stress_cpu_load`, `stress_cpu_workers`, `stress_all_except_cyclic` | `0-1`, `100`, `0`, `false` | Stress cores, load per worker in %, workers (`0` = one per core), or stress all cores except the real-time ones |
| `cat_enabled`, `cat_rt_clos`, `cat_rt_ways` | `false`, `1`, upper L3 ways | Cache partitioning for the real-time cores |
| `cstate_disable_enabled`, `cstate_max_latency_us` | `false`, `0` | Disable C-states deeper than this exit latency on the real-time cores |
| `timeout` | duration plus 900 s | Seconds for the test case |

## Profiles

<!-- esq:suite-profiles test_cyclictest -->

## Run

```bash
esq run --profile profile.horizontal.realtime.performance --filter test_id=RT-CYC-001
```
