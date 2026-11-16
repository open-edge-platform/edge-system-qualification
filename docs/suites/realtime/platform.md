# Real-Time Platform Checks

Checks whether the system is set up for real-time workloads: kernel, clock, networking, cache partitioning, and CPU power settings. Each check reports a single value, usually `1` (ready) or `0` (not ready).

| | |
|---|---|
| **Test suites** | Eight checks in `src/esq/suites/realtime/platform/`, listed below |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). ESQ checks the system directly, and the profile selects the checks. |
| **Key metric** | One per check, listed below |
| **Devices** | CPU and platform |

## Checks

| Check | Test | Key metric | What it verifies |
|-------|------|------------|------------------|
| PREEMPT-RT kernel | `test_preempt_rt` | `preempt_rt` | The kernel has full real-time preemption (`PREEMPT_RT`), and optionally the required boot parameters |
| TSN | `test_tsn` | `tsn_capable_interfaces` (interfaces) | Number of Time-Sensitive Networking capable interfaces (Intel® `igc` driver) |
| TSC clocksource | `test_tsc_clocksource` | `tsc_clocksource_active` | The kernel uses the Time Stamp Counter as its clocksource |
| PCI Multi-VC | `test_pci_multi_vc` | `pci_multi_vc` (devices) | Number of PCIe devices with more than one Virtual Channel |
| Cache partitioning | `test_cat_config` | `cat_partitioned` | Intel® Cache Allocation Technology (L3 or L2) partitioning is active |
| C-states | `test_cstate` | `rt_cores_cstate_disabled` | C-states with an exit latency above the limit are disabled on the real-time cores |
| Timer migration | `test_timer_migration` | `timer_migration_disabled` | Kernel timer migration is disabled |
| CPU frequency | `test_cpufreq` | `energy_perf_pref` (0 to 15), `rt_cores_cpufreq_performance` | The global energy performance preference, and that the real-time cores use the expected frequency governor |

`test_cpufreq` holds two checks, selected by the `test` field of the profile entry: `test_energy_perf_pref_check` and `test_cpufreq_rt_check`. The real-time cores are the isolated CPUs of the kernel command line (`isolcpus`).

`test_epb.py` and `test_tcc.py` in the same folder are not used by any shipped profile.

## Parameters

| Parameter | Check | Default | Description |
|-----------|-------|---------|-------------|
| `kernel_boot_params` | `test_preempt_rt` | unset | Boot parameters that must be set. Unset checks the kernel only. |
| `rt_cpu_ids` | `test_cstate`, `test_cpufreq` | isolated CPUs | Check only these real-time cores |
| `rt_cstate_max_latency_us` | `test_cstate` | `0` | Highest C-state exit latency allowed in µs. `0` allows only POLL. |
| `rt_cpufreq_expected_governor` | `test_cpufreq` | `performance` | Governor the real-time cores must use |
| `test` | `test_cpufreq` | required | `test_energy_perf_pref_check` or `test_cpufreq_rt_check` |
| `timeout` | all | per check | Seconds for the test case |

## Profiles

<!-- esq:suite-profiles test_preempt_rt test_tsn test_tsc_clocksource test_pci_multi_vc test_cat_config test_cstate test_timer_migration test_cpufreq -->

## Run

```bash
esq run --profile profile.horizontal.realtime.platform --filter test_id=RT-KRN-001
```
