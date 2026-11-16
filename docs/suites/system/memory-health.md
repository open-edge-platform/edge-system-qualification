# Memory Health

Checks installed RAM for faults while the OS is running, with one of two probes: the EDAC ECC error counters, or a `memtester` bit-pattern test. It collects data and does not fail on the errors it finds, unless a profile enables the KPI.

| | |
|---|---|
| **Test suite** | `test_memory_health` (`src/esq/suites/system/memory/test_memory_health.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). ESQ reads the EDAC counters or runs `memtester` on the host. The probe, test size, and iterations come from the profile. |
| **Key metric** | EDAC: `uncorrectable_errors` (errors). memtester: `memory_errors` (errors, `-1` if memtester did not complete). |
| **Devices** | System memory |

## How It Works

1. **EDAC** (`check_type: edac`): read the correctable and uncorrectable error counts per memory controller and DIMM from `/sys/devices/system/edac/mc`, and list the DIMMs with errors. Add `ras-mc-ctl` output when it is installed. Without EDAC the test reports `edac_available: false` and still passes.
2. **memtester** (`check_type: memtester`): run `memtester <size>M <iterations>` on the host and count the `FAILURE` lines. The test fails if memtester is not installed, cannot allocate the region, or exits for another reason.
3. Save the report to `esq_data/data/suites/system/memory/results/<test_id>/`.

With `memtester_size_mb: auto`, the size is `MemAvailable` minus the larger of the two reserves, capped at the locked-memory limit (`ulimit -l`). memtester locks every page it tests, so raise the limit (`ulimit -l unlimited`) to test more memory.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `check_type` | `edac` | Probe to run: `edac` or `memtester` |
| `memtester_size_mb` | `256` | Size in MB to test. `auto`, `dynamic`, `available`, `all`, or a value of 0 or less sizes it from the available memory. |
| `memtester_dynamic_reserve_mb`, `memtester_dynamic_reserve_percent` | `512`, `10.0` | Memory left free in dynamic sizing. The larger of the two is used. |
| `memtester_iterations` | `1` | memtester loops |
| `timeout` | `300` | Seconds for the memtester run |
| `kpi_refs`, `kpi_override` | `[]`, unset | KPIs to validate and their thresholds (qualification profiles), see [KPI Validation](../../guides/developer/kpi-validation.md) |

The environment variables `ENV_SUITE_MEMORY_HEALTH_MEMTESTER_SIZE_MB` and `ENV_SUITE_MEMORY_HEALTH_MEMTESTER_ITERATIONS` override the profile values at run time.

Qualification criteria are listed in [Qualifications](../../profiles/qualification.md).

## Profiles

<!-- esq:suite-profiles test_memory_health -->

## Run

```bash
esq run --profile profile.horizontal.system.memory-health --filter test_id=MEM-HLT-002
```
