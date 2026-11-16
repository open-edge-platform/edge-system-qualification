# CPU

Identifies the CPU (brand, processor number, generation, codename, segment) and reports its socket count, core counts, and frequencies. Optionally checks the CPU against a list of qualified generations or SKUs.

| | |
|---|---|
| **Test suite** | `test_cpu` (`src/esq/suites/system/cpu/test_cpu.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). ESQ checks the system directly, and the profile selects the checks. |
| **Key metric** | Only the metrics listed in `kpi_refs` or `key_metrics`: `cpu_compatibility` (1 pass, 0 fail), `sockets`, `physical_cores`, `logical_cores` (cores), `max_frequency_mhz`, `min_frequency_mhz` (MHz) |
| **Devices** | CPU |

## How It Works

1. Read the CPU details from the system information cache. The processor number is the last token of the brand string (for example `285K`).
2. Run the compatibility check. The allowlists (`processor_numbers`, `exact_generations`, `supported_generations`) are combined with OR: the CPU passes if it matches any of them. `unsupported_generations` is used only when no allowlist is set. With no list set, `cpu_compatibility` is `1` (audit mode).
3. Keep only the metrics named in `kpi_refs` or `key_metrics`. Brand, generation, codename, segment, and check mode are recorded as parameters.
4. Validate the KPIs. Thresholds are disabled by default and enabled per profile with `kpi_override`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `processor_numbers` | `[]` | Allowlist of processor numbers, for example `["155H", "285K"]`. Case-insensitive, whole-token match in the brand string. |
| `exact_generations` | `[]` | Allowlist with exact matching. Each entry is a generation string, a dict with any of `generation`, `codename`, `product_collection`, `segment`, or a list `[generation, product_collection, segment]`. |
| `supported_generations` | `[]` | Allowlist that also accepts newer generations of the same product family |
| `unsupported_generations` | `[]` | Denylist. The CPU passes if its generation is not listed and not older than a listed entry. Ignored when an allowlist is set. |
| `key_metrics` | `[]` | Metrics to report (horizontal profiles) |
| `kpi_refs`, `kpi_override` | `[]`, unset | Metrics to report and validate, and their thresholds (qualification profiles), see [KPI Validation](../../guides/developer/kpi-validation.md) |

Qualification criteria are listed in [Qualifications](../../profiles/qualification.md).

## Profiles

<!-- esq:suite-profiles test_cpu -->

## Run

```bash
esq run --profile profile.horizontal.system.cpu-sku --filter test_id=CPU-SKU-002
```
