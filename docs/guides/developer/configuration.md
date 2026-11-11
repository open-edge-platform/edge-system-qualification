# Profile & Test Configuration

This page covers the profile YAML structure, how profiles map to test files, and the per-suite `config.yml` format for KPI definitions.

---

## Profile Structure

Profiles are YAML files that define a test execution plan — which tests to run, with which parameters, and under what system requirements.

```yaml
name: "profile.horizontal.example"
description: "Example test suite"

params:
  labels:
    profile_group_name: "Example"
    profile_display_name: "Example - Basic Checks"
    type: "horizontal"   # "qualification", "horizontal", or "vertical"
    # report_section: "HEC"  # optional (any profile type); overrides which report
    #   section this profile's tests are grouped under. Defaults to each type's
    #   standard section ("Qualifications", "Horizontals", or "Verticals") when
    #   omitted. Keep this short (e.g. an abbreviation) since many profiles
    #   may each define their own section name.

  requirements:
    # Hardware
    cpu_min_cores: 4
    memory_min_gib: 8.0
    storage_min_gib: 10.0

    # Software
    os_type: ["linux"]
    docker_required: true

    # Devices
    igpu_required: false
    dgpu_required: false
    npu_required: false

suites:
  - name: "suite_name"
    sub_suites:
      - name: "sub_suite_name"
        tests:
          test_function_name:
            params:
              - test_id: "EX-001"
                display_name: "Example Test"
                devices: [cpu, igpu]
                timeout: 300
                kpi_refs:
                  - example_kpi
```

### Profile Types

Profiles are stored in `src/esq/configs/profiles/` in one of these sub-directories:

| Directory | Purpose |
|-----------|---------|
| `qualifications/` | Tests with KPI-based pass/fail criteria |
| `horizontals/` | Horizontal tests: data collection without KPI validation (`type: "horizontal"`) |
| `verticals/` | Industry-specific tests |

### Profile Labels

Each label under `params.labels` has one role. Labels are also attached to every test result, so the report reads them from there.

```yaml
params:
  labels:
    type: "horizontal"
    profile_group_name: "Vision AI"
    profile_display_name: "Vision AI - OpenVINO Benchmark"
```

**Report sections and grouping**

| Label | Required | Role |
|-------|----------|------|
| `type` | Yes | Sets the default report section: `qualification` → **Qualifications**, `horizontal` → **Horizontals**, `vertical` → **Verticals**. |
| `report_section` | No | Overrides the default section with a custom heading (for example, `HEC`). Works with any `type`. |
| `profile_group_name` | Yes | Groups related profiles in the report and sets the group heading. Profiles that share the value appear together under one heading. |

**Profile selection**

| Label | Required | Role |
|-------|----------|------|
| `profile_display_name` | Yes | Names the individual profile in the `esq run --select` picker. Make it unique and describe what the profile runs, for example `"Vision AI - OpenVINO Benchmark"`. |
| `hidden` | No | Hides the profile from interactive selection. It can still run with `--profile` or `--tag`. |
| `tags` | No | Tags used to select profiles with `--tag`. |
| `system_compatibility` | No | Qualification profiles only. Runs the CPU compatibility check before the profile and skips the profile on unsupported platforms. |

**Report presentation**

| Label | Default | Role |
|-------|---------|------|
| `display_reference` | `false` | Shows the reference column when set to `true`. It does not enable or disable KPI validation. |
| `display_status` | `true` | Hides the status column when set to `false`. |
| `display_id` | `false` | Shows the test ID column when set to `true`. |

!!! note
    - `profile_group_name` is required. A profile without it is reported under "unknown group". The old `group` label is no longer read.
    - An existing installation keeps its previously built report UI. After updating, run `esq clean --thirdparty-only` once so the report UI is rebuilt and uses `profile_group_name`.
    - `kpi_enabled` is informational. The framework does not read it today. KPI validation is controlled by `kpi_refs` and `kpi_override` on each test.
    - Profiles do not need a top-level `display_name`. The picker and report use `profile_display_name` and `profile_group_name`.

---

## Profile to Test File Mapping

The framework automatically maps profile configuration to test files based on the `suites` hierarchy:

```
Profile YAML                          Test File
────────────────────────────────────────────────────────────────
suites:
  - name: "ai"                    →   src/esq/suites/ai/
    sub_suites:
      - name: "vision"            →   src/esq/suites/ai/vision/
        tests:
          test_dlstreamer:        →   test_dlstreamer.py
            params:
              - test_id: "VSN-001"
                devices: [cpu]
```

### Targeting a Specific Function

When a test file contains multiple functions, use the `test:` field to route to the correct one:

```yaml
tests:
  test_cache:           # Discovers: cache/test_cache.py
    params:
      - test: "test_cache_import"     # Routes to test_cache_import()
        test_id: "UNIT-001"
      - test: "test_cache_creation"   # Routes to test_cache_creation()
        test_id: "UNIT-002"
```

### Parameter Flow

All parameters defined in the profile are merged with suite-level defaults and passed to the test function via the `configs` fixture:

```python
def test_dlstreamer(configs, ...):
    devices = configs.get("devices", [])
    timeout = configs.get("timeout", 300)
```

---

## Test Configuration (config.yml)

Each test directory can include a `config.yml` that defines KPI defaults and test-level parameter overrides. The framework loads this file automatically via the `suite_configs` fixture.

The `kpi` block is **optional**. Omit it entirely for data-collection suites where no pass/fail validation is needed. Tests without `kpi_refs` in their parameters will skip the validation step automatically.

```yaml
# src/esq/suites/my_domain/my_feature/config.yml
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Optional: define KPI thresholds for qualification tests.
# Remove this block for data-collection-only suites.
kpi:
  metric_name:
    name: "Human Readable Name"
    type: "numeric"       # numeric | string | boolean | list
    validation:
      operator: "gte"     # gte | lte | eq | ne | gt | lt
      reference: 100.0
      enabled: true
    unit: "unit_name"
    severity: "critical"  # critical | major | normal | minor
    description: "KPI description"
    default_value: 0.0

tests:
  test_function_name:
    params:
      - test_id: "T001"
        display_name: "Test Name"
        # Optional: reference KPIs above. Omit for collection-only tests.
        kpi_refs:
          - metric_name
```

### KPI Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Human-readable label shown in reports |
| `type` | Yes | `numeric`, `string`, `boolean`, or `list` |
| `validation.operator` | Yes | Comparison operator (`gte`, `lte`, `gt`, `lt`, `eq`, `ne`) |
| `validation.reference` | Yes | Target threshold or expected value |
| `validation.enabled` | No | Set to `false` to collect without validating (default: `true`) |
| `unit` | No | Display unit (e.g., `"ms"`, `"fps"`) |
| `severity` | No | `critical`, `major`, `normal`, or `minor` |
| `description` | No | Short explanation of what this KPI measures |
| `default_value` | No | Fallback value if the metric is not collected |

---

## Profile Inheritance

Profiles can extend a base profile to share common parameters:

```yaml
# base_profile.yml
name: "profile.base"
params:
  requirements:
    cpu_min_cores: 4
    memory_min_gib: 8.0

# extended_profile.yml
extends: "profile.base"
name: "profile.extended"
params:
  requirements:
    cpu_min_cores: 8       # Override base value
    dgpu_required: true    # Add new requirement
```

---

## Related Pages

- [System Requirements](requirements.md) — All available requirement flags
- [KPI Validation](kpi-validation.md) — Defining and using KPIs
- [Writing Tests](writing-tests.md) — Full step-by-step test creation guide
