# Writing Tests

This page walks through creating a test from scratch and explains the standard 7-step execution pattern used across all Intel® ESQ tests.

Tests can live in the `esq` extension package or in any custom extension package that builds on the `sysagent` core framework. Both follow identical structure and conventions — only the package path differs. See [Framework Architecture](index.md#framework-architecture) for an overview of the dual-package design.

---

## Test Types and Required Files

A single test function is reused across three suite types. The suite type is defined entirely by the **profile**, not by the test code. Each type maps to a different report section:

| Suite type | Report section | Purpose | Pass/fail on KPIs |
|------------|----------------|---------|-------------------|
| **Horizontal** | Test results (data collection) | General benchmarking and metric collection | No |
| **Vertical** | Vertical | Industry-specific proxy workloads | No |
| **Qualification** | Qualification | Validate metrics against Intel targets | Yes |

Because the type is profile-driven, a test does not need every file. Create only what the suite type requires:

| File | Horizontal | Vertical | Qualification | Purpose |
|------|:----------:|:--------:|:-------------:|---------|
| `test_<name>.py` | Required | Required | Required | Test implementation |
| Profile YAML | Required | Required | Required | Test parameters, devices, and (for qualification) KPI references |
| `config.yml` | Optional | Optional | Required | Shared KPI **definitions** only |

Key rules:

- **`config.yml` holds KPI definitions only.** It defines the metric name, unit, operator, reference value, and severity for each KPI. It is the single source of truth for thresholds shared across profiles.
- **Do not put test parameters in `config.yml`.** Parameters such as `test_id`, `display_name`, `devices`, `timeout`, and per-test `requirements` belong in the profile YAML. Duplicating them in `config.yml` creates two places to maintain and leads to drift.
- **Data-collection suites (horizontal and vertical) usually skip `config.yml`.** They collect metrics without pass/fail validation, so no KPI definitions are needed. Mark the metrics you want highlighted with `key_metrics` in the profile instead.
- **Qualification profiles reuse the shared definitions.** A qualification references KPIs by name with `kpi_refs` and refines only what differs (for example the threshold) with `kpi_override`. The base definition stays in `config.yml`.
- **Each result must have exactly one key metric.** Call `result.set_key_metric("metric_name")` after populating `result.metrics` to designate the single most representative value. This is the value surfaced in summary views and used as the primary comparison point. A result with no key metric, or multiple key metrics, produces incomplete reports.
- **Prerequisite guards must fail, not skip, in qualification profiles.** When a test cannot run because a required tool, capability, or kernel setting is absent, call `pytest.fail` for qualification profiles and `pytest.skip` for data-collection profiles. Use `(pytest.fail if is_qualification else pytest.skip)(reason)` so the same guard works correctly in both contexts. A silent skip in a qualification run hides a real system deficiency.
- **Do not add `test_id` or `display_name` to `result.parameters`.** These are already encoded in `result.name` (set as `f"{test_id} - {test_display_name}"`) and are auto-populated by the framework. Use `result.parameters` for domain-specific context that aids debugging — for example the device list, model name, codec, or resolution.

---

## Test ID and Domain Naming Convention

Every test parameter set carries a `test_id` that uniquely identifies it in reports. Use a consistent, hierarchical format so tests are easy to group, search, and trace.

**Format:** `<MAIN>-<SUB>-<NNN>`

| Segment | Length | Meaning | Examples |
|---------|--------|---------|----------|
| `MAIN` | 3 characters | Main domain | `VSN` (vision), `MEM` (memory), `MDA` (media), `NET` (network) |
| `SUB` | 3 characters | Sub-domain within the main domain | `OBM` (OpenVINO benchmark), `STR` (STREAM), `DEC` (decode) |
| `NNN` | 3 digits | Sequential number within the sub-domain | `001`, `002`, `003` |

Examples from the existing suites:

```
VSN-OBM-001    # Vision → OpenVINO benchmark → test 1
MEM-STR-001    # Memory → STREAM → test 1
MDA-DEC-004    # Media → Decode → test 4
NET-CON-002    # Network → Connectivity → test 2
```

Guidelines:

- Keep `MAIN` and `SUB` to three characters where practical. Use the shortest clear abbreviation when a concept does not fit neatly (for example `SNG-CPU-001` for a stress-ng CPU test).
- Qualification profiles use a prefix that reflects the qualification (for example `AES-GEN-001` for the AI Edge System generative AI test).
- The **same `test_id` may repeat across tiers** in a qualification profile when the same test runs with tier-specific parameters. The tier distinguishes the runs.
- Numbers are sequential and stable — do not renumber existing tests when adding new ones; append the next available number.

---

## Creating Your First Test

### Step 1: Create the Test File

Create a test file in the suite directory of your target extension package. Add `config.yml` only when the test needs KPI validation (qualification). Data-collection suites can omit it:

```
# In the ESQ extension package:
src/esq/suites/
└── my_domain/
    └── my_feature/
        ├── config.yml          # KPI definitions — required for qualification only
        └── test_my_feature.py  # Test implementation — always required

# In a custom extension package:
src/your_package/suites/
└── my_domain/
    └── my_feature/
        ├── config.yml          # KPI definitions — required for qualification only
        └── test_my_feature.py  # Test implementation — always required
```

### Step 2: Test Function Overview

Every test function follows the same 7-step structure. The table below summarises each step and what it produces:

| Step | Purpose | Required |
|------|---------|:--------:|
| 1. Extract parameters | Read `test_id`, `display_name`, `devices`, `timeout` from `configs` | Always |
| 2. Validate system requirements | Skip the test if the hardware or software requirements are not met | Always |
| 3. Prepare assets | Download models, videos, and build Docker\* images; skipped when already cached | When test needs assets |
| 4. Execute and cache | Run the test logic; return cached result on repeat runs | Always |
| 5. Validate KPIs | Compare metrics against thresholds; only active when `kpi_refs` is set | Qualification only |
| 6. Summarize | Write JSON summary and attach results to the Allure report | Always |
| 7. Surface outcome | Call `pytest.fail(...)` on failure or interrupt so pytest records the result cleanly | Always |

For the complete, production-ready implementation with failure and interrupt handling, see the [Test Execution Pattern](#test-execution-pattern) section below. Use that as the base for any new test.

### Step 3: Create the Configuration File

This file is **only required for qualification tests**. It defines shared KPI
thresholds and nothing else. Data-collection suites (horizontal and vertical)
can skip it entirely.

Keep test parameters out of `config.yml` — `test_id`, `display_name`,
`devices`, `timeout`, and `requirements` all live in the profile YAML (Step 4).
Defining them here as well creates two sources of truth that drift apart over
time.

```yaml
# src/<package>/suites/my_domain/my_feature/config.yml
# (e.g., src/esq/suites/my_domain/my_feature/config.yml)
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# KPI definitions only. Thresholds are disabled by default so the same
# definitions can be reused by data-collection suites (metrics collected,
# not enforced). Qualification profiles enable and refine them per test via
# kpi_refs + kpi_override.
kpi:
  inference_time:
    name: "Inference Time"
    type: "numeric"
    validation:
      operator: "lte"   # less than or equal
      reference: 0.5    # 500 ms max
      enabled: false    # enabled per-profile via kpi_override
    unit: "seconds"
    severity: "major"
    description: "Time taken for model inference"
    default_value: 999.0

  accuracy:
    name: "Model Accuracy"
    type: "numeric"
    validation:
      operator: "gte"   # greater than or equal
      reference: 0.90   # 90% minimum
      enabled: false    # enabled per-profile via kpi_override
    unit: "percentage"
    severity: "critical"
    description: "Model prediction accuracy"
    default_value: 0.0
```

### Step 4: Create a Profile

The profile is where **all test parameters live**. It sets `test_id`,
`display_name`, `devices`, `timeout`, and `requirements`. For a qualification
profile, it also references the shared KPIs with `kpi_refs` and refines
thresholds with `kpi_override`.

```yaml
# src/<package>/configs/profiles/suites/my_feature.yml
# (e.g., src/esq/configs/profiles/suites/my_feature.yml)
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

name: "profile.suite.my_feature"
description: "My Feature Test Suite"
version: "1.0.0"
params:
  labels:
    profile_display_name: "My Feature"
    group: "custom.my_feature"
    type: "suite"
  requirements:
    cpu_min_cores: 4
    memory_min_gib: 8.0
    storage_min_gib: 5.0
    os_type:
      - "linux"
    docker_required: false

suites:
  - name: "my_domain"
    sub_suites:
      - name: "my_feature"
        tests:
          test_my_feature:
            params:
              # Data-collection entry: metrics collected, no pass/fail.
              # Highlight metrics with key_metrics; no kpi_refs needed.
              - test_id: "MYF-FEA-001"
                display_name: "My Feature Test - CPU"
                devices: [cpu]
                timeout: 300
                key_metrics:
                  - inference_time
                  - accuracy

              - test_id: "MYF-FEA-002"
                display_name: "My Feature Test - GPU"
                devices: [igpu]
                timeout: 300
                key_metrics:
                  - inference_time
                requirements:
                  igpu_required: true
```

For a **qualification** profile, the same test function and the same KPI
definitions are reused. Add `kpi_refs` to enforce specific KPIs and
`kpi_override` to enable and tune the threshold:

```yaml
# src/<package>/configs/profiles/qualifications/my_feature_qual.yml
tests:
  test_my_feature:
    params:
      - test_id: "MYF-FEA-001"
        display_name: "My Feature Test - CPU"
        devices: [cpu]
        timeout: 300
        kpi_refs:
          - inference_time
          - accuracy
        kpi_override:
          inference_time:
            validation:
              reference: 0.3   # tighten to 300 ms for this qualification
              enabled: true
          accuracy:
            validation:
              enabled: true    # enable using the base 0.90 threshold
```

### Optional: Runtime Parameter Overrides via Environment Variables

Some tests support overriding profile parameters at runtime through environment variables. This lets you tune a single parameter — for example, run duration or allocation size — without editing any profile file. This is useful for quick iteration, CI customization, or hardware-specific tuning.

#### Naming convention

All runtime override variables follow this format:

```
ENV_SUITE_<TEST_FILENAME>_<PARAMETER>
```

| Segment | Meaning | Example |
|---|---|---|
| `ENV_SUITE_` | Fixed prefix distinguishing suite-level overrides | — |
| `<TEST_FILENAME>` | Test file name without `test_` and `.py`, uppercased; `-` replaced with `_` | `test_stress_ng.py` → `STRESS_NG` |
| `<PARAMETER>` | Parameter name in uppercase | `DURATION_SECONDS` |

#### When to implement this

Add an env override only when:

- The parameter directly controls test duration or allocation size — values that users commonly need to reduce for faster iteration
- Users may need to adjust it per-system without editing profiles

!!! warning
    Do **not** add overrides for parameters that affect result validity, such as thresholds, device selection, or test logic flags. These variables are for development and advanced testing only — setting them during a qualification run invalidates the results.

#### Implementation pattern

```python
# Module-level constant — single source of truth for the var name
_DURATION_ENV_VAR = "ENV_SUITE_MY_TEST_DURATION_SECONDS"

def _resolve_my_duration(configs: dict) -> int:
    """Resolve test duration, honouring the per-suite env override.

    Priority: ENV_SUITE_MY_TEST_DURATION_SECONDS env var > profile
    ``my_duration_seconds`` param > 60 default. Invalid overrides are
    ignored with a warning.
    """
    default = max(int(configs.get("my_duration_seconds", 60)), 1)

    raw_override = os.environ.get(_DURATION_ENV_VAR)
    if raw_override is None:
        return default

    try:
        override = max(int(raw_override), 1)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring non-integer %s=%r; using profile default %s",
            _DURATION_ENV_VAR, raw_override, default,
        )
        return default

    logger.info("Duration override from %s: %ss", _DURATION_ENV_VAR, override)
    return override
```

Write the resolved value back into `configs` before calling `execute_test_with_cache` so that runs with different override values are cached independently:

```python
duration = _resolve_my_duration(configs)
configs["my_duration_seconds"] = duration  # isolate cache key per override value

results = execute_test_with_cache(
    cached_result=cached_result,
    cache_result=cache_result,
    run_test_func=lambda: _execute_logic(duration, ...),
    test_name=test_name,
    configs=configs,
)
```

For the complete list of env vars supported by built-in suites, see [Runtime Environment Overrides](advanced.md#runtime-environment-overrides).

---

### Step 5: Run Your Test

```bash
# List available profiles
esq list

# Run your profile
esq -v run --profile profile.suite.my_feature

# Run a specific test using a filter
esq -d run --profile profile.suite.my_feature --filter test_id=MYF-FEA-001

# Run without cache (force fresh execution)
esq -d run -nc --profile profile.suite.my_feature

# Override a runtime parameter for this run only
ENV_SUITE_MY_TEST_DURATION_SECONDS=60 esq -v run --profile profile.suite.my_feature
```

---

## Test Execution Pattern

All Intel® ESQ tests follow a standardized 7-step pattern for consistency and reliability. Production tests wrap execution in `try` / `except` / `finally` so that:

- a failed test ends with `pytest.fail(...)` and a clear message,
- an interrupt (Ctrl+C) is surfaced as a proper outcome instead of a broken run,
- validation and summarization always run so partial results are still reported,
- cleanup always runs, even on failure.

Use the example below as the base structure for any new test.

### Full Pattern Example

```python
import logging

import pytest

from sysagent.utils.core import Metrics, Result

logger = logging.getLogger(__name__)


def test_example(
    request,
    configs,
    cached_result,
    cache_result,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """Standard test pattern with failure and interrupt handling."""

    # ================================================================
    # STEP 1: Extract Parameters
    # ================================================================
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)
    timeout = configs.get("timeout", 300)
    devices = configs.get("devices", ["cpu"])

    # Qualification tests fail the run on interrupt; data-collection tests
    # surface it as a runtime error instead. This flag drives that decision.
    is_qualification = configs.get("labels", {}).get("type") == "qualification"

    logger.info(f"Starting test: {test_display_name}")

    # ================================================================
    # STEP 2: Validate System Requirements
    # ================================================================
    validate_system_requirements_from_configs(configs)

    # Prerequisite guards — placed after is_qualification is resolved so each
    # guard can call pytest.fail (qualification) or pytest.skip (data-collection)
    # as appropriate.  A silent skip in a qualification run hides a real system
    # deficiency; pytest.fail surfaces it clearly.
    #
    # Pattern for every prerequisite check:
    #   _outcome = pytest.fail if is_qualification else pytest.skip
    #   _outcome("reason message")
    #
    # Example — binary or capability not available:
    #   if not shutil.which("my_tool"):
    #       _outcome = pytest.fail if is_qualification else pytest.skip
    #       _outcome("my_tool not found. Install it and re-run.")

    # Outcome tracking — initialized before the try block so the finally and
    # post-run blocks can always reference them.
    results = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    def cleanup():
        """Release any resources (containers, temp files, processes)."""
        # Add teardown logic here; safe to call more than once.
        pass

    try:
        # ============================================================
        # STEP 3: Prepare Assets/Dependencies
        # ============================================================
        def prepare_assets():
            return Result(
                name=f"{test_id} - Asset Preparation",
                metadata={"status": "completed"},
            )

        prepare_test(
            test_name=test_name,
            prepare_func=prepare_assets,
            configs=configs,
            name="Assets",
        )

        # ============================================================
        # STEP 4: Execute Test Logic (with caching)
        # ============================================================
        def execute_logic():
            result = Result(name=f"{test_id} - {test_display_name}")

            # Collect domain-specific metrics.
            # Every result must designate exactly ONE key metric — the most
            # representative value shown in summary views. Use set_key_metric()
            # after populating result.metrics to enforce this.
            result.metrics["throughput"] = Metrics(value=1234.5, unit="ops/sec")
            result.metrics["latency_ms"] = Metrics(value=42.0, unit="ms")
            result.set_key_metric("throughput")  # exactly one key metric per result

            # Add domain-specific parameters for the Allure report.
            # Do NOT add test_id or display_name — those are already encoded in
            # result.name and are auto-populated by the framework.
            result.parameters["Devices"] = ", ".join(devices)

            result.metadata["status"] = True
            result.update_timestamps()
            return result

        results = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=execute_logic,
            test_name=test_name,
            configs=configs,
        )

        # Mark the test failed when the execution reported a non-passing status.
        if not results.metadata.get("status", False):
            test_failed = True
            failure_message = results.metadata.get("error", f"{test_display_name} failed")

    except KeyboardInterrupt:
        # Ctrl+C or run cancellation — record it and re-surface after cleanup.
        test_interrupted = True
        failure_message = "Interrupt detected during test execution"
        logger.error(failure_message)

    except Exception as error:
        # Any unexpected error — capture the message and continue to reporting.
        test_failed = True
        failure_message = f"Unexpected error during test execution: {error}"
        logger.error(failure_message, exc_info=True)

    finally:
        # Always release resources, even on failure or interrupt.
        cleanup()

    # ================================================================
    # Ensure a result object always exists so validation and
    # summarization can record the outcome — even when execution was
    # interrupted before producing results.
    # ================================================================
    if results is None:
        results = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "Test did not complete"},
            metrics={},
        )

    # ================================================================
    # STEP 5: Validate Results Against KPIs (optional)
    # Only enforced when kpi_refs are configured for the test. Wrapped so a
    # validation error never masks the underlying test outcome.
    # ================================================================
    try:
        validate_test_results(
            results=results,
            configs=configs,
            get_kpi_config=get_kpi_config,
            test_name=test_name,
        )
    except Exception as validation_error:
        logger.error(f"Validation failed: {validation_error}")

    # ================================================================
    # STEP 6: Generate Summary (always runs)
    # ================================================================
    try:
        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as summary_error:
        logger.error(f"Summarization failed: {summary_error}", exc_info=True)

    cache_result(results)

    logger.info(f"Test completed: {test_display_name}")

    # ================================================================
    # STEP 7: Surface the outcome
    # Report interrupts and failures as proper pytest results instead of
    # leaving a broken/errored status behind.
    # ================================================================
    if test_interrupted:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
```

### Execution Flow Diagram

```
1. Parameter Extraction
   ├─> Read test_id, display_name, timeout, devices
   └─> Configure logging

2. System Validation
   ├─> Check CPU / Memory / Storage
   ├─> Check device availability
   ├─> Check software dependencies and tool prerequisites
   ├─> Data-collection profile → pytest.skip when prerequisites not met
   └─> Qualification profile  → pytest.fail when prerequisites not met

3. Asset Preparation
   ├─> Download models and videos
   ├─> Build Docker* images
   └─> Set up test environment

4. Test Execution (Cached)
   ├─> Check cache for existing results
   ├─> Cache hit  → return cached results
   ├─> Cache miss → execute test logic
   └─> On error/interrupt → record message, run cleanup, continue

5. KPI Validation (optional — only when kpi_refs are configured)
   ├─> Load KPI configurations
   ├─> Compare results vs. thresholds
   └─> Mark passed / failed / skipped

6. Result Summarization (always runs)
   ├─> Generate JSON summary
   ├─> Create Allure attachments
   └─> Generate visualizations

7. Surface Outcome
   ├─> Cleanup already ran in finally
   ├─> Interrupt  → pytest.fail (qualification) / RuntimeError (collection)
   └─> Failure    → pytest.fail with failure_message
```

---

## Next Steps

- [Configuration](configuration.md) — Profile YAML structure and `config.yml` reference
- [Fixtures Reference](fixtures.md) — All available pytest fixtures
- [Results & Metrics](results-metrics.md) — Working with `Result` and `Metrics` objects
- [KPI Validation](kpi-validation.md) — Defining and validating KPI thresholds
- [Runtime Environment Overrides](advanced.md#runtime-environment-overrides) — All available env vars and example values
