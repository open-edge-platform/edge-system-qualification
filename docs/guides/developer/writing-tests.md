# Writing Tests

This page walks through creating a test from scratch and explains the standard 7-step execution pattern used across all Intel® ESQ tests.

Tests can live in the `esq` extension package or in any custom extension package that builds on the `sysagent` core framework. Both follow identical structure and conventions — only the package path differs. See [Framework Architecture](index.md#framework-architecture) for an overview of the dual-package design.

---

## Creating Your First Test

### Step 1: Create the Test File

Create a test file in the suite directory of your target extension package:

```
# In the ESQ extension package:
src/esq/suites/
└── my_domain/
    └── my_feature/
        ├── config.yml          # KPI and test configurations
        └── test_my_feature.py  # Test implementation

# In a custom extension package:
src/your_package/suites/
└── my_domain/
    └── my_feature/
        ├── config.yml          # KPI and test configurations
        └── test_my_feature.py  # Test implementation
```

### Step 2: Basic Test Structure

```python
# src/<package>/suites/my_domain/my_feature/test_my_feature.py
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from sysagent.utils.core import Result, Metrics

logger = logging.getLogger(__name__)


def test_my_feature(
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
    """
    My custom feature test.

    This test demonstrates the standard 7-step pattern for ESQ tests.
    """
    # Step 1: Extract parameters from configs
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)
    timeout = configs.get("timeout", 300)
    devices = configs.get("devices", ["cpu"])

    logger.info(f"Starting test: {test_display_name}")

    # Step 2: Validate system requirements
    validate_system_requirements_from_configs(configs)

    # Step 3: Prepare assets/dependencies
    def prepare_assets():
        """Prepare test assets and dependencies."""
        return Result(
            name=f"{test_id} - Asset Preparation",
            metadata={"status": "completed"}
        )

    prepare_test(
        test_name=test_name,
        prepare_func=prepare_assets,
        configs=configs,
        name="Assets"
    )

    # Step 4: Execute test logic (with caching)
    def execute_logic():
        """Execute the main test logic."""
        results = Result(name=f"{test_id} - {test_display_name}")

        inference_time = 0.123  # seconds
        accuracy = 0.95         # 95%

        results.metrics["inference_time"] = Metrics(value=inference_time, unit="seconds")
        results.metrics["accuracy"] = Metrics(value=accuracy, unit="percentage")

        results.parameters["Test ID"] = test_id
        results.parameters["Display Name"] = test_display_name
        results.parameters["Devices"] = ", ".join(devices)

        results.update_timestamps()
        return results

    results = execute_test_with_cache(
        cached_result=cached_result,
        cache_result=cache_result,
        run_test_func=execute_logic,
        test_name=test_name,
        configs=configs
    )

    # Step 5: Validate results against KPIs (optional)
    # Only executes when kpi_refs are defined in the test configuration
    if configs.get("kpi_refs"):
        validation_results = validate_test_results(
            results=results,
            configs=configs,
            get_kpi_config=get_kpi_config,
            test_name=test_name
        )

    # Step 6: Generate summary
    summarize_test_results(
        results=results,
        test_name=test_name,
        configs=configs,
        get_kpi_config=get_kpi_config
    )
```

### Step 3: Create the Configuration File

```yaml
# src/<package>/suites/my_domain/my_feature/config.yml
# (e.g., src/esq/suites/my_domain/my_feature/config.yml)
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Optional: define KPI thresholds.
# Only required when tests need pass/fail validation against performance targets.
# Remove this block entirely for data-collection-only suites.
kpi:
  inference_time:
    name: "Inference Time"
    type: "numeric"
    validation:
      operator: "lte"   # less than or equal
      reference: 0.5    # 500 ms max
      enabled: true
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
      enabled: true
    unit: "percentage"
    severity: "critical"
    description: "Model prediction accuracy"
    default_value: 0.0

tests:
  test_my_feature:
    params:
      - test_id: "MF-001"
        display_name: "My Feature Test - CPU"
        devices: [cpu]
        timeout: 300
        # Optional: reference KPIs defined above. Omit for collection-only tests.
        kpi_refs:
          - inference_time
          - accuracy
        requirements:
          cpu_min_cores: 4
          memory_min_gib: 8.0
          docker_required: false

      - test_id: "MF-002"
        display_name: "My Feature Test - GPU"
        devices: [igpu]
        timeout: 300
        # No kpi_refs: metrics are collected but not validated
        requirements:
          igpu_required: true
          memory_min_gib: 8.0
```

### Step 4: Create a Profile

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
              - test_id: "MF-001"
                display_name: "My Feature Test - CPU"
                devices: [cpu]

              - test_id: "MF-002"
                display_name: "My Feature Test - GPU"
                devices: [igpu]
                requirements:
                  igpu_required: true
```

### Step 5: Run Your Test

```bash
# List available profiles
esq list

# Run your profile
esq -v run --profile profile.suite.my_feature

# Run a specific test using a filter
esq -d run --profile profile.suite.my_feature --filter test_id=MF-001

# Run without cache (force fresh execution)
esq -d run -nc --profile profile.suite.my_feature
```

---

## Test Execution Pattern

All ESQ tests follow a standardized 7-step pattern for consistency and reliability.

### Full Pattern Example

```python
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
    """Standard test pattern."""

    # ================================================================
    # STEP 1: Extract Parameters
    # ================================================================
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)
    timeout = configs.get("timeout", 300)
    devices = configs.get("devices", ["cpu"])

    logger.info(f"Starting test: {test_display_name}")

    # ================================================================
    # STEP 2: Validate System Requirements
    # ================================================================
    validate_system_requirements_from_configs(configs)

    # ================================================================
    # STEP 3: Prepare Assets/Dependencies
    # ================================================================
    def prepare_assets():
        return Result(
            name=f"{test_id} - Asset Preparation",
            metadata={"status": "completed"}
        )

    prepare_test(
        test_name=test_name,
        prepare_func=prepare_assets,
        configs=configs,
        name="Assets"
    )

    # ================================================================
    # STEP 4: Execute Test Logic (with caching)
    # ================================================================
    def execute_logic():
        results = Result(name=f"{test_id} - {test_display_name}")

        results.metrics["throughput"] = Metrics(value=1234.5, unit="ops/sec")

        results.parameters["Test ID"] = test_id
        results.parameters["Devices"] = ", ".join(devices)

        results.update_timestamps()
        return results

    results = execute_test_with_cache(
        cached_result=cached_result,
        cache_result=cache_result,
        run_test_func=execute_logic,
        test_name=test_name,
        configs=configs
    )

    # ================================================================
    # STEP 5: Validate Results Against KPIs (optional)
    # Only executes when kpi_refs are defined in the test configuration.
    # Skip this block entirely for data-collection-only tests.
    # ================================================================
    if configs.get("kpi_refs"):
        validation_results = validate_test_results(
            results=results,
            configs=configs,
            get_kpi_config=get_kpi_config,
            test_name=test_name
        )

    # ================================================================
    # STEP 6: Generate Summary
    # ================================================================
    summarize_test_results(
        results=results,
        test_name=test_name,
        configs=configs,
        get_kpi_config=get_kpi_config
    )

    # ================================================================
    # STEP 7: Complete (implicit)
    # ================================================================
    # The framework automatically:
    # - Saves results to JSON
    # - Generates the Allure report
    # - Caches results (if enabled)
```

### Execution Flow Diagram

```
1. Parameter Extraction
   ├─> Read test_id, display_name, timeout, devices
   └─> Configure logging

2. System Validation
   ├─> Check CPU / Memory / Storage
   ├─> Check device availability
   ├─> Check software dependencies
   └─> Skip test if requirements not met

3. Asset Preparation
   ├─> Download models and videos
   ├─> Build Docker* images
   └─> Set up test environment

4. Test Execution (Cached)
   ├─> Check cache for existing results
   ├─> Cache hit  → return cached results
   └─> Cache miss → execute test logic

5. KPI Validation (optional — only when kpi_refs are configured)
   ├─> Load KPI configurations
   ├─> Compare results vs. thresholds
   └─> Mark passed / failed / skipped

6. Result Summarization
   ├─> Generate JSON summary
   ├─> Create Allure attachments
   └─> Generate visualizations

7. Cleanup & Reporting
   ├─> Save results to file
   ├─> Update Allure report
   └─> Clear temporary resources
```

---

## Next Steps

- [Configuration](configuration.md) — Profile YAML structure and `config.yml` reference
- [Fixtures Reference](fixtures.md) — All available pytest fixtures
- [Results & Metrics](results-metrics.md) — Working with `Result` and `Metrics` objects
- [KPI Validation](kpi-validation.md) — Defining and validating KPI thresholds
