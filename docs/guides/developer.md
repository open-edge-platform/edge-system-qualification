# Developer Guide

Comprehensive guide for developers integrating their own pytest tests into the Intel® ESQ framework.

**Quick Reference**: Looking for a quick cheat sheet? See the [Developer Quick Reference](developer-quick-reference.md) for templates and common commands.

## Table of Contents

- [Overview](#overview)
- [Framework Architecture](#framework-architecture)
- [Creating Your First Test](#creating-your-first-test)
- [Configuration System](#configuration-system)
- [Available Fixtures](#available-fixtures)
- [System Requirements Flags](#system-requirements-flags)
- [Test Execution Pattern](#test-execution-pattern)
- [Working with Results and Metrics](#working-with-results-and-metrics)
- [KPI Validation](#kpi-validation)
- [Asset Management](#asset-management)
- [Modular Telemetry](#modular-telemetry)
- [Allure3 Report Customization](#allure3-report-customization)
- [Best Practices](#best-practices)
- [Advanced Topics](#advanced-topics)

---

## Overview

The Intel® ESQ framework provides a comprehensive pytest-based testing infrastructure with:

- **Automatic test parameterization** from YAML configuration files
- **Built-in fixtures** for caching, validation, and reporting
- **System requirement validation** with reusable flags
- **KPI-based test validation** with flexible configuration
- **Asset management** for models, videos, and files
- **Allure reporting** with rich visualizations
- **Docker integration** for containerized tests
- **Modular telemetry** for automatic background collection of CPU, memory, power, GPU, and NPU metrics during test execution — enabled entirely through profile YAML, requiring no test code changes

This guide will help you integrate your own tests into this framework and leverage its powerful features.

---

## Framework Architecture

### Dual-Package Structure

```
src/
├── sysagent/           # Core framework (reusable infrastructure)
│   ├── cli.py          # Main CLI entry point
│   ├── utils/
│   │   ├── plugins/    # Pytest fixtures and hooks
│   │   ├── core/       # Result, Metrics, Cache classes
│   │   ├── testing/    # System validation utilities
│   │   └── config/     # Configuration loaders
│   └── suites/         # Core test suites (examples)
│
└── esq/                # ESQ-specific extensions
    ├── suites/         # Domain-specific test suites
    │   ├── ai/         # AI tests (vision, audio, gen)
    │   ├── media/      # Media processing tests
    │   ├── system/     # System-level tests
    │   └── vertical/   # Vertical-specific tests
    └── configs/
        └── profiles/   # Test profiles (qualifications, suites, verticals)
```

### Test Discovery Flow

```
Profile YAML → Consolidator → Pytest Plugin → Test Function
     ↓              ↓              ↓               ↓
  params       merge with     filter by      configs fixture
               defaults    function name    (all merged params)
```

---

## Creating Your First Test

### Step 1: Create Test File

Create a test file in an appropriate suite directory:

```
src/esq/suites/
└── my_domain/
    └── my_feature/
        ├── config.yml          # KPI and test configurations
        └── test_my_feature.py  # Your test implementation
```

### Step 2: Basic Test Structure

```python
# src/esq/suites/my_domain/my_feature/test_my_feature.py
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
        # Download models, prepare data, etc.
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
        
        # Your test implementation here
        # Example: Run inference, process data, etc.
        inference_time = 0.123  # seconds
        accuracy = 0.95  # 95%
        
        # Add metrics to results
        results.metrics["inference_time"] = Metrics(
            value=inference_time,
            unit="seconds"
        )
        results.metrics["accuracy"] = Metrics(
            value=accuracy,
            unit="percentage"
        )
        
        # Add parameters
        results.parameters["Test ID"] = test_id
        results.parameters["Display Name"] = test_display_name
        results.parameters["Devices"] = ", ".join(devices)
        
        # Update timestamps
        results.update_timestamps()
        
        return results
    
    results = execute_test_with_cache(
        cached_result=cached_result,
        cache_result=cache_result,
        run_test_func=execute_logic,
        test_name=test_name,
        configs=configs
    )
    
    # Step 5: Validate results (if qualification profile)
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

### Step 3: Create Configuration File

```yaml
# src/esq/suites/my_domain/my_feature/config.yml
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# KPI definitions for this test suite
kpi:
  inference_time:
    name: "Inference Time"
    type: "numeric"
    validation:
      operator: "lte"  # less than or equal
      reference: 0.5   # 500ms max
      enabled: true
    unit: "seconds"
    severity: "major"
    description: "Time taken for model inference"
    default_value: 999.0
  
  accuracy:
    name: "Model Accuracy"
    type: "numeric"
    validation:
      operator: "gte"  # greater than or equal
      reference: 0.90  # 90% minimum
      enabled: true
    unit: "percentage"
    severity: "critical"
    description: "Model prediction accuracy"
    default_value: 0.0

# Test configurations
tests:
  test_my_feature:
    params:
      - test_id: "MF-001"
        display_name: "My Feature Test - CPU"
        devices: [cpu]
        timeout: 300
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
        kpi_refs:
          - inference_time
          - accuracy
        requirements:
          igpu_required: true
          memory_min_gib: 8.0
```

### Step 4: Create a Profile

```yaml
# src/esq/configs/profiles/suites/my_feature.yml
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

# Run specific test with filter
esq -d run --profile profile.suite.my_feature --filter test_id=MF-001

# Run without cache
esq -d run -nc --profile profile.suite.my_feature
```

---

## Configuration System

### Profile Structure

Profiles define test execution plans with parameters and requirements:

```yaml
name: "profile.suite.example"
description: "Example test suite"
version: "1.0.0"

params:
  labels:
    profile_display_name: "Example"
    group: "example.group"
    type: "suite"  # or "qualification" or "vertical"
  
  requirements:
    # Hardware requirements
    cpu_min_cores: 4
    memory_min_gib: 8.0
    storage_min_gib: 10.0
    
    # Software requirements
    os_type: ["linux"]
    docker_required: true
    
    # Device requirements
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

### Profile to Test Mapping

The framework automatically maps profile configuration to test files:

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

### Test Configuration (config.yml)

Each test directory can have a `config.yml` defining KPIs and default parameters:

```yaml
# KPI definitions
kpi:
  metric_name:
    name: "Human Readable Name"
    type: "numeric"  # or "string", "boolean", "list"
    validation:
      operator: "gte"  # gte, lte, eq, ne, gt, lt
      reference: 100.0
      enabled: true
    unit: "unit_name"
    severity: "critical"  # critical, major, normal, minor
    description: "KPI description"
    default_value: 0.0

# Test-specific parameters
tests:
  test_function_name:
    params:
      - test_id: "T001"
        display_name: "Test Name"
        kpi_refs:
          - metric_name
```

---

## Available Fixtures

The framework provides comprehensive pytest fixtures automatically available to all tests.

### Core Fixtures

#### `request`
- **Type**: pytest.FixtureRequest
- **Scope**: function
- **Description**: Standard pytest request object for accessing test metadata
- **Usage**:
  ```python
  test_name = request.node.name.split("[")[0]
  ```

#### `configs`
- **Type**: Dict[str, Any]
- **Scope**: function
- **Description**: Test configuration parameters merged from profile and config.yml
- **Usage**:
  ```python
  test_id = configs.get("test_id", "T0000")
  devices = configs.get("devices", ["cpu"])
  timeout = configs.get("timeout", 300)
  ```

### Cache Fixtures

#### `cached_result`
- **Type**: Callable[[Optional[Dict]], Optional[Union[Result, Dict]]]
- **Scope**: function
- **Description**: Retrieves cached test results if available
- **Usage**:
  ```python
  cached_data = cached_result(cache_configs={"key": "value"})
  if cached_data:
      return cached_data
  ```

#### `cache_result`
- **Type**: Callable[[Union[Result, Dict], Optional[Dict]], None]
- **Scope**: function
- **Description**: Stores test results in cache
- **Usage**:
  ```python
  cache_result(results, cache_configs={"key": "value"})
  ```

### Validation Fixtures

#### `validate_system_requirements_from_configs`
- **Type**: Callable[[Dict[str, Any]], None]
- **Scope**: function
- **Description**: Validates system meets test requirements from configs
- **Usage**:
  ```python
  validate_system_requirements_from_configs(configs)
  # Raises pytest.skip if requirements not met
  ```

#### `validate_test_results`
- **Type**: Callable[..., Dict[str, Any]]
- **Scope**: function
- **Description**: Validates test results against KPI thresholds
- **Parameters**:
  - `results`: Result object or dict
  - `configs`: Test configuration
  - `get_kpi_config`: KPI config retrieval function
  - `test_name`: Test identifier
  - `mode`: Validation mode ("all" or "any")
- **Usage**:
  ```python
  validation_results = validate_test_results(
      results=results,
      configs=configs,
      get_kpi_config=get_kpi_config,
      test_name=test_name
  )
  # Returns: {"passed": bool, "validations": {...}, "skipped": bool}
  ```

### Execution Fixtures

#### `execute_test_with_cache`
- **Type**: Callable[..., Union[Result, Dict]]
- **Scope**: function
- **Description**: Executes test with automatic caching support
- **Parameters**:
  - `cached_result`: Cached result fixture
  - `cache_result`: Cache result fixture
  - `run_test_func`: Function to execute test logic
  - `test_name`: Test identifier
  - `configs`: Test configuration
  - `cache_configs`: Optional cache-specific configs
  - `name`: Step name (default: "Analysis")
- **Usage**:
  ```python
  results = execute_test_with_cache(
      cached_result=cached_result,
      cache_result=cache_result,
      run_test_func=lambda: execute_my_test(),
      test_name=test_name,
      configs=configs
  )
  ```

#### `prepare_test`
- **Type**: Callable[..., Any]
- **Scope**: function
- **Description**: Prepares test assets with progress tracking
- **Parameters**:
  - `test_name`: Test identifier
  - `prepare_func`: Function to execute preparation
  - `configs`: Test configuration
  - `name`: Step name (default: "Preparation")
- **Usage**:
  ```python
  prepare_test(
      test_name=test_name,
      prepare_func=lambda: download_models(),
      configs=configs,
      name="Assets"
  )
  ```

### Reporting Fixtures

#### `summarize_test_results`
- **Type**: Callable[..., None]
- **Scope**: function
- **Description**: Generates test result summary with Allure attachments
- **Parameters**:
  - `results`: Result object
  - `test_name`: Test identifier
  - `configs`: Test configuration (optional)
  - `get_kpi_config`: KPI config function (optional)
  - `iteration_data`: Iteration-level data (optional)
  - `enable_visualizations`: Enable charts (default: False)
- **Usage**:
  ```python
  summarize_test_results(
      results=results,
      test_name=test_name,
      configs=configs,
      get_kpi_config=get_kpi_config
  )
  ```

### Suite Fixtures

#### `suite_configs`
- **Type**: Dict[str, Any]
- **Scope**: function
- **Description**: Loads suite-level configuration from config.yml
- **Usage**:
  ```python
  kpis = suite_configs.get("kpi", {})
  ```

#### `get_kpi_config`
- **Type**: Callable[[str], Optional[Dict[str, Any]]]
- **Scope**: function
- **Description**: Retrieves KPI configuration by name
- **Usage**:
  ```python
  kpi_config = get_kpi_config("inference_time")
  if kpi_config:
      reference = kpi_config["validation"]["reference"]
  ```

---

## System Requirements Flags

The framework provides reusable requirement validation flags. Add these to your profile or test parameters under the `requirements` key.

### Hardware Requirements

#### CPU Requirements

```yaml
requirements:
  # Minimum CPU cores
  cpu_min_cores: 4
  
  # Minimum CPU threads
  cpu_min_threads: 8
  
  # CPU brand requirements
  cpu_xeon_required: true     # Requires Intel® Xeon® processor
  cpu_core_required: true     # Requires Intel® Core™ processor (includes Ultra Desktop)
  cpu_ultra_required: true    # Requires Intel® Ultra processor
  
  # CPU socket count
  cpu_min_sockets: 1
  cpu_max_sockets: 2
```

#### Memory Requirements

```yaml
requirements:
  # Minimum total memory
  memory_min_gib: 8.0
  
  # Maximum memory (for constrained tests)
  memory_max_gib: 32.0
```

#### Storage Requirements

```yaml
requirements:
  # Total storage capacity
  storage_total_min_gib: 64.0
  
  # Free storage for test execution
  storage_min_gib: 10.0
```

#### Device Requirements

```yaml
requirements:
  # Integrated GPU (iGPU)
  igpu_required: true
  igpu_min_count: 1
  igpu_max_count: 1
  
  # Discrete GPU (dGPU)
  dgpu_required: true
  dgpu_min_count: 1
  dgpu_max_count: 4
  
  # Neural Processing Unit (NPU)
  npu_required: true
  npu_min_count: 1
  npu_max_count: 1
```

**Device Detection Logic:**
- The framework automatically detects Intel devices using OpenVINO
- Only Intel devices detected by OpenVINO are counted
- Non-Intel devices are ignored for requirements validation
- If Intel devices exist but aren't detected by OpenVINO, helpful error messages suggest driver installation

### Software Requirements

#### Operating System

```yaml
requirements:
  # Supported OS types
  os_type:
    - "linux"
    - "windows"
  
  # Specific OS versions
  os_version_min: "20.04"  # For Ubuntu
  os_version_max: "24.04"
```

#### Docker Requirements

```yaml
requirements:
  # Docker daemon required
  docker_required: true
  
  # Minimum Docker version
  docker_version_min: "20.10.0"
```

#### Software Dependencies

```yaml
requirements:
  # Python version
  python_version_min: "3.10"
  
  # Other software (validated via command presence)
  software_required:
    - name: "git"
      command: "git --version"
    - name: "ffmpeg"
      command: "ffmpeg -version"
```

### Example: Complete Requirements Block

```yaml
requirements:
  # Hardware
  cpu_min_cores: 8
  cpu_xeon_required: true
  memory_min_gib: 16.0
  storage_min_gib: 20.0
  dgpu_required: true
  dgpu_min_count: 2
  
  # Software
  os_type: ["linux"]
  docker_required: true
  python_version_min: "3.10"
  
  # Dependencies
  software_required:
    - name: "ffmpeg"
      command: "ffmpeg -version"
```

### Using Requirements in Tests

Requirements are automatically validated when you call:

```python
validate_system_requirements_from_configs(configs)
```

This will:
1. Check all specified requirements against system capabilities
2. Skip the test if requirements aren't met (using `pytest.skip`)
3. Provide detailed error messages for failed requirements
4. Log fix suggestions for common issues

**Example Validation Output:**
```
Validation failed for profile: my-profile
  CPU Requirements:
    ✗ CPU cores >= 8 (Actual: 4 cores)
  Device Requirements:
    ✗ dGPU required (Actual: 1 Intel dGPU found but not detected by OpenVINO)
    
Suggestions:
  - Upgrade to system with more CPU cores
  - Install Intel GPU drivers: sudo apt install intel-gpu-tools
```

---

## Test Execution Pattern

The framework follows a standardized 7-step pattern for all tests:

### Standard Test Pattern

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
    
    # ====================================================================
    # STEP 1: Extract Parameters
    # ====================================================================
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", test_name)
    timeout = configs.get("timeout", 300)
    devices = configs.get("devices", ["cpu"])
    
    logger.info(f"Starting test: {test_display_name}")
    
    # ====================================================================
    # STEP 2: Validate System Requirements
    # ====================================================================
    # Automatically skips test if requirements not met
    validate_system_requirements_from_configs(configs)
    
    # ====================================================================
    # STEP 3: Prepare Assets/Dependencies
    # ====================================================================
    def prepare_assets():
        """Prepare test assets."""
        # Download models, videos, files
        # Build Docker images
        # Set up test environment
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
    
    # ====================================================================
    # STEP 4: Execute Test Logic (with caching)
    # ====================================================================
    def execute_logic():
        """Main test logic."""
        results = Result(name=f"{test_id} - {test_display_name}")
        
        # Your test implementation
        # - Run benchmarks
        # - Execute inference
        # - Collect metrics
        
        # Add metrics
        results.metrics["throughput"] = Metrics(
            value=1234.5,
            unit="ops/sec"
        )
        
        # Add parameters
        results.parameters["Test ID"] = test_id
        results.parameters["Devices"] = ", ".join(devices)
        
        # Update timestamps
        results.update_timestamps()
        
        return results
    
    results = execute_test_with_cache(
        cached_result=cached_result,
        cache_result=cache_result,
        run_test_func=execute_logic,
        test_name=test_name,
        configs=configs
    )
    
    # ====================================================================
    # STEP 5: Validate Results Against KPIs
    # ====================================================================
    validation_results = validate_test_results(
        results=results,
        configs=configs,
        get_kpi_config=get_kpi_config,
        test_name=test_name
    )
    
    # ====================================================================
    # STEP 6: Generate Summary
    # ====================================================================
    summarize_test_results(
        results=results,
        test_name=test_name,
        configs=configs,
        get_kpi_config=get_kpi_config
    )
    
    # ====================================================================
    # STEP 7: Test Complete (implicit)
    # ====================================================================
    # Framework automatically:
    # - Saves results to JSON
    # - Generates Allure report
    # - Caches results (if enabled)
```

### Execution Flow

```
1. Parameter Extraction
   ├─> Read test_id, display_name, timeout, etc.
   └─> Configure logging
   
2. System Validation
   ├─> Check CPU/Memory/Storage
   ├─> Check Device availability
   ├─> Check Software dependencies
   └─> Skip test if requirements not met
   
3. Asset Preparation
   ├─> Download models/videos
   ├─> Build Docker images
   └─> Set up test environment
   
4. Test Execution (Cached)
   ├─> Check cache for existing results
   ├─> If cache hit: return cached results
   └─> If cache miss: execute test logic
   
5. KPI Validation
   ├─> Load KPI configurations
   ├─> Compare results vs thresholds
   └─> Mark passed/failed/skipped
   
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

## Working with Results and Metrics

### Result Class

The `Result` dataclass provides structured test result handling:

```python
from sysagent.utils.core import Result, Metrics

# Create result object
results = Result(
    name="T001 - My Test",
    parameters={
        "Test ID": "T001",
        "Device": "CPU",
        "Batch Size": 8
    },
    metrics={
        "throughput": Metrics(value=1234.5, unit="ops/sec"),
        "latency": Metrics(value=0.123, unit="seconds")
    },
    metadata={
        "status": "completed",
        "custom_field": "custom_value"
    }
)

# Update timestamps (auto-calculates duration)
results.update_timestamps()

# Convert to dictionary for serialization
results_dict = results.to_dict()
```

### Result Fields

| Field | Type | Purpose |
|-------|------|---------|
| `name` | `str` | Test identifier (e.g. `"T001 - My Test"`) |
| `parameters` | `Dict[str, Any]` | Human-readable test configuration shown in reports |
| `metrics` | `Dict[str, Metrics]` | Measured performance values with units |
| `metadata` | `Dict[str, Any]` | Flat key-value pairs for high-level reporting (auto-populated with timestamps and KPI status) |
| `extended_metadata` | `Dict[str, Any]` | Structured complex objects for programmatic access (not constrained to simple key-value pairs) |
| `kpis` | `Dict[str, Any]` | KPI configurations and validation results |

### `metadata` vs `extended_metadata`

Use **`metadata`** for simple property-value pairs that appear in Allure and the JSON summary as human-readable fields:

```python
results.metadata["model_version"] = "1.0.0"
results.metadata["benchmark_duration_seconds"] = 48.4
results.metadata["status"] = True
```

Use **`extended_metadata`** for structured or complex objects — nested dicts, lists, time-series data — that are intended for programmatic analysis rather than direct display:

```python
# Store a structured benchmark breakdown
results.extended_metadata["device_breakdown"] = {
    "CPU": {
        "throughput": 51.2,
        "ttft_ms": 224.0,
        "tpot_ms": 18.5,
        "samples": [...]
    }
}

# Store raw iteration data
results.extended_metadata["iterations"] = [
    {"step": 0, "latency_ms": 220.1},
    {"step": 1, "latency_ms": 218.4},
]
```

> **Note:** Telemetry data collected during test execution is automatically placed in `extended_metadata["telemetry"]` — see [Modular Telemetry](#modular-telemetry).

### Automatic Metadata

The `Result` class automatically includes the following keys in `metadata`:

```python
{
    "created_at": "2025-12-29T10:00:00+00:00",
    "updated_at": "2025-12-29T10:05:30+00:00",
    "total_duration_seconds": 330.0,
    "kpi_validation_status": "passed"  # or "failed", "skipped"
}
```

### Metrics Class

The `Metrics` dataclass structures metric data:

```python
from sysagent.utils.core import Metrics

# Create metric
metric = Metrics(
    value=123.45,
    unit="ms",
    is_key_metric=False
)

# Access metric properties
print(f"Value: {metric.value} {metric.unit}")
```

### Adding Metrics to Results

```python
# Add single metric
results.metrics["accuracy"] = Metrics(
    value=0.95,
    unit="percentage"
)

# Add multiple metrics
results.metrics.update({
    "precision": Metrics(value=0.93, unit="percentage"),
    "recall": Metrics(value=0.91, unit="percentage"),
    "f1_score": Metrics(value=0.92, unit="percentage")
})

# Device-specific metrics
for device in ["cpu", "igpu", "dgpu"]:
    results.metrics[f"throughput_{device}"] = Metrics(
        value=get_throughput(device),
        unit="fps"
    )
```

### Key Metrics

Designate a primary metric for reporting:

```python
# Set key metric
results.set_key_metric("throughput")

# Get current key metric
key_metric_name = results.get_key_metric()
if key_metric_name:
    print(f"Key metric: {key_metric_name}")
```

### Complete Example

```python
def execute_benchmark():
    """Execute benchmark and return results."""
    # Create result object
    results = Result(name=f"{test_id} - {test_display_name}")

    # Run benchmark
    start_time = time.time()
    throughput, latency = run_inference(model, data)
    elapsed = time.time() - start_time

    # Add metrics
    results.metrics["throughput"] = Metrics(
        value=throughput,
        unit="fps",
        is_key_metric=True
    )
    results.metrics["latency"] = Metrics(
        value=latency,
        unit="ms"
    )
    results.metrics["elapsed_time"] = Metrics(
        value=elapsed,
        unit="seconds"
    )

    # Add parameters (shown in Allure report)
    results.parameters["Model"] = "yolo11n"
    results.parameters["Precision"] = "INT8"
    results.parameters["Device"] = "GPU"

    # Add flat metadata (simple key-value, shown in summary)
    results.metadata["model_version"] = "1.0.0"
    results.metadata["benchmark_duration_seconds"] = elapsed

    # Add structured data to extended_metadata (complex objects, not shown directly in summary)
    results.extended_metadata["per_iteration"] = [
        {"step": i, "latency_ms": v} for i, v in enumerate(latency_trace)
    ]

    # Update timestamps
    results.update_timestamps()

    return results
```

---

## KPI Validation

### Defining KPIs

KPIs are defined in `config.yml` files:

```yaml
kpi:
  inference_time:
    name: "Inference Time"
    type: "numeric"
    validation:
      operator: "lte"  # less than or equal
      reference: 0.5   # 500ms threshold
      enabled: true
    unit: "seconds"
    severity: "major"
    description: "Time taken for model inference"
    default_value: 999.0
  
  accuracy:
    name: "Model Accuracy"
    type: "numeric"
    validation:
      operator: "gte"  # greater than or equal
      reference: 0.90  # 90% minimum
      enabled: true
    unit: "percentage"
    severity: "critical"
    description: "Model prediction accuracy"
    default_value: 0.0
  
  status:
    name: "Test Status"
    type: "string"
    validation:
      operator: "eq"  # equals
      reference: "success"
      enabled: true
    unit: ""
    severity: "critical"
    description: "Overall test status"
    default_value: "unknown"
```

### KPI Configuration Fields

| Field | Description | Required | Values |
|-------|-------------|----------|--------|
| `name` | Human-readable name | Yes | String |
| `type` | Data type | Yes | `numeric`, `string`, `boolean`, `list` |
| `validation.operator` | Comparison operator | Yes | `gte`, `lte`, `eq`, `ne`, `gt`, `lt` |
| `validation.reference` | Expected value/threshold | Yes | Varies by type |
| `validation.enabled` | Enable validation | No | `true` (default), `false` |
| `unit` | Measurement unit | No | String (e.g., "ms", "fps") |
| `severity` | Impact level | No | `critical`, `major`, `normal`, `minor` |
| `description` | KPI description | No | String |
| `default_value` | Fallback value | No | Varies by type |

### KPI Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `gte` | Greater than or equal | `actual >= reference` |
| `lte` | Less than or equal | `actual <= reference` |
| `gt` | Greater than | `actual > reference` |
| `lt` | Less than | `actual < reference` |
| `eq` | Equal to | `actual == reference` |
| `ne` | Not equal to | `actual != reference` |

### Referencing KPIs in Tests

```yaml
tests:
  test_inference:
    params:
      - test_id: "INF-001"
        display_name: "Inference Test"
        kpi_refs:
          - inference_time
          - accuracy
          - status
```

### KPI Validation in Tests

```python
# Validate results against KPIs
validation_results = validate_test_results(
    results=results,
    configs=configs,
    get_kpi_config=get_kpi_config,
    test_name=test_name
)

# Check validation status
if validation_results["passed"]:
    logger.info("All KPIs passed")
elif validation_results["skipped"]:
    logger.info(f"KPI validation skipped: {validation_results['skip_reason']}")
else:
    logger.error("Some KPIs failed")
    for kpi_name, kpi_result in validation_results["validations"].items():
        if not kpi_result["passed"]:
            logger.error(f"  {kpi_name}: {kpi_result}")
```

### Validation Result Structure

```python
{
    "passed": False,
    "skipped": False,
    "skip_reason": None,
    "validations": {
        "inference_time": {
            "passed": True,
            "actual_value": 0.45,
            "expected_value": "<= 0.5",
            "unit": "seconds",
            "operator": "lte",
            "severity": "major"
        },
        "accuracy": {
            "passed": False,
            "actual_value": 0.85,
            "expected_value": ">= 0.90",
            "unit": "percentage",
            "operator": "gte",
            "severity": "critical"
        }
    }
}
```

### Disabling KPI Validation

```yaml
kpi:
  optional_metric:
    name: "Optional Metric"
    type: "numeric"
    validation:
      operator: "gte"
      reference: 100.0
      enabled: false  # Metric collected but not validated
```

---

## Asset Management

The framework provides asset management for models, videos, and files.

### Asset Types

#### Video Assets

```yaml
assets:
  - id: "video_sample"
    type: "video"
    name: "sample_1920_1080_30fps.h264"
    url: "https://example.com/video.mp4"
    sha256: "abc123..."
    width: 1920      # Resize to this width
    height: 1080     # Resize to this height
    fps: 30          # Convert to this framerate
    codec: "h264"    # Convert to this codec (h264, h265)
    duration: 30     # Trim to this duration (seconds)
    loop: 120        # Loop video to achieve this duration
```

#### Model Assets

```yaml
assets:
  # Ultralytics model
  - id: "yolo11n"
    type: "model"
    source: "ultralytics"
    precision: "int8"
    format: "pt"
    export_args:
      dynamic: true
      half: true
  
  # KaggleHub model
  - id: "resnet-50"
    type: "model"
    source: "kagglehub"
    precision: "int8"
    format: "openvino"
    kaggle_handle: "google/resnet-v1/tensorFlow2/50-classification"
    convert_args:
      input_shape: [1, 224, 224, 3]
    quantize_args:
      calibration_samples: 512
```

#### File Assets

```yaml
assets:
  - id: "config_file"
    type: "file"
    url: "https://example.com/config.json"
    sha256: "def456..."
    path: "./configs/model.json"
```

### Using Assets in Tests

Assets are automatically prepared when specified in profile configurations. Access them in your test:

```python
# Assets prepared in standard locations
models_dir = os.path.join(data_dir, "models")
videos_dir = os.path.join(data_dir, "videos")

# Model path
model_path = os.path.join(models_dir, "yolo11n", "int8", "yolo11n.xml")

# Video path
video_path = os.path.join(videos_dir, "sample_1920_1080_30fps.h264")
```

---

## Modular Telemetry

The framework automatically collects system metrics as a background daemon thread during test execution. No test code changes are required — telemetry is enabled entirely through profile YAML.

### Enabling Telemetry in a Profile

Add a `telemetry` block inside `params` in your profile YAML:

```yaml
params:
  telemetry:
    enabled: true
    interval: 10         # seconds between samples (integer, minimum 1)
    modules:
      - name: cpu_freq          # CPU frequency (current_mhz, min_mhz, max_mhz)
        enabled: true
      - name: cpu_usage         # CPU utilisation (total_percent)
        enabled: true
        thresholds:
          total_percent:
            warning: 95         # log WARNING if CPU usage exceeds 95%
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
      - name: npu_usage         # Intel® NPU busy utilization (%) and memory (MB)
        enabled: true
      - name: npu_freq          # Intel® NPU operating frequency (MHz)
        enabled: true
```

To disable telemetry, remove the block or set `enabled: false`.

**Optional per-module keys:**

| Key | Description |
|-----|-------------|
| `metrics` | Restrict collection to a subset (e.g., `metrics: [current_mhz]`) |
| `thresholds` | Log a `WARNING` when a metric exceeds the configured value (does not fail the test) |
| `chart_type` | Chart hint for the report renderer: `line` (default), `area`, or `bar_vertical` |
| `title` / `scales` | Display labels and units for chart axes |

### Overriding the Telemetry Interval at Runtime

Priority order (highest to lowest):

1. **`--telemetry-interval SECONDS`** CLI option
2. **`CORE_TELEMETRY_INTERVAL`** environment variable
3. **`interval`** in the profile YAML

The value must be a whole number of seconds (minimum 1).

```bash
esq run --profile profile.suite.ai.gen --telemetry-interval 10
CORE_TELEMETRY_INTERVAL=10 esq run --profile profile.suite.ai.gen
```

### Available Modules

| Module | Metrics collected | Notes |
|--------|-------------------|-------|
| `cpu_freq` | `current_mhz`, `min_mhz`, `max_mhz` | |
| `cpu_usage` | `total_percent` | |
| `cpu_temp` | `package_c`, `core_max_c` | |
| `memory_usage` | `used_percent`, `available_gib`, `used_gib` | |
| `package_power` | `package_power_w`, `core_power_w`, `uncore_power_w`, `dram_power_w` | Requires RAPL; run `scripts/system-setup.sh` |
| `gpu_temp` | `gpu_{N}_pkg_c`, `gpu_{N}_vram_c` (per GPU) | Skipped if no Intel® GPU present |
| `gpu_freq` | `gpu_{N}_gt{M}_mhz` (per GPU and GT) | Skipped if no Intel® GPU present |
| `gpu_power` | `gpu_{N}_w`, `gpu_{N}_card_w` (per GPU) | Skipped if no Intel® GPU present |
| `gpu_usage` | `gpu_{N}_render_pct`, `gpu_{N}_compute_pct`, `gpu_{N}_copy_pct`, `gpu_{N}_video_pct`, `gpu_{N}_video_enh_pct` (Arc/xe); `gpu_{N}_gt{M}_pct` (iGPU/i915) | Requires `perf_event_paranoid ≤ 0` for Arc engine metrics; run `scripts/system-setup.sh` |
| `npu_usage` | `npu_{N}_busy_pct`, `npu_{N}_mem_mib` (per NPU) | Requires `intel_vpu` driver |
| `npu_freq` | `npu_{N}_freq_mhz` (per NPU) | Requires `intel_vpu` driver |

### Creating a Custom Module

Implement `BaseTelemetryModule` and register it in `src/esq/utils/telemetry/modules/`.

**Step 1 — Create the module file:**

```python
import time
from sysagent.utils.telemetry.base import BaseTelemetryModule, TelemetrySample

class DiskIoModule(BaseTelemetryModule):
    module_name = "disk_io"

    def is_available(self) -> bool:
        return True  # check dependencies here

    def collect_sample(self) -> TelemetrySample:
        raw = {"read_mb_s": 0.0}  # populate with real readings
        values = self._filter_values(raw)
        sample = TelemetrySample(timestamp=time.time(), values=values)
        self.check_thresholds(values)
        return sample
```

**Step 2 — Register in `src/esq/utils/telemetry/modules/__init__.py`:**

```python
from esq.utils.telemetry.modules.disk_io import DiskIoModule
from sysagent.utils.telemetry.registry import register as _register

_register("disk_io", DiskIoModule)
```

**Step 3 — Enable in a profile:**

```yaml
telemetry:
  enabled: true
  interval: 10
  modules:
    - name: disk_io
      enabled: true
```

---

## Best Practices

### Test Design

1. **Follow the 7-step pattern** - Maintain consistency across all tests
2. **Use descriptive test IDs** - Format: `{SUITE}-{NUM}` (e.g., `VSN-001`)
3. **Leverage caching** - Tests should support both cached and non-cached execution
4. **Validate requirements early** - Call `validate_system_requirements_from_configs` first
5. **Use proper logging** - Log important steps at appropriate levels
6. **Handle cleanup** - Always clean up resources (containers, files, processes)

### Configuration

1. **Define clear KPIs** - Set realistic thresholds based on baseline measurements
2. **Use requirement flags** - Leverage existing flags instead of custom validation
3. **Document parameters** - Add descriptions to test parameters
4. **Version profiles** - Include version numbers in profile configurations
5. **Organize profiles** - Use appropriate directories (qualifications/, suites/, verticals/)

### Error Handling

1. **Use pytest mechanisms** - `pytest.skip()`, `pytest.fail()`, `pytest.xfail()`
2. **Log errors clearly** - Include context and suggested fixes
3. **Attach diagnostics** - Use Allure attachments for logs and screenshots
4. **Clean up on failure** - Use try/finally blocks for cleanup

### Performance

1. **Cache expensive operations** - Model downloads, conversions, compilations
2. **Use timeouts** - Set appropriate timeouts for long-running operations
3. **Parallelize when possible** - Use thread pools for multi-device tests
4. **Monitor resources** - Track memory and storage usage

### Security

1. **Validate inputs** - Use allow-lists for user-provided values
2. **Avoid shell=True** - Use list-based subprocess calls
3. **Set file permissions** - Use restrictive permissions (0o750, 0o770)
4. **Handle secrets safely** - Never log sensitive information

---

## Advanced Topics

### Custom Fixtures

Create test-specific fixtures in a `conftest.py` file:

```python
# src/esq/suites/my_domain/conftest.py
import pytest

@pytest.fixture(scope="session")
def shared_resource():
    """Fixture shared across all tests in this suite."""
    resource = setup_resource()
    yield resource
    cleanup_resource(resource)

@pytest.fixture
def test_specific_resource(request):
    """Fixture for individual tests."""
    return create_resource()
```

### Multi-Device Testing

Test across multiple devices efficiently:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from sysagent.utils.system.ov_helper import get_available_devices_by_category

# Get available devices
device_dict = get_available_devices_by_category(
    device_categories=["cpu", "igpu", "dgpu"]
)

# Execute tests in parallel
with ThreadPoolExecutor(max_workers=len(device_dict)) as executor:
    futures = {
        executor.submit(run_test_on_device, device_id): device_id
        for device_id in device_dict.keys()
    }
    
    for future in as_completed(futures):
        device_id = futures[future]
        try:
            result = future.result()
            results.metrics[f"throughput_{device_id}"] = Metrics(
                value=result["throughput"],
                unit="fps"
            )
        except Exception as e:
            logger.error(f"Test failed on {device_id}: {e}")
```

### Docker Integration

Use Docker for isolated test environments:

```python
from sysagent.utils.infrastructure import DockerClient

docker_client = DockerClient()

# Build image
build_result = docker_client.build_image(
    path=dockerfile_dir,
    tag="my-test-image:latest",
    nocache=False
)

# Run container
container_result = docker_client.run_container(
    image="my-test-image:latest",
    command=["python", "test_script.py"],
    volumes={
        "/host/path": {"bind": "/container/path", "mode": "rw"}
    },
    environment={"VAR": "value"},
    timeout=300
)

# Parse results
output = container_result.get("output", "")
exit_code = container_result.get("exit_code", -1)
```

### Profile Inheritance

Profiles can extend other profiles:

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
    cpu_min_cores: 8  # Override
    dgpu_required: true  # Add new requirement
```

### Custom Metrics and Aggregation

Implement complex metric calculations:

```python
def aggregate_device_metrics(device_results):
    """Aggregate metrics across multiple devices."""
    total_throughput = sum(r["throughput"] for r in device_results.values())
    avg_latency = sum(r["latency"] for r in device_results.values()) / len(device_results)
    
    return {
        "total_throughput": total_throughput,
        "avg_latency": avg_latency,
        "device_count": len(device_results)
    }

# Use in test
for device_id, device_result in device_results.items():
    results.metrics[f"throughput_{device_id}"] = Metrics(
        value=device_result["throughput"],
        unit="fps"
    )

aggregated = aggregate_device_metrics(device_results)
results.metrics["total_throughput"] = Metrics(
    value=aggregated["total_throughput"],
    unit="fps",
    is_key_metric=True
)
```

### Iteration Data and Visualizations

Track per-iteration metrics for detailed analysis:

```python
iteration_data = {
    "iterations": [],
    "throughput": [],
    "latency": []
}

for i in range(num_iterations):
    result = run_iteration()
    iteration_data["iterations"].append(i)
    iteration_data["throughput"].append(result["throughput"])
    iteration_data["latency"].append(result["latency"])

# Summarize with visualizations
summarize_test_results(
    results=results,
    test_name=test_name,
    iteration_data=iteration_data,
    enable_visualizations=True,
    configs=configs,
    get_kpi_config=get_kpi_config
)
```

---

## Allure3 Report Customization

The project bundles a customized build of Allure3 for report generation.  This section explains how the customization is structured and how to iterate on it.

### Two-Tier Architecture

Customizations are split into two tiers so that small core changes and large new UI components are managed separately.

| Tier | Location | What it contains | Format |
|------|----------|-----------------|--------|
| **Tier 1 — Core patch** | `src/sysagent/configs/core/patches/allure3/` | Modifications to files that ship with vanilla allure3: `allurerc.mjs`, `packages/core*/src/`, `packages/plugin*/src/` | Unified diff (`.patch`) |
| **Tier 2 — Component overlay** | `src/sysagent/configs/core/overlay/allure3/` | Full source files for custom UI components that are entirely new or substantially rewritten | Plain source files |

During CLI setup the core patch is applied first, then the overlay files are copied on top.  Both tiers are committed to the repository.

#### Tier 2 overlay structure

```
src/sysagent/configs/core/overlay/allure3/
└── packages/
    └── web-awesome/
        └── src/
            ├── components/
            │   ├── Footer/          # FooterLogo.tsx, FooterVersion.tsx
            │   ├── SectionPicker/   # index.tsx
            │   ├── SectionSwitcher/ # index.tsx
            │   └── Summary/         # Custom summary section (charts, telemetry, KPIs)
            ├── locales/
            │   └── en.json          # Localisation strings
            └── stores/
                └── sections.ts      # Section registry
```

Overlay files mirror the allure3 directory tree and can be edited directly without downloading or building allure3.

### How the Patch Workflow Works

When the CLI runs it downloads vanilla allure3 at a fixed tag and applies every `*.patch` file in the patches directory using `patch --backup`.  The `--backup` flag creates a `*.orig` file alongside every patched file, recording the pre-patch (vanilla) content.

The `patch` mode of `scripts/allure3-dev.sh` uses those `*.orig` files to reconstruct the vanilla state inside a throwaway Git repository, overlays the current modified files (excluding Tier 2 overlay paths from the diff), and captures a clean unified diff with `git diff --cached`.

### Prerequisites

Set up automatically the first time you run `<cli> run`:

- `<data-dir>/thirdparty/allure3` — allure3 working copy
- `<data-dir>/thirdparty/node` — Node.js installation
- `rsync`, `patch`, `git` — system packages (`apt install rsync patch git`)

### Editing Source Files

**Tier 2 changes (component overlay)** — edit directly in the overlay source directory; no allure3 build required:

```
src/sysagent/configs/core/overlay/allure3/packages/web-awesome/src/
```

**Tier 1 changes (core patch)** — edit directly in the allure3 working copy:

```
<data-dir>/thirdparty/allure3/allurerc.mjs
<data-dir>/thirdparty/allure3/packages/core/src/
<data-dir>/thirdparty/allure3/packages/plugin-awesome/src/
<data-dir>/thirdparty/allure3/packages/plugin-log/src/
<data-dir>/thirdparty/allure3/packages/core-api/src/model.ts
```

### Developer Workflow

#### Step 1 — One-time project setup

```bash
<cli> run
```

This downloads allure3, applies the core patch, copies overlay files, and builds everything into `<data-dir>/thirdparty/allure3/`.

#### Step 2 — Edit and preview

Edit the relevant source files (see [Editing Source Files](#editing-source-files) above), then preview:

```bash
bash scripts/allure3-dev.sh test
```

This sets `singleFile: false` in `allurerc.mjs`, applies overlay files, rebuilds the `web-awesome` package, generates a multi-file report from the latest test results, and opens a browser.  Press `Ctrl+C` to stop the web server.

**Why multi-file mode?** The custom UI components (`TelemetrySection`, `BulletChart`, `AttachmentImage`, etc.) call `fetchAttachment()` from `@allurereport/web-commons` to retrieve full attachment content at runtime — things like the Core Metrics JSON, System Info JSON, and `test_summary.json` that carry telemetry, KPI results, and hardware context.  In multi-file mode each attachment is a discrete file under `./out/data/attachments/` served by the dev web server, so every `fetchAttachment()` call resolves to a real HTTP request that returns the complete JSON payload.  If the report were bundled into a single HTML file during development, attachment data would be inlined as opaque base64 blobs and the fetch path used by the components would not resolve, leaving the custom summary section empty.  Production builds use `singleFile: true` (set automatically by `patch` mode) where allure3 handles the embedding differently and `fetchAttachment()` is served from inline data; test mode keeps `singleFile: false` to make the full attachment content inspectable and to avoid the overhead of re-bundling the entire report on every UI iteration.

Repeat until the UI looks correct.

!!! note
    For changes in packages other than `web-awesome` (e.g. `core`, `plugin-awesome`, `plugin-log`), run a full rebuild first:
    ```bash
    bash scripts/allure3-dev.sh build
    bash scripts/allure3-dev.sh test
    ```

#### Step 3 — Export changes

```bash
bash scripts/allure3-dev.sh patch
```

This does two things automatically:

1. Syncs any Tier 2 files modified in the allure3 working copy back to the overlay source directory.
2. Regenerates the Tier 1 core patch from the remaining differences (Tier 2 files are excluded from the diff).

#### Step 4 — Remove the working copy

Remove the allure3 working copy so the next CLI run re-applies the updated patch and overlay from scratch:

```bash
# Option A: automatic (add --clean-allure to the patch command)
bash scripts/allure3-dev.sh patch --clean-allure

# Option B: manual
rm -rf <data-dir>/thirdparty/allure3
```

#### Step 5 — Verify and commit

```bash
# Verify the updated patch applies and the report renders
<cli> run

# Run a specific profile to verify more quickly
<cli> run --profile <profile-name>
```

Commit both the updated patch file and any changed overlay files.

### `allure3-dev.sh` Script Reference

The script lives at `scripts/allure3-dev.sh`.  All path defaults are derived from `--app-name` (defaults to `esq`).

| Option | Default | Description |
|--------|---------|-------------|
| `--app-name NAME` | `esq` | CLI data directory prefix; all paths default to `<NAME>_data/…` |
| `--allure-dir DIR` | `<app-name>_data/thirdparty/allure3` | Allure3 working copy |
| `--node-dir DIR` | `<app-name>_data/thirdparty/node` | Node.js installation |
| `--results-dir DIR` | `<app-name>_data/results/allure` | Allure results for the test preview |
| `--patches-dir DIR` | `src/sysagent/configs/core/patches/allure3` | Destination for the generated patch |
| `--overlay-dir DIR` | `src/sysagent/configs/core/overlay/allure3` | Component overlay source directory |
| `--patch-name NAME` | `allure3-v<version>.patch` | Filename for the generated patch |
| `--clean-allure` | — | Remove the allure3 working copy after `patch` mode completes |
| `--no-open` | — | Skip opening the browser (`test` mode only) |
| `--dry-run` | — | Print actions without executing them |

---

## Next Steps

1. **Review existing tests** - Study tests in `src/esq/suites/ai/` for real-world examples
2. **Create your first test** - Follow the quick start guide above
3. **Run tests** - Use `esq -v run --profile your-profile` to execute
4. **Review results** - Check JSON summary and Allure report
5. **Iterate** - Refine your tests based on results and requirements

For more information:
- [Developer Quick Reference](developer-quick-reference.md) - Cheat sheet for common tasks
- [Quick Start Guide](../getting-started/quick-start.md)
- [Troubleshooting Guide](troubleshooting.md)
- [API Reference](../api/)

---

**Questions or Issues?**
- Open an issue on [GitHub](https://github.com/intel/esq)
- Review existing tests for examples
- Check the troubleshooting guide

