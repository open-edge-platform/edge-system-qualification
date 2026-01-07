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
          memory_min_gb: 8.0
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
          memory_min_gb: 8.0
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
    memory_min_gb: 8.0
    storage_min_gb: 5.0
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
    memory_min_gb: 8.0
    storage_min_gb: 10.0
    
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
  memory_min_gb: 8.0
  
  # Maximum memory (for constrained tests)
  memory_max_gb: 32.0
```

#### Storage Requirements

```yaml
requirements:
  # Total storage capacity
  storage_total_min_gb: 64.0
  
  # Free storage for test execution
  storage_min_gb: 10.0
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
  memory_min_gb: 16.0
  storage_min_gb: 20.0
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

### Automatic Metadata

The `Result` class automatically includes:

```python
{
    "created_at": "2025-12-29T10:00:00+00:00",
    "updated_at": "2025-12-29T10:05:30+00:00",
    "total_duration_seconds": 330.0,
    "kpi_validation_status": "passed"  # or "failed", "skipped"
}
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
    
    # Add parameters
    results.parameters["Model"] = "yolo11n"
    results.parameters["Precision"] = "INT8"
    results.parameters["Device"] = "GPU"
    
    # Add custom metadata
    results.metadata["model_version"] = "1.0.0"
    results.metadata["batch_size"] = 8
    
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
    memory_min_gb: 8.0

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

