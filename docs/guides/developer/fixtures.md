# Fixtures Reference

The Intel® ESQ framework provides a set of pytest fixtures that are automatically available to all test functions. These fixtures handle caching, validation, reporting, and configuration loading.

---

## Core Fixtures

### `request`

- **Type**: `pytest.FixtureRequest`
- **Scope**: function
- **Description**: Standard pytest request object for accessing test metadata.
- **Usage**:
  ```python
  test_name = request.node.name.split("[")[0]
  ```

---

### `configs`

- **Type**: `Dict[str, Any]`
- **Scope**: function
- **Description**: Test configuration parameters merged from the active profile and `config.yml`.
- **Usage**:
  ```python
  test_id = configs.get("test_id", "T0000")
  devices = configs.get("devices", ["cpu"])
  timeout = configs.get("timeout", 300)
  ```

---

## Cache Fixtures

### `cached_result`

- **Type**: `Callable[[Optional[Dict]], Optional[Union[Result, Dict]]]`
- **Scope**: function
- **Description**: Retrieves cached test results if available. Returns `None` on a cache miss.
- **Usage**:
  ```python
  cached_data = cached_result(cache_configs={"key": "value"})
  if cached_data:
      return cached_data
  ```

---

### `cache_result`

- **Type**: `Callable[[Union[Result, Dict], Optional[Dict]], None]`
- **Scope**: function
- **Description**: Stores test results in the cache for future runs.
- **Usage**:
  ```python
  cache_result(results, cache_configs={"key": "value"})
  ```

---

## Validation Fixtures

### `validate_system_requirements_from_configs`

- **Type**: `Callable[[Dict[str, Any]], None]`
- **Scope**: function
- **Description**: Validates that the system meets all requirements defined in `configs`. Calls `pytest.skip` if any requirement is not met.
- **Usage**:
  ```python
  validate_system_requirements_from_configs(configs)
  ```

See [System Requirements](requirements.md) for all available flags.

---

### `validate_test_results`

- **Type**: `Callable[..., Dict[str, Any]]`
- **Scope**: function
- **Description**: Validates test results against KPI thresholds.
- **Parameters**:

  | Parameter | Description |
  |-----------|-------------|
  | `results` | `Result` object or dict |
  | `configs` | Test configuration dict |
  | `get_kpi_config` | KPI config retrieval fixture |
  | `test_name` | Test identifier string |
  | `mode` | `"all"` (default) or `"any"` — whether all or any KPIs must pass |

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

---

## Execution Fixtures

### `execute_test_with_cache`

- **Type**: `Callable[..., Union[Result, Dict]]`
- **Scope**: function
- **Description**: Executes test logic with automatic caching support. Returns cached results on a cache hit; executes `run_test_func` on a cache miss and caches the result.
- **Parameters**:

  | Parameter | Description |
  |-----------|-------------|
  | `cached_result` | Cached result fixture |
  | `cache_result` | Cache result fixture |
  | `run_test_func` | Callable that runs the test and returns a `Result` |
  | `test_name` | Test identifier string |
  | `configs` | Test configuration dict |
  | `cache_configs` | Optional dict of cache-specific configuration |
  | `name` | Step name shown in Allure (default: `"Analysis"`) |

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

---

### `prepare_test`

- **Type**: `Callable[..., Any]`
- **Scope**: function
- **Description**: Runs an asset preparation function with Allure progress tracking. Use this to download models, videos, or set up other dependencies before the main test.
- **Parameters**:

  | Parameter | Description |
  |-----------|-------------|
  | `test_name` | Test identifier string |
  | `prepare_func` | Callable that performs the preparation work |
  | `configs` | Test configuration dict |
  | `name` | Step name shown in Allure (default: `"Preparation"`) |

- **Usage**:
  ```python
  prepare_test(
      test_name=test_name,
      prepare_func=lambda: download_models(),
      configs=configs,
      name="Assets"
  )
  ```

---

## Reporting Fixtures

### `summarize_test_results`

- **Type**: `Callable[..., None]`
- **Scope**: function
- **Description**: Generates a test result summary and attaches it to the Allure report.
- **Parameters**:

  | Parameter | Description |
  |-----------|-------------|
  | `results` | `Result` object |
  | `test_name` | Test identifier string |
  | `configs` | Test configuration dict (optional) |
  | `get_kpi_config` | KPI config function (optional) |
  | `iteration_data` | Per-iteration metric data (optional) |
  | `enable_visualizations` | Generate charts (default: `False`) |

- **Usage**:
  ```python
  summarize_test_results(
      results=results,
      test_name=test_name,
      configs=configs,
      get_kpi_config=get_kpi_config
  )
  ```

---

## Suite Fixtures

### `suite_configs`

- **Type**: `Dict[str, Any]`
- **Scope**: function
- **Description**: Loads the suite-level configuration from `config.yml` in the test directory.
- **Usage**:
  ```python
  kpis = suite_configs.get("kpi", {})
  ```

---

### `get_kpi_config`

- **Type**: `Callable[[str], Optional[Dict[str, Any]]]`
- **Scope**: function
- **Description**: Retrieves a KPI configuration dict by name from `config.yml`.
- **Usage**:
  ```python
  kpi_config = get_kpi_config("inference_time")
  if kpi_config:
      reference = kpi_config["validation"]["reference"]
  ```

---

## Related Pages

- [Writing Tests](writing-tests.md) — How fixtures are used in the standard 7-step pattern
- [Results & Metrics](results-metrics.md) — The `Result` and `Metrics` classes
- [KPI Validation](kpi-validation.md) — Defining and using KPIs
- [System Requirements](requirements.md) — All available requirement flags
