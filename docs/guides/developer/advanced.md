# Best Practices & Advanced Topics

This page covers recommended practices for test design, configuration, and security, followed by advanced patterns for custom fixtures, multi-device testing, Docker\* integration, and iteration-level visualizations.

---

## Best Practices

### Test Design

1. **Follow the 7-step pattern** — Maintain consistency across all tests. See [Writing Tests](writing-tests.md).
2. **Use descriptive test IDs** — Format: `{SUITE}-{NUM}` (e.g., `VSN-001`).
3. **Leverage caching** — Tests should work correctly both with and without cached results.
4. **Validate requirements early** — Call `validate_system_requirements_from_configs` before any setup work.
5. **Use appropriate log levels** — Log important steps at `INFO`, debug details at `DEBUG`.
6. **Handle cleanup** — Always release resources (containers, processes, temp files) in `try/finally` blocks.

### Configuration

1. **Define clear KPIs** — Set realistic thresholds based on measured baselines.
2. **Use requirement flags** — Prefer built-in flags over custom validation logic. See [System Requirements](requirements.md).
3. **Version profiles** — Include a `version` field in all profile YAML files.
4. **Organize profiles** — Place in `qualifications/`, `suites/`, or `verticals/` as appropriate.

### Error Handling

1. **Use pytest mechanisms** — `pytest.skip()`, `pytest.fail()`, `pytest.xfail()` for controlled failure states.
2. **Log errors with context** — Include parameter values and suggested fixes in error messages.
3. **Attach diagnostics to Allure** — Use Allure attachments for logs, screenshots, and config dumps.

### Performance

1. **Cache expensive operations** — Model downloads, format conversions, and compilations should be cached.
2. **Set appropriate timeouts** — Use the `timeout` parameter in profiles; typical values are 180–600 seconds.
3. **Parallelize multi-device tests** — Use thread pools when running across multiple devices.
4. **Monitor resources** — Track memory and storage usage when handling large models.

### Security

1. **Validate user inputs** — Use allow-lists for any user-provided values.
2. **Avoid `shell=True`** — Use list-based subprocess calls.
3. **Set restrictive file permissions** — Use `0o750` or `0o770`.
4. **Never log secrets** — Keep tokens, passwords, and API keys out of log output.

---

## Custom Fixtures

Create test-specific fixtures in a `conftest.py` file within the suite directory:

```python
# src/esq/suites/my_domain/conftest.py
import pytest


@pytest.fixture(scope="session")
def shared_resource():
    """Shared across all tests in this suite."""
    resource = setup_resource()
    yield resource
    cleanup_resource(resource)


@pytest.fixture
def test_specific_resource(request):
    """Created fresh for each test."""
    return create_resource()
```

---

## Multi-Device Testing

Run tests across multiple Intel® devices in parallel:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from sysagent.utils.system.ov_helper import get_available_devices_by_category

# Discover available devices
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

---

## Docker* Integration

Use Docker\* for isolated test environments:

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

output    = container_result.get("output", "")
exit_code = container_result.get("exit_code", -1)
```

---

## Custom Metrics and Aggregation

Aggregate results from multiple devices into a single summary metric:

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

# Store per-device metrics
for device_id, device_result in device_results.items():
    results.metrics[f"throughput_{device_id}"] = Metrics(
        value=device_result["throughput"],
        unit="fps"
    )

# Store aggregated metric as the key metric
aggregated = aggregate_device_metrics(device_results)
results.metrics["total_throughput"] = Metrics(
    value=aggregated["total_throughput"],
    unit="fps",
    is_key_metric=True
)
```

---

## Iteration Data and Visualizations

Track per-iteration metrics for detailed analysis and charts in the Allure report:

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

# Pass iteration data to the summarizer to generate charts
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

## Runtime Environment Overrides

!!! warning "For development and advanced testing only"
    These variables are intended for local iteration (for example, shortening a long test run) and must **not** be set during actual runs. Overriding profile parameters invalidates qualification results.

Several built-in test suites support optional runtime parameter overrides through environment variables. Set the variable before the `esq run` command to tune a parameter for one run without editing any profile file.

All variables follow the naming convention:

```
ENV_SUITE_<TEST_FILENAME>_<PARAMETER>
```

where `<TEST_FILENAME>` is the test file name with `test_` and `.py` removed, uppercased, with hyphens replaced by underscores.

### Available variables

| Variable | Test file | Overrides | Accepted values |
|---|---|---|---|
| `ENV_SUITE_CYCLICTEST_DURATION` | `test_cyclictest.py` | `cyclic_duration` | `60s`, `30m`, `2h`, `24h` |
| `ENV_SUITE_CSTATE_RT_CPU_IDS` | `test_cstate.py` | isolcpus requirement | comma-separated CPU list, e.g. `2,3` or `2-3` |
| `ENV_SUITE_CPUFREQ_RT_CPU_IDS` | `test_cpufreq.py` | isolcpus requirement | comma-separated CPU list, e.g. `2,3` or `2-3` |
| `ENV_SUITE_STRESS_NG_DURATION_SECONDS` | `test_stress_ng.py` | `stress_duration_seconds` | Integer seconds, e.g. `120` |
| `ENV_SUITE_MEMORY_HEALTH_MEMTESTER_SIZE_MB` | `test_memory_health.py` | `memtester_size_mb` | Integer MB or `auto` for dynamic sizing |
| `ENV_SUITE_MEMORY_HEALTH_MEMTESTER_ITERATIONS` | `test_memory_health.py` | `memtester_iterations` | Positive integer, e.g. `1`, `3` |

### Usage examples

```bash
# Run cyclictest for 60 seconds instead of the profile default (24 h)
ENV_SUITE_CYCLICTEST_DURATION=60s esq -v run --profile profile.suite.realtime.performance
```

### Priority rules

Each override resolver applies the same priority order:

1. **Env var** — always wins when set; invalid values (wrong type, empty string) are logged as a warning and ignored
2. **Profile param** — the value in the YAML `params` block
3. **Built-in default** — hard-coded fallback when neither is present

A warning is logged whenever an env var value is rejected, so the active parameter is always visible in the run log.

### Cache isolation

Resolved values — whether from the env var or the profile — are written back into `configs` before `execute_test_with_cache` is called. This ensures that runs with different override values are cached independently and never share a stale result from a prior run with different parameters.

For guidance on implementing this pattern in a new test, see [Optional: Runtime Parameter Overrides via Environment Variables](writing-tests.md#optional-runtime-parameter-overrides-via-environment-variables).

---

## Related Pages

- [Writing Tests](writing-tests.md) — Full step-by-step test creation guide
- [Fixtures Reference](fixtures.md) — All available pytest fixtures
- [System Requirements](requirements.md) — Hardware and software requirement flags
- [Allure Report Customization](allure-reports.md) — Customizing the report UI
