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

## Related Pages

- [Writing Tests](writing-tests.md) — Full step-by-step test creation guide
- [Fixtures Reference](fixtures.md) — All available pytest fixtures
- [System Requirements](requirements.md) — Hardware and software requirement flags
- [Allure Report Customization](allure-reports.md) — Customizing the report UI
