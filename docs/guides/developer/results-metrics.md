# Results & Metrics

This page describes the `Result` and `Metrics` classes used to capture and report test outcomes.

---

## Result Class

The `Result` dataclass provides structured test result handling:

```python
from sysagent.utils.core import Result, Metrics

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

# Auto-calculates duration from creation and update timestamps
results.update_timestamps()

# Serialize for storage or debugging
results_dict = results.to_dict()
```

---

## Result Fields

| Field | Type | Purpose |
|-------|------|---------|
| `name` | `str` | Test identifier (e.g. `"T001 - My Test"`) |
| `parameters` | `Dict[str, Any]` | Human-readable test configuration shown in reports |
| `metrics` | `Dict[str, Metrics]` | Measured performance values with units |
| `metadata` | `Dict[str, Any]` | Flat key-value pairs for high-level reporting |
| `extended_metadata` | `Dict[str, Any]` | Structured complex objects for programmatic access |
| `kpis` | `Dict[str, Any]` | KPI configurations and validation results |

---

## `metadata` vs `extended_metadata`

Use **`metadata`** for simple property-value pairs that appear in the Allure report and JSON summary as human-readable fields:

```python
results.metadata["model_version"] = "1.0.0"
results.metadata["benchmark_duration_seconds"] = 48.4
results.metadata["status"] = True
```

Use **`extended_metadata`** for structured or complex objects — nested dicts, lists, time-series data — intended for programmatic analysis rather than direct display:

```python
# Structured benchmark breakdown
results.extended_metadata["device_breakdown"] = {
    "CPU": {
        "throughput": 51.2,
        "ttft_ms": 224.0,
        "tpot_ms": 18.5,
        "samples": [...]
    }
}

# Raw iteration data
results.extended_metadata["iterations"] = [
    {"step": 0, "latency_ms": 220.1},
    {"step": 1, "latency_ms": 218.4},
]
```

!!! note
    Telemetry data collected during test execution is automatically placed in `extended_metadata["telemetry"]`. See [Modular Telemetry](telemetry.md).

---

## Automatic Metadata Fields

The `Result` class automatically populates the following keys in `metadata`:

```python
{
    "created_at": "2025-12-29T10:00:00+00:00",
    "updated_at": "2025-12-29T10:05:30+00:00",
    "total_duration_seconds": 330.0,
    "kpi_validation_status": "passed"   # "passed" | "failed" | "skipped"
}
```

---

## Metrics Class

The `Metrics` dataclass structures a single measured value:

```python
from sysagent.utils.core import Metrics

metric = Metrics(
    value=123.45,
    unit="ms",
    is_key_metric=False
)

print(f"Value: {metric.value} {metric.unit}")
```

---

## Adding Metrics to Results

```python
# Single metric
results.metrics["accuracy"] = Metrics(value=0.95, unit="percentage")

# Multiple metrics at once
results.metrics.update({
    "precision": Metrics(value=0.93, unit="percentage"),
    "recall":    Metrics(value=0.91, unit="percentage"),
    "f1_score":  Metrics(value=0.92, unit="percentage")
})

# Device-specific metrics
for device in ["cpu", "igpu", "dgpu"]:
    results.metrics[f"throughput_{device}"] = Metrics(
        value=get_throughput(device),
        unit="fps"
    )
```

---

## Key Metrics

Designate a primary metric to highlight in reports:

```python
results.set_key_metric("throughput")

key_metric_name = results.get_key_metric()
if key_metric_name:
    print(f"Key metric: {key_metric_name}")
```

---

## Complete Example

```python
def execute_benchmark():
    """Execute benchmark and return results."""
    results = Result(name=f"{test_id} - {test_display_name}")

    start_time = time.time()
    throughput, latency = run_inference(model, data)
    elapsed = time.time() - start_time

    # Metrics
    results.metrics["throughput"] = Metrics(value=throughput, unit="fps", is_key_metric=True)
    results.metrics["latency"]    = Metrics(value=latency, unit="ms")
    results.metrics["elapsed_time"] = Metrics(value=elapsed, unit="seconds")

    # Parameters (shown in Allure report)
    results.parameters["Model"]     = "yolo11n"
    results.parameters["Precision"] = "INT8"
    results.parameters["Device"]    = "GPU"

    # Flat metadata (shown in JSON summary)
    results.metadata["model_version"] = "1.0.0"
    results.metadata["benchmark_duration_seconds"] = elapsed

    # Structured data for programmatic use
    results.extended_metadata["per_iteration"] = [
        {"step": i, "latency_ms": v} for i, v in enumerate(latency_trace)
    ]

    results.update_timestamps()
    return results
```

---

## Related Pages

- [Fixtures Reference](fixtures.md) — `execute_test_with_cache` and `summarize_test_results`
- [KPI Validation](kpi-validation.md) — Linking metrics to KPI thresholds
- [Modular Telemetry](telemetry.md) — Automatic background metric collection
