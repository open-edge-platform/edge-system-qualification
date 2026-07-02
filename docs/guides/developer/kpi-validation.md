# KPI Validation

KPI validation is **opt-in**. Tests that do not define `kpi_refs` in their parameters skip the validation step entirely — metrics are still collected and reported, but no pass/fail decision is made. This makes KPIs suitable for qualification profiles while keeping data-collection suites simple.

KPIs (Key Performance Indicators) define the pass/fail criteria for qualification tests. This page explains how to define KPIs, reference them in tests, and validate results against thresholds.

---

## Defining KPIs

KPIs are defined in a `config.yml` file within the test directory:

```yaml
kpi:
  inference_time:
    name: "Inference Time"
    type: "numeric"
    validation:
      operator: "lte"   # less than or equal
      reference: 0.5    # 500 ms threshold
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

  status:
    name: "Test Status"
    type: "string"
    validation:
      operator: "eq"    # equals
      reference: "success"
      enabled: true
    unit: ""
    severity: "critical"
    description: "Overall test status"
    default_value: "unknown"
```

---

## KPI Configuration Fields

| Field | Required | Description | Values |
|-------|----------|-------------|--------|
| `name` | Yes | Human-readable label | String |
| `type` | Yes | Data type | `numeric`, `string`, `boolean`, `list` |
| `validation.operator` | Yes | Comparison operator | `gte`, `lte`, `eq`, `ne`, `gt`, `lt` |
| `validation.reference` | Yes | Target threshold or expected value | Varies by type |
| `validation.enabled` | No | Enable or disable validation | `true` (default) or `false` |
| `unit` | No | Display unit | `"ms"`, `"fps"`, etc. |
| `severity` | No | Impact level | `critical`, `major`, `normal`, `minor` |
| `description` | No | Short explanation of the KPI | String |
| `default_value` | No | Fallback value when metric is missing | Varies by type |

### Operators

| Operator | Condition |
|----------|-----------|
| `gte` | `actual >= reference` |
| `lte` | `actual <= reference` |
| `gt` | `actual > reference` |
| `lt` | `actual < reference` |
| `eq` | `actual == reference` |
| `ne` | `actual != reference` |

---

## Referencing KPIs in Profiles

Use `kpi_refs` in the profile or `config.yml` to associate KPIs with a specific test run:

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

---

## Validating in Tests

```python
validation_results = validate_test_results(
    results=results,
    configs=configs,
    get_kpi_config=get_kpi_config,
    test_name=test_name
)

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

---

## Validation Result Structure

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

---

## Disabling KPI Validation

Set `validation.enabled: false` to collect a metric without validating it:

```yaml
kpi:
  optional_metric:
    name: "Optional Metric"
    type: "numeric"
    validation:
      operator: "gte"
      reference: 100.0
      enabled: false    # Collected but not validated
```

---

## Related Pages

- [Profile & Test Config](configuration.md) — Where to define `kpi_refs` in profiles
- [Fixtures Reference](fixtures.md) — `validate_test_results` and `get_kpi_config` fixtures
- [Results & Metrics](results-metrics.md) — Attaching metrics to `Result` objects
