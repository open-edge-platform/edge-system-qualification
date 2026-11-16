# Wind Turbine Time-Series

Runs the Wind Turbine Anomaly Detection sample application of Open Edge Platform, streams simulated sensor data into it over OPC UA or MQTT, and measures how many data points it processes and the end-to-end latency.

| | |
|---|---|
| **Test suite** | `test_wind_turbine` (`src/esq/suites/ai/timeseries/test_wind_turbine.py`) |
| **Integration type** | [Type 2: reproduce the original flow](../../guides/developer/integration-types.md#type-2-reproduce-the-original-flow). It runs the application's own Docker Compose stack and configuration from `open-edge-platform/edge-ai-suites`, pinned to a commit. |
| **Key metric** | `throughput` (data points/s, higher is better), with `latency` (s) |
| **Devices** | CPU or GPU for the analytics (`compute_device`) |

## How It Works

1. Download the application's compose file and configuration from `edge-ai-suites` at `assets_repo_ref`.
2. Start the stack with `num_streams` simulated data streams over `ingestion_mode` (`opcua` or `mqtt`).
3. Wait until the services are ready, then read the processed points and their timestamps from InfluxDB.
4. Report throughput and latency. A `scenario_matrix` entry runs several stream, ingestion, and device combinations in one test case.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `assets_repo_ref` | `main` | `edge-ai-suites` commit of the downloaded application files. The profile pins the 2026.2 release commit. |
| `ingestion_mode` | `opcua` | Data ingestion: `opcua` or `mqtt` |
| `compute_device` | `unknown` | Analytics device, `cpu` or `gpu` |
| `num_streams`, `num_data_points` | `1`, `1000` | Simulated streams and data points per stream |
| `ingestion_interval`, `continuous_simulator_ingestion` | `1s`, `true` | Simulator pacing |
| `scenario_matrix` | `[]` | Several scenarios, each with its own `test_id` and overrides, run in one test case |
| `image_suffix`, `docker_registry` | `2026.2.0`, unset | Application image tag and registry |
| `mqtt_publisher_sampling_rate`, `mqtt_publisher_streams_per_container`, `mqtt_scale_publishers`, `scale_stream_containers` | `10`, `1`, `true`, `true` | How simulated streams are spread over containers |
| `influx_*` | see code | Where and how throughput and latency are read from InfluxDB, for example `influx_min_collection_seconds` (`60`) and `influx_poll_timeout_seconds` (`180`) |
| `services_ready_timeout_seconds`, `timeout` | `300`, `1200` | Time limits in seconds |
| `ref_throughput`, `ref_latency`, `throughput_percent_delta_threshold` | unset, unset, `5.0` | Reference values. When `ref_throughput` is set, the measured throughput may be at most this many percent below it. |
| `debug_preserve_environment` | `false` | Keep the stack after the test |

## Profiles

<!-- esq:suite-profiles test_wind_turbine -->

## Run

```bash
esq run --profile profile.vertical.manufacturing.timeseries-wt --filter test_id=MFG-WTC-001
```
