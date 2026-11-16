# Loss Prevention VLM

Runs the upstream [Loss Prevention](https://github.com/intel-retail/loss-prevention) VLM workload, which combines YOLO object detection with a vision language model (VLM) for scene understanding, and measures the VLM token throughput.

| | |
|---|---|
| **Test suite** | `test_lp_vlm` (`src/esq/suites/vertical/retail/test_lp_vlm.py`) |
| **Integration type** | [Type 2: reproduce the original flow](../../guides/developer/integration-types.md#type-2-reproduce-the-original-flow). Clones `intel-retail/loss-prevention` and runs its `make run-lp` flow. |
| **Key metric** | `throughput_mean` (tokens/sec). Also `application_latency` and `avg_vlm_call_duration` (ms). |
| **Devices** | `devices` is only checked for availability. ESQ does not pass it to the application. |

## How It Works

1. Clone `intel-retail/loss-prevention` into `esq_data/thirdparty/loss-prevention` and check out `commit_id`.
2. Create the application's `.env` (generated MinIO and RabbitMQ passwords, `HUGGINGFACE_TOKEN`) if it does not exist, and run `make download-models`.
3. Run `make run-lp CAMERA_STREAM=<camera_stream> RENDER_MODE=0 REGISTRY=false`, then poll the `vlm-pipeline-runner` container every 60 seconds (15 times at most) until it exits.
4. Stop the workload with `make down-lp`, run `make consolidate-metrics`, and average the values in `results/vlm-results/vlm_performance_metrics_*.txt`. Latencies come from `metrics.csv`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `devices` | `[cpu]` | Device categories that must be available |
| `camera_stream` | `camera_to_workload_vlm.json` | Camera configuration file in the repository's `configs/` folder |
| `commit_id` | `2205dc9` | Commit of `intel-retail/loss-prevention` to check out. Used only when the repository is cloned for the first time. |
| `huggingface_token` | `HUGGINGFACE_TOKEN` environment variable | Token for the gated model. Written only when `.env` is created. |
| `registry` | `false` | `REGISTRY` value of `make download-models` |
| `download_models_timeout` | `1800` | Seconds for `make download-models` |
| `timeout` | `3600` | Seconds for `make run-lp` |

The profile pins `commit_id: "6e6353b"` (tag release `2026.2.1`).

## Profiles

<!-- esq:suite-profiles test_lp_vlm -->

## Run

```bash
esq run --profile profile.vertical.retail-lp-vlm --filter test_id=LP-VLM-001
```
