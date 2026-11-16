# Order Accuracy (Take-Away and Dine-In)

Runs the upstream [Order Accuracy](https://github.com/intel-retail/order-accuracy) applications and measures the token throughput of their vision language model (VLM) on OpenVINO™ Model Server (OVMS). Take-Away checks orders from video frames. Dine-In checks food tray images against customer orders.

| | |
|---|---|
| **Test suite** | `test_ta_vlm` (`src/esq/suites/vertical/retail/test_ta_vlm.py`), `test_di_vlm` (`src/esq/suites/vertical/retail/test_di_vlm.py`) |
| **Integration type** | [Type 2: reproduce the original flow](../../guides/developer/integration-types.md#type-2-reproduce-the-original-flow). Clones `intel-retail/order-accuracy` and runs its `make benchmark` flow. |
| **Key metric** | `throughput_mean_sec` (tokens/sec), per device, for example `throughput_mean_sec_igpu` |
| **Devices** | CPU, iGPU, dGPU. A category that matches several devices (two dGPUs) runs once on each. |

## How It Works

1. Clone `intel-retail/order-accuracy` into `esq_data/thirdparty/order-accuracy`, check out `commit_id`, and run `make update-submodules`. Both suites share this clone.
2. For each device, write `TARGET_DEVICE` into the application's `.env` and export the model with `ovms-service/setup_models.sh`. Take-Away also downloads its sample video (`make download-sample-video`). Dine-In uses the images and orders bundled with the repository.
3. Run `make benchmark` with `benchmark_workers` and `benchmark_duration` (fixed-workers mode). The target starts the application's services.
4. Run `make consolidate-metrics`, read `throughput_mean_sec` from `vlm_performance_metrics_*.txt` and `VLM_AVERAGE_CALL_DURATION` (`vlm_avg_call_duration`) from `consolidated_metrics.csv`, then stop the services with `make down`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `devices` | `[igpu]` | Device categories to test |
| `commit_id` | `65d1959` | Commit of `intel-retail/order-accuracy` to check out. Used only when the repository is cloned for the first time. |
| `benchmark_workers` | `1` | Concurrent benchmark workers (at least 1) |
| `benchmark_duration` | `200` (Take-Away), `180` (Dine-In) | Benchmark seconds (at least 10) |
| `benchmark_init_duration` | `10` | Take-Away only: warm-up seconds (at least 5) |
| `timeout` | `1800` | Seconds for `make benchmark` on one device |

The profiles pin `commit_id: "65d1959"` (tag release `v2026.2.3`).

## Comparison With the Baseline

The [VLM Prompt](../ai/vlm-prompt.md) suite (Type 1) is the baseline of these applications: it sends the same requests to the same model with only the model server and a load generator. Measured results and the reasons for the gaps are in [Comparison With the Upstream Benchmarks](../ai/vlm-prompt.md#comparison-with-the-upstream-benchmarks).

## Profiles

<!-- esq:suite-profiles test_ta_vlm test_di_vlm -->

## Run

```bash
esq run --profile profile.vertical.retail-di-vlm --filter test_id=DIO-BNK-002
```
