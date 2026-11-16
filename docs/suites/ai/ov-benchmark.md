# OpenVINO™ Benchmark

Measures the raw inference throughput of a model with OpenVINO™ `benchmark_app` on one device, without any pipeline around it.

| | |
|---|---|
| **Test suite** | `test_ov_benchmark` (`src/esq/suites/ai/vision/test_ov_benchmark.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). The model, precision, and device come from the profile. |
| **Key metric** | `throughput` (FPS) |
| **Devices** | CPU, iGPU, dGPU, NPU |

## How It Works

1. Download the model, convert it to OpenVINO IR, and quantize it to `precision`.
2. Build the benchmark image with `benchmark_app`.
3. Run `benchmark_app` on the device for `duration_secs`.
4. Report the throughput and latency.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `resnet-50-tf` | Model name, for example `resnet-50-tf`, `efficientnet-b0`, `yolo-v8s`, `clip-vit-base-patch16` |
| `precision` | `INT8` | Model precision |
| `devices` | `igpu` | Device category to test |
| `duration_secs` | `0.15` | `benchmark_app` run time in seconds. The profiles use `120`. |
| `force_download` | `false` | Download and convert the model again |
| `container_image`, `dockerfile_name`, `docker_nocache` | `openvino_bm_runner`, `Dockerfile`, `false` | Benchmark image build |
| `timeout` | `300` | Seconds for the test case |

## Adding a Workload

Add a profile entry with another model and device. For example, from `horizontals/ai_vision_ov.yml`:

```yaml
test_ov_benchmark:
  params:
    - test_id: "VSN-OBM-001"
      display_name: "OpenVINO Benchmark - resnet-50-tf INT8 (iGPU)"
      devices: [igpu]
      model: "resnet-50-tf"
      precision: "INT8"
      duration_secs: 120
```

## Profiles

<!-- esq:suite-profiles test_ov_benchmark -->

## Run

```bash
esq run --profile profile.horizontal.ai.vision-ov --filter test_id=VSN-OBM-001
```
