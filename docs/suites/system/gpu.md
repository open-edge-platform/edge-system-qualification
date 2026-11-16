# GPU Frequency

Runs OpenVINO™ `benchmark_app` with a YOLO model on the GPU and records the GPU frequency, utilization, and power, and the CPU utilization and frequency, while the GPU is under load.

| | |
|---|---|
| **Test suite** | `test_system_gpu` (`src/esq/suites/system/gpu/test_system_gpu.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). ESQ runs OpenVINO™ `benchmark_app` in its own container. The model, precision, device, and duration come from the profile. |
| **Key metric** | `frequency_max_igpu` or `frequency_max_dgpu` (GHz), for the tested `device` |
| **Devices** | iGPU, dGPU |

## How It Works

1. Download the YOLO model from Ultralytics, export it to OpenVINO™ IR, and download a test image. YOLOv5n replaces the default `yolov5s` on Intel Atom® and Celeron® processors.
2. Build the test image on top of the DL Streamer image prepared by [`test_dlstreamer`](../ai/dlstreamer.md) (the dGPU variant on a discrete GPU).
3. Run the container on each device of the `device` category. It runs `benchmark_app` for `duration_hrs` while collecting telemetry with `intel_gpu_top` (and `xpu-smi` for dGPU power).
4. Read `averages_summary.csv` and report frequency (mean on iGPU, peak on dGPU), stability (standard deviation, minimum, range), utilization, and power. Values that could not be collected are reported as `-1`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `igpu` | Device category to test: `igpu` or `dgpu` |
| `models`, `precision` | `["yolov5s"]`, `FP16` | YOLO model to export and its precision. Only the first model is benchmarked, and the container loads only the `FP16` export. |
| `duration_hrs` | `0.15` | Benchmark duration in hours. The value is converted to a whole number before it is passed on, so values below 1 run for the minimum of 0.01 hours (36 seconds). |
| `timeout` | `300` | Seconds for the container run |

## Profiles

<!-- esq:suite-profiles test_system_gpu -->

## Run

```bash
esq run --profile profile.horizontal.system.gpu-ov --filter test_id=GPU-OBM-001
```
