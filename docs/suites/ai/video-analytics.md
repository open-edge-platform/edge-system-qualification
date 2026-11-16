# Video Analytics Pipelines

Runs built-in multi-stage video analytics pipelines (decode, detection, tracking, classification) and finds the largest number of streams each device combination can sustain.

| | |
|---|---|
| **Test suite** | `test_video_analytics` (`src/esq/suites/ai/vision/test_video_analytics.py`, functions `test_va_light`, `test_va_medium`, `test_va_heavy`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). ESQ's own benchmark container runs three built-in pipelines, and the profile selects the pipeline and devices. |
| **Key metric** | `max_streams` (streams), with `avg_fps`, `gpu_freq_mhz`, and `pkg_power_w` |
| **Devices** | CPU, iGPU, dGPU, NPU, as combinations of devices for decode, detection, and classification |

| Pipeline | `test` | Models |
|----------|--------|--------|
| Light | `test_va_light` | YOLO11n detection, ResNet-50 classification |
| Medium | `test_va_medium` | YOLOv5m detection, ResNet-50 and MobileNet-v2 classification |
| Heavy | `test_va_heavy` | YOLO11m detection, ResNet-v1-50 and MobileNet-v2 classification |

## How It Works

1. Download the pipeline's models and video, and build the benchmark image on the DL Streamer image.
2. Run the pipeline in compute modes that combine the available devices for each stage.
3. Increase the stream count until the frame rate drops, and record the frame rate, GPU frequency, and package power.
4. Report the largest stream count.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `test` | required | Pipeline to run: `test_va_light`, `test_va_medium`, or `test_va_heavy` |
| `devices` | `igpu` | Device categories to combine |
| `timeout` | `600` (light), `900` (medium, heavy) | Seconds for the test case |
| `force_run` | `false` | Run again even if a result file of an earlier run exists |
| `display_output` | `0` | Show the video output on a display |
| `container_image`, `image_tag`, `dockerfile_name`, `docker_nocache` | `va_<pipeline>_bm`, `1.0`, `Dockerfile`, `false` | Benchmark image build |

## Profiles

<!-- esq:suite-profiles test_video_analytics -->

## Run

```bash
esq run --profile profile.horizontal.ai.vision-va --filter test_id=VSN-LIGHT-001
```
