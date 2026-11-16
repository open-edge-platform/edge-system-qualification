# Proxy Pipelines

Runs metro proxy pipelines (License Plate Recognition, Smart NVR, Headed Visual AI, VSaaS) with Intel® DL Streamer in a benchmark container and finds the largest number of 1080p streams that keep 30 FPS.

| | |
|---|---|
| **Test suite** | `test_proxy_pipelines` (`src/esq/suites/vertical/vision/test_proxy_pipelines.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). ESQ's own benchmark container (`src/containers/proxy_pipeline_benchmark/`) runs the pipeline that the profile selects with `pptest`. |
| **Key metric** | `max_streams` (streams), the best result across the tested devices. Also `avg_fps`, `gpu_freq_mhz`, `pkg_power_w`, `duration_s`. |
| **Devices** | iGPU, dGPU. LPR can also use the NPU when the platform has one. |

## How It Works

1. Download the models and videos of the selected pipeline into `esq_data/data/vertical/vision/results/ppl/resources`.
2. Build the device-specific DL Streamer images (as `test_dlstreamer`) and the proxy pipeline image on top of the matching one (CPU/iGPU, dGPU, or NPU).
3. Run the container with all devices of `devices`. Per device and model, it searches for the highest stream count that keeps the target FPS (`search_algorithm`) and writes a CSV file.
4. Read `max_streams` from the CSV row with the most streams and `avg_fps` from the runner log. If the CSV already exists and `force_run` is false, the benchmark is skipped and the existing CSV is read.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pptest` | `SmartAIRunner` | Pipeline: `LPRAIRunner`, `SmartAIRunner`, `VisualAIRunner`, or `VsaasAIRunner` |
| `devices` | `igpu` | Device categories to test. dGPU tests are skipped on retired metro dGPU device IDs. |
| `search_algorithm` | `adaptive` | Stream count search: `adaptive` (FPS-guided) or `linear` |
| `force_run` | `false` | Run again and replace existing result CSV files |
| `display_output` | unset | `1` shows the output on the X11 display (headless fallback if none), `0` uses `fakesink`. Unset detects the display. |
| `timeout` | `7200` | Seconds for the benchmark container |
| `docker_nocache` | `false` | Build the proxy pipeline image without the Docker cache |
| `container_image`, `image_tag` | `proxy_pl_bm_runner`, `1.0` | Name and tag of the proxy pipeline image |

The `profile.vertical.metro` profile also runs the profiles listed in its `depends_on` (memory, GPU, vision, and media horizontal profiles) in `src/esq/configs/profiles/verticals/metro.yml`.

## Profiles

<!-- esq:suite-profiles test_proxy_pipelines -->

## Run

```bash
esq run --profile profile.vertical.metro --filter test_id=METRO-PROXY-002
```
