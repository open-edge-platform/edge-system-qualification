# DL Streamer Pipeline

Runs an Intel® DL Streamer pipeline in Docker containers and finds the largest number of concurrent video streams that keep the target frame rate on each device.

| | |
|---|---|
| **Test suite** | `test_dlstreamer` (`src/esq/suites/ai/vision/test_dlstreamer.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). The pipeline, models, and videos come from the profile. |
| **Key metric** | `streams_max` (streams), per device as `streams_max_<device>` |
| **Devices** | CPU, iGPU, dGPU, NPU, or several stages on different devices |

## How It Works

1. Download the profile's `assets` (videos and models) and build the analyzer images.
2. Run a baseline on each device to estimate how many streams it can carry.
3. Search for the highest stream count whose per-stream FPS stays at or above `target_fps`, starting from the baseline.
4. Run all qualified devices together and report the total and per-device `streams_max`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pipeline` | required | GStreamer pipeline. Placeholders such as `${DEVICE_ID}`, `${RENDER_DEVICE}`, and keys of `pipeline_params` are filled per device. |
| `pipeline_params` | `{}` | Named pipeline fragments, each with a `default` and optional per-device values (`cpu`, `igpu`, `dgpu`, `npu`) |
| `target_fps` | `14.5` | Lowest per-stream FPS a stream count must keep |
| `devices` | `[]` | Device categories to test |
| `assets` | `[]` | Videos and models to download, see [Asset Management](../../guides/developer/assets.md) |
| `multi_stage_single_pipeline` | `false` | Treat the pipeline as one unit whose stages run on different devices |
| `max_streams_above_baseline` | `1` | How far above the baseline the search may go |
| `consecutive_timeout_threshold` | `4` | Timeouts in a row before the search stops |
| `pipeline_timeout` | `180` | Seconds per pipeline run |
| `timeout` | `300` | Seconds for the whole test case |
| `kpi_refs`, `kpi_validation_mode` | `[]`, `all` | KPIs to validate (qualification profiles) |
| `keep_original_videos` | `false` | Keep downloaded videos before conversion |

## Adding a Workload

Describe the application's ingredients in a profile instead of running the application. For example, the [Pallet Defect Detection](https://github.com/open-edge-platform/manufacturing-ai-suite/blob/main/industrial-edge-insights-vision/apps/pallet-defect-detection/README.md) app becomes video and model `assets`, a `pipeline`, and `pipeline_params` in `verticals/manufacturing.yml`:

```yaml
test_dlstreamer:
  params:
    - test_id: "MFG-PDD-001"
      display_name: "Pallet Defect Detection - multi-stream 480p30 H.264 gvadetect YOLOX-TINY INT8 (CPU)"
      target_fps: 29.5
      pipeline: |
        filesrc location=./videos/warehouse_30fps.avi
          ! ${PDD_DECODE_ELEMENTS}
          ! gvadetect model=./models/pallet_defect_detection_int8/deployment/Detection/model/model.xml device=CPU ${PDD_INFERENCE_ELEMENTS}
          ! queue
      devices: [cpu]
      kpi_refs: [streams_max]
```

## Profiles

<!-- esq:suite-profiles test_dlstreamer -->

## Run

```bash
esq run --profile profile.vertical.manufacturing --filter test_id=MFG-PDD-001
```
