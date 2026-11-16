# Media Pipelines

Runs GStreamer media pipelines for decode, decode plus compose, and encode on Intel® GPUs and finds the largest number of channels that keep 30 FPS each.

| | |
|---|---|
| **Test suite** | `test_media_pipelines` (`src/esq/suites/media/performance/test_media_pipelines.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). ESQ's own benchmark container runs the pipelines, and the profile selects the operation, codec, bitrate, resolution, and device. |
| **Key metric** | `max_streams` (channels), with `avg_fps` |
| **Devices** | iGPU, dGPU |

## How It Works

1. Build the media benchmark image and check that the GStreamer VA plugins are available.
2. Generate or loop the input video for the codec, bitrate, and resolution.
3. Run the `operation` pipeline with an increasing number of channels in parallel.
4. Report the largest channel count that kept 30 FPS per channel, and the average FPS.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `operation` | `decode` | `decode`, `compose` (decode and compose), or `encode` |
| `codec` | `H.264` | `H.264` or `H.265` |
| `bitrate` | `4Mbps` | Stream bitrate |
| `resolution` | `1080p` | Stream resolution and frame rate, for example `1080p30` or `4k30` |
| `device` | `igpu` | Device category |
| `display_output` | auto | Show the composed output on a display. Unset detects whether a display is available. |
| `container_image`, `image_tag`, `dockerfile_name`, `docker_nocache` | `media_perf_runner`, `1.0`, `Dockerfile`, `false` | Benchmark image build |
| `timeout` | `300` | Seconds for the test case. The profiles use `3600`. |
| `kpi_refs` | `[]` | KPIs to validate |

## Profiles

<!-- esq:suite-profiles test_media_pipelines -->

## Run

```bash
esq run --profile profile.horizontal.media.performance-pipelines --filter test_id=MDA-DEC-001
```
