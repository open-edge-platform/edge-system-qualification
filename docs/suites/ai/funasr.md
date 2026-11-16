# FunASR Speech Recognition

Converts the FunASR (Paraformer) speech-recognition model to OpenVINO™ IR and measures its inference throughput and latency with OpenVINO `benchmark_app` in a Docker container.

| | |
|---|---|
| **Test suite** | `test_funasr` (`src/esq/suites/ai/audio/test_funasr.py`, function `test_audio_funasr`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). ESQ builds its own benchmark container, and the profile selects the sub-model and device. |
| **Key metric** | `throughput` (FPS), with `avg_latency`, `min_latency`, and `max_latency` (ms) |
| **Devices** | CPU, GPU, NPU |

## How It Works

1. Build the benchmark image. It clones FunASR at a pinned commit, patches it, and converts the selected sub-model to OpenVINO IR.
2. Run `benchmark_app` for the sub-model on `device` in the container.
3. Read the benchmark report and record throughput and latency.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `CPU` | OpenVINO device (`CPU`, `GPU`, `NPU`) |
| `model` | `eb` | Sub-model: `eb` (contextual embedder, all devices) or `bb` (backbone, CPU only) |
| `operation` | `funasr` | Benchmark to run in the container |
| `container_image`, `image_tag`, `dockerfile_name`, `docker_base_image` | `robotics_funasr_benchmark`, `1.0`, `Dockerfile`, `ubuntu:24.04` | Benchmark image build |
| `docker_nocache` | `false` | Rebuild the image without the Docker cache |
| `timeout` | `300` | Seconds for the test case |

## Profiles

<!-- esq:suite-profiles test_funasr -->

## Run

```bash
esq run --profile profile.horizontal.ai.audio-funasr --filter test_id=AUD-ASR-001
```
