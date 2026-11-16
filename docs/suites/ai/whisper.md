# Whisper Speech Recognition

Transcribes a reference audio sample with a Whisper model and measures latency, real-time factor, and accuracy (word and character error rate) across model sizes, backends, and devices.

| | |
|---|---|
| **Test suite** | `test_whisper` (`src/esq/suites/ai/audio/test_whisper.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). The model size, backend, and audio come from the profile. |
| **Key metric** | `latency_ms` (ms, lower is better), with `rtf` (ratio), `wer` and `cer` (%) |
| **Devices** | CPU, iGPU, dGPU, NPU (OpenVINO backend), CPU, iGPU, dGPU (PyTorch backend) |

## How It Works

1. Download the Whisper model and, for the OpenVINO backend, export it to OpenVINO IR. Download the reference audio sample.
2. Build and start the container of the selected backend (`openvino` or `pytorch`).
3. Run `warmup_runs` transcriptions, then `num_runs` measured ones.
4. Report the average latency, the real-time factor, and the error rates against `reference_transcript`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_size` | `tiny` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large`, or `turbo` |
| `backend` | `openvino` | `openvino` or `pytorch` |
| `devices` | `[cpu]` | Device categories to test |
| `model_id`, `model_dir` | unset | Use another Hugging Face model or a local model directory |
| `wav_file_path`, `reference_transcript` | LibriSpeech sample and its transcript | 16 kHz WAV file to transcribe and its expected text |
| `warmup_runs`, `num_runs` | `1`, `5` | Unmeasured and measured transcriptions |
| `timeout`, `download_timeout`, `export_timeout`, `inference_timeout` | `300`, `3600`, `1800`, `900` | Time limits in seconds |
| `kpi_refs`, `kpi_validation_mode` | `[]`, `all` | KPIs to validate |

## Profiles

<!-- esq:suite-profiles test_whisper -->

## Run

```bash
esq run --profile profile.horizontal.ai.audio-whisper --filter test_id=AUD-WOV-001
```
