# Text Generation (LLM Serving)

Serves a large language model (LLM) with OpenVINO™ Model Server (OVMS) and measures generation throughput with a ShareGPT prompt workload.

| | |
|---|---|
| **Test suite** | `test_text_generation` (`src/esq/suites/ai/gen/test_text_generation.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). The model, precision, and load come from the profile. |
| **Key metric** | `throughput` (output tokens/sec), per device as `throughput_<device>` |
| **Devices** | CPU, iGPU, dGPU, NPU, and `HETERO` across several dGPUs |

## How It Works

1. Download the model and export it to OpenVINO IR with the OVMS export tool at `model_precision` (or download a pre-quantized model), and download the ShareGPT dataset.
2. Start an OVMS container on each device.
3. Send `test_num_prompts` ShareGPT prompts with the vLLM benchmark client at `test_request_rate`, with at most `test_max_concurrent_requests` in flight.
4. Report output tokens per second per device, plus time to first token, time per output token, and token counts in the metadata.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | `microsoft/Phi-4-mini-instruct` | Hugging Face model ID |
| `model_precision` | unset | Export precision, for example `int4`. Unset uses a pre-quantized OpenVINO model as is. |
| `quantization_params`, `quant_sym`, `quant_group_size`, `quant_ratio`, `quant_awq` | symmetric, group size -1, ratio 1.0 | Export options, as one string or as separate fields |
| `devices` | `[]` | Device categories to test |
| `test_num_prompts`, `test_request_rate`, `test_max_concurrent_requests` | `5`, `1`, `1` | Number of prompts, requests per second, and concurrent requests |
| `rest_workers` | `2` | OVMS REST worker threads. Kept low because a large value can exceed the open files limit on systems with many cores. |
| `ovms_container_image` | `openvino/model_server:2026.4.1-gpu` | Model server image |
| `hf_dataset_repo`, `ms_dataset_repo`, `dataset_filename` | ShareGPT V3 | Dataset source on Hugging Face or ModelScope |
| `timeout`, `server_timeout`, `benchmark_timeout`, `export_timeout`, `download_timeout` | `300`, `600`, `600`, `1800`, `18000` | Time limits in seconds |
| `kpi_refs`, `kpi_override`, `kpi_validation_mode`, `tiers` | `[]`, unset, `all`, unset | KPI validation and qualification tiers |
| `verified_reference_data` | `{}` | Reference results shown next to the measured values |

## Adding a Workload

Add a profile entry with another model and precision. For example, from `horizontals/ai_gen_llm.yml`:

```yaml
test_text_generation:
  params:
    - test_id: "GEN-LLM-001"
      display_name: "LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-1.5B INT4 (CPU)"
      model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
      model_precision: "int4"
      devices: [cpu]
```

Qualification criteria are listed in [Qualifications](../../profiles/qualification.md).

## Profiles

<!-- esq:suite-profiles test_text_generation -->

## Run

```bash
esq run --profile profile.horizontal.ai.gen-llm --filter test_id=GEN-LLM-001
```
