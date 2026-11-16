# Qualification Profiles

Qualification profiles validate the measured results against Intel® targets and pass or fail the system. Each test case runs one of the [test suites](../suites/index.md).

## Profiles

Each profile links to its YAML file, and each suite to its page. This list is generated from the shipped profiles when the documentation is built.

<!-- esq:profiles qualification -->

## Intel® AI Edge Systems Qualification

**Profile**: `profile.qualification.ai-edge-system`

Generative AI, [Text Generation](../suites/ai/text-generation.md) suite:

| Tier | Test ID | Test Case | Qualification Criteria |
|------|---------|-----------|------------------------|
| Entry | AES-GEN-001 | Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-1.5B INT4 | >= 10.0 tokens/sec |
| Mainstream | AES-GEN-001 | Gen AI LLM Serving Benchmark - Phi-4-mini-reasoning 3.8B INT4 | >= 10.0 tokens/sec |
| Efficiency Optimized | AES-GEN-001 | Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-7B INT4 | >= 10.0 tokens/sec |
| Scalable Performance | AES-GEN-001 | Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-14B INT4 | >= 10.0 tokens/sec |
| Scalable Performance Graphics Media | AES-GEN-001 | Gen AI LLM Serving Benchmark - Phi-4-mini-reasoning 3.8B INT4<br>Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-14B INT4<br>Gen AI LLM Serving Benchmark - Qwen3-32B INT4 | >= 10.0 tokens/sec |

Vision AI, [DL Streamer](../suites/ai/dlstreamer.md) suite:

| Tier | Test ID | Test Case | Qualification Criteria |
|------|---------|-----------|------------------------|
| Entry | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 | >= 4.0 streams |
| Mainstream | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 | >= 8.0 streams |
| Efficiency Optimized | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 | >= 25.0 streams |
| Scalable Performance | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 | >= 10.0 streams |
| Scalable Performance Graphics Media | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 | >= 40.0 streams |

```bash
esq run --profile profile.qualification.ai-edge-system
```
