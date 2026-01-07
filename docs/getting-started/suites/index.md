# Test Suites

Intel® Edge System Qualification (Intel® ESQ) provides a comprehensive collection of test suites to assess and qualify your edge system capabilities. Choose from qualification tests with pass/fail criteria, data collection suites for analysis, or industry-specific vertical tests.

## Table of Contents

- [Test Suite Types](#test-suite-types)
- [Qualifications](#qualifications)
    - [Intel® AI Edge System Qualification](#intel-ai-edge-system-qualification)
- [Vertical](#vertical)
    - [Manufacturing](#manufacturing)
    - [Retail](#retail)
- [Horizontal](#horizontal)
    - [Generative AI](#generative-ai)
    - [Vision AI - Light](#vision-ai---light)
    - [System Memory - STREAM](#system-memory---stream)

---

## Test Suite Types

| Test Suite | Purpose | Benefit |
|------|---------|----------|
| **Qualifications** | Measuring system performance to qualify against Intel® AI Edge Systems Qualifications Metrics | Gain Catalog inclusion and other marketing benefits from Intel.  |
| **Vertical** | System benchmarking vertical specific proxy workloads like retail self checkout, smart NVR and manufacturing defect detection | Gain understanding and communicate on system's potential to be used in a variety of verticals and use-cases |
| **Horizontal** | 	General system benchmarking (includes OpenVINO™ toolkit, Audio, Memory Performance) | Gain understanding on system's resource utilization and performance like System memory and GPU during select AI workload  |

---

## Qualifications

### Intel® AI Edge System Qualification

**Profile**: `profile.qualification.ai-edge-system`

**Test Cases**:

Generative AI test on text generation

| Tier | Test ID | Test Case |
|------|---------|-----------| 
| Entry | AES-GEN-001 | Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-1.5B INT4 |
| Mainstream | AES-GEN-001 | Gen AI LLM Serving Benchmark - Phi-4-mini-reasoning 3.8B INT4 |
| Efficiency Optimized | AES-GEN-001 | Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-7B INT4 |
| Scalable Performance | AES-GEN-001 | Gen AI LLM Serving Benchmark - Qwen3-32B INT4 |
| Scalable Performance Graphics Media | AES-GEN-001 | Gen AI LLM Serving Benchmark - Phi-4-mini-reasoning 3.8B INT4<br>Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-14B INT4<br>Gen AI LLM Serving Benchmark - Qwen3-32B INT4<br>Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Llama-70B INT4 |


Vision AI test using Deep Learning Streamer (DL Streamer)

| Tier | Test ID | Test Case |
|------|---------|-----------| 
| Entry | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p30 H.265 gvadetect YOLO11n INT8 gvatrack gvaclassify ResNet50 INT8 |
| Mainstream | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p30 H.265 gvadetect YOLO11n INT8 gvatrack gvaclassify ResNet50 INT8 |
| Efficiency Optimized | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p30 H.265 gvadetect YOLO11n INT8 gvatrack gvaclassify ResNet50 INT8 |
| Scalable Performance | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p30 H.265 gvadetect YOLO11n INT8 gvatrack gvaclassify ResNet50 INT8 |
| Scalable Performance Graphics Media | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p30 H.265 gvadetect YOLO11n INT8 gvatrack gvaclassify ResNet50 INT8 |

**Run this profile**:
```bash
esq run --profile profile.qualification.ai-edge-system
```

---

## Vertical

### Manufacturing

**Profile**: `profile.vertical.manufacturing`

**Test Case**:

| Test ID | Test Case |
|---------|-----------| 
| MFG-PDD-001 | Pallet Defect Detection - multi-stream 480p30 H.264 gvadetect YOLOX-TINY FP32 (CPU) |
| MFG-PDD-002 | Pallet Defect Detection - multi-stream 480p30 H.264 gvadetect YOLOX-TINY FP32 (iGPU) |
| MFG-PDD-003 | Pallet Defect Detection - multi-stream 480p30 H.264 gvadetect YOLOX-TINY FP32 (dGPU) |
| MFG-WPC-001 | Weld Porosity Classification - multi-stream 1024p30 H.264 gvaclassify EfficientNet-B0 FP16 (CPU) |
| MFG-WPC-002 | Weld Porosity Classification - multi-stream 1024p30 H.264 gvaclassify EfficientNet-B0 FP16 (iGPU) |
| MFG-WPC-003 | Weld Porosity Classification - multi-stream 1024p30 H.264 gvaclassify EfficientNet-B0 FP16 (dGPU) |


**Run this profile**:
```bash
esq run --profile profile.vertical.manufacturing
```

---

### Retail

**Profile**: `profile.vertical.retail`

**Test Case**:

| Test ID | Test Case |
|---------|-----------| 
| RTL-ASC-001 | Automated Self Checkout - multi-stream 1920p15 H.264 gvadetect YOLO11n INT8 (CPU) |
| RTL-ASC-002 | Automated Self Checkout - multi-stream 1920p15 H.264 gvadetect YOLO11n INT8 (iGPU) |
| RTL-ASC-003 | Automated Self Checkout - multi-stream 1920p15 H.264 gvadetect YOLO11n INT8 (dGPU) |
| RTL-ASC-004 | Automated Self Checkout - multi-stream 1920p15 H.264 gvadetect YOLO11n INT8 (NPU) |

**Run this profile**:
```bash
esq run --profile profile.vertical.retail
```

---

## Horizontal

### Generative AI

**Profile**: `profile.suite.ai.gen`

<details markdown="1">
<summary><b>Test Cases</b> (click to expand)</summary>

| Test ID | Test Case |
|---------|-----------| 
| GEN-LLM-001 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-1.5B INT4 (CPU) |
| GEN-LLM-002 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-1.5B INT4 (iGPU) |
| GEN-LLM-003 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-1.5B INT4 (dGPU) |
| GEN-LLM-004 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-1.5B INT4 (Hetero dGPU) |
| GEN-LLM-005 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-1.5B INT4 (NPU) |
| GEN-LLM-006 | LLM Serving Benchmark - Phi-4-mini-reasoning 3.8B INT4 (CPU) |
| GEN-LLM-007 | LLM Serving Benchmark - Phi-4-mini-reasoning 3.8B INT4 (iGPU) |
| GEN-LLM-008 | LLM Serving Benchmark - Phi-4-mini-reasoning 3.8B INT4 (dGPU) |
| GEN-LLM-009 | LLM Serving Benchmark - Phi-4-mini-reasoning 3.8B INT4 (Hetero dGPU) |
| GEN-LLM-010 | LLM Serving Benchmark - Phi-4-mini-reasoning 3.8B INT4 (NPU) |
| GEN-LLM-011 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-7B INT4 (CPU) |
| GEN-LLM-012 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-7B INT4 (iGPU) |
| GEN-LLM-013 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-7B INT4 (dGPU) |
| GEN-LLM-014 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-7B INT4 (Hetero dGPU) |
| GEN-LLM-015 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-7B INT4 (NPU) |
| GEN-LLM-016 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-14B INT4 (CPU) |
| GEN-LLM-017 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-14B INT4 (iGPU) |
| GEN-LLM-018 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-14B INT4 (dGPU) |
| GEN-LLM-019 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-14B INT4 (Hetero dGPU) |
| GEN-LLM-020 | LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-14B INT4 (NPU) |
| GEN-LLM-021 | LLM Serving Benchmark - Qwen3-32B INT4 (CPU) |
| GEN-LLM-022 | LLM Serving Benchmark - Qwen3-32B INT4 (iGPU) |
| GEN-LLM-023 | LLM Serving Benchmark - Qwen3-32B INT4 (dGPU) |
| GEN-LLM-024 | LLM Serving Benchmark - Qwen3-32B INT4 (Hetero dGPU) |
| GEN-LLM-025 | LLM Serving Benchmark - Qwen3-32B INT4 (NPU) |
| GEN-LLM-026 | LLM Serving Benchmark - DeepSeek-R1-Distill-Llama-70B INT4 (CPU) |
| GEN-LLM-027 | LLM Serving Benchmark - DeepSeek-R1-Distill-Llama-70B INT4 (iGPU) |
| GEN-LLM-028 | LLM Serving Benchmark - DeepSeek-R1-Distill-Llama-70B INT4 (dGPU) |
| GEN-LLM-029 | LLM Serving Benchmark - DeepSeek-R1-Distill-Llama-70B INT4 (Hetero dGPU) |
| GEN-LLM-030 | LLM Serving Benchmark - DeepSeek-R1-Distill-Llama-70B INT4 (NPU) |

</details>

<br>

**Run this profile**:
```bash
esq run --profile profile.suite.ai.gen
```

---

### Vision AI - Light

**Profile**: `profile.suite.ai.vision-light`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------| 
| VSN-LGT-001 | DL Streamer Analysis - multi-stream 1080p30 H.265 gvadetect YOLO11n INT8 gvatrack gvaclassify ResNet50 INT8 |
| VSN-LGT-002 | DL Streamer Analysis - multi-stream 1080p30 H.265 gvadetect YOLO11n INT8 gvatrack gvaclassify ResNet50 INT8 (CPU) |
| VSN-LGT-003 | DL Streamer Analysis - multi-stream 1080p30 H.265 gvadetect YOLO11n INT8 gvatrack gvaclassify ResNet50 INT8 (iGPU) |
| VSN-LGT-004 | DL Streamer Analysis - multi-stream 1080p30 H.265 gvadetect YOLO11n INT8 gvatrack gvaclassify ResNet50 INT8 (dGPU) |
| VSN-LGT-005 | DL Streamer Analysis - multi-stream 1080p30 H.265 gvadetect YOLO11n INT8 gvatrack gvaclassify ResNet50 INT8 (NPU) |

**Run this profile**:
```bash
esq run --profile profile.suite.ai.vision-light
```

---

### System Memory - STREAM

**Profile**: `profile.suite.system.memory-stream`

**Test Case**:

| Test ID | Test Case |
|---------|-----------| 
| MEM-STR-001 | STREAM Memory Benchmark - Copy |
| MEM-STR-002 | STREAM Memory Benchmark - Scale |
| MEM-STR-003 | STREAM Memory Benchmark - Add |
| MEM-STR-004 | STREAM Memory Benchmark - Triad |

**Run this profile**:
```bash
esq run --profile profile.suite.system.memory-stream
```
