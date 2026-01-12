# Test Suites

Intel® ESQ provides a comprehensive collection of test suites to assess and qualify your edge system capabilities. Choose from qualification tests with pass/fail criteria, data collection suites for analysis, or industry-specific vertical tests.

## Table of Contents

- [Available Test Suites](#available-test-suites)
- [Test Suite Types](#test-suite-types)
- [Qualifications](#qualifications)
    - [AI Edge System Qualification](#ai-edge-system-qualification)
- [Vertical](#vertical)
    - [Manufacturing](#manufacturing)
    - [Metro](#metro)
    - [Retail](#retail)
        - [Automated Self Checkout](#automated-self-checkout)
        - [Loss Prevention](#loss-prevention)
- [Horizontal](#horizontal)
    - [Generative AI](#generative-ai)
    - [Vision AI](#vision-ai)
        - [DL Streamer Analysis - Multi-Stream Pipelines With Multiple AI Stages](#dl-streamer-analysis---multi-stream-pipelines-with-multiple-ai-stages)
        - [DL Streamer Analysis - Verified Reference Blueprints](#verified-reference-blueprints)
        - [OpenVINO](#openvino)
    - [System GPU - OpenVINO](#system-gpu---openvino)
    - [System Memory - STREAM](#system-memory---stream)
    - [Media Performance](#media-performance)


---

## Available Test Suites

Quick reference of all available test suites and their profile names.

| Profile Name | Category | Description | Run Command |
|--------------|----------|-------------|-------------|
| `profile.qualification.ai-edge-system` | Qualification | AI Edge System qualification | `esq run --profile profile.qualification.ai-edge-system` |
| `profile.vertical.manufacturing` | Vertical | Manufacturing | `esq run --profile profile.vertical.manufacturing` |
| `profile.vertical.metro` | Vertical | Metro | `esq run --profile profile.vertical.metro` |
| `profile.vertical.retail-asc` | Vertical | Retail Automated Self-Checkout | `esq run --profile profile.vertical.retail-asc` |
| `profile.vertical.retail-lp` | Vertical | Retail Loss Prevention | `esq run --profile profile.vertical.retail-lp` |
| `profile.suite.ai.gen` | Horizontal | Gen AI profile | `esq run --profile profile.suite.ai.gen` |
| `profile.suite.ai.vision-light` | Horizontal | DL Streamer Analysis - Multi-Stream Pipelines With Multiple AI Stages | `esq run --profile profile.suite.ai.vision-light` |
| `profile.suite.ai.vision-ov` | Horizontal | OpenVINO Benchmark - Measures raw inference performance using OpenVINO Runtime API | `esq run --profile profile.suite.ai.vision-ov` |
| `profile.suite.ai.vision-vrb` | Horizontal | Vision AI profile - Verified Reference Blueprints | `esq run --profile profile.suite.ai.vision-vrb` |
| `profile.suite.system.gpu-ov` | Horizontal | System GPU Performance using OpenVINO benchmark | `esq run --profile profile.suite.system.gpu-ov` |
| `profile.suite.system.memory-stream` | Horizontal | System Memory Performance using STREAM benchmark | `esq run --profile profile.suite.system.memory-stream` |
| `profile.suite.media.performance-pipelines` | Horizontal | Media Performance | `esq run --profile profile.suite.media.performance-pipelines` |

**List all available profiles**:
```bash
esq list
```

---

## Test Suite Types

| Test Suite | Purpose | Benefit |
|------|---------|----------|
| **Qualifications** | Measuring system performance to qualify against AI Edge Systems Qualifications Metrics | Gain Catalog inclusion and other marketing benefits from Intel.  |
| **Vertical** | System benchmarking vertical specific proxy workloads like retail self checkout, smart NVR and manufacturing defect detection | Gain understanding and communicate on system's potential to be used in a variety of verticals and use-cases |
| **Horizontal** | 	General system benchmarking (includes OpenVINO™, Audio, Memory Performance) | Gain understanding on system's resource utilization and performance like System memory and GPU during select AI workload  |

---

## Qualifications

### AI Edge System Qualification

**Profile**: `profile.qualification.ai-edge-system`

**Test Cases**:

Generative AI test on text generation

| Tier | Test ID | Test Case | Qualification Criteria |
|------|---------|-----------|-----------| 
| Entry | AES-GEN-001 | Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-1.5B INT4 | >= 10.0 tokens/sec |
| Mainstream | AES-GEN-001 | Gen AI LLM Serving Benchmark - Phi-4-mini-reasoning 3.8B INT4 | >= 10.0 tokens/sec |
| Efficiency Optimized | AES-GEN-001 | Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-7B INT4 | >= 10.0 tokens/sec |
| Scalable Performance | AES-GEN-001 | Gen AI LLM Serving Benchmark - Qwen3-32B INT4 | >= 10.0 tokens/sec  |
| Scalable Performance Graphics Media | AES-GEN-001 | Gen AI LLM Serving Benchmark - Phi-4-mini-reasoning 3.8B INT4<br>Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-14B INT4<br>Gen AI LLM Serving Benchmark - Qwen3-32B INT4 | >= 10.0 tokens/sec |


Vision AI test using Intel® DLStreamer

| Tier | Test ID | Test Case | Qualification Criteria |
|------|---------|-----------|-----------| 
| Entry | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 | >=  4.0 streams |
| Mainstream | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 | >=  8.0 streams |
| Efficiency Optimized | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 | >=  25.0 streams |
| Scalable Performance | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 | >=  10.0 streams |
| Scalable Performance Graphics Media | AES-VSN-001 | Vision AI Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 | >=  40.0 streams |

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

### Metro

**Profile**: `profile.vertical.metro`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|  
| METRO-PROXY-001 | LPR Pipeline (Multi-Devices) |
| METRO-PROXY-002 | Smart NVR (iGPU) |
| METRO-PROXY-003 | Smart NVR (dGPU) |
| METRO-PROXY-004 | Headed Visual AI Proxy Pipeline (iGPU) |
| METRO-PROXY-005 | Headed Visual AI Proxy Pipeline (dGPU) |
| METRO-PROXY-006 | VSaaS Visual AI Proxy Pipeline (iGPU) |
| METRO-PROXY-007 | VSaaS Visual AI Proxy Pipeline (dGPU) |

**Run this profile**:
```bash
esq run --profile profile.vertical.metro
```

---

### Retail

#### Automated Self Checkout

**Profile**: `profile.vertical.retail-asc`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------| 
| RTL-ASC-001 | Automated Self Checkout - multi-stream 1920p15 H.264 gvadetect YOLO11n INT8 (CPU) |
| RTL-ASC-002 | Automated Self Checkout - multi-stream 1920p15 H.264 gvadetect YOLO11n INT8 (iGPU) |
| RTL-ASC-003 | Automated Self Checkout - multi-stream 1920p15 H.264 gvadetect YOLO11n INT8 (dGPU) |
| RTL-ASC-004 | Automated Self Checkout - multi-stream 1920p15 H.264 gvadetect YOLO11n INT8 (NPU) |

**Run this profile**:
```bash
esq run --profile profile.vertical.retail-asc
```

---

#### Loss Prevention

**Profile**: `profile.vertical.retail-lp`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|  
| RTL-LPP-001 | Loss Prevention - multi-stream 1080p15 Items-in-Basket H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 (iGPU) |
| RTL-LPP-002 | Loss Prevention - multi-stream 1080p15 Hidden-Items-Product-Switching H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 (iGPU) |
| RTL-LPP-003 | Loss Prevention - multi-stream 1080p15 Fake-Scan-Detection H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 (iGPU) |
| RTL-LPP-004 | Loss Prevention - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 (iGPU) |

**Run this profile**:
```bash
esq run --profile profile.vertical.retail-lp
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

### Vision AI
#### DL Streamer Analysis - Multi-Stream Pipelines With Multiple AI Stages

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

#### Verified Reference Blueprints

**Profile**: `profile.suite.ai.vision-vrb`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------| 
| VSN-VRB-001 | DL Streamer Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 |
| VSN-VRB-002 | DL Streamer Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 (CPU) |
| VSN-VRB-003 | DL Streamer Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 (iGPU) |
| VSN-VRB-004 | DL Streamer Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 (dGPU) |
| VSN-VRB-005 | DL Streamer Analysis - multi-stream 1080p15 H.264 gvadetect YOLO11n INT8 gvatrack gvaclassify EfficientNet-B0 INT8 (NPU) |

**Run this profile**:
```bash
esq run --profile profile.suite.ai.vision-vrb
```

---

#### OpenVINO

**Profile**: `profile.suite.ai.vision-ov`

<details markdown="1">
<summary><b>Test Cases</b> (click to expand)</summary>

| Test ID | Test Case |
|---------|-----------|  
| VSN-OBM-001 | OpenVINO Benchmark - resnet-50-tf INT8 (iGPU) |
| VSN-OBM-002 | OpenVINO Benchmark - resnet-50-tf INT8 (dGPU) |
| VSN-OBM-003 | OpenVINO Benchmark - resnet-50-tf INT8 (NPU) |
| VSN-OBM-004 | OpenVINO Benchmark - efficientnet-b0 INT8 (iGPU) |
| VSN-OBM-005 | OpenVINO Benchmark - efficientnet-b0 INT8 (dGPU) |
| VSN-OBM-006 | OpenVINO Benchmark - efficientnet-b0 INT8 (NPU) |
| VSN-OBM-007 | OpenVINO Benchmark - ssdlite_mobilenet_v2 INT8 (iGPU) |
| VSN-OBM-008 | OpenVINO Benchmark - ssdlite_mobilenet_v2 INT8 (dGPU) |
| VSN-OBM-009 | OpenVINO Benchmark - ssdlite_mobilenet_v2 INT8 (NPU) |
| VSN-OBM-010 | OpenVINO Benchmark - mobilenet-v2-pytorch INT8 (iGPU) |
| VSN-OBM-011 | OpenVINO Benchmark - mobilenet-v2-pytorch INT8 (dGPU) |
| VSN-OBM-012 | OpenVINO Benchmark - mobilenet-v2-pytorch INT8 (NPU) |
| VSN-OBM-013 | OpenVINO Benchmark - yolo-v5s INT8 (iGPU) |
| VSN-OBM-014 | OpenVINO Benchmark - yolo-v5s INT8 (dGPU) |
| VSN-OBM-015 | OpenVINO Benchmark - yolo-v5s INT8 (NPU) |
| VSN-OBM-016 | OpenVINO Benchmark - yolo-v8s INT8 (iGPU) |
| VSN-OBM-017 | OpenVINO Benchmark - yolo-v8s INT8 (dGPU) |
| VSN-OBM-018 | OpenVINO Benchmark - clip-vit-base-patch16 INT8 (iGPU) |
| VSN-OBM-019 | OpenVINO Benchmark - clip-vit-base-patch16 INT8 (dGPU) |

</details>

<br>

**Run this profile**:
```bash
esq run --profile profile.suite.ai.vision-ov
```

---

### System GPU - OpenVINO

**Profile**: `profile.suite.system.gpu-ov`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|  
| GPU-OBM-001 | AI GPU Frequency Measure - OV Benchmark yolo-v5s FP16 (iGPU) |
| GPU-OBM-002 | AI GPU Frequency Measure - OV Benchmark yolo-v5s FP16 (dGPU) |

**Run this profile**:
```bash
esq run --profile profile.suite.system.gpu-ov
```

---

### System Memory - STREAM

**Profile**: `profile.suite.system.memory-stream`

**Test Cases**:

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

### Media Performance
**Profile**: `profile.suite.media.performance-pipelines`

<details markdown="1">
<summary><b>Test Cases</b> (click to expand)</summary>

| Test ID | Test Case |
|---------|-----------| 
| MDA-DEC-001 | Media Decode 4Mbps H.264 1080p@30 (iGPU) |
| MDA-DEC-002 | Media Decode 16Mbps H.264 4k@30 (iGPU) |
| MDA-DEC-003 | Media Decode 4Mbps H.264 1080p@30 (dGPU) |
| MDA-DEC-004 | Media Decode 16Mbps H.264 4k@30 (dGPU) |
| MDA-DEC-005 | Media Decode 2Mbps H.265 1080p@30 (iGPU) |
| MDA-DEC-006 | Media Decode 8Mbps H.265 4k@30 (iGPU) |
| MDA-DEC-007 | Media Decode 2Mbps H.265 1080p@30 (dGPU)|
| MDA-DEC-008 | Media Decode 8Mbps H.265 4k@30 (dGPU) |
| MDA-COMP-001 | Media Decode + Compose 4Mbps H.264 1080p@30 (iGPU) |
| MDA-COMP-002 | Media Decode + Compose 16Mbps H.264 4k@30 (iGPU) |
| MDA-COMP-003 | Media Decode + Compose 2Mbps H.265 1080p@30 (iGPU) |
| MDA-COMP-004 | Media Decode + Compose 8Mbps H.265 4k@30 (iGPU) |
| MDA-COMP-005 | Media Decode + Compose 4Mbps H.264 1080p@30 (dGPU) |
| MDA-COMP-006 | Media Decode + Compose 16Mbps H.264 4k@30 (dGPU) |
| MDA-COMP-007 | Media Decode + Compose 2Mbps H.265 1080p@30 (dGPU) |
| MDA-COMP-008 | Media Decode + Compose 8Mbps H.265 4k@30 (dGPU) |
| MDA-ENC-001 | Media Encode 4Mbps H.264 1080p@30 (iGPU) |
| MDA-ENC-002 | Media Encode 16Mbps H.264 4k@30 (iGPU) |
| MDA-ENC-003 | Media Encode 4Mbps H.264 1080p@30 (dGPU) |
| MDA-ENC-004 | Media Encode 16Mbps H.264 4k@30 (dGPU) |
| MDA-ENC-005 | Media Encode 2Mbps H.265 1080p@30 (iGPU) |
| MDA-ENC-006 | Media Encode 8Mbps H.265 4k@30 (iGPU) |
| MDA-ENC-007 | Media Encode 2Mbps H.265 1080p@30 (dGPU) |
| MDA-ENC-008 | Media Encode 8Mbps H.265 4k@30 (dGPU) |

</details>

<br>

**Run this profile**:
```bash
esq run --profile profile.suite.media.performance-pipelines
```