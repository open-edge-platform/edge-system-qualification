# Test Suites

Intel® ESQ provides a comprehensive collection of test suites to assess and qualify your edge system capabilities. Choose from qualification tests with pass/fail criteria, data collection suites for analysis, or industry-specific vertical tests.

## Table of Contents

- [Available Test Suites](#available-test-suites)
- [Test Suite Types](#test-suite-types)
- [Qualifications](#qualifications)
    - [AI Edge System Qualification](#intel-ai-edge-systems-qualification)
    - [Robotics](#robotics-system-qualification)
- [HEC](#hec)
    - [Hybrid Edge Computing - Debian](#intel-hybrid-edge-computing-system-qualification-debian)
    - [Hybrid Edge Computing - EMT](#intel-hybrid-edge-computing-system-qualification-edge-microvisor-toolkit-emt)
- [Vertical](#vertical)
    - [Manufacturing](#manufacturing)
        - [Manufacturing Defect Detection](#manufacturing-defect-detection)
        - [Manufacturing Timeseries Wind Turbine](#manufacturing-timeseries-wind-turbine)
    - [Metro](#metro)
    - [Retail](#retail)
        - [Automated Self Checkout](#automated-self-checkout)
        - [Loss Prevention](#loss-prevention)
    - [Robotics](#robotics)
        - [PI0.5 RTC Benchmark](#pi05-rtc-benchmark)
        - [ACT Benchmark](#act-benchmark)
        - [FastMapping Benchmark](#fastmapping-benchmark)
        - [ADBScan Benchmark](#adbscan-benchmark)
        - [FunASR ASR Benchmark](#funasr-asr-benchmark)
- [Horizontal](#horizontal)
    - [Generative AI](#generative-ai)
        - [LLM Serving Benchmark](#llm-serving-benchmark)
        - [Chat Question and Answer Core](#chat-question-and-answer-core)
    - [Vision AI](#vision-ai)
        - [DL Streamer Analysis - Multi-Stream Pipelines With Multiple AI Stages](#dl-streamer-analysis---multi-stream-pipelines-with-multiple-ai-stages)
        - [DL Streamer Analysis - Verified Reference Blueprints](#verified-reference-blueprints)
        - [OpenVINO](#openvino)
        - [Video Analytics](#video-analytics)
    - [Real-Time Performance](#real-time-performance)
    - [Real-Time Platform](#real-time-platform)
    - [System CPU](#system-cpu)
    - [System Display](#system-display)
    - [System GPU](#system-gpu)
    - [System Memory](#system-memory)
        - [System Memory - STREAM](#system-memory---stream)
        - [System Memory Health](#system-memory-health)
    - [System Network](#system-network)
    - [System Peripheral](#system-peripheral)
    - [System Stress](#system-stress)
    - [Virtualization](#virtualization)
        - [KVM Compatibility](#kvm-compatibility)
        - [QEMU VM Reboot](#qemu-vm-reboot)
    - [Media Performance](#media-performance)


---

## Available Test Suites

Quick reference of all available test suites and their profile names.

| Profile Name | Category | Description | Run Command |
|--------------|----------|-------------|-------------|
| `profile.qualification.ai-edge-system` | Qualification | Intel® AI Edge Systems qualification | `esq run --profile profile.qualification.ai-edge-system` |
| `profile.qualification.hybrid-edge-system-debian` | HEC | Intel® Hybrid Edge Computing system qualification - Debian | `esq run --profile profile.qualification.hybrid-edge-system-debian` |
| `profile.qualification.hybrid-edge-system-emt` | HEC | Intel® Hybrid Edge Computing system qualification - Edge Microvisor Toolkit (EMT) | `esq run --profile profile.qualification.hybrid-edge-system-emt` |
| `profile.qualification.robotics` | Qualification | Robotics system qualification | `esq run --profile profile.qualification.robotics` |
| `profile.suite.ai.gen-chatqna-core` | Horizontal | Gen AI Chat QnA Core profile | `esq run --profile profile.suite.ai.gen-chatqna-core` |
| `profile.suite.ai.gen-llm` | Horizontal | Gen AI LLM serving benchmark profile | `esq run --profile profile.suite.ai.gen-llm` |
| `profile.suite.ai.vision-light` | Horizontal | DL Streamer Analysis - Multi-Stream Pipelines With Multiple AI Stages | `esq run --profile profile.suite.ai.vision-light` |
| `profile.suite.ai.vision-ov` | Horizontal | OpenVINO™ Toolkit Benchmark - Measures raw inference performance using OpenVINO Runtime API | `esq run --profile profile.suite.ai.vision-ov` |
| `profile.suite.ai.vision-va` | Horizontal | Multi-stage video analytics pipelines with detection, tracking, and classification | `esq run --profile profile.suite.ai.vision-va` |
| `profile.suite.ai.vision-vrb` | Horizontal | Vision AI profile - Verified Reference Blueprints | `esq run --profile profile.suite.ai.vision-vrb` |
| `profile.suite.media.performance-pipelines` | Horizontal | Media Performance | `esq run --profile profile.suite.media.performance-pipelines` |
| `profile.suite.realtime.performance` | Horizontal | Real-time wakeup latency measurement with cyclictest under stress load profile | `esq run --profile profile.suite.realtime.performance` |
| `profile.suite.realtime.platform` | Horizontal | Real-time platform capability detection and configuration checks profile | `esq run --profile profile.suite.realtime.platform` |
| `profile.suite.system.cpu-sku` | Horizontal | CPU SKU characterisation suite — processor identification, generation, core counts, and frequency collection profile | `esq run --profile profile.suite.system.cpu-sku` |
| `profile.suite.system.display` | Horizontal | System display related tests | `esq run --profile profile.suite.system.display` |
| `profile.suite.system.gpu-ov` | Horizontal | System GPU Performance using OpenVINO™ Toolkit benchmark | `esq run --profile profile.suite.system.gpu-ov` |
| `profile.suite.system.memory-health` | Horizontal | System memory health — detects and helps identify faulty RAM modules at runtime | `esq run --profile profile.suite.system.memory-health` |
| `profile.suite.system.memory-stream` | Horizontal | System Memory Performance using STREAM benchmark | `esq run --profile profile.suite.system.memory-stream` |
| `profile.suite.system.network` | Horizontal | System network related tests | `esq run --profile profile.suite.system.network` |
| `profile.suite.system.peripheral` | Horizontal | System Peripheral Device Detection and Enumeration | `esq run --profile profile.suite.system.peripheral` |
| `profile.suite.system.stress` | Horizontal | Host machine stress-ng suite with CPU, memory, iGPU, and mixed stress scenarios | `esq run --profile profile.suite.system.stress` |
| `profile.suite.virtualization.kvm` | Horizontal | KVM/QEMU* virtualization compatibility validation | `esq run --profile profile.suite.virtualization.kvm` |
| `profile.suite.virtualization.kvm-qemu` | Horizontal | QEMU*/KVM VM reboot testing | `esq run --profile profile.suite.virtualization.kvm-qemu` |
| `profile.vertical.manufacturing` | Vertical | Manufacturing | `esq run --profile profile.vertical.manufacturing` |
| `profile.vertical.manufacturing.timeseries-wt` | Vertical | Manufacturing Timeseries Wind Turbine scenarios | `esq run --profile profile.vertical.manufacturing.timeseries-wt` |
| `profile.vertical.metro` | Vertical | Metro proxy workloads (LPR, Smart NVR, Visual AI, VSaaS) | `esq run --profile profile.vertical.metro` |
| `profile.vertical.retail-asc` | Vertical | Retail Automated Self-Checkout | `esq run --profile profile.vertical.retail-asc` |
| `profile.vertical.retail-lp` | Vertical | Retail Loss Prevention | `esq run --profile profile.vertical.retail-lp` |
| `profile.vertical.retail-lp-vlm` | Vertical | Retail Loss Prevention Visual Language Model | `esq run --profile profile.vertical.retail-lp-vlm` |
| `profile.vertical.robotics-act` | Vertical | Robotics ACT inference benchmark | `esq run --profile profile.vertical.robotics-act` |
| `profile.vertical.robotics-adbscan` | Vertical | Robotics ADBScan point cloud benchmark | `esq run --profile profile.vertical.robotics-adbscan` |
| `profile.vertical.robotics-fastmapping` | Vertical | Robotics FastMapping SLAM benchmark | `esq run --profile profile.vertical.robotics-fastmapping` |
| `profile.vertical.robotics-funasr` | Vertical | Robotics FunASR speech-recognition benchmark | `esq run --profile profile.vertical.robotics-funasr` |
| `profile.vertical.robotics-pi5` | Vertical | Robotics PI0.5 RTC inference benchmark | `esq run --profile profile.vertical.robotics-pi5` |

**List all available profiles**:
```bash
esq list
```

---

## Test Suite Types

| Test Suite | Purpose | Benefit |
|------|---------|----------|
| **Qualifications** | Measuring system performance to qualify against  Intel® AI Edge Systems Qualifications Metrics | Gain Catalog inclusion and other marketing benefits from Intel.  |
| **Vertical** | System benchmarking vertical specific proxy workloads like retail self checkout, smart NVR and manufacturing defect detection | Gain understanding and communicate on system's potential to be used in a variety of verticals and use-cases |
| **Horizontal** | 	General system benchmarking (includes OpenVINO™ Toolkit, Audio, Memory Performance) | Gain understanding on system's resource utilization and performance like System memory and GPU during select AI workload  |

---

## Qualifications

### Intel® AI Edge Systems Qualification

**Profile**: `profile.qualification.ai-edge-system`

**Test Cases**:

Generative AI test on text generation

| Tier | Test ID | Test Case | Qualification Criteria |
|------|---------|-----------|-----------| 
| Entry | AES-GEN-001 | Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-1.5B INT4 | >= 10.0 tokens/sec |
| Mainstream | AES-GEN-001 | Gen AI LLM Serving Benchmark - Phi-4-mini-reasoning 3.8B INT4 | >= 10.0 tokens/sec |
| Efficiency Optimized | AES-GEN-001 | Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-7B INT4 | >= 10.0 tokens/sec |
| Scalable Performance | AES-GEN-001 | Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-14B INT4 | >= 10.0 tokens/sec  |
| Scalable Performance Graphics Media | AES-GEN-001 | Gen AI LLM Serving Benchmark - Phi-4-mini-reasoning 3.8B INT4<br>Gen AI LLM Serving Benchmark - DeepSeek-R1-Distill-Qwen-14B INT4<br>Gen AI LLM Serving Benchmark - Qwen3-32B INT4 | >= 10.0 tokens/sec |


Vision AI test using Intel® DL Streamer

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

### Robotics System Qualification

**Profile**: `profile.qualification.robotics`

**Test Cases**:

| Test ID | Test Case | Qualification Criteria |
|---------|-----------|------------------------|
| ROB-RT-001 | PREEMPT-RT Kernel Check | PREEMPT_RT kernel active with required RT boot parameters configured |
| ROB-TSN-001 | TSN Detection | >= 1 TSN-capable Intel® igc NIC with PTP hardware clock detected |
| ROB-TCC-001 | TCC Detection | TCC-capable hardware detected (invariant TSC flags present) |
| ROB-CYC-001 | RT Wakeup Latency (Robotics) | Data collection (max_latency_us in µs, lower is better) |
| ROB-RTC-001 | PI0.5 RTC Benchmark (GPU) | Benchmark completes without error |
| ROB-ACT-001 | ACT Benchmark (GPU) | Benchmark completes without error |
| ROB-FMP-001 | FastMapping Benchmark (CPU) | Benchmark completes without error |
| ROB-ADB-001 | ADBScan Benchmark (CPU) | Benchmark completes without error |
| ROB-ASR-001 | FunASR ASR Benchmark (CPU) | Benchmark completes without error |
| ROB-ASR-002 | FunASR ASR Benchmark (GPU) | Benchmark completes without error |
| ROB-ASR-003 | FunASR ASR Benchmark (NPU) | Benchmark completes without error |

**Run this profile**:
```bash
esq run --profile profile.qualification.robotics
```

---

## HEC

### Intel® Hybrid Edge Computing System Qualification - Debian

**Profile**: `profile.qualification.hybrid-edge-system-debian`

**Test Cases**:

| Test ID | Test Case | Qualification Criteria |
|---------|-----------|------------------------|
| HEC-CPU-001 | CPU Compatibility | Matches an allowed platform configuration |
| HEC-STR-001 | CPU 100% (stress-ng) | Completes 15-minute run without system failure |
| HEC-MEM-001 | Memory Bit-Pattern Test (memtester, dynamic, 1 iter) | = 0 memory errors |
| HEC-KVM-001 | Peripheral VM Capacity | >= 1 VM |
| HEC-QEMU-001 | QEMU* VM Reboot Test (KVM, OVMF/UEFI, 5x) | All 5 reboot cycles complete with guest OS ready |

**Run this profile**:
```bash
esq run --profile profile.qualification.hybrid-edge-system-debian
```

---

### Intel® Hybrid Edge Computing System Qualification - Edge Microvisor Toolkit (EMT)

**Profile**: `profile.qualification.hybrid-edge-system-emt`

**Test Cases**:

| Test ID | Test Case | Qualification Criteria |
|---------|-----------|------------------------|
| HEC-CPU-001 | CPU Compatibility | Matches an allowed platform configuration |
| HEC-STR-001 | CPU 100% (stress-ng) | Completes 15-minute run without system failure |
| HEC-MEM-001 | Memory Bit-Pattern Test (memtester, dynamic, 1 iter) | = 0 memory errors |
| HEC-KVM-001 | Peripheral VM Capacity | >= 1 VM |
| HEC-QEMU-001 | QEMU* VM Reboot Test (KVM, OVMF/UEFI, 5x) | All 5 reboot cycles complete with guest OS ready |

**Run this profile**:
```bash
esq run --profile profile.qualification.hybrid-edge-system-emt
```

---

## Vertical

### Manufacturing

#### Manufacturing Defect Detection 

**Profile**: `profile.vertical.manufacturing`

<details markdown="1">
<summary><b>Test Cases</b> (click to expand)</summary>

| Test ID | Test Case |
|---------|-----------| 
| MFG-PDD-001 | Pallet Defect Detection - multi-stream 480p30 H.264 gvadetect YOLOX-TINY INT8 (CPU) |
| MFG-PDD-002 | Pallet Defect Detection - multi-stream 480p30 H.264 gvadetect YOLOX-TINY INT8 (iGPU) |
| MFG-PDD-003 | Pallet Defect Detection - multi-stream 480p30 H.264 gvadetect YOLOX-TINY INT8 (dGPU) |
| MFG-PDD-004 | Pallet Defect Detection - multi-stream 480p30 H.264 gvadetect YOLOX-TINY FP32 (CPU) |
| MFG-PDD-005 | Pallet Defect Detection - multi-stream 480p30 H.264 gvadetect YOLOX-TINY FP32 (iGPU) |
| MFG-PDD-006 | Pallet Defect Detection - multi-stream 480p30 H.264 gvadetect YOLOX-TINY FP32 (dGPU) |
| MFG-PDD-007 | Pallet Defect Detection - multi-stream 480p30 H.264 gvadetect YOLOX-TINY FP16 (CPU) |
| MFG-PDD-008 | Pallet Defect Detection - multi-stream 480p30 H.264 gvadetect YOLOX-TINY FP16 (iGPU) |
| MFG-PDD-009 | Pallet Defect Detection - multi-stream 480p30 H.264 gvadetect YOLOX-TINY FP16 (dGPU) |
| MFG-WPC-001 | Weld Porosity Classification - multi-stream 1024p30 H.264 gvaclassify EfficientNet-B0 FP16 (CPU) |
| MFG-WPC-002 | Weld Porosity Classification - multi-stream 1024p30 H.264 gvaclassify EfficientNet-B0 FP16 (iGPU) |
| MFG-WPC-003 | Weld Porosity Classification - multi-stream 1024p30 H.264 gvaclassify EfficientNet-B0 FP16 (dGPU) |
| MFG-WPC-004 | Weld Porosity Classification - multi-stream 1024p30 H.264 gvaclassify EfficientNet-B0 INT8 (CPU) |
| MFG-WPC-005 | Weld Porosity Classification - multi-stream 1024p30 H.264 gvaclassify EfficientNet-B0 INT8 (iGPU) |
| MFG-WPC-006 | Weld Porosity Classification - multi-stream 1024p30 H.264 gvaclassify EfficientNet-B0 INT8 (dGPU) |
| MFG-WPC-007 | Weld Porosity Classification - multi-stream 1024p30 H.264 gvaclassify EfficientNet-B0 FP32 (CPU) |
| MFG-WPC-008 | Weld Porosity Classification - multi-stream 1024p30 H.264 gvaclassify EfficientNet-B0 FP32 (iGPU) |
| MFG-WPC-009 | Weld Porosity Classification - multi-stream 1024p30 H.264 gvaclassify EfficientNet-B0 FP32 (dGPU) |

</details>

<br>

**Run this profile**:
```bash
esq run --profile profile.vertical.manufacturing
```

---

#### Manufacturing Timeseries Wind Turbine

**Profile**: `profile.vertical.manufacturing.timeseries-wt`

**Test Case**:

| Test ID | Test Case |
|---------|-----------| 
| MFG-WTC-001 | Wind Turbine Timeseries - Combined Functional Flow |
| MFG-WTS-001 | TS Wind Turbine - s40p500 CPU OPCUA |
| MFG-WTS-002 | TS Wind Turbine - s40p500 GPU OPCUA |
| MFG-WTS-003 | TS Wind Turbine - s40p500 CPU MQTT |
| MFG-WTS-004 | TS Wind Turbine - s40p500 GPU MQTT |

**Run this profile**:
```bash
esq run --profile profile.vertical.manufacturing.timeseries-wt
```

---

### Metro

**Profile**: `profile.vertical.metro`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| METRO-PROXY-001 | LPR Pipeline (Multi-Devices) - LPR proxy workload on iGPU and dGPU for multi-device throughput scaling. |
| METRO-PROXY-002 | Smart NVR (iGPU) - Smart NVR proxy workload on iGPU with display output for real-time analytics. |
| METRO-PROXY-003 | Smart NVR (dGPU) - Smart NVR proxy workload on dGPU for high-density stream analytics. |
| METRO-PROXY-004 | Headed Visual AI Proxy Pipeline (iGPU) - Headed Visual AI proxy workload on iGPU with display output for interactive analytics. |
| METRO-PROXY-005 | Headed Visual AI Proxy Pipeline (dGPU) - Headed Visual AI proxy workload on dGPU with display output for interactive analytics. |
| METRO-PROXY-006 | VSaaS Visual AI Proxy Pipeline (iGPU) - VSaaS Visual AI proxy workload on iGPU with multi-model inference and encode stages. |
| METRO-PROXY-007 | VSaaS Visual AI Proxy Pipeline (dGPU) - VSaaS Visual AI proxy workload on dGPU for scalable stream analytics. |

**Run this profile**:
```bash
esq run --profile profile.vertical.metro
```

> **Note:** Running `esq run --profile profile.vertical.metro` also runs dependent profiles:
> `profile.suite.system.memory-stream`,
> `profile.suite.system.gpu-ov`,
> `profile.suite.ai.vision-ov`,
> `profile.suite.media.performance-pipelines`, and
> `profile.suite.ai.vision-va`.
> Total execution time depends on your hardware capabilities and available accelerators.

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

**Profile**: `profile.vertical.retail-lp-vlm`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------| 
| LP-VLM-001 | Loss Prevention VLM - 1080p15 H.264 gvadetect YOLO11n FP16 (CPU) VLM analysis Qwen2.5-VL-7B-Instruct INT8 (GPU) |

**Run this profile**:
```bash
esq run --profile profile.vertical.retail-lp-vlm
```

---

### Robotics

#### PI0.5 RTC Benchmark

**Profile**: `profile.vertical.robotics-pi5`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| PI5-RTC-001 | PI0.5 RTC Benchmark (GPU) |

**Run this profile**:
```bash
esq run --profile profile.vertical.robotics-pi5
```

---

#### ACT Benchmark

**Profile**: `profile.vertical.robotics-act`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| ROB-ACT-001 | ACT Benchmark (GPU) |

**Run this profile**:
```bash
esq run --profile profile.vertical.robotics-act
```

---

#### FastMapping Benchmark

**Profile**: `profile.vertical.robotics-fastmapping`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| ROB-FMP-001 | FastMapping Benchmark (CPU) |

**Run this profile**:
```bash
esq run --profile profile.vertical.robotics-fastmapping
```

---

#### ADBScan Benchmark

**Profile**: `profile.vertical.robotics-adbscan`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| ROB-ADB-001 | ADBScan Benchmark (CPU) |

**Run this profile**:
```bash
esq run --profile profile.vertical.robotics-adbscan
```

---

#### FunASR ASR Benchmark

**Profile**: `profile.vertical.robotics-funasr`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| ROB-ASR-001 | FunASR ASR Benchmark (CPU) |
| ROB-ASR-002 | FunASR ASR Benchmark (GPU) |
| ROB-ASR-003 | FunASR ASR Benchmark (NPU) |

**Run this profile**:
```bash
esq run --profile profile.vertical.robotics-funasr
```

---

## Horizontal

### Generative AI

#### LLM Serving Benchmark

**Profile**: `profile.suite.ai.gen-llm`

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
esq run --profile profile.suite.ai.gen-llm
```

---

#### Chat Question and Answer Core

**Profile**: `profile.suite.ai.gen-chatqna-core`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| GEN-CHAT-CONS-001 | Chat Q&A Core - Consolidated Functional Flow |
| GEN-CHAT-001 | Chat Q&A Core - OpenVINO* CPU | 
| GEN-CHAT-002 | Chat Q&A Core - OpenVINO* GPU |
| GEN-CHAT-003 | Chat Q&A Core - Ollama* CPU |
| GEN-CHAT-004 | Chat Q&A Core - Ollama* CPU (5 prompts) |

**Run this profile**:
```bash
esq run --profile profile.suite.ai.gen-chatqna-core
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

#### Video Analytics

**Profile**: `profile.suite.ai.vision-va`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| VSN-LIGHT-001 | VA Light - All Available Devices (YOLO11n + ResNet-50, H.265) |
| VSN-MEDIUM-001 | VA Medium - All Available Devices (YOLOv5m + ResNet-50 + MobileNet-v2, H.265) |
| VSN-HEAVY-001 | VA Heavy - All Available Devices (YOLO11m + ResNet-v1-50 + MobileNet-v2, H.265) |

**Run this profile**:
```bash
esq run --profile profile.suite.ai.vision-va
```

---

### Real-Time Performance

**Profile**: `profile.suite.realtime.performance`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| RT-CYC-001 | RT Wakeup Latency |
| RT-CYC-002 | RT Wakeup Latency (Optimized) |

**Run this profile**:
```bash
esq run --profile profile.suite.realtime.performance
```

---

### Real-Time Platform

**Profile**: `profile.suite.realtime.platform`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| RT-KRN-001 | PREEMPT-RT Kernel Check |
| RT-TSN-001 | TSN Detection |
| RT-TSC-001 | TSC Active Clocksource |
| RT-PCI-001 | PCI Multi-VC Detection |
| RT-CAT-001 | L3/L2 CAT Configuration |
| RT-CST-001 | RT Core C-State Disabled |
| RT-TMR-001 | Timer Migration Disabled |
| RT-FRQ-001 | Energy Performance Preference |
| RT-FRQ-002 | RT Core CPU Frequency Scaling Governor |

**Run this profile**:
```bash
esq run --profile profile.suite.realtime.platform
```

---

### System CPU

**Profile**: `profile.suite.system.cpu-sku`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| CPU-SKU-001 | CPU Socket Count |
| CPU-SKU-002 | Physical Core Count |
| CPU-SKU-003 | Logical Core Count |
| CPU-SKU-004 | Maximum CPU Frequency |
| CPU-SKU-005 | Minimum CPU Frequency |

**Run this profile**:
```bash
esq run --profile profile.suite.system.cpu-sku
```

---

### System Display

**Profile**: `profile.suite.system.display`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| SYS-DISP-001 | All Display Ports |
| SYS-DISP-002 | HDMI Ports |
| SYS-DISP-003 | DisplayPort |

**Run this profile**:
```bash
esq run --profile profile.suite.system.display
```

---

### System GPU

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
### System Memory

#### System Memory - STREAM

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

---

#### System Memory Health

**Profile**: `profile.suite.system.memory-health`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| MEM-HLT-001 | Memory ECC Error Scan (EDAC) |
| MEM-HLT-002 | Memory Bit-Pattern Test (memtester, 512 MB, 1 iter) |

**Run this profile**:
```bash
esq run --profile profile.suite.system.memory-health
```

---

### System Network

**Profile**: `profile.suite.system.network`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| NET-CON-001 | Active Network Connections |
| NET-CON-002 | Wired Network Connectivity |
| NET-CON-003 | Wireless Network Connectivity |

**Run this profile**:
```bash
esq run --profile profile.suite.system.network
```

---

### System Peripheral

**Profile**: `profile.suite.system.peripheral`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| PER-USB-001 | USB - Controllers |
| PER-USB-002 | USB - Devices |
| PER-USB-003 | USB - Keyboards |
| PER-USB-004 | USB - Mouse |
| PER-USB-005 | USB - Storage Devices |
| PER-USB-006 | USB - Audio Devices |
| PER-USB-007 | USB - Cameras |
| PER-USB-008 | USB - Hubs |
| PER-PS2-001 | PS/2 - Keyboards (Wired) |
| PER-PS2-002 | PS/2 - Mouse (Wired) |
| PER-NET-001 | Network - Controllers |
| PER-GFX-001 | Graphics - Controllers |
| PER-GFX-002 | Graphics - Connected Monitors |
| PER-INPUT-001 | Input - Total Keyboards (USB + PS/2) |
| PER-INPUT-002 | Input - Total Mouse (USB + PS/2) |

**Run this profile**:
```bash
esq run --profile profile.suite.system.peripheral
```

---

### System Stress

**Profile**: `profile.suite.system.stress`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| SNG-CPU-001 | CPU 100% (stress-ng) |
| SNG-CPU-002 | CPU 50% (stress-ng) |
| SNG-MEM-001 | Memory 100% (stress-ng) |
| SNG-MEM-002 | Memory 50% (stress-ng) |
| SNG-IGP-001 | iGPU Rendering (stress-ng) |
| SNG-MIX-001 | CPU 100% + iGPU Rendering (stress-ng) |
| SNG-MIX-002 | CPU 50% + iGPU Rendering (stress-ng) |

**Run this profile**:
```bash
esq run --profile profile.suite.system.stress
```

---

### Virtualization

#### KVM Compatibility

**Profile**: `profile.suite.virtualization.kvm`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| KVM-CPT-001 | Comprehensive KVM Compatibility |
| KVM-CPT-002 | Basic KVM Compatibility |
| KVM-CPT-003 | Device Passthrough Compatibility (VT-d/VFIO) |
| KVM-CPT-004 | Nested Virtualization Compatibility |
| KVM-CAP-001 | Peripheral VM Capacity |

**Run this profile**:
```bash
esq run --profile profile.suite.virtualization.kvm
```

---

#### QEMU* VM Reboot

**Profile**: `profile.suite.virtualization.kvm-qemu`

**Test Cases**:

| Test ID | Test Case |
|---------|-----------|
| KVM-RBT-001 | QEMU* VM Reboot Test (KVM, OVMF/UEFI, 2x) |
| KVM-RBT-002 | QEMU* VM Reboot Test (KVM, SeaBIOS, 2x) |

**Run this profile**:
```bash
esq run --profile profile.suite.virtualization.kvm-qemu
```

---

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

---

