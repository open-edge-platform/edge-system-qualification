# Test Suites

Each test case of a [profile](../profiles/index.md) runs a **test suite** with the parameters set in the profile. One suite can serve many profiles, for example the DL Streamer suite runs the manufacturing, retail, and qualification vision test cases.

This section has one page per test suite. Each page describes what the suite measures, its [integration type](../guides/developer/integration-types.md), its key metric, all its parameters, and the profiles that use it.

## Test Suites

Type 1 is a [generic suite](../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients) that takes its workload from the profile. Type 2 [reproduces the original flow](../guides/developer/integration-types.md#type-2-reproduce-the-original-flow) of an upstream application. System checks, such as CPU or network checks, are also Type 1: ESQ runs the check directly and the profile selects what to check.

| Category | Suite | What it measures | Type | Key metric |
|----------|-------|------------------|------|------------|
| AI | [DL Streamer](ai/dlstreamer.md) | Concurrent video streams of a DL Streamer pipeline | Type 1 | `streams_max` |
|  | [VLM Prompt](ai/vlm-prompt.md) | VLM token throughput and stream density with prompts and images | Type 1 | `output_token_throughput`, `streams_max` |
|  | [Text Generation](ai/text-generation.md) | LLM serving throughput on OpenVINO Model Server | Type 1 | `throughput` |
|  | [Chat Q&A Core](ai/chatqna-core.md) | RAG chat application query latency | Type 2 | `p95_query_latency_ms` |
|  | [OpenVINO Benchmark](ai/ov-benchmark.md) | Raw model inference throughput | Type 1 | `throughput` |
|  | [Video Analytics](ai/video-analytics.md) | Streams of built-in light, medium, and heavy pipelines | Type 1 | `max_streams` |
|  | [FunASR](ai/funasr.md) | Speech recognition model inference throughput | Type 1 | `throughput` |
|  | [Whisper](ai/whisper.md) | Speech recognition latency and accuracy | Type 1 | `latency_ms` |
|  | [ADBScan](ai/adbscan.md) | Point-cloud clustering rate | Type 1 | `throughput` |
|  | [Wind Turbine](ai/wind-turbine.md) | Time-series anomaly detection throughput | Type 2 | `throughput` |
| Media | [Media Pipelines](media/media-pipelines.md) | Decode, compose, and encode channels at 30 FPS | Type 1 | `max_streams` |
| Real-Time | [Cyclictest](realtime/cyclictest.md) | Real-time wakeup latency under load | Type 1 | `max_latency_us` |
|  | [Platform Checks](realtime/platform.md) | Kernel, clock, cache, and power settings for real-time | Type 1 | One per check |
| System | [CPU](system/cpu.md) | CPU identification, cores, frequencies, and compatibility | Type 1 | Per `kpi_refs` |
|  | [Display](system/display.md) | Connected display ports | Type 1 | `total_connected` |
|  | [GPU](system/gpu.md) | GPU frequency and power under AI load | Type 1 | `frequency_max_<device>` |
|  | [Memory Health](system/memory-health.md) | ECC errors and memtester faults | Type 1 | `uncorrectable_errors`, `memory_errors` |
|  | [Memory STREAM](system/memory-stream.md) | Memory bandwidth | Type 1 | `best_rate` |
|  | [Network](system/network.md) | Connected network interfaces | Type 1 | `total_connected` |
|  | [Peripheral](system/peripheral.md) | USB, PS/2, graphics, and monitor devices | Type 1 | Count of the checked type |
|  | [stress-ng](system/stress-ng.md) | Stability under CPU, memory, and GPU stress | Type 1 | `cpu_bogo_ops_per_real_time` |
| Vertical | [Order Accuracy](vertical/order-accuracy.md) | Take-Away and Dine-In applications, VLM throughput | Type 2 | `throughput_mean_sec` |
|  | [Loss Prevention VLM](vertical/loss-prevention-vlm.md) | Loss Prevention application, VLM throughput | Type 2 | `throughput_mean` |
|  | [Proxy Pipelines](vertical/proxy-pipelines.md) | Metro proxy pipeline streams at 30 FPS | Type 1 | `max_streams` |
| Virtualization | [KVM](virtualization/kvm.md) | KVM capability, VM capacity, and VM reboot time | Type 1 | Per test |
