# Memory Bandwidth (STREAM)

Measures sustainable memory bandwidth with the STREAM benchmark for one of its four operations.

| | |
|---|---|
| **Test suite** | `test_memory_stream` (`src/esq/suites/system/memory/test_memory_stream.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). It wraps the STREAM benchmark, and the profile selects the operation. |
| **Key metric** | `best_rate` (MB/s), with `avg_time`, `min_time`, and `max_time` |
| **Devices** | System memory |

## How It Works

1. Build the benchmark image. It clones STREAM and compiles it with OpenMP and an array size well above the CPU cache size.
2. Run STREAM in the container.
3. Report the best rate and the run times of `operation`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `operation` | `Copy` | STREAM operation: `Copy`, `Scale`, `Add`, or `Triad` |
| `container_image`, `image_tag`, `dockerfile_name`, `docker_nocache` | `stream_memory_benchmark`, `1.0`, `Dockerfile`, `false` | Benchmark image build |
| `timeout` | `300` | Seconds for the test case |

## Profiles

<!-- esq:suite-profiles test_memory_stream -->

## Run

```bash
esq run --profile profile.horizontal.system.memory-stream --filter test_id=MEM-STR-001
```
