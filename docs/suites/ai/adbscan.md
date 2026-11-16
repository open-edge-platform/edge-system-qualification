# ADBScan Clustering

Runs the ROS 2 ADBScan point-cloud clustering node with the ROS benchmark framework in a Docker container and measures its processing rate, latency, and jitter.

| | |
|---|---|
| **Test suite** | `test_adbscan` (`src/esq/suites/ai/clustering/test_adbscan.py`, function `test_clustering_adbscan`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). ESQ builds its own benchmark container from the published ROS packages, and the profile selects the device. |
| **Key metric** | `throughput` (Hz), with `mean_latency` (ms) and jitter statistics |
| **Devices** | CPU |

## How It Works

1. Build the benchmark image on the ROS base image, with the `adbscan-ros2` and `benchmark-framework` packages.
2. Run the ADBScan benchmark in the container.
3. Read the benchmark report and record throughput, latency, and jitter.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `gpu` | Device passed to the benchmark. The profile sets `CPU`. |
| `operation` | `adbscan` | Benchmark to run in the container |
| `container_image`, `image_tag`, `dockerfile_name`, `docker_base_image` | `robotics_adbscan_benchmark`, `1.0`, `Dockerfile`, `amd64/ros:jazzy-ros-base` | Benchmark image build |
| `docker_nocache` | `false` | Rebuild the image without the Docker cache |
| `timeout` | `300` | Seconds for the test case |

## Profiles

<!-- esq:suite-profiles test_adbscan -->

## Run

```bash
esq run --profile profile.horizontal.ai.clustering-adbscan --filter test_id=CLU-ADB-001
```
