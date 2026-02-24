# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Video Analytics (VA) Benchmark Container
#
# Based on Intel DL Streamer with additional benchmark utilities

ARG COMMON_BASE_IMAGE=intel/dlstreamer:2025.2.0-ubuntu24
FROM ${COMMON_BASE_IMAGE}

USER root

# Install additional dependencies for benchmarking
# Matches proxy test suite packages for telemetry compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-psutil \
    lshw \
    pciutils \
    sudo \
    intel-gpu-tools \
    xpu-smi \
    numactl \
    libgsl27 \
    libasound2-plugins \
    libasound2-data \
    libxmlrpc-core-c3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure sudoers for telemetry tools (matches proxy test suite)
RUN set -e -o pipefail && \
    echo "dlstreamer ALL=(ALL) NOPASSWD: /usr/bin/intel_gpu_top,/usr/bin/rm,/usr/bin/xpu-smi,/usr/local/bin/intel_gpu_top,/usr/local/bin/xpu-smi" | \
    tee /etc/sudoers.d/dlstreamer && \
    chmod 0440 /etc/sudoers.d/dlstreamer

# Create working directories
RUN mkdir -p /home/dlstreamer/output \
    && mkdir -p /home/dlstreamer/share/models \
    && mkdir -p /home/dlstreamer/sample_video \
    && mkdir -p /home/dlstreamer/sink

# Copy benchmark utilities
COPY esq_utils /home/dlstreamer/esq_utils

# Copy model-proc files for YOLO post-processing
COPY model_proc /home/dlstreamer/model_proc

# Copy benchmark scripts
COPY base_benchmark.py /home/dlstreamer/base_benchmark.py
COPY va_benchmark.py /home/dlstreamer/va_benchmark.py
COPY va_medium_benchmark.py /home/dlstreamer/va_medium_benchmark.py
COPY va_heavy_benchmark.py /home/dlstreamer/va_heavy_benchmark.py
COPY run_video_analytics_benchmark.sh /home/dlstreamer/run_video_analytics_benchmark.sh
COPY run_pipeline.sh /home/dlstreamer/run_pipeline.sh

# Make scripts executable
RUN chmod +x /home/dlstreamer/run_video_analytics_benchmark.sh \
    && chmod +x /home/dlstreamer/run_pipeline.sh

# Set working directory
WORKDIR /home/dlstreamer

# Set permissions for output directory
RUN chmod 777 /home/dlstreamer/output

# Switch to non-root user for container execution
USER dlstreamer

# Add healthcheck for operational monitoring
HEALTHCHECK CMD [ "cat", "/dev/null" ]

# Default command
CMD ["/bin/bash"]
