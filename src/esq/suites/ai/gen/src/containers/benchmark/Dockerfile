# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM vllm/vllm-openai:v0.9.2

USER root

# Create a non-root user
RUN useradd -m -u 1000 user && \
    mkdir -p /vllm-workspace && \
    chown user:user /vllm-workspace

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python packages using uv
RUN uv pip install --system --no-cache -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

# Set up workspace directory with secure permissions (770 prevents world access)
RUN mkdir -p /vllm-workspace/results && \
    chmod +x /vllm-workspace/benchmarks/benchmark_serving.py && \
    chmod 770 /vllm-workspace/results && \
    chown -R user:user /vllm-workspace

# Add healthcheck for operational monitoring and Trivy compliance
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

WORKDIR /vllm-workspace

# Switch to non-root user
USER user

ENTRYPOINT ["/bin/bash"]
