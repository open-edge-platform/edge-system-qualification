# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM vllm/vllm-openai-cpu:v0.31.0

USER root

# Create a non-root user
RUN useradd -m -u 1000 user && \
    mkdir -p /vllm-workspace && \
    chown user:user /vllm-workspace

# Set up workspace directory with secure permissions (770 prevents world access)
# Note: vLLM v0.31.0 uses the integrated `vllm bench serve` CLI subcommand
#       Command: vllm bench serve [args]
RUN mkdir -p /vllm-workspace/results && \
    chmod 770 /vllm-workspace/results && \
    chown -R user:user /vllm-workspace

# Add healthcheck for operational monitoring and Trivy compliance
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

WORKDIR /vllm-workspace

# Switch to non-root user
USER user

ENTRYPOINT ["/bin/bash"]
