# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM openvino/model_server:2025.4.1-gpu

USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        software-properties-common \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Update NPU drivers to latest version
RUN curl -O https://raw.githubusercontent.com/open-edge-platform/edge-developer-kit-reference-scripts/49473c01212ee84c53b0cff75581d0c9b6201794/npu_installer.sh 
RUN chmod +x ./npu_installer.sh 
# Create udev directory for container environment and run installer (ignore udev reload failures)
RUN mkdir -p /etc/udev/rules.d && ./npu_installer.sh || true

USER ovms

# Add health check to prevent trivy dockerfile issues
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/config || exit 1

# Use the default OVMS entrypoint
CMD ["/ovms/bin/ovms"]
    