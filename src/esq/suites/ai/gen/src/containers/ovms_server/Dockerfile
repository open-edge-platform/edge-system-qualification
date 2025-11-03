# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM openvino/model_server:2025.3-gpu

USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        software-properties-common \
        curl \
    && rm -rf /var/lib/apt/lists/*

USER ovms

# Add health check to prevent trivy dockerfile issues
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/config || exit 1

# Use the default OVMS entrypoint
CMD ["/ovms/bin/ovms"]
    