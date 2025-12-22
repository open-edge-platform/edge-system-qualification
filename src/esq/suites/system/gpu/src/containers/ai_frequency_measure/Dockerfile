# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG COMMON_BASE_IMAGE

FROM ${COMMON_BASE_IMAGE}

ARG ENABLE_PRC_NETWORK
ARG NO_POC_PLATFORM
ARG IS_XEON_PLATFORM

USER root

# hadolint ignore=DL4006,SC3040
RUN set -e -o pipefail; \
    if [ "$IS_XEON_PLATFORM" = "true" ] ; then \
        apt-get update && apt-get install --no-install-recommends build-essential python3-dev python3-pip python3-setuptools cython3 pciutils git curl unzip numactl wget cmake python3-pip -y; \
    else \
        apt-get update && apt-get install --no-install-recommends build-essential python3-dev python3-pip python3-setuptools cython3 pciutils git curl unzip intel-gpu-tools xpu-smi numactl wget cmake python3-pip -y; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Install OpenVINO inside the container
RUN set -e -o pipefail; \
    if [ -f /home/dlstreamer/dlstreamer/scripts/install_dependencies/install_openvino.sh ]; then \
      bash /home/dlstreamer/dlstreamer/scripts/install_dependencies/install_openvino.sh && \
      ov_dir=$(find /opt/intel -maxdepth 1 -type d -name "openvino_*" | sort -r | head -n 1) && \
      if [ -n "$ov_dir" ]; then \
        ln -sf "$ov_dir" /opt/intel/openvino; \
        echo "Linked $ov_dir -> /opt/intel/openvino"; \
      else \
        echo "OpenVINO folder not found after install"; \
      fi; \
    else \
      echo "install_openvino.sh not found in image"; \
    fi

# Clean up cache to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Create necessary folders and set ownership
# Note: /home/dlstreamer/share will be mounted from esq_data/data/vertical/metro at runtime
RUN mkdir -p /home/dlstreamer/output && \
    mkdir -p /home/dlstreamer/share/models && \
    mkdir -p /home/dlstreamer/share/images && \
    chown -R dlstreamer:dlstreamer /home/dlstreamer/output && \
    chown -R dlstreamer:dlstreamer /home/dlstreamer/share && \
    chmod -R 777 /home/dlstreamer/output

# Copy requirements
COPY  --chown=dlstreamer:dlstreamer requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

USER dlstreamer
ENV HOME=/home/dlstreamer
WORKDIR /home/dlstreamer

# Entrypoint will handle setting up OpenVINO environment
COPY  --chown=dlstreamer:dlstreamer entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Copy run script
COPY --chown=dlstreamer:dlstreamer AI_box_runtimeDL.py .
COPY --chown=dlstreamer:dlstreamer bcmk_telemetry.py gpu_util.py ./

HEALTHCHECK CMD [ "cat", "/dev/null" ]
ENTRYPOINT ["./entrypoint.sh"]
