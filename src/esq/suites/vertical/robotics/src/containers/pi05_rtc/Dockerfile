# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG COMMON_BASE_IMAGE

FROM ${COMMON_BASE_IMAGE} AS build

USER root
RUN apt-get update -y && \
    apt-get install --no-install-recommends -y \
        git \
        curl \
        gpg \
        build-essential \
        python3-dev \
        libgl1 \
        libglib2.0-0 \
        linux-libc-dev \
        software-properties-common && \
    apt-get install --reinstall --no-install-recommends -y \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Update CA certificates to ensure the latest trusted root certificates are available
RUN update-ca-certificates --fresh

COPY setup.sh .
RUN chmod +x setup.sh && bash setup.sh

# Staged build
FROM ${COMMON_BASE_IMAGE}

# Install the Intel GPU user-space runtime from the kobuk-team/intel-graphics PPA.
# Provides the OpenCL ICD loader (libOpenCL.so.1) required by the OpenVINO GPU
# plugin and the Level Zero loader (libze_loader.so.1) used by both GPU and NPU
# plugins, plus the Intel compute driver needed to use /dev/dri. libtbb12 is a
# runtime dependency of the Intel NPU compiler installed below.
USER root
RUN apt-get update -y && \
    apt-get install --no-install-recommends -y \
        software-properties-common \
        curl \
        gpg \
        ca-certificates && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get update -y && \
    apt-get install --no-install-recommends -y \
        python3 \
        python3-pip \
        intel-opencl-icd \
        intel-ocloc \
        libze1 \
        libze-intel-gpu1 \
        libtbb12 && \
    rm -rf /var/lib/apt/lists/*

# Install the Intel NPU user-space driver (Level Zero NPU runtime + compiler) from
# the official linux-npu-driver release. These packages are not published to an apt
# repository, so they are fetched as a verified release tarball of .deb files. The
# version matches the host kernel driver (intel_vpu 1.33.0) so the OpenVINO NPU
# plugin can access the device at /dev/accel/accel0.
ARG NPU_DRIVER_VERSION="v1.33.0"
ARG NPU_DRIVER_TARBALL="linux-npu-driver-v1.33.0.20260529-26625960453-ubuntu2404.tar.gz"
ARG NPU_DRIVER_SHA256="48cc219a5bd882a7cadaef523d9bb0c1e03c1b9b5b8562a29ad5ee4b9bfbb3bc"
# Use bash with pipefail so failures in piped commands (e.g. curl | sha256sum)
# are detected. The default /bin/sh (dash) does not support "set -o pipefail".
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -fsSL -o /tmp/npu-driver.tar.gz \
        "https://github.com/intel/linux-npu-driver/releases/download/${NPU_DRIVER_VERSION}/${NPU_DRIVER_TARBALL}" && \
    echo "${NPU_DRIVER_SHA256}  /tmp/npu-driver.tar.gz" | sha256sum -c - && \
    mkdir -p /tmp/npu-driver && \
    tar -xf /tmp/npu-driver.tar.gz -C /tmp/npu-driver && \
    find /tmp/npu-driver -name 'intel-*npu_*.deb' ! -name '*dbgsym*' -exec dpkg -i {} + && \
    rm -rf /tmp/npu-driver /tmp/npu-driver.tar.gz

RUN pip install --break-system-packages openvino

# Create a dedicated system group and unprivileged user
RUN groupadd -r appgroup && useradd -r -g appgroup -m -d /home/appuser appuser

USER appuser
ENV HOME=/home/appuser
WORKDIR /home/appuser

COPY --from=build --chown=appuser:appgroup /resources /resources
COPY --chown=appuser:appgroup test.sh .
RUN chmod a+rx test.sh && chmod o+rx "$HOME"
HEALTHCHECK CMD [ "cat", "/dev/null" ]
ENTRYPOINT [ "./test.sh" ]
