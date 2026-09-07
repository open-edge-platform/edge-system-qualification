# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        wget \
        ffmpeg \
        python3-pip \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install GPU userspace drivers via Intel Kobuk PPA (kernel modules run on the host)
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        intel-opencl-icd \
        libze-intel-gpu1 \
        libze1 \
        intel-media-va-driver-non-free \
        libmfx-gen1 \
        libvpl2 \
        va-driver-all \
        clinfo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Provide the uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pytorch/requirements.txt .

RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt

COPY pytorch/pytorch_backend.py .
COPY utils/metrics_helper.py .

RUN useradd --create-home user

RUN chown -R user:user /app

USER user

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch" || exit 1

CMD ["python3", "pytorch_backend.py"]
