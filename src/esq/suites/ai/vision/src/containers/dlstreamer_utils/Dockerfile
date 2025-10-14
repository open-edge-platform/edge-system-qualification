# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM debian:12-slim
ARG DEBIAN_FRONTEND=noninteractive

# Install necessary packages for video processing
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        python3 \
        ffmpeg \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-tools \
        gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -r user && \
    useradd -r -g user -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user:user /home/user

# Set up working directory and permissions
WORKDIR /app
COPY main.py /app/
RUN chown -R user:user /app

# Switch to non-root user
USER user

# Set the working directory for the non-root user
WORKDIR /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 --version || exit 1
