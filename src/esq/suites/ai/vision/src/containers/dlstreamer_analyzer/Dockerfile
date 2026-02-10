# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM intel/dlstreamer:2025.2.0-ubuntu24

USER root

# Set up directories with proper ownership (dlstreamer user already exists with UID 1000)
RUN mkdir -p /app && \
    chown dlstreamer:dlstreamer /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        python3-gi \
        python3-gst-1.0 \
        gir1.2-gst-plugins-base-1.0 \
    && rm -rf /var/lib/apt/lists/*

# Copy main.py and make it executable
COPY main.py /app/main.py
RUN chmod +x /app/main.py && \
    chown dlstreamer:dlstreamer /app/main.py

# Set up workspace with secure permissions
RUN mkdir -p /mnt/results /mnt/logs /mnt/videos && \
    chmod 770 /mnt /mnt/results /mnt/logs /mnt/videos && \
    chown -R dlstreamer:dlstreamer /mnt

# Add healthcheck for operational monitoring and security compliance
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 --version || exit 1

WORKDIR /app

# Switch to non-root user
USER dlstreamer

ENTRYPOINT ["/app/main.py"]
