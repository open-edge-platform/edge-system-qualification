# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Dockerfile for sysbench performance testing in Ubuntu 24.04
FROM ubuntu:24.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Update package list and install required packages for performance testing
RUN apt-get update && apt-get install -y --no-install-recommends \
    sysbench \
    build-essential \
    htop \
    procps \
    vim \
    jq \
    python3 \
    python3-pip \
    coreutils \
    util-linux \
    && rm -rf /var/lib/apt/lists/*
# Add a simple healthcheck to verify the container is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD [ -f /app/sysbench_benchmark.sh ] && [ -x /app/sysbench_benchmark.sh ] || exit 1

# Ubuntu 24.04 ships with a default non-root user ubuntu:1000:1000
# Create a working directory and set ownership to ubuntu user
WORKDIR /app
RUN chown ubuntu:ubuntu /app

# Create results directory with proper permissions for volume mounting
RUN mkdir -p /results && chown ubuntu:ubuntu /results && chmod 755 /results

# Create temp directory with proper permissions
RUN mkdir -p /tmp && chmod 1777 /tmp

# Copy our test scripts
COPY sysbench_benchmark.sh /app/
COPY sysbench_analysis.py /app/

# Make the bash script executable and set ownership to ubuntu user
RUN chmod +x /app/sysbench_benchmark.sh \
    && chown ubuntu:ubuntu /app/sysbench_benchmark.sh /app/sysbench_analysis.py

# Python packages are not needed for this container since we use built-in modules
# RUN pip3 install --no-cache-dir [packages]

# Switch to non-root ubuntu user (UID 1000)
USER ubuntu

# Set the default command
CMD ["/bin/bash"]
