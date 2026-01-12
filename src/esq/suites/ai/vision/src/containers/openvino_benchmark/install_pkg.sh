#!/bin/bash
#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
set -e -o pipefail

IS_XEON_PLATFORM=${IS_XEON_PLATFORM:-false}
echo "Evaluating IS_XEON_PLATFORM as: $IS_XEON_PLATFORM" >&2

if [ "$IS_XEON_PLATFORM" = "true" ]; then
	apt-get update && apt-get install libgsl27 libasound2-plugins libasound2-data libxmlrpc-core-c3 git curl unzip sysstat libgl1 numactl cmake python3-pip -y; \
else

    apt-get update -y && apt-get install libgsl27 libasound2-plugins libasound2-data libxmlrpc-core-c3 git curl unzip intel-gpu-tools xpu-smi sysstat numactl cmake python3-pip -y; \
fi
apt-get clean && rm -rf /var/lib/apt/lists/*

# Install OpenVINO inside the container
if [ -f /home/dlstreamer/dlstreamer/scripts/install_dependencies/install_openvino.sh ]; then \
   # shellcheck disable=SC1091 # DLStreamer install script path not available at static analysis time
   source /home/dlstreamer/dlstreamer/scripts/install_dependencies/install_openvino.sh && \
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
apt-get clean && rm -rf /var/lib/apt/lists/*
