#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

args=("$@")

if [ -f /opt/intel/openvino/setupvars.sh ]; then
	# shellcheck disable=SC1091 # OpenVINO path not available at static analysis time
	source /opt/intel/openvino/setupvars.sh
else
    echo "[WARNING] OpenVINO setupvars.sh not found in container. Did you mount it?"
fi

if [ -f /opt/intel/openvino/samples/cpp/build_samples.sh ]; then
	if ! (cd /opt/intel/openvino/samples/cpp && ./build_samples.sh > /dev/null); then
		echo "Build benchmark_app Failed. Stop test."
		exit 255
	fi
else
	echo "[WARNING] OpenVINO samples build script not found in container."
fi

echo "OpenVINO environment initialized"

# Validate that share directory is mounted with correct structure
# Expected: esq_data/data/vertical/metro mounted to /home/dlstreamer/share
if [ ! -d "/home/dlstreamer/share" ]; then
    echo "[ERROR] Share directory not found at /home/dlstreamer/share"
    echo "[ERROR] Expected structure: /home/dlstreamer/share/{models,images,videos}"
    echo "[ERROR] Ensure data directory (esq_data/data/vertical/metro) is mounted correctly"
    exit 1
fi

# Check if models directory exists
if [ ! -d "/home/dlstreamer/share/models" ]; then
    echo "[ERROR] Models directory not found at /home/dlstreamer/share/models"
    echo "[ERROR] Ensure models are downloaded to esq_data/data/vertical/metro/models"
    ls -la /home/dlstreamer/share/ 2>/dev/null || echo "Share directory not accessible"
    exit 1
fi

echo "Share directory validation passed"

exec python3 AI_box_runtimeDL.py "${args[@]}"
