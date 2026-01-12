#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

args=("$@")

# Use standard OpenVINO installation path
# Note: The base DLStreamer image provides OpenVINO at /opt/intel/openvino (symlink)
# The installer is responsible for creating this symlink to the versioned directory
OPENVINO_DIR="/opt/intel/openvino"

if [ ! -d "$OPENVINO_DIR" ]; then
    echo "[ERROR] OpenVINO not found at standard path: $OPENVINO_DIR"
    echo "[ERROR] Ensure OpenVINO is properly installed with symlink at /opt/intel/openvino"
    exit 1
fi

echo "[INFO] Using OpenVINO from: $OPENVINO_DIR"

# Source OpenVINO environment
if [ -f "$OPENVINO_DIR/setupvars.sh" ]; then
	# shellcheck disable=SC1091 # OpenVINO path not available at static analysis time
	source "$OPENVINO_DIR/setupvars.sh"
else
    echo "[ERROR] OpenVINO setupvars.sh not found at: $OPENVINO_DIR/setupvars.sh"
    exit 1
fi

# Build benchmark_app samples
if [ -f "$OPENVINO_DIR/samples/cpp/build_samples.sh" ]; then
	if ! (cd "$OPENVINO_DIR/samples/cpp" && ./build_samples.sh > /dev/null 2>&1); then
		echo "[ERROR] Failed to build benchmark_app. Check OpenVINO installation."
		exit 255
	fi
	echo "[INFO] benchmark_app built successfully"
else
	echo "[ERROR] build_samples.sh not found at: $OPENVINO_DIR/samples/cpp/build_samples.sh"
	exit 1
fi

# Execute the benchmark runner script
exec python3 run_benchmark.py "${args[@]}"
