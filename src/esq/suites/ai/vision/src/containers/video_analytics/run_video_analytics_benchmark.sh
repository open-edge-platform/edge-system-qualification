#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Video Analytics Benchmark Sequential Runner
#
# This script runs video analytics benchmarks for multiple devices SEQUENTIALLY.
# Changed from parallel to sequential execution to prevent OOM when running
# multiple devices simultaneously. Each device gets full system memory.
#
# Arguments: <device1> [device2] ... <display_output> <is_mtl> <has_igpu> <config_file> <benchmark_script>
#
# Example: ./run_video_analytics_benchmark.sh iGPU dGPU 0 true true none va_benchmark

set -e

# Cleanup function to terminate orphaned processes on signal
# shellcheck disable=SC2317  # Function is invoked indirectly via trap
cleanup() {
    echo "[CLEANUP] Received termination signal, cleaning up..."
    # Kill any orphaned gst-launch-1.0 processes
    pkill -9 -f "gst-launch-1.0" 2>/dev/null || true
    echo "[CLEANUP] Cleanup complete"
}

# Trap signals for cleanup
trap cleanup SIGTERM SIGINT SIGHUP EXIT

# Parse arguments
ARGS=("$@")
ARG_COUNT=$#

# Last 4 arguments are special
BENCHMARK_SCRIPT="${ARGS[$((ARG_COUNT-1))]}"
CONFIG_FILE="${ARGS[$((ARG_COUNT-2))]}"
HAS_IGPU="${ARGS[$((ARG_COUNT-3))]}"
IS_MTL="${ARGS[$((ARG_COUNT-4))]}"
DISPLAY_OUTPUT="${ARGS[$((ARG_COUNT-5))]}"

# Remaining arguments are devices
DEVICE_COUNT=$((ARG_COUNT-5))
DEVICES=("${ARGS[@]:0:$DEVICE_COUNT}")

echo "========================================"
echo "Video Analytics Benchmark Sequential Runner"
echo "========================================"
echo "Benchmark Script: ${BENCHMARK_SCRIPT}"
echo "Config File: ${CONFIG_FILE}"
echo "Is MTL: ${IS_MTL}"
echo "Has iGPU: ${HAS_IGPU}"
echo "Display Output: ${DISPLAY_OUTPUT}"
echo "Devices: ${DEVICES[*]}"
echo "Device Count: ${DEVICE_COUNT}"
echo "========================================"

# Validate benchmark script
if [ -z "${BENCHMARK_SCRIPT}" ]; then
    echo "[ERROR] Benchmark script not specified"
    exit 1
fi

# Map benchmark script name to Python file
case "${BENCHMARK_SCRIPT}" in
    "va_benchmark"|"va_light")
        PYTHON_SCRIPT="va_benchmark.py"
        CSV_NAME="va_proxy_pipeline"
        ;;
    "va_medium_benchmark"|"va_medium")
        PYTHON_SCRIPT="va_medium_benchmark.py"
        CSV_NAME="va_medium_pipeline"
        ;;
    "va_heavy_benchmark"|"va_heavy")
        PYTHON_SCRIPT="va_heavy_benchmark.py"
        CSV_NAME="va_heavy_pipeline"
        ;;
    *)
        echo "[ERROR] Unknown benchmark script: ${BENCHMARK_SCRIPT}"
        echo "Available options: va_benchmark, va_light, va_medium_benchmark, va_medium, va_heavy_benchmark, va_heavy"
        exit 1
        ;;
esac

# Check if script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "[ERROR] Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

echo "[INFO] Using Python script: ${PYTHON_SCRIPT}"

# Initialize CSV file if needed
OUTPUT_DIR="/home/dlstreamer/output"

# Use custom CSV filename from environment variable if provided, otherwise use default
if [ -n "${VA_CSV_FILENAME}" ]; then
    CSV_FILE="${OUTPUT_DIR}/${VA_CSV_FILENAME}"
    echo "[INFO] Using custom CSV filename from environment: ${VA_CSV_FILENAME}"
else
    CSV_FILE="${OUTPUT_DIR}/${CSV_NAME}.csv"
    echo "[INFO] Using default CSV filename: ${CSV_NAME}.csv"
fi

if [ ! -f "${CSV_FILE}" ]; then
    echo "[INFO] Initializing CSV file: ${CSV_FILE}"
    echo "TC Name,Model,Mode,Devices,Result,Streams,GPU Freq,Pkg Power,Ref Platform,Ref FPS,Ref GPU Freq,Ref Pkg Power,Duration(s),Error" > "${CSV_FILE}"
fi

# Run benchmark for each device
EXIT_CODES=()

# Sequential execution to avoid OOM when running multiple devices
# Each device gets full system memory for its benchmark
for device in "${DEVICES[@]}"; do
    echo "[INFO] ========================================"
    echo "[INFO] Starting benchmark for device: ${device}"
    echo "[INFO] ========================================"
    
    # Determine monitor number based on device type
    case "${device}" in
        iGPU|GPU|GPU.0)
            MONITOR_NUM=0
            ;;
        dGPU|dGPU.0|GPU.1)
            MONITOR_NUM=1
            ;;
        dGPU.1|GPU.2)
            MONITOR_NUM=2
            ;;
        CPU)
            MONITOR_NUM=0
            ;;
        NPU)
            MONITOR_NUM=0
            ;;
        *)
            echo "[WARNING] Unknown device type: ${device}, using monitor 0"
            MONITOR_NUM=0
            ;;
    esac
    
    LOG_FILE="${OUTPUT_DIR}/va_${device}_benchmark.log"
    
    echo "[INFO] Running: python3 ${PYTHON_SCRIPT} --device ${device} --monitor_num ${MONITOR_NUM} --is_mtl ${IS_MTL} --has_igpu ${HAS_IGPU} --config_file ${CONFIG_FILE}"
    
    # Run benchmark SEQUENTIALLY (not in background) to avoid OOM
    # This matches the LPR benchmark pattern which was changed from parallel to sequential
    # to prevent memory exhaustion when multiple devices run simultaneously
    if python3 "${PYTHON_SCRIPT}" \
        --device "${device}" \
        --monitor_num "${MONITOR_NUM}" \
        --is_mtl "${IS_MTL}" \
        --has_igpu "${HAS_IGPU}" \
        --config_file "${CONFIG_FILE}" \
        2>&1 | tee "${LOG_FILE}"; then
        echo "[INFO] Benchmark for ${device} completed successfully"
        EXIT_CODES+=("0")
    else
        exit_code=$?
        echo "[ERROR] Benchmark for ${device} failed with exit code ${exit_code}"
        EXIT_CODES+=("${exit_code}")
    fi
    
    echo "[INFO] Finished benchmark for device: ${device}"
    echo ""
done

echo "========================================"
echo "Video Analytics Benchmark Summary"
echo "========================================"

# Check if any benchmark failed
FAILED=0
for i in "${!DEVICES[@]}"; do
    device=${DEVICES[$i]}
    exit_code=${EXIT_CODES[$i]}
    
    if [ "${exit_code}" == "0" ]; then
        echo "[OK] ${device}: Success"
    else
        echo "[FAIL] ${device}: Exit code ${exit_code}"
        FAILED=1
    fi
done

echo "========================================"

if [ "${FAILED}" == "1" ]; then
    echo "[WARNING] Some benchmarks failed"
    exit 1
else
    echo "[SUCCESS] All benchmarks completed successfully"
    exit 0
fi
