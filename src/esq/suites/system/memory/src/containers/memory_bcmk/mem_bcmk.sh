#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on error

OUTPUT_DIR=/home/dlstreamer/output
STREAM_DIR=/home/dlstreamer/STREAM
RESULT_FILE=${OUTPUT_DIR}/memory_benchmark_runner.result

echo "========================================"
echo "Memory Benchmark Container Started"
echo "========================================"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output Directory: ${OUTPUT_DIR}"
echo "STREAM Directory: ${STREAM_DIR}"
echo "Result File: ${RESULT_FILE}"
echo "========================================"

run_mem_bcmk()
{
    echo "[INFO] Checking for STREAM binary..."
    if [ ! -f ${STREAM_DIR}/stream ]; then
        echo "[ERROR] STREAM binary not found at: ${STREAM_DIR}/stream"
        echo "[ERROR] Directory contents:"
        ls -la ${STREAM_DIR}/ || echo "[ERROR] Directory does not exist"
        exit 255
    fi
    echo "[INFO] STREAM binary found: ${STREAM_DIR}/stream"

    echo "[INFO] Checking output directory..."
    if [ ! -d ${OUTPUT_DIR} ]; then
        echo "[ERROR] Output directory does not exist: ${OUTPUT_DIR}"
        exit 254
    fi
    echo "[INFO] Output directory exists and is writable"

    echo "[INFO] Running STREAM benchmark..."
    echo "[INFO] Command: ${STREAM_DIR}/stream > ${RESULT_FILE}"
    echo "========================================"

    ${STREAM_DIR}/stream > ${RESULT_FILE} 2>&1
    EXIT_CODE=$?

    echo "========================================"
    echo "[INFO] STREAM benchmark completed with exit code: ${EXIT_CODE}"

    if [ -f ${RESULT_FILE} ]; then
        RESULT_SIZE=$(stat -f "%z" ${RESULT_FILE} 2>/dev/null || stat -c "%s" ${RESULT_FILE} 2>/dev/null || echo "unknown")
        echo "[INFO] Result file created: ${RESULT_FILE} (${RESULT_SIZE} bytes)"
        echo "[INFO] Result file preview (first 10 lines):"
        head -10 ${RESULT_FILE}
        echo "..."
    else
        echo "[ERROR] Result file was not created: ${RESULT_FILE}"
        exit 253
    fi

    echo "========================================"
    echo "[INFO] Memory Benchmark Container Finished Successfully"
    echo "========================================"

    return ${EXIT_CODE}
}

run_mem_bcmk
