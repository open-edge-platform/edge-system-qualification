#!/bin/bash
#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Script metadata
baseFileName=$(basename "$0")
time='date +"%Y-%m-%d %H:%M:%S"'

# Terminal colors
_yellow='\033[1;33m'
_reset='\033[0m'

# Arguments passed from test: device_id operation codec bitrate resolution display_output is_mtl has_igpu
DEVICE=$1
OPERATION=$2
CODEC=$3
BITRATE=$4
RESOLUTION=$5
DISPLAY_OUTPUT=$6  # 0=headless (fakesink), 1=display enabled (xvimagesink)
IS_MTL=$7
HAS_IGPU=$8

# Convert display_output to monitor_num for Python script (monitor_num is count of displays)
MONITOR_NUM=$DISPLAY_OUTPUT

DETAIL_LOG_FILE=/home/dlstreamer/output/media_performance_benchmark_runner.log
CONTAINER_NAME=media_bm_runner
CONTAINER_TAG=1.0

format_log_section()
{
    body=$1
    _returnCode=$2  # Reserved for future use (e.g., exit code logging)
    funcName=$3
    lineNo=$4
    detailsLogFile=$5

    # Calculate prefix length and create separator line
    local prefix
    prefix="[$(eval "$time")][$baseFileName][$funcName][line:$lineNo]"
    local len=${#prefix}
    local splitStr
    splitStr=$(printf '%0.s-' $(seq 1 $((120-len))))

    echo -e "\n"
    echo "${prefix}${splitStr}" | tee -a "$detailsLogFile"
    echo "${prefix}${_yellow}${body}${_reset}" 2>&1 | tee -a "$detailsLogFile"
    echo "${prefix}${splitStr}" | tee -a "$detailsLogFile"
}

run_benchmark() {
    format_log_section "Running Media Performance Benchmark with params: [container: ${CONTAINER_NAME}:${CONTAINER_TAG}; device:${DEVICE}; operation:${OPERATION}; codec:${CODEC}; bitrate:${BITRATE}; resolution:${RESOLUTION}]." "" "${FUNCNAME[0]}" "$LINENO" "${DETAIL_LOG_FILE}"

    # Run media benchmark with parameters from test
    python3 ./media_benchmark.py \
        --device "${DEVICE}" \
        --monitor_num "${MONITOR_NUM}" \
        --is_mtl "${IS_MTL}" \
        --has_igpu "${HAS_IGPU}" \
        --operation "${OPERATION}" \
        --codec "${CODEC}" \
        --bitrate "${BITRATE}" \
        --resolution "${RESOLUTION}"

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        format_log_section "Media Performance Benchmark execution [device:${DEVICE}] finished successfully" "0" "${FUNCNAME[0]}" "$LINENO" "${DETAIL_LOG_FILE}"
    else
        format_log_section "Media Performance Benchmark execution [device:${DEVICE}] failed with exit code: ${exit_code}" "${exit_code}" "${FUNCNAME[0]}" "$LINENO" "${DETAIL_LOG_FILE}"
    fi

    return $exit_code
}

run_benchmark
