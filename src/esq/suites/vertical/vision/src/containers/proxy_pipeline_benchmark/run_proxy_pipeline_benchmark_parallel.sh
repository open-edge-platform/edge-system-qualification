#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Script metadata
baseFileName=$(basename "$0")
time='date +"%Y-%m-%d %H:%M:%S"'

# Terminal colors
_yellow='\033[1;33m'
_reset='\033[0m'

opt_d_dGPU=("$@")
MONITOR_NUM=${opt_d_dGPU[-5]}
IS_MTL=${opt_d_dGPU[-4]}
HAS_IGPU=${opt_d_dGPU[-3]}
CFG_FILE=${opt_d_dGPU[-2]}
PL_NAME=${opt_d_dGPU[-1]}

# Quote array indices to prevent glob expansion
unset 'opt_d_dGPU[-1]'
unset 'opt_d_dGPU[-1]'
unset 'opt_d_dGPU[-1]'
unset 'opt_d_dGPU[-1]'
unset 'opt_d_dGPU[-1]'

case $PL_NAME in 
  "smart_nvr_benchmark")
    TC_NAME="Smart NVR Proxy Pipeline"
    FILE_PREFIX="smart_nvr_proxy_pipeline"
    ;;
  "headed_visual_ai_benchmark")
    TC_NAME="Headed Visual AI Proxy Pipeline"
    FILE_PREFIX="headed_visual_ai_proxy_pipeline"
    ;;
  "ai_vsaas_benchmark")
    TC_NAME="AI VSaaS Proxy Pipeline"
    FILE_PREFIX="ai_vsaas_proxy_pipeline"
    ;;
  "lpr_benchmark")
    TC_NAME="LPR Proxy Pipeline"
    FILE_PREFIX="lpr_proxy_pipeline"
    ;;
esac

OUTPUT_DIR=/home/dlstreamer/output

if [ "${CFG_FILE}" != "none" ]; then
  DETAIL_LOG_FILE=${OUTPUT_DIR}/${FILE_PREFIX}_runner_with_config.log
  CSV_FILE=${OUTPUT_DIR}/${FILE_PREFIX}_with_config.csv
else
  DETAIL_LOG_FILE=${OUTPUT_DIR}/${FILE_PREFIX}_runner.log
fi

CONTAINER_NAME=proxy_pl_bm_runner
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
    echo -e "${prefix}${_yellow}${body}${_reset}" 2>&1 | tee -a "$detailsLogFile"
    echo "${prefix}${splitStr}" | tee -a "$detailsLogFile"
}

parallel_run() {
    # Changed to sequential execution to avoid double wait() race condition
    # Run each device one at a time instead of parallel with &
    device_cnt=0
    for _dev in "${opt_d_dGPU[@]}"; do
        if [ "${CFG_FILE}" != "none" ]; then
            format_log_section "Running ${TC_NAME} with config: [config file: ${CFG_FILE}, device: ${_dev}]." "" "${FUNCNAME[0]}" "$LINENO" "${DETAIL_LOG_FILE}"
        else
            format_log_section "Running ${TC_NAME} with params: [container: ${CONTAINER_NAME}:${CONTAINER_TAG}; device:${_dev}]." "" "${FUNCNAME[0]}" "$LINENO" "${DETAIL_LOG_FILE}"
        fi

        # Run synchronously (no & background operator)
        python3 "${PL_NAME}.py" --device "${_dev}" --monitor_num "${MONITOR_NUM}" --is_mtl "${IS_MTL}" --has_igpu "${HAS_IGPU}" --config_file "${CFG_FILE}"

        # Check exit status immediately
        if [ "${CFG_FILE}" != "none" ]; then
            format_log_section "${TC_NAME} execution with config: [config file: ${CFG_FILE}, device: ${_dev}] is finished" "" "${FUNCNAME[0]}" "$LINENO" "${DETAIL_LOG_FILE}"
            format_log_section "Check result in ${CSV_FILE}" "" "${FUNCNAME[0]}" "$LINENO" "${DETAIL_LOG_FILE}"
        else
            format_log_section "${TC_NAME} execution [device:${_dev}] is finished" "" "${FUNCNAME[0]}" "$LINENO" "${DETAIL_LOG_FILE}"
        fi
        device_cnt=$((device_cnt+1))
    done
}

parallel_run
