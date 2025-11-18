#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

OUTPUT_DIR=/home/dlstreamer/output
STREAM_DIR=/home/dlstreamer/STREAM
RESULT_FILE=${OUTPUT_DIR}/memory_benchmark_runner.result

run_mem_bcmk()
{
    if [ ! -f ${STREAM_DIR}/stream ]; then
        echo "Not found STREAM binary for the test."
        exit 255
    fi
    ${STREAM_DIR}/stream  > ${RESULT_FILE} 

}

run_mem_bcmk
