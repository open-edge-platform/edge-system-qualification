#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on error

echo "========================================"
echo "Building STREAM Benchmark"
echo "========================================"

# Function to check the L3 cache size, then return the STREAM_ARRAY_SIZE size which more than 4 times of cache size.
get_stream_array_size() {
  default_val=120000000
  l3_cache_result=()
  IFS=" " read -r -a l3_cache_result <<< "$(getconf LEVEL3_CACHE_SIZE)"
  [ ${#l3_cache_result[@]} -eq 0 ] && return ${default_val}
  l3_cache_byte=${l3_cache_result[-1]}
  if [ "${l3_cache_byte}" -ne -1 ];then
    echo $((5 * l3_cache_byte)) # so we use 5 times of cache size
  else
    echo ${default_val}
  fi
}

STREAM_GIT_URL=${1:-"https://github.com/jeffhammond/STREAM.git"}
STREAM_ARRAY_SIZE=120000000

echo "[INFO] STREAM Git URL: ${STREAM_GIT_URL}"
echo "[INFO] STREAM Array Size: ${STREAM_ARRAY_SIZE}"
echo "[INFO] Removing old STREAM directory..."
rm -rf STREAM

echo "[INFO] Cloning STREAM repository..."
git clone "${STREAM_GIT_URL}" STREAM

echo "[INFO] Entering STREAM directory..."
cd STREAM

echo "[INFO] Compiling STREAM benchmark..."
echo "[INFO] Compiler: gcc -O2 -fopenmp -DSTREAM_ARRAY_SIZE=${STREAM_ARRAY_SIZE} -o stream stream.c"
gcc -O2 -fopenmp -DSTREAM_ARRAY_SIZE=${STREAM_ARRAY_SIZE} -o stream stream.c

if [ -f stream ]; then
    echo "[INFO] STREAM binary compiled successfully"
    ls -lh stream
else
    echo "[ERROR] STREAM binary compilation failed"
    exit 1
fi

echo "========================================"
echo "[INFO] STREAM Build Complete"
echo "========================================"
