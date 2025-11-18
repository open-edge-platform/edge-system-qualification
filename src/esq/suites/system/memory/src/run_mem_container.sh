#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

OUTPUT_DIR="$1"

if [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <output-dir>"
    exit 1
fi

# shellcheck disable=SC1091
source "${OUTPUT_DIR}"/../common.sh

CONTAINER_NAME=stream_memory_benchmark
CONTAINER_FULL_NAME=stream_memory_benchmark
CONTAINER_TAG=1.0
CONTAINER_HOME=/home/dlstreamer

TC_NAME=Memory_Benchmark_Test
RESULT_FILE=${OUTPUT_DIR}/memory_benchmark_runner.result
CSV_FILE=${OUTPUT_DIR}/memory_benchmark_runner.csv

OUTPUT_MOUNT=(--volume "${OUTPUT_DIR}:${CONTAINER_HOME}/output")

init_csv_file()
{
  FORCE=${1:-0}
  if [ "$FORCE" -ne 0 ]; then
    rm -f "${CSV_FILE}"
    touch "${CSV_FILE}"
  fi
  if [ ! -f "${CSV_FILE}" ]; then
     echo "Memory_BM_Runner,Function,Best Rate,Avg Time,Min Time,Max Time,Result" >> "${CSV_FILE}"
     echo "${TC_NAME},NA,NA" >> "${CSV_FILE}"
  fi

}

format_bcmk_results()
{
  result_data=$1
  TFlag=$2
  IFS=$'\n' read -r -d '' -a lines <<< "$result_data"
  header="Memory_BM_Runner",$(echo "${lines[0]}" | sed 's/  \{1,\}/,/g'|cut -d ',' -f 1-5)",Result"
  echo "$header" >> "${CSV_FILE}"
  # Convert data into CSV format
  for (( i = 1; i < ${#lines[@]}; i++ )); do
    # Extract function name
    function_name=$(echo "${lines[i]}" | awk '{gsub(":",","); print $1}')
    # Replace spaces with commas and semicolons
    line_csv=$(echo "${lines[i]}" | awk '{$1=""; sub("^ *",""); gsub(/ +/, ","); print $0}')
    echo "${TC_NAME},""${function_name}${line_csv}"",${TFlag}" >> "${CSV_FILE}"
  done

}

update_csv()
{
  test_result=$1
  if [ "$test_result" -eq 0 ]; then
      TFlag="No Error"
      detail_rst=$(grep -E -A4 "^Function" "${RESULT_FILE}")
  else
      TFlag="FAIL"
      detail_rst=$(grep -i -E "error|fail" "${RESULT_FILE}")
  fi
  #since only one result in csv file, just re create csv file.
  rm -f "${CSV_FILE}"
  format_bcmk_results "${detail_rst}" "${TFlag}"
}

run_mem_bcmk()
{
  mkdir -p "${OUTPUT_DIR}"
  docker run --rm \
        "${OUTPUT_MOUNT[@]}" \
        --name $CONTAINER_NAME \
        ${CONTAINER_FULL_NAME}:${CONTAINER_TAG}
  retCode=$?
  update_csv $retCode
}

init_csv_file 1

run_mem_bcmk
