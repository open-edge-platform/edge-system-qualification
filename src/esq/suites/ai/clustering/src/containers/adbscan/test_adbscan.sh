#!/bin/bash

# shellcheck disable=SC1090,SC1091 # Script path not available at static analysis time
source "/opt/ros/${ROS_DISTRO}/setup.bash"

# Prepare environment
cd "/opt/ros/${ROS_DISTRO}/benchmarking" || exit
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

NO_PROMPT=1 make adbscan-benchmark

# Parse results
throughput=$(find monitoring_sessions/ -name kpi.json -print0 | xargs -0 cat | jq .throughput_hz | awk '{ sum+=$1; count++ } END { print sum/count }')
mean_latency=$(find monitoring_sessions/ -name kpi.json -print0 | xargs -0 cat | jq .mean_latency_ms | awk '{ sum+=$1; count++ } END { print sum/count }')
max_jitter=$(find monitoring_sessions/ -name kpi.json -print0 | xargs -0 cat | jq .max_jitter_ms | awk '{ sum+=$1; count++ } END { print sum/count }')
min_jitter=$(find monitoring_sessions/ -name kpi.json -print0 | xargs -0 cat | jq .min_jitter_ms | awk '{ sum+=$1; count++ } END { print sum/count }')
mean_jitter=$(find monitoring_sessions/ -name kpi.json -print0 | xargs -0 cat | jq .mean_jitter_ms | awk '{ sum+=$1; count++ } END { print sum/count }')
jitter_stdev=$(find monitoring_sessions/ -name kpi.json -print0 | xargs -0 cat | jq .jitter_stdev_ms | awk '{ sum+=$1; count++ } END { print sum/count }')
iterations=$(find monitoring_sessions/ -name kpi.json | wc -l)

mkdir -p "${HOME}/output"

# Write report
echo "{\"execution_results\": {\"throughput\": \"${throughput}\", \"mean_latency\": \"${mean_latency}\", \"max_jitter\": \"${max_jitter}\", \"min_jitter\": \"${min_jitter}\", \"mean_jitter\": \"${mean_jitter}\", \"jitter_stdev\": \"${jitter_stdev}\", \"iterations\": \"${iterations}\"}}" > "${HOME}/output/benchmark_report.json"

