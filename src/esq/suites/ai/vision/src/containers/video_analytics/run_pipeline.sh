#!/bin/bash
#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Video Analytics Pipeline Runner
#
# This script runs a GStreamer pipeline for the VA benchmark.
# It handles environment setup and pipeline execution with proper
# process management for telemetry collection.

device=$1
gst_cmd=$2

gst_pid=""

# Cleanup function to ensure child processes are terminated
# This prevents orphaned gst-launch-1.0 processes when bash is killed
cleanup() {
    if [ -n "$gst_pid" ] && kill -0 "$gst_pid" 2>/dev/null; then
        # First try graceful termination (SIGTERM)
        kill -TERM "$gst_pid" 2>/dev/null
        # Wait briefly for graceful shutdown
        sleep 0.5
        # If still running, force kill (SIGKILL)
        if kill -0 "$gst_pid" 2>/dev/null; then
            kill -KILL "$gst_pid" 2>/dev/null
        fi
    fi
    rm -f /tmp/gst_pid_"${device}".txt
}

# Trap signals to ensure cleanup happens when this script is terminated
# SIGTERM (15) - normal termination request from Python process.terminate()
# SIGINT (2) - Ctrl+C
# SIGHUP (1) - terminal disconnect
# EXIT - script exit (including normal completion)
trap cleanup SIGTERM SIGINT SIGHUP EXIT

# Source OpenVINO environment (use symlink for version independence)
if [ -f /opt/intel/openvino/setupvars.sh ]; then
    # shellcheck disable=SC1091 # OpenVINO path not available at static analysis time
    source /opt/intel/openvino/setupvars.sh
fi

# Note: DLStreamer environment is pre-configured in the base container image
# GST_PLUGIN_PATH is already set during container build
# oneAPI is NOT required for DLStreamer container - it has its own OpenVINO/GPU runtime

if [ "$device" != "CPU" ]; then
    # Source oneAPI if available (optional, not required for DLStreamer)
    if [ -f /opt/intel/oneapi/setvars.sh ]; then
        # shellcheck disable=SC1091 # OneAPI path not available at static analysis time
        source /opt/intel/oneapi/setvars.sh
    fi
fi

# SECURITY NOTE: SC2086 is intentionally disabled here
# The gst_cmd variable contains a GStreamer pipeline command that MUST be word-split
# for gst-launch-1.0 to parse correctly. The command originates from controlled
# Python source code (not user input), making this safe from injection attacks.
# Example: "filesrc location=file.mp4 ! decoder ! sink" must become separate args
# DO NOT quote ${gst_cmd} as it will break GStreamer pipeline parsing.
# shellcheck disable=SC2086 # GStreamer pipeline requires word-splitting from controlled source
GST_DEBUG=3 gst-launch-1.0 -e -v ${gst_cmd} &

gst_pid=$!

echo $gst_pid > /tmp/gst_pid_"${device}".txt
wait $gst_pid
# Cleanup is handled by trap on EXIT
