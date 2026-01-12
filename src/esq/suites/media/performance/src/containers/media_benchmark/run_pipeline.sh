#!/bin/bash
#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

device=$1
gst_cmd=$2

# Source OpenVINO environment (use symlink for version independence)
if [ -f /opt/intel/openvino/setupvars.sh ]; then
    # shellcheck disable=SC1091 # OpenVINO path not available at static analysis time
    source /opt/intel/openvino/setupvars.sh
fi

# Note: DLStreamer environment is pre-configured in the base container image
# GST_PLUGIN_PATH is already set during container build

if [ "$device" != "CPU" ]; then
    # shellcheck disable=SC1091 # OneAPI path not available at static analysis time
    source /opt/intel/oneapi/setvars.sh
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
rm /tmp/gst_pid_"${device}".txt
