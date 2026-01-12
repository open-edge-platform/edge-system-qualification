#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

device=${1:-"CPU"}

if [ -f /opt/intel/openvino/setupvars.sh ]; then
    # shellcheck disable=SC1091 # OpenVINO path not available at static analysis time
    source /opt/intel/openvino/setupvars.sh
fi

# Source DLStreamer environment if available
if [ -f /home/dlstreamer/dlstreamer/scripts/setup_env.sh ]; then
    # shellcheck disable=SC1091 # DLStreamer script path not available at static analysis time
    source /home/dlstreamer/dlstreamer/scripts/setup_env.sh
fi

gst-inspect-1.0 gvafpscounter

if [ "$device" != "CPU" ]; then
    # shellcheck disable=SC1091 # OneAPI path not available at static analysis time
    source /opt/intel/oneapi/setvars.sh
fi

gst-inspect-1.0 va
