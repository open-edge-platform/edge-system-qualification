#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

LOOPS=${1:-18}
VIDEO_RESOLUTION=${2:-1080p}
VIDEO_TYPE=${3:-h264}
FPS=${4:-30}

VIDEO_IN="/home/dlstreamer/sample_video/car_${VIDEO_RESOLUTION}${FPS}_10s_${VIDEO_TYPE}.mp4"
VIDEO_OUT="/home/dlstreamer/sample_video/car_${VIDEO_RESOLUTION}${FPS}_180s_${VIDEO_TYPE}.mp4"

rm -f "${VIDEO_OUT}"

if [[ ${VIDEO_TYPE} =~ "h265" ]]; then
    pipeline="gst-launch-1.0 -e concat name=c ! h265parse ! mp4mux ! filesink location=${VIDEO_OUT}"
else
    pipeline="gst-launch-1.0 -e concat name=c ! h264parse ! mp4mux ! filesink location=${VIDEO_OUT}"
fi

# Loop counter intentionally unused - just repeating the pipeline concatenation
for _ in $(seq 1 "${LOOPS}"); do
    pipeline+=" filesrc location=${VIDEO_IN} ! qtdemux ! queue ! c. "
done

# Source OpenVINO environment (use symlink for version independence)
if [ -f /opt/intel/openvino/setupvars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/openvino/setupvars.sh
fi

# Note: DLStreamer environment is pre-configured in the base container image
# (GST_PLUGIN_PATH, PYTHONPATH, etc. are already set)
# No need to source setup_env.sh which is for development builds

eval "${pipeline}"
gst-discoverer-1.0 "${VIDEO_OUT}"
