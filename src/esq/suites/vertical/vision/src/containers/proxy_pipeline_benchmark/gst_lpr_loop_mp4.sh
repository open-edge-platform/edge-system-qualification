#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

LOOPS=${1:-6}
# VIDEO_RESOLUTION=${2:-1280p}  # Reserved for future use if needed
VIDEO_TYPE=${3:-h264}
# FPS=${4:-30}  # Reserved for future use if needed

VIDEO_IN="/home/dlstreamer/sample_video/lpr/ParkingVideo.mp4"
VIDEO_OUT="/home/dlstreamer/sample_video/lpr/ParkingVideo_1min.mp4"

echo "[DEBUG] Creating looped video from ${VIDEO_IN} to ${VIDEO_OUT}" >&2
echo "[DEBUG] Loops: ${LOOPS}" >&2

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

eval "${pipeline}"
gst-discoverer-1.0 "${VIDEO_OUT}"
