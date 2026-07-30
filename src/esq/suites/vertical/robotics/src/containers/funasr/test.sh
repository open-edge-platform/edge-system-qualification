#!/bin/bash

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on error

_DEVICE=${DEVICE:-"CPU"}
# Which FunASR sub-model to benchmark:
#   eb -> contextual embedder (model_eb.xml), portable across CPU/GPU/NPU.
#   bb -> backbone/decoder   (model_bb.xml), the representative ASR compute
#         (CPU only in the current conversion; needs speech_lengths fed via -i).
_MODEL=${MODEL:-"eb"}
_FEATS=${FEATS_LENGTH:-30}
_NUM_HOTWORDS=${NUM_HOTWORDS:-2}
_NITER=${NITER:-30}

common_args=(
  -d "$_DEVICE"
  -hint latency
  -report_type average_counters
  -json_stats
  -report_folder "$HOME/output"
  -niter "$_NITER"
)

if [ "$_MODEL" = "bb" ]; then
  benchmark_app \
    -m /resources/model_bb.xml \
    "${common_args[@]}" \
    -shape "speech[1,${_FEATS},560],speech_lengths[1],bias_embed[1,${_NUM_HOTWORDS},512]" \
    -i "speech_lengths:/resources/speech_lengths.npy"
else
  benchmark_app \
    -m /resources/model_eb.xml \
    "${common_args[@]}" \
    -shape "[6,10]"
fi
