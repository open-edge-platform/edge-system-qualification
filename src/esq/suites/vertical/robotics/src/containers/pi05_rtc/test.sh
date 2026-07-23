#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on error

_DEVICE=${DEVICE:-"CPU"}

# Recommended configuration (FP16, 3 cameras, 64 tokens, 10 denoising steps):
benchmark_app \
  -m /resources/model.xml \
  -d "$_DEVICE" \
  -hint latency \
  -report_type average_counters \
  -json_stats \
  -report_folder "$HOME/output" \
  -niter 30 \
  -shape "img_0[1,3,224,224],img_1[1,3,224,224],img_2[1,3,224,224],lang_tokens[1,64],lang_masks[1,64],state[1,64],noise[1,10,32]"
