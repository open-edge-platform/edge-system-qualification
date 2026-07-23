#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on error

_DEVICE=${DEVICE:-"CPU"}

benchmark_app \
  -m /resources/model.xml \
  -d "$_DEVICE" \
  -hint latency \
  -report_type average_counters \
  -json_stats \
  -report_folder "$HOME/output"