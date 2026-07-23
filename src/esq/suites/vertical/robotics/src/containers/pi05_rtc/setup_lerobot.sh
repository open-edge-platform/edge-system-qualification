#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit on error

git clone --branch release-2026.1.0 --depth 1 --filter=blob:none --sparse https://github.com/open-edge-platform/edge-ai-suites.git
cd edge-ai-suites
git sparse-checkout set robotics-ai-suite
cd robotics-ai-suite/pipelines/pi05-rtc-ov
git submodule update --init lerobot
cd lerobot
git config user.name "Anonymous"
git config user.email "none@example.com"
git am ../patches/*.patch
curl -LsSf https://astral.sh/uv/install.sh | sh
# shellcheck disable=SC1091 # Script path not available at static analysis time
source "$HOME"/.local/bin/env
uv sync --extra pi-ov
cd examples/pi05_with_openvino
uv run --extra pi-ov --with nncf scripts/convert_ov_rtc.py --ov_output_dir pi05_rtc_lerobot_ov_ir --compress_int8 --override
mkdir -p /resources
cp pi05_rtc_lerobot_ov_ir_4c_INT8/model* /resources
