#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

mkdir -p /resources
curl -fsSL https://huggingface.co/OpenVINO/pi05-libero-fp16-ov/resolve/main/pi05.bin?download=true -o /resources/model.bin
curl -fsSL https://huggingface.co/OpenVINO/pi05-libero-fp16-ov/resolve/main/pi05.xml?download=true -o /resources/model.xml
