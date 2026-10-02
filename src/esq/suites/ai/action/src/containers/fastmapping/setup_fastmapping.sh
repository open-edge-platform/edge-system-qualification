#!/bin/bash

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

eatmydata apt-get update
eatmydata apt-get install -y "ros-${ROS_DISTRO}-rtabmap-ros" "ros-${ROS_DISTRO}-fast-mapping"
