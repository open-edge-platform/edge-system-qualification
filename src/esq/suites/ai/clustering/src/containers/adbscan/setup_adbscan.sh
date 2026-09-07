#!/bin/bash

eatmydata apt-get update
eatmydata apt-get install -y "ros-${ROS_DISTRO}-benchmark-framework" "ros-${ROS_DISTRO}-adbscan-ros2"
