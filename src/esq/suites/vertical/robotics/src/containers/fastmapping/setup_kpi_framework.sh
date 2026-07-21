#!/bin/bash

export DEBIAN_FRONTEND=noninteractive

# Add AMR APT repo
curl -sSf https://eci.intel.com/repos/gpg-keys/GPG-PUB-KEY-INTEL-ECI.gpg -o /usr/share/keyrings/eci-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/eci-archive-keyring.gpg] https://amrdocs.intel.com/repos/$(lsb_release -cs) amr main" | tee /etc/apt/sources.list.d/amr.list > /dev/null

# Add oneAPI APT repo
curl -sSf https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list

# Pin oneAPI release to 2025.3
{
    echo -e "Package: intel-oneapi-runtime-*\nPin: version 2025.3.*\nPin-Priority: 1001\n";
    echo -e "Package: intel-oneapi-compiler-*\nPin: version 2025.3.*\nPin-Priority: 1001\n";
    echo -e "Package: intel-oneapi-mkl-*\nPin: version 2025.3.*\nPin-Priority: 1001\n";
} > /etc/apt/preferences.d/oneapi

# Add RealSense APT repo
mkdir -p /root/.gnupg
curl -sSf https://librealsense.realsenseai.com/Debian/librealsenseai.asc | gpg --dearmor | tee /etc/apt/keyrings/librealsenseai.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/librealsense.list

# Add OpenVINO APT repo
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/openvino ubuntu24 main" > /etc/apt/sources.list.d/intel-openvino.list

# Pin OpenVINO release to 2025.3.0
{
    echo -e "Package: openvino-libraries-dev\nPin: version 2025.3.0*\nPin-Priority: 1001\n";
    echo -e "Package: openvino\nPin: version 2025.3.0*\nPin-Priority: 1001\n";
    echo -e "Package: ros-jazzy-openvino-wrapper-lib\nPin: version 2025.3.0*\nPin-Priority: 1002\n";
    echo -e "Package: ros-jazzy-openvino-node\nPin: version 2025.3.0*\nPin-Priority: 1002";
} > /etc/apt/preferences.d/intel-openvino

# Add OSRFoundation APT repo
curl https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Retain proxy variables with sudo
echo "Defaults env_keep += \"http_proxy https_proxy\"" | tee -a /etc/sudoers

# shellcheck disable=SC1090,SC1091 # Script path not available at static analysis time
source ~/.bashrc
apt-get update
apt-get install -y eatmydata
eatmydata apt-get install --no-install-recommends -y \
  wget \
  libogre-1.12-dev \
  libgz-math7-dev \
  libgz-math7 \
  xvfb \
  jq \
  "ros-${ROS_DISTRO}-benchmark-framework"
eatmydata apt-get -q -y -o Dpkg::Options::="--force-confnew" -o Dpkg::Options::="--force-confdef" upgrade
eatmydata apt-get clean
rm -rf /var/lib/apt/lists/*

cd "/opt/ros/${ROS_DISTRO}/benchmarking" || exit
make install
PATH="$HOME/.local/bin:$PATH" uv sync
sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' .venv/pyvenv.cfg
chown -R appuser:appgroup "/opt/ros/${ROS_DISTRO}/benchmarking"
