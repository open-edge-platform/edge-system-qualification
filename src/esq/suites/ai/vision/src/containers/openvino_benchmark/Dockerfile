# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG COMMON_BASE_IMAGE

FROM ${COMMON_BASE_IMAGE}

ARG YOLOV5_GIT_BRANCH
ARG YOLOV5_GIT_URL

ARG ENABLE_PRC_NETWORK
ARG PYPI_MIRROR_URL
ARG PYPI_TRUSTED_HOST
ARG NO_POC_PLATFORM
#ARG IS_XEON_PLATFORM

#install git
USER root

COPY install_pkg.sh /usr/local/bin/install_pkg.sh
RUN chmod +x /usr/local/bin/install_pkg.sh
RUN IS_XEON_PLATFORM=$IS_XEON_PLATFORM /usr/local/bin/install_pkg.sh

USER dlstreamer
ENV HOME=/home/dlstreamer
WORKDIR /home/dlstreamer

# set mirror url if user set
RUN if [ -n "${PYPI_MIRROR_URL}" ]; then \
        pip3 config set global.index-url ${PYPI_MIRROR_URL}; \
    fi && \
    if [ -n "${PYPI_TRUSTED_HOST}" ]; then \
        pip3 config set global.trusted-host ${PYPI_TRUSTED_HOST}; \
    fi

# Install Python packages from requirements.txt
COPY requirements.txt ./
RUN pip3 install --break-system-packages -r requirements.txt --no-cache-dir

# Entrypoint will handle setting up OpenVINO environment
COPY  --chown=dlstreamer:dlstreamer entrypoint.sh ./
RUN chmod +x entrypoint.sh

COPY --chown=dlstreamer:dlstreamer spr_bcmk.py run_benchmark.py ref_bcmk.py bcmk_ref.csv ./
COPY --chown=dlstreamer:dlstreamer bcmk_telemetry.py gpu_util.py ./

HEALTHCHECK CMD [ "cat", "/dev/null" ]
ENTRYPOINT ["./entrypoint.sh"]
