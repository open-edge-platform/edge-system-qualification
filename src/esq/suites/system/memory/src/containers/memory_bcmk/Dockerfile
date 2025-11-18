# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG COMMON_BASE_IMAGE

FROM ${COMMON_BASE_IMAGE}

ARG STREAM_GIT_URL

USER root
RUN apt-get update --no-install-recommends -y && apt-get install --no-install-recommends -y git gcc  &&  rm -rf /var/lib/apt/lists/*

USER dlstreamer
ENV HOME=/home/dlstreamer
WORKDIR /home/dlstreamer

COPY --chown=dlstreamer:dlstreamer build_stream.sh .
RUN chmod u+x build_stream.sh && bash build_stream.sh ${STREAM_GIT_URL}

COPY --chown=dlstreamer:dlstreamer mem_bcmk.sh .
RUN chmod u+x mem_bcmk.sh
HEALTHCHECK CMD [ "cat", "/dev/null" ]
ENTRYPOINT [ "./mem_bcmk.sh" ]
