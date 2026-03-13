# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG COMMON_BASE_IMAGE

FROM ${COMMON_BASE_IMAGE}

ARG IS_XEON_PLATFORM

USER root

RUN set -e -o pipefail; \
    if [ "$IS_XEON_PLATFORM" = "true" ] ; then \
        apt-get update && apt-get install --no-install-recommends numactl libgsl27 libasound2-plugins libasound2-data libxmlrpc-core-c3 -y; \
    else \
        apt-get update -y && apt-get install --no-install-recommends -y intel-gpu-tools xpu-smi numactl libgsl27 libasound2-plugins libasound2-data libxmlrpc-core-c3 ; \
    fi  && \
    rm -rf /var/lib/apt/lists/*

# Install OpenVINO inside the container
RUN set -e -o pipefail; \
   if [ -f /home/dlstreamer/dlstreamer/scripts/install_dependencies/install_openvino.sh ]; then \
   	source /home/dlstreamer/dlstreamer/scripts/install_dependencies/install_openvino.sh && \
   	ov_dir=$(find /opt/intel -maxdepth 1 -type d -name "openvino_*" | sort -r | head -n 1) && \
   	if [ -n "$ov_dir" ]; then \
   		ln -sf "$ov_dir" /opt/intel/openvino; \
       		echo "Linked $ov_dir -> /opt/intel/openvino"; \
   	else \
       		echo "OpenVINO folder not found after install"; \
   	fi; \
   else \
	echo "install_openvino.sh not found in image"; \
   fi && \
   apt-get clean && rm -rf /var/lib/apt/lists/*

USER dlstreamer
ENV HOME=/home/dlstreamer
WORKDIR /home/dlstreamer

# Copy ESQ media utilities from extended build context (esq/ directory)
# Build context is set to esq/ by test_media_pipelines.py, so we reference utils/ directly
# These provide secure subprocess wrappers and shared benchmarking/telemetry functionality
COPY --chown=dlstreamer:dlstreamer utils/media/__init__.py ./esq_utils/media/
COPY --chown=dlstreamer:dlstreamer utils/media/pipeline_utils.py ./esq_utils/media/
COPY --chown=dlstreamer:dlstreamer utils/media/telemetry.py ./esq_utils/media/
COPY --chown=dlstreamer:dlstreamer utils/media/pmu_gpu_collector.py ./esq_utils/media/
COPY --chown=dlstreamer:dlstreamer utils/media/validation.py ./esq_utils/media/
COPY --chown=dlstreamer:dlstreamer utils/media/container_utils.py ./esq_utils/media/
COPY --chown=dlstreamer:dlstreamer utils/media/memory_utils.py ./esq_utils/media/
COPY --chown=dlstreamer:dlstreamer utils/media/adaptive_search.py ./esq_utils/media/

# Copy benchmark scripts from container directory
COPY --chown=dlstreamer:dlstreamer suites/media/performance/src/containers/media_benchmark/gst_loop_mp4.sh .
COPY --chown=dlstreamer:dlstreamer suites/media/performance/src/containers/media_benchmark/media_benchmark.py .
COPY --chown=dlstreamer:dlstreamer suites/media/performance/src/containers/media_benchmark/run_media_benchmark_parallel.sh .
COPY --chown=dlstreamer:dlstreamer suites/media/performance/src/containers/media_benchmark/run_pipeline.sh .
COPY --chown=dlstreamer:dlstreamer suites/media/performance/src/containers/media_benchmark/check_va_plugins.sh .

RUN mkdir -p /home/dlstreamer/sink && chown -R dlstreamer:dlstreamer /home/dlstreamer/sink && chmod -R 770 /home/dlstreamer/sink
RUN mkdir -p /home/dlstreamer/output && chown -R dlstreamer:dlstreamer /home/dlstreamer/output && chmod -R 770 /home/dlstreamer/output

HEALTHCHECK CMD [ "cat", "/dev/null" ]
