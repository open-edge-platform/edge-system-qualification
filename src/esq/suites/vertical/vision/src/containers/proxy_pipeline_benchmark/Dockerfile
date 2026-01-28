# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG COMMON_BASE_IMAGE

FROM ${COMMON_BASE_IMAGE}

ARG ENABLE_PRC_NETWORK
ARG IS_XEON_PLATFORM

USER root

RUN set -e -o pipefail; \
    if [ "$IS_XEON_PLATFORM" = "true" ] ; then \
        apt-get update && apt-get install --no-install-recommends sudo numactl libgsl27 libasound2-plugins libasound2-data libxmlrpc-core-c3 -y; \
        echo "dlstreamer ALL=(ALL) NOPASSWD: /usr/bin/ln,/usr/local/bin/ln" | \
        tee /etc/sudoers.d/user-conf && \
        chmod 0440 /etc/sudoers.d/user-conf ; \
    else \
        apt-get update -y && apt-get install --no-install-recommends -y sudo intel-gpu-tools xpu-smi numactl libgsl27 libasound2-plugins libasound2-data libxmlrpc-core-c3; \
        set -e -o pipefail && \
            echo "dlstreamer ALL=(ALL) NOPASSWD: /usr/bin/intel_gpu_top,/usr/bin/rm,/usr/bin/xpu-smi,/usr/local/bin/intel_gpu_top,/usr/local/bin/xpu-smi" | \
            tee /etc/sudoers.d/user-conf && \
            chmod 0440 /etc/sudoers.d/user-conf ; \
    fi  && \
    rm -rf /var/lib/apt/lists/*

# Install OpenVINO inside the container
RUN set -e -o pipefail && \
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

RUN mkdir -p /home/dlstreamer/output && chown -R dlstreamer:dlstreamer /home/dlstreamer/output && chmod -R 770 /home/dlstreamer/output
RUN mkdir -p $HOME/sample_video/ && chown -R dlstreamer:dlstreamer $HOME/sample_video/ && chmod -R 770 $HOME/sample_video/
RUN mkdir -p /home/dlstreamer/sink && chown -R dlstreamer:dlstreamer /home/dlstreamer/sink && chmod -R 770 /home/dlstreamer/sink
RUN mkdir -p $HOME/share/models/ && chown -R dlstreamer:dlstreamer $HOME/share/models/ && chmod -R 770 $HOME/share/models/

# Copy ESQ media utilities from extended build context (esq/ directory)
# Build context is set to esq/ by test_proxy_pipelines.py, so we reference utils/ directly
# These provide secure subprocess wrappers and shared benchmarking/telemetry functionality
COPY --chown=dlstreamer:dlstreamer utils/media/__init__.py ./esq_utils/media/
COPY --chown=dlstreamer:dlstreamer utils/media/pipeline_utils.py ./esq_utils/media/
COPY --chown=dlstreamer:dlstreamer utils/media/telemetry.py ./esq_utils/media/
COPY --chown=dlstreamer:dlstreamer utils/media/validation.py ./esq_utils/media/
COPY --chown=dlstreamer:dlstreamer utils/media/container_utils.py ./esq_utils/media/
COPY --chown=dlstreamer:dlstreamer utils/media/adaptive_search.py ./esq_utils/media/

# Copy benchmark scripts from container directory
COPY --chown=dlstreamer:dlstreamer suites/vertical/vision/src/containers/proxy_pipeline_benchmark/gst_loop_mp4.sh .
COPY --chown=dlstreamer:dlstreamer suites/vertical/vision/src/containers/proxy_pipeline_benchmark/smart_nvr_benchmark.py .
COPY --chown=dlstreamer:dlstreamer suites/vertical/vision/src/containers/proxy_pipeline_benchmark/headed_visual_ai_benchmark.py .
COPY --chown=dlstreamer:dlstreamer suites/vertical/vision/src/containers/proxy_pipeline_benchmark/ai_vsaas_benchmark.py .
COPY --chown=dlstreamer:dlstreamer suites/vertical/vision/src/containers/proxy_pipeline_benchmark/base_plbenchmark.py .
COPY --chown=dlstreamer:dlstreamer suites/vertical/vision/src/containers/proxy_pipeline_benchmark/run_proxy_pipeline_benchmark_parallel.sh .
COPY --chown=dlstreamer:dlstreamer suites/vertical/vision/src/containers/proxy_pipeline_benchmark/gst_lpr_loop_mp4.sh .
COPY --chown=dlstreamer:dlstreamer suites/vertical/vision/src/containers/proxy_pipeline_benchmark/lpr_benchmark.py .
COPY --chown=dlstreamer:dlstreamer suites/vertical/vision/src/containers/proxy_pipeline_benchmark/run_pipeline.sh .
COPY --chown=dlstreamer:dlstreamer suites/vertical/vision/src/containers/proxy_pipeline_benchmark/check_va_plugins.sh .

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/compiler/2024.2/lib

HEALTHCHECK CMD [ "cat", "/dev/null" ]
