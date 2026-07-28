ARG COMMON_BASE_IMAGE=amd64/ros:jazzy-ros-base

FROM ${COMMON_BASE_IMAGE} AS build

ENV DEBIAN_FRONTEND=noninteractive

# Create a dedicated system group and unprivileged user
RUN groupadd -r appgroup && useradd -r -g appgroup -m -d /home/appuser appuser

COPY setup_kpi_framework.sh /
RUN /setup_kpi_framework.sh

COPY setup_adbscan.sh /
RUN /setup_adbscan.sh

USER appuser
ENV HOME=/home/appuser
WORKDIR /home/appuser

COPY --chown=appuser:appgroup test_adbscan.sh .
RUN chmod a+rx test_adbscan.sh
HEALTHCHECK CMD [ "cat", "/dev/null" ]
ENTRYPOINT [ "./test_adbscan.sh" ]
