# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Catalog of reusable service specs for timeseries style deployments."""

from esq.utils.services.interfaces import ServiceSpec

TIMESERIES_COMMON_SERVICES = {
    "nginx": ServiceSpec(
        name="nginx",
        kind="ingress",
        compose_service_name="nginx",
        default_group="core",
    ),
    "mqtt-broker": ServiceSpec(
        name="mqtt-broker",
        kind="messaging",
        compose_service_name="mqtt-broker",
        default_group="core",
    ),
    "mqtt-publisher": ServiceSpec(
        name="mqtt-publisher",
        kind="telemetry-publisher",
        compose_service_name="mqtt-publisher",
        default_group="telemetry",
    ),
    "opc-ua-server": ServiceSpec(
        name="opc-ua-server",
        kind="ot-server",
        compose_service_name="opc-ua-server",
        default_group="core",
    ),
    "rest-api": ServiceSpec(
        name="rest-api",
        kind="api",
        compose_service_name="rest-api",
        default_group="core",
    ),
}
