# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Supplemental GPU memory-utilization and temperature collection."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from esq.utils.telemetry.providers.metrics_manager.device_inventory import (
    gpu_groups_by_pci_bdf,
)
from esq.utils.telemetry.providers.platform_telemetry.base import BaseCollector
from esq.utils.telemetry.providers.platform_telemetry.models import MetricSample
from esq.utils.telemetry.providers.platform_telemetry.tools.metric_sources import (
    METRIC_MEMORY_UTILIZATION,
    METRIC_TEMPERATURE_C,
    SysfsBackfillSource,
    XpuSmiBackfillSource,
)
from esq.utils.telemetry.providers.platform_telemetry.tools.xpu_smi_utils import XpuSmiUtils

logger = logging.getLogger(__name__)


class GpuCollector(BaseCollector):
    """Collect only GPU metrics absent from the Metrics Manager API."""

    name = "gpu_gaps"

    def __init__(self) -> None:
        self._running = False
        self._device_groups_by_bdf: dict[str, str] = {}
        self._xpu_source = XpuSmiBackfillSource(XpuSmiUtils())

    def set_device_inventory(self, devices: list[dict[str, Any]]) -> None:
        self._device_groups_by_bdf = gpu_groups_by_pci_bdf(devices)

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def collect_once(self) -> list[MetricSample]:
        if not self._running:
            return []

        self._xpu_source.refresh()
        drm_root = Path("/sys/class/drm")
        cards = sorted(
            card
            for card in drm_root.iterdir()
            if card.name.startswith("card") and card.name[4:].isdigit()
        ) if drm_root.exists() else []
        return self._collect_cards(cards)

    def _collect_cards(self, cards: list[Path]) -> list[MetricSample]:
        samples: list[MetricSample] = []
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for card in cards:
            pci = SysfsBackfillSource.card_pci_address(card)
            device_group = self._device_groups_by_bdf.get(pci)
            if not device_group:
                logger.warning(
                    "Skipping supplemental GPU metrics for %s: no matching Metrics Manager capability",
                    pci or card.name,
                )
                continue

            memory_reading = self._xpu_source.read(pci, METRIC_MEMORY_UTILIZATION)
            if memory_reading is None:
                memory_reading = SysfsBackfillSource.memory_utilization(card)
            temperature_reading = SysfsBackfillSource.temperature_c(card)

            for metric, reading, unit in (
                (METRIC_MEMORY_UTILIZATION, memory_reading, "%"),
                (METRIC_TEMPERATURE_C, temperature_reading, "°C"),
            ):
                if reading is None:
                    continue
                samples.append(
                    MetricSample(
                        timestamp_utc=timestamp,
                        collector=self.name,
                        device=device_group,
                        metric_name=f"gpu.{metric}",
                        value=reading.value,
                        unit=unit,
                        tags={
                            "vendor": "intel",
                            "metric_origin": reading.origin,
                            "card": card.name,
                            "pci": pci,
                        },
                    )
                )
        return samples
