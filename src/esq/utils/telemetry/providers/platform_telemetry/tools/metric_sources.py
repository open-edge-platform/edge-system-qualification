# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Supplemental GPU memory and temperature sources for external telemetry."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from esq.utils.telemetry.providers.platform_telemetry.tools.xpu_smi_utils import XpuSmiUtils

METRIC_TEMPERATURE_C = "temperature_c"
METRIC_MEMORY_UTILIZATION = "memory_utilization"


class MetricReading(NamedTuple):
    value: float
    origin: str


class XpuSmiBackfillSource:
    """PCI-addressed GPU memory-utilization source backed by xpu-smi."""

    def __init__(self, utils: XpuSmiUtils | None = None) -> None:
        self._utils = utils or XpuSmiUtils()
        self._memory_util: dict[str, float] = {}

    def refresh(self) -> None:
        self._memory_util = self._utils.memory_util_map()

    def read(self, pci: str, metric: str) -> MetricReading | None:
        if metric != METRIC_MEMORY_UTILIZATION:
            return None
        value = self._memory_util.get(pci.lower())
        return MetricReading(value, "xpu_smi") if value is not None else None


class SysfsBackfillSource:
    """Read GPU temperature and memory utilization from DRM sysfs."""

    @staticmethod
    def temperature_c(card: Path) -> MetricReading | None:
        hwmon_base = card / "device" / "hwmon"
        for temp_file in sorted(hwmon_base.glob("hwmon*/temp*_input")):
            try:
                temperature = float(temp_file.read_text(encoding="utf-8").strip()) / 1000.0
            except (OSError, ValueError):
                continue
            if temperature >= 0:
                return MetricReading(temperature, "sysfs")
        return None

    @staticmethod
    def memory_utilization(card: Path) -> MetricReading | None:
        candidates = (
            ("mem_info_vram_used", "mem_info_vram_total"),
            ("mem_info_gtt_used", "mem_info_gtt_total"),
            ("mem_info_system_used", "mem_info_system_total"),
        )
        for used_name, total_name in candidates:
            try:
                used = float((card / "device" / used_name).read_text(encoding="utf-8").strip())
                total = float((card / "device" / total_name).read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            if total > 0:
                value = max(0.0, min(100.0, used / total * 100.0))
                return MetricReading(round(value, 2), "sysfs")
        return None

    @staticmethod
    def card_pci_address(card: Path) -> str:
        try:
            return (card / "device").resolve(strict=True).name.lower()
        except OSError:
            return ""
