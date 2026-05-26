# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""BaseCollector contract used by all platform_telemetry in-process collectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .models import MetricSample


class BaseCollector(ABC):
    name: str = "base"

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def collect_once(self) -> List[MetricSample]:
        pass
