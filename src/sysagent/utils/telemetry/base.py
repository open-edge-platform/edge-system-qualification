# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Base classes for the modular telemetry framework.

Defines the abstract interface that all telemetry modules must implement,
along with shared data structures for configuration, samples, and summaries.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TelemetryConfig:
    """
    Configuration for a single telemetry module instance.

    Attributes:
        enabled: Whether this module is active.
        interval: Sampling interval in seconds (shared across all modules by the collector).
        metrics: Metric names to collect. Empty list means collect all.
        thresholds: Per-metric warning thresholds: ``{metric_name: {"warning": value}}``.
        chart_type: Visualization hint for report renderers.
            Values: ``"line"`` (default), ``"area"``, ``"bar_vertical"``.
            ``None`` means use the module's ``get_default_config()`` default (falls back to ``"line"``).
        title: Chart title config. Keys: ``display`` (bool), ``text`` (str).
            When omitted, the renderer auto-generates a title from the module name.
        scales: Per-metric display hints. Key is metric name; value supports:
            ``display`` (bool), ``label`` (str), ``unit`` (str).
        axes: Dual y-axis config for modules with mixed units (e.g. ``%`` and ``GB``).
            Each entry defines one axis with keys: ``id`` (str), ``position``
            (``"left"``/``"right"``), ``metrics`` (list of metric names), ``label`` (str).
            When empty, all metrics share a single y-axis.
    """

    enabled: bool = True
    interval: int = 10
    metrics: List[str] = field(default_factory=list)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    chart_type: Optional[str] = None
    title: Dict[str, Any] = field(default_factory=dict)
    scales: Dict[str, Any] = field(default_factory=dict)
    axes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TelemetrySample:
    """
    A single telemetry sample snapshot.

    Attributes:
        timestamp: Unix timestamp when the sample was collected.
        values: Dict of metric_name -> value pairs.
    """

    timestamp: float = field(default_factory=time.time)
    values: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp, "values": self.values}


class BaseTelemetryModule(ABC):
    """
    Abstract base class for all telemetry modules.

    Each module is responsible for collecting one category of metrics
    (e.g., CPU frequency, memory usage, package power).

    Module authors must implement:
    - ``module_name`` class attribute (unique string identifier)
    - ``collect_sample()`` method

    Module authors may override:
    - ``is_available()`` to check if dependencies exist
    - ``compute_averages()`` for custom aggregation logic
    """

    #: Unique identifier used in profile YAML ``modules[].name``
    module_name: str = "base"

    def __init__(self, config: TelemetryConfig) -> None:
        self.config = config
        self._samples: List[TelemetrySample] = []

    def get_default_config(self) -> Dict[str, Any]:
        """
        Return default chart configuration for this module.

        Override in subclasses to provide module-specific defaults for
        ``title``, ``scales``, ``thresholds``, ``chart_type``, and ``axes``.
        The collector merges these defaults with any profile YAML overrides;
        profile values always take precedence.

        For GPU modules, this method may call hardware discovery functions
        to generate per-device scales and thresholds dynamically.

        Returns:
            Dict with any subset of keys: ``chart_type`` (str),
            ``title`` (dict), ``scales`` (dict), ``thresholds`` (dict),
            ``axes`` (list).  Return an empty dict for no defaults.
        """
        return {}

    def _apply_default_config(self) -> None:
        """
        Merge ``get_default_config()`` into ``self.config``.

        Called by the collector immediately after module instantiation.
        Profile YAML values always win; module defaults only fill in
        values that the profile left unset (empty dict / empty list / None).
        """
        defaults = self.get_default_config()
        if not defaults:
            # No defaults defined → only ensure chart_type has a string value.
            if self.config.chart_type is None:
                self.config.chart_type = "line"
            return

        # chart_type: apply module default when profile left it unset (None).
        if self.config.chart_type is None:
            self.config.chart_type = defaults.get("chart_type", "line")

        # title: module default as base, profile keys override per-key.
        if "title" in defaults:
            self.config.title = {**defaults["title"], **self.config.title}

        # scales: module defaults as base, profile keys override per-key.
        if "scales" in defaults:
            self.config.scales = {**defaults["scales"], **self.config.scales}

        # thresholds: module defaults as base, profile keys override per-key.
        if "thresholds" in defaults:
            self.config.thresholds = {**defaults["thresholds"], **self.config.thresholds}

        # axes: use module default only when profile provided none.
        if "axes" in defaults and not self.config.axes:
            self.config.axes = list(defaults["axes"])

        # Ensure chart_type always has a string value after merging.
        if self.config.chart_type is None:
            self.config.chart_type = "line"

    def is_available(self) -> bool:
        """Return True if this module can collect data on the current system."""
        return True

    @abstractmethod
    def collect_sample(self) -> TelemetrySample:
        """
        Collect a single sample of metrics.

        Returns:
            TelemetrySample with current metric values.
            Implementations should return an empty sample (no values) rather
            than raising exceptions on transient read failures.
        """
        ...

    def _should_collect_metric(self, metric_name: str) -> bool:
        """Return True if this metric should be collected based on config."""
        if not self.config.metrics:
            return True
        return metric_name in self.config.metrics

    def _filter_values(self, raw_values: Dict[str, Any]) -> Dict[str, Any]:
        """Filter raw values dict to only include configured metrics."""
        if not self.config.metrics:
            return raw_values
        return {k: v for k, v in raw_values.items() if k in self.config.metrics}

    def check_thresholds(self, values: Dict[str, Any]) -> None:
        """
        Log warning messages for any metric that exceeds its configured threshold.

        Args:
            values: Current metric values from a TelemetrySample.
        """
        for metric_name, threshold_config in self.config.thresholds.items():
            if not isinstance(threshold_config, dict):
                continue
            warning_val = threshold_config.get("warning")
            if warning_val is None:
                continue
            current = values.get(metric_name)
            if current is not None and isinstance(current, (int, float)):
                if current > warning_val:
                    logger.warning(
                        "[Telemetry] %s.%s = %.2f exceeds warning threshold %.2f",
                        self.module_name,
                        metric_name,
                        current,
                        warning_val,
                    )

    def compute_averages(self) -> Dict[str, Any]:
        """
        Compute average values across all collected samples.

        Returns:
            Dict of metric_name -> average_value for numeric metrics.
            Non-numeric metrics are excluded from averages.
        """
        if not self._samples:
            return {}

        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for sample in self._samples:
            for key, value in sample.values.items():
                if isinstance(value, (int, float)):
                    sums[key] = sums.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + 1

        return {key: round(sums[key] / counts[key], 4) for key in sums if counts[key] > 0}

    def compute_min_max(self) -> Dict[str, Dict[str, float]]:
        """
        Compute min and max values across all collected samples.

        Returns:
            Dict of metric_name -> {"min": ..., "max": ...} for numeric metrics.
        """
        if not self._samples:
            return {}

        mins: Dict[str, float] = {}
        maxs: Dict[str, float] = {}

        for sample in self._samples:
            for key, value in sample.values.items():
                if isinstance(value, (int, float)):
                    if key not in mins or value < mins[key]:
                        mins[key] = value
                    if key not in maxs or value > maxs[key]:
                        maxs[key] = value

        return {key: {"min": round(mins[key], 4), "max": round(maxs[key], 4)} for key in mins}

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary dict containing averages, min/max, sample count, and configuration.

        Returns:
            Dict with keys: module, configs, sample_count, averages, min_max, samples (list of dicts).
            The ``configs`` key holds chart/display configuration (metrics, thresholds,
            chart_type, title, scales, axes).
        """
        averages = self.compute_averages()
        min_max = self.compute_min_max()
        return {
            "module": self.module_name,
            "configs": {
                "metrics": self.config.metrics or [],
                "thresholds": self.config.thresholds or {},
                "chart_type": self.config.chart_type or "line",
                "title": self.config.title or {},
                "scales": self.config.scales or {},
                "axes": self.config.axes or [],
            },
            "sample_count": len(self._samples),
            "averages": averages,
            "min_max": min_max,
            "samples": [s.to_dict() for s in self._samples],
        }

    def reset(self) -> None:
        """Clear all collected samples."""
        self._samples.clear()
