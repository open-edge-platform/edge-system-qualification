# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Chart data structures for embedding visualization data in test ``extended_metadata``.

Charts placed in ``Result.extended_metadata["charts"]`` are automatically detected
and rendered by the Allure report overlay (``ChartsSection`` component) as
interactive SVG charts — **no PNG attachments needed**.

Structure
---------
A chart is a dict produced by ``Chart.to_dict()``:

.. code-block:: json

    {
      "id":       "latency_histogram",
      "title":    "RT Latency Distribution",
      "type":     "line",
      "x_label":  "Latency",
      "y_label":  "Sample Count",
      "x_unit":   "µs",
      "y_unit":   "samples",
      "log_y":    true,
      "log_x":    false,
      "series": [
        {
          "label": "CPU 2 (Thread 0)",
          "color": "#1e88e5",
          "data":  [{"x": 0.0, "y": 17.0}, {"x": 1.0, "y": 118856.0}, ...]
        }
      ],
      "metadata": {
        "description": "..."
      }
    }

Usage
-----
.. code-block:: python

    from sysagent.utils.core.charts import Chart, ChartSeries

    series = ChartSeries(
        label="CPU 2 (Thread 0)",
        data=[{"x": 1.0, "y": 118856.0}, {"x": 2.0, "y": 44.0}],
        color="#1e88e5",
    )
    chart = Chart(
        id="latency_histogram",
        title="RT Latency Distribution",
        type="line",
        x_label="Latency", x_unit="µs",
        y_label="Sample Count", y_unit="samples",
        log_y=True,
        series=[series],
    )
    # Embed in test Result
    result.extended_metadata["charts"] = [chart.to_dict()]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# High-contrast categorical palette — mirrors ChartsSection.tsx SERIES_COLORS so
# any explicit colour assignments match the auto-assigned renderer colours.
_SERIES_COLORS: tuple[str, ...] = (
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#8c564b",  # brown
    "#bcbd22",  # olive
    "#7f7f7f",  # grey
)


@dataclass
class ChartSeries:
    """One named data series for a chart.

    Parameters
    ----------
    label:
        Human-readable series name displayed in the chart legend.
    data:
        List of ``{"x": float, "y": float}`` dicts.  For histogram charts,
        include **only non-zero** data points (sparse representation) —
        the renderer skips ``y == 0`` automatically.
    color:
        Optional hex colour string (e.g. ``"#1e88e5"``).  When ``None``,
        the renderer assigns a colour from the shared palette via
        :meth:`default_color`.
    """

    label: str
    data: list[dict[str, float]]
    color: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict for embedding in ``extended_metadata``."""
        d: dict[str, Any] = {"label": self.label, "data": self.data}
        if self.color is not None:
            d["color"] = self.color
        return d

    @staticmethod
    def default_color(index: int) -> str:
        """Return the palette colour for the given series *index* (0-based)."""
        return _SERIES_COLORS[index % len(_SERIES_COLORS)]


@dataclass
class Chart:
    """Standardized chart descriptor for ``extended_metadata["charts"]``.

    The Allure report overlay (``ChartsSection``) reads this structure and renders
    it as an interactive SVG chart without requiring any attached image files,
    keeping report archives small.

    Parameters
    ----------
    id:
        Unique identifier within the test result (e.g. ``"latency_histogram"``).
    title:
        Human-readable chart title shown in the card header.
    type:
        Chart type: ``"line"`` (only type currently rendered; others reserved).
    x_label / y_label:
        Axis label text.
    x_unit / y_unit:
        Unit suffix appended to the axis label (e.g. ``"µs"``, ``"samples"``).
    log_y / log_x:
        Whether to use a log₁₀ scale on the respective axis.
    series:
        Ordered list of :class:`ChartSeries` instances.
    metadata:
        Optional free-form dict (string values) for extra annotations such as
        ``"description"`` or ``"source"``.  The ``"description"`` key is shown
        as a caption below the chart title in the report.
    """

    id: str
    title: str
    type: str  # "line" | "step" | "bar" | "scatter"
    x_label: str
    y_label: str
    series: list[ChartSeries] = field(default_factory=list)
    x_unit: str = ""
    y_unit: str = ""
    log_y: bool = False
    log_x: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    # Minimum axis range: when set, the axis is guaranteed to span at least
    # from x_min to x_max (or y_min to y_max).  Data outside these bounds
    # still expands the range.  Use to keep charts comparable across runs
    # (e.g. always show 0–700 µs for cyclictest latency histograms).
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict for embedding in ``extended_metadata``."""
        d: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
            "log_y": self.log_y,
            "log_x": self.log_x,
            "series": [s.to_dict() for s in self.series],
            "metadata": self.metadata,
        }
        if self.x_min is not None:
            d["x_min"] = self.x_min
        if self.x_max is not None:
            d["x_max"] = self.x_max
        if self.y_min is not None:
            d["y_min"] = self.y_min
        if self.y_max is not None:
            d["y_max"] = self.y_max
        return d
