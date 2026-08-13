# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Parsing and chart generation for cyclictest JSON output.

Handles the native JSON schema written by ``cyclictest --json=FILENAME``,
derives per-run summary metrics, and builds the latency distribution chart
embedded in the Allure report.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def parse_cyclictest_json(json_path: str) -> tuple[list[dict], dict, str]:
    """Parse the JSON file written by cyclictest and return thread data.

    Returns ``(threads, top_level_meta, error_message)``.  ``error_message``
    is empty on success.
    """
    if not json_path or not os.path.exists(json_path):
        return [], {}, f"cyclictest JSON output not found: {json_path}"

    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as err:
        return [], {}, f"Failed to read cyclictest JSON output: {err}"

    thread_data = data.get("thread")
    if not isinstance(thread_data, dict) or not thread_data:
        return [], {}, "cyclictest JSON output contains no thread data"

    threads = []
    for thread_id, info in sorted(thread_data.items(), key=lambda x: int(x[0])):
        if not isinstance(info, dict):
            continue
        raw_hist = info.get("histogram", {})
        histogram = {int(k): int(v) for k, v in raw_hist.items() if str(k).lstrip("-").isdigit()}
        threads.append(
            {
                "id": int(thread_id),
                "cpu": int(info.get("cpu", -1)),
                "node": int(info.get("node", -1)),
                "min_us": int(info.get("min", -1)),
                "avg_us": float(info.get("avg", -1.0)),
                "max_us": int(info.get("max", -1)),
                "cycles": int(info.get("cycles", -1)),
                "histogram": histogram,
            }
        )

    if not threads:
        return [], {}, "cyclictest JSON thread data could not be parsed"

    # cyclictest serialises some top-level keys with a trailing colon —
    # try both forms.
    rt_ver_raw = data.get("rt_test_version:", data.get("rt_test_version", ""))
    top_level_meta = {
        "resolution_ns": int(data.get("resolution_in_ns", 0)),
        "rt_test_version": str(rt_ver_raw).strip(),
        "num_threads": int(data.get("num_threads", len(threads))),
    }
    return threads, top_level_meta, ""


def derive_summary_metrics(threads: list[dict]) -> dict:
    """Derive overall latency summary from per-thread stats."""
    if not threads:
        return {"max_latency_us": -1.0, "avg_latency_us": -1.0, "min_latency_us": -1.0, "total_cycles": 0}
    max_lat = float(max(t["max_us"] for t in threads))
    min_lat = float(min(t["min_us"] for t in threads))
    total_cycles = sum(t["cycles"] for t in threads if t["cycles"] >= 0)
    if total_cycles > 0:
        avg_lat = sum(t["avg_us"] * t["cycles"] for t in threads) / total_cycles
    else:
        avg_lat = sum(t["avg_us"] for t in threads) / len(threads)
    return {
        "max_latency_us": max_lat,
        "avg_latency_us": round(avg_lat, 2),
        "min_latency_us": min_lat,
        "total_cycles": total_cycles,
    }


def build_histogram_chart(threads: list[dict], run_info: dict | None = None) -> dict | None:
    """Build a RT latency distribution line chart from cyclictest histogram data.

    Each thread becomes one series in the chart.  Only non-zero histogram
    entries are included (the histogram is sparse — most µs buckets are
    empty).  The Y-axis uses a log₁₀ scale because the sample distribution
    is heavily skewed: >99.99 % of samples land at 1 µs while rare
    high-latency outliers appear at 50–200 µs, creating a dynamic range of
    ~5 orders of magnitude.

    The x-axis minimum range is **0 – 200 µs** so that typical runs share a
    consistent baseline for visual comparison.  If any measured latency
    exceeds 200 µs the axis expands automatically to show the full data; no
    samples are ever clipped.

    *run_info* is an optional dict of display-ready key/value pairs (both
    strings) that the report overlay renders as an info grid below the
    legend — useful for surfacing run duration, optimization status, and
    latency summary without cluttering the chart itself.

    Embeds the chart dict in ``extended_metadata["charts"]`` so the Allure
    report overlay renders it as an inline SVG without any attached images.
    """
    from sysagent.utils.core.charts import Chart, ChartSeries

    if not threads:
        return None

    series_list: list[ChartSeries] = []
    for thread in threads:
        histogram: dict = thread.get("histogram", {})
        if not histogram:
            continue
        data = [{"x": float(lat_us), "y": float(count)} for lat_us, count in sorted(histogram.items()) if count > 0]
        if not data:
            continue
        max_us = int(max(p["x"] for p in data))
        label = f"CPU {thread['cpu']} (max {max_us} µs)"
        series_list.append(ChartSeries(label=label, data=data))

    if not series_list:
        return None

    meta: dict = {
        "description": (
            "Per-CPU wakeup-latency histogram measured by cyclictest (log\u2081\u2080 Y-axis). "
            "A lower and narrower distribution indicates better real-time performance."
        ),
    }
    if run_info:
        meta.update(run_info)

    return Chart(
        id="latency_histogram",
        title="RT Latency Distribution",
        type="step",
        x_label="Latency",
        y_label="Number of latency samples",
        x_unit="µs",
        log_y=True,
        x_min=0.0,
        x_max=200.0,
        series=series_list,
        metadata=meta,
    ).to_dict()
