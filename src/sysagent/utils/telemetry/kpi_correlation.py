# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
KPI ↔ Telemetry correlation builder.

Computes a per-test correlation block linking the result's single key metric
(``Metrics.is_key_metric=True``) to the resource consumption captured in the
telemetry summary.  The block is injected into
``result.extended_metadata["telemetry"]["kpi_correlation"]`` so the Allure
report can render four elements:

  1. KPI banner row  (key metric + PASS/FAIL + target if available)
  2. Cost-per-KPI-unit table  (one row per <device, resource metric>)
  3. Resource attribution stacked bar  (% of total Δ-power per device)
  4. Dominant-device annotation hint  (rendered as a chart badge)

Direction (``higher_is_better`` / ``lower_is_better``) is auto-detected from
the key metric unit.  Mixed resource units are preserved per row — no global
Joules normalization is performed.

Sysfs-mode telemetry (the sysfs module suite without ``device_name`` per
module) is not supported and yields ``None`` so the renderer can omit the
section gracefully.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# --- direction detection ----------------------------------------------------

# Higher-is-better tokens: throughput / rate units. Checked first so units
# like "streams" or "fps" don't accidentally match the substring "ns" in the
# lower-is-better set below.
_HIGHER_BETTER_UNIT_TOKENS = (
    "fps",
    "mb/s", "gb/s", "kb/s", "tb/s",
    "mb_s", "gb_s", "kb_s",
    "samples/s", "tokens/s", "iters/s", "ops/s", "req/s",
    "streams",
    "iops",
    "mflops", "gflops", "tflops",
)

# Lower-is-better tokens: latency / duration units. Matched as whole words
# (alphanumeric boundaries) to avoid colliding with throughput unit text.
_LOWER_BETTER_WORD_TOKENS = (
    "ms", "millisecond", "milliseconds",
    "us", "microsecond", "microseconds",
    "ns", "nanosecond", "nanoseconds",
    "s", "sec", "second", "seconds",
    "latency",
)

_WORD_RE = re.compile(r"[a-z]+")


def _detect_direction(unit: Optional[str]) -> str:
    """Return ``higher_is_better`` or ``lower_is_better`` from unit string."""
    if not unit:
        return "higher_is_better"
    u = str(unit).strip().lower()
    # Higher-better is checked first because throughput tokens are more
    # specific (and their substrings could otherwise match latency words).
    for token in _HIGHER_BETTER_UNIT_TOKENS:
        if token in u:
            return "higher_is_better"
    # Lower-better only matches whole-word alphabetic tokens.
    words = set(_WORD_RE.findall(u))
    if words & set(_LOWER_BETTER_WORD_TOKENS):
        return "lower_is_better"
    return "higher_is_better"


# --- helpers ---------------------------------------------------------------

# Sentinel emitted by collectors when a metric could not be read. Treated
# as "no data" so it never poisons deltas, costs, or share calculations.
MISSING_VALUE: float = -1.0


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _real_float(value: Any) -> Optional[float]:
    """Like ``_safe_float`` but also drops the MISSING_VALUE sentinel."""
    fv = _safe_float(value)
    if fv is None or fv == MISSING_VALUE:
        return None
    return fv


def _samples_window(samples: List[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    """Return (first_ts, last_ts) of a sample list, else None."""
    if not samples:
        return None
    timestamps: List[float] = []
    for s in samples:
        ts = _safe_float(s.get("timestamp"))
        if ts is not None:
            timestamps.append(ts)
    if not timestamps:
        return None
    return (min(timestamps), max(timestamps))


def _baseline_avg(module: Dict[str, Any], metric: str) -> Optional[float]:
    """Pull idle baseline avg for a metric, if a prerun baseline is attached."""
    base = module.get("baseline") or {}
    metrics = base.get("metrics") or {}
    entry = metrics.get(metric) or {}
    # Drop the -1 sentinel so deltas are not computed against fake values.
    return _real_float(entry.get("avg"))


def _format_cost_unit(resource_unit: str, kpi_unit: str, direction: str) -> str:
    """Build a human-readable ``cost`` unit label.

    Resource units are kept as-is (W, %, MB/s, °C). The frontend applies SI
    prefix scaling (m, µ) so values like 0.0003 W/(MB/s) render as
    300 µW/(MB/s) rather than being converted to J/MB.
    """
    r = (resource_unit or "").strip() or "unit"
    k = (kpi_unit or "").strip() or "kpi"
    if direction == "higher_is_better":
        # cost = Δ_resource / kpi_value  ⇒  "<r> per <k>"
        return f"{r} / {k}"
    # cost = Δ_resource × kpi_value  ⇒  "<r>·<k>"
    return f"{r}·{k}"


# --- noise-floor classification --------------------------------------------

# Per-metric absolute noise floors. Below these |Δ| values, the change is
# within sensor accuracy and should not be presented as a meaningful cost.
# Token → (floor, classification hint). Tokens are matched against the
# lowercase metric key by substring.
_NOISE_FLOORS: Tuple[Tuple[str, float], ...] = (
    ("power_w", 0.5),          # 0.5 W — platform power-meter accuracy
    ("bandwidth_mb_s", 50.0),  # 50 MB/s
    ("temperature_c", 1.0),    # 1 °C
    ("memory_utilization", 2.0),  # 2 %
    ("utilization", 2.0),      # 2 %  (catch-all for *_utilization)
)


def _signal_class(metric: str, delta_avg: Optional[float], is_primary: bool) -> Tuple[str, bool]:
    """Classify a cost row and indicate whether its cost number is meaningful.

    Returns ``(signal_class, cost_meaningful)`` where ``signal_class`` is one
    of ``primary`` / ``significant`` / ``low`` / ``negative`` and
    ``cost_meaningful`` is False when the underlying Δ falls below the
    metric-class noise floor (cost should then be rendered as “—”).

    - ``primary``     → the device's anchor metric, always shown. Cost may
                        still be flagged not-meaningful when the device sat
                        idle (e.g. NPU during an iGPU-only test).
    - ``negative``    → workload value below idle (delta < 0); typically jitter.
    - ``low``         → |delta| under the metric-class noise floor.
    - ``significant`` → above the floor; cost is meaningful.
    """
    if delta_avg is None:
        return ("low", False)
    if delta_avg < 0:
        cls = "primary" if is_primary else "negative"
        return (cls, False)
    mlow = metric.lower()
    floor: Optional[float] = None
    for token, threshold in _NOISE_FLOORS:
        if token in mlow:
            floor = threshold
            break
    below_floor = floor is not None and abs(delta_avg) < floor
    if below_floor:
        cls = "primary" if is_primary else "low"
        return (cls, False)
    return ("significant", True)

# --- main builder ----------------------------------------------------------

def build_kpi_correlation(result: Any, telemetry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build the KPI-to-telemetry correlation dict.

    Args:
        result: ``Result`` instance with ``metrics`` (``Metrics`` dataclass map)
                and ``metadata``.
        telemetry: ``result.extended_metadata["telemetry"]`` dict, already
                   populated with ``modules`` and (optionally) ``baseline``
                   merged onto each module.

    Returns:
        A correlation dict (see module docstring) or ``None`` when no key
        metric / no fw_sys-style modules are available.
    """
    if not telemetry or not isinstance(telemetry, dict):
        return None
    modules = telemetry.get("modules") or {}
    if not modules:
        return None

    # --- resolve key metric -------------------------------------------------
    metrics_map = getattr(result, "metrics", None) or {}
    key_name: Optional[str] = None
    key_metric_obj: Any = None
    for name, m in metrics_map.items():
        if getattr(m, "is_key_metric", False):
            key_name = name
            key_metric_obj = m
            break
    if key_name is None or key_metric_obj is None:
        # No key metric set on the result — nothing to correlate.
        return None

    key_value = _safe_float(getattr(key_metric_obj, "value", None))
    if key_value is None or key_value == 0:
        # Unable to compute cost without a finite, non-zero KPI value.
        return None
    key_unit = str(getattr(key_metric_obj, "unit", "") or "")
    direction = _detect_direction(key_unit)

    # --- duration -----------------------------------------------------------
    duration_s: Optional[float] = None
    metadata = getattr(result, "metadata", {}) or {}
    duration_s = _safe_float(metadata.get("total_duration_seconds"))
    if duration_s is None:
        windows: List[Tuple[float, float]] = []
        for mod in modules.values():
            # When a module nests per-device summaries under ``device_groups``
            # the parent itself may not carry ``samples``; inspect the children
            # so the duration heuristic still works for the new layout.
            nested = mod.get("device_groups")
            if isinstance(nested, list) and nested:
                for sub in nested:
                    if isinstance(sub, dict):
                        w = _samples_window(sub.get("samples") or [])
                        if w:
                            windows.append(w)
                continue
            w = _samples_window(mod.get("samples") or [])
            if w:
                windows.append(w)
        if windows:
            duration_s = max(w[1] for w in windows) - min(w[0] for w in windows)

    # --- per-device aggregation --------------------------------------------
    # We only correlate when the module advertises a ``device_name`` field
    # (platform_telemetry virtual modules); otherwise we cannot bucket cleanly.
    # Modules may expose per-device telemetry either directly (legacy flat
    # layout) or via a ``device_groups`` list (nested layout where the
    # parent key keeps its original module name, e.g. ``platform_telemetry``).
    device_modules: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for mod_name, mod in modules.items():
        nested = mod.get("device_groups")
        if isinstance(nested, list) and nested:
            for sub in nested:
                if not isinstance(sub, dict):
                    continue
                dev = str(sub.get("device_name") or "").strip()
                if not dev:
                    continue
                sub_name = str(sub.get("module") or mod_name)
                device_modules.setdefault(dev, []).append((sub_name, sub))
            continue
        dev = str(mod.get("device_name") or "").strip()
        if not dev:
            continue
        device_modules.setdefault(dev, []).append((mod_name, mod))

    if not device_modules:
        return None

    # Resource metrics we surface in the cost table, ordered by interpretive
    # priority. Substring match against the metric key. ``frequency_mhz`` is
    # intentionally omitted: clock rate tells you what the silicon *did*,
    # not the *cost* of doing it (that information is already captured by
    # power and utilization).
    resource_priority = (
        "power_w",
        "utilization",
        "bandwidth_mb_s",
        "memory_utilization",
        "temperature_c",
    )

    cost_rows: List[Dict[str, Any]] = []
    delta_power_by_device: Dict[str, float] = {}
    primary_metric_by_device: Dict[str, str] = {}

    for device, mods in device_modules.items():
        # Aggregate (averages, min_max, scales) across the device's modules.
        # In fw_sys mode each device is a single virtual module, but we keep
        # this loop tolerant of future multi-module devices.
        merged_avg: Dict[str, float] = {}
        merged_minmax: Dict[str, Dict[str, float]] = {}
        merged_scales: Dict[str, Dict[str, Any]] = {}
        merged_baseline: Dict[str, float] = {}
        sample_count = 0
        for _mod_name, mod in mods:
            for k, v in (mod.get("averages") or {}).items():
                fv = _real_float(v)
                if fv is not None:
                    merged_avg[k] = fv
            for k, v in (mod.get("min_max") or {}).items():
                if isinstance(v, dict):
                    vmin = _real_float(v.get("min"))
                    vmax = _real_float(v.get("max"))
                    # Skip metrics where both ends are missing-data sentinels.
                    if vmin is None and vmax is None:
                        continue
                    merged_minmax[k] = {
                        "min": vmin if vmin is not None else 0.0,
                        "max": vmax if vmax is not None else 0.0,
                    }
            scales = (mod.get("configs") or {}).get("scales") or {}
            for k, v in scales.items():
                if isinstance(v, dict):
                    merged_scales[k] = v
            for k in list(merged_avg.keys()):
                b = _baseline_avg(mod, k)
                if b is not None:
                    merged_baseline[k] = b
            sample_count = max(sample_count, int(mod.get("sample_count") or 0))

        # Pick the primary resource metric for this device (first match in
        # priority list that has a workload average).
        primary = next(
            (
                metric
                for token in resource_priority
                for metric in merged_avg
                if token in metric.lower()
            ),
            None,
        )
        if primary:
            primary_metric_by_device[device] = primary

        # Track power Δ for share computation.
        power_metric = next((k for k in merged_avg if "power_w" in k.lower()), None)
        if power_metric is not None:
            workload = merged_avg[power_metric]
            idle = merged_baseline.get(power_metric)
            delta = workload - idle if idle is not None else workload
            if delta > 0:
                delta_power_by_device[device] = delta

        # Build cost rows per metric (mixed units preserved).
        for metric, workload_avg in merged_avg.items():
            mlow = metric.lower()
            if not any(token in mlow for token in resource_priority):
                continue
            # Skip metrics that are flat-zero on both idle and workload
            # (e.g. dGPU bandwidth during a CPU memory benchmark): no
            # signal, just clutter.
            idle_avg_check = merged_baseline.get(metric)
            if (
                abs(workload_avg) < 1e-9
                and (idle_avg_check is None or abs(idle_avg_check) < 1e-9)
            ):
                continue
            workload_peak = (merged_minmax.get(metric) or {}).get("max")
            idle_avg = idle_avg_check
            # Without an idle baseline we cannot derive a meaningful delta /
            # cost: emit the workload values but leave delta + cost as None
            # so the renderer prints "—" instead of a misleading number.
            if idle_avg is None:
                delta_avg: Optional[float] = None
                cost_value: Optional[float] = None
            else:
                delta_avg = workload_avg - idle_avg
                if direction == "higher_is_better":
                    cost_value = delta_avg / key_value if key_value else None
                else:
                    cost_value = delta_avg * key_value
            resource_unit = str((merged_scales.get(metric) or {}).get("unit") or "")
            signal_class, cost_meaningful = _signal_class(
                metric, delta_avg, is_primary=(metric == primary)
            )
            if not cost_meaningful:
                # Hide noisy / non-meaningful cost numbers; the row still
                # carries workload + idle averages for transparency.
                cost_value = None
            cost_rows.append(
                {
                    "device": device,
                    "metric": metric,
                    "metric_unit": resource_unit,
                    "workload_avg": workload_avg,
                    "workload_peak": workload_peak,
                    "idle_avg": idle_avg,
                    "delta_avg": delta_avg,
                    "cost_value": cost_value,
                    "cost_unit": _format_cost_unit(resource_unit, key_unit, direction),
                    "is_primary": metric == primary,
                    "signal_class": signal_class,
                    "cost_meaningful": cost_meaningful,
                }
            )

    # --- device attribution shares -----------------------------------------
    total_dpower = sum(delta_power_by_device.values())
    device_attribution: List[Dict[str, Any]] = []
    for device in sorted(device_modules.keys()):
        dpower = delta_power_by_device.get(device)
        share = (dpower / total_dpower) if (dpower and total_dpower) else None
        device_attribution.append(
            {
                "device": device,
                "delta_power_w": dpower,
                "share_power": share,
                "primary_metric": primary_metric_by_device.get(device),
            }
        )

    # Dominant device = largest power share, fall back to largest delta even
    # without a baseline.
    dominant_device: Optional[str] = None
    if delta_power_by_device:
        dominant_device = max(delta_power_by_device, key=delta_power_by_device.get)

    # --- target / validation ------------------------------------------------
    validation_status = str(metadata.get("kpi_validation_status") or "skipped")
    target_obj: Optional[Dict[str, Any]] = None
    kpis = getattr(result, "kpis", {}) or {}
    # kpis is shaped {kpi_name: {"config": {...}, "validation": {...}}}; pick
    # the entry whose config references our key metric, if any.
    for _kpi_name, kpi_block in kpis.items():
        cfg = (kpi_block or {}).get("config") or {}
        if cfg.get("metric") == key_name or cfg.get("name") == key_name:
            tgt = cfg.get("target") or cfg.get("threshold")
            if tgt is not None:
                target_obj = {"value": tgt, "op": cfg.get("op") or cfg.get("operator")}
                break

    return {
        "key_metric": {
            "name": key_name,
            "unit": key_unit,
            "value": key_value,
            "direction": direction,
            "validation_status": validation_status,
            "target": target_obj,
        },
        "duration_s": duration_s,
        "device_attribution": device_attribution,
        "cost_per_kpi_unit": cost_rows,
        "render_hints": {
            "dominant_device": dominant_device,
        },
    }


__all__ = ["build_kpi_correlation"]
