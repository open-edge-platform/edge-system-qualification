# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Thin per-PCI snapshot helper around the ``intel_gpu_top`` CLI.

This module isolates every ``intel_gpu_top`` subprocess invocation behind
a single class so ``qmassa_collector.py`` can stay focused on qmmd
scrape and per-metric backfill orchestration.

``intel_gpu_top`` is used **only** as a per-metric backfill for GPUs
already discovered by qmmd (iGPU engine utilization, frequency, package
power and memory bandwidth on platforms where the qmmd Prometheus
surface is incomplete).  It is not a standalone collector and never
drives device discovery.

The CLI streams JSON forever; we run it for ~200ms-1.5s, parse the
first complete sample, cache the result for one second to amortise
across all metrics of the device, and return.  All invocations are
unprivileged; ``CAP_PERFMON`` / perf paranoid configuration must be
granted up front by the ESQ pre-requisites script.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 # Wraps intel_gpu_top CLI for iGPU telemetry backfill.
import time
from pathlib import Path
from typing import Dict, Optional


class IntelGpuTopUtils:
    """Cached, unprivileged wrapper around the ``intel_gpu_top -J`` snapshot."""

    _CACHE_TTL_S = 1.0

    def __init__(self) -> None:
        self._cache_map: Dict[str, Dict[str, float]] = {}
        self._cache_ts: float = 0.0

    @staticmethod
    def is_available() -> bool:
        """Return True when the ``intel_gpu_top`` executable is on ``PATH``."""
        return any(
            (Path(path) / "intel_gpu_top").exists()
            for path in os.environ.get("PATH", "").split(":")
        )

    def snapshot(self, pci: str) -> Optional[Dict[str, float]]:
        """One-shot ``intel_gpu_top -J`` snapshot for ``pci`` (PCI slot).

        Returns a dict with any subset of ``utilization``, ``frequency_mhz``,
        ``power_w`` and ``bandwidth_mb_s``; absent keys mean intel_gpu_top
        did not surface that metric for the device.  ``None`` means the
        command was unavailable or the JSON stream could not be parsed.
        """
        if not pci:
            return None

        now = time.monotonic()
        cached = self._cache_map.get(pci.lower())
        if cached is not None and (now - self._cache_ts) < self._CACHE_TTL_S:
            return cached

        if not self.is_available():
            return None

        cmd = ["intel_gpu_top", "-J", "-s", "200", "-o", "-", "-d", f"pci:slot={pci}"]

        # intel_gpu_top streams JSON forever; the TimeoutExpired path is
        # the expected completion. Partial stdout lives on the exception.
        out = ""
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            out = proc.stdout or ""
        except subprocess.TimeoutExpired as exc:
            raw = exc.stdout or ""
            out = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        except OSError:
            return None

        sample = self._first_json_object(out)
        if sample is None:
            return None

        result: Dict[str, float] = {}
        # Engines: pick the maximum 'busy' percentage across listed engines.
        engines = sample.get("engines") or {}
        if isinstance(engines, dict):
            busys = [
                float(eng.get("busy", 0.0))
                for eng in engines.values()
                if isinstance(eng, dict)
            ]
            if busys:
                result["utilization"] = max(0.0, min(100.0, max(busys)))

        freq = sample.get("frequency") or {}
        if isinstance(freq, dict):
            actual = freq.get("actual")
            if actual is not None:
                try:
                    result["frequency_mhz"] = max(0.0, float(actual))
                except (TypeError, ValueError):
                    pass

        power = sample.get("power") or {}
        if isinstance(power, dict):
            # Only publish a value when intel_gpu_top reports a non-zero
            # GPU-domain reading. A 0 W reading means the render slice is
            # RC6 powergated; emitting it would pin power_w=0 and block
            # other fallbacks (xpu-smi / sysfs) from running.
            for key in ("GPU", "gpu"):
                if key in power:
                    try:
                        watts = float(power[key])
                    except (TypeError, ValueError):
                        continue
                    if watts > 0:
                        result["power_w"] = watts
                    break

        imc = sample.get("imc-bandwidth") or {}
        if isinstance(imc, dict):
            try:
                read = float(imc.get("reads", 0.0))
                write = float(imc.get("writes", 0.0))
                result["bandwidth_mb_s"] = max(0.0, read + write)
            except (TypeError, ValueError):
                pass

        self._cache_map[pci.lower()] = result
        self._cache_ts = now
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_json_object(text: str) -> Optional[dict]:
        """Extract the first balanced JSON object from a possibly-truncated stream."""
        if not text:
            return None
        # Skip a leading '[' so we land on the object directly.
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None
