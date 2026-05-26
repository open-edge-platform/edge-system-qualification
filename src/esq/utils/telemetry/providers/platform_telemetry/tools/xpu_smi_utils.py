# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Thin per-PCI metric helpers around the ``xpu-smi`` CLI.

This module isolates every ``xpu-smi`` subprocess invocation behind a
single class so ``qmassa_collector.py`` can stay focused on qmmd scrape
and per-metric backfill orchestration.

``xpu-smi`` is used **only** as a per-metric backfill for GPUs already
discovered by qmmd (typically dGPU power / memory bandwidth on
platforms where the qmmd Prometheus surface is incomplete).  It is not
a standalone collector and never drives device discovery — that
responsibility stays with qmassa_collector.

All subprocesses are invoked unprivileged; access (``xpu-smi`` group /
permissions) must be granted up front by the ESQ pre-requisites script.
"""

from __future__ import annotations

import json
import os
import re
import subprocess  # nosec B404 # Wraps xpu-smi CLI for GPU telemetry backfill.
import time
from pathlib import Path
from typing import Dict, List, Optional


class XpuSmiUtils:
    """Cached, unprivileged wrapper around the ``xpu-smi`` CLI."""

    # Bandwidth output is sampled at fixed cadence by xpu-smi itself; cache
    # the last result for this many seconds so repeated metric backfill
    # calls inside a single qmassa scrape do not re-spawn the process.
    _DUMP_CACHE_TTL_S = 5.0

    def __init__(self) -> None:
        # Shared cache populated by a single ``xpu-smi dump`` invocation
        # covering utilization (m=0), memory utilization (m=5) and
        # memory bandwidth (m=6,7). Keyed by lower-cased PCI BDF.
        self._dump_bw_map: Dict[str, float] = {}
        self._dump_util_map: Dict[str, float] = {}
        self._dump_mem_util_map: Dict[str, float] = {}
        self._dump_cache_ts: float = 0.0

    @staticmethod
    def is_available() -> bool:
        """Return True when the ``xpu-smi`` executable is on ``PATH``."""
        return any(
            (Path(path) / "xpu-smi").exists()
            for path in os.environ.get("PATH", "").split(":")
        )

    # ------------------------------------------------------------------
    # Public per-metric maps (PCI BDF lower-case → value)
    # ------------------------------------------------------------------

    def power_map_w(self) -> Dict[str, float]:
        """Per-device power in watts, keyed by lower-cased PCI BDF."""
        if not self.is_available():
            return {}

        try:
            discovery = self._run_json(["discovery", "-j"])
        except Exception:
            return {}

        devices = discovery.get("device_list") if isinstance(discovery, dict) else []
        if not isinstance(devices, list):
            return {}

        power_map: Dict[str, float] = {}
        for dev in devices:
            if not isinstance(dev, dict):
                continue
            device_id = dev.get("device_id")
            bdf = str(dev.get("bdf_address") or dev.get("pci_bdf_address") or "").lower().strip()
            if device_id is None or not bdf:
                continue
            try:
                stats = self._run_json(["stats", "-d", str(device_id), "-j"])
            except Exception:
                continue

            power = self._extract_power_w(stats)
            if power is None:
                power = self._dump_power_w(str(device_id))
            if power is not None:
                power_map[bdf] = power

        return power_map

    def bandwidth_map_mb_s(self) -> Dict[str, float]:
        """Per-device memory read+write throughput in MB/s, by lower-cased PCI BDF."""
        self._refresh_dump_cache()
        return dict(self._dump_bw_map)

    def utilization_map(self) -> Dict[str, float]:
        """Per-device GPU engine utilization in percent (0–100)."""
        self._refresh_dump_cache()
        return dict(self._dump_util_map)

    def memory_util_map(self) -> Dict[str, float]:
        """Per-device GPU memory utilization in percent (0–100)."""
        self._refresh_dump_cache()
        return dict(self._dump_mem_util_map)

    def _refresh_dump_cache(self) -> None:
        """Run one ``xpu-smi dump`` and populate util/mem_util/bw caches."""
        if not self.is_available():
            return
        now = time.monotonic()
        if now - self._dump_cache_ts < self._DUMP_CACHE_TTL_S:
            return

        try:
            discovery = self._run_json(["discovery", "-j"])
        except Exception:
            return

        devices = discovery.get("device_list") if isinstance(discovery, dict) else []
        if not isinstance(devices, list):
            return

        id_to_bdf: Dict[str, str] = {}
        for dev in devices:
            if not isinstance(dev, dict):
                continue
            device_id = dev.get("device_id")
            bdf = str(dev.get("bdf_address") or dev.get("pci_bdf_address") or "").lower().strip()
            if device_id is None or not bdf:
                continue
            id_to_bdf[str(device_id)] = bdf

        if not id_to_bdf:
            return

        # Columns after timestamp, device_id: util(%), mem_util(%), read(kB/s), write(kB/s)
        cmd = ["xpu-smi", "dump", "-d", "-1", "-m", "0,5,6,7", "-n", "1"]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5, check=False,
            )
            output = proc.stdout if proc.returncode == 0 else ""
        except Exception:
            output = ""

        bw_map: Dict[str, float] = {}
        util_map: Dict[str, float] = {}
        mem_util_map: Dict[str, float] = {}
        for raw_line in output.splitlines()[1:]:
            cols = [c.strip() for c in raw_line.split(",")]
            if len(cols) < 6:
                continue
            dev_id = cols[1]
            bdf = id_to_bdf.get(dev_id)
            if not bdf:
                continue

            util_v = self._parse_numeric(cols[2])
            if util_v is not None:
                util_map[bdf] = max(0.0, min(100.0, util_v))

            mem_util_v = self._parse_numeric(cols[3])
            if mem_util_v is not None:
                mem_util_map[bdf] = max(0.0, min(100.0, mem_util_v))

            read_v = self._parse_numeric(cols[4])
            write_v = self._parse_numeric(cols[5])
            if read_v is not None and write_v is not None:
                bw_map[bdf] = max(0.0, (read_v + write_v) / 1024.0)

        self._dump_bw_map = bw_map
        self._dump_util_map = util_map
        self._dump_mem_util_map = mem_util_map
        self._dump_cache_ts = now

    @staticmethod
    def _parse_numeric(cell: str) -> Optional[float]:
        if not cell or cell.upper() == "N/A":
            return None
        m = re.search(r"[-+]?\d*\.?\d+", cell)
        if not m:
            return None
        try:
            return float(m.group(0))
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_json(args: List[str]) -> dict:
        proc = subprocess.run(
            ["xpu-smi", *args],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "xpu-smi command failed")
        payload = proc.stdout.strip()
        if not payload:
            raise RuntimeError("xpu-smi returned empty output")
        return json.loads(payload)

    @staticmethod
    def _dump_power_w(device_id: str) -> Optional[float]:
        """Fallback for environments where ``stats -j`` omits a power value."""
        try:
            proc = subprocess.run(
                ["xpu-smi", "dump", "-d", device_id, "-m", "1", "-n", "1"],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            )
        except Exception:
            return None

        if proc.returncode != 0:
            return None

        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if len(lines) < 2:
            return None

        # Example row: "11:17:51.247,    0,  43.25"
        row = lines[-1]
        cols = [c.strip() for c in row.split(",")]
        if not cols:
            return None
        value = cols[-1]
        if value.upper() == "N/A":
            return None

        try:
            watts = float(value)
        except ValueError:
            m = re.search(r"[-+]?\d*\.?\d+", value)
            if not m:
                return None
            watts = float(m.group(0))

        return max(0.0, watts)

    @classmethod
    def _extract_power_w(cls, payload: object) -> Optional[float]:
        preferred_keys = {
            "gpu_power",
            "gpu_power_w",
            "power",
            "power_w",
            "power_draw",
            "board_power",
            "average_power",
        }

        def walk(node: object) -> Optional[float]:
            if isinstance(node, dict):
                for key, value in node.items():
                    key_l = str(key).lower()
                    if key_l in preferred_keys:
                        parsed = cls._to_watts(value)
                        if parsed is not None:
                            return parsed
                    # Avoid pulling power limit/threshold values.
                    if "power" in key_l and all(
                        x not in key_l for x in ("limit", "max", "min", "threshold", "cap")
                    ):
                        parsed = cls._to_watts(value)
                        if parsed is not None:
                            return parsed
                for value in node.values():
                    nested = walk(value)
                    if nested is not None:
                        return nested
            elif isinstance(node, list):
                for value in node:
                    nested = walk(value)
                    if nested is not None:
                        return nested
            return None

        return walk(payload)

    @staticmethod
    def _to_watts(value: object) -> Optional[float]:
        if isinstance(value, (int, float)):
            v = float(value)
        elif isinstance(value, str):
            m = re.search(r"[-+]?\d*\.?\d+", value)
            if not m:
                return None
            v = float(m.group(0))
            text = value.lower()
            if "mw" in text and "w" in text:
                v /= 1000.0
            return max(0.0, v)
        else:
            return None

        if v < 0:
            return None
        # Heuristic: values in micro-watts are much larger than practical board power.
        if v > 10_000.0:
            v /= 1_000_000.0
        return max(0.0, v)
