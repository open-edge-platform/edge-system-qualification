# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NPU telemetry collector backed by the upstream Intel npu-monitor-tool.

The tool is sparse-cloned at runtime by
``npu_monitor_fetcher.ensure_tool()``.  Each ``collect_once()`` call
runs the tool once with ``--csv`` redirected into a per-invocation
working directory, parses the most recent CSV row and emits
``MetricSample`` instances using the same metric names the in-process
``NpuCollector`` produces.  This makes it a drop-in *primary* NPU
collector, with the in-process PMT collector kept as a fallback for
hosts without ``git``/network/Python access to clone the tool.

CSV schema (from upstream ``npu-monitor-tool.py``)::

    timestamp,power,frequency,bandwidth,tile_config,temperature,utilization,memory_usage

* ``power``         – watts (float)
* ``frequency``     – display frequency in Hz (int) — converted to MHz
* ``bandwidth``     – MB/s (float)
* ``temperature``   – °C (int)
* ``utilization``   – percentage (int)
* ``memory_usage``  – MiB (float, ``-1.0`` when unsupported)
"""

from __future__ import annotations

import csv
import logging
import shutil
import subprocess  # nosec B404 # For npu-monitor-tool subprocess invocation
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..base import BaseCollector
from ..models import MetricSample
from .npu_monitor_fetcher import ensure_tool

logger = logging.getLogger(__name__)


class NpuMonitorToolCollector(BaseCollector):
    """Run npu-monitor-tool out-of-process and parse its CSV output."""

    name = "npu_monitor_tool"

    def __init__(
        self,
        *,
        interval_ms: int = 200,
        run_timeout_s: float = 6.0,
        clone_timeout_s: float = 30.0,
    ) -> None:
        self._running = False
        self._tool_path: Optional[Path] = None
        self._interval_ms = max(50, int(interval_ms))
        self._run_timeout_s = float(run_timeout_s)
        self._clone_timeout_s = float(clone_timeout_s)
        self._unavailable_reason = ""

    # ── BaseCollector interface ───────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        try:
            self._tool_path = ensure_tool(timeout_s=self._clone_timeout_s)
        except Exception as exc:
            self._tool_path = None
            self._unavailable_reason = f"fetch_error:{exc}"
        if self._tool_path is None and not self._unavailable_reason:
            self._unavailable_reason = "tool_unavailable"

    def stop(self) -> None:
        self._running = False

    def collect_once(self) -> List[MetricSample]:
        if not self._running or self._tool_path is None:
            return []

        try:
            row = self._run_once_and_read_csv()
        except Exception as exc:
            logger.debug("[%s] tool execution failed: %s", self.name, exc)
            return []

        if not row:
            return []

        return self._row_to_samples(row)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_command(self, run_dir: Path) -> list[str]:
        """Build the subprocess command line.

        The upstream tool takes a single one-shot reading when ``--interval``
        is omitted — but it still sleeps once for ``DEFAULT_INTERVAL_MS``
        before printing.  We pass ``-i`` explicitly to make the cadence
        configurable from the collector caller.
        """
        cmd = [
            sys.executable,
            str(self._tool_path),
            "-i",
            str(self._interval_ms),
            "--csv",
        ]
        return cmd

    def _run_once_and_read_csv(self) -> Optional[dict]:
        """Run the tool inside a temp dir and return the latest CSV row."""
        run_dir = Path(tempfile.mkdtemp(prefix="esq_npumon_"))
        try:
            cmd = self._build_command(run_dir)

            # The upstream loop is infinite when -i is supplied, so we
            # cap with timeout + bound the runtime to one sample plus a
            # small safety margin.  When the timeout fires the script's
            # CSV row has already been flushed because csv_file.flush()
            # runs inside the loop.
            soft_runtime_s = max(self._interval_ms / 1000.0 + 0.5, 0.6)

            try:
                subprocess.run(
                    cmd,
                    cwd=str(run_dir),
                    timeout=soft_runtime_s,
                    check=False,
                    capture_output=True,
                )
            except subprocess.TimeoutExpired:
                # Expected: the upstream tool loops forever once -i is set.
                pass
            except OSError as exc:
                logger.debug("[%s] subprocess failed: %s", self.name, exc)
                return None

            output_dir = run_dir / "npu_output"
            if not output_dir.is_dir():
                return None

            csv_files = sorted(output_dir.glob("npu_*.csv"))
            if not csv_files:
                return None

            latest = csv_files[-1]
            try:
                with latest.open("r", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
            except OSError:
                return None
            if not rows:
                return None
            return rows[-1]
        finally:
            # ``ignore_errors`` keeps cleanup resilient — the OS will reclaim
            # ``/tmp`` even if a stray file fails to unlink.
            shutil.rmtree(run_dir, ignore_errors=True)

    def _row_to_samples(self, row: dict) -> List[MetricSample]:
        now = datetime.utcnow().isoformat() + "Z"
        tags = {"vendor": "Intel", "metric_origin": "npu-monitor-tool"}
        samples: List[MetricSample] = []

        utilization = self._to_float(row.get("utilization"))
        if utilization is not None:
            samples.append(
                MetricSample(
                    timestamp_utc=now,
                    collector=self.name,
                    device="NPU",
                    metric_name="npu.utilization",
                    value=max(0.0, min(100.0, utilization)),
                    unit="%",
                    tags=tags,
                )
            )

        power = self._to_float(row.get("power"))
        if power is not None:
            samples.append(
                MetricSample(
                    timestamp_utc=now,
                    collector=self.name,
                    device="NPU",
                    metric_name="npu.power_w",
                    value=max(0.0, power),
                    unit="W",
                    tags=tags,
                )
            )

        # Upstream emits frequency in Hz (display freq); convert to MHz so it
        # aligns with our other collectors and the unit map in platform_telemetry.
        freq_hz = self._to_float(row.get("frequency"))
        if freq_hz is not None:
            samples.append(
                MetricSample(
                    timestamp_utc=now,
                    collector=self.name,
                    device="NPU",
                    metric_name="npu.frequency_mhz",
                    value=max(0.0, freq_hz / 1_000_000.0),
                    unit="MHz",
                    tags=tags,
                )
            )

        bandwidth_mb_s = self._to_float(row.get("bandwidth"))
        if bandwidth_mb_s is not None:
            samples.append(
                MetricSample(
                    timestamp_utc=now,
                    collector=self.name,
                    device="NPU",
                    metric_name="npu.bandwidth_mb_s",
                    value=max(0.0, bandwidth_mb_s),
                    unit="MB/s",
                    tags=tags,
                )
            )

        temperature = self._to_float(row.get("temperature"))
        if temperature is not None:
            samples.append(
                MetricSample(
                    timestamp_utc=now,
                    collector=self.name,
                    device="NPU",
                    metric_name="npu.temperature_c",
                    value=max(0.0, temperature),
                    unit="°C",
                    tags=tags,
                )
            )

        memory_mb = self._to_float(row.get("memory_usage"))
        memory_origin = "npu_monitor_tool"
        if memory_mb is None or memory_mb < 0:
            # npu-monitor-tool returns -1 when the NPU driver does not
            # expose memory usage in the way the tool expects; fall back to
            # the accel sysfs node which is present on recent ivpu builds.
            sysfs_mb = self._sysfs_npu_memory_mb()
            if sysfs_mb is not None:
                memory_mb = sysfs_mb
                memory_origin = "sysfs"
            else:
                memory_mb = 0.0
                memory_origin = "unavailable"

        mem_tags = dict(tags)
        mem_tags["metric_origin"] = memory_origin
        # Published as an absolute size (MB) rather than a percent;
        # see NpuCollector for the naming rationale.
        samples.append(
            MetricSample(
                timestamp_utc=now,
                collector=self.name,
                device="NPU",
                metric_name="npu.memory_used_mb",
                value=memory_mb,
                unit="MB",
                tags=mem_tags,
            )
        )

        return samples

    @staticmethod
    def _sysfs_npu_memory_mb() -> Optional[float]:
        """Read NPU memory usage in MB from the accel sysfs node.

        On recent ivpu drivers the accel subsystem exposes
        ``/sys/class/accel/accel*/device/npu_memory_utilization`` containing
        a byte count.  Returns ``None`` when no such node exists or it
        cannot be parsed.
        """
        from pathlib import Path  # local import to keep module import surface tiny

        accel_root = Path("/sys/class/accel")
        if not accel_root.exists():
            return None
        for accel in sorted(accel_root.iterdir()):
            node = accel / "device" / "npu_memory_utilization"
            try:
                raw = node.read_text().strip()
                bytes_used = float(raw)
                if bytes_used < 0:
                    continue
                return round(bytes_used / (1024.0 * 1024.0), 3)
            except (OSError, ValueError):
                continue
        return None

    @staticmethod
    def _to_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
