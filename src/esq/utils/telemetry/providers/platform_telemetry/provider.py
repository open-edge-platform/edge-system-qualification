# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""In-process telecollect provider for platform_telemetry.

Historically this module was a thin HTTP client that talked to an
external telecollect daemon on :9700.  In practice the daemon was rarely
running on target hosts, leaving the framework-level telemetry path
without real GPU/NPU data.  The provider is now an **in-process**
collector stack that hosts the same qmassa (GPU), Intel PMT (NPU) and
psutil/RAPL/vmstat (CPU) collectors directly inside the esq Python
process — so no separate service has to be launched alongside the test
runner.

The provider name (``telecollect_collector``) is preserved for backward
compatibility with existing profile YAML, result schemas, and downstream
diagnostics; only the implementation has changed.

Sampling cadence: synchronous.  ``collect_sample()`` runs one
``collect_once()`` per collector and returns immediately, which lets the
existing ``TelemetryCollector`` background thread (started just before
``run-test`` and stopped right after) drive the cadence without a
second thread.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from .base import BaseCollector
from .models import MetricSample
from .prerequisites import detect, log_snapshot

from ..base import BaseFrameworkTelemetryProvider

logger = logging.getLogger(__name__)


class PlatformTelemetryProvider(BaseFrameworkTelemetryProvider):
    """In-process collector stack (qmassa + NPU PMT + psutil/RAPL)."""

    name = "platform_telemetry"

    def __init__(self, options: Dict[str, Any]) -> None:
        super().__init__(options)
        # Provider runs fully in-process; no HTTP endpoint is consulted.
        # qmassa / qmmd communication is loopback-only and resolved below.
        self._qmmd_url = str(
            options.get("telecollect_collector_qmmd_url")
            or options.get("platform_telemetry_qmmd_url")
            or "http://127.0.0.1:9000/metrics"
        )
        # qmassa is the **primary** GPU telemetry source for platform_telemetry.
        # When qmmd is not yet running we attempt a one-time install (cargo)
        # and start the daemon ourselves so that iGPU/dGPU metrics are always
        # gathered through the qmassa+xpu-smi stack.  Failure is non-fatal:
        # QmassaCollector transparently falls back to sysfs.
        self._qmmd_install_if_missing = bool(
            options.get("platform_telemetry_qmmd_install_if_missing", True)
        )
        self._qmmd_install_cargo_if_missing = bool(
            options.get("platform_telemetry_qmmd_install_cargo_if_missing", True)
        )
        self._qmmd_install_timeout_s = float(
            options.get("platform_telemetry_qmmd_install_timeout_s", 600.0)
        )
        self._qmmd_start_timeout_s = float(
            options.get("platform_telemetry_qmmd_start_timeout_s", 6.0)
        )
        self._qmmd_port = int(options.get("platform_telemetry_qmmd_port", 9000))

        self._capabilities: Dict[str, Any] = detect(self._qmmd_url)
        # If qmmd isn't reachable yet but a DRM card is present, try to
        # install + launch it before instantiating QmassaCollector so the
        # collector picks the qmmd path on its very first scrape.
        if (
            self._capabilities.get("drm_card")
            and not self._capabilities.get("qmmd")
        ):
            try:
                from esq.utils.telemetry.providers.platform_telemetry.tools.qmmd_installer import (
                    ensure_qmmd_running,
                )

                running_url = ensure_qmmd_running(
                    port=self._qmmd_port,
                    install_timeout_s=self._qmmd_install_timeout_s,
                    start_timeout_s=self._qmmd_start_timeout_s,
                    install_if_missing=self._qmmd_install_if_missing,
                    install_cargo_if_missing=self._qmmd_install_cargo_if_missing,
                )
            except Exception as exc:
                logger.debug(
                    "[%s] qmmd auto-bring-up raised: %s", self.name, exc
                )
                running_url = None
            if running_url:
                self._qmmd_url = running_url
                self._capabilities["qmmd"] = True
                self._capabilities["qmmd_url"] = running_url
        log_snapshot(self._capabilities)

        # Register a hard-kill / Ctrl+C safety net so any in-process
        # collector threads we start (and the qmmd daemon if we spawned it)
        # are reaped even when the normal pytest teardown path is bypassed.
        try:
            from esq.utils.telemetry.providers.platform_telemetry.cleanup import register

            register(self.close)
        except Exception:
            pass

        self._collectors: List[BaseCollector] = []
        self._started = False
        self._last_error = ""

        # CPU collector — gated on psutil availability.  Without psutil the
        # rest of the CPU pipeline cannot run, so fail closed for it but
        # still allow GPU/NPU collectors to start.
        if self._capabilities.get("psutil"):
            try:
                from esq.utils.telemetry.providers.platform_telemetry.system.cpu_collector import CpuCollector

                self._collectors.append(CpuCollector())
            except Exception as exc:
                logger.debug("[%s] CpuCollector unavailable: %s", self.name, exc)

        # GPU collector — qmassa is enabled whenever a DRM card or qmmd
        # endpoint is present.  qmmd / sysfs / xpu-smi fallback chain is
        # handled inside QmassaCollector itself.
        if self._capabilities.get("drm_card") or self._capabilities.get("qmmd"):
            try:
                from esq.utils.telemetry.providers.platform_telemetry.tools.qmassa_collector import QmassaCollector

                self._collectors.append(QmassaCollector(qmmd_url=self._qmmd_url))
            except Exception as exc:
                logger.debug("[%s] QmassaCollector unavailable: %s", self.name, exc)

        # NPU collector — default to the in-process PMT collector
        # (reads only /sys/class/intel_pmt, granted by system-setup.sh).
        # Set ``platform_telemetry_npu_use_upstream_tool`` to opt in to the
        # upstream npu-monitor-tool, which additionally needs debugfs
        # traversal (system-setup.sh Module 6).
        npu_added = False
        use_upstream_tool = bool(
            options.get("platform_telemetry_npu_use_upstream_tool", False)
        )
        if (
            use_upstream_tool
            and self._capabilities.get("intel_vpu")
            and self._capabilities.get("intel_pmt")
            and self._capabilities.get("git")
        ):
            try:
                from esq.utils.telemetry.providers.platform_telemetry.tools.npu_monitor_collector import (
                    NpuMonitorToolCollector,
                )

                self._collectors.append(
                    NpuMonitorToolCollector(
                        interval_ms=int(
                            options.get("platform_telemetry_npu_tool_interval_ms", 200)
                        ),
                        run_timeout_s=float(
                            options.get("platform_telemetry_npu_tool_run_timeout_s", 6.0)
                        ),
                        clone_timeout_s=float(
                            options.get("platform_telemetry_npu_tool_clone_timeout_s", 30.0)
                        ),
                    )
                )
                npu_added = True
            except Exception as exc:
                logger.debug(
                    "[%s] NpuMonitorToolCollector unavailable: %s", self.name, exc
                )

        # Default PMT-based in-process NPU collector.
        if (
            not npu_added
            and self._capabilities.get("intel_vpu")
            and self._capabilities.get("intel_pmt")
        ):
            try:
                from esq.utils.telemetry.providers.platform_telemetry.tools.npu_collector import NpuCollector

                self._collectors.append(NpuCollector())
            except Exception as exc:
                logger.debug("[%s] NpuCollector unavailable: %s", self.name, exc)

        if not self._collectors:
            logger.info(
                "[%s] no in-process collectors available on this host (capabilities=%s)",
                self.name,
                self._capabilities,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._started:
            return
        for collector in self._collectors:
            try:
                collector.start()
            except Exception as exc:
                logger.debug(
                    "[%s] collector %s failed to start: %s",
                    self.name,
                    getattr(collector, "name", type(collector).__name__),
                    exc,
                )
        self._started = True

    def close(self) -> None:
        for collector in self._collectors:
            try:
                collector.stop()
            except Exception:
                pass
        # If we spawned qmmd ourselves, tear it down with the provider so
        # we don't leak a daemon between test sessions.
        try:
            from esq.utils.telemetry.providers.platform_telemetry.tools.qmmd_installer import (
                stop_qmmd,
            )

            stop_qmmd()
        except Exception:
            pass
        self._started = False

    # ------------------------------------------------------------------
    # BaseFrameworkTelemetryProvider interface
    # ------------------------------------------------------------------

    def health(self) -> bool:
        if not self._collectors:
            return False
        self._ensure_started()
        return True

    def collect_sample(self) -> Dict[str, Any]:
        if not self._collectors:
            return {}
        self._ensure_started()

        flattened: Dict[str, float] = {}
        for collector in self._collectors:
            try:
                samples = collector.collect_once() or []
            except Exception as exc:
                self._last_error = f"{getattr(collector, 'name', '?')}: {exc}"
                logger.debug("[%s] collect_once failed: %s", self.name, self._last_error)
                continue
            for sample in samples:
                key = self._sample_to_key(sample)
                if key is None:
                    continue
                try:
                    flattened[key] = float(sample.value)
                except (TypeError, ValueError):
                    continue
        return flattened

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "qmmd_url": self._qmmd_url,
            "collectors": [
                getattr(c, "name", type(c).__name__) for c in self._collectors
            ],
            "started": self._started,
            "capabilities": dict(self._capabilities),
            "last_error": self._last_error,
            "health": bool(self._collectors),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _device_prefix(device_name: str) -> str:
        normalized = str(device_name or "").strip()
        if not normalized:
            return ""
        if normalized == "CPU":
            return "cpu"
        if normalized == "iGPU":
            return "igpu"
        if normalized == "dGPU":
            return "dgpu"
        if normalized == "NPU":
            return "npu"
        # Numeric-suffixed multi-device cases: iGPU2, dGPU2 -> dgpu_1
        m = re.fullmatch(r"(iGPU|dGPU)(\d+)", normalized)
        if m:
            kind = "igpu" if m.group(1) == "iGPU" else "dgpu"
            idx = int(m.group(2))
            # First device emits 'iGPU'/'dGPU' (no suffix); 2..N use indexed form.
            return f"{kind}_{idx - 1}" if idx >= 2 else kind
        if normalized.startswith("dGPU[") and normalized.endswith("]"):
            inside = normalized[5:-1].strip()
            if inside.isdigit():
                return f"dgpu_{inside}"
        return (
            normalized.lower()
            .replace("[", "_")
            .replace("]", "")
            .replace(" ", "_")
            .replace("-", "_")
        )

    @classmethod
    def _sample_to_key(cls, sample: MetricSample) -> str | None:
        if sample is None:
            return None
        prefix = cls._device_prefix(sample.device)
        if not prefix:
            return None
        metric = str(sample.metric_name or "")
        # MetricSamples use dotted names like 'cpu.utilization', 'gpu.power_w'.
        # The leading scope is redundant with the device prefix, so drop it.
        suffix = metric.split(".", 1)[1] if "." in metric else metric
        suffix = suffix.replace(".", "_")
        if not suffix:
            return None
        return f"{prefix}_{suffix}"
