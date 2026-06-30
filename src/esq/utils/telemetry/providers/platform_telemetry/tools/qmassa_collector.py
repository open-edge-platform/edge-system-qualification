# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import time
import http.client
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from esq.utils.telemetry.providers.platform_telemetry.base import BaseCollector
from esq.utils.telemetry.providers.platform_telemetry.tools.intel_gpu_top_utils import IntelGpuTopUtils
from esq.utils.telemetry.providers.platform_telemetry.tools.metric_sources import (
    METRIC_FREQUENCY_MHZ,
    METRIC_MEMORY_UTILIZATION,
    METRIC_POWER_W,
    METRIC_TEMPERATURE_C,
    METRIC_UTILIZATION,
    IntelGpuTopBackfillSource,
    MetricReading,
    MetricSource,
    SysfsBackfillSource,
    XpuSmiBackfillSource,
)
from esq.utils.telemetry.providers.platform_telemetry.tools.xpu_smi_utils import XpuSmiUtils
from esq.utils.telemetry.providers.platform_telemetry.models import MetricSample

# Default qmmd Prometheus endpoint (override via QMMD_URL env var or constructor)
_DEFAULT_QMMD_URL = "http://127.0.0.1:9000/metrics"

# Sentinel emitted when a telemetry source cannot be read. -1 is used so
# missing data is visible in JSON and distinguishable from a real 0
# (e.g. RC6 power-gated power_w, idle 0% utilization).
MISSING_VALUE: float = -1.0

# Bound the QMMD_URL surface to a strict allow-list. qmmd is a host-local
# Prometheus exporter; only loopback addresses are valid targets. Any
# user-supplied override that fails this check is rejected and the safe
# default is used instead.
_ALLOWED_QMMD_SCHEMES = frozenset({"http", "https"})
_ALLOWED_QMMD_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _sanitize_qmmd_url(candidate: str) -> str:
    """Return ``candidate`` if it points at loopback qmmd, else the default."""
    try:
        parsed = urllib.parse.urlparse(candidate)
    except (TypeError, ValueError):
        return _DEFAULT_QMMD_URL
    if parsed.scheme not in _ALLOWED_QMMD_SCHEMES:
        return _DEFAULT_QMMD_URL
    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_QMMD_HOSTS:
        return _DEFAULT_QMMD_URL
    try:
        port = parsed.port if parsed.port is not None else 9000
    except ValueError:
        return _DEFAULT_QMMD_URL
    if not (1 <= int(port) <= 65535):
        return _DEFAULT_QMMD_URL
    return candidate


# Prometheus line parser: metric_name{labels} value
_PROM_LINE_RE = re.compile(r'^(\w+)\{([^}]*)\}\s+([\d.eE+\-]+|NaN)')
# Label pair parser
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


class QmassaCollector(BaseCollector):
    """GPU metrics collector that scrapes the qmmd Prometheus endpoint.

    qmmd (qmassa metrics daemon) is the authoritative source for Intel GPU
    engine utilization, frequency, power, and temperature on Linux.  It uses
    the xe/i915 kernel DRM perf PMU interfaces and hwmon to gather real data.

    Fallback: if qmmd is unavailable, attempts direct sysfs reads (limited).
    """

    name = "qmassa"

    def __init__(self, devices: list[str] | None = None, qmmd_url: str | None = None):
        self._running = False
        raw_url = qmmd_url or os.environ.get("QMMD_URL", _DEFAULT_QMMD_URL)
        self._qmmd_url = _sanitize_qmmd_url(raw_url)
        # _pci_to_friendly: populated on first successful scrape
        # Maps PCI slot (e.g. "0000:03:00.0") → friendly name ("iGPU"/"dGPU")
        self._pci_to_friendly: Dict[str, str] = {}
        # device key -> integrated/discrete/unknown, parsed from qmmd_gpu_info
        self._qmmd_device_class: Dict[str, str] = {}
        # PCI slot -> DRM card path for actual card devices only.
        self._drm_cards_by_pci: Optional[Dict[str, Path]] = None
        self._qmmd_available: Optional[bool] = None  # None = not probed yet
        # Per-metric backfill sources, queried in priority order by
        # ``_supplement_qmmd_metrics_from_sysfs``. Each source exposes the
        # uniform ``MetricSource.read(pci, metric)`` API so the collector
        # no longer needs to know provider-specific call shapes.
        self._xpu_source = XpuSmiBackfillSource(XpuSmiUtils())
        self._igt_source = IntelGpuTopBackfillSource(IntelGpuTopUtils())
        self._sysfs_source = SysfsBackfillSource(
            card_resolver=lambda pci: self._discover_drm_cards().get(pci),
        )
        self._backfill_sources: List[MetricSource] = [
            self._xpu_source,
            self._igt_source,
            self._sysfs_source,
        ]

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def collect_once(self) -> List[MetricSample]:
        if not self._running:
            return []

        # Prime each backfill source so per-cycle adapters (e.g. xpu-smi)
        # snapshot their PCI-keyed maps exactly once per ``collect_once``.
        for src in self._backfill_sources:
            src.refresh()

        raw = self._fetch_qmmd()
        if raw is not None:
            self._qmmd_available = True
            return self._parse_qmmd_metrics(raw)

        # qmmd unavailable — fall back to sysfs
        self._qmmd_available = False
        return self._collect_sysfs_fallback()

    # ── qmmd scraping ────────────────────────────────────────────────────────

    def _fetch_qmmd(self) -> Optional[str]:
        try:
            parsed = urllib.parse.urlparse(self._qmmd_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                return None

            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"

            conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
            conn = conn_cls(parsed.hostname, parsed.port, timeout=3)
            try:
                conn.request("GET", path, headers={"Accept": "text/plain; version=0.0.4"})
                resp = conn.getresponse()
                if resp.status != 200:
                    return None
                return resp.read().decode("utf-8", errors="replace")
            finally:
                conn.close()
        except (http.client.HTTPException, OSError, ValueError):
            return None

    def _parse_qmmd_metrics(self, text: str) -> List[MetricSample]:
        """Parse Prometheus text from qmmd and produce MetricSamples."""
        # Phase 1: build PCI→friendly name map from qmmd_gpu_info
        self._update_pci_map(text)

        now = datetime.utcnow().isoformat() + "Z"
        # Accumulators keyed by PCI slot
        engine_util: Dict[str, float] = {}   # max engine util per device
        engine_util_origin: Dict[str, str] = {}
        freq_hz: Dict[str, float] = {}       # gt0 actual freq (Hz)
        freq_hz_origin: Dict[str, str] = {}
        power_w: Dict[str, float] = {}       # "gpu" domain power (W)
        power_pkg_w: Dict[str, float] = {}   # "package" domain power (W)
        power_origin: Dict[str, str] = {}
        temp_c: Dict[str, float] = {}        # GPU temperature in C
        temp_origin: Dict[str, str] = {}
        mem_used_bytes: Dict[str, Dict[str, float]] = {}
        mem_total_bytes: Dict[str, Dict[str, float]] = {}

        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            m = _PROM_LINE_RE.match(line)
            if not m:
                continue
            metric, labels_str, value_str = m.group(1), m.group(2), m.group(3)
            try:
                value = float(value_str)
            except ValueError:
                continue
            labels = dict(_LABEL_RE.findall(labels_str))
            device = labels.get("device", "")

            if metric == "qmmd_gpu_engine_utilization_ratio":
                # ratio 0.0–1.0; take max across all engines per device
                engine_util[device] = max(engine_util.get(device, 0.0), value)
                engine_util_origin[device] = "qmmd"

            elif metric == "qmmd_gpu_actual_frequency_hertz":
                # use gt0 (primary render GT); fall back to any freq.
                # Ignore zero readings so the supplement chain
                # (intel_gpu_top / sysfs) can backfill — qmmd reports
                # 0 Hz for i915 iGPUs when no OA stream is open.
                if value <= 0:
                    continue
                freq_id = labels.get("freq_id", "")
                if freq_id == "gt0" or device not in freq_hz:
                    freq_hz[device] = value
                    freq_hz_origin[device] = "qmmd"

            elif metric == "qmmd_gpu_power_watts":
                domain = labels.get("domain", "")
                if domain == "gpu":
                    power_w[device] = value
                    power_origin[device] = "qmmd"
                elif domain == "package":
                    power_pkg_w[device] = value
                    power_origin.setdefault(device, "qmmd")

            elif metric == "qmmd_gpu_memory_used_bytes":
                mem_type = str(labels.get("mem_type", "")).lower() or "unknown"
                mem_used_bytes.setdefault(device, {})[mem_type] = value

            elif metric == "qmmd_gpu_memory_total_bytes":
                mem_type = str(labels.get("mem_type", "")).lower() or "unknown"
                mem_total_bytes.setdefault(device, {})[mem_type] = value

            elif metric in {
                "qmmd_gpu_temperature_celsius",
                "qmmd_gpu_temperature_deg_c",
                "qmmd_gpu_temperature",
            }:
                if device not in temp_c:
                    temp_c[device] = max(0.0, value)
                    temp_origin[device] = "qmmd"

        signal_devices = set(engine_util) | set(power_w) | set(power_pkg_w) | set(temp_c)
        memory_devices = set(mem_used_bytes) | set(mem_total_bytes)
        # Ignore frequency-only ghost devices when richer telemetry exists.
        all_devices = signal_devices | memory_devices
        if not all_devices:
            all_devices = set(freq_hz)
        all_devices = self._filter_real_qmmd_devices(all_devices)

        for pci in all_devices:
            self._supplement_qmmd_metrics_from_sysfs(
                pci,
                engine_util,
                engine_util_origin,
                freq_hz,
                freq_hz_origin,
                power_w,
                power_pkg_w,
                power_origin,
                temp_c,
                temp_origin,
                bw_mb_s,
                bw_origin,
            )

        self._refresh_friendly_map_from_activity(
            all_devices,
            engine_util,
            freq_hz,
            power_w,
            power_pkg_w,
        )

        # Build samples
        samples: List[MetricSample] = []
        for pci in sorted(all_devices):
            friendly = self._pci_to_friendly.get(pci, pci)
            # Track per-metric availability. When a metric family is absent
            # from qmmd / fallback sources we deliberately DO NOT emit a
            # MetricSample for it — the downstream chart treats the missing
            # key as a gap so the user can distinguish "no data" from a
            # genuine zero reading (e.g. RC6 power-gated power_w).
            if pci in engine_util:
                util_pct = round(engine_util[pci] * 100.0, 2)
                util_origin = engine_util_origin.get(pci, "qmmd")
                util_available = True
            else:
                util_pct = None
                util_origin = "qmmd_partial"
                util_available = False

            if pci in freq_hz:
                freq_mhz = round(freq_hz[pci] / 1_000_000.0, 1)
                freq_origin = freq_hz_origin.get(pci, "qmmd")
                freq_available = True
            else:
                freq_mhz = None
                freq_origin = "qmmd_partial"
                freq_available = False
            # Prefer device-domain GPU power; fall back to package when GPU
            # domain is unavailable for the driver/device.
            if pci in power_w:
                pw_raw = power_w[pci]
                power_domain = "gpu"
                power_metric_origin = power_origin.get(pci, "qmmd")
                power_available = "true"
            elif pci in power_pkg_w:
                pw_raw = power_pkg_w[pci]
                power_domain = "package"
                power_metric_origin = power_origin.get(pci, "qmmd")
                power_available = "true"
            else:
                pw_raw = None
                power_domain = "unavailable"
                power_metric_origin = "qmmd_partial"
                power_available = "false"
            # Keep milliwatt-scale visibility for low iGPU idle power.
            pw = round(pw_raw, 3) if pw_raw is not None else None

            if pci in temp_c:
                temperature_c = round(max(0.0, temp_c[pci]), 1)
                temperature_origin = temp_origin.get(pci, "qmmd")
                temp_available = True
            else:
                if self._qmmd_device_class.get(pci) == "integrated":
                    proxy_temp, proxy_origin = SysfsBackfillSource.cpu_package_temp_proxy_c()
                    if proxy_temp is not None:
                        temperature_c = round(max(0.0, proxy_temp), 1)
                        temperature_origin = proxy_origin
                        temp_available = True
                    else:
                        temperature_c = None
                        temperature_origin = proxy_origin
                        temp_available = False
                else:
                    temperature_c = None
                    temperature_origin = "placeholder_sysfs_unavailable"
                    temp_available = False

            mem_pct, mem_origin, mem_domain, mem_available = self._memory_utilization_from_qmmd(
                pci,
                mem_used_bytes,
                mem_total_bytes,
            )
            try_sysfs_mem = mem_available != "true" or (
                mem_origin == "qmmd" and mem_domain == "smem" and mem_pct <= 0.0
            )
            if try_sysfs_mem:
                # Walk the unified backfill chain (xpu-smi → intel_gpu_top
                # → sysfs). xpu-smi readings are taken unconditionally;
                # sysfs is only accepted if it actually improves on the
                # qmmd smem=0% baseline (preserves original semantics).
                card = self._discover_drm_cards().get(pci)
                had_qmmd_zero_baseline = mem_available == "true"
                for src in self._backfill_sources:
                    r = src.read(pci, METRIC_MEMORY_UTILIZATION, card=card)
                    if r is None:
                        continue
                    if src.name == "sysfs" and had_qmmd_zero_baseline and r.value <= 0.0:
                        continue
                    mem_pct = max(0.0, min(100.0, r.value))
                    mem_origin = r.origin
                    mem_domain = r.domain or "device"
                    mem_available = "true"
                    break

            # Emit a MetricSample for every metric on every device; use
            # MISSING_VALUE when the source cannot be read. Aggregations
            # filter the sentinel out.
            samples.append(MetricSample(
                timestamp_utc=now,
                collector=self.name,
                device=friendly,
                metric_name="gpu.utilization",
                value=util_pct if (util_available and util_pct is not None) else MISSING_VALUE,
                unit="%",
                tags={"vendor": "intel", "metric_origin": util_origin, "pci": pci},
            ))
            samples.append(MetricSample(
                timestamp_utc=now,
                collector=self.name,
                device=friendly,
                metric_name="gpu.frequency_mhz",
                value=freq_mhz if (freq_available and freq_mhz is not None) else MISSING_VALUE,
                unit="MHz",
                tags={"vendor": "intel", "metric_origin": freq_origin, "pci": pci},
            ))
            samples.append(MetricSample(
                timestamp_utc=now,
                collector=self.name,
                device=friendly,
                metric_name="gpu.power_w",
                value=pw if (power_available == "true" and pw is not None) else MISSING_VALUE,
                unit="W",
                tags={
                    "vendor": "intel",
                    "metric_origin": power_metric_origin,
                    "pci": pci,
                    "power_domain": power_domain,
                    "power_available": power_available,
                },
            ))
            samples.append(MetricSample(
                timestamp_utc=now,
                collector=self.name,
                device=friendly,
                metric_name="gpu.memory_utilization",
                value=mem_pct if (mem_available == "true" and mem_pct is not None) else MISSING_VALUE,
                unit="%",
                tags={
                    "vendor": "intel",
                    "metric_origin": mem_origin,
                    "pci": pci,
                    "memory_domain": mem_domain,
                    "memory_available": mem_available,
                },
            ))
            samples.append(MetricSample(
                timestamp_utc=now,
                collector=self.name,
                device=friendly,
                metric_name="gpu.temperature_c",
                value=temperature_c if (temp_available and temperature_c is not None) else MISSING_VALUE,
                unit="\u00b0C",
                tags={"vendor": "intel", "metric_origin": temperature_origin, "pci": pci},
            ))
        return samples

    @staticmethod
    def _memory_utilization_from_qmmd(
        pci: str,
        mem_used_bytes: Dict[str, Dict[str, float]],
        mem_total_bytes: Dict[str, Dict[str, float]],
    ) -> tuple[float, str, str, str]:
        used_by_type = mem_used_bytes.get(pci, {})
        total_by_type = mem_total_bytes.get(pci, {})

        for mem_type in ("vram", "smem"):
            used = float(used_by_type.get(mem_type, 0.0))
            total = float(total_by_type.get(mem_type, 0.0))
            if total > 0.0:
                pct = max(0.0, min(100.0, (used / total) * 100.0))
                return round(pct, 2), "qmmd", mem_type, "true"

        return 0.0, "qmmd_partial", "unavailable", "false"

    def _update_pci_map(self, text: str) -> None:
        """Build (or refresh) PCI→friendly name map from qmmd_gpu_info lines."""
        self._qmmd_device_class = {}
        for line in text.splitlines():
            if not line.startswith("qmmd_gpu_info{"):
                continue
            m = _PROM_LINE_RE.match(line)
            if not m:
                continue
            labels = dict(_LABEL_RE.findall(m.group(2)))
            pci = labels.get("device", "")
            dev_type = str(labels.get("dev_type", ""))
            if not pci:
                continue
            if "Integrated" in dev_type:
                self._qmmd_device_class[pci] = "integrated"
            elif "Discrete" in dev_type:
                self._qmmd_device_class[pci] = "discrete"
            else:
                self._qmmd_device_class.setdefault(pci, "unknown")

    def _refresh_friendly_map_from_activity(
        self,
        devices: set[str],
        engine_util: Dict[str, float],
        freq_hz: Dict[str, float],
        power_w: Dict[str, float],
        power_pkg_w: Dict[str, float],
    ) -> None:
        def is_active(dev: str) -> int:
            return int(dev in engine_util or dev in freq_hz or dev in power_w or dev in power_pkg_w)

        integrated = [d for d in devices if self._qmmd_device_class.get(d) == "integrated"]
        discrete = [d for d in devices if self._qmmd_device_class.get(d) == "discrete"]

        integrated.sort(key=lambda dev: (-is_active(dev), dev))
        discrete.sort(key=lambda dev: (-is_active(dev), dev))

        for i, dev in enumerate(integrated):
            self._pci_to_friendly[dev] = "iGPU" if i == 0 else f"iGPU{i + 1}"
        for i, dev in enumerate(discrete):
            self._pci_to_friendly[dev] = "dGPU" if i == 0 else f"dGPU{i + 1}"

    def _discover_drm_cards(self) -> Dict[str, Path]:
        if self._drm_cards_by_pci is not None:
            return self._drm_cards_by_pci

        cards: Dict[str, Path] = {}
        drm_root = Path("/sys/class/drm")
        if drm_root.exists():
            for card in sorted(drm_root.iterdir()):
                if not (card.name.startswith("card") and card.name[4:].isdigit()):
                    continue
                pci = SysfsBackfillSource.card_pci_address(card)
                if pci:
                    cards[pci] = card

        self._drm_cards_by_pci = cards
        return cards

    def _filter_real_qmmd_devices(self, devices: set[str]) -> set[str]:
        if not devices:
            return devices

        drm_cards = self._discover_drm_cards()
        if not drm_cards:
            return devices

        filtered = {device for device in devices if device in drm_cards}
        return filtered or devices

    def _supplement_qmmd_metrics_from_sysfs(
        self,
        pci: str,
        engine_util: Dict[str, float],
        engine_util_origin: Dict[str, str],
        freq_hz: Dict[str, float],
        freq_hz_origin: Dict[str, str],
        power_w: Dict[str, float],
        power_pkg_w: Dict[str, float],
        power_origin: Dict[str, str],
        temp_c: Dict[str, float],
        temp_origin: Dict[str, str],
        bw_mb_s: Dict[str, float],
        bw_origin: Dict[str, str],
    ) -> None:
        """Per-metric backfill chain: qmmd → ``self._backfill_sources``.

        For each metric still missing from the qmmd snapshot we walk the
        unified source list in priority order (xpu-smi → intel_gpu_top →
        sysfs) and stop at the first source that returns a reading. All
        provider-specific call shapes live behind the ``MetricSource``
        protocol; the collector only sees ``MetricReading(value, origin)``.
        """
        card = self._discover_drm_cards().get(pci)
        if card is None:
            return

        def first_reading(metric: str):
            for src in self._backfill_sources:
                reading = src.read(pci, metric, card=card)
                if reading is not None:
                    return reading
            return None

        # ── utilization (sources return percent, stored as fraction) ─────
        # Backfill when qmmd is missing utilization, and also when qmmd
        # reports a persistent zero baseline while another source sees
        # positive activity.
        should_try_util_backfill = pci not in engine_util or engine_util.get(pci, 0.0) <= 0.0
        if should_try_util_backfill:
            r = first_reading(METRIC_UTILIZATION)
            if r is not None:
                backfill_fraction = max(0.0, min(1.0, r.value / 100.0))
                if pci not in engine_util or backfill_fraction > engine_util.get(pci, 0.0):
                    engine_util[pci] = backfill_fraction
                    engine_util_origin[pci] = r.origin

        # ── frequency (sources return MHz, stored as Hz) ─────────────────
        if pci not in freq_hz:
            r = first_reading(METRIC_FREQUENCY_MHZ)
            if r is not None:
                freq_hz[pci] = max(0.0, r.value) * 1_000_000.0
                freq_hz_origin[pci] = r.origin

        # ── power ────────────────────────────────────────────────────────
        if pci not in power_w and pci not in power_pkg_w:
            r = first_reading(METRIC_POWER_W)
            if r is not None:
                power_w[pci] = r.value
                power_origin[pci] = r.origin

        # ── temperature ──────────────────────────────────────────────────
        if pci not in temp_c:
            r = first_reading(METRIC_TEMPERATURE_C)
            if r is not None:
                temp_c[pci] = r.value
                temp_origin[pci] = r.origin

    # ── sysfs fallback ───────────────────────────────────────────────────────

    def _collect_sysfs_fallback(self) -> List[MetricSample]:
        """Best-effort metrics when qmmd is not running.

        Uses the same ``self._backfill_sources`` chain as the post-qmmd
        backfill so this path also benefits from xpu-smi / intel_gpu_top
        readings (not just raw sysfs). Each metric is emitted exactly
        once per discovered DRM card; missing readings carry
        ``MISSING_VALUE`` with an "unavailable" origin tag.
        """
        now = datetime.utcnow().isoformat() + "Z"
        samples: List[MetricSample] = []

        # Auto-discover all DRM card directories
        drm_root = Path("/sys/class/drm")
        cards = sorted(
            p for p in drm_root.iterdir()
            if p.name.startswith("card") and p.name[4:].isdigit()
        ) if drm_root.exists() else []

        # (metric_name, MetricSample unit, sample metric_name suffix, fallback origin)
        metric_specs = (
            (METRIC_UTILIZATION,    "%",    "gpu.utilization",    "placeholder_sysfs_unavailable"),
            (METRIC_FREQUENCY_MHZ,  "MHz",  "gpu.frequency_mhz",  "placeholder_sysfs_unavailable"),
            (METRIC_POWER_W,        "W",    "gpu.power_w",        "placeholder_sysfs_unavailable"),
            (METRIC_TEMPERATURE_C,  "°C",   "gpu.temperature_c",  "placeholder_sysfs_unavailable"),
        )

        def first_reading(pci: str, card: Path, metric: str):
            for src in self._backfill_sources:
                reading = src.read(pci, metric, card=card)
                if reading is not None:
                    return reading
            return None

        igpu_count = dGpu_count = 0
        for card in cards:
            is_integrated = SysfsBackfillSource.is_integrated(card)
            if is_integrated:
                igpu_count += 1
                friendly = "iGPU" if igpu_count == 1 else f"iGPU{igpu_count}"
            else:
                dGpu_count += 1
                friendly = "dGPU" if dGpu_count == 1 else f"dGPU{dGpu_count}"

            pci = SysfsBackfillSource.card_pci_address(card)

            for metric, unit, sample_name, unavailable_origin in metric_specs:
                r = first_reading(pci, card, metric)

                # Integrated GPUs on platforms without iGPU hwmon temp
                # nodes fall back to a CPU-package thermal proxy so the
                # chart shows something instead of "no data".
                if r is None and metric == METRIC_TEMPERATURE_C and is_integrated:
                    proxy_v, proxy_origin = SysfsBackfillSource.cpu_package_temp_proxy_c()
                    if proxy_v is not None:
                        r = MetricReading(proxy_v, proxy_origin)

                if r is not None:
                    value = r.value
                    origin = r.origin
                else:
                    value = MISSING_VALUE
                    origin = unavailable_origin

                samples.append(MetricSample(
                    timestamp_utc=now,
                    collector=self.name,
                    device=friendly,
                    metric_name=sample_name,
                    value=value,
                    unit=unit,
                    tags={"vendor": "intel", "metric_origin": origin, "card": card.name, "pci": pci or ""},
                ))
        return samples
