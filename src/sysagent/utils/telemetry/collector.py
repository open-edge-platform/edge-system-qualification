# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Telemetry collector orchestrator.

Manages multiple telemetry modules running in a single background thread.
The collector is driven by profile YAML configuration and integrates with
the Result data structure for automatic metric and sample attachment.
"""

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


def _parse_module_configs(telemetry_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse the ``telemetry.modules`` list from profile YAML.

    Each entry may be a dict with keys: name, enabled, metrics, thresholds.
    Returns a list of parsed module config dicts (only enabled entries).
    """
    raw_modules = telemetry_cfg.get("modules", [])
    parsed = []
    for entry in raw_modules:
        if not isinstance(entry, dict):
            continue
        if not entry.get("enabled", True):
            continue
        parsed.append(
            {
                "name": entry.get("name", ""),
                "metrics": entry.get("metrics", []) or [],
                "thresholds": entry.get("thresholds", {}) or {},
                "chart_type": entry.get("chart_type") or None,
                "title": entry.get("title", {}) or {},
                "scales": entry.get("scales", {}) or {},
                "axes": entry.get("axes", []) or [],
            }
        )
    return parsed


class TelemetryCollector:
    """
    Orchestrates one or more telemetry modules in a daemon background thread.

    Configuration is read from the ``telemetry`` key of a test's ``configs`` dict,
    which is populated from the profile YAML:

    .. code-block:: yaml

        telemetry:
          enabled: true
          interval: 10         # seconds between samples
          modules:
            - name: cpu_freq
              enabled: true
              metrics: []      # empty = all metrics
              thresholds:
                current_mhz:
                  warning: 4500
            - name: cpu_usage
              enabled: true
            - name: memory_usage
              enabled: true
              # axes: dual y-axis config
              # Metrics with incompatible units (e.g. % vs GB) can be split
              # across a left axis ("y") and a right axis ("y1") so that each
              # unit range is independently scaled on the chart.
              axes:
                - id: "y"
                  position: "left"
                  metrics: [used_percent]
                  label: "Usage (%)"
                - id: "y1"
                  position: "right"
                  metrics: [available_gib, used_gib]
                  label: "Memory (GB)"
            - name: package_power   # registered by esq package
              enabled: true
              thresholds:
                package_power_w:
                  warning: 150

    After ``stop()`` is called the collector stores the final summary and merges
    it into a ``Result`` object via ``apply_to_result()``.  All telemetry data
    is written exclusively to ``result.extended_metadata["telemetry"]``; nothing
    is added to ``result.metadata``.
    """

    def __init__(self, configs: Dict[str, Any]) -> None:
        """
        Initialise the collector from a test ``configs`` dict.

        Args:
            configs: Full test configuration dict (from profile YAML).
        """
        telemetry_cfg: Dict[str, Any] = configs.get("telemetry") or {}
        self.enabled: bool = bool(telemetry_cfg.get("enabled", False))
        self.interval: int = int(telemetry_cfg.get("interval", 10))

        # Allow env-var override of the configured interval.
        # CORE_TELEMETRY_INTERVAL takes precedence over the profile YAML value,
        # enabling per-run overrides without modifying any profile file.
        # CLI option --telemetry-interval sets this env var; it can also be set directly.
        # Priority: CLI option (highest) > CORE_TELEMETRY_INTERVAL env var > profile YAML (lowest).
        _env_interval = os.environ.get("CORE_TELEMETRY_INTERVAL", "").strip()
        if _env_interval:
            try:
                _override = int(_env_interval)
                if _override >= 1:
                    logger.debug(
                        "Telemetry interval overridden by CORE_TELEMETRY_INTERVAL: %ds (profile configured: %ds)",
                        _override,
                        self.interval,
                    )
                    self.interval = _override
                else:
                    logger.warning("CORE_TELEMETRY_INTERVAL='%s' must be >= 1; ignoring override.", _env_interval)
            except ValueError:
                logger.warning("CORE_TELEMETRY_INTERVAL='%s' is not a valid integer; ignoring override.", _env_interval)

        self._modules: List[Any] = []  # List[BaseTelemetryModule]
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._summary: Optional[Dict[str, Any]] = None

        if self.enabled:
            self._load_modules(telemetry_cfg)

    # ------------------------------------------------------------------
    # Module loading
    # ------------------------------------------------------------------

    def _load_modules(self, telemetry_cfg: Dict[str, Any]) -> None:
        """Instantiate and validate requested telemetry modules."""
        # Ensure core modules are registered (idempotent)
        try:
            import sysagent.utils.telemetry.modules  # noqa: F401 – triggers registration
        except Exception as exc:
            logger.debug("Could not import core telemetry modules: %s", exc)

        # Discover extension telemetry modules via entry points (installation-method-agnostic).
        # Packages that contribute telemetry modules declare:
        #   [project.entry-points.sysagent]
        #   sysagent_telemetry = "my_package.utils.telemetry.modules"
        # The entry point module must expose get_telemetry_modules() -> {name: class}.
        # The collector calls this function explicitly rather than relying on import
        # side-effects, which can be skipped when the module is already cached in
        # sys.modules (e.g. the config loader also iterates the sysagent entry-point
        # group and may import the same module before the collector runs).
        try:
            import importlib
            import importlib.metadata

            eps = importlib.metadata.entry_points()
            telemetry_eps = eps.select(group="sysagent") if hasattr(eps, "select") else eps.get("sysagent", [])
            for ep in telemetry_eps:
                if ep.name == "sysagent_telemetry":
                    try:
                        mod = importlib.import_module(ep.value)
                        logger.debug("Loaded telemetry modules from entry point: %s -> %s", ep.name, ep.value)
                        # Prefer the explicit function-based API (get_telemetry_modules)
                        # over import side-effects — immune to sys.modules caching issues.
                        if hasattr(mod, "get_telemetry_modules"):
                            from sysagent.utils.telemetry import registry as _tel_registry

                            for _mod_name, _mod_cls in mod.get_telemetry_modules().items():
                                _tel_registry.register(_mod_name, _mod_cls)
                    except Exception as exc:
                        logger.debug("Failed to load telemetry entry point '%s': %s", ep.value, exc)
        except Exception as exc:
            logger.debug("Could not discover telemetry entry points: %s", exc)

        from sysagent.utils.telemetry.base import TelemetryConfig
        from sysagent.utils.telemetry.registry import get as registry_get

        module_entries = _parse_module_configs(telemetry_cfg)
        if not module_entries:
            # Fall back: enable all registered modules with defaults
            from sysagent.utils.telemetry.registry import list_modules

            for name, cls in list_modules().items():
                cfg = TelemetryConfig(enabled=True, interval=self.interval)
                self._try_add_module(name, cls, cfg)
        else:
            for entry in module_entries:
                name = entry["name"]
                cls = registry_get(name)
                if cls is None:
                    logger.debug("Telemetry module '%s' not found in registry (package not imported?). Skipping.", name)
                    continue
                cfg = TelemetryConfig(
                    enabled=True,
                    interval=self.interval,
                    metrics=entry["metrics"],
                    thresholds=entry["thresholds"],
                    chart_type=entry["chart_type"],
                    title=entry["title"],
                    scales=entry["scales"],
                    axes=entry["axes"],
                )
                self._try_add_module(name, cls, cfg)

    def _try_add_module(self, name: str, cls: Type[Any], cfg: Any) -> None:
        """Instantiate a module and add it if it is available on this system."""
        try:
            instance = cls(cfg)
            instance._apply_default_config()  # merge module-level defaults with profile config
            if instance.is_available():
                self._modules.append(instance)
                logger.debug("Telemetry module loaded: %s", name)
            else:
                logger.debug("Telemetry module '%s' not available on this system; skipping.", name)
        except Exception as exc:
            logger.debug("Failed to load telemetry module '%s': %s", name, exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background collection thread (no-op if disabled or no modules)."""
        if not self.enabled or not self._modules:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._collect_loop,
            name="TelemetryCollector",
            daemon=True,
        )
        self._thread.start()
        logger.debug(
            "Telemetry collection started (interval=%.1fs, modules=%s)",
            self.interval,
            [m.module_name for m in self._modules],
        )

    def stop(self) -> None:
        """
        Stop the background thread and compute the final summary.

        Safe to call multiple times — subsequent calls after the first are no-ops.
        Safe to call even if the collector was never started.
        """
        if self._stop_event.is_set():
            return  # Already stopped; nothing to do
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval * 2, 5))
            self._thread = None
        self._summary = self._build_summary()
        logger.debug("Telemetry collection stopped; %d module(s) summarised.", len(self._modules))

    def _collect_one(self) -> None:
        """Collect a single sample from every module (called from background thread or stop)."""
        for module in self._modules:
            try:
                sample = module.collect_sample()
                if not sample.values:
                    # Delta-based modules (gpu_usage, gpu_power, etc.) return an
                    # empty sample on the very first call because they need two
                    # readings to compute a delta.  Transient sysfs read failures
                    # also produce empty samples.  In both cases there is nothing
                    # useful to store — skip rather than recording empty entries.
                    logger.debug(
                        "[Telemetry] %s (sample #%d): empty — skipping",
                        module.module_name,
                        len(module._samples) + 1,
                    )
                    continue
                module._samples.append(sample)
                values_str = "  ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in sample.values.items()
                )
                logger.debug(
                    "[Telemetry] %s (sample #%d): %s",
                    module.module_name,
                    len(module._samples),
                    values_str,
                )
            except Exception as exc:
                logger.debug("Error collecting telemetry from %s: %s", module.module_name, exc)

    def _collect_loop(self) -> None:
        """Background thread: collect an immediate sample at start, then one every ``interval`` seconds."""
        # Collect at t=0 so the test ramp-up phase is captured
        self._collect_one()
        while not self._stop_event.wait(self.interval):
            self._collect_one()

    # ------------------------------------------------------------------
    # Summary & result integration
    # ------------------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        """Build aggregated summary across all modules."""
        if not self._modules:
            return {}

        per_module: Dict[str, Any] = {}

        for module in self._modules:
            per_module[module.module_name] = module.get_summary()

        return {
            "enabled": True,
            "interval_s": self.interval,
            "modules": per_module,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Return the telemetry summary.

        If the collector has been stopped, returns the finalised summary.
        If still running, builds a snapshot from samples collected so far
        (without stopping the background thread).

        Returns an empty dict if telemetry is disabled or no modules are loaded.
        """
        if self._summary is not None:
            return self._summary
        # Collector still running — return a live snapshot
        return self._build_summary()

    def apply_to_result(self, result: Any) -> None:
        """
        Merge telemetry data into a ``Result`` object.

        Stops collection if the background thread is still running so that
        the full set of samples is captured before computing averages.
        Safe to call multiple times (subsequent calls are no-ops after the
        first stop).

        ``result.extended_metadata["telemetry"]`` receives the full summary
        (per-module averages, min/max, and raw sample time-series).
        No telemetry keys are written to ``result.metadata``; all telemetry
        data lives exclusively in ``extended_metadata``.

        Args:
            result: A ``sysagent.utils.core.Result`` instance (or any object with
                    ``metadata`` and ``extended_metadata`` dicts).
        """
        # Ensure collection is finalised before summarising
        if not self._stop_event.is_set():
            self.stop()

        summary = self.get_summary()
        if not summary:
            return

        # Store full summary in extended_metadata only
        result.extended_metadata["telemetry"] = summary

        logger.debug(
            "Telemetry data applied to result (%d averaged metrics)",
            len(summary.get("combined_averages", {})),
        )
