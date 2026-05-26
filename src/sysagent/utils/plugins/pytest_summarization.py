# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test result summarization utilities for core testing framework.

This module provides fixtures and functions for summarizing test results
with standardized charts and reporting.

Now supports Result dataclass structure:
- Renders a table for 'data' (metric, value, unit)
- Renders a table for 'parameters' (key, value)
- Renders a table for 'kpis' (only for KPIs with validation enabled)
"""

import json
import logging
import re
from typing import Any, Dict, Optional

import allure
import pytest

from sysagent.utils.core import Result
from sysagent.utils.plugins.pytest_telemetry import get_telemetry_collector
from sysagent.utils.telemetry.kpi_correlation import build_kpi_correlation

logger = logging.getLogger(__name__)


def _infer_device_from_metric(metric_name: str) -> str:
    """Infer a coarse compute-device label from normalized telemetry metric key."""
    tokens = metric_name.lower().split("_")
    for token in tokens:
        if token in {"cpu", "gpu", "igpu", "dgpu", "npu"}:
            return token
    if "mem" in tokens:
        return "memory"
    if "temp" in tokens:
        return "thermal"
    return "system"


def _to_header_unit(metric_name: str, scales: Dict[str, Any]) -> str:
    unit = ((scales or {}).get(metric_name) or {}).get("unit")
    return str(unit) if unit else "value"


def _build_telemetry_metric_csvs(results_dict: Dict[str, Any]) -> Dict[str, str]:
    """
    Build per-metric CSV tables from telemetry summary.

    Each CSV has one row per inferred device and columns:
      device, avg(<unit>), min(<unit>), max(<unit>)
    """
    csv_outputs: Dict[str, str] = {}
    telemetry = (results_dict.get("extended_metadata") or {}).get("telemetry") or {}
    modules = telemetry.get("modules") or {}
    if not isinstance(modules, dict):
        return csv_outputs

    # metric_name -> device -> {avg,min,max,unit}
    tables: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _iter_leaf_modules(mods):
        """Yield ``(name, data)`` pairs flattening ``device_groups`` if present.

        Modules whose summary nests per-device entries under ``device_groups``
        (preserving the original module file name as the parent key) need
        each device flattened back to a metric source for CSV aggregation.
        """
        for name, data in mods.items():
            if not isinstance(data, dict):
                continue
            nested = data.get("device_groups")
            if isinstance(nested, list) and nested:
                for sub in nested:
                    if isinstance(sub, dict):
                        yield str(sub.get("module") or name), sub
                continue
            yield str(name), data

    for module_name, module_data in _iter_leaf_modules(modules):
        averages = module_data.get("averages") or {}
        min_max = module_data.get("min_max") or {}
        scales = ((module_data.get("configs") or {}).get("scales") or {})
        if not isinstance(averages, dict) or not isinstance(min_max, dict):
            continue

        for metric_name, avg_val in averages.items():
            if not isinstance(avg_val, (int, float)):
                continue
            # Skip the -1 (MISSING_VALUE) sentinel so unavailable metrics
            # don't appear as "-1.0000" rows in the CSV.
            if float(avg_val) == -1.0:
                continue
            mm = min_max.get(metric_name) or {}
            min_val = mm.get("min")
            max_val = mm.get("max")
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                continue
            if float(min_val) == -1.0 and float(max_val) == -1.0:
                continue

            device = _infer_device_from_metric(str(metric_name))
            unit = _to_header_unit(str(metric_name), scales if isinstance(scales, dict) else {})

            metric_table = tables.setdefault(str(metric_name), {})
            metric_table[device] = {
                "module": str(module_name),
                "avg": float(avg_val),
                "min": float(min_val),
                "max": float(max_val),
                "unit": unit,
            }

    # Ordering for table rows: System first, then CPU, iGPU, dGPU(idx), NPU, others.
    _DEVICE_PRIORITY = {"system": 0, "cpu": 1, "igpu": 2, "dgpu": 3, "gpu": 3, "npu": 4}

    def _device_order_key(dev: str) -> tuple[int, int, str]:
        low = dev.lower()
        base = low.split("[", 1)[0]
        rank = _DEVICE_PRIORITY.get(base, 9)
        idx = 0
        if "[" in low and low.endswith("]"):
            try:
                idx = int(low.split("[", 1)[1][:-1])
            except ValueError:
                idx = 0
        return (rank, idx, low)

    for metric_name, per_device in tables.items():
        # Stable ordering: device priority (System, CPU, iGPU, dGPU, NPU) then name.
        devices = sorted(per_device.keys(), key=_device_order_key)
        sample = per_device[devices[0]] if devices else {"unit": "value"}
        unit = sample.get("unit", "value")
        lines = [
            f"device,avg ({unit}),min ({unit}),max ({unit}),module",
        ]
        for device in devices:
            row = per_device[device]
            lines.append(
                f"{device},{row['avg']:.4f},{row['min']:.4f},{row['max']:.4f},{row['module']}"
            )

        safe_metric = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric_name).strip("_")
        csv_outputs[f"Telemetry CSV - {safe_metric}"] = "\n".join(lines)

    return csv_outputs


def _merge_idle_baseline(results, request_node) -> None:
    """Attach the optional pre-run idle baseline to each telemetry module as ``module["baseline"]``."""
    baseline = getattr(request_node, "_sysagent_telemetry_baseline", None)
    if not baseline:
        return
    try:
        telemetry = results.extended_metadata.get("telemetry") or {}
    except AttributeError:
        return
    if not telemetry:
        return
    modules = telemetry.setdefault("modules", {})

    def _find_target(module_name: str):
        """Locate a baseline target by name in either layout.

        Returns the module dict (or a nested ``device_groups`` entry) whose
        identity matches ``module_name``. The new layout keeps the parent
        module key matching the source file (``platform_telemetry``) and nests
        per-device summaries under ``device_groups``; the old layout fanned
        each virtual name (``platform_telemetry_cpu`` …) out as a top-level key.
        """
        direct = modules.get(module_name)
        if direct is not None:
            return direct
        for parent in modules.values():
            if not isinstance(parent, dict):
                continue
            nested = parent.get("device_groups")
            if not isinstance(nested, list):
                continue
            for sub in nested:
                if isinstance(sub, dict) and str(sub.get("module") or "") == module_name:
                    return sub
        return None

    for module_name, module_baseline in baseline.items():
        metrics = module_baseline.get("metrics") or {}
        if not metrics:
            continue
        target = _find_target(module_name)
        if target is None:
            continue
        target["baseline"] = {
            "device_name": module_baseline.get("device_name"),
            "duration_s": module_baseline.get("duration_s"),
            "interval_s": module_baseline.get("interval_s"),
            "sample_count": module_baseline.get("sample_count", 0),
            "metrics": metrics,
        }


def _attach_kpi_correlation(results, configs: Optional[Dict[str, Any]] = None) -> None:
    """Compute and attach the KPI ↔ telemetry correlation block at ``telemetry.kpi_correlation``.

    Honours the profile-level toggle ``telemetry.kpi_correlation`` (boolean,
    default ``True``). Profiles that do not need the cost-per-KPI section
    can set ``params.telemetry.kpi_correlation: false`` to suppress it.
    """
    if configs is not None:
        telemetry_cfg = (configs.get("telemetry") or {})
        if telemetry_cfg.get("kpi_correlation", True) is False:
            return
    try:
        telemetry = (results.extended_metadata or {}).get("telemetry") or {}
    except AttributeError:
        return
    if not telemetry:
        return
    correlation = build_kpi_correlation(results, telemetry)
    if correlation:
        telemetry["kpi_correlation"] = correlation


@pytest.fixture
def summarize_test_results(request):
    """
    Fixture that returns a function to summarize test results.

    This fixture standardizes test result summarization and Allure reporting
    for tests using the Result structure.
    - Renders a table for 'data' (metric, value, unit)
    - Renders a table for 'parameters' (key, value)
    - Attaches raw result as JSON
    - Optionally visualizes iteration data if provided

    Usage in tests:
        summarize_test_results(
            results=results,  # Result or dict
            test_name=test_name,
            iteration_data=iteration_data,  # Optional
            configs=configs,
            get_kpi_config=get_kpi_config
        )

    Returns:
        A function that performs test result summarization and creates appropriate
        Allure report steps
    """
    logger = logging.getLogger(__name__)

    def _summarize_results(
        results: Result,
        configs: Optional[Dict[str, Any]] = None,
        get_kpi_config: Optional[callable] = None,
        test_name: str = "Unknown",
        iteration_data: Optional[Dict[str, Any]] = None,
        enable_visualizations: bool = False,
    ) -> None:
        logger.info(f"Generating test result summary for test: {test_name}")
        results_dict = results.to_dict()

        # Apply live telemetry to the result. On cache-hit, restore the cached
        # telemetry stash instead (a fresh apply would overwrite it with the
        # short cache-hit-overhead samples).
        try:
            telemetry_collector = get_telemetry_collector(request.node)
            # Treat presence of the cache stash as the authoritative cache-hit
            # signal; the boolean flag is unreliable across fixture invocations.
            _stashed_telemetry = getattr(request.node, "_cached_telemetry", None)
            result_from_cache = bool(_stashed_telemetry) or getattr(
                request.node, "_result_from_cache", False
            )
            if telemetry_collector is not None and not result_from_cache:
                telemetry_collector.apply_to_result(results)
                _merge_idle_baseline(results, request.node)
                _attach_kpi_correlation(results, configs)
                results_dict = results.to_dict()
            elif result_from_cache:
                if _stashed_telemetry:
                    results.extended_metadata["telemetry"] = _stashed_telemetry
                _attach_kpi_correlation(results, configs)
                results_dict = results.to_dict()
        except Exception as _tel_exc:
            logger.debug("Could not apply telemetry to result: %s", _tel_exc)

        # Update result name
        if configs:
            test_id = configs.get("test_id", "T0000")
            display_name = configs.get("display_name", test_name)
            results_dict["name"] = f"{test_id} - {display_name}"

        visualization_enabled = bool(enable_visualizations)
        plt_module = None
        create_results_table = None
        create_time_series_chart = None

        if visualization_enabled:
            try:
                import matplotlib.pyplot as plt_module  # type: ignore

                from sysagent.utils.reporting.visualization import (
                    create_results_table as _create_results_table,
                )
                from sysagent.utils.reporting.visualization import (
                    create_time_series_chart as _create_time_series_chart,
                )

                create_results_table = _create_results_table
                create_time_series_chart = _create_time_series_chart
            except ImportError as exc:
                logger.error(
                    "Visualization utilities not available for test summarization: %s",
                    exc,
                )
                visualization_enabled = False
                plt_module = None
            except Exception as exc:  # pragma: no cover - unexpected import failure
                logger.error("Failed to initialize visualization utilities: %s", exc)
                visualization_enabled = False
                plt_module = None

        try:
            with allure.step("Summarize test results"):
                logger.debug(f"Generating detailed test result summary for test: {test_name}")

                # Skip summarization if no results available
                if not results_dict:
                    logger.warning("No test results available to summarize")
                    allure.attach(
                        json.dumps(
                            {"skipped_reason": "No test results to summarize"},
                            indent=2,
                        ),
                        name="Summary Skipped",
                        attachment_type=allure.attachment_type.JSON,
                    )
                    logger.info("Skipped test results summary - no data available")
                    return
                else:
                    allure.attach(
                        json.dumps(results_dict, indent=2),
                        name="Core Metrics Test Results",
                        attachment_type=allure.attachment_type.JSON,
                    )

                    # Telemetry CSV attachments are intentionally disabled.
                    # The report now presents telemetry in the dedicated Telemetry section,
                    # and these per-metric CSV attachments create duplicate tables outside it.

                if iteration_data:
                    allure.attach(
                        json.dumps(iteration_data, indent=2),
                        name="Iteration Data",
                        attachment_type=allure.attachment_type.JSON,
                    )

                if visualization_enabled and create_results_table and plt_module:
                    with allure.step("Generate test results table"):
                        try:
                            metrics = results_dict.get("metrics", {})
                            if metrics:
                                display_name = results_dict.get("parameters", {}).get("Display Name", test_name)
                                image_bytes, fig = create_results_table(
                                    metrics,
                                    title=f"{display_name} - Metrics",
                                    columns=["Metric", "Value", "Unit"],
                                )
                                allure.attach(
                                    image_bytes,
                                    name="Metrics Table",
                                    attachment_type=allure.attachment_type.PNG,
                                )
                                plt_module.close(fig)
                            else:
                                logger.warning("No metrics section found in results for metrics table.")

                        except Exception as exc:
                            logger.error(
                                "Error generating test results tables for summary: %s",
                                exc,
                            )
                            allure.attach(
                                json.dumps(
                                    {"error": str(exc), "results": results_dict},
                                    indent=2,
                                ),
                                name="Results Summary Error",
                                attachment_type=allure.attachment_type.JSON,
                            )
                else:
                    logger.debug("Visualization disabled; skipping metrics table generation")

                if visualization_enabled and create_time_series_chart and iteration_data:
                    with allure.step("Generate iteration trend charts"):
                        try:
                            chart_path = create_time_series_chart(iteration_data, test_name)
                            if chart_path:
                                with open(chart_path, "rb") as chart_file:
                                    allure.attach(
                                        chart_file.read(),
                                        name=f"{test_name} Performance Chart",
                                        attachment_type=allure.attachment_type.PNG,
                                    )
                                logger.debug("Added time series chart from %s", chart_path)
                            else:
                                logger.warning("Failed to create time series chart")
                        except Exception as exc:
                            logger.error(
                                "Error generating iteration data for summary: %s",
                                exc,
                            )
                            allure.attach(
                                json.dumps(
                                    {"error": str(exc), "iteration_data": iteration_data},
                                    indent=2,
                                ),
                                name="Iteration Summary Error",
                                attachment_type=allure.attachment_type.JSON,
                            )
                elif iteration_data:
                    logger.debug("Visualization disabled; skipping iteration chart generation")

                logger.info("Test result summary completed successfully")

        except Exception as exc:  # pragma: no cover - unexpected summarization failure
            logger.error("Unexpected error in test result summarization: %s", exc)

    return _summarize_results
