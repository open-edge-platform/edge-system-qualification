# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""InfluxDB metric extraction helpers for timeseries suites."""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sysagent.utils.core import run_command

logger = logging.getLogger(__name__)

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")

_THROUGHPUT_RATE_HINTS = ("throughput", "per_second", "fps", "rate")
_THROUGHPUT_COUNTER_HINTS = ("processed_data_points", "processed_points", "total", "count")
_INVALID_THROUGHPUT_HINTS = (
    "time",
    "timestamp",
    "quality",
    "latency",
    "status",
    "error",
    "alarm",
)


def _validate_identifier(name: str, label: str) -> str:
    if not _SAFE_IDENTIFIER.fullmatch(name):
        raise ValueError(f"Invalid {label}: {name}")
    return name


def _extract_json_payload(text: str) -> Dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end < start:
        raise ValueError("No JSON payload found in influx output")
    return json.loads(text[start : end + 1])


def _is_invalid_throughput_field(field_name: str) -> bool:
    lowered = str(field_name).lower()
    return any(token in lowered for token in _INVALID_THROUGHPUT_HINTS)


def _select_throughput_field(fields: List[str], explicit_field: Optional[str]) -> Tuple[Optional[str], str]:
    if explicit_field and explicit_field in fields:
        return explicit_field, "explicit"

    for field in fields:
        lowered = field.lower()
        if any(token in lowered for token in _THROUGHPUT_RATE_HINTS):
            return field, "rate_like"

    for field in fields:
        lowered = field.lower()
        if any(token in lowered for token in _THROUGHPUT_COUNTER_HINTS):
            return field, "counter_like"

    for field in fields:
        if not _is_invalid_throughput_field(field):
            return field, "fallback_non_time"

    return None, "no_valid_field"


def _first_numeric_value(result_payload: Dict) -> Optional[float]:
    results = result_payload.get("results", [])
    if not results:
        return None

    series = results[0].get("series", [])
    if not series:
        return None

    values = series[0].get("values", [])
    if not values:
        return None

    row = values[0]
    if not isinstance(row, list):
        return None

    columns = series[0].get("columns", [])

    # Prefer numeric values from non-time columns to avoid selecting epoch timestamps.
    if isinstance(columns, list) and len(columns) == len(row):
        for column_name, value in zip(columns, row):
            if str(column_name).lower() == "time":
                continue
            if isinstance(value, (int, float)):
                return float(value)

    # Fallback for payloads without columns metadata.
    for value in row:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _first_time_value(result_payload: Dict) -> Optional[float]:
    """Return the epoch-seconds for the first row time column, if available."""
    results = result_payload.get("results", [])
    if not results:
        return None

    series = results[0].get("series", [])
    if not series:
        return None

    values = series[0].get("values", [])
    if not values:
        return None

    row = values[0]
    if not isinstance(row, list) or not row:
        return None

    columns = series[0].get("columns", [])
    time_index = 0
    if isinstance(columns, list) and columns:
        try:
            time_index = columns.index("time")
        except ValueError:
            time_index = 0

    if time_index >= len(row):
        return None

    time_value = row[time_index]
    if isinstance(time_value, (int, float)):
        # Influx may return numeric epoch in ns/us/ms depending on precision.
        # Normalize to epoch seconds.
        numeric = float(time_value)
        if numeric > 1e17:
            return numeric / 1_000_000_000.0
        if numeric > 1e14:
            return numeric / 1_000_000.0
        if numeric > 1e11:
            return numeric / 1_000.0
        return numeric

    if isinstance(time_value, str):
        normalized = time_value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None

    return None


def _run_influx_query(
    query: str,
    username: str,
    password: str,
    database: Optional[str],
    container_name: str,
    timeout: int,
) -> Dict:
    cmd = [
        "docker",
        "exec",
        container_name,
        "influx",
        "-username",
        username,
        "-password",
        password,
        "-execute",
        query,
        "-format",
        "json",
    ]
    if database:
        cmd[6:6] = ["-database", database]
    result = run_command(cmd, timeout=timeout)
    if not result.success and "No such container" in (result.stderr or result.stdout):
        # Fallback for cases where the service container is running with a different container name.
        fallback_cmd = [
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.service={container_name}",
            "--format",
            "{{.ID}}",
        ]
        fallback = run_command(fallback_cmd, timeout=30)
        fallback_id = ""
        if fallback.success:
            fallback_id = next((line.strip() for line in fallback.stdout.splitlines() if line.strip()), "")

        if fallback_id:
            cmd[2] = fallback_id
            result = run_command(cmd, timeout=timeout)

    if not result.success:
        raise RuntimeError(result.stderr or result.stdout)
    return _extract_json_payload(result.stdout)


def list_databases(
    username: str,
    password: str,
    container_name: str = "ia-influxdb",
    timeout: int = 120,
) -> List[str]:
    payload = _run_influx_query(
        query="SHOW DATABASES",
        username=username,
        password=password,
        database=None,
        container_name=container_name,
        timeout=timeout,
    )

    databases: List[str] = []
    try:
        series = payload.get("results", [])[0].get("series", [])
        if series:
            for row in series[0].get("values", []):
                if isinstance(row, list) and row:
                    db_name = str(row[0])
                    if db_name and db_name not in {"_internal"}:
                        databases.append(db_name)
    except Exception:
        pass

    return databases


def list_measurements(
    username: str,
    password: str,
    database: str = "datain",
    container_name: str = "ia-influxdb",
    timeout: int = 120,
) -> List[str]:
    payload = _run_influx_query(
        query="SHOW MEASUREMENTS",
        username=username,
        password=password,
        database=database,
        container_name=container_name,
        timeout=timeout,
    )

    measurements: List[str] = []
    try:
        series = payload.get("results", [])[0].get("series", [])
        if series:
            for row in series[0].get("values", []):
                if isinstance(row, list) and row:
                    name = str(row[0])
                    if name:
                        measurements.append(name)
    except Exception:
        pass

    return measurements


def list_field_keys(
    measurement: str,
    username: str,
    password: str,
    database: str = "datain",
    container_name: str = "ia-influxdb",
    timeout: int = 120,
) -> List[str]:
    safe_measurement = _validate_identifier(measurement, "measurement")
    payload = _run_influx_query(
        query=f'SHOW FIELD KEYS FROM "{safe_measurement}"',
        username=username,
        password=password,
        database=database,
        container_name=container_name,
        timeout=timeout,
    )

    fields: List[str] = []
    try:
        series = payload.get("results", [])[0].get("series", [])
        if series:
            for row in series[0].get("values", []):
                if isinstance(row, list) and row:
                    field_name = str(row[0])
                    if field_name:
                        fields.append(field_name)
    except Exception:
        pass

    return fields


def resolve_measurement(
    preferred_measurement: Optional[str],
    username: str,
    password: str,
    database: str = "datain",
    container_name: str = "ia-influxdb",
    timeout: int = 120,
) -> Optional[str]:
    measurements = list_measurements(
        username=username,
        password=password,
        database=database,
        container_name=container_name,
        timeout=timeout,
    )
    if not measurements:
        return None

    if preferred_measurement and preferred_measurement in measurements:
        return preferred_measurement

    anomaly_candidates = [m for m in measurements if "anomaly" in m.lower()]
    if anomaly_candidates:
        return anomaly_candidates[0]

    return measurements[0]


def compute_processed_points_latency_from_influx(
    username: str,
    password: str,
    processed_measurement: str = "wind-turbine-anomaly-data",
    processed_field: str = "wind_speed",
    latency_field: str = "end_end_time",
    latency_scale: float = 1_000_000_000.0,
    throughput_window_seconds: int = 180,
    since_time_epoch_seconds: Optional[float] = None,
    database: str = "datain",
    container_name: str = "ia-influxdb",
    timeout: int = 120,
) -> Tuple[Optional[float], Optional[float], Dict[str, object]]:
    """Extract strict app KPI semantics from processed anomaly measurement.

    Throughput: processed points/sec derived from COUNT(wind_speed) over point time span
    Latency: MEAN(end_end_time) / 1e9 seconds
    """
    safe_measurement = _validate_identifier(processed_measurement, "measurement")
    safe_processed_field = _validate_identifier(processed_field, "processed field")
    safe_latency_field = _validate_identifier(latency_field, "latency field")

    time_filter = ""
    if since_time_epoch_seconds is not None:
        since_time_ns = int(float(since_time_epoch_seconds) * 1_000_000_000.0)
        time_filter = f" WHERE time > {since_time_ns}"
    elif throughput_window_seconds and int(throughput_window_seconds) > 0:
        time_filter = f" WHERE time > now() - {max(1, int(throughput_window_seconds))}s"

    candidate_databases: List[str] = [database] if database else []
    try:
        discovered_databases = list_databases(
            username=username,
            password=password,
            container_name=container_name,
            timeout=timeout,
        )
        for discovered_db in discovered_databases:
            if discovered_db and discovered_db not in candidate_databases:
                candidate_databases.append(discovered_db)
    except Exception as discovery_error:
        logger.debug("Failed to discover Influx databases: %s", discovery_error)

    # Strict processed KPI extraction should only search data databases.
    # The telegraf database is typically for agent/internal metrics and can
    # produce misleading terminal errors for processed measurements.
    for fallback_db in ["dataout", "datain"]:
        if fallback_db not in candidate_databases:
            candidate_databases.append(fallback_db)

    last_error: Optional[str] = None
    per_database_errors: Dict[str, str] = {}
    for candidate_db in candidate_databases:
        try:
            measurements = list_measurements(
                username=username,
                password=password,
                database=candidate_db,
                container_name=container_name,
                timeout=timeout,
            )
            if safe_measurement not in measurements:
                last_error = (
                    f"Measurement '{safe_measurement}' not found in database '{candidate_db}'"
                )
                per_database_errors[candidate_db] = last_error
                continue

            fields = list_field_keys(
                measurement=safe_measurement,
                username=username,
                password=password,
                database=candidate_db,
                container_name=container_name,
                timeout=timeout,
            )
            if safe_processed_field not in fields:
                last_error = (
                    f"Field '{safe_processed_field}' missing in measurement '{safe_measurement}' "
                    f"for database '{candidate_db}'"
                )
                per_database_errors[candidate_db] = last_error
                continue
            if safe_latency_field not in fields:
                last_error = (
                    f"Field '{safe_latency_field}' missing in measurement '{safe_measurement}' "
                    f"for database '{candidate_db}'"
                )
                per_database_errors[candidate_db] = last_error
                continue

            count_payload = _run_influx_query(
                query=(
                    f'SELECT COUNT("{safe_processed_field}") FROM "{safe_measurement}"'
                    f"{time_filter}"
                ),
                username=username,
                password=password,
                database=candidate_db,
                container_name=container_name,
                timeout=timeout,
            )
            count_value = _first_numeric_value(count_payload)

            first_payload = _run_influx_query(
                query=(
                    f'SELECT FIRST("{safe_processed_field}") FROM "{safe_measurement}"'
                    f"{time_filter}"
                ),
                username=username,
                password=password,
                database=candidate_db,
                container_name=container_name,
                timeout=timeout,
            )
            first_time = _first_time_value(first_payload)

            last_payload = _run_influx_query(
                query=(
                    f'SELECT LAST("{safe_processed_field}") FROM "{safe_measurement}"'
                    f"{time_filter}"
                ),
                username=username,
                password=password,
                database=candidate_db,
                container_name=container_name,
                timeout=timeout,
            )
            last_time = _first_time_value(last_payload)

            latency_payload = _run_influx_query(
                query=(
                    f'SELECT MEAN("{safe_latency_field}") FROM "{safe_measurement}"'
                    f"{time_filter}"
                ),
                username=username,
                password=password,
                database=candidate_db,
                container_name=container_name,
                timeout=timeout,
            )
            latency_raw = _first_numeric_value(latency_payload)

            if count_value is None or latency_raw is None:
                last_error = (
                    f"No KPI values from measurement '{safe_measurement}' in database '{candidate_db}'"
                )
                per_database_errors[candidate_db] = last_error
                continue

            throughput_points_per_second = float(count_value)
            duration_seconds = None
            if first_time is not None and last_time is not None and last_time > first_time:
                duration_seconds = float(last_time - first_time)
                throughput_points_per_second = float(count_value) / duration_seconds

            latency_seconds = float(latency_raw) / float(latency_scale)

            return (
                float(throughput_points_per_second),
                float(latency_seconds),
                {
                    "database": candidate_db,
                    "measurement": safe_measurement,
                    "throughput_field": safe_processed_field,
                    "throughput_query_mode": "count_over_time_span_points_per_second",
                    "throughput_count": float(count_value),
                    "throughput_duration_seconds": duration_seconds,
                    "throughput_window_seconds": int(max(1, int(throughput_window_seconds))),
                    "throughput_time_filter": time_filter or "none",
                    "latency_field": safe_latency_field,
                    "latency_query_mode": "mean_latency_scaled_to_seconds",
                    "latency_scale": latency_scale,
                },
            )
        except Exception as candidate_error:
            last_error = str(candidate_error)
            per_database_errors[candidate_db] = last_error

    return (
        None,
        None,
        {
            "error": last_error or "No measurements found in InfluxDB",
            "candidate_databases": candidate_databases,
            "per_database_errors": per_database_errors,
            "processed_measurement": safe_measurement,
            "processed_field": safe_processed_field,
            "latency_field": safe_latency_field,
        },
    )


def compute_throughput_latency_from_influx(
    username: str,
    password: str,
    preferred_measurement: Optional[str] = None,
    throughput_field: Optional[str] = None,
    latency_field: Optional[str] = None,
    window_seconds: int = 30,
    database: str = "datain",
    container_name: str = "ia-influxdb",
    timeout: int = 120,
    require_latency: bool = False,
) -> Tuple[Optional[float], Optional[float], Dict[str, object]]:
    candidate_databases: List[str] = [database] if database else []

    try:
        discovered_databases = list_databases(
            username=username,
            password=password,
            container_name=container_name,
            timeout=timeout,
        )
        for discovered_db in discovered_databases:
            if discovered_db and discovered_db not in candidate_databases:
                candidate_databases.append(discovered_db)
    except Exception as discovery_error:
        logger.debug("Failed to discover Influx databases: %s", discovery_error)

    for fallback_db in ["datain", "dataout", "telegraf"]:
        if fallback_db not in candidate_databases:
            candidate_databases.append(fallback_db)

    last_error: Optional[str] = None
    for candidate_db in candidate_databases:
        try:
            measurement = resolve_measurement(
                preferred_measurement=preferred_measurement,
                username=username,
                password=password,
                database=candidate_db,
                container_name=container_name,
                timeout=timeout,
            )
            if not measurement:
                last_error = f"No measurements found in database '{candidate_db}'"
                continue

            fields = list_field_keys(
                measurement=measurement,
                username=username,
                password=password,
                database=candidate_db,
                container_name=container_name,
                timeout=timeout,
            )
            if not fields:
                last_error = f"No fields found for measurement '{measurement}' in database '{candidate_db}'"
                continue

            chosen_throughput_field, throughput_field_selection = _select_throughput_field(fields, throughput_field)
            if not chosen_throughput_field:
                last_error = (
                    f"No throughput-like field available for measurement '{measurement}' in database '{candidate_db}'"
                )
                continue
            chosen_latency_field = (
                latency_field if latency_field in fields else ("latency" if "latency" in fields else None)
            )
            if require_latency and latency_field and chosen_latency_field is None:
                last_error = (
                    f"Measurement '{measurement}' in database '{candidate_db}' has no latency field "
                    f"(requested '{latency_field}')"
                )
                continue

            safe_measurement = _validate_identifier(measurement, "measurement")
            safe_count_field = _validate_identifier(chosen_throughput_field, "throughput field")

            field_lower = chosen_throughput_field.lower()
            is_rate_like_field = any(token in field_lower for token in _THROUGHPUT_RATE_HINTS)
            is_counter_like_field = any(token in field_lower for token in _THROUGHPUT_COUNTER_HINTS)

            throughput_value = None
            throughput_query_mode = "unknown"

            if throughput_field_selection == "fallback_non_time":
                # Generic numeric fields are not throughput; estimate rate by point count in window.
                count_query = (
                    f'SELECT COUNT("{safe_count_field}") FROM "{safe_measurement}" '
                    f"WHERE time > now() - {int(window_seconds)}s"
                )
                count_payload = _run_influx_query(
                    query=count_query,
                    username=username,
                    password=password,
                    database=candidate_db,
                    container_name=container_name,
                    timeout=timeout,
                )
                count_value = _first_numeric_value(count_payload)
                if count_value is not None and window_seconds > 0:
                    throughput_value = float(count_value) / float(window_seconds)
                throughput_query_mode = "count_per_window_fallback_field"

            if throughput_value is None and is_counter_like_field and window_seconds > 0:
                # Counter-like fields are cumulative; derive rate from delta over window.
                last_query = (
                    f'SELECT LAST("{safe_count_field}") FROM "{safe_measurement}" '
                    f"WHERE time > now() - {int(window_seconds)}s"
                )
                first_query = (
                    f'SELECT FIRST("{safe_count_field}") FROM "{safe_measurement}" '
                    f"WHERE time > now() - {int(window_seconds)}s"
                )
                last_payload = _run_influx_query(
                    query=last_query,
                    username=username,
                    password=password,
                    database=candidate_db,
                    container_name=container_name,
                    timeout=timeout,
                )
                first_payload = _run_influx_query(
                    query=first_query,
                    username=username,
                    password=password,
                    database=candidate_db,
                    container_name=container_name,
                    timeout=timeout,
                )
                last_value = _first_numeric_value(last_payload)
                first_value = _first_numeric_value(first_payload)
                if last_value is not None and first_value is not None:
                    delta = float(last_value) - float(first_value)
                    if delta >= 0:
                        throughput_value = delta / float(window_seconds)
                        throughput_query_mode = "counter_delta_per_window"

            if throughput_value is None:
                # For rate-like fields, LAST is typically already points/s.
                throughput_query = (
                    f'SELECT LAST("{safe_count_field}") FROM "{safe_measurement}" '
                    f"WHERE time > now() - {int(window_seconds)}s"
                )
                throughput_payload = _run_influx_query(
                    query=throughput_query,
                    username=username,
                    password=password,
                    database=candidate_db,
                    container_name=container_name,
                    timeout=timeout,
                )
                throughput_value = _first_numeric_value(throughput_payload)
                throughput_query_mode = "last_field_rate" if is_rate_like_field else "last_field"

            if throughput_value is None:
                # Final fallback for raw telemetry measurements.
                count_query = (
                    f'SELECT COUNT("{safe_count_field}") FROM "{safe_measurement}" '
                    f"WHERE time > now() - {int(window_seconds)}s"
                )
                count_payload = _run_influx_query(
                    query=count_query,
                    username=username,
                    password=password,
                    database=candidate_db,
                    container_name=container_name,
                    timeout=timeout,
                )
                count_value = _first_numeric_value(count_payload)
                if count_value is not None and window_seconds > 0:
                    throughput_value = float(count_value) / float(window_seconds)
                throughput_query_mode = "count_per_window"

            throughput = None
            if throughput_value is not None:
                throughput = float(throughput_value)

            latency = None
            if chosen_latency_field:
                safe_latency_field = _validate_identifier(chosen_latency_field, "latency field")
                latency_query = (
                    f'SELECT MEAN("{safe_latency_field}") FROM "{safe_measurement}" '
                    f"WHERE time > now() - {int(window_seconds)}s"
                )
                latency_payload = _run_influx_query(
                    query=latency_query,
                    username=username,
                    password=password,
                    database=candidate_db,
                    container_name=container_name,
                    timeout=timeout,
                )
                latency = _first_numeric_value(latency_payload)

            metadata: Dict[str, object] = {
                "database": candidate_db,
                "measurement": measurement,
                "fields": fields,
                "throughput_field": chosen_throughput_field,
                "throughput_field_selection": throughput_field_selection,
                "throughput_query_mode": throughput_query_mode,
                "latency_field": chosen_latency_field,
                "latency_missing": chosen_latency_field is None,
                "window_seconds": window_seconds,
            }
            return throughput, latency, metadata
        except Exception as candidate_error:
            last_error = str(candidate_error)

    return (
        None,
        None,
        {
            "error": last_error or "No measurements found in InfluxDB",
            "candidate_databases": candidate_databases,
            "preferred_measurement": preferred_measurement,
        },
    )
