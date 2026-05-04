# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Performance report generation helpers for timeseries suites."""

import csv
import logging
import math
from pathlib import Path
from typing import Dict, List, Sequence, Union

logger = logging.getLogger(__name__)

ValueType = Union[str, int, float, bool]

CSV_COLUMNS = [
    "execution_order",
    "test_id",
    "display_name",
    "stream_count",
    "data_points",
    "compute_device",
    "ingestion_mode",
    "cur_throughput",
    "cur_latency",
    "ref_platform",
    "ref_throughput",
    "ref_latency",
    "throughput_delta",
    "status",
]

PRESENTATION_CSV_BASE_COLUMNS = [
    "execution_order",
    "test_id",
    "display_name",
    "cur_throughput",
    "cur_latency",
    "throughput_delta",
    "status",
]


def _collect_reference_platforms(rows: Sequence[Dict[str, str]]) -> List[str]:
    """Collect unique reference platform labels from CSV rows."""
    platforms = []
    seen = set()
    for row in rows:
        platform = str(row.get("ref_platform", "")).strip()
        if not platform or platform in seen:
            continue
        seen.add(platform)
        platforms.append(platform)

    return platforms


def _platform_header_name(rows: Sequence[Dict[str, str]]) -> str:
    """Return heading text used for reference columns in presentation CSV."""
    platforms = _collect_reference_platforms(rows)
    if len(platforms) == 1:
        return platforms[0]
    if platforms:
        return "Reference Platform"
    return "Ref_Platform"


def _simplify_cpu_name(name: str) -> str:
    """Remove 'Intel ' prefix from a CPU brand string for compact column headers."""
    simplified = name.strip()
    if simplified.lower().startswith("intel "):
        simplified = simplified[6:]
    return simplified


def _get_current_system_cpu() -> str:
    """Return the running system's CPU brand string."""
    try:
        import cpuinfo

        cpu_data = cpuinfo.get_cpu_info()
        return str(cpu_data.get("brand_raw", "")).strip() or "Unknown"
    except Exception:
        pass
    # Fallback: read /proc/cpuinfo directly
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Unknown"


def _add_platform_legend_box(
    plt, rows: Sequence[Dict[str, str]], put_color: str = "tab:blue", rp_color: str = "tab:orange"
) -> None:
    """Render platform under test and reference platform with plot-matched colors."""
    try:
        from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
    except Exception:
        return

    ref_platforms = _collect_reference_platforms(rows)
    put_text = _get_current_system_cpu()
    rp_text = ", ".join(ref_platforms) if ref_platforms else "Unknown"

    put_line = TextArea(
        f"PUT: {put_text}",
        textprops={"color": put_color, "fontsize": 8, "fontweight": "bold"},
    )
    rp_line = TextArea(
        f"RP: {rp_text}",
        textprops={"color": rp_color, "fontsize": 8, "fontweight": "bold"},
    )
    packed = VPacker(children=[put_line, rp_line], align="left", pad=0, sep=2)

    anchored = AnchoredOffsetbox(
        loc="lower left",
        child=packed,
        pad=0.25,
        borderpad=0.35,
        frameon=True,
        bbox_to_anchor=(0.01, 0.01),
        bbox_transform=plt.gcf().transFigure,
    )
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor("gray")
    anchored.patch.set_alpha(0.92)
    plt.gcf().add_artist(anchored)


def _is_valid_number(value: object) -> bool:
    try:
        parsed = float(str(value))
        return not math.isnan(parsed) and not math.isinf(parsed)
    except Exception:
        return False


def _draw_reference_segment_markers(
    plt,
    positions: Sequence[float],
    ref_values: Sequence[float],
    half_width: float,
    color: str,
    label: str,
    decimals: int = 2,
) -> None:
    """Draw per-bar reference as a short horizontal line plus marker and value label."""
    label_drawn = False
    for x, y in zip(positions, ref_values):
        if not _is_valid_number(y):
            continue
        y_value = float(y)
        current_label = label if not label_drawn else None
        plt.hlines(
            y=y_value,
            xmin=x - half_width,
            xmax=x + half_width,
            colors=color,
            linestyles="--",
            linewidth=1.3,
            label=current_label,
        )
        plt.plot([x], [y_value], marker="o", color=color, markersize=4)
        plt.text(
            x,
            y_value,
            _format_numeric(y_value, decimals=decimals),
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )
        label_drawn = True


def _format_numeric(value: ValueType, decimals: int = 4) -> str:
    """Format numeric values for CSV readability and avoid scientific notation."""
    try:
        parsed = float(value)
    except Exception:
        return str(value)

    if math.isnan(parsed) or math.isinf(parsed):
        return ""

    formatted = f"{parsed:.{decimals}f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def ensure_timeseries_report_paths(base_dir: str) -> Dict[str, str]:
    """Create report output directory and return artifact paths."""
    report_dir = Path(base_dir) / "data" / "suites" / "ai" / "timeseries" / "wt" / "results"
    report_dir.mkdir(parents=True, exist_ok=True)

    return {
        "report_dir": str(report_dir),
        "csv_path": str(report_dir / "timeseries_performance.csv"),
        "presentation_csv_path": str(report_dir / "timeseries_performance_presentation.csv"),
        "throughput_plot_path": str(report_dir / "timeseries_throughput.png"),
        "latency_plot_path": str(report_dir / "timeseries_latency.png"),
        "throughput_grouped_plot_path": str(report_dir / "timeseries_throughput_grouped.png"),
        "latency_grouped_plot_path": str(report_dir / "timeseries_latency_grouped.png"),
        "throughput_scenario_grouped_plot_path": str(report_dir / "timeseries_throughput_by_scenario.png"),
        "latency_scenario_grouped_plot_path": str(report_dir / "timeseries_latency_by_scenario.png"),
    }


def _normalize_csv_schema(csv_path: Path) -> None:
    """Normalize existing CSV files to current schema while preserving row order."""
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        if fieldnames == CSV_COLUMNS:
            return
        rows = list(reader)

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row_index, source_row in enumerate(rows, start=1):
            row = {column: source_row.get(column, "") for column in CSV_COLUMNS}
            existing_order = str(source_row.get("execution_order", "")).strip()
            row["execution_order"] = existing_order if existing_order else str(row_index)
            writer.writerow(row)


def _get_next_execution_order(csv_path: Path) -> int:
    """Return next execution order index for append-only CSV logging."""
    if not csv_path.exists():
        return 1

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        if not rows:
            return 1

        try:
            return int(rows[-1].get("execution_order", 0)) + 1
        except Exception:
            return len(rows) + 1


def append_performance_row(csv_path: str, row_data: Dict[str, ValueType]) -> None:
    """Append one performance row preserving execution order."""
    csv_file_path = Path(csv_path)
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_csv_schema(csv_file_path)
    execution_order = _get_next_execution_order(csv_file_path)

    row = {
        "execution_order": str(execution_order),
        "test_id": row_data.get("test_id", ""),
        "display_name": row_data.get("display_name", ""),
        "stream_count": str(row_data.get("stream_count", "")),
        "data_points": str(row_data.get("data_points", "")),
        "compute_device": str(row_data.get("compute_device", "")),
        "ingestion_mode": str(row_data.get("ingestion_mode", "")),
        "cur_throughput": _format_numeric(row_data.get("cur_throughput", "")),
        "cur_latency": _format_numeric(row_data.get("cur_latency", "")),
        "ref_platform": str(row_data.get("ref_platform", "")),
        "ref_throughput": _format_numeric(row_data.get("ref_throughput", "")),
        "ref_latency": _format_numeric(row_data.get("ref_latency", "")),
        "throughput_delta": _format_numeric(row_data.get("throughput_delta", "")),
        "status": str(row_data.get("status", "")),
    }

    write_header = not csv_file_path.exists()
    with csv_file_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def generate_presentation_csv(csv_path: str, presentation_csv_path: str) -> None:
    """Generate a compact CSV view for presentation-friendly tables."""
    source_csv = Path(csv_path)
    target_csv = Path(presentation_csv_path)
    target_csv.parent.mkdir(parents=True, exist_ok=True)

    if not source_csv.exists():
        return

    with source_csv.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    put_cpu = _simplify_cpu_name(_get_current_system_cpu())
    platform_header = _simplify_cpu_name(_platform_header_name(rows))
    put_throughput_column = f"{put_cpu} throughput"
    put_latency_column = f"{put_cpu} latency"
    ref_throughput_column = f"{platform_header} ref_throughput"
    ref_latency_column = f"{platform_header} ref_latency"
    presentation_columns = [
        "execution_order",
        "test_id",
        "display_name",
        put_throughput_column,
        put_latency_column,
        ref_throughput_column,
        ref_latency_column,
        "throughput_delta",
        "status",
    ]

    with target_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=presentation_columns)
        writer.writeheader()
        for row in rows:
            ref_throughput = str(row.get("ref_throughput", "")).strip()
            ref_latency = str(row.get("ref_latency", "")).strip()

            writer.writerow(
                {
                    "execution_order": row.get("execution_order", ""),
                    "test_id": row.get("test_id", ""),
                    "display_name": row.get("display_name", ""),
                    put_throughput_column: row.get("cur_throughput", ""),
                    put_latency_column: row.get("cur_latency", ""),
                    ref_throughput_column: ref_throughput,
                    ref_latency_column: ref_latency,
                    "throughput_delta": row.get("throughput_delta", ""),
                    "status": row.get("status", ""),
                }
            )


def generate_performance_graphs(
    csv_path: str,
    throughput_plot_path: str,
    latency_plot_path: str,
    throughput_grouped_plot_path: str,
    latency_grouped_plot_path: str,
    throughput_scenario_grouped_plot_path: str,
    latency_scenario_grouped_plot_path: str,
    include_trend_graphs: bool = True,
    include_grouped_graphs: bool = True,
) -> List[str]:
    """Generate trend and grouped bar performance plots from CSV history."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except Exception:
        logger.warning("matplotlib is not available; skipping graph generation")
        return []

    csv_file_path = Path(csv_path)
    if not csv_file_path.exists():
        return []

    with csv_file_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        return []

    x_values = []
    labels = []
    throughput_values = []
    ref_throughput_values = []
    latency_values = []
    ref_latency_values = []
    has_latency_column = bool(rows and ("cur_latency" in rows[0] or "latency" in rows[0]))

    for row in rows:
        execution_order = len(x_values) + 1

        label = str(row.get("display_name", "") or row.get("test_id", "") or f"run-{execution_order}")
        label = label.replace("TS Wind Turbine - ", "", 1)

        try:
            throughput = float(row.get("cur_throughput", row.get("throughput", "nan")))
        except Exception:
            throughput = float("nan")

        try:
            ref_throughput = float(row.get("ref_throughput", "nan"))
        except Exception:
            ref_throughput = float("nan")

        latency = float("nan")
        if has_latency_column:
            try:
                latency = float(row.get("cur_latency", row.get("latency", "nan")))
            except Exception:
                latency = float("nan")

        ref_latency = float("nan")
        if has_latency_column:
            try:
                ref_latency = float(row.get("ref_latency", "nan"))
            except Exception:
                ref_latency = float("nan")

        x_values.append(execution_order)
        labels.append(label)
        throughput_values.append(throughput)
        latency_values.append(latency)
        ref_throughput_values.append(ref_throughput)
        ref_latency_values.append(ref_latency)

    generated_files: List[str] = []
    if include_trend_graphs:
        plt.figure(figsize=(12, 5))
        plt.plot(x_values, throughput_values, marker="o", label="Measured")
        plt.plot(x_values, ref_throughput_values, marker="x", linestyle="--", label="Reference")
        plt.title("TS-WT - Throughput Comparison")
        plt.xlabel("Execution Modes")
        plt.ylabel("Throughput (points/s)")
        plt.grid(True, alpha=0.3)
        plt.xticks(x_values, labels, rotation=45, ha="right")
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        plt.gca().ticklabel_format(style="plain", axis="y", useOffset=False)
        plt.legend()
        _add_platform_legend_box(plt=plt, rows=rows, put_color="tab:blue", rp_color="tab:orange")
        plt.tight_layout()
        plt.savefig(throughput_plot_path)
        plt.close()
        generated_files.append(throughput_plot_path)

        if has_latency_column:
            plt.figure(figsize=(12, 5))
            plt.plot(x_values, latency_values, marker="o", color="tab:orange", label="Measured")
            plt.plot(x_values, ref_latency_values, marker="x", linestyle="--", color="tab:red", label="Reference")
            plt.title("TS-WT - Latency Comparison")
            plt.xlabel("Execution Modes")
            plt.ylabel("Latency (s)")
            plt.grid(True, alpha=0.3)
            plt.xticks(x_values, labels, rotation=45, ha="right")
            plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            plt.gca().ticklabel_format(style="plain", axis="y", useOffset=False)
            plt.legend()
            _add_platform_legend_box(plt=plt, rows=rows, put_color="tab:orange", rp_color="tab:red")
            plt.tight_layout()
            plt.savefig(latency_plot_path)
            plt.close()
            generated_files.append(latency_plot_path)
    else:
        last_index = len(rows) - 1
        measured_throughput = throughput_values[last_index]
        reference_throughput = ref_throughput_values[last_index]
        run_label = labels[last_index]

        plt.figure(figsize=(7, 5))
        x_points = [0, 1]
        measured_line = [0.0, measured_throughput]
        reference_line = [0.0, reference_throughput]
        plt.plot(x_points, measured_line, marker="o", color="tab:blue", label="Measured")
        plt.plot(x_points, reference_line, marker="x", linestyle="--", color="tab:green", label="Reference")
        plt.title("TS-WT - Throughput Comparison")
        plt.xlabel("Execution Mode")
        plt.ylabel("Throughput (points/s)")
        plt.xticks([1], [run_label], rotation=20, ha="right")
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        plt.gca().ticklabel_format(style="plain", axis="y", useOffset=False)
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        _add_platform_legend_box(plt=plt, rows=rows, put_color="tab:blue", rp_color="tab:green")
        plt.tight_layout()
        plt.savefig(throughput_grouped_plot_path)
        plt.close()
        generated_files.append(throughput_grouped_plot_path)

        if has_latency_column:
            measured_latency = latency_values[last_index]
            reference_latency = ref_latency_values[last_index]
            plt.figure(figsize=(7, 5))
            x_points = [0, 1]
            measured_line = [0.0, measured_latency]
            reference_line = [0.0, reference_latency]
            plt.plot(x_points, measured_line, marker="o", color="tab:orange", label="Measured")
            plt.plot(x_points, reference_line, marker="x", linestyle="--", color="tab:red", label="Reference")
            plt.title("TS-WT - Latency Comparison")
            plt.xlabel("Execution Mode")
            plt.ylabel("Latency (s)")
            plt.xticks([1], [run_label], rotation=20, ha="right")
            plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            plt.gca().ticklabel_format(style="plain", axis="y", useOffset=False)
            plt.grid(True, axis="y", alpha=0.3)
            plt.legend()
            _add_platform_legend_box(plt=plt, rows=rows, put_color="tab:orange", rp_color="tab:red")
            plt.tight_layout()
            plt.savefig(latency_grouped_plot_path)
            plt.close()
            generated_files.append(latency_grouped_plot_path)

    # Grouped charts require mode/device/stream/data columns from expanded schema.
    has_grouping_columns = (
        include_trend_graphs
        and include_grouped_graphs
        and all(rows and col in rows[0] for col in ["stream_count", "data_points", "compute_device", "ingestion_mode"])
    )
    if rows and has_grouping_columns:
        last_row = rows[-1]
        target_stream_count = str(last_row.get("stream_count", ""))
        target_data_points = str(last_row.get("data_points", ""))

        # Keep only rows from the same stream/data scenario and from passed runs.
        scoped_rows = [
            row
            for row in rows
            if str(row.get("stream_count", "")) == target_stream_count
            and str(row.get("data_points", "")) == target_data_points
        ]

        latest_by_key = {}
        for row_index, row in enumerate(scoped_rows, start=1):
            device = str(row.get("compute_device", "unknown")).lower()
            mode = str(row.get("ingestion_mode", "")).lower()
            if mode not in {"opcua", "mqtt"}:
                continue
            key = (device, mode)
            try:
                order = int(row.get("execution_order", row_index))
            except Exception:
                order = row_index
            prev = latest_by_key.get(key)
            if prev is None or order >= prev[0]:
                latest_by_key[key] = (order, row)

        devices = sorted({key[0] for key in latest_by_key.keys()})
        if devices:
            x_positions = list(range(len(devices)))
            bar_width = 0.35

            opcua_throughput = []
            mqtt_throughput = []
            opcua_latency = []
            mqtt_latency = []
            opcua_ref_throughput = []
            mqtt_ref_throughput = []
            opcua_ref_latency = []
            mqtt_ref_latency = []

            for device in devices:
                opcua_row = latest_by_key.get((device, "opcua"), (0, {}))[1]
                mqtt_row = latest_by_key.get((device, "mqtt"), (0, {}))[1]

                try:
                    opcua_throughput.append(float(opcua_row.get("cur_throughput", opcua_row.get("throughput", "nan"))))
                except Exception:
                    opcua_throughput.append(float("nan"))

                try:
                    opcua_ref_throughput.append(float(opcua_row.get("ref_throughput", "nan")))
                except Exception:
                    opcua_ref_throughput.append(float("nan"))

                try:
                    mqtt_throughput.append(float(mqtt_row.get("cur_throughput", mqtt_row.get("throughput", "nan"))))
                except Exception:
                    mqtt_throughput.append(float("nan"))

                try:
                    mqtt_ref_throughput.append(float(mqtt_row.get("ref_throughput", "nan")))
                except Exception:
                    mqtt_ref_throughput.append(float("nan"))

                try:
                    opcua_latency.append(float(opcua_row.get("cur_latency", opcua_row.get("latency", "nan"))))
                except Exception:
                    opcua_latency.append(float("nan"))

                try:
                    opcua_ref_latency.append(float(opcua_row.get("ref_latency", "nan")))
                except Exception:
                    opcua_ref_latency.append(float("nan"))

                try:
                    mqtt_latency.append(float(mqtt_row.get("cur_latency", mqtt_row.get("latency", "nan"))))
                except Exception:
                    mqtt_latency.append(float("nan"))

                try:
                    mqtt_ref_latency.append(float(mqtt_row.get("ref_latency", "nan")))
                except Exception:
                    mqtt_ref_latency.append(float("nan"))

            left_positions = [x - (bar_width / 2.0) for x in x_positions]
            right_positions = [x + (bar_width / 2.0) for x in x_positions]

            plt.figure(figsize=(10, 5))
            plt.bar(left_positions, opcua_throughput, width=bar_width, label="OPC-UA")
            plt.bar(right_positions, mqtt_throughput, width=bar_width, label="MQTT")
            _draw_reference_segment_markers(
                plt=plt,
                positions=left_positions,
                ref_values=opcua_ref_throughput,
                half_width=bar_width / 2.0,
                color="black",
                label="Ref OPC-UA",
            )
            _draw_reference_segment_markers(
                plt=plt,
                positions=right_positions,
                ref_values=mqtt_ref_throughput,
                half_width=bar_width / 2.0,
                color="dimgray",
                label="Ref MQTT",
            )
            plt.title("TS-WT - Throughput Comparison")
            plt.xlabel("Compute Device")
            plt.ylabel("Throughput (points/s)")
            plt.xticks(x_positions, [d.upper() for d in devices])
            plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            plt.gca().ticklabel_format(style="plain", axis="y", useOffset=False)
            plt.grid(True, axis="y", alpha=0.3)
            plt.legend()
            _add_platform_legend_box(plt=plt, rows=rows, put_color="tab:blue", rp_color="tab:orange")
            plt.tight_layout()
            plt.savefig(throughput_grouped_plot_path)
            plt.close()
            generated_files.append(throughput_grouped_plot_path)

            plt.figure(figsize=(10, 5))
            plt.bar(left_positions, opcua_latency, width=bar_width, label="OPC-UA")
            plt.bar(right_positions, mqtt_latency, width=bar_width, label="MQTT")
            _draw_reference_segment_markers(
                plt=plt,
                positions=left_positions,
                ref_values=opcua_ref_latency,
                half_width=bar_width / 2.0,
                color="black",
                label="Ref OPC-UA",
            )
            _draw_reference_segment_markers(
                plt=plt,
                positions=right_positions,
                ref_values=mqtt_ref_latency,
                half_width=bar_width / 2.0,
                color="dimgray",
                label="Ref MQTT",
            )
            plt.title("TS-WT - Latency Comparison")
            plt.xlabel("Compute Device")
            plt.ylabel("Latency (s)")
            plt.xticks(x_positions, [d.upper() for d in devices])
            plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            plt.gca().ticklabel_format(style="plain", axis="y", useOffset=False)
            plt.grid(True, axis="y", alpha=0.3)
            plt.legend()
            _add_platform_legend_box(plt=plt, rows=rows, put_color="tab:orange", rp_color="tab:red")
            plt.tight_layout()
            plt.savefig(latency_grouped_plot_path)
            plt.close()
            generated_files.append(latency_grouped_plot_path)

    # Scenario-based grouped bars: x-axis is scenario (sXpY), bars are device+mode combinations.
    if include_trend_graphs and include_grouped_graphs and rows:
        latest_by_triplet = {}
        for row_index, row in enumerate(rows, start=1):
            scenario = f"s{row.get('stream_count', '')}p{row.get('data_points', '')}"
            device = str(row.get("compute_device", "unknown")).lower()
            mode = str(row.get("ingestion_mode", "")).lower()
            if mode not in {"opcua", "mqtt"}:
                continue
            key = (scenario, device, mode)
            try:
                order = int(row.get("execution_order", row_index))
            except Exception:
                order = row_index
            prev = latest_by_triplet.get(key)
            if prev is None or order >= prev[0]:
                latest_by_triplet[key] = (order, row)

        scenarios = sorted({k[0] for k in latest_by_triplet.keys()})
        combos = sorted({(k[1], k[2]) for k in latest_by_triplet.keys()})

        if scenarios and combos:
            x_positions = list(range(len(scenarios)))
            group_width = 0.8
            bar_width = group_width / max(len(combos), 1)

            def combo_label(device: str, mode: str) -> str:
                return f"{device.upper()}-{mode.upper()}"

            plt.figure(figsize=(max(10, len(scenarios) * 1.5), 5))
            for idx, (device, mode) in enumerate(combos):
                values = []
                ref_values = []
                for scenario in scenarios:
                    row = latest_by_triplet.get((scenario, device, mode), (0, {}))[1]
                    try:
                        values.append(float(row.get("cur_throughput", row.get("throughput", "nan"))))
                    except Exception:
                        values.append(float("nan"))
                    try:
                        ref_values.append(float(row.get("ref_throughput", "nan")))
                    except Exception:
                        ref_values.append(float("nan"))

                offset = -group_width / 2 + (idx + 0.5) * bar_width
                bar_positions = [x + offset for x in x_positions]
                plt.bar(bar_positions, values, width=bar_width, label=combo_label(device, mode))
                _draw_reference_segment_markers(
                    plt=plt,
                    positions=bar_positions,
                    ref_values=ref_values,
                    half_width=bar_width / 2.0,
                    color="black",
                    label="Reference" if idx == 0 else "",
                )

            plt.title("TS-WT - Throughput Comparison")
            plt.xlabel("Scenario")
            plt.ylabel("Throughput (points/s)")
            plt.xticks(x_positions, scenarios)
            plt.grid(True, axis="y", alpha=0.3)
            plt.legend()
            _add_platform_legend_box(plt=plt, rows=rows, put_color="tab:blue", rp_color="tab:orange")
            plt.tight_layout()
            plt.savefig(throughput_scenario_grouped_plot_path)
            plt.close()
            generated_files.append(throughput_scenario_grouped_plot_path)

            plt.figure(figsize=(max(10, len(scenarios) * 1.5), 5))
            for idx, (device, mode) in enumerate(combos):
                values = []
                ref_values = []
                for scenario in scenarios:
                    row = latest_by_triplet.get((scenario, device, mode), (0, {}))[1]
                    try:
                        values.append(float(row.get("cur_latency", row.get("latency", "nan"))))
                    except Exception:
                        values.append(float("nan"))
                    try:
                        ref_values.append(float(row.get("ref_latency", "nan")))
                    except Exception:
                        ref_values.append(float("nan"))

                offset = -group_width / 2 + (idx + 0.5) * bar_width
                bar_positions = [x + offset for x in x_positions]
                plt.bar(bar_positions, values, width=bar_width, label=combo_label(device, mode))
                _draw_reference_segment_markers(
                    plt=plt,
                    positions=bar_positions,
                    ref_values=ref_values,
                    half_width=bar_width / 2.0,
                    color="black",
                    label="Reference" if idx == 0 else "",
                )

            plt.title("TS-WT - Latency Comparison")
            plt.xlabel("Scenario")
            plt.ylabel("Latency")
            plt.xticks(x_positions, scenarios)
            plt.grid(True, axis="y", alpha=0.3)
            plt.legend()
            _add_platform_legend_box(plt=plt, rows=rows, put_color="tab:orange", rp_color="tab:red")
            plt.tight_layout()
            plt.savefig(latency_scenario_grouped_plot_path)
            plt.close()
            generated_files.append(latency_scenario_grouped_plot_path)

    return generated_files
