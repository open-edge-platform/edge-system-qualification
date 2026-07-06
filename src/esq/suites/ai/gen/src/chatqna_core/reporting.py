# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "execution_order",
    "test_id",
    "display_name",
    "backend_runtime",
    "compute_device",
    "model_config",
    "corpus_id",
    "prompt_set_id",
    "service_startup_seconds",
    "document_ingestion_seconds",
    "p50_query_latency_ms",
    "p95_query_latency_ms",
    "ttft_ms",
    "query_success_rate",
    "ref_platform",
    "ref_p50_query_latency_ms",
    "ref_p95_query_latency_ms",
    "ref_ttft_ms",
    "status",
]

PRESENTATION_CSV_COLUMNS = [
    "execution_order",
    "test_id",
    "display_name",
    "backend_runtime",
    "compute_device",
    "p50_query_latency_ms",
    "p95_query_latency_ms",
    "ttft_ms",
    "ref_platform",
    "status",
]


def ensure_chatqna_report_paths(base_dir: str, consolidated: bool = False) -> Dict[str, str]:
    """Return file paths for ChatQnA Core reporting artifacts.

    When *consolidated* is True, paths use a ``_consolidated`` suffix so that
    the consolidated multi-scenario test keeps its own CSV and graphs
    completely separate from the individual per-scenario tests.
    """
    report_dir = Path(base_dir) / "data" / "suites" / "ai" / "gen" / "chatqna_core" / "results"
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix = "chatqna_core_consolidated" if consolidated else "chatqna_core_individual"
    return {
        "report_dir": str(report_dir),
        "csv_path": str(report_dir / f"{prefix}_performance.csv"),
        "presentation_csv_path": str(report_dir / f"{prefix}_performance_presentation.csv"),
        "metadata_json_path": str(report_dir / f"{prefix}_latest_metadata.json"),
        "p50_plot_path": str(report_dir / f"{prefix}_p50_latency.png"),
        "p95_plot_path": str(report_dir / f"{prefix}_p95_latency.png"),
        "ttft_plot_path": str(report_dir / f"{prefix}_ttft.png"),
        "combined_plot_path": str(report_dir / f"{prefix}_combined_latency.png"),
    }


def _get_next_execution_order(csv_path: Path) -> int:
    if not csv_path.exists():
        return 1
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return 1
    try:
        return int(rows[-1].get("execution_order", 0)) + 1
    except Exception:
        return len(rows) + 1


def append_performance_row(csv_path: str, row_data: Dict[str, object]) -> None:
    """Append a single performance row to the CSV file."""
    csv_file_path = Path(csv_path)
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_file_path.exists()

    row = {column: str(row_data.get(column, "")) for column in CSV_COLUMNS}
    row["execution_order"] = str(_get_next_execution_order(csv_file_path))

    with csv_file_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def generate_presentation_csv(csv_path: str, presentation_csv_path: str) -> None:
    """Regenerate the presentation CSV from all rows in *csv_path*."""
    source_csv = Path(csv_path)
    target_csv = Path(presentation_csv_path)
    if not source_csv.exists():
        return

    with source_csv.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    with target_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PRESENTATION_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in PRESENTATION_CSV_COLUMNS})


def write_scenario_metadata(metadata_json_path: str, scenario_metadata: Dict[str, object]) -> None:
    metadata_path = Path(metadata_json_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(scenario_metadata, indent=2), encoding="utf-8")


def _parse_metric_value(raw: str) -> float:
    """Return the float value, or NaN if the value is -1 (sentinel for 'no data')."""
    import math

    try:
        v = float(raw)
        return math.nan if v < 0 else v
    except Exception:
        return math.nan


def _annotate_bars(ax, bars, values) -> None:
    """Add value labels above each bar; show 'N/A' for NaN."""
    import math

    for bar, value in zip(bars, values):
        if math.isnan(value):
            label = "N/A"
            y = bar.get_height()
        else:
            label = f"{value:.0f}"
            y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y * 1.02 if y > 0 else 0.5,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def _bar_chart(plot_path: str, labels: List[str], values: List[float], title: str, ylabel: str) -> bool:
    """Render a single-metric bar chart. Returns True if saved successfully."""
    import math

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x_positions = list(range(len(labels)))
    bar_values = [0.0 if math.isnan(v) else v for v in values]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5))
    bars = ax.bar(x_positions, bar_values, color="#0071C5", edgecolor="white", width=0.6)
    _annotate_bars(ax, bars, values)

    # Connect data points with a line when there is more than one run
    valid_x = [x for x, v in zip(x_positions, values) if not math.isnan(v)]
    valid_v = [v for v in values if not math.isnan(v)]
    if len(valid_x) > 1:
        ax.plot(valid_x, valid_v, color="#FF6B35", linewidth=1.5, marker="o", markersize=5, zorder=3)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Scenario / Run", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    return True


def generate_performance_graphs(
    csv_path: str,
    p50_plot_path: str,
    p95_plot_path: str,
    ttft_plot_path: str,
) -> List[str]:
    """Generate bar/line charts from all rows in *csv_path*."""
    import math

    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("matplotlib is not available; skipping Chat Q&A Core graph generation")
        return []

    csv_file_path = Path(csv_path)
    if not csv_file_path.exists():
        return []

    with csv_file_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        return []

    labels = [row.get("display_name", row.get("test_id", f"run-{i + 1}")) for i, row in enumerate(rows)]
    p50_values = [_parse_metric_value(row.get("p50_query_latency_ms", "-1")) for row in rows]
    p95_values = [_parse_metric_value(row.get("p95_query_latency_ms", "-1")) for row in rows]
    ttft_values = [_parse_metric_value(row.get("ttft_ms", "-1")) for row in rows]

    generated_files: List[str] = []

    if _bar_chart(p50_plot_path, labels, p50_values, "Chat Q&A Core - P50 Query Latency", "Latency (ms)"):
        generated_files.append(p50_plot_path)
    if _bar_chart(p95_plot_path, labels, p95_values, "Chat Q&A Core - P95 Query Latency", "Latency (ms)"):
        generated_files.append(p95_plot_path)
    if _bar_chart(ttft_plot_path, labels, ttft_values, "Chat Q&A Core - Time-to-First-Token (TTFT)", "TTFT (ms)"):
        generated_files.append(ttft_plot_path)

    # --- Combined multi-metric line graph across all runs ---
    combined_plot_path = str(Path(p50_plot_path).parent / "chatqna_core_combined_latency.png")
    is_individual_csv = "_individual_" in Path(csv_path).name
    try:
        if is_individual_csv:
            x_positions = list(range(1, len(labels) + 1))
            xtick_positions = [0] + x_positions
            xtick_labels = ["Start (0,0)"] + labels
        else:
            x_positions = list(range(len(labels)))
            xtick_positions = x_positions
            xtick_labels = labels

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 5))

        metric_series = [
            (p50_values, "#0071C5", "P50 Latency"),
            (p95_values, "#FF6B35", "P95 Latency"),
            (ttft_values, "#6DC066", "TTFT"),
        ]

        has_any_data = False
        for values, color, metric_label in metric_series:
            valid_x = [x for x, v in zip(x_positions, values) if not math.isnan(v)]
            valid_v = [v for v in values if not math.isnan(v)]
            if valid_x:
                if is_individual_csv:
                    plot_x = [0] + valid_x
                    plot_v = [0.0] + valid_v
                else:
                    plot_x = valid_x
                    plot_v = valid_v

                ax.plot(plot_x, plot_v, color=color, linewidth=2, marker="o", markersize=6, label=metric_label)
                for vx, vv in zip(valid_x, valid_v):
                    ax.annotate(
                        f"{vv:.0f}",
                        xy=(vx, vv),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        color=color,
                    )
                has_any_data = True

        if has_any_data:
            ax.set_title("Chat Q&A Core - Latency Overview (all metrics)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Scenario / Run", fontsize=10)
            ax.set_ylabel("Latency (ms)", fontsize=10)
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=8)

            # Position legend to avoid overlapping with plot data and axis labels:
            # - Individual runs (single point): legend below chart in available whitespace
            # - Consolidated runs (multiple scenarios): legend in upper-left to avoid x-axis labels
            if is_individual_csv:
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.18),
                    ncol=3,
                    fontsize=9,
                    frameon=False,
                )
                fig.subplots_adjust(bottom=0.25)
            else:
                handles, legend_labels = ax.get_legend_handles_labels()
                fig.legend(
                    handles,
                    legend_labels,
                    loc="upper left",
                    bbox_to_anchor=(0.01, 0.99),
                    fontsize=9,
                    frameon=False,
                    ncol=1,
                )
                fig.subplots_adjust(top=0.82, bottom=0.15)

            ax.grid(alpha=0.3, linestyle="--")
            ax.set_ylim(bottom=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            plt.savefig(combined_plot_path, dpi=120)
            generated_files.append(combined_plot_path)

        plt.close(fig)
    except Exception as combined_error:
        logger.warning("Could not generate combined latency chart: %s", combined_error)

    return generated_files
