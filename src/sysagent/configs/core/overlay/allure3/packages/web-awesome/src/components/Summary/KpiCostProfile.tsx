import { FunctionComponent } from "preact";
import { useMemo, useState } from "preact/hooks";
import * as styles from "./TelemetryChartStyle.scss";

/**
 * KpiCostProfile
 *
 * Renders the KPI ↔ Telemetry correlation block produced by the Python
 * helper at extended_metadata.telemetry.kpi_correlation. Surfaces four
 * elements:
 *
 *   1. KPI banner row     (key metric + PASS/FAIL + target if any)
 *   2. Resource attribution stacked bar (% of Δ-power per device)
 *   3. Cost-per-KPI-unit table (per device × resource metric)
 *   4. Dominant-device hint (data only — chart badge is wired in TelemetrySection)
 *
 * The component renders nothing when no kpi_correlation block is present
 * (sysfs-mode telemetry, or tests without a key metric).
 */

export interface KpiCorrelationBlock {
  key_metric?: {
    name: string;
    unit?: string;
    value: number;
    direction: "higher_is_better" | "lower_is_better";
    validation_status?: string;
    target?: { value: any; op?: string | null } | null;
  };
  duration_s?: number | null;
  device_attribution?: Array<{
    device: string;
    delta_power_w?: number | null;
    share_power?: number | null;
    primary_metric?: string | null;
  }>;
  cost_per_kpi_unit?: Array<{
    device: string;
    metric: string;
    metric_unit?: string;
    workload_avg?: number;
    workload_peak?: number | null;
    idle_avg?: number | null;
    delta_avg?: number;
    cost_value?: number | null;
    cost_unit?: string;
    is_primary?: boolean;
    /**
     * Backend classification of row signal strength:
     *  - "primary"     anchor metric for the device, always shown
     *  - "significant" Δ over idle exceeds the metric-class noise floor
     *  - "low"         Δ below noise floor (sensor jitter)
     *  - "negative"    workload value below idle (counter-driving / jitter)
     */
    signal_class?: "primary" | "significant" | "low" | "negative";
    /**
     * False when the underlying Δ falls below the metric-class noise
     * floor; backend nulls ``cost_value`` in that case and the renderer
     * shows "— below sensor noise".
     */
    cost_meaningful?: boolean;
  }>;
  render_hints?: {
    dominant_device?: string | null;
  };
}

interface KpiCostProfileProps {
  correlation?: KpiCorrelationBlock | null;
}

const DEVICE_COLORS: Record<string, string> = {
  CPU: "#0071c5",
  iGPU: "#16a085",
  dGPU: "#8e44ad",
  NPU: "#e67e22",
};

const deviceColor = (device: string): string => {
  if (DEVICE_COLORS[device]) return DEVICE_COLORS[device];
  if (device.startsWith("dGPU")) return DEVICE_COLORS.dGPU;
  return "#7f8c8d";
};

const fmt = (value: number | null | undefined, digits = 3): string => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) {
    return value.toExponential(2);
  }
  return value.toFixed(digits);
};

// SI prefix table (engineering-notation steps). The cost value tends to be
// dominated by the ratio Δ_resource / KPI, which for high-throughput tests
// drops far below 1. Auto-scaling avoids exponential notation in the table.
// We cap at micro (µ) — anything smaller is sensor noise, not signal.
const SI_PREFIXES: Array<{ exp: number; prefix: string }> = [
  { exp: 9, prefix: "G" },
  { exp: 6, prefix: "M" },
  { exp: 3, prefix: "k" },
  { exp: 0, prefix: "" },
  { exp: -3, prefix: "m" },
  { exp: -6, prefix: "µ" },
];

const pickSiPrefix = (absValue: number): { exp: number; prefix: string } => {
  if (!Number.isFinite(absValue) || absValue === 0) return { exp: 0, prefix: "" };
  for (const entry of SI_PREFIXES) {
    const scaled = absValue / Math.pow(10, entry.exp);
    if (scaled >= 1 && scaled < 1000) return entry;
  }
  // Below the smallest tracked prefix → effectively zero (sensor noise).
  return { exp: -6, prefix: "µ" };
};

/**
 * Render cost value + unit with an SI prefix applied to the resource side
 * of the unit (e.g. "4.58e-5 % / MB/s" -> "45.8 µ% / MB/s"). Falls back to
 * fixed-point if the unit string can't be split.
 */
const formatCost = (
  value: number | null | undefined,
  unit: string | undefined,
): { value: string; unit: string } => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return { value: "—", unit: unit ?? "" };
  }
  const u = String(unit ?? "");
  const sep = u.includes(" / ") ? " / " : u.includes("·") ? "·" : "";
  if (!sep) return { value: fmt(value, 4), unit: u };

  const [resource, kpi] = u.split(sep);
  const { exp, prefix } = pickSiPrefix(Math.abs(value));
  if (exp === 0) return { value: fmt(value, 4), unit: u };
  const scaled = value / Math.pow(10, exp);
  const display = Math.abs(scaled) >= 100 ? scaled.toFixed(1) : scaled.toFixed(2);
  return { value: display, unit: `${prefix}${resource}${sep}${kpi}` };
};

/**
 * Decide whether a cost row carries enough signal to surface in the
 * default (compact) view. Backend annotates each row with ``signal_class``
 * and ``cost_meaningful``; the renderer shows ``significant`` rows and
 * ``primary`` rows whose cost is meaningful by default. The "Show all"
 * toggle additionally reveals below-idle and below-sensor-noise rows for
 * transparency.
 */
const rowFilter = (
  row: {
    signal_class?: string;
    delta_avg?: number | null;
    cost_value?: number | null;
    cost_meaningful?: boolean;
  },
  showAll: boolean,
): boolean => {
  if (showAll) return row.signal_class !== "low"; // "low" never shown
  const cls = row.signal_class;
  // Hide rows whose delta went below idle baseline (cost can't be computed).
  if (typeof row.delta_avg === "number" && row.delta_avg < 0) return false;
  // Hide rows whose cost is non-meaningful (Δ under sensor noise floor).
  // ``cost_meaningful`` is the explicit backend signal; fall back to
  // detecting a null cost_value with a non-negative delta when the field
  // is missing (older results).
  if (row.cost_meaningful === false) return false;
  if (
    row.cost_meaningful === undefined
    && (row.cost_value === null || row.cost_value === undefined)
    && typeof row.delta_avg === "number"
    && row.delta_avg >= 0
  ) {
    return false;
  }
  if (!cls) return true; // legacy / missing classification — show it
  return cls === "primary" || cls === "significant";
};

const fmtPct = (frac: number | null | undefined): string => {
  if (typeof frac !== "number" || !Number.isFinite(frac)) return "—";
  return `${(frac * 100).toFixed(1)}%`;
};

const validationClass = (status?: string): string => {
  const s = String(status || "").toLowerCase();
  if (s === "passed") return styles.kpiBannerPass;
  if (s === "failed") return styles.kpiBannerFail;
  return styles.kpiBannerSkipped;
};

const validationLabel = (status?: string): string => {
  const s = String(status || "").toLowerCase();
  if (s === "passed") return "PASS";
  if (s === "failed") return "FAIL";
  return "—";
};

export const KpiCostProfile: FunctionComponent<KpiCostProfileProps> = ({
  correlation,
}) => {
  const [showAllRows, setShowAllRows] = useState(false);

  if (!correlation || !correlation.key_metric) return null;

  const km = correlation.key_metric;
  const attribution = correlation.device_attribution ?? [];
  const costRows = correlation.cost_per_kpi_unit ?? [];
  const dominant = correlation.render_hints?.dominant_device;

  // Default view shows primary + significant rows. "Show all" additionally
  // reveals below-idle rows; sub-noise-floor rows are never shown because
  // they're sensor jitter, not signal.
  const visibleRows = useMemo(
    () => costRows.filter((row) => rowFilter(row, showAllRows)),
    [costRows, showAllRows],
  );
  const lowCount = costRows.filter((r) => r.signal_class === "low").length;
  // ``hiddenCount`` MUST be computed against the *default* (compact)
  // filter, not the currently-displayed set. Otherwise once the user
  // expands ("Show low-signal rows"), all rows become visible, the count
  // collapses to zero, and the "Hide low-signal rows" toggle disappears
  // — leaving the user no way to collapse the table back. By anchoring
  // to ``rowFilter(row, /* showAll */ false)`` the toggle stays
  // discoverable in both states.
  const defaultVisibleSet = useMemo(
    () => new Set(costRows.filter((r) => rowFilter(r, false))),
    [costRows],
  );
  const hiddenCount = costRows.filter(
    (r) => !defaultVisibleSet.has(r) && r.signal_class !== "low",
  ).length;

  // Stacked bar segments: filter to devices that have a measurable share.
  const stackedSegments = useMemo(
    () => attribution.filter((row) => typeof row.share_power === "number" && row.share_power! > 0),
    [attribution],
  );

  const targetText = (() => {
    if (!km.target) return null;
    const op = km.target.op || (km.direction === "higher_is_better" ? "≥" : "≤");
    return `${op} ${km.target.value} ${km.unit ?? ""}`.trim();
  })();

  const directionLabel = km.direction === "higher_is_better" ? "↑ higher is better" : "↓ lower is better";

  return (
    <div className={styles.kpiCostProfile}>
      <div className={styles.kpiCostProfileHeader}>
        <span className={styles.kpiCostProfileTitle}>
          Key Metrics ↔ Resource Cost Profile
        </span>
      </div>

      <div className={styles.kpiCostProfileBody}>
          {/* 1. KPI banner row */}
          <div className={`${styles.kpiBanner} ${validationClass(km.validation_status)}`}>
            <div className={styles.kpiBannerKey}>
              <span className={styles.kpiBannerLabel}>Key metric</span>
              <span className={styles.kpiBannerName}>{km.name}</span>
            </div>
            <div className={styles.kpiBannerValueGroup}>
              <span className={styles.kpiBannerValue}>{fmt(km.value, 2)}</span>
              <span className={styles.kpiBannerUnit}>{km.unit}</span>
            </div>
            <div className={styles.kpiBannerMeta}>
              <span className={styles.kpiBannerDirection}>{directionLabel}</span>
              {targetText && (
                <span className={styles.kpiBannerTarget}>target {targetText}</span>
              )}
              {typeof correlation.duration_s === "number" && (
                <span className={styles.kpiBannerDuration}>
                  duration {correlation.duration_s.toFixed(2)} s
                </span>
              )}
            </div>
            {(() => {
              const label = validationLabel(km.validation_status);
              // Suppress the status pill when there's no actionable PASS/FAIL
              // verdict (label is the placeholder em-dash) — the empty box
              // had no semantic value and just added visual noise.
              if (label === "—") return null;
              return (
                <div className={styles.kpiBannerStatus}>{label}</div>
              );
            })()}
          </div>

          {/* 2. Stacked resource-attribution bar */}
          {stackedSegments.length > 0 && (
            <div className={styles.kpiAttributionWrap}>
              <div className={styles.kpiAttributionTitle}>
                Power budget attribution
                {dominant && (
                  <span className={styles.kpiDominantBadge}>
                    dominant: <strong>{dominant}</strong>
                  </span>
                )}
              </div>
              <div className={styles.kpiAttributionBar} role="img" aria-label="device power share">
                {stackedSegments.map((seg) => {
                  const pct = (seg.share_power ?? 0) * 100;
                  return (
                    <div
                      key={seg.device}
                      className={styles.kpiAttributionSegment}
                      style={{
                        width: `${pct.toFixed(2)}%`,
                        background: deviceColor(seg.device),
                      }}
                      title={`${seg.device}: ${pct.toFixed(1)}% (${fmt(seg.delta_power_w, 2)} W over idle)`}
                    >
                      {pct >= 8 ? `${seg.device} ${pct.toFixed(0)}%` : ""}
                    </div>
                  );
                })}
              </div>
              <div className={styles.kpiAttributionLegend}>
                {stackedSegments.map((seg) => (
                  <span key={seg.device} className={styles.kpiAttributionLegendItem}>
                    <span
                      className={styles.kpiAttributionSwatch}
                      style={{ background: deviceColor(seg.device) }}
                      aria-hidden="true"
                    />
                    {seg.device} · ΔP {fmt(seg.delta_power_w, 2)} W · {fmtPct(seg.share_power)}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* 3. Cost-per-KPI-unit table */}
          {costRows.length > 0 && (
            <div className={styles.kpiCostTableWrap}>
              <div className={styles.kpiCostTableHeaderRow}>
                <div className={styles.kpiCostTableTitle}>
                  Cost per Key-Metric unit ({km.unit || "key metric"})
                </div>
                {hiddenCount > 0 && !showAllRows && (
                  <button
                    type="button"
                    className={styles.kpiCostTableToggle}
                    onClick={() => setShowAllRows(true)}
                    aria-expanded={false}
                    title={
                      lowCount > 0
                        ? `${hiddenCount} low-signal row(s) hidden (below idle baseline or below sensor noise floor); additional ${lowCount} sub-noise-floor row(s) permanently suppressed.`
                        : `${hiddenCount} low-signal row(s) hidden (below idle baseline or below sensor noise floor).`
                    }
                  >
                    <span className={styles.kpiCostTableToggleIcon} aria-hidden="true">+</span>
                    Show low-signal rows ({hiddenCount})
                  </button>
                )}
                {showAllRows && hiddenCount > 0 && (
                  <button
                    type="button"
                    className={styles.kpiCostTableToggle}
                    onClick={() => setShowAllRows(false)}
                    aria-expanded={true}
                    title="Collapse low-signal rows back to the default view."
                  >
                    <span className={styles.kpiCostTableToggleIcon} aria-hidden="true">−</span>
                    Hide low-signal rows ({hiddenCount})
                  </button>
                )}
              </div>
              <table className={styles.kpiCostTable}>
                <thead>
                  <tr>
                    <th>Device</th>
                    <th>Resource metric</th>
                    <th>Idle avg</th>
                    <th>Workload avg</th>
                    <th>Peak</th>
                    <th>Δ over idle</th>
                    <th>Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleRows.map((row, idx) => {
                    // When the cost can't be computed meaningfully, render
                    // "—" with a short tag explaining why instead of a
                    // misleading number.
                    //   - delta < 0  → workload drew *less* than idle baseline
                    //                  (background tasks present during prerun,
                    //                  core-parking, scheduler effects)
                    //   - delta below sensor noise floor → device sat idle
                    //                  during the workload (e.g. NPU on a
                    //                  CPU-only test)
                    const deltaBelowIdle =
                      typeof row.delta_avg === "number" && row.delta_avg < 0;
                    const costNotMeaningful =
                      row.cost_value === null ||
                      row.cost_value === undefined ||
                      !Number.isFinite(row.cost_value as number);
                    let cost: { value: string; unit: string };
                    if (deltaBelowIdle) {
                      cost = { value: "—", unit: "below idle baseline" };
                    } else if (costNotMeaningful) {
                      cost = { value: "—", unit: "below sensor noise" };
                    } else {
                      cost = formatCost(row.cost_value, row.cost_unit);
                    }
                    return (
                      <tr
                        key={`${row.device}-${row.metric}-${idx}`}
                        className={row.is_primary ? styles.kpiCostRowPrimary : undefined}
                      >
                        <td>
                          <span
                            className={styles.kpiDeviceDot}
                            style={{ background: deviceColor(row.device) }}
                            aria-hidden="true"
                          />
                          {row.device}
                        </td>
                        <td>
                          {row.metric}
                          {row.metric_unit ? (
                            <span className={styles.kpiCostUnitHint}> ({row.metric_unit})</span>
                          ) : null}
                        </td>
                        <td>{fmt(row.idle_avg, 3)}</td>
                        <td>{fmt(row.workload_avg, 3)}</td>
                        <td>{fmt(row.workload_peak, 3)}</td>
                        <td>{fmt(row.delta_avg, 3)}</td>
                        <td>
                          <strong>{cost.value}</strong>
                          <span className={styles.kpiCostUnitHint}> {cost.unit}</span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <div className={styles.kpiCostTableFootnote}>
                {km.direction === "higher_is_better"
                  ? "cost = Δ_resource ÷ KPI value (lower cost ⇒ more efficient)"
                  : "cost = Δ_resource × KPI value (lower cost ⇒ more efficient)"}
                {" · "}
                <span title={
                  "“below idle baseline” → workload drew less than the pre-run idle window " +
                  "(background tasks present at idle, or core-parking during the workload). " +
                  "“below sensor noise” → Δ within sensor accuracy (≈0.5 W power, 2% util, 50 MB/s bandwidth)."
                }>
                  cost shows “—” when Δ is negative or below sensor noise (hover for details)
                </span>
              </div>
            </div>
          )}
        </div>
    </div>
  );
};
