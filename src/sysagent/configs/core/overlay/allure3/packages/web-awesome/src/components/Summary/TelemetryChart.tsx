import { FunctionComponent } from "preact";
import { useState, useRef, useEffect } from "preact/hooks";
import * as styles from "./TelemetryChartStyle.scss";

// ─── Vivid, high-contrast palette (works in both light & dark themes) ────────
// Mid-saturation tones chosen so each series remains distinguishable on
// both white and near-black backgrounds without becoming neon.
const SERIES_COLORS = [
  "#1e88e5", // Blue 600         — vivid blue
  "#00bcd4", // Cyan 500         — bright cyan/teal
  "#43a047", // Green 600        — vivid grass green
  "#fb8c00", // Orange 600       — warm amber
  "#e53935", // Red 600          — vivid red
  "#8e24aa", // Purple 600       — vivid purple
  "#ffb300", // Amber 600        — gold
  "#26a69a", // Teal 400         — sea-green
  "#7e57c2", // Deep-Purple 400  — periwinkle
  "#ec407a", // Pink 400         — magenta
];

// ─── Layout constants ─────────────────────────────────────────────────────────
const CARD_M  = { top: 10, right: 12, bottom: 40, left: 54 };
const CARD_H  = 230;
const MODAL_M = { top: 16, right: 18, bottom: 50, left: 64 };
const MODAL_H = 520;
const MODAL_W = 980;

// Missing-data sentinel: the backend emits -1 when a source could not be
// read. Treat it like null on every render path (axis, tooltip, line gap,
// bar, threshold).
const MISSING_VALUE = -1;
const isMissing = (v: unknown): boolean =>
  v == null || (typeof v === "number" && (!Number.isFinite(v) || v === MISSING_VALUE));

// ─── Types ────────────────────────────────────────────────────────────────────
export interface TelemetrySample {
  timestamp: number;
  values: Record<string, number>;
}
export interface TelemetryScaleConfig {
  display?: boolean;
  label?: string;
  unit?: string;
}
export interface TelemetryTitleConfig {
  display?: boolean;
  text?: string;
}
export interface TelemetryAxisConfig {
  id: string;
  position: "left" | "right";
  metrics: string[];
  label?: string;
}
interface TtEntry {
  key: string; label: string; value: number | null; unit: string; color: string;
}
interface Tooltip {
  svgX: number; cX: number; cY: number; time: string; entries: TtEntry[];
}
export interface TelemetryBaselineMetric {
  min?: number;
  avg?: number;
  max?: number;
}
export interface TelemetryChartProps {
  moduleName: string;
  chartType: "line" | "area" | "bar_vertical";
  samples: TelemetrySample[];
  averages?: Record<string, number>;
  minMax?: Record<string, { min: number; max: number }>;
  thresholds?: Record<string, { warning?: number }>;
  visibleMetrics?: string[];
  configTitle?: TelemetryTitleConfig;
  configScales?: Record<string, TelemetryScaleConfig>;
  axes?: TelemetryAxisConfig[];
  /**
   * Optional pre-run idle baseline stats per metric key.  When supplied, the
   * chart overlays a horizontal reference band (min..max) and a dashed avg
   * line on the workload's y-axis so users can compare load against idle.
   * The overlay is duration-independent (purely y-axis), so any test/idle
   * duration ratio works.
   */
  baseline?: Record<string, TelemetryBaselineMetric>;
  summaryMeta?: { cliName: string; platform: string; timestamp: string } | null;
  testId?: string;
  compact?: boolean;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function fmtY(v: number): string {
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
  if (Math.abs(v) < 10) return v.toFixed(1);
  return v.toFixed(0);
}

/**
 * Range-aware tick formatter.  When 5 evenly-spaced ticks span a small range
 * (e.g. 46.3 .. 47.9 W) the default integer rounding produces visible
 * duplicates ("46, 47, 47, 48, 48"). This formatter picks decimal precision
 * based on the tick *step* so adjacent ticks always render distinctly.
 */
function fmtYRange(v: number, step: number): string {
  const absV = Math.abs(v);
  if (absV >= 1000) return `${(v / 1000).toFixed(1)}k`;
  const s = Math.abs(step) || 0;
  // Decimal places ~ ceil(-log10(step)); clamp to 0..3 for readability.
  let decimals = 0;
  if (s > 0) {
    decimals = Math.min(3, Math.max(0, Math.ceil(-Math.log10(s)) + 0));
  } else if (absV < 10) {
    decimals = 1;
  }
  // Ensure small magnitudes still show a fractional digit when step rounds to 0 places.
  if (decimals === 0 && absV < 10 && s < 1) decimals = 1;
  return v.toFixed(decimals);
}

function resolveSeriesColor(metricKey: string, index: number): string {
  const key = String(metricKey || "").toLowerCase();
  if (key.includes("temperature")) return "#d94841";
  if (key.includes("power")) return "#2d9c74";
  if (key.includes("memory") && key.includes("utilization")) return "#7b61c8";
  if (key.includes("bandwidth")) return "#0f9fb8";
  if (key.includes("frequency")) return "#2878c8";
  if (key.includes("utilization")) return "#f08c2e";
  return SERIES_COLORS[index % SERIES_COLORS.length];
}

function exportFullPng(
  svgEl: SVGSVGElement,
  w: number,
  chartH: number,
  filename: string,
  title: string,
  metricKeys: string[],
  chartType: "line" | "area" | "bar_vertical",
  configScales?: Record<string, TelemetryScaleConfig>,
  averages?: Record<string, number>,
  minMax?: Record<string, { min: number; max: number }>,
  baseline?: Record<string, TelemetryBaselineMetric>,
) {
  const FONT         = "Arial, Helvetica, sans-serif";
  const TITLE_H      = 36;
  const hasStats     = metricKeys.some(k =>
    averages?.[k] !== undefined
    || minMax?.[k] !== undefined
    || baseline?.[k]?.avg !== undefined
  );
  // Each legend item rendered in kpi-card style needs:
  //   row 1: series symbol + metric label                     (16px)
  //   row 2: avg-pill + value + unit + Min/Max line           (22px)
  //   row 3: Idle + Over-Idle delta pill                      (18px)
  // Without stats we fall back to a single 26px row.
  const ROW_H        = hasStats ? 64 : 26;
  const LEG_PAD      = 16;
  const EXTRA_RIGHT  = 64;  // room for "warn" threshold labels that overflow the chart right edge
  const EXTRA_BOTTOM = 16;  // breathing room between bottom x-axis label and legend
  const PER_ROW      = Math.min(metricKeys.length, 2);  // max 2 per row to prevent label overlap
  const legRows      = Math.ceil(metricKeys.length / PER_ROW);
  const LEG_H        = LEG_PAD + legRows * ROW_H + LEG_PAD;
  const exportW      = w + EXTRA_RIGHT;
  const legStartY    = TITLE_H + chartH + EXTRA_BOTTOM;
  const totalH       = legStartY + LEG_H;
  const ns           = "http://www.w3.org/2000/svg";

  const root = document.createElementNS(ns, "svg");
  root.setAttribute("xmlns", ns);
  root.setAttribute("width", String(exportW));
  root.setAttribute("height", String(totalH));
  root.setAttribute("font-family", FONT);

  // White background
  const bg = document.createElementNS(ns, "rect");
  bg.setAttribute("width", String(exportW)); bg.setAttribute("height", String(totalH)); bg.setAttribute("fill", "#fff");
  root.appendChild(bg);

  // Chart title
  const tEl = document.createElementNS(ns, "text");
  tEl.setAttribute("x", String(exportW / 2)); tEl.setAttribute("y", "24");
  tEl.setAttribute("text-anchor", "middle"); tEl.setAttribute("font-size", "14");
  tEl.setAttribute("font-weight", "600"); tEl.setAttribute("fill", "#1a1a1a");
  tEl.textContent = title;
  root.appendChild(tEl);

  // Chart content (shifted down below title)
  const chartG = document.createElementNS(ns, "g");
  chartG.setAttribute("transform", `translate(0,${TITLE_H})`);
  const clone = svgEl.cloneNode(true) as SVGSVGElement;
  while (clone.firstChild) chartG.appendChild(clone.firstChild);
  root.appendChild(chartG);

  // Idle baseline legend chip — pinned to the top-right of the chart
  // area so the downloaded PNG carries the same visual reference the
  // user sees in the on-screen card / full-view (a small dashed line
  // labelled "Idle"). Only rendered when at least one metric in this
  // chart has an idle baseline value.
  const hasIdleBaseline = baseline
    ? metricKeys.some((k) => typeof baseline[k]?.avg === "number")
    : false;
  if (hasIdleBaseline) {
    const chipPadX  = 6;
    const chipPadY  = 3;
    const chipFont  = 10;
    const labelText = "Idle";
    const labelW    = labelText.length * 6 + 2;   // approx width @ font-size 10
    const swatchW   = 14;
    const swatchGap = 5;
    const chipW     = chipPadX * 2 + swatchW + swatchGap + labelW;
    const chipH     = chipFont + chipPadY * 2;
    // Anchor 8px in from the right edge of the original chart width and
    // 6px down from the top of the chart area (matches the on-screen
    // ``.idleChip`` offsets).
    const chipX     = w - chipW - 8;
    const chipY     = TITLE_H + 6;

    const chipBg = document.createElementNS(ns, "rect");
    chipBg.setAttribute("x", String(chipX));
    chipBg.setAttribute("y", String(chipY));
    chipBg.setAttribute("width", String(chipW));
    chipBg.setAttribute("height", String(chipH));
    chipBg.setAttribute("rx", "10");
    chipBg.setAttribute("fill", "rgba(255, 255, 255, 0.92)");
    chipBg.setAttribute("stroke", "#d8e0e8");
    chipBg.setAttribute("stroke-width", "1");
    root.appendChild(chipBg);

    // Dashed swatch — same dash pattern as the in-chart baseline line.
    const swatchY = chipY + chipH / 2;
    const swatch  = document.createElementNS(ns, "line");
    swatch.setAttribute("x1", String(chipX + chipPadX));
    swatch.setAttribute("y1", String(swatchY));
    swatch.setAttribute("x2", String(chipX + chipPadX + swatchW));
    swatch.setAttribute("y2", String(swatchY));
    swatch.setAttribute("stroke", "#1f2d3d");
    swatch.setAttribute("stroke-width", "1.5");
    swatch.setAttribute("stroke-dasharray", "3 2");
    swatch.setAttribute("stroke-opacity", "0.9");
    root.appendChild(swatch);

    const lbl = document.createElementNS(ns, "text");
    lbl.setAttribute("x", String(chipX + chipPadX + swatchW + swatchGap));
    lbl.setAttribute("y", String(swatchY));
    lbl.setAttribute("dominant-baseline", "middle");
    lbl.setAttribute("font-size", String(chipFont));
    lbl.setAttribute("font-weight", "600");
    lbl.setAttribute("fill", "#1f2d3d");
    lbl.textContent = labelText;
    root.appendChild(lbl);
  }

  // Legend separator
  const sep = document.createElementNS(ns, "line");
  sep.setAttribute("x1", "16"); sep.setAttribute("y1", String(legStartY + 2));
  sep.setAttribute("x2", String(exportW - 16)); sep.setAttribute("y2", String(legStartY + 2));
  sep.setAttribute("stroke", "#e0e0e0"); sep.setAttribute("stroke-width", "1");
  root.appendChild(sep);

  // Legend items
  const ITEM_W = (exportW - 32) / PER_ROW;
  const baseY  = legStartY + LEG_PAD;

  metricKeys.forEach((key, i) => {
    const color      = resolveSeriesColor(key, i);
    const label      = configScales?.[key]?.label ?? key.replace(/_/g, " ");
    const unit       = configScales?.[key]?.unit;
    const avg        = averages?.[key];
    const mm         = minMax?.[key];
    const idleAvg    = baseline?.[key]?.avg;
    const unitSuffix = unit ? ` ${unit}` : "";
    const row        = Math.floor(i / PER_ROW);
    const col        = i % PER_ROW;
    const ix         = 16 + col * ITEM_W;
    const labelY     = baseY + row * ROW_H + (hasStats ? 13 : ROW_H / 2);

    // Symbol rect
    const sym = document.createElementNS(ns, "rect");
    if (chartType === "line") {
      sym.setAttribute("x", String(ix)); sym.setAttribute("y", String(labelY - 1));
      sym.setAttribute("width", "16"); sym.setAttribute("height", "2"); sym.setAttribute("rx", "1");
    } else if (chartType === "area") {
      sym.setAttribute("x", String(ix)); sym.setAttribute("y", String(labelY - 4));
      sym.setAttribute("width", "12"); sym.setAttribute("height", "8"); sym.setAttribute("rx", "1");
      sym.setAttribute("opacity", "0.65");
    } else {
      sym.setAttribute("x", String(ix)); sym.setAttribute("y", String(labelY - 5));
      sym.setAttribute("width", "10"); sym.setAttribute("height", "10"); sym.setAttribute("rx", "1");
      sym.setAttribute("opacity", "0.82");
    }
    sym.setAttribute("fill", color);
    root.appendChild(sym);

    // Label text
    const txt = document.createElementNS(ns, "text");
    txt.setAttribute("x", String(ix + 22)); txt.setAttribute("y", String(labelY));
    txt.setAttribute("dominant-baseline", "middle");
    txt.setAttribute("font-size", "11"); txt.setAttribute("fill", "#333");
    txt.textContent = `${label}${unit ? ` (${unit})` : ""}`;
    root.appendChild(txt);

    // kpi-card style stats block (rows 2 & 3 of the legend item) ── mirrors
    // the per-metric ``kpiCard`` rendered in ``TelemetrySection`` so the
    // downloaded PNG carries the same numeric context the user sees in
    // the report. Layout:
    //   row 2: [avg pill] <big value> <unit>     Min ..   Max ..
    //   row 3: Idle ..    [Over Idle] avg +N% \u00B7 peak +N%
    if (hasStats && (avg !== undefined || mm || idleAvg !== undefined)) {
      const ROW2_Y = baseY + row * ROW_H + 28;
      const ROW3_Y = baseY + row * ROW_H + 50;

      // ── Row 2: avg pill + value + unit + Min/Max ─────────────────────
      let cursorX = ix;
      if (avg !== undefined) {
        // "avg" pill (mirrors ``.kpiAggregateTag``)
        const pillW = 22;
        const pillH = 12;
        const pillBg = document.createElementNS(ns, "rect");
        pillBg.setAttribute("x", String(cursorX));   pillBg.setAttribute("y", String(ROW2_Y - 9));
        pillBg.setAttribute("width", String(pillW)); pillBg.setAttribute("height", String(pillH));
        pillBg.setAttribute("rx", "2");
        pillBg.setAttribute("fill", "#1f2d3d");
        root.appendChild(pillBg);

        const pillTxt = document.createElementNS(ns, "text");
        pillTxt.setAttribute("x", String(cursorX + pillW / 2));
        pillTxt.setAttribute("y", String(ROW2_Y - 3));
        pillTxt.setAttribute("text-anchor", "middle");
        pillTxt.setAttribute("dominant-baseline", "middle");
        pillTxt.setAttribute("font-size", "8");
        pillTxt.setAttribute("font-weight", "700");
        pillTxt.setAttribute("fill", "#fff");
        pillTxt.textContent = "avg";
        root.appendChild(pillTxt);
        cursorX += pillW + 4;

        // big value (mirrors ``.kpiValue``)
        const valTxt = document.createElementNS(ns, "text");
        valTxt.setAttribute("x", String(cursorX));   valTxt.setAttribute("y", String(ROW2_Y));
        valTxt.setAttribute("dominant-baseline", "middle");
        valTxt.setAttribute("font-size", "16");
        valTxt.setAttribute("font-weight", "700");
        valTxt.setAttribute("fill", "#1f2d3d");
        const valStr = avg.toFixed(2);
        valTxt.textContent = valStr;
        root.appendChild(valTxt);
        cursorX += valStr.length * 9;

        // unit (mirrors ``.kpiUnit``)
        if (unit) {
          const unitTxt = document.createElementNS(ns, "text");
          unitTxt.setAttribute("x", String(cursorX));   unitTxt.setAttribute("y", String(ROW2_Y));
          unitTxt.setAttribute("dominant-baseline", "middle");
          unitTxt.setAttribute("font-size", "10");
          unitTxt.setAttribute("fill", "#5a6b7a");
          unitTxt.textContent = unit;
          root.appendChild(unitTxt);
          cursorX += String(unit).length * 6 + 8;
        }
      }
      // Min / Max (mirrors ``.kpiStatsRow``)
      if (mm) {
        const stTxt = document.createElementNS(ns, "text");
        stTxt.setAttribute("x", String(cursorX + 8));   stTxt.setAttribute("y", String(ROW2_Y));
        stTxt.setAttribute("dominant-baseline", "middle");
        stTxt.setAttribute("font-size", "10");
        stTxt.setAttribute("fill", "#5a6b7a");
        stTxt.textContent = `Min ${mm.min.toFixed(2)}${unitSuffix}    Max ${mm.max.toFixed(2)}${unitSuffix}`;
        root.appendChild(stTxt);
      }

      // ── Row 3: Idle + Over Idle delta pill ────────────────────────────
      const idleX = ix;
      let row3Cursor = idleX;
      if (idleAvg !== undefined) {
        const idleTxt = document.createElementNS(ns, "text");
        idleTxt.setAttribute("x", String(row3Cursor));  idleTxt.setAttribute("y", String(ROW3_Y));
        idleTxt.setAttribute("dominant-baseline", "middle");
        idleTxt.setAttribute("font-size", "10");
        idleTxt.setAttribute("font-weight", "600");
        idleTxt.setAttribute("fill", "#37474f");
        const idleStr = `Idle ${idleAvg.toFixed(2)}${unitSuffix}`;
        idleTxt.textContent = idleStr;
        root.appendChild(idleTxt);
        row3Cursor += idleStr.length * 6 + 12;
      }
      // Over Idle delta pill
      const formatDelta = (loadValue: number | undefined): string | null => {
        if (typeof idleAvg !== "number" || typeof loadValue !== "number") return null;
        const diff = loadValue - idleAvg;
        if (Math.abs(idleAvg) >= 0.1) {
          const pct  = (diff / idleAvg) * 100;
          const sign = pct >= 0 ? "+" : "";
          return `${sign}${pct.toFixed(0)}%`;
        }
        const sign = diff >= 0 ? "+" : "";
        return `${sign}${diff.toFixed(2)}${unitSuffix}`;
      };
      const avgD  = avg     !== undefined ? formatDelta(avg)     : null;
      const peakD = mm?.max !== undefined ? formatDelta(mm.max)  : null;
      if (avgD || peakD) {
        const parts: string[] = [];
        if (avgD)  parts.push(`avg ${avgD}`);
        if (peakD) parts.push(`peak ${peakD}`);
        const deltaText = parts.join("  \u00B7  ");

        const tagW = 50, tagH = 12;
        const valApproxW = deltaText.length * 5.5 + 6;
        const boxW = tagW + 6 + valApproxW + 6;
        const boxH = 16;

        // Box (mirrors ``.kpiBaselineDeltaBox``)
        const box = document.createElementNS(ns, "rect");
        box.setAttribute("x", String(row3Cursor));         box.setAttribute("y", String(ROW3_Y - boxH / 2));
        box.setAttribute("width", String(boxW));           box.setAttribute("height", String(boxH));
        box.setAttribute("rx", "3");
        box.setAttribute("fill", "#f7f9fb");
        box.setAttribute("stroke", "#d8e0e8");
        box.setAttribute("stroke-width", "0.6");
        root.appendChild(box);

        // Inner "Over Idle" tag (mirrors ``.kpiBaselineDeltaTag``)
        const tag = document.createElementNS(ns, "rect");
        tag.setAttribute("x", String(row3Cursor + 3));     tag.setAttribute("y", String(ROW3_Y - tagH / 2));
        tag.setAttribute("width", String(tagW));           tag.setAttribute("height", String(tagH));
        tag.setAttribute("rx", "2");
        tag.setAttribute("fill", "#1f2d3d");
        root.appendChild(tag);

        const tagTxt = document.createElementNS(ns, "text");
        tagTxt.setAttribute("x", String(row3Cursor + 3 + tagW / 2));
        tagTxt.setAttribute("y", String(ROW3_Y));
        tagTxt.setAttribute("text-anchor", "middle");
        tagTxt.setAttribute("dominant-baseline", "middle");
        tagTxt.setAttribute("font-size", "8");
        tagTxt.setAttribute("font-weight", "700");
        tagTxt.setAttribute("fill", "#fff");
        tagTxt.textContent = "Over Idle";
        root.appendChild(tagTxt);

        const valTxt = document.createElementNS(ns, "text");
        valTxt.setAttribute("x", String(row3Cursor + 3 + tagW + 6));
        valTxt.setAttribute("y", String(ROW3_Y));
        valTxt.setAttribute("dominant-baseline", "middle");
        valTxt.setAttribute("font-size", "10");
        valTxt.setAttribute("font-weight", "600");
        valTxt.setAttribute("fill", "#1f2d3d");
        valTxt.textContent = deltaText;
        root.appendChild(valTxt);
      }
    }
  });

  // Serialize → canvas → download at 2× resolution
  const svgStr = new XMLSerializer().serializeToString(root);
  const url    = URL.createObjectURL(
    new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" }),
  );
  const img = new Image();
  img.onload = () => {
    const scale = 2;
    const cv    = document.createElement("canvas");
    cv.width = exportW * scale; cv.height = totalH * scale;
    const ctx = cv.getContext("2d");
    if (!ctx) { URL.revokeObjectURL(url); return; }
    ctx.scale(scale, scale); ctx.drawImage(img, 0, 0);
    const a = document.createElement("a");
    a.download = filename; a.href = cv.toDataURL("image/png"); a.click();
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

// ─── Component ────────────────────────────────────────────────────────────────
export const TelemetryChart: FunctionComponent<TelemetryChartProps> = ({
  moduleName, chartType, samples, averages, minMax, thresholds,
  visibleMetrics, configTitle, configScales, axes, baseline,
  summaryMeta, testId, compact = false,
}) => {
  // All hooks must come before any conditional return ──────────────────────
  const containerRef  = useRef<HTMLDivElement>(null);
  const cardSvgRef    = useRef<SVGSVGElement>(null);
  const modalSvgRef   = useRef<SVGSVGElement>(null);
  const modalChartRef = useRef<HTMLDivElement>(null);

  const [cardW, setCardW]         = useState(400);
  const [cardTt, setCardTt]       = useState<Tooltip | null>(null);
  const [modalTt, setModalTt]     = useState<Tooltip | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setCardW(w);
    });
    ro.observe(el);
    setCardW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!modalOpen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setModalOpen(false); };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [modalOpen]);

  // ── Guard ────────────────────────────────────────────────────────────────
  if (!samples || samples.length === 0) return null;
  // Collect all keys across all samples — the first sample may have `values: {}`
  const allKeys = Array.from(new Set(samples.flatMap((s) => Object.keys(s.values))));
  const metricKeys = visibleMetrics?.length ? allKeys.filter((k) => visibleMetrics.includes(k)) : allKeys;
  if (metricKeys.length === 0) return null;

  // ── Derived values ───────────────────────────────────────────────────────

  // ── Multi-axis setup ─────────────────────────────────────────────────────
  const leftAxisCfg  = axes?.find(a => a.position === "left");
  const rightAxisCfg = axes?.find(a => a.position === "right");

  // Map each visible metric key to its axis side (default: left)
  const metricAxisMap: Record<string, "left" | "right"> = {};
  metricKeys.forEach(k => { metricAxisMap[k] = "left"; });
  if (axes && axes.length > 0) {
    axes.forEach(ax => ax.metrics.forEach(m => {
      if (metricKeys.includes(m)) metricAxisMap[m] = ax.position as "left" | "right";
    }));
  }

  const leftMetrics  = metricKeys.filter(k => metricAxisMap[k] !== "right");
  const rightMetrics = metricKeys.filter(k => metricAxisMap[k] === "right");
  const hasRightAxis = rightMetrics.length > 0;

  // Pre-run idle baseline avg per axis side. Including it in the y-domain
  // guarantees the dashed reference line is always visible inside the chart.
  // (min/max are summarized in the KPI card, not drawn on the plot.)
  const baselineExtrema = (keys: string[]): number[] => {
    if (!baseline) return [];
    return keys.flatMap(k => {
      const b = baseline[k];
      return b && typeof b.avg === "number" ? [b.avg] : [];
    });
  };

  // Left Y domain (include warning thresholds + baseline so all overlays stay in range)
  // Exclude missing samples so they don't drag the axis below zero.
  const leftVals = samples.flatMap(s => leftMetrics.flatMap(k => {
    const v = s.values[k];
    return isMissing(v) ? [] : [v as number];
  }));
  const leftThreshVals = thresholds
    ? Object.entries(thresholds as Record<string, { warning?: number }>).flatMap(([k, t]) =>
        leftMetrics.includes(k) && t.warning !== undefined ? [t.warning] : [])
    : [];
  const leftBaselineVals = baselineExtrema(leftMetrics);
  const leftRawMin = leftVals.length > 0 ? Math.min(...leftVals, ...leftBaselineVals) : (leftBaselineVals.length > 0 ? Math.min(...leftBaselineVals) : 0);
  const leftRawMax = leftVals.length > 0 ? Math.max(...leftVals, ...leftThreshVals, ...leftBaselineVals) : (leftBaselineVals.length > 0 ? Math.max(...leftBaselineVals) : 1);
  const leftPad    = (leftRawMax - leftRawMin) * 0.1 || 1;
  const leftYMin   = Math.max(0, leftRawMin - leftPad);
  const leftYMax   = leftRawMax + leftPad;

  // Right Y domain (only meaningful when right-axis metrics exist)
  const rightVals = hasRightAxis ? samples.flatMap(s => rightMetrics.flatMap(k => {
    const v = s.values[k];
    return isMissing(v) ? [] : [v as number];
  })) : [];
  const rightThreshVals = hasRightAxis && thresholds
    ? Object.entries(thresholds as Record<string, { warning?: number }>).flatMap(([k, t]) =>
        rightMetrics.includes(k) && t.warning !== undefined ? [t.warning] : [])
    : [];
  const rightBaselineVals = hasRightAxis ? baselineExtrema(rightMetrics) : [];
  const rightRawMin = rightVals.length > 0 ? Math.min(...rightVals, ...rightBaselineVals) : (rightBaselineVals.length > 0 ? Math.min(...rightBaselineVals) : 0);
  const rightRawMax = rightVals.length > 0 ? Math.max(...rightVals, ...rightThreshVals, ...rightBaselineVals) : (rightBaselineVals.length > 0 ? Math.max(...rightBaselineVals) : 1);
  const rightPad    = (rightRawMax - rightRawMin) * 0.1 || 1;
  const rightYMin   = Math.max(0, rightRawMin - rightPad);
  const rightYMax   = rightRawMax + rightPad;

  // Per-axis unit labels (shown in tick values when no axis label overrides them)
  const leftAxisUnit  = leftMetrics.length > 0 &&
    leftMetrics.every(k => (configScales?.[k]?.unit ?? "") === (configScales?.[leftMetrics[0]]?.unit ?? ""))
    ? (configScales?.[leftMetrics[0]]?.unit ?? "") : "";
  const rightAxisUnit = rightMetrics.length > 0 &&
    rightMetrics.every(k => (configScales?.[k]?.unit ?? "") === (configScales?.[rightMetrics[0]]?.unit ?? ""))
    ? (configScales?.[rightMetrics[0]]?.unit ?? "") : "";

  // Derive a meaningful axis label from axes config, scale label+unit, or unit alone.
  // When the axis label is shown, tick values omit the unit suffix to avoid duplication.
  const deriveAxisLabel = (
    axisConfig: TelemetryAxisConfig | undefined,
    metrics: string[],
    unit: string,
  ): string => {
    if (axisConfig?.label) return axisConfig.label;
    if (metrics.length === 0) return unit;
    const firstLabel = configScales?.[metrics[0]]?.label ?? "";
    const sharedLabel = firstLabel && metrics.every(k => (configScales?.[k]?.label ?? "") === firstLabel)
      ? firstLabel : "";
    if (sharedLabel) return unit ? `${sharedLabel} (${unit})` : sharedLabel;
    return unit;
  };

  const leftAxisLabel  = deriveAxisLabel(leftAxisCfg,  leftMetrics,  leftAxisUnit);
  const rightAxisLabel = deriveAxisLabel(rightAxisCfg, rightMetrics, rightAxisUnit);

  const minTs = samples[0].timestamp;
  const maxTs = samples[samples.length - 1].timestamp;

  // X-axis: auto-switch to minutes when total elapsed > 60 s
  const totalDuration = maxTs - minTs; // seconds
  const useMinutes    = totalDuration > 60;
  const xTimeUnit     = useMinutes ? "min" : "s";
  const fmtElapsed    = (ts: number) => {
    const elapsed = ts - minTs;
    return useMinutes ? (elapsed / 60).toFixed(1) : elapsed.toFixed(0);
  };

  // Effective margins: widen right side when a second Y-axis is active;
  // widen left side to accommodate rotated Y-axis label in single-axis mode too.
  const RIGHT_AXIS_W  = 46; // px reserved for right-axis ticks + rotated label
  const LEFT_LABEL_W  = 0;  // rotated label fits within base CARD_M/MODAL_M left margin
  const cardMargin    = {
    ...CARD_M,
    left:  CARD_M.left  + LEFT_LABEL_W,
    right: hasRightAxis ? RIGHT_AXIS_W : CARD_M.right,
  };
  const modalMargin   = {
    ...MODAL_M,
    left:  MODAL_M.left  + LEFT_LABEL_W,
    right: hasRightAxis ? RIGHT_AXIS_W : MODAL_M.right,
  };

  const fallbackTitle = moduleName.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  const exportTitle   = configTitle?.text || fallbackTitle;
  const displayName   = configTitle?.display === false ? "" : exportTitle;
  // Filename-safe sanitiser: lower-case, collapse any non-alphanumeric
  // run to a single underscore, trim leading/trailing underscores.
  const sanitize      = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  const safeModule    = sanitize(moduleName);
  // Per-chart slug — when ``configTitle.text`` is supplied (e.g.
  // ``"CPU Utilization"`` from the per-metric kpiCard) we use it as a
  // compact, human-readable identifier (e.g. ``cpu_utilization``)
  // instead of the verbose module name. Falls back to the module name
  // for the aggregate Metrics-Graphs view.
  const safeChartId   = configTitle?.text ? sanitize(configTitle.text) : safeModule;
  // Filename intentionally drops the platform string — it tends to be
  // very long ("gigabyte_technology__z890_aorus_master_ai_core_ultra_9_285k_b580")
  // and bloats the filename. Test ID + chart slug + timestamp are
  // sufficient to uniquely identify a chart from a given test run.
  const pngFilename   = summaryMeta && testId
    ? `${sanitize(summaryMeta.cliName)}_telemetry_${sanitize(testId)}_${safeChartId}_${sanitize(summaryMeta.timestamp)}.png`
    : `${safeChartId}_telemetry.png`;

  // ── Tooltip builder ──────────────────────────────────────────────────────
  const buildTooltip = (
    e: MouseEvent,
    svgEl: SVGSVGElement,
    containerEl: HTMLElement,
    width: number,
    margin: typeof CARD_M,
  ): Tooltip | null => {
    const innerW    = Math.max(10, width - margin.left - margin.right);
    const barGroupW = innerW / samples.length;
    const svgRect   = svgEl.getBoundingClientRect();
    const cRect     = containerEl.getBoundingClientRect();
    const mouseX    = e.clientX - svgRect.left;
    const mouseY    = e.clientY - cRect.top;

    const xScale = (ts: number) =>
      maxTs === minTs
        ? margin.left + innerW / 2
        : margin.left + ((ts - minTs) / (maxTs - minTs)) * innerW;

    let idx = 0;
    if (chartType === "bar_vertical") {
      idx = Math.max(0, Math.min(samples.length - 1, Math.floor((mouseX - margin.left) / barGroupW)));
    } else {
      let minD = Infinity;
      samples.forEach((s, i) => {
        const d = Math.abs(xScale(s.timestamp) - mouseX);
        if (d < minD) { minD = d; idx = i; }
      });
    }

    const s       = samples[idx];
    const cursorX = chartType === "bar_vertical"
      ? margin.left + idx * barGroupW + barGroupW / 2
      : xScale(s.timestamp);

    return {
      svgX: cursorX, cX: cursorX, cY: mouseY,
      time: new Date(s.timestamp * 1000).toLocaleTimeString(),
      entries: metricKeys.map((k, i) => ({
        key:   k,
        label: configScales?.[k]?.label ?? k.replace(/_/g, " "),
        value: isMissing(s.values[k]) ? null : (s.values[k] as number),
        unit:  configScales?.[k]?.unit ?? "",
          color: resolveSeriesColor(k, i),
      })),
    };
  };

  // ── Tooltip JSX ─────────────────────────────────────────────────────────
  const Tip = ({ tt, containerWidth, isModal = false }: { tt: Tooltip; containerWidth: number; isModal?: boolean }) => (
    <div
      className={styles.tooltip}
      style={{
        left: tt.cX > containerWidth / 2 ? `${tt.cX - 175}px` : `${tt.cX + 10}px`,
        top: `${Math.max(4, tt.cY - 28)}px`,
      }}
    >
      <div className={styles.tooltipTime} style={isModal ? { fontSize: "10px" } : undefined}>{tt.time}</div>
      {tt.entries.map((v) => (
        <div key={v.key} className={styles.tooltipRow}>
          <span className={styles.tooltipDot} style={{ background: v.color }} />
          <span className={styles.tooltipKey} style={isModal ? { fontSize: "10px" } : undefined}>{v.label}</span>
          <span className={styles.tooltipVal} style={isModal ? { fontSize: "11px" } : undefined}>{v.value != null ? `${v.value.toFixed(2)}${v.unit ? ` ${v.unit}` : ""}` : "—"}</span>
        </div>
      ))}
    </div>
  );

  // ── Legend JSX ───────────────────────────────────────────────────────────
  const Legend = () => (
    <div className={styles.legend}>
      {metricKeys.map((key, i) => {
        const color      = resolveSeriesColor(key, i);
        const label      = configScales?.[key]?.label ?? key.replace(/_/g, " ");
        const unit       = configScales?.[key]?.unit;
        const avg        = averages?.[key];
        const mm         = minMax?.[key];
        const idleAvg    = baseline?.[key]?.avg;
        const unitSuffix = unit ? ` ${unit}` : "";
        return (
          <div key={key} className={styles.legendItem}>
            <div className={styles.legendItemHeader}>
              {chartType === "line" && <span className={styles.legendLine} style={{ background: color }} />}
              {chartType === "area" && <span className={styles.legendArea} style={{ background: color }} />}
              {chartType === "bar_vertical" && <span className={styles.legendBar} style={{ background: color }} />}
              <span className={styles.legendLabel}>{label}{unit ? ` (${unit})` : ""}</span>
            </div>
            {(avg !== undefined || mm || idleAvg !== undefined) && (
              <div className={styles.legendStatGroup}>
                {mm && (
                  <span className={`${styles.legendStat} ${styles.legendStatMin}`}>
                    <span className={styles.legendStatLabel}>min</span>
                    <span className={styles.legendStatValue}>{mm.min.toFixed(2)}{unitSuffix}</span>
                  </span>
                )}
                {avg !== undefined && (
                  <span className={`${styles.legendStat} ${styles.legendStatAvg}`}>
                    <span className={styles.legendStatLabel}>avg</span>
                    <span className={styles.legendStatValue}>{avg.toFixed(2)}{unitSuffix}</span>
                  </span>
                )}
                {mm && (
                  <span className={`${styles.legendStat} ${styles.legendStatMax}`}>
                    <span className={styles.legendStatLabel}>max</span>
                    <span className={styles.legendStatValue}>{mm.max.toFixed(2)}{unitSuffix}</span>
                  </span>
                )}
                {idleAvg !== undefined && (
                  <span className={`${styles.legendStat} ${styles.legendStatIdle}`}>
                    <span className={styles.legendStatLabel}>idle</span>
                    <span className={styles.legendStatValue}>{idleAvg.toFixed(2)}{unitSuffix}</span>
                  </span>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );

  // ── SVG chart renderer ───────────────────────────────────────────────────
  const ChartSVG = ({
    width, chartH, margin, svgRef, tooltip, onMouseMove, onMouseLeave, tickFontSize = 10,
  }: {
    width: number;
    chartH: number;
    margin: typeof CARD_M;
    svgRef: { current: SVGSVGElement | null };
    tooltip: Tooltip | null;
    onMouseMove: (e: MouseEvent) => void;
    onMouseLeave: () => void;
    tickFontSize?: number;
  }) => {
    const innerW    = Math.max(10, width - margin.left - margin.right);
    const innerH    = chartH - margin.top - margin.bottom;
    const barGroupW = innerW / samples.length;

    const xScale = (ts: number) =>
      maxTs === minTs
        ? margin.left + innerW / 2
        : margin.left + ((ts - minTs) / (maxTs - minTs)) * innerW;

    // Per-axis Y scale factories
    const makeYScale = (yMin: number, yMax: number) => (val: number) =>
      margin.top + innerH - ((val - yMin) / (yMax - yMin)) * innerH;

    const yScaleLeft  = makeYScale(leftYMin, leftYMax);
    const yScaleRight = hasRightAxis ? makeYScale(rightYMin, rightYMax) : yScaleLeft;
    const yScaleFor   = (key: string) => metricAxisMap[key] === "right" ? yScaleRight : yScaleLeft;
    const yMinFor     = (key: string) => metricAxisMap[key] === "right" ? rightYMin : leftYMin;

    const leftTicks  = Array.from({ length: 5 }, (_, i) => leftYMin  + (i / 4) * (leftYMax  - leftYMin));
    const rightTicks = hasRightAxis
      ? Array.from({ length: 5 }, (_, i) => rightYMin + (i / 4) * (rightYMax - rightYMin))
      : [];
    // Tick step drives decimal precision so adjacent labels never duplicate
    // (e.g. avoid "46, 47, 47, 48, 48" when the range is small).
    const leftTickStep  = (leftYMax  - leftYMin)  / 4;
    const rightTickStep = (rightYMax - rightYMin) / 4;

    const maxXT = Math.min(7, samples.length);
    const xTickIdxs =
      samples.length <= maxXT
        ? samples.map((_, i) => i)
        : Array.from({ length: maxXT }, (_, i) =>
            Math.round((i / (maxXT - 1)) * (samples.length - 1)),
          );

    // Build line/area paths as contiguous sub-paths separated by gaps.
    // Missing samples (null/undefined or -1) break the line so users can
    // tell unavailable data apart from a genuine 0 reading.
    type Segment = Array<{ x: number; y: number }>;
    const buildSegments = (key: string): Segment[] => {
      const ys = yScaleFor(key);
      const segs: Segment[] = [];
      let current: Segment = [];
      for (const s of samples) {
        const v = s.values[key];
        if (isMissing(v)) {
          if (current.length > 0) {
            segs.push(current);
            current = [];
          }
          continue;
        }
        current.push({ x: xScale(s.timestamp), y: ys(v as number) });
      }
      if (current.length > 0) segs.push(current);
      return segs;
    };

    const linePath = (key: string) =>
      buildSegments(key)
        .map(seg => seg.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" "))
        .join(" ");

    const areaPath = (key: string) => {
      const ys    = yScaleFor(key);
      const baseY = ys(yMinFor(key)).toFixed(1);
      // Close each contiguous segment back to the baseline so gaps remain
      // visually empty instead of being filled by a single sweeping polygon.
      return buildSegments(key)
        .map(seg => {
          if (seg.length === 0) return "";
          const top = seg.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
          const lastX  = seg[seg.length - 1].x.toFixed(1);
          const firstX = seg[0].x.toFixed(1);
          return `${top} L${lastX},${baseY} L${firstX},${baseY} Z`;
        })
        .join(" ");
    };

    const rightAxisX = margin.left + innerW;

    return (
      <svg
        ref={svgRef as any}
        width={width}
        height={chartH}
        onMouseMove={onMouseMove as any}
        onMouseLeave={onMouseLeave}
        style={{ display: "block", overflow: "visible", fontFamily: "Arial, Helvetica, sans-serif" }}
      >
        {/* Grid (based on left Y ticks) */}
        {leftTicks.map((tick, i) => (
          <line key={`g${i}`}
            x1={margin.left} y1={yScaleLeft(tick).toFixed(1)}
            x2={(margin.left + innerW).toFixed(1)} y2={yScaleLeft(tick).toFixed(1)}
            stroke="#e8ecf0" strokeWidth="0.5" />
        ))}

        {/* Left Y-axis ticks — omit unit suffix when the rotated axis label already carries it */}
        {leftTicks.map((tick, i) => (
          <text key={`yt${i}`}
            x={margin.left - 5} y={yScaleLeft(tick).toFixed(1)}
            textAnchor="end" dominantBaseline="middle" fontSize={tickFontSize} className={styles.axisTick}>
            {fmtYRange(tick, leftTickStep)}{!leftAxisLabel && leftAxisUnit ? ` ${leftAxisUnit}` : ""}
          </text>
        ))}

        {/* Left Y-axis label (rotated −90°, mirroring the right-axis gap so spacing looks symmetric) */}
        {leftAxisLabel && (() => {
          const lx = margin.left - (RIGHT_AXIS_W - 6); // mirrors right-axis: label center same distance from ticks
          return (
            <text
              transform={`rotate(-90, ${lx.toFixed(1)}, ${(margin.top + innerH / 2).toFixed(1)})`}
              x={lx.toFixed(1)}
              y={(margin.top + innerH / 2).toFixed(1)}
              textAnchor="middle"
              fontSize={tickFontSize}
              className={styles.axisTick}>
              {leftAxisLabel}
            </text>
          );
        })()}

        {/* Right Y-axis ticks — omit unit suffix when the axis label already carries it */}
        {rightTicks.map((tick, i) => (
          <text key={`ryt${i}`}
            x={(rightAxisX + 5).toFixed(1)} y={yScaleRight(tick).toFixed(1)}
            textAnchor="start" dominantBaseline="middle" fontSize={tickFontSize} className={styles.axisTick}>
            {fmtYRange(tick, rightTickStep)}{!rightAxisLabel && rightAxisUnit ? ` ${rightAxisUnit}` : ""}
          </text>
        ))}

        {/* Right Y-axis label (rotated +90°, at far-right edge of RIGHT_AXIS_W, clear of tick text) */}
        {hasRightAxis && rightAxisLabel && (
          <text
            transform={`rotate(90, ${(rightAxisX + RIGHT_AXIS_W - 6).toFixed(1)}, ${(margin.top + innerH / 2).toFixed(1)})`}
            x={(rightAxisX + RIGHT_AXIS_W - 6).toFixed(1)}
            y={(margin.top + innerH / 2).toFixed(1)}
            textAnchor="middle"
            fontSize={tickFontSize}
            className={styles.axisTick}>
            {rightAxisLabel}
          </text>
        )}

        {xTickIdxs.map((idx) => {
          const s = samples[idx];
          const x = chartType === "bar_vertical"
            ? margin.left + idx * barGroupW + barGroupW / 2
            : xScale(s.timestamp);
          return (
            <text key={`xt${idx}`}
              x={x.toFixed(1)} y={(margin.top + innerH + tickFontSize + 3).toFixed(1)}
              textAnchor="middle" fontSize={tickFontSize} className={styles.axisTick}>
              {fmtElapsed(s.timestamp)}
            </text>
          );
        })}

        {/* X-axis label */}
        <text
          x={(margin.left + innerW / 2).toFixed(1)}
          y={(margin.top + innerH + tickFontSize * 2 + 8).toFixed(1)}
          textAnchor="middle" fontSize={tickFontSize} className={styles.axisTick}>
          Test Duration ({xTimeUnit})
        </text>

        {/* Axis lines */}
        <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + innerH} stroke="#ccc" strokeWidth="1" />
        <line x1={margin.left} y1={margin.top + innerH} x2={(margin.left + innerW).toFixed(1)} y2={margin.top + innerH} stroke="#ccc" strokeWidth="1" />
        {hasRightAxis && (
          <line x1={rightAxisX.toFixed(1)} y1={margin.top} x2={rightAxisX.toFixed(1)} y2={margin.top + innerH} stroke="#ccc" strokeWidth="1" />
        )}

        {/* Idle baseline overlay (rendered behind workload series).
            Only the dashed reference line at the idle average is drawn — the
            min/max range is summarized numerically in the KPI card below the
            chart, so we keep the plot itself clean. The line uses a distinct
            charcoal tone so it never collides with any metric series color
            (which span red/green/blue/teal/orange/purple). The overlay is
            duration-independent (purely y-axis), so any test/idle duration
            ratio works. */}
        {baseline && metricKeys.map((key) => {
          const b = baseline[key];
          if (!b || typeof b.avg !== "number") return null;
          const ys = yScaleFor(key);
          const xLeft  = margin.left;
          const xRight = margin.left + innerW;
          const avgY  = ys(b.avg as number);
          return (
            <g key={`baseline-${key}`} style={{ pointerEvents: "none" }}>
              <line
                x1={xLeft.toFixed(1)} y1={avgY.toFixed(1)}
                x2={xRight.toFixed(1)} y2={avgY.toFixed(1)}
                stroke="#1f2d3d" strokeWidth="1.25" strokeDasharray="5 3" strokeOpacity="0.85"
              />
            </g>
          );
        })}

        {/* Series */}
        {metricKeys.map((key, i) => {
          const color = resolveSeriesColor(key, i);
          // Path commands need at least 2 points to be visible. When the
          // collector recorded only one sample (short tests, or sampler
          // start latency on the first test in a session) we fall back
          // to drawing a marker dot at the sole sample so the chart
          // doesn't appear empty.
          const renderSinglePoint = samples.length === 1 && chartType !== "bar_vertical";
          if (renderSinglePoint) {
            const ys = yScaleFor(key);
            const cx = xScale(samples[0].timestamp).toFixed(1);
            const cy = ys(samples[0].values[key] ?? 0).toFixed(1);
            return (
              <g key={key}>
                <circle cx={cx} cy={cy} r="3.5" fill={color} fillOpacity="0.95" />
                <circle cx={cx} cy={cy} r="6"  fill={color} fillOpacity="0.20" />
              </g>
            );
          }
          if (chartType === "line") {
            return (
              <g key={key}>
                <path d={areaPath(key)} fill={color} fillOpacity="0.18" stroke="none" />
                <path d={linePath(key)} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
              </g>
            );
          }
          if (chartType === "area") {
            return (
              <g key={key}>
                <path d={areaPath(key)} fill={color} fillOpacity="0.28" stroke="none" />
                <path d={linePath(key)} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
              </g>
            );
          }
          if (chartType === "bar_vertical") {
            const numS       = metricKeys.length;
            const totalBarW  = barGroupW * 0.75;
            const singleBarW = totalBarW / numS;
            const groupOff   = -totalBarW / 2;
            const ys         = yScaleFor(key);
            const baseY      = ys(yMinFor(key));
            return samples.map((s, sIdx) => {
              const val = s.values[key];
              // Skip the bar entirely when the sample has no data
              // (null/undefined/NaN or the explicit -1 missing sentinel)
              // so a gap is visible instead of a zero-height bar.
              if (isMissing(val)) return null;
              const bY  = ys(val as number);
              const bH  = Math.max(0, baseY - bY);
              const bX  = margin.left + sIdx * barGroupW + barGroupW / 2 + groupOff + i * singleBarW;
              return (
                <rect key={`${key}-${sIdx}`}
                  x={bX.toFixed(1)} y={bY.toFixed(1)}
                  width={Math.max(1, singleBarW - 1).toFixed(1)} height={bH.toFixed(1)}
                  fill={color} fillOpacity="0.82" rx="1" />
              );
            });
          }
          return null;
        })}

        {/* Thresholds — color reflects whether any sample exceeds the warning level */}
        {thresholds && (() => {
          // Orange→red tones when exceeded; muted grey (theme-neutral) when safe
          const WARN_EXCEEDED = ["#f57c00", "#e53935", "#e65100", "#c62828", "#bf360c"];
          const WARN_SAFE     = "#9e9e9e"; // grey-500, readable in both light & dark themes
          const warnEntries = (Object.entries(thresholds as Record<string, { warning?: number }>)
            .filter(([k, t]) => t.warning !== undefined && metricKeys.includes(k)));
          return warnEntries.map(([key, thresh], wIdx) => {
            const warnVal   = thresh.warning!;
            const exceeded  = samples.some(s => {
              const v = s.values[key];
              return !isMissing(v) && (v as number) > warnVal;
            });
            const warnColor = exceeded ? WARN_EXCEEDED[wIdx % WARN_EXCEEDED.length] : WARN_SAFE;
            const unit      = configScales?.[key]?.unit ?? "";
            const warnText  = `warn: ${fmtY(warnVal)}${unit ? ` ${unit}` : ""}`;
            return (
              <g key={key}>
                <line
                  x1={margin.left} y1={yScaleFor(key)(warnVal).toFixed(1)}
                  x2={(margin.left + innerW).toFixed(1)} y2={yScaleFor(key)(warnVal).toFixed(1)}
                  stroke={warnColor} strokeWidth="1" strokeDasharray="5,3" />
                <text
                  x={(margin.left + 4).toFixed(1)}
                  y={(yScaleFor(key)(warnVal) - 3).toFixed(1)}
                  fontSize={tickFontSize - 1} fill={warnColor}>
                  {warnText}
                </text>
              </g>
            );
          });
        })()}

        {/* Hover cursor */}
        {tooltip && (
          <line
            x1={tooltip.svgX.toFixed(1)} y1={margin.top}
            x2={tooltip.svgX.toFixed(1)} y2={margin.top + innerH}
            stroke="#0071c5" strokeWidth="1" strokeDasharray="3,3" />
        )}

        {/* Mouse capture overlay */}
        <rect x={margin.left} y={margin.top} width={innerW} height={innerH} fill="transparent" />
      </svg>
    );
  };

  // ── Event handlers ────────────────────────────────────────────────────────
  const onCardMove = (e: MouseEvent) => {
    if (!cardSvgRef.current || !containerRef.current) return;
    setCardTt(buildTooltip(e, cardSvgRef.current, containerRef.current, cardW, cardMargin));
  };
  const onModalMove = (e: MouseEvent) => {
    if (!modalSvgRef.current || !modalChartRef.current) return;
    setModalTt(buildTooltip(e, modalSvgRef.current, modalChartRef.current, MODAL_W, modalMargin));
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className={styles.chartCard} ref={containerRef}>
      {/* Card header */}
      {!compact && (
        <div className={styles.chartCardHeader}>
          <span className={styles.chartCardTitle}>{displayName}</span>
          <div className={styles.chartCardActions}>
            <button
              className={styles.pngButton}
              type="button"
              onClick={() => cardSvgRef.current && exportFullPng(cardSvgRef.current, cardW, CARD_H, pngFilename, exportTitle, metricKeys, chartType, configScales, averages, minMax, baseline)}
              title={`Download ${exportTitle} chart as PNG`}
            >
              Download
            </button>
            <button className={styles.expandButton} type="button" onClick={() => setModalOpen(true)} title="Open larger view">
              Expand
            </button>
          </div>
        </div>
      )}

      {/* Compact-mode action strip — sits ABOVE the chart so the buttons
          never overlap the y-axis / tick labels / idleChip. Right-aligned
          so the user finds them in the conventional toolbar location. */}
      {compact && (
        <div className={styles.chartCardActionsCompact}>
          <button
            className={styles.pngButton}
            type="button"
            onClick={() => cardSvgRef.current && exportFullPng(cardSvgRef.current, cardW, CARD_H, pngFilename, exportTitle, metricKeys, chartType, configScales, averages, minMax, baseline)}
            title={`Download ${exportTitle} chart as PNG`}
          >
            ⬇
          </button>
          <button
            className={styles.expandButton}
            type="button"
            onClick={() => setModalOpen(true)}
            title={`Expand ${exportTitle} chart to full view`}
          >
            ⛶
          </button>
        </div>
      )}

      {/* Card chart */}
      <div style={{ position: "relative" }}>
        <ChartSVG
          width={cardW} chartH={CARD_H} margin={cardMargin}
          svgRef={cardSvgRef} tooltip={cardTt} tickFontSize={10}
          onMouseMove={onCardMove} onMouseLeave={() => setCardTt(null)}
        />
        {cardTt && <Tip tt={cardTt} containerWidth={cardW} />}
      </div>

      {/* Card legend */}
  {!compact && <Legend />}

      {/* Modal */}
      {modalOpen && (
        <div
          className={styles.modalOverlay}
          onClick={(e) => { if ((e.target as HTMLElement) === e.currentTarget) setModalOpen(false); }}
        >
          <div className={styles.modalContent}>
            <div className={styles.modalHeader}>
              <span className={styles.modalTitle}>{exportTitle}</span>
              <button
                className={styles.pngButton}
                type="button"
                onClick={() => modalSvgRef.current && exportFullPng(modalSvgRef.current, MODAL_W, MODAL_H, pngFilename, exportTitle, metricKeys, chartType, configScales, averages, minMax, baseline)}
                title={`Download ${exportTitle} chart as PNG`}
              >
                Download
              </button>
              <button className={styles.modalClose} type="button" onClick={() => setModalOpen(false)} title="Close (Esc)" aria-label="Close">
                ×
              </button>
            </div>
            <div style={{ position: "relative", overflow: "hidden" }} ref={modalChartRef}>
              {/* Idle baseline legend chip — mirrors the same chip rendered
                  by ``TelemetrySection`` over each per-card chart so that
                  the user can identify the dashed reference line in the
                  expanded full-view too. Multiple metrics with idle
                  baselines collapse into a single chip ("Idle") since the
                  dashed line styling is identical across series. */}
              {baseline && metricKeys.some((k) => typeof baseline[k]?.avg === "number") && (
                <span
                  className={styles.idleChip}
                  title="Pre-run idle baseline (dashed reference line on the chart)"
                >
                  <svg
                    className={styles.idleChipSwatch}
                    aria-hidden="true"
                    width="18"
                    height="8"
                    viewBox="0 0 18 8"
                  >
                    <line
                      x1="0"
                      y1="4"
                      x2="18"
                      y2="4"
                      stroke="currentColor"
                      stroke-width="1.5"
                      stroke-dasharray="3 2"
                    />
                  </svg>
                  Idle
                </span>
              )}
              <ChartSVG
                width={MODAL_W} chartH={MODAL_H} margin={modalMargin}
                svgRef={modalSvgRef} tooltip={modalTt} tickFontSize={12}
                onMouseMove={onModalMove} onMouseLeave={() => setModalTt(null)}
              />
              {modalTt && <Tip tt={modalTt} containerWidth={MODAL_W} isModal={true} />}
            </div>
            {/* Stats summary block — reuses the exact same classes as the
                per-metric ``kpiCard`` rendered by ``TelemetrySection`` so
                the expanded view has an identical look-and-feel: the
                ``avg`` pill + value, the Min/Max stats row, and the Idle
                + OVER IDLE delta row. Rendered once per metric key. */}
            {metricKeys.map((key) => {
              const unit       = configScales?.[key]?.unit;
              const unitSuffix = unit ? ` ${unit}` : "";
              const avg        = averages?.[key];
              const mm         = minMax?.[key];
              const idleAvg    = baseline?.[key]?.avg;
              if (avg === undefined && !mm && idleAvg === undefined) return null;
              const formatDelta = (loadValue: number | undefined): string | null => {
                if (typeof idleAvg !== "number" || typeof loadValue !== "number") return null;
                const diff = loadValue - idleAvg;
                if (Math.abs(idleAvg) >= 0.1) {
                  const pct  = (diff / idleAvg) * 100;
                  const sign = pct >= 0 ? "+" : "";
                  return `${sign}${pct.toFixed(0)}%`;
                }
                const sign = diff >= 0 ? "+" : "";
                return `${sign}${diff.toFixed(2)}${unitSuffix}`;
              };
              const avgDeltaText  = avg !== undefined  ? formatDelta(avg) : null;
              const peakDeltaText = mm?.max !== undefined ? formatDelta(mm.max) : null;
              const avgDeltaLabel  = avgDeltaText  ? `avg ${avgDeltaText}`  : null;
              const peakDeltaLabel = peakDeltaText ? `peak ${peakDeltaText}` : null;
              return (
                <div key={key} className={styles.modalKpiSummary}>
                  <div className={styles.kpiValueRow}>
                    <span className={styles.kpiAggregateTag} title="Average across all workload samples">avg</span>
                    <span className={styles.kpiValue} title="Average across all workload samples">{avg !== undefined ? avg.toFixed(2) : "--"}</span>
                    <span className={styles.kpiUnit}>{unit ?? ""}</span>
                  </div>
                  <div className={styles.kpiStatsRow}>
                    <span>Min {mm?.min !== undefined ? `${mm.min.toFixed(2)}${unitSuffix}` : "--"}</span>
                    <span>Max {mm?.max !== undefined ? `${mm.max.toFixed(2)}${unitSuffix}` : "--"}</span>
                  </div>
                  {(typeof idleAvg === "number" || avgDeltaLabel || peakDeltaLabel) && (
                    <div className={styles.kpiBaselineRow}>
                      {typeof idleAvg === "number" ? (
                        <span className={styles.kpiBaselineLabel}>Idle {idleAvg.toFixed(2)}{unitSuffix}</span>
                      ) : (
                        <span />
                      )}
                      {(avgDeltaLabel || peakDeltaLabel) && (
                        <span
                          className={styles.kpiBaselineDeltaBox}
                          title="Workload avg / peak relative to pre-run idle baseline"
                        >
                          <span className={styles.kpiBaselineDeltaTag}>Over Idle</span>
                          <span className={styles.kpiBaselineDelta}>
                            {[avgDeltaLabel, peakDeltaLabel].filter(Boolean).join(" \u00B7 ")}
                          </span>
                        </span>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
