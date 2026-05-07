import { FunctionComponent } from "preact";
import { useState, useRef, useEffect } from "preact/hooks";
import * as styles from "./TelemetryChartStyle.scss";

// ─── Series colour palette ───────────────────────────────────────────────────
// Colours are ordered so adjacent indices rotate across four hue families
// (blue → purple → teal → grey → …), maximising visual separation on line
// and area charts.  For any series beyond the palette, getSeriesColor() falls
// back to a golden-ratio HSL generator so the sequence extends indefinitely
// without repeating.  Avoid bright green (≈ 90-150 ° hue) — it is reserved
// for pass / fail status indicators in the rest of the UI.
const SERIES_COLORS: readonly string[] = [
  // Round 1 — vivid anchors, maximally spread
  "#0071c5", // Blue       · Intel Blue (primary, vivid)
  "#8e24aa", // Purple     · Purple 600 (vivid)
  "#00838f", // Teal       · Cyan 800 (mid teal)
  "#546e7a", // Grey       · Blue Grey 600 (mid)
  // Round 2 — extreme lightness shift in each family
  "#1a237e", // Blue       · Indigo 900 (near-black navy)
  "#ce93d8", // Purple     · Purple 200 (very light / pastel)
  "#26c6da", // Teal       · Cyan 400 (bright / vivid)
  "#263238", // Grey       · Blue Grey 900 (near-black)
  // Round 3 — bright / deep contrast
  "#00b0ff", // Blue       · Light Blue A400 (bright sky)
  "#4a148c", // Purple     · Purple 900 (very dark violet)
  "#006064", // Teal       · Cyan 900 (very dark)
  "#b0bec5", // Grey       · Blue Grey 200 (light silver)
  // Round 4 — mid-range fillers
  "#1565c0", // Blue       · Blue 800 (deep navy)
  "#7b1fa2", // Purple     · Purple 800
  "#00acc1", // Teal       · Cyan 600 (lighter)
  "#78909c", // Grey       · Blue Grey 400 (medium-light)
  // Round 5 — remaining distinct entries
  "#0288d1", // Blue       · Light Blue 700
  "#3949ab", // Purple     · Indigo 600 (blue-purple blend)
  "#00695c", // Teal       · Teal 800 (dark green-teal)
  "#37474f", // Grey       · Blue Grey 700 (medium-dark)
];

/**
 * Returns the series colour for the given 0-based index.
 * Uses the hand-crafted palette for the first 20 entries;
 * beyond that a golden-ratio HSL algorithm generates new colours
 * so the sequence extends indefinitely without repeating.
 */
function getSeriesColor(index: number): string {
  if (index < SERIES_COLORS.length) return SERIES_COLORS[index];
  // Golden-angle hue distribution — spreads colours evenly around the wheel.
  // Two lightness levels and three saturation bands give further variety.
  const goldenAngle = 137.508;
  const hue = Math.round((index * goldenAngle) % 360);
  // Skip the green band (90-150°) reserved for status colours.
  const adjustedHue = hue >= 90 && hue <= 150 ? (hue + 60) % 360 : hue;
  const sat = 55 + (index % 3) * 12;  // 55 / 67 / 79 %
  const lit = 38 + (index % 2) * 18;  // 38 / 56 %
  return `hsl(${adjustedHue},${sat}%,${lit}%)`;
}

// ─── Layout constants ─────────────────────────────────────────────────────────
const CARD_M  = { top: 10, right: 12, bottom: 40, left: 54 };
const CARD_H  = 230;
const MODAL_M = { top: 16, right: 18, bottom: 50, left: 64 };
const MODAL_H = 520;
const MODAL_W = 980;

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
  key: string; label: string; value: number; unit: string; color: string;
}
interface Tooltip {
  svgX: number; cX: number; cY: number; time: string; entries: TtEntry[];
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
  summaryMeta?: { cliName: string; platform: string; timestamp: string } | null;
  testId?: string;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function fmtY(v: number): string {
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)}k`;
  if (Math.abs(v) < 10) return v.toFixed(1);
  return v.toFixed(0);
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
) {
  const FONT         = "Arial, Helvetica, sans-serif";
  const TITLE_H      = 36;
  const hasStats     = metricKeys.some(k => averages?.[k] !== undefined || minMax?.[k] !== undefined);
  const ROW_H        = hasStats ? 42 : 26;
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
    const color      = getSeriesColor(i);
    const label      = configScales?.[key]?.label ?? key.replace(/_/g, " ");
    const unit       = configScales?.[key]?.unit;
    const avg        = averages?.[key];
    const mm         = minMax?.[key];
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

    // Min / Avg / Max stat badges (second row)
    if (hasStats && (avg !== undefined || mm)) {
      const badgeH   = 15;
      const badgeW   = 80;
      const badgeGap = 4;
      const badgesY  = baseY + row * ROW_H + 26;
      let   badgeX   = ix;

      const drawBadge = (
        bx: number,
        badgeLabel: string,
        badgeValue: string,
        bgFill: string,
        borderStroke: string,
        textColor: string,
      ) => {
        const g = document.createElementNS(ns, "g");

        const bg = document.createElementNS(ns, "rect");
        bg.setAttribute("x", String(bx));          bg.setAttribute("y", String(badgesY));
        bg.setAttribute("width", String(badgeW));  bg.setAttribute("height", String(badgeH));
        bg.setAttribute("rx", "2");
        bg.setAttribute("fill", bgFill);           bg.setAttribute("stroke", borderStroke);
        bg.setAttribute("stroke-width", "0.5");
        g.appendChild(bg);

        const lbl = document.createElementNS(ns, "text");
        lbl.setAttribute("x", String(bx + 4));    lbl.setAttribute("y", String(badgesY + badgeH / 2));
        lbl.setAttribute("dominant-baseline", "middle");
        lbl.setAttribute("font-size", "8");        lbl.setAttribute("font-weight", "700");
        lbl.setAttribute("fill", textColor);       lbl.setAttribute("opacity", "0.75");
        lbl.textContent = badgeLabel;
        g.appendChild(lbl);

        const val = document.createElementNS(ns, "text");
        val.setAttribute("x", String(bx + badgeW - 4));  val.setAttribute("y", String(badgesY + badgeH / 2));
        val.setAttribute("dominant-baseline", "middle");  val.setAttribute("text-anchor", "end");
        val.setAttribute("font-size", "10");               val.setAttribute("font-weight", "600");
        val.setAttribute("fill", textColor);
        val.textContent = badgeValue;
        g.appendChild(val);

        root.appendChild(g);
      };

      if (mm) {
        drawBadge(badgeX, "MIN", `${mm.min.toFixed(2)}${unitSuffix}`, "rgba(0,172,193,0.12)", "rgba(0,172,193,0.40)", "#00838f");
        badgeX += badgeW + badgeGap;
      }
      if (avg !== undefined) {
        drawBadge(badgeX, "AVG", `${avg.toFixed(2)}${unitSuffix}`, "rgba(0,113,197,0.10)", "rgba(0,113,197,0.35)", "#0071c5");
        badgeX += badgeW + badgeGap;
      }
      if (mm) {
        drawBadge(badgeX, "MAX", `${mm.max.toFixed(2)}${unitSuffix}`, "rgba(57,73,171,0.12)", "rgba(57,73,171,0.40)", "#3949ab");
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
  visibleMetrics, configTitle, configScales, axes,
  summaryMeta, testId,
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

  // Left Y domain (include warning thresholds so lines always stay in range)
  const leftVals = samples.flatMap(s => leftMetrics.map(k => s.values[k] ?? 0));
  const leftThreshVals = thresholds
    ? Object.entries(thresholds as Record<string, { warning?: number }>).flatMap(([k, t]) =>
        leftMetrics.includes(k) && t.warning !== undefined ? [t.warning] : [])
    : [];
  const leftRawMin = leftVals.length > 0 ? Math.min(...leftVals) : 0;
  const leftRawMax = leftVals.length > 0 ? Math.max(...leftVals, ...leftThreshVals) : 1;
  const leftPad    = (leftRawMax - leftRawMin) * 0.1 || 1;
  const leftYMin   = Math.max(0, leftRawMin - leftPad);
  const leftYMax   = leftRawMax + leftPad;

  // Right Y domain (only meaningful when right-axis metrics exist)
  const rightVals = hasRightAxis ? samples.flatMap(s => rightMetrics.map(k => s.values[k] ?? 0)) : [];
  const rightThreshVals = hasRightAxis && thresholds
    ? Object.entries(thresholds as Record<string, { warning?: number }>).flatMap(([k, t]) =>
        rightMetrics.includes(k) && t.warning !== undefined ? [t.warning] : [])
    : [];
  const rightRawMin = rightVals.length > 0 ? Math.min(...rightVals) : 0;
  const rightRawMax = rightVals.length > 0 ? Math.max(...rightVals, ...rightThreshVals) : 1;
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
  const displayName   = configTitle?.display && configTitle?.text ? configTitle.text : fallbackTitle;
  const sanitize      = (s: string) => s.toLowerCase().replace(/-/g, "_");
  const safeModule    = sanitize(moduleName);
  const pngFilename   = summaryMeta && testId
    ? `${sanitize(summaryMeta.cliName)}_telemetry_${sanitize(testId)}_${safeModule}_${sanitize(summaryMeta.platform)}_${sanitize(summaryMeta.timestamp)}.png`
    : `${safeModule}_telemetry.png`;

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
        value: s.values[k] ?? 0,
        unit:  configScales?.[k]?.unit ?? "",
        color: getSeriesColor(i),
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
          <span className={styles.tooltipVal} style={isModal ? { fontSize: "11px" } : undefined}>{v.value.toFixed(2)}{v.unit ? ` ${v.unit}` : ""}</span>
        </div>
      ))}
    </div>
  );

  // ── Legend JSX ───────────────────────────────────────────────────────────
  const Legend = () => (
    <div className={styles.legend}>
      {metricKeys.map((key, i) => {
        const color      = getSeriesColor(i);
        const label      = configScales?.[key]?.label ?? key.replace(/_/g, " ");
        const unit       = configScales?.[key]?.unit;
        const avg        = averages?.[key];
        const mm         = minMax?.[key];
        const unitSuffix = unit ? ` ${unit}` : "";
        return (
          <div key={key} className={styles.legendItem}>
            <div className={styles.legendItemHeader}>
              {chartType === "line" && <span className={styles.legendLine} style={{ background: color }} />}
              {chartType === "area" && <span className={styles.legendArea} style={{ background: color }} />}
              {chartType === "bar_vertical" && <span className={styles.legendBar} style={{ background: color }} />}
              <span className={styles.legendLabel}>{label}{unit ? ` (${unit})` : ""}</span>
            </div>
            {(avg !== undefined || mm) && (
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

    const maxXT = Math.min(7, samples.length);
    const xTickIdxs =
      samples.length <= maxXT
        ? samples.map((_, i) => i)
        : Array.from({ length: maxXT }, (_, i) =>
            Math.round((i / (maxXT - 1)) * (samples.length - 1)),
          );

    const linePath = (key: string) =>
      samples
        .map((s, i) => `${i === 0 ? "M" : "L"}${xScale(s.timestamp).toFixed(1)},${yScaleFor(key)(s.values[key] ?? 0).toFixed(1)}`)
        .join(" ");

    const areaPath = (key: string) => {
      const ys     = yScaleFor(key);
      const lp     = linePath(key);
      const lastX  = xScale(samples[samples.length - 1].timestamp).toFixed(1);
      const firstX = xScale(samples[0].timestamp).toFixed(1);
      const baseY  = ys(yMinFor(key)).toFixed(1);
      return `${lp} L${lastX},${baseY} L${firstX},${baseY} Z`;
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
            textAnchor="end" dominantBaseline="middle" fontSize={tickFontSize} fill="#888">
            {fmtY(tick)}{!leftAxisLabel && leftAxisUnit ? ` ${leftAxisUnit}` : ""}
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
              fill="#888">
              {leftAxisLabel}
            </text>
          );
        })()}

        {/* Right Y-axis ticks — omit unit suffix when the axis label already carries it */}
        {rightTicks.map((tick, i) => (
          <text key={`ryt${i}`}
            x={(rightAxisX + 5).toFixed(1)} y={yScaleRight(tick).toFixed(1)}
            textAnchor="start" dominantBaseline="middle" fontSize={tickFontSize} fill="#888">
            {fmtY(tick)}{!rightAxisLabel && rightAxisUnit ? ` ${rightAxisUnit}` : ""}
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
            fill="#888">
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
              textAnchor="middle" fontSize={tickFontSize} fill="#888">
              {fmtElapsed(s.timestamp)}
            </text>
          );
        })}

        {/* X-axis label */}
        <text
          x={(margin.left + innerW / 2).toFixed(1)}
          y={(margin.top + innerH + tickFontSize * 2 + 8).toFixed(1)}
          textAnchor="middle" fontSize={tickFontSize} fill="#888">
          Elapsed ({xTimeUnit})
        </text>

        {/* Axis lines */}
        <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + innerH} stroke="#ccc" strokeWidth="1" />
        <line x1={margin.left} y1={margin.top + innerH} x2={(margin.left + innerW).toFixed(1)} y2={margin.top + innerH} stroke="#ccc" strokeWidth="1" />
        {hasRightAxis && (
          <line x1={rightAxisX.toFixed(1)} y1={margin.top} x2={rightAxisX.toFixed(1)} y2={margin.top + innerH} stroke="#ccc" strokeWidth="1" />
        )}

        {/* Series */}
        {metricKeys.map((key, i) => {
          const color = getSeriesColor(i);
          if (chartType === "line") {
            return <path key={key} d={linePath(key)} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />;
          }
          if (chartType === "area") {
            return (
              <g key={key}>
                <path d={areaPath(key)} fill={color} fillOpacity="0.12" stroke="none" />
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
              const val = s.values[key] ?? 0;
              const bY  = ys(val);
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
            const exceeded  = samples.some(s => (s.values[key] ?? 0) > warnVal);
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
      <div className={styles.chartCardHeader}>
        <span className={styles.chartCardTitle}>{displayName}</span>
        <button className={styles.expandButton} type="button" onClick={() => setModalOpen(true)} title="Open larger view">
          Expand
        </button>
      </div>

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
      <Legend />

      {/* Modal */}
      {modalOpen && (
        <div
          className={styles.modalOverlay}
          onClick={(e) => { if ((e.target as HTMLElement) === e.currentTarget) setModalOpen(false); }}
        >
          <div className={styles.modalContent}>
            <div className={styles.modalHeader}>
              <span className={styles.modalTitle}>{displayName}</span>
              <button
                className={styles.pngButton}
                type="button"
                onClick={() => modalSvgRef.current && exportFullPng(modalSvgRef.current, MODAL_W, MODAL_H, pngFilename, displayName, metricKeys, chartType, configScales, averages, minMax)}
                title={`Download ${displayName} chart as PNG`}
              >
                Download
              </button>
              <button className={styles.modalClose} type="button" onClick={() => setModalOpen(false)} title="Close (Esc)" aria-label="Close">
                ×
              </button>
            </div>
            <div style={{ position: "relative", overflow: "hidden" }} ref={modalChartRef}>
              <ChartSVG
                width={MODAL_W} chartH={MODAL_H} margin={modalMargin}
                svgRef={modalSvgRef} tooltip={modalTt} tickFontSize={12}
                onMouseMove={onModalMove} onMouseLeave={() => setModalTt(null)}
              />
              {modalTt && <Tip tt={modalTt} containerWidth={MODAL_W} isModal={true} />}
            </div>
            <Legend />
          </div>
        </div>
      )}
    </div>
  );
};
