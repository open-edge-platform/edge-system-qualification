// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * ChartsSection — inline SVG chart renderer for the Allure report overlay.
 *
 * Reads ``extended_metadata.charts[]`` from the Core Metrics Test Results
 * attachment and renders each chart as a responsive inline SVG with:
 *   - Log or linear Y-axis
 *   - Multiple named series / legend
 *   - Interactive crosshair tooltip
 *   - Collapsible section header (mirrors TelemetrySection UX)
 *
 * No attached image files are needed — charts are stored as data in
 * ``extended_metadata`` and rendered entirely client-side, keeping the
 * report archive small.
 *
 * Supported chart types (``chart.type``):
 *   "line"  — connected polyline per series (default for histograms)
 */

import { FunctionComponent } from "preact";
import { useState, useRef, useEffect } from "preact/hooks";
import * as styles from "./ChartsSectionStyle.scss";

// ─── Types ────────────────────────────────────────────────────────────────────

interface ChartPoint {
  x: number;
  y: number;
}

interface ChartSeriesData {
  label: string;
  data: ChartPoint[];
  color?: string;
}

interface ChartData {
  id: string;
  title: string;
  /** Currently only "line" and "step" are rendered; other types reserved for future use. */
  type: "line" | "step" | "bar" | "scatter";
  x_label: string;
  y_label: string;
  x_unit: string;
  y_unit: string;
  log_y: boolean;
  log_x: boolean;
  series: ChartSeriesData[];
  metadata?: Record<string, string>;
  /** Minimum axis range. When set, the axis spans at least from x_min to x_max
   *  (or y_min to y_max). Data outside these bounds still expands the axis.
   *  Use to keep charts comparable across runs (e.g. always 0–700 µs for
   *  cyclictest latency histograms). */
  x_min?: number;
  x_max?: number;
  y_min?: number;
  y_max?: number;
}

interface TooltipEntry {
  label: string;
  color: string;
  nearestX: number;
  nearestY: number;
}

interface TooltipState {
  /** Left edge of the crosshair (SVG-space pixel). Used to draw the vertical guide. */
  lineX: number;
  /** Mouse position in the SVG container for left/right flip logic. */
  mouseX: number;
  mouseY: number;
  entries: TooltipEntry[];
}

// ─── Constants ────────────────────────────────────────────────────────────────

const MARGIN = { top: 20, right: 30, bottom: 50, left: 68 };
const CHART_H = 260;

/** High-contrast categorical palette — max hue separation for up to 10 series. */
const SERIES_COLORS = [
  "#1f77b4",  // blue
  "#d62728",  // red
  "#2ca02c",  // green
  "#ff7f0e",  // orange
  "#9467bd",  // purple
  "#17becf",  // cyan
  "#e377c2",  // pink
  "#8c564b",  // brown
  "#bcbd22",  // olive
  "#7f7f7f",  // grey
];

// ─── Scale / axis helpers ─────────────────────────────────────────────────────

function fmtCount(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(0)}K`;
  return String(Math.round(v));
}

function fmtX(v: number): string {
  return v % 1 === 0 ? String(Math.round(v)) : v.toFixed(1);
}

/**
 * Generate Y-axis tick values for a log₁₀ scale.
 * Returns one tick per integer power of 10 within [yMin, yMax].
 */
function logYTicks(yMin: number, yMax: number): number[] {
  const ticks: number[] = [];
  const lo = Math.floor(Math.log10(Math.max(yMin, 1)));
  const hi = Math.ceil(Math.log10(Math.max(yMax, 1)));
  for (let p = lo; p <= hi; p++) ticks.push(Math.pow(10, p));
  return ticks;
}

/** Generate ~5 evenly spaced linear Y-axis ticks. */
function linearYTicks(yMin: number, yMax: number, n = 5): number[] {
  const step = (yMax - yMin) / n;
  return Array.from({ length: n + 1 }, (_, i) => yMin + step * i);
}

/** Filename-safe sanitiser: lower-case, collapse non-alphanumeric to underscores. */
function sanitizeForFilename(s: string): string {
  return String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

/** Format compact timestamp ``YYMMDD_HHMM`` → ``20YY-MM-DD  HH:MM``. */
function formatPngTimestamp(ts: string): string {
  const m = ts.match(/^(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})$/);
  return m ? `20${m[1]}-${m[2]}-${m[3]}  ${m[4]}:${m[5]}` : ts;
}

/**
 * Build an ordered array of human-readable info lines from raw ``systemInfo``
 * (the object returned by ``getSystemInfoFromAttachment``).  Used to populate
 * the hardware/software footer strip in the exported PNG.
 */
function buildSystemInfoLines(systemInfo: any): string[] {
  const lines: string[] = [];
  if (!systemInfo) return lines;
  const hw = systemInfo?.hardware;
  const sw = systemInfo?.software;

  // Product / board (no label)
  if (hw?.dmi?.system) {
    const parts: string[] = [];
    if (hw.dmi.system.vendor)       parts.push(String(hw.dmi.system.vendor));
    if (hw.dmi.system.product_name) parts.push(String(hw.dmi.system.product_name));
    if (parts.length > 0) {
      const prod = parts.join(" ");
      lines.push(hw.dmi.motherboard?.name ? `${prod}  \u2022  ${hw.dmi.motherboard.name}` : prod);
    }
  }

  // CPU brand + installed memory on one line, no labels
  if (hw?.cpu?.brand) {
    const mem = hw?.memory;
    const gib = mem?.dimms?.installed_ram_gib ?? mem?.installed_ram_gib
              ?? mem?.usable_ram_gib          ?? mem?.total_gib;
    let line = String(hw.cpu.brand);
    if (gib !== undefined) line += `  \u2022  ${Number(gib).toFixed(0)} GB of RAM`;
    lines.push(line);
  }

  // OS — use pretty_name for full version, fallback to constructed string
  if (sw?.os) {
    const os   = sw.os;
    const dist = os.distribution;
    let line   = "";
    if (dist?.pretty_name)              line = String(dist.pretty_name);
    else if (dist?.name && dist?.version_id) line = `${dist.name} ${dist.version_id}`;
    else if (os.name)                   line = String(os.name);
    if (os.release) line += `  \u2022  kernel ${os.release}`;
    if (line) lines.push(line);
  }

  return lines;
}

/**
 * Export a rendered SVG element as a retina-quality PNG download.
 * Inlines computed styles so the standalone SVG renders correctly outside
 * the browser's CSS cascade (module class names are hashed at build time).
 */
function exportChartAsPng(
  svgEl: SVGSVGElement,
  chart: ChartData,
  width: number,
  height: number,
  filename: string,
  summaryMeta?: { cliName: string; platform: string; timestamp: string; version?: string } | null,
  systemInfo?: any,
): void {
  const ns   = "http://www.w3.org/2000/svg";
  const FONT = "Arial, Helvetica, sans-serif";

  // ── Dimensions ──────────────────────────────────────────────────────────────
  const TITLE_H    = 36;                       // title row only (no description)
  const EXTRA_BTM  = 8;                        // breathing room below chart area
  const LEG_PAD    = 12;
  const LEG_ITEM_H = 22;
  // System-info header — dynamically sized to fit all hardware/software lines
  const infoLines    = buildSystemInfoLines(systemInfo);
  const SYS_LINE_H   = 14;                     // px per info line
  const SYS_PAD      = 8;                      // top+bottom padding inside header
  const hasHeader    = systemInfo != null || summaryMeta != null;
  const rightLnCount = summaryMeta ? 2 : 0;    // version + timestamp
  const maxSysLines  = Math.max(infoLines.length, rightLnCount);
  const HDR_H        = hasHeader ? Math.max(maxSysLines * SYS_LINE_H + SYS_PAD * 2, 28) : 0;
  const PER_ROW    = Math.min(chart.series.length || 1, 3);
  const legRows    = chart.series.length > 0 ? Math.ceil(chart.series.length / PER_ROW) : 0;
  const LEG_H      = chart.series.length > 0 ? LEG_PAD + legRows * LEG_ITEM_H + LEG_PAD : 0;
  // Run info metadata grid — all metadata entries except "description"
  const metaEntries  = Object.entries(chart.metadata ?? {}).filter(([k]) => k !== "description");
  const META_COLS    = 4;
  const META_ITEM_H  = 18;
  const META_PAD     = 8;
  const metaRows     = metaEntries.length > 0 ? Math.ceil(metaEntries.length / META_COLS) : 0;
  const META_H       = metaRows > 0 ? META_PAD + metaRows * META_ITEM_H + META_PAD : 0;
  const totalH     = HDR_H + TITLE_H + height + EXTRA_BTM + LEG_H + META_H;

  // ── Build composite SVG ─────────────────────────────────────────────────────
  const root = document.createElementNS(ns, "svg");
  root.setAttribute("xmlns", ns);
  root.setAttribute("width",  String(width));
  root.setAttribute("height", String(totalH));
  root.setAttribute("font-family", FONT);

  // White background
  const bg = document.createElementNS(ns, "rect");
  bg.setAttribute("width",  String(width));
  bg.setAttribute("height", String(totalH));
  bg.setAttribute("fill",   "#fff");
  root.appendChild(bg);

  // ── System info header (hardware + software + version/timestamp) ────────────────
  if (hasHeader) {
    // Light gray background strip
    const hdrBg = document.createElementNS(ns, "rect");
    hdrBg.setAttribute("x",      "0");
    hdrBg.setAttribute("y",      "0");
    hdrBg.setAttribute("width",  String(width));
    hdrBg.setAttribute("height", String(HDR_H));
    hdrBg.setAttribute("fill",   "#f5f5f5");
    root.appendChild(hdrBg);

    // Bottom separator line
    const hdrSep = document.createElementNS(ns, "line");
    hdrSep.setAttribute("x1",           "0");
    hdrSep.setAttribute("y1",           String(HDR_H));
    hdrSep.setAttribute("x2",           String(width));
    hdrSep.setAttribute("y2",           String(HDR_H));
    hdrSep.setAttribute("stroke",       "#e0e0e0");
    hdrSep.setAttribute("stroke-width", "1");
    root.appendChild(hdrSep);

    // ── Left column: hardware + software info lines ───────────────────────
    infoLines.forEach((line, i) => {
      const textEl = document.createElementNS(ns, "text");
      textEl.setAttribute("x",                 "12");
      textEl.setAttribute("y",                 String(SYS_PAD + SYS_LINE_H / 2 + i * SYS_LINE_H));
      textEl.setAttribute("dominant-baseline", "middle");
      textEl.setAttribute("font-size",         "9");
      textEl.setAttribute("fill",              "#555");
      textEl.textContent = line;
      root.appendChild(textEl);
    });

    // ── Right column: version (top-right) then timestamp below it ─────────
    if (summaryMeta) {
      const rightX     = String(width - 12);
      const hasVersion = !!(summaryMeta.version);
      const topLineY   = SYS_PAD + SYS_LINE_H / 2;

      // Version — first line (top-aligned)
      if (hasVersion) {
        const verEl = document.createElementNS(ns, "text");
        verEl.setAttribute("x",                 rightX);
        verEl.setAttribute("y",                 String(topLineY));
        verEl.setAttribute("text-anchor",       "end");
        verEl.setAttribute("dominant-baseline", "middle");
        verEl.setAttribute("font-size",         "9");
        verEl.setAttribute("fill",              "#666");
        verEl.textContent = summaryMeta.version!;
        root.appendChild(verEl);
      }

      // Timestamp — second line (below version)
      const tsEl = document.createElementNS(ns, "text");
      tsEl.setAttribute("x",                 rightX);
      tsEl.setAttribute("y",                 String(topLineY + (hasVersion ? SYS_LINE_H : 0)));
      tsEl.setAttribute("text-anchor",       "end");
      tsEl.setAttribute("dominant-baseline", "middle");
      tsEl.setAttribute("font-size",         "9");
      tsEl.setAttribute("fill",              "#666");
      tsEl.textContent = formatPngTimestamp(summaryMeta.timestamp);
      root.appendChild(tsEl);
    }
  }

  // Title
  const titleEl = document.createElementNS(ns, "text");
  titleEl.setAttribute("x", String(width / 2));
  titleEl.setAttribute("y", String(HDR_H + 22));
  titleEl.setAttribute("text-anchor",  "middle");
  titleEl.setAttribute("font-size",    "14");
  titleEl.setAttribute("font-weight",  "600");
  titleEl.setAttribute("fill",         "#1a1a1a");
  titleEl.textContent = chart.title;
  root.appendChild(titleEl);

  // Chart SVG content shifted down by HDR_H + TITLE_H
  const chartG = document.createElementNS(ns, "g");
  chartG.setAttribute("transform", `translate(0,${HDR_H + TITLE_H})`);
  const clone = svgEl.cloneNode(true) as SVGSVGElement;
  const PROPS = ["stroke", "stroke-width", "stroke-dasharray", "fill", "font-size", "font-weight", "font-family"] as const;
  const origEls  = Array.from(svgEl.querySelectorAll("*"));
  const cloneEls = Array.from(clone.querySelectorAll("*"));
  origEls.forEach((orig, idx) => {
    const cs  = window.getComputedStyle(orig as Element);
    const cel = cloneEls[idx] as SVGElement;
    PROPS.forEach((p) => {
      const v = cs.getPropertyValue(p);
      if (v) cel.style.setProperty(p, v);
    });
    // ── Force light-mode colors regardless of the active UI theme ───────────
    // getComputedStyle resolves CSS variables from the live DOM, so in dark
    // mode it returns dark-theme overrides (e.g. axisTick fill #d0d6dc,
    // svgAxis stroke #5a6370). Since the exported PNG always has a white
    // background we override with the hardcoded light-mode values so the
    // chart text/lines are always readable dark-on-white.
    const cl = (orig as Element).classList;
    if (cl.contains(styles.axisTick)) {
      cel.style.setProperty("fill",        "#555");
      cel.style.setProperty("font-weight", "500");
    }
    if (cl.contains(styles.svgAxis)) {
      cel.style.setProperty("stroke", "#bbb");
    }
    if (cl.contains(styles.svgGrid)) {
      cel.style.setProperty("stroke", "#e0e0e0");
    }
    cel.removeAttribute("class");
  });
  while (clone.firstChild) chartG.appendChild(clone.firstChild);
  root.appendChild(chartG);

  // ── Legend (below chart) ─────────────────────────────────────────────────────
  if (chart.series.length > 0) {
    const legStartY = HDR_H + TITLE_H + height + EXTRA_BTM;

    // Separator line
    const sep = document.createElementNS(ns, "line");
    sep.setAttribute("x1", "16");
    sep.setAttribute("y1", String(legStartY + 2));
    sep.setAttribute("x2", String(width - 16));
    sep.setAttribute("y2", String(legStartY + 2));
    sep.setAttribute("stroke",       "#e0e0e0");
    sep.setAttribute("stroke-width", "1");
    root.appendChild(sep);

    const ITEM_W = (width - 32) / PER_ROW;
    const baseY  = legStartY + LEG_PAD;

    chart.series.forEach((s, i) => {
      const color  = s.color ?? SERIES_COLORS[i % SERIES_COLORS.length];
      const row    = Math.floor(i / PER_ROW);
      const col    = i % PER_ROW;
      const ix     = 16 + col * ITEM_W;
      const itemY  = baseY + row * LEG_ITEM_H + LEG_ITEM_H / 2;

      // Color swatch
      const sym = document.createElementNS(ns, "rect");
      sym.setAttribute("x",      String(ix));
      sym.setAttribute("y",      String(itemY - 1));
      sym.setAttribute("width",  "16");
      sym.setAttribute("height", "2");
      sym.setAttribute("rx",     "1");
      sym.setAttribute("fill",   color);
      root.appendChild(sym);

      // Label
      const txt = document.createElementNS(ns, "text");
      txt.setAttribute("x",                  String(ix + 22));
      txt.setAttribute("y",                  String(itemY));
      txt.setAttribute("dominant-baseline",  "middle");
      txt.setAttribute("font-size",          "11");
      txt.setAttribute("fill",               "#333");
      txt.textContent = s.label;
      root.appendChild(txt);
    });
  }

  // ── Run info metadata grid ───────────────────────────────────────────────────
  if (META_H > 0) {
    const metaStartY = HDR_H + TITLE_H + height + EXTRA_BTM + LEG_H;

    // Light background strip
    const metaBg = document.createElementNS(ns, "rect");
    metaBg.setAttribute("x",      "0");
    metaBg.setAttribute("y",      String(metaStartY));
    metaBg.setAttribute("width",  String(width));
    metaBg.setAttribute("height", String(META_H));
    metaBg.setAttribute("fill",   "#fafafa");
    root.appendChild(metaBg);

    // Top separator
    const metaSep = document.createElementNS(ns, "line");
    metaSep.setAttribute("x1",           "0");
    metaSep.setAttribute("y1",           String(metaStartY));
    metaSep.setAttribute("x2",           String(width));
    metaSep.setAttribute("y2",           String(metaStartY));
    metaSep.setAttribute("stroke",       "#e0e0e0");
    metaSep.setAttribute("stroke-width", "1");
    root.appendChild(metaSep);

    const META_ITEM_W = (width - 32) / META_COLS;
    metaEntries.forEach(([key, value], i) => {
      const row = Math.floor(i / META_COLS);
      const col = i % META_COLS;
      const ix  = 16 + col * META_ITEM_W;
      const iy  = metaStartY + META_PAD + row * META_ITEM_H + META_ITEM_H / 2;

      // Key (left-aligned, dimmed)
      const keyEl = document.createElementNS(ns, "text");
      keyEl.setAttribute("x",                 String(ix));
      keyEl.setAttribute("y",                 String(iy));
      keyEl.setAttribute("dominant-baseline", "middle");
      keyEl.setAttribute("font-size",         "9");
      keyEl.setAttribute("fill",              "#888");
      keyEl.textContent = key;
      root.appendChild(keyEl);

      // Value (right-aligned within cell, bold)
      const valEl = document.createElementNS(ns, "text");
      valEl.setAttribute("x",                 String(ix + META_ITEM_W - 8));
      valEl.setAttribute("y",                 String(iy));
      valEl.setAttribute("text-anchor",       "end");
      valEl.setAttribute("dominant-baseline", "middle");
      valEl.setAttribute("font-size",         "9");
      valEl.setAttribute("font-weight",       "600");
      valEl.setAttribute("fill",              "#333");
      valEl.textContent = String(value);
      root.appendChild(valEl);
    });
  }

  // ── Serialize → blob URL → canvas (2×) → download ───────────────────────────
  const svgStr = new XMLSerializer().serializeToString(root);
  const url = URL.createObjectURL(
    new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" }),
  );
  const img = new Image();
  img.onload = () => {
    const SCALE = 2;
    const cv = document.createElement("canvas");
    cv.width  = width  * SCALE;
    cv.height = totalH * SCALE;
    const ctx = cv.getContext("2d");
    if (!ctx) { URL.revokeObjectURL(url); return; }
    ctx.scale(SCALE, SCALE);
    ctx.drawImage(img, 0, 0);
    const a = document.createElement("a");
    a.download = filename;
    a.href = cv.toDataURL("image/png");
    a.click();
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

// ─── SVG line chart ───────────────────────────────────────────────────────────

const LineChartSvg: FunctionComponent<{
  chart: ChartData;
  width: number;
  onTooltip: (tt: TooltipState | null) => void;
  tooltip?: TooltipState | null;
  chartHeight?: number;
  svgRef?: { current: SVGSVGElement | null };
}> = ({ chart, width, onTooltip, tooltip, chartHeight = CHART_H, svgRef }) => {
  const innerW = Math.max(width - MARGIN.left - MARGIN.right, 10);
  const innerH = chartHeight - MARGIN.top - MARGIN.bottom;

  // ── Collect data extents ──────────────────────────────────────────────────
  const allPoints: ChartPoint[] = chart.series.flatMap((s) => s.data);
  if (allPoints.length === 0) return null;

  const xVals = allPoints.map((p) => p.x);
  const yPositive = allPoints.map((p) => p.y).filter((y) => y > 0);
  if (yPositive.length === 0) return null;

  const xMinData = Math.min(...xVals);
  const xMaxData = Math.max(...xVals);
  const yMinData = Math.min(...yPositive);
  const yMaxData = Math.max(...yPositive);

  // Apply minimum axis range overrides so charts from different runs share
  // the same axis bounds and can be visually compared.
  // x_min / x_max guarantee the axis spans at least that range (data outside
  // still expands it); y_min / y_max work the same way.
  const xMin = chart.x_min != null ? Math.min(chart.x_min, xMinData) : xMinData;
  const xMax = chart.x_max != null ? Math.max(chart.x_max, xMaxData) : xMaxData;
  const yMin = chart.y_min != null ? Math.min(chart.y_min, yMinData) : yMinData;
  const yMax = chart.y_max != null ? Math.max(chart.y_max, yMaxData) : yMaxData;

  // For step charts: detect nominal bin width as the minimum x-difference
  // between consecutive unique x values. Used for:
  //   1. Zero-gap bridging — bins with no data are shown as explicit drops to
  //      the x-axis rather than flat horizontal lines at the previous count.
  //   2. Right-edge extension — xMax is padded by one binWidth so the last
  //      bin's right edge is visible and the staircase closes cleanly.
  const binWidth: number = (() => {
    if (chart.type !== "step") return 1;
    const sorted = [...new Set(xVals)].sort((a, b) => a - b);
    let w = 1;
    for (let i = 1; i < sorted.length; i++) {
      const d = sorted[i] - sorted[i - 1];
      if (i === 1 || d < w) w = d;
    }
    return w > 0 ? w : 1;
  })();
  // Extend the visible x range by one bin for step charts.
  const effectiveXMax = chart.type === "step" ? xMax + binWidth : xMax;

  // ── Scale functions ───────────────────────────────────────────────────────
  const scaleX = (x: number): number =>
    ((x - xMin) / (effectiveXMax - xMin || 1)) * innerW + MARGIN.left;

  // Log₁₀ Y scale: maps log(y) linearly to pixel height.
  // Points with y ≤ 0 are clipped just below the bottom axis (not drawn).
  //
  // For log-Y charts we pad the bottom of the scale by half a decade so that
  // bins with y = yMin (typically 1 for a histogram) sit visibly above the
  // x-axis.  Without padding, log₁₀(1) = 0 = logLo, which maps y=1 to the
  // exact same pixel as the axis baseline — making single-count bins appear
  // absent.  The half-decade shift gives y=1 roughly 0.5 / (logRange + 0.5)
  // of the chart height, which is clearly perceivable on any realistic data
  // range (e.g. ~9 % for a 5-decade span, ~33 % for a 1-decade span).
  const LOG_Y_BOTTOM_PAD = chart.log_y ? 0.5 : 0;
  const logLo = Math.log10(Math.max(yMin, 1)) - LOG_Y_BOTTOM_PAD;
  const logHi = Math.log10(Math.max(yMax, 1));
  const logRange = logHi - logLo || 1;

  const scaleY = chart.log_y
    ? (y: number): number => {
        if (y <= 0) return MARGIN.top + innerH + 4; // clip below axis — filtered later
        return MARGIN.top + innerH - ((Math.log10(y) - logLo) / logRange) * innerH;
      }
    : (y: number): number =>
        MARGIN.top + innerH - ((y - yMin) / (yMax - yMin || 1)) * innerH;

  // ── Axis ticks ────────────────────────────────────────────────────────────
  const yTicks = chart.log_y ? logYTicks(yMin, yMax) : linearYTicks(yMin, yMax);

  // X ticks: at most 10, evenly spaced across the effective data range.
  const xTickCount = Math.min(10, Math.max(xVals.length, 2));
  const xTicks: number[] = [];
  for (let i = 0; i <= xTickCount; i++) {
    xTicks.push(xMin + (effectiveXMax - xMin) * (i / xTickCount));
  }

  // ── Tooltip / crosshair hit detection ────────────────────────────────────
  const handleMouseMove = (evt: MouseEvent): void => {
    const svgEl = evt.currentTarget as SVGSVGElement;
    const rect = svgEl.getBoundingClientRect();
    const mouseX = evt.clientX - rect.left; // pixel inside SVG

    // Ignore mouse outside the plotting area
    if (mouseX < MARGIN.left || mouseX > MARGIN.left + innerW) {
      onTooltip(null);
      return;
    }

    // Convert pixel → data x
    const dataX = xMin + ((mouseX - MARGIN.left) / innerW) * (effectiveXMax - xMin);

    // Build tooltip entries — one per series that has data at the cursor position.
    //
    // Step/histogram charts: the bin at p.x covers [p.x, p.x + binWidth).
    // Only add an entry when the cursor falls inside a non-zero bin for that
    // series.  Using a strict bin-range check prevents "ghost" dots from
    // appearing for series that have no data at the hovered position (which
    // the old "nearest non-zero" approach produced for sparse histograms).
    //
    // Line charts: keep the existing nearest-point snapping behaviour.
    const entries: TooltipEntry[] = [];
    chart.series.forEach((s, i) => {
      if (!s.data.length) return;
      const color = s.color ?? SERIES_COLORS[i % SERIES_COLORS.length];

      let match: ChartPoint | null = null;

      if (chart.type === "step") {
        // Exact bin-range lookup: cursor must be inside [p.x, p.x + binWidth).
        const hit = s.data.find((p) => p.y > 0 && dataX >= p.x && dataX < p.x + binWidth);
        match = hit ?? null;
      } else {
        // Nearest-point snap for line charts.
        let closest = s.data[0];
        let minDist = Math.abs(s.data[0].x - dataX);
        for (const p of s.data) {
          const d = Math.abs(p.x - dataX);
          if (d < minDist) {
            minDist = d;
            closest = p;
          }
        }
        if (closest.y > 0) match = closest;
      }

      if (match) {
        entries.push({ label: s.label, color, nearestX: match.x, nearestY: match.y });
      }
    });

    if (!entries.length) {
      onTooltip(null);
      return;
    }

    onTooltip({
      // Crosshair follows the mouse rather than snapping to a data-point x so
      // it tracks smoothly even for step charts with sparse bins.
      lineX: mouseX,
      mouseX,
      mouseY: evt.clientY - rect.top,
      entries,
    });
  };

  // ── Colour helper ─────────────────────────────────────────────────────────
  const seriesColor = (s: ChartSeriesData, i: number): string =>
    s.color ?? SERIES_COLORS[i % SERIES_COLORS.length];


  return (
    <svg
      ref={svgRef}
      width={width}
      height={chartHeight}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => onTooltip(null)}
    >
      {/* ── Y grid lines ─────────────────────────────────────────────────── */}
      {yTicks.map((y) => {
        const py = scaleY(y);
        if (py < MARGIN.top - 2 || py > MARGIN.top + innerH + 2) return null;
        return (
          <line
            key={`yg-${y}`}
            x1={MARGIN.left}
            y1={py}
            x2={MARGIN.left + innerW}
            y2={py}
            className={styles.svgGrid}
            strokeWidth="1"
            strokeDasharray="3,3"
          />
        );
      })}

      {/* ── Data polylines (one per series) ─────────────────────────────── */}
      {chart.series.map((s, i) => {
        const color = seriesColor(s, i);
        const filtered = s.data.filter((p) => p.y > 0);
        if (!filtered.length) return null;

        // Each entry in polylineSegments is one <polyline> points string.
        // For log-Y step charts we emit multiple segments (one per contiguous
        // non-zero run) so that empty bins produce genuine blank space instead
        // of a zero-bridge drawn at chartBottom.  On a log₁₀ scale
        // chartBottom === scaleY(yMin) (both equal MARGIN.top + innerH), so
        // a bridge drawn at chartBottom is visually indistinguishable from a
        // bin with count = yMin — making empty regions look like real data.
        const polylineSegments: string[][] = [];
        const chartBottom = (MARGIN.top + innerH).toFixed(2);

        if (chart.type === "step") {
          if (chart.log_y) {
            // ── Log-Y: split into one segment per contiguous non-zero run ──
            // Gaps produce blank SVG space rather than a zero bridge.
            let seg: string[] = [];

            for (let j = 0; j < filtered.length; j++) {
              const p = filtered[j];

              if (j === 0) {
                // Rise from axis at the left edge of the first bin.
                seg.push(`${scaleX(p.x).toFixed(2)},${chartBottom}`);
              } else {
                const prev = filtered[j - 1];
                const gap = p.x - prev.x;

                if (gap > binWidth * 1.5) {
                  // Gap: close current segment (right edge of prev bin + drop to axis).
                  seg.push(`${scaleX(prev.x + binWidth).toFixed(2)},${scaleY(prev.y).toFixed(2)}`);
                  seg.push(`${scaleX(prev.x + binWidth).toFixed(2)},${chartBottom}`);
                  polylineSegments.push(seg);
                  seg = [];
                  // New segment: rise from axis at the next bin's left edge.
                  seg.push(`${scaleX(p.x).toFixed(2)},${chartBottom}`);
                } else {
                  // Consecutive bins: step-after horizontal from prev.y.
                  seg.push(`${scaleX(p.x).toFixed(2)},${scaleY(prev.y).toFixed(2)}`);
                }
              }

              seg.push(`${scaleX(p.x).toFixed(2)},${scaleY(p.y).toFixed(2)}`);
            }

            // Close the last segment (right edge + drop to axis).
            if (seg.length > 0) {
              const lastP = filtered[filtered.length - 1];
              seg.push(`${scaleX(lastP.x + binWidth).toFixed(2)},${scaleY(lastP.y).toFixed(2)}`);
              seg.push(`${scaleX(lastP.x + binWidth).toFixed(2)},${chartBottom}`);
              polylineSegments.push(seg);
            }
          } else {
            // ── Linear-Y: one polyline with baseline anchors & gap bridges ──
            const stepPts: string[] = [];

            // Anchor start: rise from x-axis at the left edge of the first bin.
            stepPts.push(`${scaleX(filtered[0].x).toFixed(2)},${chartBottom}`);

            for (let j = 0; j < filtered.length; j++) {
              const p = filtered[j];

              if (j > 0) {
                const prev = filtered[j - 1];
                const gap = p.x - prev.x;

                if (gap > binWidth * 1.5) {
                  // Gap: end previous bin at its right edge, drop to zero,
                  // hold at zero until the next non-zero bin starts.
                  stepPts.push(`${scaleX(prev.x + binWidth).toFixed(2)},${scaleY(prev.y).toFixed(2)}`);
                  stepPts.push(`${scaleX(prev.x + binWidth).toFixed(2)},${chartBottom}`);
                  stepPts.push(`${scaleX(p.x).toFixed(2)},${chartBottom}`);
                } else {
                  // Consecutive bins: normal step-after horizontal.
                  stepPts.push(`${scaleX(p.x).toFixed(2)},${scaleY(prev.y).toFixed(2)}`);
                }
              }

              stepPts.push(`${scaleX(p.x).toFixed(2)},${scaleY(p.y).toFixed(2)}`);
            }

            // Anchor end: extend last bin to its right edge, then drop to zero.
            const lastP = filtered[filtered.length - 1];
            stepPts.push(`${scaleX(lastP.x + binWidth).toFixed(2)},${scaleY(lastP.y).toFixed(2)}`);
            stepPts.push(`${scaleX(lastP.x + binWidth).toFixed(2)},${chartBottom}`);
            polylineSegments.push(stepPts);
          }
        } else {
          // Standard diagonal polyline (type === "line")
          polylineSegments.push(
            filtered.map((p) => `${scaleX(p.x).toFixed(2)},${scaleY(p.y).toFixed(2)}`)
          );
        }

        return (
          <g key={`series-${s.label}`}>
            {polylineSegments.map((segPts, segIdx) => (
              <polyline
                key={`${s.label}-${segIdx}`}
                points={segPts.join(" ")}
                fill="none"
                stroke={color}
                strokeWidth="1.8"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            ))}
          </g>
        );
      })}

      {/* ── X axis ───────────────────────────────────────────────────────── */}
      <line
        x1={MARGIN.left}
        y1={MARGIN.top + innerH}
        x2={MARGIN.left + innerW}
        y2={MARGIN.top + innerH}
        className={styles.svgAxis}
        strokeWidth="1"
      />
      {xTicks.map((x, ti) => {
        const px = scaleX(x);
        // Skip first/last ticks when they'd overlap the axis ends
        if (ti === 0 && px < MARGIN.left + 2) return null;
        if (ti === xTicks.length - 1 && px > MARGIN.left + innerW - 2) return null;
        return (
          <g key={`xt-${x}`}>
            <line
              x1={px}
              y1={MARGIN.top + innerH}
              x2={px}
              y2={MARGIN.top + innerH + 4}
              className={styles.svgAxis}
              strokeWidth="1"
            />
            <text
              x={px}
              y={MARGIN.top + innerH + 15}
              textAnchor="middle"
              fontSize="10"
              className={styles.axisTick}
            >
              {fmtX(Math.round(x * 10) / 10)}
            </text>
          </g>
        );
      })}
      {/* X axis label */}
      <text
        x={MARGIN.left + innerW / 2}
        y={chartHeight - 6}
        textAnchor="middle"
        fontSize="11"
        className={styles.axisTick}
      >
        {chart.x_label}
        {chart.x_unit ? ` (${chart.x_unit})` : ""}
      </text>

      {/* ── Y axis ───────────────────────────────────────────────────────── */}
      <line
        x1={MARGIN.left}
        y1={MARGIN.top}
        x2={MARGIN.left}
        y2={MARGIN.top + innerH}
        className={styles.svgAxis}
        strokeWidth="1"
      />
      {yTicks.map((y) => {
        const py = scaleY(y);
        if (py < MARGIN.top - 2 || py > MARGIN.top + innerH + 2) return null;
        return (
          <g key={`yt-${y}`}>
            <line
              x1={MARGIN.left - 4}
              y1={py}
              x2={MARGIN.left}
              y2={py}
              className={styles.svgAxis}
              strokeWidth="1"
            />
            <text
              x={MARGIN.left - 7}
              y={py + 4}
              textAnchor="end"
              fontSize="10"
              className={styles.axisTick}
            >
              {fmtCount(y)}
            </text>
          </g>
        );
      })}
      {/* Y axis label — rotated, centred along the axis */}
      <text
        transform={`rotate(-90)`}
        x={-(MARGIN.top + innerH / 2)}
        y={MARGIN.left - 50}
        textAnchor="middle"
        fontSize="11"
        className={styles.axisTick}
      >
        {chart.y_label}
        {chart.y_unit ? ` (${chart.y_unit})` : ""}
      </text>

      {/* ── Hover crosshair & intersection dots ──────────────────────── */}
      {tooltip && (
        <>
          <line
            x1={tooltip.lineX.toFixed(1)}
            y1={MARGIN.top}
            x2={tooltip.lineX.toFixed(1)}
            y2={MARGIN.top + innerH}
            stroke="#0071c5"
            strokeWidth="1"
            strokeDasharray="3,3"
          />
          {tooltip.entries.map((e) => {
            if (e.nearestY <= 0) return null;
            const cy = scaleY(e.nearestY);
            if (cy < MARGIN.top - 2 || cy > MARGIN.top + innerH + 2) return null;
            return (
              <circle
                key={e.label}
                cx={tooltip.lineX}
                cy={cy}
                r="4"
                fill={e.color}
                fillOpacity="0.9"
                stroke="white"
                strokeWidth="1.5"
              />
            );
          })}
        </>
      )}
    </svg>
  );
};

const MODAL_W = 1100;
const MODAL_CHART_H = 480;

// ─── Chart card (title + SVG + legend + tooltip) ─────────────────────────────

const ChartCard: FunctionComponent<{
  chart: ChartData;
  summaryMeta?: { cliName: string; platform: string; timestamp: string; version?: string } | null;
  testId?: string;
  systemInfo?: any;
}> = ({ chart, summaryMeta, testId, systemInfo }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cardSvgRef = useRef<SVGSVGElement>(null);
  const modalSvgRef = useRef<SVGSVGElement>(null);
  const [svgWidth, setSvgWidth] = useState(600);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalTooltip, setModalTooltip] = useState<TooltipState | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  // Close the kebab menu when the user clicks outside the dropdown.
  useEffect(() => {
    if (!menuOpen) return;
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [menuOpen]);

  // Responsive width — re-measure whenever the card is resized.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w && w > 0) setSvgWidth(w);
    });
    ro.observe(el);
    setSvgWidth(el.clientWidth || 600);
    return () => ro.disconnect();
  }, []);

  const description = chart.metadata?.description ?? "";

  // Filename mirrors TelemetryChart naming convention.
  const pngFilename = (() => {
    const parts: string[] = [];
    if (summaryMeta) parts.push(sanitizeForFilename(summaryMeta.cliName));
    parts.push("chart");
    if (testId) parts.push(sanitizeForFilename(testId));
    parts.push(sanitizeForFilename(chart.id));
    if (summaryMeta) parts.push(sanitizeForFilename(summaryMeta.timestamp));
    return parts.join("_") + ".png";
  })();

  const handleDownload = () => {
    if (!cardSvgRef.current) return;
    exportChartAsPng(cardSvgRef.current, chart, svgWidth, CHART_H, pngFilename, summaryMeta, systemInfo);
  };

  const handleModalDownload = () => {
    if (!modalSvgRef.current) return;
    exportChartAsPng(modalSvgRef.current, chart, MODAL_W, MODAL_CHART_H, pngFilename, summaryMeta, systemInfo);
  };

  return (
    <>
      <div className={styles.chartCard}>
        {/* Header: title + action buttons */}
        <div className={styles.chartCardHeader}>
          <span className={styles.chartCardTitle}>{chart.title}</span>
          <div className={styles.chartCardActions}>
            {/* Maximize — opens larger modal view */}
            <button
              className={styles.iconButton}
              type="button"
              onClick={() => setModalOpen(true)}
              title="Maximize"
            >
              ▢
            </button>
            {/* ⋮ kebab — click to reveal per-chart actions */}
            <div className={styles.menuWrap} ref={menuRef}>
              <button
                className={styles.iconButton}
                type="button"
                onClick={() => setMenuOpen((v) => !v)}
                title="Menu"
              >
                ≡
              </button>
              {menuOpen && (
                <div className={styles.menuDropdown}>
                  <button
                    className={styles.menuDropdownItem}
                    type="button"
                    onClick={() => { handleDownload(); setMenuOpen(false); }}
                  >
                    Download PNG
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Optional description caption */}
        {description && <p className={styles.chartDescription}>{description}</p>}

        {/* SVG chart + floating tooltip */}
        <div className={styles.chartSvgWrap} ref={containerRef}>
          <LineChartSvg
            chart={chart}
            width={svgWidth}
            onTooltip={setTooltip}
            tooltip={tooltip}
            svgRef={cardSvgRef}
          />

          {tooltip && (
            <div
              className={styles.tooltip}
              style={{
                left:
                  tooltip.mouseX > svgWidth / 2
                    ? `${tooltip.mouseX - 190}px`
                    : `${tooltip.mouseX + 14}px`,
                top: `${Math.max(tooltip.mouseY - 24, 4)}px`,
              }}
            >
              {tooltip.entries.map((e) => (
                <div key={e.label} className={styles.tooltipRow}>
                  <span className={styles.tooltipDot} style={{ background: e.color }} />
                  <span className={styles.tooltipKey}>{e.label}</span>
                  <span className={styles.tooltipVal}>
                    {`${fmtX(e.nearestX)}\u00a0\u00b5s\u00a0\u2192\u00a0${fmtCount(e.nearestY)}`}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Legend: one row per series */}
        <div className={styles.chartLegend}>
          {chart.series.map((s, i) => {
            const color = s.color ?? SERIES_COLORS[i % SERIES_COLORS.length];
            return (
              <div key={s.label} className={styles.legendItem}>
                <span className={styles.legendSwatch} style={{ background: color }} />
                <span className={styles.legendLabel}>{s.label}</span>
              </div>
            );
          })}
        </div>

        {/* Run info grid — all metadata entries except the description caption */}
        {(() => {
          const infoEntries = Object.entries(chart.metadata ?? {}).filter(([k]) => k !== "description");
          if (infoEntries.length === 0) return null;
          return (
            <div className={styles.chartMetaGrid}>
              {infoEntries.map(([key, value]) => (
                <div key={key} className={styles.chartMetaItem}>
                  <span className={styles.chartMetaKey}>{key}</span>
                  <span className={styles.chartMetaValue}>{String(value)}</span>
                </div>
              ))}
            </div>
          );
        })()}
      </div>

      {/* Modal — larger expanded view */}
      {modalOpen && (
        <div className={styles.modalOverlay} onClick={() => setModalOpen(false)}>
          <div
            className={styles.modalContent}
            onClick={(e: MouseEvent) => e.stopPropagation()}
          >
            <div className={styles.modalHeader}>
              <span className={styles.modalTitle}>{chart.title}</span>
              <div className={styles.chartCardActions}>
                <button
                  className={styles.iconButton}
                  type="button"
                  onClick={handleModalDownload}
                  title={`Download ${chart.title} as PNG`}
                  style={{ width: "auto", padding: "0 8px", fontSize: "11px" }}
                >
                  Download PNG
                </button>
                <button
                  className={`${styles.iconButton} ${styles.iconButtonDanger}`}
                  type="button"
                  onClick={() => setModalOpen(false)}
                  title="Close"
                >
                  &#x2715;
                </button>
              </div>
            </div>
            {description && (
              <p className={styles.chartDescription} style={{ padding: "0 4px" }}>
                {description}
              </p>
            )}
            <div style={{ position: "relative" }}>
              <LineChartSvg
                chart={chart}
                width={MODAL_W}
                chartHeight={MODAL_CHART_H}
                onTooltip={setModalTooltip}
                tooltip={modalTooltip}
                svgRef={modalSvgRef}
              />
              {modalTooltip && (
                <div
                  className={styles.tooltip}
                  style={{
                    left:
                      modalTooltip.mouseX > MODAL_W / 2
                        ? `${modalTooltip.mouseX - 210}px`
                        : `${modalTooltip.mouseX + 14}px`,
                    top: `${Math.max(modalTooltip.mouseY - 24, 4)}px`,
                  }}
                >
                  {modalTooltip.entries.map((e) => (
                    <div key={e.label} className={styles.tooltipRow}>
                      <span className={styles.tooltipDot} style={{ background: e.color }} />
                      <span className={styles.tooltipKey}>{e.label}</span>
                      <span className={styles.tooltipVal}>
                        {`${fmtX(e.nearestX)}\u00a0\u00b5s\u00a0\u2192\u00a0${fmtCount(e.nearestY)}`}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className={styles.chartLegend}>
              {chart.series.map((s, i) => {
                const color = s.color ?? SERIES_COLORS[i % SERIES_COLORS.length];
                return (
                  <div key={s.label} className={styles.legendItem}>
                    <span className={styles.legendSwatch} style={{ background: color }} />
                    <span className={styles.legendLabel}>{s.label}</span>
                  </div>
                );
              })}
            </div>

            {/* Run info grid — all metadata entries except the description caption */}
            {(() => {
              const infoEntries = Object.entries(chart.metadata ?? {}).filter(([k]) => k !== "description");
              if (infoEntries.length === 0) return null;
              return (
                <div className={styles.chartMetaGrid}>
                  {infoEntries.map(([key, value]) => (
                    <div key={key} className={styles.chartMetaItem}>
                      <span className={styles.chartMetaKey}>{key}</span>
                      <span className={styles.chartMetaValue}>{String(value)}</span>
                    </div>
                  ))}
                </div>
              );
            })()}
          </div>
        </div>
      )}
    </>
  );
};

// ─── ChartsSection (exported) ─────────────────────────────────────────────────

interface ChartsSectionProps {
  /**
   * The ``extended_metadata`` object from the Core Metrics Test Results JSON.
   * Expected shape: ``{ charts?: ChartData[], ... }``.
   */
  extendedMetadata?: any;
  /** Summary metadata forwarded from the parent Summary component; used to
   *  build meaningful PNG filenames (e.g. ``esq_chart_<testId>_<id>_<timestamp>.png``). */
  summaryMeta?: { cliName: string; platform: string; timestamp: string; version?: string } | null;
  /** Test ID used in PNG export filenames (mirrors TelemetrySection's ``testId`` prop). */
  testId?: string;
  /** Raw system info from ``getSystemInfoFromAttachment``; used to populate the
   *  hardware/software footer strip in exported PNG files. */
  systemInfo?: any;
}

/**
 * Renders the ``Charts`` collapsible section inside a test's expanded detail
 * row.  The section is invisible when ``extendedMetadata.charts`` is absent
 * or empty, so it never appears for tests that don't produce chart data.
 */
export const ChartsSection: FunctionComponent<ChartsSectionProps> = ({
  extendedMetadata,
  summaryMeta,
  testId,
  systemInfo,
}) => {
  // Expanded by default so charts are immediately visible on row open.
  const [isExpanded, setIsExpanded] = useState(true);

  const charts: ChartData[] = Array.isArray(extendedMetadata?.charts)
    ? extendedMetadata.charts
    : [];

  if (charts.length === 0) return null;

  return (
    <div className={styles.chartsSection}>
      <button
        type="button"
        className={styles.chartsSectionHeader}
        onClick={() => setIsExpanded((v) => !v)}
      >
        <span className={styles.chartsToggle}>{isExpanded ? "\u2212" : "+"}</span>
        <span className={styles.chartsSectionTitle}>Charts</span>
      </button>

      {isExpanded && (
        <div className={styles.chartsGrid}>
          {charts.map((chart) => (
            <ChartCard key={chart.id} chart={chart} summaryMeta={summaryMeta} testId={testId} systemInfo={systemInfo} />
          ))}
        </div>
      )}
    </div>
  );
};
