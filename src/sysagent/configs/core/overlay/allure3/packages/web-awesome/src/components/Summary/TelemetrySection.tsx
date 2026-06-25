import { FunctionComponent } from "preact";
import { useMemo, useState } from "preact/hooks";
import { TelemetryChart } from "./TelemetryChart";
import { KpiCostProfile } from "./KpiCostProfile";
import * as styles from "./TelemetryChartStyle.scss";

// Backend sentinel meaning "source could not be read". Rendered as "no data".
const MISSING_VALUE = -1;

interface TelemetrySectionProps {
  /** The `extended_metadata` object from the Core Metrics Test Results JSON */
  extendedMetadata?: any;
  /** Summary metadata (cli_name, platform, timestamp) from test_summary.json */
  summaryMeta?: { cliName: string; platform: string; timestamp: string } | null;
  /** The full UUID of the test result, used to generate unique PNG filenames */
  testId?: string;
}

/**
 * TelemetrySection
 *
 * Renders a collapsible section that displays one SVG chart per telemetry
 * module found in `extended_metadata.telemetry.modules`.  The chart type
 * (line | area | bar_vertical) is determined by each module's
 * `configs.chart_type` field.
 */
export const TelemetrySection: FunctionComponent<TelemetrySectionProps> = ({
  extendedMetadata,
  summaryMeta,
  testId,
}) => {
  type MetricCard = {
    device: string;
    module: string;
    metric: string;
    metricLabel: string;
    unit: string;
    avg?: number;
    min?: number;
    max?: number;
    samples: number;
    hasData: boolean;
    moduleData?: any;
  };

  type DeviceGroup = {
    device: string;
    modules: Array<[string, any]>;
    metricRows: MetricCard[];
  };

  const DEVICE_ORDER = ["System", "CPU", "iGPU", "dGPU", "NPU", "Other"];
  const METRIC_ORDER = [
    "utilization",
    "frequency_mhz",
    "temperature_c",
    "power_w",
    "memory_utilization",
    "memory_used_mb",
    "bandwidth_mb_s",
  ];

  const toTitle = (text: string) =>
    text
      .replace(/[_.]+/g, " ")
      .replace(/\b\w/g, (s) => s.toUpperCase())
      .trim();

  const inferDevice = (moduleName: string, moduleData?: any, metricKey?: string): string => {
    // 1. Module-provided device_name wins (platform_telemetry / external
    //    collectors set this per-module). Sysfs modules do not.
    const provided = String(moduleData?.device_name || "").trim();
    if (provided) return provided;

    const name   = String(moduleName || "").toLowerCase();
    const metric = String(metricKey  || "").toLowerCase();

    // 2. Per-metric prefix routing for sysfs gpu_*/npu_* modules whose
    //    metric keys carry the device index (``gpu_0_pkg_c``, ``npu_0_busy_pct``).
    //    Sysfs gpu modules populate ``configs.scales[metric].label`` with a
    //    human-readable hint produced by ``_drm.get_gpu_label_map()``
    //    (e.g. "iGPU Pkg", "Arc A770 Pkg"); use it to disambiguate iGPU vs dGPU.
    if (/^gpu_\d+_/.test(metric)) {
      const label = String(moduleData?.configs?.scales?.[metricKey ?? ""]?.label || "").toLowerCase();
      if (label.includes("igpu")) return "iGPU";
      if (label.includes("dgpu") || label.includes("arc") || label.includes("data center")) return "dGPU";
      return "GPU";
    }
    if (/^npu_\d+_/.test(metric)) return "NPU";

    // 3. Module-name fallback (no per-metric hint).
    if (name.includes("dgpu")) return "dGPU";
    if (name.includes("igpu")) return "iGPU";
    if (name.startsWith("gpu_") || name === "gpu_temp" || name === "gpu_freq" ||
        name === "gpu_power" || name === "gpu_usage") return "GPU";
    if (name.includes("npu")) return "NPU";
    if (name.startsWith("cpu_")) return "CPU";
    // memory_usage and power/package rails are host-level telemetry.
    if (name === "memory_usage" || name.startsWith("package_") || name === "power") return "System";
    return "Other";
  };

  const deviceSortOrder = (device: string): [number, number, string] => {
    if (device === "System") return [0, 0, device];
    if (device === "CPU") return [1, 0, device];
    if (device === "iGPU") return [2, 0, device];
    if (device === "dGPU") return [3, 0, device];
    if (device.startsWith("dGPU[") && device.endsWith("]")) {
      const idx = Number.parseInt(device.slice(5, -1), 10);
      return [3, Number.isFinite(idx) ? idx + 1 : 99, device];
    }
    if (device === "NPU") return [4, 0, device];
    const base = DEVICE_ORDER.indexOf(device);
    return [base === -1 ? 99 : base, 0, device];
  };

  const canonicalMetricKey = (metric: string, moduleName?: string): string | null => {
    const m   = String(metric || "").toLowerCase();
    const mod = String(moduleName || "").toLowerCase();

    // ---- Memory family (must precede generic _percent / _gib matches) -----
    //   Long form (platform_telemetry):  ``memory_utilization`` (%),
    //                                  ``memory_used_mb`` (size),
    //                                  ``memory_available`` (size)
    //   Sysfs memory_usage module:     ``used_percent`` (%) /
    //                                  ``available_gib`` (size) / ``used_gib`` (size)
    //   Sysfs gpu/npu mem keys:        ``gpu_0_mem_mib`` / ``npu_0_mem_mib`` (size)
    //
    // Size-style keys (``_mb`` / ``_mib`` / ``_gib`` / ``memory_used`` /
    // ``memory_available``) are routed to the dedicated ``memory_used_mb``
    // bucket so the chart axis stays in MB and the value is never
    // mistaken for a 0-100 % reading. Only true percent metrics
    // (``memory_utilization`` / ``used_percent``) land in
    // ``memory_utilization``.
    if (m.includes("memory") && m.includes("utilization")) return "memory_utilization";
    if (mod === "memory_usage" && /(^|_)used_percent($|_)/.test(m)) return "memory_utilization";
    if (m.includes("memory_used") || m.includes("memory_available")) return "memory_used_mb";
    if (/(^|_)mem(_|$)/.test(m) || /_(gib|mib|mb)(_|$)/.test(m) || /_(gib|mib|mb)$/.test(m)) return "memory_used_mb";
    if (mod === "memory_usage") return "memory_used_mb";

    if (m.includes("bandwidth")) return "bandwidth_mb_s";

    // ---- Frequency: "frequency" word OR ``_mhz`` suffix ------------------
    //   sysfs cpu_freq:   ``current_mhz``
    //   sysfs gpu_freq:   ``gpu_0_gt0_mhz``
    //   sysfs npu_freq:   ``npu_0_freq_mhz``
    if (m.includes("frequency") || /_mhz$/.test(m) || /_mhz_/.test(m)) return "frequency_mhz";

    // ---- Temperature: "temperature" word OR sysfs ``_c`` suffix ----------
    //   sysfs cpu_temp:   ``package_c`` / ``core_max_c``
    //   sysfs gpu_temp:   ``gpu_0_pkg_c`` / ``gpu_0_vram_c``
    if (m.includes("temperature")) return "temperature_c";
    if (/_c$/.test(m) && !/_(pct|gib|mib|mb_s|mhz|hz)_c$/.test(m)) return "temperature_c";

    // ---- Power: "power" word OR sysfs terminal ``_w`` suffix --------------
    //   sysfs gpu_power:  ``gpu_0_w`` / ``gpu_0_card_w``
    //   sysfs power:      ``<rail>_power_w``
    if (m.includes("power") || /_w$/.test(m)) return "power_w";

    // ---- Utilization: explicit OR sysfs ``_pct`` / ``_percent`` / ``_busy`` -
    //   sysfs cpu_usage:  ``total_percent``
    //   sysfs gpu_usage:  ``gpu_0_gt0_pct`` / ``gpu_0_render_pct``
    //   sysfs npu_usage:  ``npu_0_busy_pct``
    if (m.includes("utilization_idle") || m.includes("utilization_sys")) return null;
    if (m.includes("utilization") || /_(pct|percent|busy)$/.test(m) || /_(busy)_/.test(m)) return "utilization";

    return null;
  };

  const metricOrder = (metric: string): number => {
    const idx = METRIC_ORDER.findIndex((k) => metric === k);
    return idx === -1 ? 99 : idx;
  };

  const friendlyMetric = (metric: string): string => {
    const m = canonicalMetricKey(metric) ?? String(metric || "").toLowerCase();
    if (m === "memory_utilization") return "Memory Utilization";
    if (m === "memory_used_mb") return "Memory Used (MB)";
    if (m === "utilization") return "Utilization";
    if (m === "frequency_mhz") return "Frequency";
    if (m === "power_w") return "Power";
    if (m === "bandwidth_mb_s") return "Bandwidth";
    if (m === "temperature_c") return "Temperature";
    return toTitle(metric);
  };

  const memoryUsedPreferenceRank = (metric: string): number => {
    const m = String(metric || "").toLowerCase();
    if (m.includes("memory_used") || /(^|_)used(_|$)/.test(m)) return 2;
    if (m.includes("memory_available") || /(^|_)available(_|$)/.test(m)) return 1;
    return 0;
  };

  // Filename-safe sanitiser: lower-case, collapse any non-alphanumeric run
  // to a single underscore, trim leading/trailing underscores. Mirrors the
  // helper used by ``TelemetryChart`` for PNG export filenames so the CSV
  // and PNG exports from the same test run share a common naming scheme.
  const sanitizeForFilename = (s: string): string =>
    String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");

  // Build a CSV filename that mirrors the PNG export pattern produced by
  // ``TelemetryChart`` — ``<cli>_telemetry_<testId>_metrics_<timestamp>.csv``
  // when summary metadata + test id are available, falling back to a generic
  // ``telemetry_metrics.csv`` for standalone previews.
  const buildCsvFilename = (): string => {
    if (summaryMeta && testId) {
      const cli = sanitizeForFilename(summaryMeta.cliName);
      const tid = sanitizeForFilename(testId);
      const ts  = sanitizeForFilename(summaryMeta.timestamp);
      return `${cli}_telemetry_${tid}_metrics_${ts}.csv`;
    }
    return "telemetry_metrics.csv";
  };

  const exportRowsToCsv = (rows: MetricCard[]) => {
    if (!rows.length) return;
    const headers = ["Device", "Module", "Metric", "Unit", "Avg", "Max", "Min", "Samples"];
    const lines = rows.map((row) => [
      row.device,
      row.module,
      row.metricLabel,
      row.unit,
      row.avg ?? "",
      row.max ?? "",
      row.min ?? "",
      row.samples,
    ]);
    const csv = [headers, ...lines]
      .map((line) =>
        line
          .map((value) => {
            const text = String(value ?? "");
            return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
          })
          .join(","),
      )
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = buildCsvFilename();
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  };

  const [isExpanded, setIsExpanded] = useState(false);

  const telemetry = extendedMetadata?.telemetry;

  // Nothing to render if telemetry is disabled or has no modules
  if (!telemetry?.enabled || !telemetry.modules) return null;

  // Flatten nested ``device_groups`` into ``[virtualModuleName, summary]``
  // pairs so the rest of the rendering pipeline can treat each device as an
  // independent chart source. The new backend layout keeps the parent key
  // matching the source-file module name (``platform_telemetry``) and nests
  // per-device entries underneath; older payloads still come through as flat
  // top-level keys and pass through untouched.
  const moduleEntries: Array<[string, any]> = [];
  for (const [name, mod] of Object.entries(telemetry.modules as Record<string, any>)) {
    if (mod && typeof mod === "object" && Array.isArray((mod as any).device_groups) && (mod as any).device_groups.length > 0) {
      for (const sub of (mod as any).device_groups) {
        if (sub && typeof sub === "object") {
          const subName = String((sub as any).module || name);
          moduleEntries.push([subName, sub]);
        }
      }
      continue;
    }
    moduleEntries.push([name, mod]);
  }
  if (moduleEntries.length === 0) return null;

  const deviceGroups = useMemo<DeviceGroup[]>(() => {
    // Per-metric routing: for sysfs modules a single module (e.g. ``gpu_temp``)
    // can carry metrics for multiple physical devices (``gpu_0_*`` ⇒ iGPU,
    // ``gpu_1_*`` ⇒ dGPU). We therefore build the (device, canonical metric)
    // table by visiting every (module, metric) pair individually.
    const cardsByDevice = new Map<string, Map<string, MetricCard>>();
    const modulesByDevice = new Map<string, Map<string, any>>();

    moduleEntries.forEach(([moduleName, mod]) => {
      const samples = mod?.samples ?? [];
      const scales = mod?.configs?.scales ?? {};
      const configMetrics: string[] = mod?.configs?.metrics ?? [];
      const sampleKeys = Object.keys(samples[0]?.values ?? {});
      const averageKeys = Object.keys(mod?.averages ?? {});
      const minMaxKeys = Object.keys(mod?.min_max ?? {});
      const metricKeys = configMetrics.length > 0
        ? configMetrics
        : Array.from(new Set([...sampleKeys, ...averageKeys, ...minMaxKeys]));

      metricKeys.forEach((metric) => {
        const canonical = canonicalMetricKey(metric, moduleName);
        if (!canonical) return;
        const device = inferDevice(moduleName, mod, metric);

        const cards = cardsByDevice.get(device) ?? new Map<string, MetricCard>();
        if (cards.has(canonical)) {
          // For the unified Memory Used card, prefer ``used`` signals over
          // ``available`` when both are present in the same module.
          if (canonical === "memory_used_mb") {
            const existing = cards.get(canonical);
            const existingRawMetric = String(existing?.moduleData?.rawMetric || "");
            const shouldReplace =
              memoryUsedPreferenceRank(metric) > memoryUsedPreferenceRank(existingRawMetric);
            if (shouldReplace) {
              cards.delete(canonical);
            } else {
              return;
            }
          } else {
          // Already have a card for this (device, canonical) pair; keep the
          // first one to avoid double-counting metrics that map to the same
          // bucket (e.g. ``gpu_0_pkg_c`` + ``gpu_0_vram_c`` both map to
          // ``temperature_c``). The chart renders the first metric only.
            return;
          }
        }
        cardsByDevice.set(device, cards);

        const mods = modulesByDevice.get(device) ?? new Map<string, any>();
        if (!mods.has(moduleName)) mods.set(moduleName, mod);
        modulesByDevice.set(device, mods);

        const minMax = mod?.min_max?.[metric] ?? {};
        // "No data" = no finite non-zero reading, and -1 (MISSING_VALUE)
        // is treated as missing too. Such metrics render the
        // "No telemetry data collected" placeholder instead of a flat
        // line or a -1 spike.
        let sawNonZero = false;
        for (const s of samples) {
          const v = s?.values?.[metric];
          if (v != null && Number.isFinite(v) && v !== 0 && v !== MISSING_VALUE) {
            sawNonZero = true;
            break;
          }
        }
        const avgVal = mod?.averages?.[metric];
        const minVal = minMax?.min;
        const maxVal = minMax?.max;
        const isValidAggregate = (v: unknown): boolean =>
          typeof v === "number" && Number.isFinite(v) && v !== 0 && v !== MISSING_VALUE;
        const summaryHasNonZero =
          isValidAggregate(avgVal) || isValidAggregate(minVal) || isValidAggregate(maxVal);
        const hasData = sawNonZero || summaryHasNonZero;
        // Map -1 (MISSING_VALUE) aggregates to undefined so the UI shows
        // "--" instead of "-1.00".
        const sanitizeAggregate = (v: unknown): number | undefined =>
          typeof v === "number" && Number.isFinite(v) && v !== MISSING_VALUE
            ? v
            : undefined;
        const displayAvg = sanitizeAggregate(avgVal);
        const displayMin = sanitizeAggregate(minVal);
        const displayMax = sanitizeAggregate(maxVal);
        cards.set(canonical, {
          device,
          module: moduleName,
          metric: canonical,
          metricLabel: friendlyMetric(canonical),
          unit: scales?.[metric]?.unit ?? "",
          avg: displayAvg,
          min: displayMin,
          max: displayMax,
          samples: samples.length,
          hasData,
          moduleData: {
            moduleName,
            mod,
            rawMetric: metric,
            scaleCfg: scales?.[metric] ?? {},
          },
        });
      });
    });

    // Fold bare "dGPU"/"iGPU" into the first indexed variant when both exist,
    // preferring cards that actually have data. Avoids duplicate device sections.
    const foldBareIntoIndexed = (bare: string) => {
      if (!cardsByDevice.has(bare)) return;
      const indexed = Array.from(cardsByDevice.keys())
        .filter((k: string) => k.startsWith(`${bare}[`) && k.endsWith("]"))
        .sort();
      if (indexed.length === 0) return;
      const target = indexed[0];
      const bareCards = cardsByDevice.get(bare) as Map<string, MetricCard>;
      const targetCards = cardsByDevice.get(target) as Map<string, MetricCard>;
      bareCards.forEach((card: MetricCard, key: string) => {
        const existing = targetCards.get(key);
        if (!existing || (!existing.hasData && card.hasData)) {
          targetCards.set(key, { ...card, device: target });
        }
      });
      const bareModules = modulesByDevice.get(bare);
      if (bareModules) {
        const targetModules = modulesByDevice.get(target) ?? new Map<string, any>();
        bareModules.forEach((m: any, name: string) => {
          if (!targetModules.has(name)) targetModules.set(name, m);
        });
        modulesByDevice.set(target, targetModules);
      }
      cardsByDevice.delete(bare);
      modulesByDevice.delete(bare);
    };
    foldBareIntoIndexed("dGPU");
    foldBareIntoIndexed("iGPU");

    const sortedDevices = Array.from(cardsByDevice.keys()).sort((a, b) => {
      const [a1, a2] = deviceSortOrder(a);
      const [b1, b2] = deviceSortOrder(b);
      return a1 - b1 || a2 - b2 || a.localeCompare(b);
    });

    return sortedDevices.map((device) => {
      const metricCards = cardsByDevice.get(device) ?? new Map<string, MetricCard>();
      const modules: Array<[string, any]> = Array.from(
        (modulesByDevice.get(device) ?? new Map<string, any>()).entries(),
      );

      // Render only metrics the collector actually produced for this device.
      // Sysfs (built-in) modules expose a narrower set than platform_telemetry —
      // e.g. ``cpu_*`` modules have no bandwidth/memory; ``gpu_*`` modules
      // have no bandwidth. Emitting empty placeholders for those buckets
      // creates "no data" cards that confuse the reader.
      const metricRows = METRIC_ORDER
        .filter((metric) => metricCards.has(metric))
        .map((metric) => metricCards.get(metric) as MetricCard);

      return { device, modules, metricRows };
    });
  }, [moduleEntries]);

  const dashboardRows = useMemo<MetricCard[]>(() => deviceGroups.flatMap((group: DeviceGroup) => group.metricRows), [deviceGroups]);

  // Derive the list of Primary Compute Devices actually present in this
  // report. Multi-tile dGPUs (``dGPU[0]``, ``dGPU[1]``) collapse to a
  // single ``dGPU`` token, and unknown buckets are filtered out so the
  // caption stays focused on real silicon.
  const primaryDevices: string[] = useMemo(() => {
    const seen = new Set<string>();
    const order: string[] = [];
    deviceGroups.forEach((g: DeviceGroup) => {
      let token = g.device;
      if (token.startsWith("dGPU[")) token = "dGPU";
      if (!seen.has(token) && ["CPU", "iGPU", "dGPU", "NPU"].includes(token)) {
        seen.add(token);
        order.push(token);
      }
    });
    return order;
  }, [deviceGroups]);
  const primaryDevicesLabel = primaryDevices.join(" / ");

  const toggle = () => setIsExpanded((v: boolean) => !v);

  const deviceClass = (device: string): string => {
    if (device === "CPU") return styles.deviceCPU;
    if (device === "iGPU") return styles.deviceiGPU;
    if (device === "dGPU" || device.startsWith("dGPU[")) return styles.devicedGPU;
    if (device === "NPU") return styles.deviceNPU;
    return styles.deviceOther;
  };

  return (
    <div className={styles.telemetrySection}>
      {/* ── Section header (click to expand / collapse) ── */}
      <div
        className={styles.telemetrySectionHeader}
        onClick={toggle}
        role="button"
        aria-expanded={isExpanded}
      >
        <button
          className={styles.telemetryToggle}
          type="button"
          tabIndex={-1}
          aria-hidden="true"
        >
          {isExpanded ? "−" : "+"}
        </button>
        <h4 className={styles.telemetrySectionTitle}>Telemetry</h4>
      </div>

      {/* ── Charts grid (one card per module) ── */}
      {isExpanded && (
        <>
          {/*
            ``telemetry`` is guaranteed non-null past the early return on
            line 156 (``if (!telemetry?.enabled || !telemetry.modules) return null;``).
            Drop the optional chain on the access path so the static
            analyser does not re-introduce the null assumption here.
          */}
          <KpiCostProfile correlation={telemetry.kpi_correlation} />
          <div className={styles.telemetryToolbar}>
            <div />
            <button className={styles.telemetryExportBtn} type="button" onClick={() => exportRowsToCsv(dashboardRows)}>
              Export CSV
            </button>
          </div>
          <div className={styles.telemetryGraphsWrap}>
            <div className={styles.telemetryGraphsTitle}>Metrics Graphs</div>
            {primaryDevicesLabel && (
              <div className={styles.telemetrySectionCaption}>
                <strong>Compute Device:</strong> {primaryDevicesLabel}
              </div>
            )}
          {deviceGroups.map((group) => {
            // Per-device sub-sections are always expanded (collapse/expand
            // affordance intentionally removed). The outer Telemetry section
            // toggle still controls visibility of the whole block.
            return (
              <section key={group.device} className={styles.deviceGroup}>
                <div className={`${styles.deviceGroupHeader} ${deviceClass(group.device)}`}>
                  <span className={styles.deviceGroupTitle}>{group.device}</span>
                </div>

                <div className={styles.chartsGrid}>
                    {group.metricRows.map((row) => {
                      const moduleName = row.moduleData?.moduleName;
                      const mod = row.moduleData?.mod;
                      const rawMetric = row.moduleData?.rawMetric;
                      const scaleCfg = row.moduleData?.scaleCfg ?? {};
                      const chartType = (((mod?.configs?.chart_type as string) || "line") as "line" | "area" | "bar_vertical");
                      const unitText = row.unit ? ` ${row.unit}` : "";
                      const baselineForMetric = rawMetric ? (mod?.baseline?.metrics?.[rawMetric] as { min?: number; avg?: number; max?: number } | undefined) : undefined;
                      // Treat -1 (MISSING_VALUE) as "no baseline" so the
                      // delta chip is hidden rather than misleading.
                      const rawIdleAvg = baselineForMetric?.avg;
                      const idleAvg =
                        typeof rawIdleAvg === "number" && rawIdleAvg !== MISSING_VALUE
                          ? rawIdleAvg
                          : undefined;
                      const workloadAvg = row.avg;
                      const workloadMax = row.max;

                      const formatDelta = (loadValue: number | undefined, label: string): string | null => {
                        if (typeof idleAvg !== "number" || typeof loadValue !== "number") return null;
                        if (loadValue === MISSING_VALUE) return null;
                        const diff = loadValue - idleAvg;
                        let value: string;
                        if (Math.abs(idleAvg) >= 0.1) {
                          const pct = (diff / idleAvg) * 100;
                          const sign = pct >= 0 ? "+" : "";
                          value = `${sign}${pct.toFixed(0)}%`;
                        } else {
                          const sign = diff >= 0 ? "+" : "";
                          value = `${sign}${diff.toFixed(2)}${unitText}`;
                        }
                        return `${label} ${value}`;
                      };
                      const avgDeltaText  = formatDelta(workloadAvg, "avg");
                      const peakDeltaText = formatDelta(workloadMax, "peak");

                      return (
                        <article key={`${group.device}-${row.metric}`} className={styles.kpiCard}>
                          <div className={styles.kpiCardName}>{row.metricLabel}</div>
                          <div className={styles.metricSingleChart}>
                            {typeof idleAvg === "number" && (
                              <span className={styles.idleChip} title={`Pre-run idle baseline: ${idleAvg.toFixed(2)}${unitText}`}>
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
                            {row.hasData && moduleName && mod && rawMetric ? (
                              <TelemetryChart
                                moduleName={moduleName}
                                chartType={chartType}
                                samples={mod?.samples ?? []}
                                thresholds={mod?.configs?.thresholds}
                                visibleMetrics={[rawMetric]}
                                configTitle={{ display: false, text: `${group.device} ${row.metricLabel}` }}
                                configScales={{ [rawMetric]: { ...scaleCfg, label: row.metricLabel } }}
                                axes={mod?.configs?.axes}
                                baseline={baselineForMetric ? { [rawMetric]: baselineForMetric } : undefined}
                                averages={typeof row.avg === "number" ? { [rawMetric]: row.avg } : undefined}
                                minMax={
                                  typeof row.min === "number" && typeof row.max === "number"
                                    ? { [rawMetric]: { min: row.min, max: row.max } }
                                    : undefined
                                }
                                summaryMeta={summaryMeta}
                                testId={testId}
                                compact={true}
                              />
                            ) : (
                              <div className={styles.emptyMetricState}>No telemetry data available</div>
                            )}
                          </div>
                          <div className={styles.kpiValueRow}>
                            <span className={styles.kpiAggregateTag} title="Average across all workload samples">avg</span>
                            <span className={styles.kpiValue} title="Average across all workload samples">{row.avg !== undefined ? row.avg.toFixed(2) : "--"}</span>
                            <span className={styles.kpiUnit}>{row.unit}</span>
                          </div>
                          <div className={styles.kpiStatsRow}>
                            <span>Min {row.min !== undefined ? `${row.min.toFixed(2)}${unitText}` : "--"}</span>
                            <span>Max {row.max !== undefined ? `${row.max.toFixed(2)}${unitText}` : "--"}</span>
                          </div>
                          {(typeof idleAvg === "number" || avgDeltaText || peakDeltaText) && (
                            <div className={styles.kpiBaselineRow}>
                              {typeof idleAvg === "number" ? (
                                <span className={styles.kpiBaselineLabel}>Idle {idleAvg.toFixed(2)}{unitText}</span>
                              ) : (
                                <span />
                              )}
                              {(avgDeltaText || peakDeltaText) && (
                                <span
                                  className={styles.kpiBaselineDeltaBox}
                                  title="Workload avg / peak relative to pre-run idle baseline"
                                >
                                  <span className={styles.kpiBaselineDeltaTag}>Over Idle</span>
                                  <span className={styles.kpiBaselineDelta}>
                                    {[avgDeltaText, peakDeltaText].filter(Boolean).join(" \u00B7 ")}
                                  </span>
                                </span>
                              )}
                            </div>
                          )}
                        </article>
                      );
                    })}
                  </div>
              </section>
            );
          })}
          </div>
          <div className={styles.telemetryMetricTableWrap}>
            <div className={styles.telemetryMetricTableTitle}>Metrics Table</div>
            {primaryDevicesLabel && (
              <div className={styles.telemetrySectionCaption}>
                <strong>Compute Device:</strong> {primaryDevicesLabel}
              </div>
            )}
            {deviceGroups.map((group) => {
              // Per-device table is always expanded; collapse affordance
              // intentionally removed per UX review.
              return (
                <div key={group.device} className={styles.telemetryMetricTableDevice}>
                  <div className={`${styles.tableGroupHeader} ${deviceClass(group.device)}`}>
                    <span className={styles.tableGroupTitle}>{group.device}</span>
                  </div>
                  {(() => {
                    // Hide the Idle column for collectors that do not capture
                    // a pre-run idle baseline (e.g. sysfs profiles without
                    // ``telemetry.prerun.enabled``). Decide once per device
                    // group based on whether any row has an idle value.
                    const idleValues: Array<number | undefined> = group.metricRows.map(
                      (row: MetricCard): number | undefined =>
                        row.moduleData?.mod?.baseline?.metrics?.[row.moduleData?.rawMetric]?.avg,
                    );
                    const hasIdle = idleValues.some((v) => typeof v === "number");
                    return (
                      <table className={styles.telemetryMetricTable}>
                        <thead>
                          <tr>
                            <th>Metric</th>
                            <th>Unit</th>
                            {hasIdle && <th>Idle</th>}
                            <th>Avg</th>
                            <th>Max</th>
                            <th>Min</th>
                            <th>Samples</th>
                          </tr>
                        </thead>
                        <tbody>
                          {group.metricRows.map((row, rowIdx) => {
                            const idleAvgVal = idleValues[rowIdx];
                            return (
                              <tr key={`${row.device}-${row.metric}`}>
                                <td>{row.metricLabel}</td>
                                <td>{row.unit}</td>
                                {hasIdle && (
                                  <td>{typeof idleAvgVal === "number" && idleAvgVal !== MISSING_VALUE ? idleAvgVal.toFixed(4) : ""}</td>
                                )}
                                <td>{row.avg !== undefined ? row.avg.toFixed(4) : ""}</td>
                                <td>{row.max !== undefined ? row.max.toFixed(4) : ""}</td>
                                <td>{row.min !== undefined ? row.min.toFixed(4) : ""}</td>
                                <td>{row.samples}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    );
                  })()}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
};
