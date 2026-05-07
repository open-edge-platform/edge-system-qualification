import { FunctionComponent } from "preact";
import { useState } from "preact/hooks";
import { TelemetryChart } from "./TelemetryChart";
import * as styles from "./TelemetryChartStyle.scss";

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
  const [isExpanded, setIsExpanded] = useState(false);

  const telemetry = extendedMetadata?.telemetry;

  // Nothing to render if telemetry is disabled or has no modules
  if (!telemetry?.enabled || !telemetry.modules) return null;

  const moduleEntries = Object.entries(
    telemetry.modules as Record<string, any>
  );
  if (moduleEntries.length === 0) return null;

  const toggle = () => setIsExpanded((v) => !v);

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
        <div className={styles.chartsGrid}>
          {moduleEntries.map(([key, mod]) => {
            const chartType = (
              (mod?.configs?.chart_type as string) || "line"
            ) as "line" | "area" | "bar_vertical";

            // If configs.metrics is a non-empty array, restrict to those keys
            const configMetrics: string[] = mod?.configs?.metrics ?? [];

            return (
              <TelemetryChart
                key={key}
                moduleName={key}
                chartType={chartType}
                samples={mod?.samples ?? []}
                averages={mod?.averages}
                minMax={mod?.min_max}
                thresholds={mod?.configs?.thresholds}
                visibleMetrics={configMetrics.length > 0 ? configMetrics : undefined}
                configTitle={mod?.configs?.title}
                configScales={mod?.configs?.scales}
                axes={mod?.configs?.axes}
                summaryMeta={summaryMeta}
                testId={testId}
              />
            );
          })}
        </div>
      )}
    </div>
  );
};
