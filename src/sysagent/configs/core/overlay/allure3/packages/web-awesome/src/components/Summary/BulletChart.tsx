import { FunctionComponent } from "preact";
import * as styles from "./BulletChartStyle.scss";

interface BulletChartData {
  metric: string;
  value: number;
  reference: number;
  unit: string;
  status: "passed" | "failed" | "broken" | "skipped" | "unknown";
}

interface BulletChartProps {
  data: BulletChartData[];
}

export const BulletChart: FunctionComponent<BulletChartProps> = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className={styles.emptyState}>
        No metrics data available for visualization
      </div>
    );
  }

  return (
    <div className={styles.bulletChartContainer}>
      {data.map((metric, index) => {
        // Calculate percentages for positioning
        const maxValue = Math.max(metric.value, metric.reference, metric.reference * 1.2);
        const valuePercent = (metric.value / maxValue) * 100;
        const referencePercent = (metric.reference / maxValue) * 100;
        const referenceZoneStart = 0; // Start from 0
        const referenceZoneWidth = referencePercent - referenceZoneStart;

        const statusClass = metric.status === "passed" ? styles.passed : styles.failed;

        return (
          <div key={index} className={styles.bulletRow}>
            {/* Metric title */}
            <div className={styles.metricTitle}>{metric.metric}</div>

            {/* Chart area */}
            <div className={styles.chartArea}>
              {/* Background range */}
              <div className={styles.backgroundRange}>
                {/* Reference threshold zone */}
                <div
                  className={`${styles.referenceZone} ${statusClass}`}
                  style={{
                    left: `${referenceZoneStart}%`,
                    width: `${referenceZoneWidth}%`,
                  }}
                />

                {/* Actual value bar */}
                <div
                  className={`${styles.actualBar} ${statusClass}`}
                  style={{ width: `${valuePercent}%` }}
                />

                {/* Reference marker line */}
                <div
                  className={styles.referenceMarker}
                  style={{ left: `${referencePercent}%` }}
                />

                {/* Reference value label (above the marker) */}
                <div
                  className={styles.referenceLabel}
                  style={{ left: `${referencePercent}%` }}
                >
                  Ref: {metric.reference.toFixed(1)} {metric.unit}
                </div>

                {/* Actual value label (to the right of the bar) */}
                <div
                  className={styles.actualLabel}
                  style={{ left: `${valuePercent}%`, marginLeft: '5px' }}
                >
                  Actual: {metric.value.toFixed(1)} {metric.unit}
                </div>
              </div>

              {/* Scale labels */}
              <div className={styles.scaleLabels}>
                <span className={styles.minLabel}>0</span>
                <span className={styles.maxLabel}>
                  {maxValue.toFixed(1)} {metric.unit}
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};
