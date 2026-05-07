/**
 * Attachment helper functions for Summary components.
 *
 * These are standalone async utilities that fetch and parse specific JSON
 * attachments produced by the CLI test runner (Core Metrics, System Info,
 * test_summary.json, etc.).  Each function is pure (no component state)
 * so they can be imported from any module.
 */

import { fetchAttachment } from "@allurereport/web-commons";

// ─── Key metric ───────────────────────────────────────────────────────────────

/**
 * Returns the most important metric from a test result's "Core Metrics Test
 * Results" attachment.  Prefers metrics flagged with `is_key_metric: true`;
 * falls back to the first metric found.
 */
export async function getMetricsFromAttachment(
  testResult: any,
): Promise<{ metric: string; value: string; unit: string }> {
  if (!testResult?.attachments || !Array.isArray(testResult.attachments)) {
    return { metric: "N/A", value: "N/A", unit: "N/A" };
  }

  const metricsAttachment = testResult.attachments.find(
    (attachment: any) =>
      attachment.link?.name === "Core Metrics Test Results" &&
      attachment.link?.contentType === "application/json",
  );

  if (!metricsAttachment) {
    return { metric: "N/A", value: "N/A", unit: "N/A" };
  }

  try {
    const result = await fetchAttachment(
      metricsAttachment.link.id,
      metricsAttachment.link.ext,
      metricsAttachment.link.contentType,
    );

    if (!result || !result.text) {
      return { metric: "N/A", value: "N/A", unit: "N/A" };
    }

    const metricsData = JSON.parse(result.text);

    if (metricsData?.metrics && typeof metricsData.metrics === "object") {
      const metrics = Object.entries(metricsData.metrics);
      if (metrics.length > 0) {
        let selectedMetric: [string, unknown] | null = null;

        for (const [metricKey, metricValue] of metrics) {
          if (
            typeof metricValue === "object" &&
            metricValue !== null &&
            (metricValue as any).is_key_metric === true
          ) {
            selectedMetric = [metricKey, metricValue];
            break;
          }
        }

        if (!selectedMetric) {
          selectedMetric = metrics[0];
        }

        const [metricKey, metricValue] = selectedMetric;
        if (
          typeof metricValue === "object" &&
          metricValue !== null &&
          "value" in metricValue &&
          "unit" in metricValue
        ) {
          return {
            metric: metricKey,
            value: String((metricValue as any).value),
            unit: String((metricValue as any).unit),
          };
        }
        return { metric: metricKey, value: String(metricValue), unit: "" };
      }
    }
  } catch (error) {
    console.error("Error fetching metrics attachment:", error);
  }

  return { metric: "N/A", value: "N/A", unit: "N/A" };
}

// ─── Full metrics object ──────────────────────────────────────────────────────

/**
 * Returns the entire parsed JSON from the "Core Metrics Test Results"
 * attachment (used by MetadataTable and for KPI / reference lookups).
 */
export async function getFullMetricsFromAttachment(
  testResult: any,
): Promise<any | null> {
  if (!testResult?.attachments || !Array.isArray(testResult.attachments)) {
    return null;
  }

  const metricsAttachment = testResult.attachments.find(
    (attachment: any) =>
      attachment.link?.name === "Core Metrics Test Results" &&
      attachment.link?.contentType === "application/json",
  );

  if (!metricsAttachment) {
    return null;
  }

  try {
    const result = await fetchAttachment(
      metricsAttachment.link.id,
      metricsAttachment.link.ext,
      metricsAttachment.link.contentType,
    );

    if (!result || !result.text) {
      return null;
    }

    return JSON.parse(result.text);
  } catch (error) {
    console.error("Error fetching full metrics attachment:", error);
  }

  return null;
}

// ─── KPI reference value ──────────────────────────────────────────────────────

/**
 * Extracts the formatted reference/threshold string from the first KPI that
 * has a `validation.reference` value (e.g. ">= 40 tokens/sec").
 */
export async function getReferenceFromAttachment(
  testResult: any,
): Promise<string> {
  const fullMetrics = await getFullMetricsFromAttachment(testResult);

  if (!fullMetrics?.kpis || Object.keys(fullMetrics.kpis).length === 0) {
    return "N/A";
  }

  const operatorDisplayMap: Record<string, string> = {
    eq: "==",
    neq: "!=",
    gt: ">",
    gte: ">=",
    lt: "<",
    lte: "<=",
    between: "between",
    contains: "contains",
    not_contains: "not contains",
    matches: "matches",
    in: "in",
    not_in: "not in",
  };

  for (const [, kpiData] of Object.entries(fullMetrics.kpis)) {
    if (kpiData && typeof kpiData === "object") {
      const config = (kpiData as any).config || {};
      const validation = config.validation || {};

      if (validation.reference !== undefined) {
        const operator = validation.operator || "gte";
        const operatorSymbol = operatorDisplayMap[operator] || operator;
        const reference = validation.reference;
        const unit = config.unit || "";

        return `${operatorSymbol} ${reference}${unit ? ` ${unit}` : ""}`;
      }
    }
  }

  return "N/A";
}

// ─── System information ───────────────────────────────────────────────────────

/**
 * Fetches the "Core CLI System Information" JSON attachment from the special
 * `core / test_system` test result and returns the parsed system info object.
 */
export async function getSystemInfoFromAttachment(
  individualTestResults: Record<string, any>,
): Promise<any | null> {
  if (Object.keys(individualTestResults).length === 0) {
    return null;
  }

  const systemTest = Object.values(individualTestResults).find(
    (testResult: any) => {
      const suiteLabel = testResult.labels?.find(
        (label: any) => label.name === "suite",
      )?.value;
      const packageLabel = testResult.labels?.find(
        (label: any) => label.name === "package",
      )?.value;
      return suiteLabel === "core" && packageLabel === "test_system";
    },
  );

  if (!systemTest) {
    return null;
  }

  const systemAttachment = (systemTest as any)?.attachments?.find(
    (attachment: any) =>
      attachment.link?.name === "Core CLI System Information" &&
      attachment.link?.contentType === "application/json",
  );

  if (!systemAttachment) {
    return null;
  }

  try {
    const result = await fetchAttachment(
      systemAttachment.link.id,
      systemAttachment.link.ext || ".json",
      systemAttachment.link.contentType,
    );

    if (result?.text) {
      const systemInfo = JSON.parse(result.text);
      return systemInfo.system_info || systemInfo;
    }
  } catch (error) {
    console.error("Error fetching system information attachment:", error);
  }

  return null;
}

// ─── Summary metadata ─────────────────────────────────────────────────────────

/**
 * Finds the `test_summary.json` attachment and returns a map of
 * Allure result UUID → description for every test entry in the summary.
 * All UUIDs listed in `all_run_uuids` are mapped so that any historical
 * run UUID resolves to its description.
 */
export async function getTestDescriptionsFromSummary(
  individualTestResults: Record<string, any>,
): Promise<Record<string, string>> {
  const summaryTest = Object.values(individualTestResults).find(
    (testResult: any) =>
      testResult.labels?.find((l: any) => l.name === "package")?.value === "test_summary",
  );

  if (!summaryTest) return {};

  const summaryAttachment = (summaryTest as any)?.attachments?.find(
    (a: any) =>
      a.link?.name?.includes("test_summary.json") &&
      a.link?.contentType === "application/json",
  );

  if (!summaryAttachment) return {};

  try {
    const result = await fetchAttachment(
      summaryAttachment.link.id,
      summaryAttachment.link.ext || ".json",
      summaryAttachment.link.contentType,
    );

    if (result?.text) {
      const data = JSON.parse(result.text);
      const map: Record<string, string> = {};
      for (const test of (data?.tests ?? [])) {
        const description: string = test?.description ?? "";
        if (!description) continue;
        // Key by test_name — matched against the Allure result's `name` field
        if (test?.test_name) {
          map[test.test_name] = description;
        }
      }
      return map;
    }
  } catch (error) {
    console.error("Error fetching test descriptions from summary:", error);
  }

  return {};
}

/**
 * Finds the `test_summary.json` attachment on the CLI Summary test
 * (`package = "test_summary"`) and returns `{ cliName, platform, timestamp }`.
 * These values are used to generate unique telemetry chart PNG filenames.
 */
export async function getSummaryMetaFromAttachment(
  individualTestResults: Record<string, any>,
): Promise<{ cliName: string; platform: string; timestamp: string } | null> {
  if (Object.keys(individualTestResults).length === 0) {
    return null;
  }

  const summaryTest = Object.values(individualTestResults).find(
    (testResult: any) => {
      const packageLabel = testResult.labels?.find(
        (label: any) => label.name === "package",
      )?.value;
      return packageLabel === "test_summary";
    },
  );

  if (!summaryTest) {
    return null;
  }

  const summaryAttachment = (summaryTest as any)?.attachments?.find(
    (attachment: any) =>
      attachment.link?.name?.includes("test_summary.json") &&
      attachment.link?.contentType === "application/json",
  );

  if (!summaryAttachment) {
    return null;
  }

  try {
    const result = await fetchAttachment(
      summaryAttachment.link.id,
      summaryAttachment.link.ext || ".json",
      summaryAttachment.link.contentType,
    );

    if (result?.text) {
      const data = JSON.parse(result.text);
      const meta = data?.summary?.metadata;
      if (meta?.cli_name && meta?.platform && meta?.timestamp) {
        return {
          cliName: String(meta.cli_name),
          platform: String(meta.platform),
          timestamp: String(meta.timestamp),
        };
      }
    }
  } catch (error) {
    console.error("Error fetching summary metadata attachment:", error);
  }

  return null;
}
