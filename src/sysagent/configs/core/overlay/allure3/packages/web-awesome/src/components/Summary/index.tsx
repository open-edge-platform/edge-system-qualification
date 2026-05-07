import { useEffect, useState } from "preact/hooks";
import { fetchAttachment } from "@allurereport/web-commons";
import {
  getMetricsFromAttachment,
  getFullMetricsFromAttachment,
  getReferenceFromAttachment,
  getSystemInfoFromAttachment,
  getSummaryMetaFromAttachment,
  getTestDescriptionsFromSummary,
} from "./attachmentHelpers";
import * as styles from "./styles.scss";
import { availableSections } from "../../stores/sections";
import { testResultNavStore, fetchTestResultNav, testResultStore, fetchTestResult } from "../../stores/testResults";
import { BulletChart } from "./BulletChart";
import * as bulletChartStyles from "./BulletChartStyle.scss";
import { TelemetrySection } from "./TelemetrySection";

// Summary table components
interface SortableTableProps {
  title: string;
  data: (string | number)[][];
  headers: string[];
  onSort?: (columnIndex: number, direction: "asc" | "desc") => void;
  onRowClick?: (rowIndex: number) => void;
  expandedRows?: boolean[];
  renderExpandedContent?: (rowIndex: number) => any;
  getRowId?: (rowIndex: number) => string;
}

// System Information table component (for Hardware and Software info)
interface SystemInfoTableProps {
  title: string;
  data: { component: string; details: string[]; packageData?: { [key: string]: string }; expandable?: boolean }[];
  expandedPackageDetails?: Set<string>;
  onTogglePackageDetails?: (packageType: string) => void;
}

// Metadata table component for test metrics
interface MetadataTableProps {
  metricsData: any;
}

// Attachment Image component for Summary
interface AttachmentImageProps {
  attachment: {
    link: {
      id: string;
      contentType: string;
      ext?: string;
    };
    name?: string;
  };
  onError?: () => void;
}

const AttachmentImage = ({ attachment, onError }: AttachmentImageProps) => {
  const [imageData, setImageData] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const loadAttachment = async () => {
      try {
        setLoading(true);
        setError(false);

        const attachmentId = attachment.link.id;
        const contentType = attachment.link.contentType;

        // Determine file extension from content type if not provided
        let ext = attachment.link.ext || "";
        if (!ext) {
          ext = contentType === "image/png" ? ".png" :
               contentType === "image/jpeg" ? ".jpg" :
               contentType === "image/jpg" ? ".jpg" : "";
        }

        // Use the same pattern as web-components Attachment: fetchAttachment function
        const result = await fetchAttachment(attachmentId, ext, contentType);

        if (result?.img) {
          setImageData(result.img);
        } else {
          throw new Error("Failed to load attachment - no image data returned");
        }

      } catch (err) {
        console.error("Failed to load attachment:", err);
        setError(true);
        onError?.();
      } finally {
        setLoading(false);
      }
    };

    loadAttachment();
  }, [attachment.link.id, attachment.link.contentType, onError]);

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center', 
        height: '100%',
        color: '#666',
        fontSize: '12px'
      }}>
        Loading...
      </div>
    );
  }

  if (error || !imageData) {
    return (
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center', 
        height: '100%',
        color: '#999',
        fontSize: '12px',
        fontStyle: 'italic'
      }}>
        Failed to load image
      </div>
    );
  }

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = imageData;
    link.download = attachment.name || 'attachment';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <img 
        src={imageData}
        alt={attachment.name || "Attachment"}
        style={{ 
          maxWidth: '100%', 
          maxHeight: '100%', 
          objectFit: 'contain',
          cursor: 'pointer'
        }}
        onClick={() => window.open(imageData, '_blank')}
        onError={() => {
          setError(true);
          onError?.();
        }}
      />
      <button
        className={styles["image-download-button"]}
        onClick={(e) => {
          e.stopPropagation();
          handleDownload();
        }}
      >
        Download
      </button>
    </div>
  );
};

// Attachment CSV component for Summary
interface AttachmentCSVProps {
  attachment: {
    link: {
      id: string;
      name: string;
      contentType: string;
      ext?: string;
    };
    name?: string;
  };
  onError?: () => void;
}

const AttachmentCSV = ({ attachment, onError }: AttachmentCSVProps) => {
  const [csvData, setCsvData] = useState<string[][] | null>(null);
  const [csvText, setCsvText] = useState<string>(''); // Store raw CSV text for download
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const loadAttachment = async () => {
      try {
        setLoading(true);
        setError(false);

        const attachmentId = attachment.link.id;
        const contentType = attachment.link.contentType;
        const ext = attachment.link.ext || ".csv";

        // Fetch CSV attachment as text
        const result = await fetchAttachment(attachmentId, ext, contentType);

        if (result?.text) {
          // Store raw CSV text for download
          setCsvText(result.text);
          // Parse CSV text into table data
          const parsedData = parseCSV(result.text);
          setCsvData(parsedData);
        } else {
          throw new Error("Failed to load CSV attachment - no text data returned");
        }

      } catch (err) {
        console.error("Failed to load CSV attachment:", err);
        setError(true);
        onError?.();
      } finally {
        setLoading(false);
      }
    };

    loadAttachment();
  }, [attachment.link.id, attachment.link.contentType, onError]);

  // Simple CSV parser
  const parseCSV = (text: string): string[][] => {
    const lines = text.trim().split('\n');
    return lines.map(line => {
      // Simple CSV parsing - handles basic cases
      // For more complex CSV with quoted fields, a proper CSV parser would be needed
      const cells: string[] = [];
      let currentCell = '';
      let insideQuotes = false;
      
      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
          insideQuotes = !insideQuotes;
        } else if (char === ',' && !insideQuotes) {
          cells.push(currentCell.trim());
          currentCell = '';
        } else {
          currentCell += char;
        }
      }
      cells.push(currentCell.trim());
      
      return cells;
    });
  };

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center', 
        padding: '20px',
        color: '#666',
        fontSize: '12px'
      }}>
        Loading CSV data...
      </div>
    );
  }

  if (error || !csvData || csvData.length === 0) {
    return (
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center', 
        padding: '20px',
        color: '#999',
        fontSize: '12px',
        fontStyle: 'italic'
      }}>
        Failed to load CSV data
      </div>
    );
  }

  const headers = csvData[0];
  const rows = csvData.slice(1);

  const handleDownload = () => {
    const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = attachment.name || 'data.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ position: 'relative' }}>
      <div className={styles["csv-title-container"]}>
        <h4 className={styles["csv-title"]}>{attachment.name || attachment.link?.name || 'CSV Data'}</h4>
        <button
          className={styles["download-button"]}
          onClick={handleDownload}
        >
          Download CSV
        </button>
      </div>
      <table className={styles["csv-table"]}>
        <thead>
          <tr>
            {headers.map((header, index) => (
              <th key={index}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => (
                <td key={cellIndex}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// Helper function to transform KPI data for display
const transformKpisData = (kpis: Record<string, any>, metricsData?: any) => {
  const operatorDisplayMap: Record<string, string> = {
    "eq": "==",
    "neq": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "between": "between",
    "contains": "contains",
    "not_contains": "not contains",
    "matches": "matches",
    "in": "in",
    "not_in": "not in"
  };

  const transformedData: Array<{
    name: string;
    value: string;
    reference: string;
    status: string;
    unit: string;
    validationMode: string;
  }> = [];

  // Extract validation mode from the first KPI (should be consistent across all KPIs)
  let validationMode = "all"; // default
  const firstKpiData = Object.values(kpis)[0];
  if (firstKpiData && typeof firstKpiData === 'object') {
    validationMode = firstKpiData.mode || "all";
  }

  Object.entries(kpis).forEach(([key, kpiData]) => {
    if (kpiData && typeof kpiData === 'object') {
      const config = kpiData.config || {};
      const validation = kpiData.validation || {};
      
      const operator = config.validation?.operator || validation.operator || "==";
      const operatorSymbol = operatorDisplayMap[operator] || operator;
      const referenceValue = config.validation?.reference || validation.expected_value || "N/A";
      let actualValue = validation.actual_value !== undefined ? validation.actual_value : "N/A";
      const unit = config.unit || validation.unit || "";
      const passed = validation.passed !== undefined ? validation.passed : false;
      const enabled = config.validation?.enabled !== false; // default to true if not specified
      
      // For skip mode with empty validation, try to get actual value from metrics data
      if (validationMode === "skip" && Object.keys(validation).length === 0 && metricsData?.metrics) {
        const matchingMetric = metricsData.metrics[key];
        if (matchingMetric && typeof matchingMetric === 'object' && matchingMetric.value !== undefined) {
          actualValue = matchingMetric.value;
          // Use unit from metrics if not available in config
          if (!unit && matchingMetric.unit) {
            const metricUnit = matchingMetric.unit;
            const name = config.name || validation.kpi_name || key;
            // Extract reference from config if available, otherwise show "Reference Only"
            const configReference = config.validation?.reference;
            const reference = configReference !== undefined 
              ? `${operatorSymbol} ${configReference}${metricUnit ? ` ${metricUnit}` : ""}`
              : "Reference Only";
            const actualDisplay = `${actualValue}${metricUnit ? ` ${metricUnit}` : ""}`;
            
            transformedData.push({
              name,
              value: actualDisplay,
              reference,
              status: "SKIPPED",
              unit: metricUnit,
              validationMode
            });
            return;
          }
        }
      }
      
      const name = config.name || validation.kpi_name || key;
      // For skip mode, extract reference from config if available
      let reference;
      if (validationMode === "skip") {
        const configReference = config.validation?.reference;
        reference = configReference !== undefined 
          ? `${operatorSymbol} ${configReference}${unit ? ` ${unit}` : ""}`
          : "Reference Only";
      } else {
        reference = `${operatorSymbol} ${referenceValue}${unit ? ` ${unit}` : ""}`;
      }
      const actualDisplay = `${actualValue}${unit ? ` ${unit}` : ""}`;
      
      // Determine status based on validation mode and enabled flag
      let status = "FAILED";
      if (validationMode === "skip") {
        status = "SKIPPED";
      } else if (!enabled) {
        status = "SKIPPED";
      } else if (passed) {
        status = "PASSED";
      }

      transformedData.push({
        name,
        value: actualDisplay,
        reference,
        status,
        unit,
        validationMode
      });
    }
  });

  return { data: transformedData, validationMode };
};

// Helper function to render KPIs section with custom table structure
const renderKpisSection = (
  transformResult: {
    data: Array<{
      name: string;
      value: string;
      reference: string;
      status: string;
      unit: string;
      validationMode: string;
    }>;
    validationMode: string;
  },
  expandedSections: Set<string>,
  toggleSection: (sectionId: string) => void
) => {
  if (transformResult.data.length === 0) {
    return null;
  }

  const { data: kpisData, validationMode } = transformResult;
  const isExpanded = expandedSections.has('kpis');

  return (
    <div className={styles["kpis-section"]}>
      <div 
        className={`${styles["kpis-section-header"]} ${isExpanded ? styles["expanded"] : ""}`}
        onClick={() => toggleSection('kpis')}
      >
        <button
          className={styles["toggle-button-small"]}
          type="button"
          style={{ marginRight: '8px' }}
        >
          {isExpanded ? '−' : '+'}
        </button>
        <h4 className={styles["metadata-section-title"]}>KPIs</h4>
        <span className={styles["kpis-validation-mode"]}>
          {validationMode}
        </span>
      </div>
      {isExpanded && (
        <>
          <div className={styles["kpis-description"]}>
            {validationMode === 'skip' 
              ? 'All KPI validations are disabled and provided for reference purposes only' 
              : validationMode === 'any' 
                ? 'At least one KPI must pass for overall pass status' 
                : 'All KPIs must pass for overall pass status'}
          </div>
          <table className={styles["kpis-table"]}>
            <thead>
              <tr>
                <th className={`${styles["kpis-table-header"]} ${styles["col-30"]}`}>
                  KPI Name
                </th>
                <th className={`${styles["kpis-table-header"]} ${styles["col-30"]}`}>
                  Reference
                </th>
                <th className={`${styles["kpis-table-header"]} ${styles["col-20"]}`}>
                  Value
                </th>
              </tr>
            </thead>
            <tbody>
              {kpisData.map((kpi, idx) => (
                <tr key={idx}>
                  <td className={`${styles["kpis-table-cell"]} ${styles["name-cell"]}`}>
                    {kpi.name}
                  </td>
                  <td className={`${styles["kpis-table-cell"]} ${styles["value-cell"]}`}>
                    {kpi.reference}
                  </td>
                  <td className={`${styles["kpis-table-cell"]} ${styles["value-cell"]}`}>
                    {kpi.value}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
};

/** Small show-more/show-less component for potentially long descriptions. */
const DescriptionBlock = ({ text }: { text: string }) => {
  const [expanded, setExpanded] = useState(false);
  const LIMIT = 220;
  const isTruncatable = text.length > LIMIT;
  const truncated = !expanded && isTruncatable;
  return (
    <div className={styles["test-description"]}>
      <span className={styles["test-description-text"]}>
        {truncated ? text.slice(0, LIMIT).trimEnd() : text}
        {isTruncatable && (
          <button
            type="button"
            className={styles["test-description-toggle"]}
            onClick={(e: MouseEvent) => { e.stopPropagation(); setExpanded(v => !v); }}
          >
            {expanded ? '\u00a0Show less' : '\u2026\u00a0Show more'}
          </button>
        )}
      </span>
    </div>
  );
};

const MetadataTable = ({ metricsData }: MetadataTableProps) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set()); // Collapsed by default
  
  if (!metricsData) {
    return null;
  }

  const toggleSection = (sectionName: string) => {
    setExpandedSections((prev: Set<string>) => {
      const newSet = new Set(prev);
      if (newSet.has(sectionName)) {
        newSet.delete(sectionName);
      } else {
        newSet.add(sectionName);
      }
      return newSet;
    });
  };

  // Helper function to render value as string
  const renderValue = (value: any): string => {
    if (value === null || value === undefined) {
      return 'N/A';
    }
    if (typeof value === 'object') {
      if (Array.isArray(value)) {
        return JSON.stringify(value);
      }
      return JSON.stringify(value, null, 2);
    }
    return String(value);
  };

  // Helper function to render an individual section
  const renderIndividualSection = (title: string, data: Record<string, any>, sectionId: string) => {
    const entries = Object.entries(data);
    if (entries.length === 0) {
      return null;
    }

    const isExpanded = expandedSections.has(sectionId);

    return (
      <div className={styles["metadata-section"]}>
        <div 
          className={`${styles["metadata-section-header"]} ${isExpanded ? styles["expanded"] : ""}`}
          onClick={() => toggleSection(sectionId)}
        >
          <button 
            className={styles["toggle-button-small"]}
            type="button"
            style={{ marginRight: '8px' }}
          >
            {isExpanded ? "−" : "+"}
          </button>
          <h4 className={styles["metadata-section-title"]}>{title}</h4>
        </div>
        {isExpanded && (
          <table className={styles["metadata-table"]}>
            <thead>
              <tr>
                <th className={`${styles["metadata-table-header"]} ${styles["property-column"]}`}>
                  Property
                </th>
                <th className={styles["metadata-table-header"]}>
                  Value
                </th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([key, value], idx) => (
                <tr key={idx}>
                  <td className={`${styles["metadata-table-cell"]} ${styles["property-cell"]}`}>
                    {key}
                  </td>
                  <td className={`${styles["metadata-table-cell"]} ${styles["value-cell"]}`}>
                    {renderValue(value).includes('\n') ? (
                      <pre className={styles["metadata-table-pre"]}>
                        {renderValue(value)}
                      </pre>
                    ) : (
                      <span className={styles["metadata-table-span"]}>{renderValue(value)}</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    );
  };

  // Prepare individual sections
  const sections = [];

  // // Parameters section - excluded
  // if (metricsData.parameters && Object.keys(metricsData.parameters).length > 0) {
  //   sections.push(renderIndividualSection('Parameters', metricsData.parameters));
  // }

  // Metrics section
  if (metricsData.metrics && Object.keys(metricsData.metrics).length > 0) {
    // Transform metrics to display value and unit properly
    const metricsForDisplay: Record<string, any> = {};
    Object.entries(metricsData.metrics).forEach(([key, metricData]: [string, any]) => {
      if (metricData && typeof metricData === 'object' && metricData.value !== undefined) {
        metricsForDisplay[key] = metricData.unit 
          ? `${metricData.value} ${metricData.unit}`
          : metricData.value;
      } else {
        metricsForDisplay[key] = metricData;
      }
    });
    sections.push(renderIndividualSection('Metrics', metricsForDisplay, 'metrics'));
  }

  // Metadata section
  if (metricsData.metadata && Object.keys(metricsData.metadata).length > 0) {
    sections.push(renderIndividualSection('Metadata', metricsData.metadata, 'metadata'));
  }

  // KPIs section - transform to display readable format
  if (metricsData.kpis && Object.keys(metricsData.kpis).length > 0) {
    const transformResult = transformKpisData(metricsData.kpis, metricsData);
    sections.push(renderKpisSection(transformResult, expandedSections, toggleSection));
  }

  if (sections.length === 0) {
    return null;
  }

  // Return all sections as individual components
  return (
    <>
      {sections}
    </>
  );
};

const SystemInfoTable = ({ title, data, expandedPackageDetails, onTogglePackageDetails }: SystemInfoTableProps) => {
  return (
    <div className={styles["table-wrapper"]}>
      <h3>{title}</h3>
      <table className={styles["custom-table"]}>
        <thead>
          <tr>
            <th className={styles["system-info-header"]}>Component</th>
            <th className={styles["system-info-header"]}>Details</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr key={idx}>
              <td className={styles["system-info-component"]}>{row.component}</td>
              <td className={styles["system-info-details"]}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  {/* Expandable package button positioned on the left */}
                  {row.expandable && row.packageData && onTogglePackageDetails && (
                    <button
                      className={styles["expansion-toggle-button"]}
                      onClick={() => onTogglePackageDetails(row.component)}
                      type="button"
                      style={{ 
                        // backgroundColor: 'var(--bg-base-secondary)',
                        // color: 'var(--text-primary)',
                        border: '0px solid var(--on-border-secondary)'
                      }}
                    >
                      {expandedPackageDetails?.has(row.component) ? '−' : '+'}
                    </button>
                  )}
                  
                  {/* Details content */}
                  <div style={{ flex: 1 }}>
                    {row.details.map((detail, detailIdx) => (
                      <div
                        key={detailIdx}
                        className={detail.startsWith('\t')
                          ? styles["system-info-sub-detail-line"]
                          : styles["system-info-detail-line"]}
                      >
                        {detail.startsWith('\t') ? detail.slice(1) : detail}
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Expanded package list */}
                {row.expandable && row.packageData && expandedPackageDetails?.has(row.component) && (
                  <div style={{ 
                    marginTop: '12px', 
                    backgroundColor: 'white', 
                    borderRadius: '4px'
                  }}>
                    <table style={{
                      width: '100%',
                      borderCollapse: 'collapse',
                      fontSize: '12px',
                      border: '1px solid #e0e0e0'
                    }}>
                      <thead>
                        <tr>
                          <th style={{ 
                            backgroundColor: '#f8f9fa', 
                            color: '#333', 
                            textAlign: 'left', 
                            fontWeight: '500', 
                            padding: '8px 12px', 
                            border: '1px solid #e0e0e0',
                            width: '60%'
                          }}>
                            Package
                          </th>
                          <th style={{ 
                            backgroundColor: '#f8f9fa', 
                            color: '#333', 
                            textAlign: 'left', 
                            fontWeight: '500', 
                            padding: '8px 12px', 
                            border: '1px solid #e0e0e0'
                          }}>
                            Version
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(row.packageData)
                          .sort(([a], [b]) => a.toLowerCase().localeCompare(b.toLowerCase()))
                          .map(([name, version]) => (
                            <tr key={name}>
                              <td style={{
                                padding: '8px 12px',
                                border: '1px solid #e0e0e0',
                                backgroundColor: 'white',
                                fontWeight: '500',
                                color: '#555',
                                verticalAlign: 'top',
                                fontFamily: 'monospace',
                                wordBreak: 'break-word',
                                fontSize: '12px'
                              }}>
                                {name}
                              </td>
                              <td style={{
                                padding: '8px 12px',
                                border: '1px solid #e0e0e0',
                                backgroundColor: 'white',
                                color: '#333',
                                verticalAlign: 'top',
                                fontFamily: 'monospace',
                                fontSize: '11px'
                              }}>
                                {version || 'unknown'}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const SortableTable = ({ 
  title, 
  data, 
  headers, 
  onRowClick, 
  expandedRows = [], 
  renderExpandedContent,
  getRowId
}: SortableTableProps) => {
  const [sortConfig, setSortConfig] = useState<{ column: number; direction: "asc" | "desc" } | null>(null);
  const [sortedData, setSortedData] = useState(data);
  const [originalToSortedMapping, setOriginalToSortedMapping] = useState<number[]>([]);

  // Update sorted data when original data changes
  useEffect(() => {
    setSortedData(data);
    setSortConfig(null);
    // Initialize mapping to original order
    setOriginalToSortedMapping(data.map((_, index) => index));
  }, [data]);

  // Helper function to extract numeric value from percentage strings
  const parsePercentage = (value: string | number): number => {
    if (typeof value === "number") return value;
    const str = String(value);
    if (str.endsWith("%")) {
      return parseFloat(str.slice(0, -1));
    }
    return parseFloat(str) || 0;
  };

  // Helper function to extract seconds from duration strings
  const parseDuration = (value: string | number): number => {
    if (typeof value === "number") return value;
    const str = String(value);
    
    // Handle formats like "2m 30.5s", "45.2s", "1h 30m 45.2s"
    let totalSeconds = 0;
    
    // Extract hours (must be followed by 'h')
    const hoursMatch = str.match(/(\d+(?:\.\d+)?)h/);
    if (hoursMatch) {
      totalSeconds += parseFloat(hoursMatch[1]) * 3600;
    }
    
    // Extract minutes (must be followed by 'm' but not 'ms')
    const minutesMatch = str.match(/(\d+(?:\.\d+)?)m(?!s)/);
    if (minutesMatch) {
      totalSeconds += parseFloat(minutesMatch[1]) * 60;
    }
    
    // Extract seconds (must be followed by 's')
    const secondsMatch = str.match(/(\d+(?:\.\d+)?)s/);
    if (secondsMatch) {
      totalSeconds += parseFloat(secondsMatch[1]);
    }
    
    // If no time units found, try to parse as plain number (assume seconds)
    if (totalSeconds === 0) {
      const plainNumber = parseFloat(str);
      if (!isNaN(plainNumber)) {
        totalSeconds = plainNumber;
      }
    }
    
    return totalSeconds;
  };

  // Helper function to extract timestamp for date sorting
  const parseTimestamp = (value: string | number): number => {
    if (typeof value === "number") return value;
    const str = String(value);
    
    // Remove "LATEST" suffix if present
    const cleanStr = str.replace(" LATEST", "");
    const date = new Date(cleanStr);
    return date.getTime();
  };

  // Helper function to determine column type and get sortable value
  const getSortableValue = (value: string | number, columnIndex: number): number => {
    const header = headers[columnIndex].toLowerCase();
    
    // Handle percentage columns
    if (header.includes("rate") || header.includes("percent")) {
      return parsePercentage(value);
    }
    
    // Handle duration columns
    if (header.includes("duration") || header.includes("longest") || header.includes("current")) {
      return parseDuration(value);
    }
    
    // Handle timestamp columns
    if (header.includes("generated") || header.includes("timestamp")) {
      return parseTimestamp(value);
    }
    
    // Handle numeric values
    if (typeof value === "number") {
      return value;
    }
    
    // Try to parse as number, fallback to 0 for string comparison
    const numValue = parseFloat(String(value));
    return isNaN(numValue) ? 0 : numValue;
  };

  const handleSort = (columnIndex: number) => {
    let direction: "asc" | "desc" = "asc";
    
    if (sortConfig && sortConfig.column === columnIndex && sortConfig.direction === "asc") {
      direction = "desc";
    }

    // Create array of {row, originalIndex} objects for sorting
    const rowsWithOriginalIndex = data.map((row, originalIndex) => ({
      row,
      originalIndex
    }));

    // Sort the array
    const sorted = [...rowsWithOriginalIndex].sort((a, b) => {
      const aVal = a.row[columnIndex];
      const bVal = b.row[columnIndex];
      
      // Handle different data types
      let comparison = 0;
      
      if (typeof aVal === "number" && typeof bVal === "number") {
        comparison = aVal - bVal;
      } else {
        const header = headers[columnIndex].toLowerCase();
        
        // For columns that should be sorted numerically (percentages, durations, timestamps)
        if (header.includes("rate") || header.includes("percent") || 
            header.includes("duration") || header.includes("longest") || 
            header.includes("current") || header.includes("generated") || 
            header.includes("timestamp")) {
          
          const aSortValue = getSortableValue(aVal, columnIndex);
          const bSortValue = getSortableValue(bVal, columnIndex);
          comparison = aSortValue - bSortValue;
        } else {
          // Try numeric comparison first
          const aNum = parseFloat(String(aVal));
          const bNum = parseFloat(String(bVal));
          
          if (!isNaN(aNum) && !isNaN(bNum)) {
            comparison = aNum - bNum;
          } else {
            // Fall back to string comparison
            const aStr = String(aVal).toLowerCase();
            const bStr = String(bVal).toLowerCase();
            comparison = aStr.localeCompare(bStr);
          }
        }
      }
      
      return direction === "asc" ? comparison : -comparison;
    });

    // Extract sorted data and mapping
    const sortedDataOnly = sorted.map(item => item.row);
    const newMapping = sorted.map(item => item.originalIndex);

    setSortedData(sortedDataOnly);
    setOriginalToSortedMapping(newMapping);
    setSortConfig({ column: columnIndex, direction });
  };

  const getSortIcon = (columnIndex: number) => {
    if (!sortConfig || sortConfig.column !== columnIndex) {
      return " ⇅";
    }
    return sortConfig.direction === "asc" ? " ↑" : " ↓";
  };

  const getStatusClassName = (status: string) => {
    const statusLower = String(status).toLowerCase();
    switch (statusLower) {
      case "passed":
        return styles["status-passed"];
      case "failed":
        return styles["status-failed"];
      case "broken":
        return styles["status-broken"];
      case "skipped":
        return styles["status-skipped"];
      default:
        return styles["status-unknown"];
    }
  };

  const getStatusCountClassName = (header: string) => {
    const headerLower = header.toLowerCase();
    switch (headerLower) {
      case "passed":
        return styles["status-passed"];
      case "failed":
        return styles["status-failed"];
      case "broken":
        return styles["status-broken"];
      case "skipped":
        return styles["status-skipped"];
      case "unknown":
        return styles["status-unknown"];
      default:
        return null;
    }
  };

  const isStatusColumn = (header: string) => {
    return header.toLowerCase() === "status";
  };

  const isStatusCountColumn = (header: string) => {
    const headerLower = header.toLowerCase();
    return ["passed", "failed", "broken", "skipped", "unknown"].includes(headerLower);
  };

  const isLatestColumn = (header: string) => {
    return header.toLowerCase() === "latest";
  };

  const isGeneratedColumn = (header: string) => {
    return header.toLowerCase() === "generated";
  };

  const renderGeneratedCell = (cell: string | number) => {
    const cellStr = String(cell);
    if (cellStr.includes(" LATEST")) {
      const timestamp = cellStr.replace(" LATEST", "");
      return (
        <span>
          {timestamp}{" "}
          <span className={styles["status-latest"]}>LATEST</span>
        </span>
      );
    }
    return cellStr;
  };

  return (
    <div className={styles["table-wrapper"]}>
      <h3>{title}</h3>
      <table className={styles["custom-table"]}>
        <thead>
          <tr>
            {onRowClick && (
              <th className={styles["expansion-header"]}>
                {/* No label for the expansion column */}
              </th>
            )}
            {headers.map((header, index) => (
              <th
                key={header}
                className={styles["sortable-header"]}
                onClick={() => handleSort(index)}
                title={`Click to sort by ${header}`}
              >
                {header}{getSortIcon(index)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.flatMap((row, idx) => {
            // Get the original index for this sorted row
            const originalIdx = originalToSortedMapping[idx];

            const mainRow = (
              <tr 
                key={`row-${originalIdx}`}
                className={onRowClick ? styles["clickable-row"] : ""}
                onClick={() => onRowClick?.(originalIdx)}
                style={{ cursor: onRowClick ? "pointer" : "default" }}
              >
                {onRowClick && (
                  <td className={styles["expansion-indicator"]}>
                    <span className={styles["expansion-toggle-button"]}>
                      {expandedRows?.[originalIdx] ? "−" : "+"}
                    </span>
                  </td>
                )}
                {row.map((cell, cidx) => (
                  <td key={cidx}>
                    {isStatusColumn(headers[cidx]) ? (
                      <span className={getStatusClassName(String(cell))}>
                        {String(cell)}
                      </span>
                    ) : isStatusCountColumn(headers[cidx]) ? (
                      <span className={Number(cell) > 0 ? getStatusCountClassName(headers[cidx]) : styles["status-zero"]}>
                        {typeof cell === "number" ? cell.toLocaleString() : cell}
                      </span>
                    ) : isGeneratedColumn(headers[cidx]) ? (
                      renderGeneratedCell(cell)
                    ) : isLatestColumn(headers[cidx]) && String(cell) === "LATEST" ? (
                      <span className={styles["status-latest"]}>
                        {String(cell)}
                      </span>
                    ) : (
                      typeof cell === "number" ? cell.toLocaleString() : cell
                    )}
                  </td>
                ))}
              </tr>
            );

            const expandedRow = expandedRows?.[originalIdx] && renderExpandedContent ? (
              <tr key={`expanded-${originalIdx}`} className={styles["details-row"]}>
                <td colSpan={onRowClick ? headers.length + 1 : headers.length} className={styles["details-cell"]}>
                  {renderExpandedContent(originalIdx)}
                </td>
              </tr>
            ) : null;

            return expandedRow ? [mainRow, expandedRow] : [mainRow];
          })}
        </tbody>
      </table>
    </div>
  );
};

export const Summary = () => {
  const [expandedProfiles, setExpandedProfiles] = useState<Set<string>>(new Set());
  const [allExpanded, setAllExpanded] = useState(false);
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [expandedSystemSections, setExpandedSystemSections] = useState<Set<string>>(new Set(["hardware", "software"])); // Expanded by default
  const [expandedTestDetails, setExpandedTestDetails] = useState<Set<string>>(new Set()); // For individual test details
  const [expandedPackageDetails, setExpandedPackageDetails] = useState<Set<string>>(new Set()); // For package lists
  const [individualTestResults, setIndividualTestResults] = useState<Record<string, any>>({});
  const [loadingTests, setLoadingTests] = useState(true);
  const [testMetrics, setTestMetrics] = useState<Record<string, any>>({}); // Store metrics data from attachments
  const [testFullMetrics, setTestFullMetrics] = useState<Record<string, any>>({}); // Store full metrics data for metadata tables
  const [testReferences, setTestReferences] = useState<Record<string, string>>({}); // Store reference data from KPIs
  
  // State for qualifications profiles
  const [expandedQualificationProfiles, setExpandedQualificationProfiles] = useState<Set<string>>(new Set());
  const [qualificationSearchTerm, setQualificationSearchTerm] = useState<string>("");

  // State for vertical profiles
  const [expandedVerticalProfiles, setExpandedVerticalProfiles] = useState<Set<string>>(new Set());
  const [verticalSearchTerm, setVerticalSearchTerm] = useState<string>("");

  // Test descriptions keyed by Allure result UUID, loaded from test_summary.json
  const [testDescriptions, setTestDescriptions] = useState<Record<string, string>>({});

  useEffect(() => {
    // Fetch navigation data and individual test results
    const fetchAllTestData = async () => {
      try {
        setLoadingTests(true);
        
        // First fetch the navigation data to get all test IDs
        await fetchTestResultNav();
        const navData = testResultNavStore.value.data;
        
        if (navData && Array.isArray(navData)) {
          // Fetch each individual test result
          const fetchPromises = navData.map(testId => fetchTestResult(testId));
          await Promise.all(fetchPromises);
          
          // Now get all the fetched data from the store
          const allFetchedResults = testResultStore.value.data || {};
          setIndividualTestResults(allFetchedResults);
        }
      } catch (error) {
        console.error("Error fetching test data:", error);
      } finally {
        setLoadingTests(false);
      }
    };
    
    fetchAllTestData();
    
    // Auto-add summary section to available sections when summary data exists
    if (!availableSections.value.includes("summary")) {
      availableSections.value = [...availableSections.value, "summary"];
    }
  }, []);

  // Fetch metrics from test attachments when individual test results are loaded
  useEffect(() => {
    const fetchAllMetrics = async () => {
      if (loadingTests || Object.keys(individualTestResults).length === 0) {
        return;
      }

      const metricsData: Record<string, any> = {};
      const fullMetricsData: Record<string, any> = {};
      const referencesData: Record<string, string> = {};
      
      // Fetch metrics for each test
      for (const testResult of Object.values(individualTestResults)) {
        const metrics = await getMetricsFromAttachment(testResult);
        const fullMetrics = await getFullMetricsFromAttachment(testResult);
        const reference = await getReferenceFromAttachment(testResult);
        if ((testResult as any).id) {
          metricsData[(testResult as any).id] = metrics;
          fullMetricsData[(testResult as any).id] = fullMetrics;
          referencesData[(testResult as any).id] = reference;
        }
      }
      
      setTestMetrics(metricsData);
      setTestFullMetrics(fullMetricsData);
      setTestReferences(referencesData);
    };

    fetchAllMetrics();
  }, [loadingTests, individualTestResults]);

  const toggleProfile = (profileName: string) => {
    setExpandedProfiles((prev: Set<string>) => {
      const newSet = new Set(prev);
      if (newSet.has(profileName)) {
        newSet.delete(profileName);
      } else {
        newSet.add(profileName);
      }
      return newSet;
    });
  };

  const expandAllProfiles = () => {
    // Get all profile names from the currently rendered profiles (filteredProfilesData)
    const allProfileNames = filteredProfilesData.map(profile => profile.profileName);
    setExpandedProfiles(new Set(allProfileNames));
    setAllExpanded(true);
  };

  const collapseAllProfiles = () => {
    setExpandedProfiles(new Set());
    setAllExpanded(false);
  };

  // Qualification profile management functions
  const toggleQualificationProfile = (profileName: string) => {
    setExpandedQualificationProfiles((prev: Set<string>) => {
      const newSet = new Set(prev);
      if (newSet.has(profileName)) {
        newSet.delete(profileName);
      } else {
        newSet.add(profileName);
      }
      return newSet;
    });
  };

  const expandAllQualificationProfiles = () => {
    // We need to reference filteredQualificationProfilesData but it's defined later
    // So we'll get all qualification profile names from the base data
    const allQualificationProfileNames = qualificationProfilesData.map((profile: any) => profile.profileName);
    setExpandedQualificationProfiles(new Set(allQualificationProfileNames));
  };

  const collapseAllQualificationProfiles = () => {
    setExpandedQualificationProfiles(new Set());
  };

  const clearQualificationSearch = () => {
    setQualificationSearchTerm("");
  };

  const handleQualificationSearchChange = (event: Event) => {
    const target = event.target as HTMLInputElement;
    setQualificationSearchTerm(target.value);
  };

  // Vertical profile management functions
  const toggleVerticalProfile = (profileName: string) => {
    setExpandedVerticalProfiles((prev: Set<string>) => {
      const newSet = new Set(prev);
      if (newSet.has(profileName)) {
        newSet.delete(profileName);
      } else {
        newSet.add(profileName);
      }
      return newSet;
    });
  };

  const expandAllVerticalProfiles = () => {
    const allVerticalProfileNames = verticalProfilesData.map((profile: any) => profile.profileName);
    setExpandedVerticalProfiles(new Set(allVerticalProfileNames));
  };

  const collapseAllVerticalProfiles = () => {
    setExpandedVerticalProfiles(new Set());
  };

  const clearVerticalSearch = () => {
    setVerticalSearchTerm("");
  };

  const handleVerticalSearchChange = (event: Event) => {
    const target = event.target as HTMLInputElement;
    setVerticalSearchTerm(target.value);
  };

  const clearSearch = () => {
    setSearchTerm("");
  };

  const handleSearchChange = (event: Event) => {
    const target = event.target as HTMLInputElement;
    setSearchTerm(target.value);
  };

  const toggleSystemSection = (sectionName: string) => {
    setExpandedSystemSections((prev: Set<string>) => {
      const newSet = new Set(prev);
      if (newSet.has(sectionName)) {
        newSet.delete(sectionName);
      } else {
        newSet.add(sectionName);
      }
      return newSet;
    });
  };

  const expandAllSystemSections = () => {
    setExpandedSystemSections(new Set(["hardware", "software"]));
  };

  const collapseAllSystemSections = () => {
    setExpandedSystemSections(new Set());
  };

  const toggleTestDetails = (testId: string) => {
    setExpandedTestDetails(prev => {
      const newSet = new Set(prev);
      if (newSet.has(testId)) {
        newSet.delete(testId);
      } else {
        newSet.add(testId);
      }
      return newSet;
    });
  };

  const togglePackageDetails = (packageType: string) => {
    setExpandedPackageDetails(prev => {
      const newSet = new Set(prev);
      if (newSet.has(packageType)) {
        newSet.delete(packageType);
      } else {
        newSet.add(packageType);
      }
      return newSet;
    });
  };

  // Helper functions
  const formatDuration = (seconds: number): string => {
    if (seconds < 60) {
      return `${seconds.toFixed(1)}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(1)}s`;
  };

  /** Returns the description string for a given testId, or null if absent. */
  const getTestDescription = (testId: string): string | null => {
    const testName = individualTestResults[testId]?.name;
    return (testName && testDescriptions[testName]) || null;
  };

  // Create profile table data for test results (type: "suite")
  const createTestResultProfilesFromTestResults = (groupBy: "group" | "parentSuite" = "group") => {
    if (loadingTests || Object.keys(individualTestResults).length === 0) {
      return [];
    }

    // Group tests by specified grouping method for suite type only
    const profileGroups: Record<string, any[]> = {};
    
    Object.values(individualTestResults).forEach((testResult: any) => {
      const parentSuite = testResult.labels?.find((label: any) => label.name === "parentSuite")?.value;
      const groupLabel = testResult.labels?.find((label: any) => label.name === "group")?.value;
      const typeLabel = testResult.labels?.find((label: any) => label.name === "type")?.value;
      
      // Include only profiles that have "type" label with value "suite" and exclude "core" type
      if (typeLabel === "suite") {
        // Determine grouping key based on groupBy parameter
        let groupingKey: string;
        if (groupBy === "group") {
          groupingKey = groupLabel || "unknown group";
        } else {
          groupingKey = parentSuite || "unknown suite";
        }
        
        if (groupingKey && (groupBy === "group" || parentSuite)) { // For parentSuite grouping, ensure parentSuite exists
          if (!profileGroups[groupingKey]) {
            profileGroups[groupingKey] = [];
          }
          profileGroups[groupingKey].push(testResult);
        }
      }
    });

    // Create profile table data
    return Object.entries(profileGroups).map(([profileName, tests]) => {
      // Get profile_display_name from the first test in the group
      const firstTest = tests[0];
      const profileDisplayName = firstTest?.labels?.find((label: any) => label.name === "profile_display_name")?.value;
      const displayStatus = firstTest?.labels?.find((label: any) => label.name === "display_status")?.value;
      const displayReference = firstTest?.labels?.find((label: any) => label.name === "display_reference")?.value;
      const displayId = firstTest?.labels?.find((label: any) => label.name === "display_id")?.value;
      
      // Find tier_display_name from any test in the group (not just the first one)
      let tierDisplayName: string | undefined;
      for (const test of tests) {
        const tierLabel = test.labels?.find((label: any) => label.name === "tier_display_name")?.value;
        if (tierLabel) {
          tierDisplayName = tierLabel;
          break;
        }
      }
      
      // Check if status column should be hidden
      const hideStatusColumn = displayStatus === "False" || displayStatus === "false";
      
      // Check if reference column should be displayed (must be explicitly set to "True" or "true")
      const showReferenceColumn = displayReference === "True" || displayReference === "true";
      
      // Check if ID column should be hidden (default is hidden unless explicitly set to "True" or "true")
      const hideIdColumn = displayId !== "True" && displayId !== "true";

      const testsData = tests.map((test: any) => {
        // Get metrics from cached data
        const cachedMetrics = testMetrics[test.id] || { metric: "N/A", value: "N/A", unit: "N/A" };
        const cachedReference = testReferences[test.id] || "N/A";
        
        // Calculate history and retries data
        const historyCount = test.history?.length || 0;
        const retriesCount = test.retries?.length || 0;
        const totalDuration = test.duration || 0;

        // Get test title from test_title label, fallback to test name
        const testTitle = test.labels?.find((label: any) => label.name === "test_title")?.value;

        return {
          id: test.id.substring(0, 8), // Display ID (shortened)
          fullId: test.id, // Full ID for accessing individual test results
          testName: testTitle || test.name, // Use test_title label if available, otherwise use test name
          metric: cachedMetrics.metric,
          value: cachedMetrics.value,
          unit: cachedMetrics.unit,
          reference: cachedReference,
          status: test.status,
          // Additional data for expanded content
          historyCount,
          retriesCount,
          duration: formatDuration(totalDuration / 1000), // Convert from ms to seconds
          testResult: test // Store full test result for detailed view
        };
      });

      // Get the type from the first test in the group to identify profile type
      const firstTestForType = tests[0];
      const profileType = firstTestForType?.labels?.find((label: any) => label.name === "type")?.value;

      return {
        profileName: profileDisplayName || profileName.replace(/^profile\.(suite|qualification)\./, ""), // Use profile_display_name or fallback to processed profileName
        testsData,
        hideStatusColumn, // Pass this flag to the table rendering
        showReferenceColumn, // Pass flag for showing reference column
        hideIdColumn, // Pass flag for hiding ID column
        profileType, // Add profile type for conditional rendering
        tierDisplayName // Add tier display name for badge rendering
      };
    });
  };

  // Create profile table data for qualifications (type: "qualification")
  const createQualificationProfilesFromTestResults = (groupBy: "group" | "parentSuite" = "group") => {
    if (loadingTests || Object.keys(individualTestResults).length === 0) {
      return [];
    }

    // Group tests by specified grouping method for qualification type only
    const profileGroups: Record<string, any[]> = {};
    
    Object.values(individualTestResults).forEach((testResult: any) => {
      const parentSuite = testResult.labels?.find((label: any) => label.name === "parentSuite")?.value;
      const groupLabel = testResult.labels?.find((label: any) => label.name === "group")?.value;
      const typeLabel = testResult.labels?.find((label: any) => label.name === "type")?.value;
      
      // Include only profiles that have "type" label with value "qualification"
      if (typeLabel === "qualification") {
        // Determine grouping key based on groupBy parameter
        let groupingKey: string;
        if (groupBy === "group") {
          groupingKey = groupLabel || "unknown group";
        } else {
          groupingKey = parentSuite || "unknown suite";
        }
        
        if (groupingKey && (groupBy === "group" || parentSuite)) { // For parentSuite grouping, ensure parentSuite exists
          if (!profileGroups[groupingKey]) {
            profileGroups[groupingKey] = [];
          }
          profileGroups[groupingKey].push(testResult);
        }
      }
    });

    // Create profile table data similar to test results but for qualifications
    const profilesArray = Object.entries(profileGroups).map(([profileName, tests]) => {
      // Get profile_display_name from the first test in the group
      const firstTest = tests[0];
      const profileDisplayName = firstTest?.labels?.find((label: any) => label.name === "profile_display_name")?.value;
      const displayStatus = firstTest?.labels?.find((label: any) => label.name === "display_status")?.value;
      const displayReference = firstTest?.labels?.find((label: any) => label.name === "display_reference")?.value;
      const displayId = firstTest?.labels?.find((label: any) => label.name === "display_id")?.value;
      
      // Find tier_display_name from any test in the group (not just the first one)
      let tierDisplayName: string | undefined;
      for (const test of tests) {
        const tierLabel = test.labels?.find((label: any) => label.name === "tier_display_name")?.value;
        if (tierLabel) {
          tierDisplayName = tierLabel;
          break;
        }
      }
      
      // Check if status column should be hidden
      const hideStatusColumn = displayStatus === "False" || displayStatus === "false";
      
      // Check if reference column should be displayed (must be explicitly set to "True" or "true")
      const showReferenceColumn = displayReference === "True" || displayReference === "true";
      
      // Check if ID column should be hidden (default is hidden unless explicitly set to "True" or "true")
      const hideIdColumn = displayId !== "True" && displayId !== "true";

      // Calculate overall status and test count for the profile
      const totalTests = tests.length;
      const passedTests = tests.filter(test => test.status === "passed").length;
      const failedTests = tests.filter(test => test.status === "failed").length;
      const brokenTests = tests.filter(test => test.status === "broken").length;
      const skippedTests = tests.filter(test => test.status === "skipped").length;
      const unknownTests = tests.filter(test => test.status === "unknown").length;
      
      // Determine overall status - only passed/failed for qualifications
      // Any non-passed test (failed, broken, skipped, unknown) results in "failed" status
      let overallStatus = "passed";
      if (failedTests > 0 || brokenTests > 0 || skippedTests > 0 || unknownTests > 0) {
        overallStatus = "failed";
      }

      const testsData = tests.map((test: any) => {
        // Get metrics from cached data
        const cachedMetrics = testMetrics[test.id] || { metric: "N/A", value: "N/A", unit: "N/A" };
        const cachedReference = testReferences[test.id] || "N/A";
        
        // Calculate history and retries data
        const historyCount = test.history?.length || 0;
        const retriesCount = test.retries?.length || 0;
        const totalDuration = test.duration || 0;

        // Get test title from test_title label, fallback to test name
        const testTitle = test.labels?.find((label: any) => label.name === "test_title")?.value;

        return {
          id: test.id.substring(0, 8), // Display ID (shortened)
          fullId: test.id, // Full ID for accessing individual test results
          testName: testTitle || test.name, // Use test_title label if available, otherwise use test name
          metric: cachedMetrics.metric,
          value: cachedMetrics.value,
          unit: cachedMetrics.unit,
          reference: cachedReference,
          status: test.status,
          // Additional data for expanded content
          historyCount,
          retriesCount,
          duration: formatDuration(totalDuration / 1000), // Convert from ms to seconds
          testResult: test // Store full test result for detailed view
        };
      });

      // Get the type from the first test in the group to identify profile type
      const firstTestForType = tests[0];
      const profileType = firstTestForType?.labels?.find((label: any) => label.name === "type")?.value;

      return {
        profileName: profileDisplayName || profileName.replace(/^profile\.(suite|qualification)\./, ""), // Use profile_display_name or fallback to processed profileName
        testsData,
        hideStatusColumn, // Pass this flag to the table rendering
        showReferenceColumn, // Pass flag for showing reference column
        hideIdColumn, // Pass flag for hiding ID column
        profileType, // Add profile type for conditional rendering
        tierDisplayName, // Add tier display name for badge rendering
        // Additional data for qualification profiles
        totalTests,
        passedTests,
        failedTests,
        brokenTests,
        skippedTests,
        unknownTests,
        overallStatus
      };
    });

    // Sort profiles: first by type (qualification type label), then by profile name alphabetically
    return profilesArray.sort((a, b) => {
      // Sort by type first (qualification < vertical, or other ordering if needed)
      const typeComparison = (a.profileType || '').localeCompare(b.profileType || '');
      if (typeComparison !== 0) {
        return typeComparison;
      }
      
      // Then sort by profile name alphabetically
      return a.profileName.localeCompare(b.profileName);
    });
  };

  const individualProfilesData = createTestResultProfilesFromTestResults();
  const qualificationProfilesData = createQualificationProfilesFromTestResults();

  // Create profile table data for vertical profiles (type: "vertical")
  const createVerticalProfilesFromTestResults = (groupBy: "group" | "parentSuite" = "group") => {
    if (loadingTests || Object.keys(individualTestResults).length === 0) {
      return [];
    }

    // Group tests by specified grouping method for vertical type only
    const profileGroups: Record<string, any[]> = {};
    
    Object.values(individualTestResults).forEach((testResult: any) => {
      const parentSuite = testResult.labels?.find((label: any) => label.name === "parentSuite")?.value;
      const groupLabel = testResult.labels?.find((label: any) => label.name === "group")?.value;
      const typeLabel = testResult.labels?.find((label: any) => label.name === "type")?.value;
      
      // Include only profiles that have "type" label with value "vertical"
      if (typeLabel === "vertical") {
        // Determine grouping key based on groupBy parameter
        let groupingKey: string;
        if (groupBy === "group") {
          groupingKey = groupLabel || "unknown group";
        } else {
          groupingKey = parentSuite || "unknown suite";
        }
        
        if (groupingKey && (groupBy === "group" || parentSuite)) { // For parentSuite grouping, ensure parentSuite exists
          if (!profileGroups[groupingKey]) {
            profileGroups[groupingKey] = [];
          }
          profileGroups[groupingKey].push(testResult);
        }
      }
    });

    // Create profile table data similar to qualifications
    const profilesArray = Object.entries(profileGroups).map(([profileName, tests]) => {
      // Get profile_display_name from the first test in the group
      const firstTest = tests[0];
      const profileDisplayName = firstTest?.labels?.find((label: any) => label.name === "profile_display_name")?.value;
      const displayStatus = firstTest?.labels?.find((label: any) => label.name === "display_status")?.value;
      const displayReference = firstTest?.labels?.find((label: any) => label.name === "display_reference")?.value;
      const displayId = firstTest?.labels?.find((label: any) => label.name === "display_id")?.value;
      
      // Find tier_display_name from any test in the group (not just the first one)
      let tierDisplayName: string | undefined;
      for (const test of tests) {
        const tierLabel = test.labels?.find((label: any) => label.name === "tier_display_name")?.value;
        if (tierLabel) {
          tierDisplayName = tierLabel;
          break;
        }
      }
      
      // Check if status column should be hidden
      const hideStatusColumn = displayStatus === "False" || displayStatus === "false";
      
      // Check if reference column should be displayed (must be explicitly set to "True" or "true")
      const showReferenceColumn = displayReference === "True" || displayReference === "true";
      
      // Check if ID column should be hidden (default is hidden unless explicitly set to "True" or "true")
      const hideIdColumn = displayId !== "True" && displayId !== "true";

      // Calculate overall status and test count for the profile
      const totalTests = tests.length;
      const passedTests = tests.filter(test => test.status === "passed").length;
      const failedTests = tests.filter(test => test.status === "failed").length;
      const brokenTests = tests.filter(test => test.status === "broken").length;
      const skippedTests = tests.filter(test => test.status === "skipped").length;
      const unknownTests = tests.filter(test => test.status === "unknown").length;
      
      // Determine overall status
      let overallStatus = "passed";
      if (failedTests > 0 || brokenTests > 0 || skippedTests > 0 || unknownTests > 0) {
        overallStatus = "failed";
      }

      const testsData = tests.map((test: any) => {
        // Get metrics from cached data
        const cachedMetrics = testMetrics[test.id] || { metric: "N/A", value: "N/A", unit: "N/A" };
        const cachedReference = testReferences[test.id] || "N/A";
        
        // Calculate history and retries data
        const historyCount = test.history?.length || 0;
        const retriesCount = test.retries?.length || 0;
        const totalDuration = test.duration || 0;

        // Get test title from test_title label, fallback to test name
        const testTitle = test.labels?.find((label: any) => label.name === "test_title")?.value;

        return {
          id: test.id.substring(0, 8), // Display ID (shortened)
          fullId: test.id, // Full ID for accessing individual test results
          testName: testTitle || test.name, // Use test_title label if available, otherwise use test name
          metric: cachedMetrics.metric,
          value: cachedMetrics.value,
          unit: cachedMetrics.unit,
          reference: cachedReference,
          status: test.status,
          // Additional data for expanded content
          historyCount,
          retriesCount,
          duration: formatDuration(totalDuration / 1000), // Convert from ms to seconds
          testResult: test // Store full test result for detailed view
        };
      });

      // Get the type from the first test in the group to identify profile type
      const firstTestForType = tests[0];
      const profileType = firstTestForType?.labels?.find((label: any) => label.name === "type")?.value;

      return {
        profileName: profileDisplayName || profileName.replace(/^profile\.(suite|qualification|vertical)\./, ""), // Use profile_display_name or fallback to processed profileName
        testsData,
        hideStatusColumn, // Pass this flag to the table rendering
        showReferenceColumn, // Pass flag for showing reference column
        hideIdColumn, // Pass flag for hiding ID column
        profileType, // Add profile type for conditional rendering
        tierDisplayName, // Add tier display name for badge rendering
        // Additional data for vertical profiles
        totalTests,
        passedTests,
        failedTests,
        brokenTests,
        skippedTests,
        unknownTests,
        overallStatus
      };
    });

    // Sort profiles by profile name alphabetically
    return profilesArray.sort((a, b) => {
      return a.profileName.localeCompare(b.profileName);
    });
  };

  const verticalProfilesData = createVerticalProfilesFromTestResults();

  // Filter profiles and their tests based on search term
  const filteredProfilesData = individualProfilesData.map(profile => {
    if (!searchTerm.trim()) {
      return profile; // No search term, return all data
    }

    const searchLower = searchTerm.toLowerCase();
    
    // Filter tests within this profile
    const filteredTests = profile.testsData.filter((test: any) => {
      // Check if any field contains the search term
      return [
        test.id,
        test.testName,
        test.metric,
        test.value,
        test.unit,
        test.status
      ].some(field => String(field).toLowerCase().includes(searchLower));
    });

    // Only include profiles that have matching tests or whose name matches
    const profileNameMatches = profile.profileName.toLowerCase().includes(searchLower);
    
    return {
      profileName: profile.profileName,
      testsData: filteredTests,
      hideStatusColumn: profile.hideStatusColumn, // Preserve the hideStatusColumn flag
      showReferenceColumn: profile.showReferenceColumn, // Preserve the showReferenceColumn flag
      hideIdColumn: profile.hideIdColumn, // Preserve the hideIdColumn flag
      profileType: profile.profileType, // Preserve the profileType flag
      tierDisplayName: profile.tierDisplayName, // Preserve the tierDisplayName flag
      hasMatches: filteredTests.length > 0 || profileNameMatches
    };
  }).filter(profile => (profile as any).hasMatches || !searchTerm.trim());

  // Filter qualification profiles and their tests based on search term
  const filteredQualificationProfilesData = qualificationProfilesData.map(profile => {
    if (!qualificationSearchTerm.trim()) {
      return profile; // No search term, return all data
    }

    const searchLower = qualificationSearchTerm.toLowerCase();
    
    // Filter tests within this profile
    const filteredTests = profile.testsData.filter((test: any) => {
      // Check if any field contains the search term
      return [
        test.id,
        test.testName,
        test.metric,
        test.value,
        test.unit,
        test.status
      ].some(field => String(field).toLowerCase().includes(searchLower));
    });

    // Only include profiles that have matching tests or whose name matches
    const profileNameMatches = profile.profileName.toLowerCase().includes(searchLower);
    
    return {
      ...profile, // Include all original properties (totalTests, overallStatus, etc.)
      testsData: filteredTests,
      hasMatches: filteredTests.length > 0 || profileNameMatches
    };
  }).filter(profile => (profile as any).hasMatches || !qualificationSearchTerm.trim());

  // Filter vertical profiles and their tests based on search term
  const filteredVerticalProfilesData = verticalProfilesData.map(profile => {
    if (!verticalSearchTerm.trim()) {
      return profile; // No search term, return all data
    }

    const searchLower = verticalSearchTerm.toLowerCase();
    
    // Filter tests within this profile
    const filteredTests = profile.testsData.filter((test: any) => {
      // Check if any field contains the search term
      return [
        test.id,
        test.testName,
        test.metric,
        test.value,
        test.unit,
        test.status
      ].some(field => String(field).toLowerCase().includes(searchLower));
    });

    // Only include profiles that have matching tests or whose name matches
    const profileNameMatches = profile.profileName.toLowerCase().includes(searchLower);
    
    return {
      ...profile, // Include all original properties (totalTests, overallStatus, etc.)
      testsData: filteredTests,
      hasMatches: filteredTests.length > 0 || profileNameMatches
    };
  }).filter(profile => (profile as any).hasMatches || !verticalSearchTerm.trim());

  // State for system information
  const [systemInfo, setSystemInfo] = useState<any>(null);

  // Fetch system information on component mount
  useEffect(() => {
    const loadSystemInfo = async () => {
      const info = await getSystemInfoFromAttachment(individualTestResults);
      setSystemInfo(info);
    };

    if (!loadingTests && Object.keys(individualTestResults).length > 0) {
      loadSystemInfo();
    }
  }, [loadingTests, individualTestResults]);

  // State for summary metadata (cli_name, platform, timestamp from test_summary.json)
  const [summaryMeta, setSummaryMeta] = useState<{ cliName: string; platform: string; timestamp: string } | null>(null);

  // Fetch summary metadata and test descriptions from the same test_summary.json attachment
  useEffect(() => {
    const loadSummaryData = async () => {
      const [meta, descriptions] = await Promise.all([
        getSummaryMetaFromAttachment(individualTestResults),
        getTestDescriptionsFromSummary(individualTestResults),
      ]);
      setSummaryMeta(meta);
      setTestDescriptions(descriptions);
    };

    if (!loadingTests && Object.keys(individualTestResults).length > 0) {
      loadSummaryData();
    }
  }, [loadingTests, individualTestResults]);

  // Helper function to convert bytes to GiB, rendered with 1 decimal place and labelled as "GB"
  const bytesToGB = (bytes: number): string => {
    return (bytes / 1073741824).toFixed(1); // 1 GiB = 1024³ bytes
  };

  // Helper function to map internal tier names to display tier names
  const mapTierDisplayName = (tierName: string | undefined): string | undefined => {
    if (!tierName) return undefined;
    if (tierName === "Scalable AI Graphics Media") {
      return "Scalable Performance Graphics Media";
    }
    return tierName;
  };

  // Helper function to get tier display name from any test in a section
  const getSectionTierDisplayName = (profilesData: any[]): string | undefined => {
    for (const profile of profilesData) {
      if (profile.tierDisplayName) {
        return mapTierDisplayName(profile.tierDisplayName);
      }
    }
    return undefined;
  };

  // Hardware Information data preparation from attachment
  const hardwareInfoData: { component: string; details: string[]; packageData?: { [key: string]: string }; expandable?: boolean }[] = [];
  
  if (systemInfo?.hardware) {
    const hardware = systemInfo.hardware;
    
    // Product information (DMI system, BIOS, board) - formatted for marketing-style presentation
    if (hardware.dmi) {
      const dmi = hardware.dmi;
      const productDetails: string[] = [];
      
      if (dmi.system) {
        productDetails.push(`${dmi.system.vendor} ${dmi.system.product_name}`);
        if (dmi.motherboard) {
          productDetails.push(`${dmi.motherboard.name} motherboard`);
        }
      }
      
      if (productDetails.length > 0) {
        hardwareInfoData.push({ component: "Product", details: productDetails });
      }
    }
    
    // CPU information - simplified marketing format
    if (hardware.cpu) {
      const cpu = hardware.cpu;
      const cpuDetails: string[] = [];
      
      cpuDetails.push(`${cpu.brand}`);
      
      // Combine core counts and frequency information in one line
      let coreFreqLine = '';
      if (cpu.sockets && cpu.sockets > 1) {
        coreFreqLine = `${cpu.sockets} sockets • `;
      }
      coreFreqLine += `${cpu.count} cores • ${cpu.logical_count} threads`;
      if (cpu.frequency) {
        coreFreqLine += ` • Current: ${cpu.frequency.current.toFixed(0)} MHz • Range: ${Number(cpu.frequency.min).toFixed(0)}-${Number(cpu.frequency.max).toFixed(0)} MHz`;
      }
      cpuDetails.push(coreFreqLine);
      
      hardwareInfoData.push({ component: "CPU", details: cpuDetails });
    }
    
    // Graphics information with sysfs freq/power config
    if (hardware.gpu?.devices && hardware.gpu.devices.length > 0) {
      const allGpuDetails: string[] = [];

      hardware.gpu.devices.forEach((gpu: any, index: number) => {
        // Blank separator between devices
        if (index > 0) allGpuDetails.push("");

        // ── Header: name • VRAM • EUs • driver ──────────────────────────
        const deviceName = gpu.openvino?.full_device_name || gpu.device_name;
        const driver = gpu.driver || gpu.sysfs?.driver;
        let headerLine = deviceName;
        if (gpu.openvino?.memory_gib !== undefined) {
          headerLine += ` • ${gpu.openvino.memory_gib.toFixed(1)} GB VRAM`;
        } else if (gpu.openvino?.memory_bytes) {
          headerLine += ` • ${bytesToGB(gpu.openvino.memory_bytes)} GB VRAM`;
        }
        if (gpu.openvino?.execution_units) {
          headerLine += ` • ${gpu.openvino.execution_units} EUs`;
        }
        if (driver) headerLine += ` • ${driver}`;
        allGpuDetails.push(headerLine);

        // ── Frequency limits — each GT engine as "Freq <gt> ..." segment ──
        // Terms follow Linux sysfs / Intel GPU spec sheet conventions:
        //   base  = rpn_mhz  hardware floor (shown only when ≠ configured min)
        //   min   = min_mhz  configured minimum clock
        //   eff   = rpe_mhz  efficient / balanced point (shown when distinct)
        //   max   = max_mhz  configured maximum clock
        //   boost = rp0_mhz  rated peak boost (shown only when > max_mhz)
        const freqLimits = gpu.sysfs?.freq_limits;
        if (freqLimits && Object.keys(freqLimits).length > 0) {
          const gtSegments: string[] = [];
          Object.entries(freqLimits).forEach(([gt, lim]: [string, any]) => {
            const parts: string[] = [];
            if (lim.rpn_mhz !== undefined && lim.rpn_mhz !== lim.min_mhz) {
              parts.push(`base ${lim.rpn_mhz} MHz`);
            }
            if (lim.min_mhz !== undefined) parts.push(`min ${lim.min_mhz} MHz`);
            if (lim.rpe_mhz !== undefined &&
                lim.rpe_mhz !== lim.min_mhz &&
                lim.rpe_mhz !== lim.rpn_mhz) {
              parts.push(`eff ${lim.rpe_mhz} MHz`);
            }
            if (lim.max_mhz !== undefined) parts.push(`max ${lim.max_mhz} MHz`);
            if (lim.rp0_mhz !== undefined && lim.rp0_mhz !== lim.max_mhz) {
              parts.push(`boost ${lim.rp0_mhz} MHz`);
            }
            if (parts.length > 0) {
              gtSegments.push(`\tFreq ${gt}  ${parts.join("  ")}`);
            }
          });
          // Each GT gets its own sub-detail line
          gtSegments.forEach(seg => allGpuDetails.push(seg));
        }

        // ── Power limits ─────────────────────────────────────────────────
        // Linux hwmon power sysfs semantics (kernel docs):
        //   cap_w  = power1_cap  — configurable power cap/limit (adjustable; ≤ max_w)
        //   max_w  = power1_max  — hardware absolute power ceiling (cap cannot exceed this)
        //   crit_w = power1_crit — critical threshold; triggers emergency throttle/shutdown
        // NOTE: cap_w is NOT "TDP" — TDP is a fixed manufacturer spec.
        //       The cap is a software-settable limit that defaults to TDP but can differ.
        const pl = gpu.sysfs?.power_limits;
        if (pl && Object.keys(pl).length > 0) {
          const powerParts: string[] = [];
          if (pl.cap_w !== undefined)  powerParts.push(`Power cap ${pl.cap_w} W`);
          if (pl.max_w !== undefined)  powerParts.push(`Power max ${pl.max_w} W`);
          if (pl.crit_w !== undefined) powerParts.push(`Power crit ${pl.crit_w} W`);

          // Fallback: read from channels if top-level fields are absent
          if (powerParts.length === 0 && pl.channels) {
            Object.entries(pl.channels).forEach(([_ch, chLim]: [string, any]) => {
              if (chLim.cap_w !== undefined)  powerParts.push(`Power cap ${chLim.cap_w} W`);
              if (chLim.max_w !== undefined)  powerParts.push(`Power max ${chLim.max_w} W`);
              if (chLim.crit_w !== undefined) powerParts.push(`Power crit ${chLim.crit_w} W`);
            });
          }

          if (powerParts.length > 0) {
            allGpuDetails.push(`\t${powerParts.join("  •  ")}`);
          }
        }
      });

      hardwareInfoData.push({ component: "Graphics", details: allGpuDetails });
    }

    // Storage information - show individual devices with model, interface, size and partition info
    if (hardware.storage?.devices && hardware.storage.devices.length > 0) {
      const allStorageDetails: string[] = [];
      
      hardware.storage.devices.forEach((device: any, index: number) => {
        let storageLine = '';
        
        storageLine += `${device.model || 'Unknown Model'}`;
        if (device.interface) {
          storageLine += ` (${device.interface})`;
        }
        storageLine += ` • ${bytesToGB(device.size)} GB capacity`;
        
        // Check if this device has partitions and show root partition info
        if (device.partitions && device.partitions.length > 0) {
          const rootPartition = device.partitions.find((p: any) => p.mountpoint === '/');
          if (rootPartition) {
            storageLine += ` • ${bytesToGB(rootPartition.free)} GB available (${rootPartition.percent.toFixed(0)}% used)`;
          }
        }
        
        allStorageDetails.push(storageLine);
      });
      
      hardwareInfoData.push({ component: "Storage", details: allStorageDetails });
    }
    
    // Memory information - with DIMM slot information if available
    if (hardware.memory) {
      const memory = hardware.memory;
      const memoryDetails: string[] = [];
      const dimms = memory.dimms;

      const installedRamGib = dimms?.installed_ram_gib;
      const usableRamGib = memory.usable_ram_gib;
      const availableRamGib = memory.available_gib;
      if (installedRamGib !== undefined && usableRamGib !== undefined) {
        let memLine = `${installedRamGib.toFixed(1)} GB (${usableRamGib.toFixed(1)} GB usable)`;
        if (availableRamGib !== undefined) {
          memLine += ` • ${availableRamGib.toFixed(1)} GB available (${memory.percent.toFixed(0)}% used)`;
        }
        if (dimms?.available && dimms.slot_count !== undefined) {
          memLine += ` • ${dimms.installed_count} of ${dimms.slot_count} slots`;
        }
        memoryDetails.push(memLine);
      } else {
        memoryDetails.push(`${bytesToGB(memory.total)} GB • ${bytesToGB(memory.available)} GB available (${memory.percent.toFixed(0)}% used)`);
      }

      // Per-slot details
      if (dimms?.available && dimms.slot_count !== undefined) {
        // Per-slot details if devices are available
        if (dimms.devices && dimms.devices.length > 0) {
          dimms.devices.forEach((dimm: any, idx: number) => {
            const locator = dimm.locator || `Slot ${idx + 1}`;
            const isInstalled = dimm.installed === true || (dimm.size && dimm.size !== 'No Module Installed' && dimm.type !== 'Unknown');

            if (isInstalled) {
              let slotLine = `${locator}`;
              if (dimm.size) slotLine += ` • ${dimm.size}`;
              if (dimm.type) slotLine += ` ${dimm.type}`;
              if (dimm.speed_mts) slotLine += ` @ ${dimm.speed_mts} MT/s`;
              if (dimm.manufacturer) slotLine += ` • ${dimm.manufacturer.trim()}`;
              if (dimm.part_number) slotLine += ` ${dimm.part_number.trim()}`;
              memoryDetails.push(`\t${slotLine}`);
            } else {
              memoryDetails.push(`\t${locator} • No Module Installed`);
            }
          });
        }
      }

      hardwareInfoData.push({ component: "Memory", details: memoryDetails });
    }

    // Power information - Intel RAPL (Running Average Power Limit) interface
    if (hardware.power?.control_types && hardware.power.control_types.length > 0) {
      const powerDetails: string[] = [];
      
      hardware.power.control_types.forEach((controlType: any) => {
        if (controlType.name === "intel-rapl" && controlType.zones && controlType.zones.length > 0) {
          controlType.zones.forEach((zone: any) => {
            // Only display package-level zones (CPU sockets)
            if (zone.name && zone.name.startsWith("package-")) {
              const zoneLabel = zone.name.replace("package-", "CPU Package ");
              
              // First line: Package with energy information
              let packageLine = `${zoneLabel}`;
              
              if (zone.max_energy_range_uj) {
                const currentKJ = zone.energy_uj ? (zone.energy_uj / 1000000000).toFixed(0) : "0";
                const maxRangeKJ = (zone.max_energy_range_uj / 1000000000).toFixed(0);
                const energyPercent = zone.energy_uj ? ((zone.energy_uj / zone.max_energy_range_uj) * 100).toFixed(0) : "0";
                packageLine += ` • Energy counter: ${currentKJ} kJ / ${maxRangeKJ} kJ (${energyPercent}%)`;
              }
              
              // Add subzones count if available
              if (zone.subzones && zone.subzones.length > 0) {
                packageLine += ` • ${zone.subzones.length} subzones`;
              }
              
              powerDetails.push(packageLine);
              
              // Second line: Limits - dynamically display all available power constraints
              if (zone.constraints && zone.constraints.length > 0) {
                const constraintParts: string[] = [];
                
                zone.constraints.forEach((constraint: any) => {
                  if (constraint.power_limit_uw && constraint.power_limit_uw > 0) {
                    const powerW = (constraint.power_limit_uw / 1000000).toFixed(0);
                    const constraintName = constraint.name || 'unknown';
                    
                    // Format constraint name for display - abbreviated
                    let displayName = constraintName.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase());
                    displayName = displayName.replace('Long Term', 'PL1').replace('Short Term', 'PL2').replace('Peak Power', 'Peak');
                    
                    let constraintStr = `${displayName}: ${powerW} W`;
                    
                    // Add max power if available to show headroom/margin
                    if (constraint.max_power_uw && constraint.max_power_uw > 0) {
                      const maxPowerW = (constraint.max_power_uw / 1000000).toFixed(0);
                      const marginPercent = ((constraint.power_limit_uw / constraint.max_power_uw) * 100).toFixed(0);
                      constraintStr += ` / ${maxPowerW} W (${marginPercent}%)`;
                    }
                    
                    // Add time window if available (and not peak_power which doesn't use time window)
                    if (constraint.time_window_us && constraintName !== 'peak_power') {
                      const timeMs = (constraint.time_window_us / 1000);
                      if (timeMs < 1) {
                        constraintStr += ` @${constraint.time_window_us}µs`;
                      } else if (timeMs < 1000) {
                        constraintStr += ` @${timeMs.toFixed(1)}ms`;
                      } else {
                        constraintStr += ` @${(timeMs / 1000).toFixed(1)}s`;
                      }
                    }
                    
                    constraintParts.push(constraintStr);
                  }
                });
                
                if (constraintParts.length > 0) {
                  const constraintCount = constraintParts.length;
                  powerDetails.push(`${constraintCount} ${constraintCount === 1 ? 'limit' : 'limits'} • ${constraintParts.join(' • ')}`);
                }
              }
            }
          });
        }
      });
      
      if (powerDetails.length > 0) {
        hardwareInfoData.push({ component: "Power", details: powerDetails });
      }
    }
  }

  // Software Information data preparation from attachment
  const softwareInfoData: { component: string; details: string[]; packageData?: { [key: string]: string }; expandable?: boolean }[] = [];
  
  if (systemInfo?.software) {
    const software = systemInfo.software;
    
    // Operating System information - marketing-style presentation
    if (software.os) {
      const os = software.os;
      const osDetails: string[] = [];
      
      if (os.distribution?.pretty_name) {
        osDetails.push(`${os.distribution.pretty_name}`);
      } else if (os.name) {
        osDetails.push(`${os.name}`);
      }
      
      if (os.kernel?.version) {
        osDetails.push(`Kernel ${os.kernel.version}`);
      } else if (os.release) {
        osDetails.push(`Kernel ${os.release}`);
      }
      
      if (osDetails.length > 0) {
        softwareInfoData.push({ component: "Operating System", details: osDetails });
      }
    }
    
    // Python information
    if (software.python) {
      const python = software.python;
      const pythonDetails: string[] = [];
      
      if (python.version) {
        const versionMatch = python.version.match(/^(\d+\.\d+\.\d+)/);
        const cleanVersion = versionMatch ? versionMatch[1] : python.version;
        pythonDetails.push(`Python ${cleanVersion}`);
      }
      
      if (python.in_virtualenv && python.virtualenv?.name) {
        pythonDetails.push(`Virtual environment: ${python.virtualenv.name}`);
      }
      
      if (pythonDetails.length > 0) {
        softwareInfoData.push({ component: "Python Runtime", details: pythonDetails });
      }
    }
    
    // System packages information
    if (software.system_packages) {
      const systemPackages = software.system_packages;
      const packageDetails: string[] = [];
      
      // Show single line summary
      if (systemPackages.total_installed) {
        packageDetails.push(`${Object.keys(systemPackages.packages || {}).length} tracked • ${systemPackages.total_installed} installed on system`);
      }
      
      if (packageDetails.length > 0) {
        softwareInfoData.push({ 
          component: "System Packages", 
          details: packageDetails,
          packageData: systemPackages.packages || {},
          expandable: true
        });
      }
    }
    
    // Python packages information
    if (software.python_packages) {
      const pythonPackages = software.python_packages;
      const packageDetails: string[] = [];
      
      // Show single line summary
      if (pythonPackages.total_installed) {
        packageDetails.push(`${Object.keys(pythonPackages.packages || {}).length} tracked • ${pythonPackages.total_installed} installed on system`);
      }
      
      if (packageDetails.length > 0) {
        softwareInfoData.push({ 
          component: "Python Packages", 
          details: packageDetails,
          packageData: pythonPackages.packages || {},
          expandable: true
        });
      }
    }
  }

  return (
    <div className={styles.overview}>
      {/* System Information */}
      {(hardwareInfoData.length > 0 || softwareInfoData.length > 0) && (
        <div className={styles["overview-grid-item"]}>
          <div className={styles["collapsible-section"]}>
            <div className={styles["collapsible-header"]}>
              <div className={styles["header-left"]}>
                <h3 className={styles["section-title"]}>System Information</h3>
              </div>
              <div className={styles["expand-collapse-controls"]}>
                <button 
                  className={styles["secondary-button"]}
                  onClick={expandAllSystemSections}
                  type="button"
                >
                  Expand All
                </button>
                <button 
                  className={styles["secondary-button"]}
                  onClick={collapseAllSystemSections}
                  type="button"
                >
                  Collapse All
                </button>
              </div>
            </div>
            
            <div className={styles["profiles-container"]}>
              {hardwareInfoData.length > 0 && (
                <div className={styles["profile-section"]}>
                  <div 
                    className={styles["profile-header"]}
                    onClick={() => toggleSystemSection("hardware")}
                  >
                    <button 
                      className={styles["toggle-button"]}
                      type="button"
                      aria-expanded={expandedSystemSections.has("hardware")}
                    >
                      {expandedSystemSections.has("hardware") ? "−" : "+"}
                    </button>
                    <h4 className={styles["profile-title"]}>Hardware</h4>
                  </div>
                  
                  {expandedSystemSections.has("hardware") && (
                    <div className={styles["profile-content"]}>
                      <SystemInfoTable
                        title=""
                        data={hardwareInfoData}
                        expandedPackageDetails={expandedPackageDetails}
                        onTogglePackageDetails={togglePackageDetails}
                      />
                    </div>
                  )}
                </div>
              )}

              {softwareInfoData.length > 0 && (
                <div className={styles["profile-section"]}>
                  <div 
                    className={styles["profile-header"]}
                    onClick={() => toggleSystemSection("software")}
                  >
                    <button 
                      className={styles["toggle-button"]}
                      type="button"
                      aria-expanded={expandedSystemSections.has("software")}
                    >
                      {expandedSystemSections.has("software") ? "−" : "+"}
                    </button>
                    <h4 className={styles["profile-title"]}>Software</h4>
                  </div>
                  
                  {expandedSystemSections.has("software") && (
                    <div className={styles["profile-content"]}>
                      <SystemInfoTable
                        title=""
                        data={softwareInfoData}
                        expandedPackageDetails={expandedPackageDetails}
                        onTogglePackageDetails={togglePackageDetails}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Qualifications */}
      {qualificationProfilesData.length > 0 && (
        <div className={styles["overview-grid-item"]}>
          <div className={styles["collapsible-section"]}>
            <div className={styles["collapsible-header-enhanced"]}>
              <div className={styles["header-left-enhanced"]}>
                <h3 className={styles["section-title"]}>Qualifications</h3>
              </div>
              <div className={styles["header-right-enhanced"]}>
                <div className={styles["search-container-inline"]}>
                  <div className={styles["search-box"]}>
                    <input
                      type="text"
                      className={styles["search-input"]}
                      placeholder="Search"
                      value={qualificationSearchTerm}
                      onInput={handleQualificationSearchChange}
                    />
                    <button
                      className={styles["clear-button"]}
                      onClick={clearQualificationSearch}
                      type="button"
                      title="Clear search"
                    >
                      ×
                    </button>
                  </div>
                </div>
                <div className={styles["expand-collapse-controls"]}>
                  <button 
                    className={styles["secondary-button"]}
                    onClick={expandAllQualificationProfiles}
                    type="button"
                  >
                    Expand All
                  </button>
                  <button 
                    className={styles["secondary-button"]}
                    onClick={collapseAllQualificationProfiles}
                    type="button"
                  >
                    Collapse All
                  </button>
                </div>
              </div>
            </div>

            {/* Qualification Summary Section - Bullet Chart + Tier Bar */}
            {(() => {
              const sectionTierDisplayName = getSectionTierDisplayName(filteredQualificationProfilesData);
              if (!sectionTierDisplayName) return null;
              
              // Tiers ordered from highest to lowest (top to bottom display)
              // This creates a bottom-to-top visual hierarchy where Entry is at the bottom
              const tiers = [
                "Scalable Performance Graphics Media",
                "Scalable Performance",
                "Efficiency Optimized",
                "Mainstream",
                "Entry"
              ];

              // Collect all metrics from all qualification profiles for bullet chart
              // Extract numeric reference values from full metrics data (KPIs)
              const allMetrics = filteredQualificationProfilesData.flatMap(profile =>
                profile.testsData
                  .filter((test: any) => {
                    // Get the full metrics data from cache to extract numeric reference
                    const fullMetrics = testFullMetrics[test.fullId];
                    
                    // Parse the actual value from test data
                    const valueStr = String(test.value);
                    if (valueStr === "N/A" || !valueStr) {
                      return false;
                    }
                    
                    const value = parseFloat(valueStr);
                    if (isNaN(value) || value <= 0) {
                      return false;
                    }
                    
                    // Extract numeric reference from KPIs in full metrics
                    if (!fullMetrics?.kpis) {
                      return false;
                    }
                    
                    // Check if we can extract a valid numeric reference from KPIs
                    let hasValidReference = false;
                    for (const kpiData of Object.values(fullMetrics.kpis)) {
                      if (kpiData && typeof kpiData === 'object') {
                        const config = (kpiData as any).config || {};
                        const validation = config.validation || {};
                        if (validation.reference !== undefined) {
                          const ref = parseFloat(String(validation.reference));
                          if (!isNaN(ref) && ref > 0) {
                            hasValidReference = true;
                            break;
                          }
                        }
                      }
                    }
                    
                    return hasValidReference;
                  })
                  .map((test: any) => {
                    // Get the full metrics data from cache
                    const fullMetrics = testFullMetrics[test.fullId];
                    
                    // Parse the actual value
                    const valueStr = String(test.value);
                    const value = parseFloat(valueStr);
                    
                    // Extract numeric reference from KPIs (first valid one)
                    let reference = 0;
                    for (const kpiData of Object.values(fullMetrics.kpis)) {
                      if (kpiData && typeof kpiData === 'object') {
                        const config = (kpiData as any).config || {};
                        const validation = config.validation || {};
                        if (validation.reference !== undefined) {
                          const ref = parseFloat(String(validation.reference));
                          if (!isNaN(ref) && ref > 0) {
                            reference = ref;
                            break;
                          }
                        }
                      }
                    }
                    
                    return {
                      metric: String(test.metric || test.testName || "Unknown Metric"),
                      value: value,
                      reference: reference,
                      unit: String(test.unit || ""),
                      status: String(test.status)
                    };
                  })
              );

              // Sanitize string fields using character-by-character copy to break
              // Coverity taint chain from external JSON attachment data (DOM_XSS).
              type SafeStatus = "passed" | "failed" | "broken" | "skipped" | "unknown";
              const SAFE_STATUSES: SafeStatus[] = ["passed", "failed", "broken", "skipped", "unknown"];
              const safeMetrics: Array<{metric: string; value: number; reference: number; unit: string; status: SafeStatus}> = [];
              for (const raw of allMetrics) {
                let safeName = "";
                for (const ch of raw.metric) {
                  if (ch !== "<" && ch !== ">" && ch !== "&" && ch !== '"' && ch !== "'") safeName += ch;
                }
                let safeUnit = "";
                for (const ch of raw.unit) {
                  if (ch !== "<" && ch !== ">" && ch !== "&" && ch !== '"' && ch !== "'") safeUnit += ch;
                }
                const safeStatus: SafeStatus = SAFE_STATUSES.includes(raw.status as SafeStatus)
                  ? (raw.status as SafeStatus)
                  : "unknown";
                safeMetrics.push({
                  metric: safeName || "Unknown",
                  value: Number(raw.value),
                  reference: Number(raw.reference),
                  unit: safeUnit,
                  status: safeStatus
                });
              }

              return (
                <div className={styles["qualification-summary-section"]}>
                  {/* Left side: Bullet Chart (3/4) */}
                  <div className={styles["bullet-chart-section"]}>
                    {/* <div className={styles["bullet-chart-title"]}>Metrics Overview</div> */}
                    <div className={bulletChartStyles.bulletChartWrapper}>
                      {safeMetrics.length > 0 ? (
                        <BulletChart data={safeMetrics} />
                      ) : (
                        <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '12px' }}>
                          No metrics data available for visualization
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Right side: Vertical Tier Bar (1/4) */}
                  <div className={styles["tier-bar-section"]}>
                    {/* <div className={styles["tier-bar-title"]}>System Tier</div> */}
                    <div className={styles["tier-bar"]}>
                      {tiers.map((tier) => (
                        <div 
                          key={tier}
                          className={`${styles["tier-item"]} ${tier === sectionTierDisplayName ? styles["active"] : ""}`}
                        >
                          <span className={styles["tier-item-name"]}>{tier}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })()}
            
            <div className={styles["profiles-container"]}>
              {filteredQualificationProfilesData.map((profileData, index) => (
                profileData.testsData.length > 0 && (
                  <div key={index} className={styles["profile-section"]}>
                    <div 
                      className={`${styles["profile-header"]} ${styles["qualification-profile-header"]}`}
                      onClick={() => toggleQualificationProfile(profileData.profileName)}
                    >
                      <button 
                        className={styles["toggle-button"]}
                        type="button"
                        aria-expanded={expandedQualificationProfiles.has(profileData.profileName)}
                      >
                        {expandedQualificationProfiles.has(profileData.profileName) ? "−" : "+"}
                      </button>
                      <div className={styles["qualification-profile-info"]}>
                        <h4 className={styles["profile-title"]}>{profileData.profileName}</h4>
                      </div>
                      <div className={styles["qualification-profile-summary"]}>
                        {profileData.profileType === "qualification" && (
                          <span className={`${styles["status-badge"]} ${styles[`status-${profileData.overallStatus}`]}`}>
                            {profileData.overallStatus}
                          </span>
                        )}

                        <span className={styles["table-test-count"]}>
                          {profileData.totalTests}
                        </span>
                      </div>
                    </div>
                    
                    {expandedQualificationProfiles.has(profileData.profileName) && (
                      <div className={styles["profile-content"]}>
                        <SortableTable
                          title=""
                          headers={(() => {
                            const baseHeaders = [];
                            if (!profileData.hideIdColumn) {
                              baseHeaders.push("ID");
                            }
                            baseHeaders.push("Test Name", "Metric");
                            if (profileData.showReferenceColumn) {
                              baseHeaders.push("Reference");
                            }
                            baseHeaders.push("Value", "Unit");
                            if (!profileData.hideStatusColumn) {
                              baseHeaders.push("Status");
                            }
                            return baseHeaders;
                          })()}
                          data={profileData.testsData.map((test: any) => {
                            const baseData = [];
                            if (!profileData.hideIdColumn) {
                              baseData.push(test.id);
                            }
                            baseData.push(test.testName, test.metric);
                            if (profileData.showReferenceColumn) {
                              baseData.push(test.reference);
                            }
                            baseData.push(test.value, test.unit);
                            if (!profileData.hideStatusColumn) {
                              baseData.push(test.status);
                            }
                            return baseData;
                          })}
                          onRowClick={(rowIndex: number) => {
                            const test = profileData.testsData[rowIndex];
                            toggleTestDetails(test.fullId);
                          }}
                          expandedRows={profileData.testsData.map((test: any) => 
                            expandedTestDetails.has(test.fullId)
                          )}
                          renderExpandedContent={(rowIndex: number) => {
                            const test = profileData.testsData[rowIndex];
                            const testId = test.fullId; // Use the full ID to access individual test results
                            const testAlias = individualTestResults[testId]?.parameters?.find((p: any) => p.name === "Test Id")?.value?.replace(/^'|'$/g, "").trim();
                            
                            return (
                              <div className={styles["test-details"]}>
                                {individualTestResults[testId] ? (
                                  <div>
                                    {/* Error Messages for Failed Tests */}
                                    {test.status === 'failed' && individualTestResults[testId]?.error?.message && (
                                      <div className={styles["detail-section-error"]}>
                                        <h4 className={styles["detail-section-title-dark"]}>Error Details</h4>
                                        <pre className={styles["detail-section-message"]}>
                                          {individualTestResults[testId].error.message}
                                        </pre>
                                      </div>
                                    )}

                                    {/* Status Details for Skipped and Broken Tests */}
                                    {(test.status === 'skipped' || test.status === 'broken') && individualTestResults[testId]?.error?.message && (
                                      <div className={test.status === 'skipped' ? styles["detail-section-skipped"] : styles["detail-section-broken"]}>
                                        <h4 className={styles["detail-section-title-dark"]}>
                                          {test.status === 'skipped' ? 'Skip Details' : 'Broken Details'}
                                        </h4>
                                        {individualTestResults[testId].error.message && (
                                          <pre className={styles["detail-section-message"]}>
                                            {individualTestResults[testId].error.message}
                                          </pre>
                                        )}
                                      </div>
                                    )}

                                    {/* Overview Section */}
                                    <div className={styles["detail-section"]}>
                                      <h4 className={styles["detail-section-title"]}>Overview</h4>
                                      {(() => { const desc = getTestDescription(testId); return desc ? <DescriptionBlock text={desc} /> : null; })()}
                                      <div className={styles["detail-grid-3col"]}>
                                        <div>
                                          <span className={styles["detail-label"]}>History: </span>
                                          <span>{individualTestResults[testId]?.history?.length || "1"}</span>
                                        </div>
                                        <div>
                                          <span className={styles["detail-label"]}>Retries: </span>
                                          <span>{individualTestResults[testId]?.retries?.length || '0'}</span>
                                        </div>
                                        <div>
                                          <span className={styles["detail-label"]}>Duration: </span>
                                          <span>{individualTestResults[testId]?.duration 
                                            ? `${(individualTestResults[testId].duration / 1000).toFixed(2)}s`
                                            : 'N/A'}</span>
                                        </div>
                                      </div>
                                    </div>

                                    {/* Metadata Section */}
                                    <MetadataTable metricsData={testFullMetrics[testId]} />

                                    {/* Telemetry Section */}
                                    <TelemetrySection extendedMetadata={testFullMetrics[testId]?.extended_metadata} summaryMeta={summaryMeta} testId={testAlias} />

                                    {/* Attachments Section */}
                                    {(() => {
                                      const testData = individualTestResults[testId];
                                      const imageAttachments = testData?.attachments?.filter((attachment: any) => 
                                        attachment.link?.contentType === "image/png" || 
                                        attachment.link?.contentType === "image/jpeg"
                                      ) || [];
                                      const csvAttachments = testData?.attachments?.filter((attachment: any) => 
                                        attachment.link?.contentType === "text/csv"
                                      ) || [];
                                      
                                      const hasAttachments = imageAttachments.length > 0 || csvAttachments.length > 0;
                                      
                                      return hasAttachments ? (
                                        <div className={styles["attachment-section"]}>
                                          {/* Image Attachments */}
                                          {imageAttachments.length > 0 && (
                                            <div className={styles["attachment-container"]} style={{ marginBottom: csvAttachments.length > 0 ? '12px' : '0' }}>
                                              <h4 className={styles["detail-section-title"]}>Image Attachments</h4>
                                              <div className={styles["attachment-image-grid"]}>
                                                {imageAttachments.map((attachment: any, index: number) => {
                                                  return (
                                                    <div key={index} className={styles["attachment-image-item"]}>
                                                      <div className={styles["attachment-image-wrapper"]}>
                                                        <AttachmentImage 
                                                          attachment={attachment}
                                                          onError={() => console.warn('Failed to load attachment:', attachment.link?.id)}
                                                        />
                                                      </div>
                                                      <div className={styles["attachment-image-name"]}>
                                                        {attachment.name || attachment.link?.name || `Attachment ${index + 1}`}
                                                      </div>
                                                    </div>
                                                  );
                                                })}
                                              </div>
                                            </div>
                                          )}
                                          
                                          {/* CSV Attachments - Full Width */}
                                          {csvAttachments.length > 0 && (
                                            <div className={styles["csv-attachments"]}>
                                              {csvAttachments.map((attachment: any, index: number) => (
                                                <div key={index} className={styles["attachment-container"]} style={{ 
                                                  marginBottom: index < csvAttachments.length - 1 ? '12px' : '0'
                                                }}>
                                                  <AttachmentCSV 
                                                    attachment={attachment}
                                                    onError={() => console.warn('Failed to load CSV attachment:', attachment.link?.id)}
                                                  />
                                                </div>
                                              ))}
                                            </div>
                                          )}
                                        </div>
                                      ) : null;
                                    })()}
                                  </div>
                                ) : (
                                  <div style={{ padding: '16px', textAlign: 'center', color: '#666' }}>
                                    Loading test details...
                                  </div>
                                )}
                              </div>
                            );
                          }}
                        />
                      </div>
                    )}
                  </div>
                )
              ))}
              
              {qualificationSearchTerm && filteredQualificationProfilesData.every(profile => profile.testsData.length === 0) && (
                <div className={styles["no-results"]}>
                  <p>No matching qualification profiles or tests found for "{qualificationSearchTerm}"</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Vertical Profiles */}
      {verticalProfilesData.length > 0 && (
        <div className={styles["overview-grid-item"]}>
          <div className={styles["collapsible-section"]}>
            <div className={styles["collapsible-header-enhanced"]}>
              <div className={styles["header-left-enhanced"]}>
                <h3 className={styles["section-title"]}>Vertical</h3>
              </div>
              <div className={styles["header-right-enhanced"]}>
                <div className={styles["search-container-inline"]}>
                  <div className={styles["search-box"]}>
                    <input
                      type="text"
                      className={styles["search-input"]}
                      placeholder="Search"
                      value={verticalSearchTerm}
                      onInput={handleVerticalSearchChange}
                    />
                    <button
                      className={styles["clear-button"]}
                      onClick={clearVerticalSearch}
                      type="button"
                      title="Clear search"
                    >
                      ×
                    </button>
                  </div>
                </div>
                <div className={styles["expand-collapse-controls"]}>
                  <button 
                    className={styles["secondary-button"]}
                    onClick={expandAllVerticalProfiles}
                    type="button"
                  >
                    Expand All
                  </button>
                  <button 
                    className={styles["secondary-button"]}
                    onClick={collapseAllVerticalProfiles}
                    type="button"
                  >
                    Collapse All
                  </button>
                </div>
              </div>
            </div>
            
            <div className={styles["profiles-container"]}>
              {filteredVerticalProfilesData.map((profileData, index) => (
                profileData.testsData.length > 0 && (
                  <div key={index} className={styles["profile-section"]}>
                    <div 
                      className={`${styles["profile-header"]} ${styles["qualification-profile-header"]}`}
                      onClick={() => toggleVerticalProfile(profileData.profileName)}
                    >
                      <button 
                        className={styles["toggle-button"]}
                        type="button"
                        aria-expanded={expandedVerticalProfiles.has(profileData.profileName)}
                      >
                        {expandedVerticalProfiles.has(profileData.profileName) ? "−" : "+"}
                      </button>
                      <div className={styles["qualification-profile-info"]}>
                        <h4 className={styles["profile-title"]}>{profileData.profileName}</h4>
                      </div>
                      <div className={styles["qualification-profile-summary"]}>
                        <span className={styles["table-test-count"]}>
                          {profileData.totalTests}
                        </span>
                      </div>
                    </div>
                    
                    {expandedVerticalProfiles.has(profileData.profileName) && (
                      <div className={styles["profile-content"]}>
                        <SortableTable
                          title=""
                          headers={(() => {
                            const baseHeaders = [];
                            if (!profileData.hideIdColumn) {
                              baseHeaders.push("ID");
                            }
                            baseHeaders.push("Test Name", "Metric");
                            if (profileData.showReferenceColumn) {
                              baseHeaders.push("Reference");
                            }
                            baseHeaders.push("Value", "Unit");
                            if (!profileData.hideStatusColumn) {
                              baseHeaders.push("Status");
                            }
                            return baseHeaders;
                          })()}
                          data={profileData.testsData.map((test: any) => {
                            const baseData = [];
                            if (!profileData.hideIdColumn) {
                              baseData.push(test.id);
                            }
                            baseData.push(test.testName, test.metric);
                            if (profileData.showReferenceColumn) {
                              baseData.push(test.reference);
                            }
                            baseData.push(test.value, test.unit);
                            if (!profileData.hideStatusColumn) {
                              baseData.push(test.status);
                            }
                            return baseData;
                          })}
                          onRowClick={(rowIndex: number) => {
                            const test = profileData.testsData[rowIndex];
                            toggleTestDetails(test.fullId);
                          }}
                          expandedRows={profileData.testsData.map((test: any) => 
                            expandedTestDetails.has(test.fullId)
                          )}
                          renderExpandedContent={(rowIndex: number) => {
                            const test = profileData.testsData[rowIndex];
                            const testId = test.fullId; // Use the full ID to access individual test results
                            const testAlias = individualTestResults[testId]?.parameters?.find((p: any) => p.name === "Test Id")?.value?.replace(/^'|'$/g, "").trim();
                            
                            return (
                              <div className={styles["test-details"]}>
                                {individualTestResults[testId] ? (
                                  <div>
                                    {/* Error Messages for Failed Tests */}
                                    {test.status === 'failed' && individualTestResults[testId]?.error?.message && (
                                      <div className={styles["detail-section-error"]}>
                                        <h4 className={styles["detail-section-title-dark"]}>Error Details</h4>
                                        <pre className={styles["detail-section-message"]}>
                                          {individualTestResults[testId].error.message}
                                        </pre>
                                      </div>
                                    )}

                                    {/* Status Details for Skipped and Broken Tests */}
                                    {(test.status === 'skipped' || test.status === 'broken') && individualTestResults[testId]?.error?.message && (
                                      <div className={test.status === 'skipped' ? styles["detail-section-skipped"] : styles["detail-section-broken"]}>
                                        <h4 className={styles["detail-section-title-dark"]}>
                                          {test.status === 'skipped' ? 'Skip Details' : 'Broken Details'}
                                        </h4>
                                        {individualTestResults[testId].error.message && (
                                          <pre className={styles["detail-section-message"]}>
                                            {individualTestResults[testId].error.message}
                                          </pre>
                                        )}
                                      </div>
                                    )}

                                    {/* Overview Section */}
                                    <div className={styles["detail-section"]}>
                                      <h4 className={styles["detail-section-title"]}>Overview</h4>
                                      {(() => { const desc = getTestDescription(testId); return desc ? <DescriptionBlock text={desc} /> : null; })()}
                                      <div className={styles["detail-grid-3col"]}>
                                        <div>
                                          <span className={styles["detail-label"]}>History: </span>
                                          <span>{individualTestResults[testId]?.history?.length || "1"}</span>
                                        </div>
                                        <div>
                                          <span className={styles["detail-label"]}>Retries: </span>
                                          <span>{individualTestResults[testId]?.retries?.length || '0'}</span>
                                        </div>
                                        <div>
                                          <span className={styles["detail-label"]}>Duration: </span>
                                          <span>{individualTestResults[testId]?.duration 
                                            ? `${(individualTestResults[testId].duration / 1000).toFixed(2)}s`
                                            : 'N/A'}</span>
                                        </div>
                                      </div>
                                    </div>

                                    {/* Metadata Section */}
                                    <MetadataTable metricsData={testFullMetrics[testId]} />

                                    {/* Telemetry Section */}
                                    <TelemetrySection extendedMetadata={testFullMetrics[testId]?.extended_metadata} summaryMeta={summaryMeta} testId={testAlias} />

                                    {/* Attachments Section */}
                                    {(() => {
                                      const testData = individualTestResults[testId];
                                      const imageAttachments = testData?.attachments?.filter((attachment: any) => 
                                        attachment.link?.contentType === "image/png" || 
                                        attachment.link?.contentType === "image/jpeg"
                                      ) || [];
                                      const csvAttachments = testData?.attachments?.filter((attachment: any) => 
                                        attachment.link?.contentType === "text/csv"
                                      ) || [];
                                      
                                      const hasAttachments = imageAttachments.length > 0 || csvAttachments.length > 0;
                                      
                                      return hasAttachments ? (
                                        <div className={styles["attachment-section"]}>
                                          {/* Image Attachments */}
                                          {imageAttachments.length > 0 && (
                                            <div className={styles["attachment-container"]} style={{ marginBottom: csvAttachments.length > 0 ? '12px' : '0' }}>
                                              <h4 className={styles["detail-section-title"]}>Image Attachments</h4>
                                              <div className={styles["attachment-image-grid"]}>
                                                {imageAttachments.map((attachment: any, index: number) => {
                                                  return (
                                                    <div key={index} className={styles["attachment-image-item"]}>
                                                      <div className={styles["attachment-image-wrapper"]}>
                                                        <AttachmentImage 
                                                          attachment={attachment}
                                                          onError={() => console.warn('Failed to load attachment:', attachment.link?.id)}
                                                        />
                                                      </div>
                                                      <div className={styles["attachment-image-name"]}>
                                                        {attachment.name || attachment.link?.name || `Attachment ${index + 1}`}
                                                      </div>
                                                    </div>
                                                  );
                                                })}
                                              </div>
                                            </div>
                                          )}
                                          
                                          {/* CSV Attachments - Full Width */}
                                          {csvAttachments.length > 0 && (
                                            <div className={styles["csv-attachments"]}>
                                              {csvAttachments.map((attachment: any, index: number) => (
                                                <div key={index} className={styles["attachment-container"]} style={{ 
                                                  marginBottom: index < csvAttachments.length - 1 ? '12px' : '0'
                                                }}>
                                                  <AttachmentCSV 
                                                    attachment={attachment}
                                                    onError={() => console.warn('Failed to load CSV attachment:', attachment.link?.id)}
                                                  />
                                                </div>
                                              ))}
                                            </div>
                                          )}
                                        </div>
                                      ) : null;
                                    })()}
                                  </div>
                                ) : (
                                  <div style={{ padding: '16px', textAlign: 'center', color: '#666' }}>
                                    Loading test details...
                                  </div>
                                )}
                              </div>
                            );
                          }}
                        />
                      </div>
                    )}
                  </div>
                )
              ))}
              
              {verticalSearchTerm && filteredVerticalProfilesData.every(profile => profile.testsData.length === 0) && (
                <div className={styles["no-results"]}>
                  <p>No matching vertical profiles or tests found for "{verticalSearchTerm}"</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Individual Profile Tables */}
      {individualProfilesData.length > 0 && (
        <div className={styles["overview-grid-item"]}>
          <div className={styles["collapsible-section"]}>
            <div className={styles["collapsible-header-enhanced"]}>
              <div className={styles["header-left-enhanced"]}>
                <h3 className={styles["section-title"]}>Test Results</h3>
                {(() => {
                  const sectionTierDisplayName = getSectionTierDisplayName(filteredProfilesData);
                  return sectionTierDisplayName ? (
                    <span className={styles["test-tier-label"]}>
                      {sectionTierDisplayName}
                    </span>
                  ) : null;
                })()}
              </div>
              <div className={styles["header-right-enhanced"]}>
                <div className={styles["search-container-inline"]}>
                  <div className={styles["search-box"]}>
                    <input
                      type="text"
                      className={styles["search-input"]}
                      placeholder="Search"
                      value={searchTerm}
                      onInput={handleSearchChange}
                    />
                    <button
                      className={styles["clear-button"]}
                      onClick={clearSearch}
                      type="button"
                      title="Clear search"
                    >
                      ×
                    </button>
                  </div>
                </div>
                <div className={styles["expand-collapse-controls"]}>
                  <button 
                    className={styles["secondary-button"]}
                    onClick={expandAllProfiles}
                    type="button"
                  >
                    Expand All
                  </button>
                  <button 
                    className={styles["secondary-button"]}
                    onClick={collapseAllProfiles}
                    type="button"
                  >
                    Collapse All
                  </button>
                </div>
              </div>
            </div>
            
            <div className={styles["profiles-container"]}>
              {filteredProfilesData.map((profileData, index) => (
                profileData.testsData.length > 0 && (
                  <div key={index} className={styles["profile-section"]}>
                    <div 
                      className={styles["profile-header"]}
                      onClick={() => toggleProfile(profileData.profileName)}
                    >
                      <button 
                        className={styles["toggle-button"]}
                        type="button"
                        aria-expanded={expandedProfiles.has(profileData.profileName)}
                      >
                        {expandedProfiles.has(profileData.profileName) ? "−" : "+"}
                      </button>
                      <div className={styles["qualification-profile-info"]}>
                        <h4 className={styles["profile-title"]}>{profileData.profileName}</h4>
                      </div>
                      <div className={styles["qualification-profile-summary"]}>
                        {(() => {
                          const totalTests = profileData.testsData.length;
                          
                          // Only show status badge for qualification type profiles
                          if (profileData.profileType === "qualification") {
                            const failedTests = profileData.testsData.filter(test => test.status === "failed" || test.status === "broken").length;
                            const overallStatus = failedTests > 0 ? "failed" : "passed";
                            
                            return (
                              <>
                                <span className={`${styles["status-badge"]} ${styles[`status-${overallStatus}`]}`}>
                                  {overallStatus}
                                </span>
                                <span className={styles["table-test-count"]}>
                                  {totalTests}
                                </span>
                              </>
                            );
                          } else {
                            // For non-qualification profiles, show test count only
                            return (
                              <>
                                <span className={styles["table-test-count"]}>
                                  {totalTests}
                                </span>
                              </>
                            );
                          }
                        })()}
                      </div>
                    </div>
                    
                    {expandedProfiles.has(profileData.profileName) && (
                      <div className={styles["profile-content"]}>
                        <SortableTable
                          title=""
                          headers={(() => {
                            const baseHeaders = [];
                            if (!profileData.hideIdColumn) {
                              baseHeaders.push("ID");
                            }
                            baseHeaders.push("Test Name", "Metric");
                            if (profileData.showReferenceColumn) {
                              baseHeaders.push("Reference");
                            }
                            baseHeaders.push("Value", "Unit");
                            if (!profileData.hideStatusColumn) {
                              baseHeaders.push("Status");
                            }
                            return baseHeaders;
                          })()}
                          data={profileData.testsData.map((test: any) => {
                            const baseData = [];
                            if (!profileData.hideIdColumn) {
                              baseData.push(test.id);
                            }
                            baseData.push(test.testName, test.metric);
                            if (profileData.showReferenceColumn) {
                              baseData.push(test.reference);
                            }
                            baseData.push(
                              test.value,
                              test.unit
                            );
                            if (!profileData.hideStatusColumn) {
                              baseData.push(test.status);
                            }
                            return baseData;
                          })}
                          onRowClick={(rowIndex: number) => {
                            const test = profileData.testsData[rowIndex];
                            toggleTestDetails(test.fullId);
                          }}
                          expandedRows={profileData.testsData.map((test: any) => 
                            expandedTestDetails.has(test.fullId)
                          )}
                          renderExpandedContent={(rowIndex: number) => {
                            const test = profileData.testsData[rowIndex];
                            const testId = test.fullId; // Use the full ID to access individual test results
                            const testAlias = individualTestResults[testId]?.parameters?.find((p: any) => p.name === "Test Id")?.value?.replace(/^'|'$/g, "").trim();
                            
                            return (
                              <div className={styles["test-details"]}>
                                {individualTestResults[testId] ? (
                                  <div>
                                    {/* Error Messages for Failed Tests */}
                                    {test.status === 'failed' && individualTestResults[testId]?.error?.message && (
                                      <div className={styles["detail-section-error"]}>
                                        <h4 className={styles["detail-section-title-dark"]}>Error Details</h4>
                                        <pre className={styles["detail-section-message"]}>
                                          {individualTestResults[testId].error.message}
                                        </pre>
                                      </div>
                                    )}

                                    {/* Status Details for Skipped and Broken Tests */}
                                    {(test.status === 'skipped' || test.status === 'broken') && individualTestResults[testId]?.error?.message && (
                                      <div className={test.status === 'skipped' ? styles["detail-section-skipped"] : styles["detail-section-broken"]}>
                                        <h4 className={styles["detail-section-title-dark"]}>
                                          {test.status === 'skipped' ? 'Skip Details' : 'Broken Details'}
                                        </h4>
                                        {individualTestResults[testId].error.message && (
                                          <pre className={styles["detail-section-message"]}>
                                            {individualTestResults[testId].error.message}
                                          </pre>
                                        )}
                                      </div>
                                    )}

                                    {/* Overview Section */}
                                    <div className={styles["detail-section"]}>
                                      <h4 className={styles["detail-section-title"]}>Overview</h4>
                                      {(() => { const desc = getTestDescription(testId); return desc ? <DescriptionBlock text={desc} /> : null; })()}
                                      <div className={styles["detail-grid-3col"]}>
                                        <div>
                                          <span className={styles["detail-label"]}>History: </span>
                                          <span>{individualTestResults[testId]?.history?.length || "1"}</span>
                                        </div>
                                        <div>
                                          <span className={styles["detail-label"]}>Retries: </span>
                                          <span>{individualTestResults[testId]?.retries?.length || '0'}</span>
                                        </div>
                                        <div>
                                          <span className={styles["detail-label"]}>Duration: </span>
                                          <span>{individualTestResults[testId]?.duration 
                                            ? `${(individualTestResults[testId].duration / 1000).toFixed(2)}s`
                                            : 'N/A'}</span>
                                        </div>
                                      </div>
                                    </div>

                                    {/* Metadata Section */}
                                    <MetadataTable metricsData={testFullMetrics[testId]} />

                                    {/* Telemetry Section */}
                                    <TelemetrySection extendedMetadata={testFullMetrics[testId]?.extended_metadata} summaryMeta={summaryMeta} testId={testAlias} />

                                    {/* Attachments Section */}
                                    {(() => {
                                      const testData = individualTestResults[testId];
                                      const imageAttachments = testData?.attachments?.filter((attachment: any) => 
                                        attachment.link?.contentType === "image/png" || 
                                        attachment.link?.contentType === "image/jpeg"
                                      ) || [];
                                      const csvAttachments = testData?.attachments?.filter((attachment: any) => 
                                        attachment.link?.contentType === "text/csv"
                                      ) || [];
                                      
                                      const hasAttachments = imageAttachments.length > 0 || csvAttachments.length > 0;
                                      
                                      return hasAttachments ? (
                                        <div className={styles["attachment-section"]}>
                                          {/* Image Attachments */}
                                          {imageAttachments.length > 0 && (
                                            <div className={styles["attachment-container"]} style={{ marginBottom: csvAttachments.length > 0 ? '12px' : '0' }}>
                                              <h4 className={styles["detail-section-title"]}>Image Attachments</h4>
                                              <div className={styles["attachment-image-grid"]}>
                                                {imageAttachments.map((attachment: any, index: number) => {
                                                  return (
                                                    <div key={index} className={styles["attachment-image-item"]}>
                                                      <div className={styles["attachment-image-wrapper"]}>
                                                        <AttachmentImage 
                                                          attachment={attachment}
                                                          onError={() => console.warn('Failed to load attachment:', attachment.link?.id)}
                                                        />
                                                      </div>
                                                      <div className={styles["attachment-image-name"]}>
                                                        {attachment.name || attachment.link?.name || `Attachment ${index + 1}`}
                                                      </div>
                                                    </div>
                                                  );
                                                })}
                                              </div>
                                            </div>
                                          )}
                                          
                                          {/* CSV Attachments - Full Width */}
                                          {csvAttachments.length > 0 && (
                                            <div className={styles["csv-attachments"]}>
                                              {csvAttachments.map((attachment: any, index: number) => (
                                                <div key={index} className={styles["attachment-container"]} style={{ 
                                                  marginBottom: index < csvAttachments.length - 1 ? '12px' : '0'
                                                }}>
                                                  <AttachmentCSV 
                                                    attachment={attachment}
                                                    onError={() => console.warn('Failed to load CSV attachment:', attachment.link?.id)}
                                                  />
                                                </div>
                                              ))}
                                            </div>
                                          )}
                                        </div>
                                      ) : null;
                                    })()}
                                  </div>
                                ) : (
                                  <div style={{ padding: '16px', textAlign: 'center', color: '#666' }}>
                                    Loading test details...
                                  </div>
                                )}
                              </div>
                            );
                          }}
                        />
                      </div>
                    )}
                  </div>
                )
              ))}
              
              {searchTerm && filteredProfilesData.every(profile => profile.testsData.length === 0) && (
                <div className={styles["no-results"]}>
                  <p>No matching profiles or tests found for "{searchTerm}"</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
