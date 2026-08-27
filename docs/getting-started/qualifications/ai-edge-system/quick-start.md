# Quick Start

The AI Edge System qualification measures generative AI and vision AI performance against Intel® AI Edge Systems Qualification Metrics.

!!! info
    Refer to [Intel® ESQ for Intel® AI Edge Systems](https://www.intel.com/content/www/us/en/developer/articles/guide/esq-for-ai-edge-systems.html) for the official main qualification page.

Before you begin, confirm your system meets the [Requirements](requirements.md). No additional installation steps beyond the standard setup below are needed for this qualification.

## Installation

### 1. System Drivers

@import "../../../includes/quick-start/system-drivers.md"

### 2. Docker Engine

@import "../../../includes/quick-start/docker-engine.md"

### 3. Python Package Manager

@import "../../../includes/quick-start/python-package-manager.md"

### 4. System Setup

@import "../../../includes/quick-start/system-setup.md"

### 5. Intel® ESQ

@import "../../../includes/quick-start/esq-install.md"

## Qualification

### 1. Run

@import "../../../includes/quick-start/run.md"

Run this qualification directly:

```bash
esq run -t aes
```

### 2. Submit

After the run completes, review the generated report before submitting your results. The report path is printed at the end of the run, for example:

```text
Report available at: esq_data/reports/allure/esq_report_285k_260826_1430.html
```

Confirm that every applicable test passed and that the collected metrics reflect your system's expected performance. Then proceed to [submit your qualified system's test report](https://builders.intel.com/ecosystem-engagement/solution-hub/systems/qualification-form).

### 3. Uninstall

@import "../../../includes/quick-start/uninstall.md"

---

Continue to [Qualification Profiles](../index.md) →
