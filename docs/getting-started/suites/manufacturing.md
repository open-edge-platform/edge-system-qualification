# Manufacturing

The Manufacturing test suite focuses on vertical use cases specific to industrial and manufacturing environments.

## Overview

The Manufacturing test suite targets critical industrial use cases where AI delivers measurable value. This suite currently focuses on:

- **Pallet Defect Detection**: Automated identification and classification of defects in pallets, improving quality control and reducing manual inspection effort.
- **Weld Porosity Detection**: AI-driven analysis of welds to detect porosity and other anomalies, supporting predictive maintenance and ensuring weld integrity in manufacturing processes.

## Running the Test Suite

To run the Manufacturing AI test suite, use the following command with the manufacturing profile:

```bash
esq run --profile profile.vertical.manufacturing
```

This command executes all tests designed to validate manufacturing-specific AI workloads and provides performance metrics relevant to industrial deployment scenarios.

!!! tip "Verbose Output"
    Use the `--verbose` option to see detailed information while running tests:
    
---

Need to run the retail test suite? Check out the [Retail test suite](retail.md) →
