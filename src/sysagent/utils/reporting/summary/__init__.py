# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Results summary generation package.

Provides functionality to generate JSON summary results from Allure test results,
including durations, test status, and metadata for creating test summary tables.
"""

from .extractor import TestResultsExtractor
from .generator import CoreResultsSummaryGenerator
from .parser import AllureResultsParser
from .table import TestSummaryTableGenerator

# For backward compatibility, export the main classes
__all__ = [
    "AllureResultsParser",
    "CoreResultsSummaryGenerator",
    "TestResultsExtractor",
    "TestSummaryTableGenerator",
]
