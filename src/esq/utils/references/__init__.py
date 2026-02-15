# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
References Utilities Package.

This package provides utilities for handling verified reference data from profile
configurations, including filtering by CPU generation and creating Allure attachments.
"""

from .data_handler import (
    add_reference_data_to_result,
    attach_reference_data_to_allure,
    convert_reference_data_to_csv,
    filter_reference_data_by_generation,
)

__all__ = [
    "filter_reference_data_by_generation",
    "convert_reference_data_to_csv",
    "attach_reference_data_to_allure",
    "add_reference_data_to_result",
]
