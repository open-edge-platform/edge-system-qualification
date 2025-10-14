# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Allure reporting utilities package.

Provides comprehensive Allure report generation capabilities split into
focused modules for better maintainability.
"""

# Import all the split modules
from .generator import generate_allure_report, install_allure_cli_from_repo
from .patch import apply_patch
from .utils import (
    _cleanup_old_final_reports,
    _generate_final_report_copy,
    _generate_short_timestamp,
    _get_app_name,
    _get_comprehensive_system_info_for_filename,
    _normalize_cpu_brand,
    _normalize_filename_component,
    _normalize_intel_gpu_name,
    update_allure_title_with_metrics,
)

# Import constants from the config loader
try:
    from sysagent.utils.config import get_allure_version

    ALLURE_VERSION = get_allure_version()
except ImportError:
    ALLURE_VERSION = "2.20.1"  # fallback version

ALLURE_DIR_NAME = "allure3"

__all__ = [
    # Main functions
    "generate_allure_report",
    "install_allure_cli_from_repo",
    "update_allure_title_with_metrics",
    # Patch functions
    "apply_patch",
    "_apply_patch_with_patch_command",
    "_apply_patch_with_git_apply",
    # Utility functions
    "_generate_final_report_copy",
    "_get_app_name",
    "_generate_short_timestamp",
    "_get_comprehensive_system_info_for_filename",
    "_cleanup_old_final_reports",
    "_normalize_cpu_brand",
    "_normalize_filename_component",
    "_normalize_intel_gpu_name",
    # Constants
    "ALLURE_VERSION",
    "ALLURE_DIR_NAME",
]
