# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Helper utilities for CLI operations.

Contains utility functions that support various CLI commands
but don't belong to any specific command module.
"""
from typing import List, Tuple


def get_test_names_from_profile(profile_configs) -> List[Tuple[str, str, str]]:
    """
    Extract test names from a profile configs. Works for both qualification and suite profiles.
    
    Args:
        profile_configs: The profile configurations
        
    Returns:
        List[Tuple[str, str, str]]: List of (suite_name, sub_suite_name, test_name) tuples
    """
    test_names = []
    
    for suite in profile_configs.get("suites", []):
        suite_name = suite.get("name")
        
        for sub_suite in suite.get("sub_suites", []):
            sub_suite_name = sub_suite.get("name")
            tests_config = sub_suite.get("tests", {})
            
            # Tests should be a dictionary with test names as keys
            for test_name in tests_config.keys():
                test_names.append((suite_name, sub_suite_name, test_name))
    
    return test_names
