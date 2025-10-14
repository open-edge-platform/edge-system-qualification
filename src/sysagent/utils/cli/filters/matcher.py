# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Filter matching utilities for test parameter filtering.

Provides functions to match filter values against test parameters and apply
filters to collections of test parameters.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def match_filter_value(param_value: Any, filter_value: Any) -> bool:
    """
    Check if a parameter value matches a filter value.
    
    Args:
        param_value: Value from test parameter
        filter_value: Value from filter expression
        
    Returns:
        True if values match, False otherwise
    """
    # Handle None values
    if param_value is None and filter_value is None:
        return True
    if param_value is None or filter_value is None:
        return False
    
    # Handle list parameters (e.g., devices)
    if isinstance(param_value, list):
        if isinstance(filter_value, list):
            # Check if any element in filter_value is in param_value
            return any(item in param_value for item in filter_value)
        else:
            # Check if filter_value is in param_value list
            return filter_value in param_value
    elif isinstance(filter_value, list):
        # Check if param_value is in filter_value list
        return param_value in filter_value
    
    # Handle string matching (case-insensitive)
    if isinstance(param_value, str) and isinstance(filter_value, str):
        return param_value.lower() == filter_value.lower()
    
    # Handle exact matching for other types
    return param_value == filter_value


def apply_test_filters(test_params: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply filters to test parameters.
    
    Args:
        test_params: List of test parameter configurations
        filters: Dictionary of filter criteria
        
    Returns:
        Filtered list of test parameters
    """
    if not filters:
        return test_params
    
    filtered_params = []
    
    for param_data in test_params:
        param_config = param_data.get('param', {})
        
        # Check if all filters match
        matches_all_filters = True
        for filter_key, filter_value in filters.items():
            param_value = param_config.get(filter_key)
            
            if not match_filter_value(param_value, filter_value):
                matches_all_filters = False
                logger.debug(f"Filter mismatch: {filter_key}={param_value} does not match filter value {filter_value}")
                break
        
        if matches_all_filters:
            filtered_params.append(param_data)
            logger.debug(f"Test parameter matched all filters: {param_config.get('test_id', 'unknown')} - {param_config.get('display_name', 'unknown')}")
    
    return filtered_params
