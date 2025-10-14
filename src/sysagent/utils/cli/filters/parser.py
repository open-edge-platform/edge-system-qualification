# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Filter expression parser for Docker-style test filtering.

Provides functionality to parse command-line filter expressions into structured
filter dictionaries that can be applied to test parameters.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def parse_filters(filters: List[str]) -> Dict[str, Any]:
    """
    Parse filter expressions from command line arguments.
    
    Args:
        filters: List of filter expressions in format "key=value"
        
    Returns:
        Dict mapping filter keys to values
        
    Raises:
        ValueError: If filter format is invalid
    """
    parsed_filters = {}
    
    if not filters:
        return parsed_filters
    
    for filter_expr in filters:
        if '=' not in filter_expr:
            raise ValueError(f"Invalid filter format: '{filter_expr}'. Expected format: 'key=value'")
        
        key, value = filter_expr.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        if not key:
            raise ValueError(f"Empty filter key in expression: '{filter_expr}'")
        
        # Handle special values
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)
        # Remove quotes if present
        elif value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        
        # Handle list values (comma-separated)
        if isinstance(value, str) and ',' in value:
            value = [v.strip() for v in value.split(',')]
        
        parsed_filters[key] = value
        logger.debug(f"Parsed filter: {key} = {value} (type: {type(value)})")
    
    return parsed_filters
