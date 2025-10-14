# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CLI utilities package for the core framework.

This package contains modular CLI command implementations,
handlers, and utilities to keep the main CLI file lightweight.
"""

# Import functions that don't cause circular dependencies
from .handlers import handle_interrupt
from .parsers import create_argument_parser
from .helpers import get_test_names_from_profile

# Commands will be imported dynamically as needed to avoid circular imports

__all__ = [
    'handle_interrupt',
    'create_argument_parser', 
    'get_test_names_from_profile'
]
