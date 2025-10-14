# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CLI command implementations package.

Contains individual command implementations for the CLI,
organized by functionality to maintain clean separation of concerns.
"""

# Commands are imported dynamically to avoid circular imports
# Use get_command_function() to safely import and get command functions

def get_command_function(command_name: str):
    """
    Dynamically import and return a command function.
    
    Args:
        command_name: Name of the command to import
        
    Returns:
        The command function
    """
    if command_name == 'run_tests':
        from .run import run_tests
        return run_tests
    elif command_name == 'run_system_info':
        from .info import run_system_info
        return run_system_info
    elif command_name == 'list_available_items':
        from .list import list_available_items
        return list_available_items
    elif command_name == 'clean_data_dir':
        from .clean import clean_data_dir
        return clean_data_dir
    elif command_name == 'generate_report':
        from .report import generate_report
        return generate_report
    elif command_name == 'attach_logs':
        from .attach import attach_logs
        return attach_logs
    elif command_name == 'attach_summaries':
        from .attach import attach_summaries
        return attach_summaries
    elif command_name == 'attach_system':
        from .attach import attach_system
        return attach_system
    elif command_name == 'generate_summary':
        from .summary import generate_summary
        return generate_summary
    elif command_name == 'check_dependencies':
        from .deps import check_dependencies
        return check_dependencies
    elif command_name == 'list_dependencies':
        from .deps import list_dependencies
        return list_dependencies
    else:
        raise ValueError(f"Unknown command: {command_name}")


__all__ = ['get_command_function']
