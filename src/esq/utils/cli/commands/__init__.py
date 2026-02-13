# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
ESQ-specific CLI command implementations.

This package contains ESQ-specific command overrides that extend
or replace the default sysagent commands with domain-specific logic.
"""


def get_command_function(command_name: str):
    """
    Get ESQ-specific command function, or None if not overridden.

    Args:
        command_name: Name of the command to import

    Returns:
        The command function, or None if command is not overridden
    """
    if command_name == "run_tests":
        from .run import run_tests

        return run_tests
    else:
        # Command not overridden by ESQ, fall back to sysagent
        return None


__all__ = ["get_command_function"]
