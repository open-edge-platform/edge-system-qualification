# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
ESQ-specific argument parser configuration.

Extends the generic sysagent parser by overriding only the 'run' command
with ESQ-specific options like --all and --qualification-only flags.
All other commands (info, list, clean, deps, etc.) inherit from sysagent.
"""

import argparse
import logging

from sysagent.utils.config.config_loader import get_cli_aware_project_name

logger = logging.getLogger(__name__)


def get_cli_name() -> str:
    """Get the CLI command name dynamically based on the project name."""
    project_name = get_cli_aware_project_name()
    return project_name.lower()


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create ESQ-specific argument parser by extending the base sysagent parser.

    This function:
    1. Gets the base parser from sysagent (with all standard commands)
    2. Removes the generic 'run' subparser
    3. Adds ESQ-specific 'run' subparser with custom options
    4. Keeps all other commands (info, list, clean, deps, report, summary) unchanged

    Returns:
        argparse.ArgumentParser: Parser with ESQ-specific run command
    """
    # Import base parser creator from sysagent
    from sysagent.utils.cli.parsers import create_argument_parser as create_base_parser

    # Get the base parser with all standard commands
    parser = create_base_parser()
    cli_name = get_cli_name()

    # Access the subparsers to modify the 'run' command
    # Note: This requires accessing the _subparsers attribute which contains all subparsers
    subparsers_action = None
    for action in parser._subparsers._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers_action = action
            break

    if subparsers_action is None:
        logger.error("Could not find subparsers in base parser")
        return parser

    # Remove the generic 'run' subparser
    if "run" in subparsers_action.choices:
        del subparsers_action.choices["run"]

    # Add ESQ-specific 'run' subparser with custom options
    run_parser = subparsers_action.add_parser(
        "run",
        help="Run tests with system validation",
        description=f"""
Run Intel® Edge System Qualification (ESQ).

USAGE PATTERNS:
  1. Default run (interactive prompts):
     {cli_name} run
     • Run qualification and vertical profiles
     • Options to skip vertical profiles
     • Exits if system doesn't meet requirements and vertical profile is skipped

  2. Run all profiles:
     {cli_name} run --all
     • Skips qualification profiles if system not supported

  3. Run qualification profiles only:
     {cli_name} run --qualification-only
     • Runs only qualification profiles
     • Exits if system doesn't meet requirements

  4. Run specific profile:
     {cli_name} run -p PROFILE_NAME
     • Runs the specified profile with validation
     
  5. Run with filters:
     {cli_name} run -p PROFILE_NAME --filter test_id=T0069

EXAMPLES:
  {cli_name} run                                                    # Interactive mode with prompts
  {cli_name} run --all                                              # All profiles
  {cli_name} run --qualification-only                               # Qualification profiles only
  {cli_name} run -p profile.qualification.ai-edge-system            # Specific profile
  {cli_name} run -p profile.suite.ai.vision --filter test_id=T0001  # Filtered test

HARDWARE REQUIREMENTS:
  Refer to the documentation for supported hardware and system requirements.

For available profiles, use: {cli_name} list
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Main execution options (grouped for clarity)
    execution_group = run_parser.add_argument_group("EXECUTION OPTIONS")
    execution_group.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Run all profile types after system validation (skips qualification if unsupported)",
    )
    execution_group.add_argument(
        "--qualification-only",
        "-qo",
        action="store_true",
        help="Run qualification profiles only (exits if system requirements not met)",
    )
    execution_group.add_argument(
        "--profile",
        "-p",
        metavar="PROFILE_NAME",
        help="Run a specific profile (e.g., profile.qualification.ai-edge-system)",
    )

    # Individual test selection (grouped together)
    test_group = run_parser.add_argument_group("INDIVIDUAL TEST SELECTION")
    test_group.add_argument("--suite", "-s", metavar="SUITE_NAME", help="Test suite name (e.g., 'ai')")
    test_group.add_argument(
        "--sub-suite", "-ss", metavar="SUB_SUITE_NAME", help="Test sub-suite name (requires --suite, e.g., 'vision')"
    )
    test_group.add_argument(
        "--test", "-t", metavar="TEST_NAME", help="Test name (requires --sub-suite, e.g., 'test_dlstreamer')"
    )

    # Advanced options (grouped separately)
    advanced_group = run_parser.add_argument_group("ADVANCED OPTIONS")
    advanced_group.add_argument(
        "--suites-dir", metavar="DIRECTORY", help="Custom directory containing test suites (overrides default)"
    )
    advanced_group.add_argument(
        "--skip-system-check", "-ssc", action="store_true", help="Skip system validation before running tests"
    )
    advanced_group.add_argument("--no-cache", "-nc", action="store_true", help="Run tests without using cached results")
    advanced_group.add_argument(
        "--filter",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "Filter tests by parameter values (e.g., --filter test_id=T0069 --filter display_name='CPU Test'). "
            "Can be used multiple times for multiple filters."
        ),
    )
    advanced_group.add_argument(
        "--set-prompt",
        action="append",
        metavar="PROMPT=ANSWER",
        help=(
            "Override specific prompt answers for CI automation "
            "(e.g., --set-prompt run_configuration=y --set-prompt system_validation=n). "
            "Use for non-interactive execution."
        ),
    )

    # Add common options to run command
    run_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Display more detailed output with medium traceback"
    )
    run_parser.add_argument("--debug", "-d", action="store_true", help="Display debug output with full traceback")
    run_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip interactive prompts. If system unsupported: continue with suite/vertical profiles (skip qualification)",
    )
    run_parser.add_argument(
        "--no-mask",
        "-nm",
        action="store_true",
        help="Disable masking of data (IP addresses, MAC addresses, serial numbers)",
    )

    # Return the modified parser (all other commands inherited from sysagent)
    return parser
