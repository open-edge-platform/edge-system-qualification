# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Argument parser setup for CLI commands.

Contains the main argument parser configuration and all subparsers
for different CLI commands, keeping argument definitions centralized.
Uses dynamic app name instead of hardcoded values.
"""

import argparse

from sysagent.utils.config import get_dist_version
from sysagent.utils.config.config_loader import get_cli_aware_project_name


def get_cli_name() -> str:
    """Get the CLI command name dynamically based on the project name."""
    project_name = get_cli_aware_project_name()
    return project_name.lower()


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the main argument parser with all subcommands.

    Returns:
        argparse.ArgumentParser: Configured parser ready for argument parsing
    """
    cli_name = get_cli_name()

    parser = argparse.ArgumentParser(
        description="System Testing CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--version", action="version", version=f"{get_dist_version()}")

    parser.add_argument("--force", "-f", action="store_true", help="Force operations like reinstalling dependencies")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Display more detailed output with medium traceback"
    )

    parser.add_argument("--debug", "-d", action="store_true", help="Display debug output with full traceback")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run tests (qualification profiles by default, specific profile, or individual tests)",
        description=f"""
Run tests using one of four main patterns:

USAGE PATTERNS:
  1. Run qualification and vertical profiles with opt-out prompt (default):
     {cli_name} run

  2. Run all available profiles:
     {cli_name} run --all

  3. Run a specific profile:
     {cli_name} run -p PROFILE_NAME

  4. Run with filters:
     {cli_name} run -p PROFILE_NAME --filter test_id=T0069
     {cli_name} run -p PROFILE_NAME --filter display_name="CPU Test"
     {cli_name} run -p PROFILE_NAME --filter test_id=T0069 --filter devices=cpu

EXAMPLES:
  {cli_name} run                                      # Run qualification + vertical profiles (with opt-out prompt)
  {cli_name} run --force                              # Skip prompt, use default (include vertical profiles)
  {cli_name} run --all                                # Run all profile types
  {cli_name} run -p profile.suite.ai.vision           # Run specific profile

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
        help=("Run all profile types. By default, qualification and vertical profiles are run with an opt-out prompt."),
    )
    execution_group.add_argument(
        "--profile", "-p", metavar="PROFILE_NAME", help="Run a specific profile (e.g., profile.suite.ai-vision)"
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
        help="Filter tests by parameter values (e.g., --filter test_id=T0069 --filter display_name='CPU Test'). Can be used multiple times for multiple filters.",
    )

    # Add common options to run command
    run_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Display more detailed output with medium traceback"
    )
    run_parser.add_argument("--debug", "-d", action="store_true", help="Display debug output with full traceback")
    run_parser.add_argument(
        "--force", "-f", action="store_true", help="Skip interactive prompts and use default behavior"
    )

    # System Info command
    info_parser = subparsers.add_parser("info", help="Show system information with hardware and software details")
    info_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Display more detailed output with medium traceback"
    )
    info_parser.add_argument("--debug", "-d", action="store_true", help="Display debug output with full traceback")

    # List command
    list_parser = subparsers.add_parser("list", help="List available profiles and tests")
    list_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Display more detailed output with medium traceback"
    )
    list_parser.add_argument("--debug", "-d", action="store_true", help="Display debug output with full traceback")

    # Clean command
    clean_parser = subparsers.add_parser(
        "clean", help="Clean test data directories, logs, Allure results, reports, and history"
    )

    # Add common options to clean command
    clean_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Display more detailed output with medium traceback"
    )
    clean_parser.add_argument("--debug", "-d", action="store_true", help="Display debug output with full traceback")

    # Group for additional cleaning (in addition to default results/logs cleaning)
    additional_group = clean_parser.add_argument_group("ADDITIONAL CLEANING (with results)")
    additional_group.add_argument(
        "--cache", "-c", action="store_true", help="Clean test cache directory as well (test result cache files)"
    )
    additional_group.add_argument(
        "--thirdparty",
        "-t",
        action="store_true",
        help="Clean entire thirdparty directory (includes Node.js and Allure installations)",
    )
    additional_group.add_argument(
        "--data", action="store_true", help="Clean entire data directory (models, videos, test data, etc.)"
    )
    additional_group.add_argument(
        "--all", "-a", action="store_true", help="Clean all directories (equivalent to --cache --thirdparty --data)"
    )

    # Group for specific directory cleaning only (without touching results/logs)
    specific_group = clean_parser.add_argument_group("SPECIFIC DIRECTORY CLEANING (only)")
    specific_group.add_argument(
        "--cache-only", action="store_true", help="Clean only the cache directory (do not clean results/logs)"
    )
    specific_group.add_argument(
        "--thirdparty-only", action="store_true", help="Clean only the thirdparty directory (do not clean results/logs)"
    )
    specific_group.add_argument(
        "--data-only", action="store_true", help="Clean only the data directory (do not clean results/logs)"
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate Allure report from test results")
    report_parser.add_argument(
        "--report-name", "-n", help="Custom name for the Allure report (default: 'Edge System Qualification Report')"
    )
    report_parser.add_argument(
        "--report-version",
        help="Version string to use for allureVersion in the Allure configuration (default: software version)",
    )

    # Add common options to report command
    report_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Display more detailed output with medium traceback"
    )
    report_parser.add_argument("--debug", "-d", action="store_true", help="Display debug output with full traceback")

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Generate JSON summary from existing test results")
    summary_parser.add_argument("--output-file", "-o", help="Custom output filename for the summary JSON")

    # Add common options to summary command
    summary_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Display more detailed output with medium traceback"
    )
    summary_parser.add_argument("--debug", "-d", action="store_true", help="Display debug output with full traceback")

    # Dependencies command
    deps_parser = subparsers.add_parser(
        "deps",
        help="Check and manage system dependencies",
        description=f"""
Check system dependencies and get installation instructions.

EXAMPLES:
  {cli_name} deps                           # Check all dependencies
  {cli_name} deps --list                    # List all available dependencies
  {cli_name} deps --check curl git          # Check specific dependencies
  {cli_name} deps --instructions            # Show installation instructions
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    deps_group = deps_parser.add_mutually_exclusive_group()
    deps_group.add_argument(
        "--check", nargs="*", metavar="DEPENDENCY_NAME", help="Check specific dependencies (default: check all)"
    )
    deps_group.add_argument("--list", action="store_true", help="List all available dependencies and their information")

    deps_parser.add_argument(
        "--instructions", "-i", action="store_true", help="Show installation instructions for missing dependencies"
    )

    # Add common options to deps command
    deps_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Display more detailed output with medium traceback"
    )
    deps_parser.add_argument("--debug", "-d", action="store_true", help="Display debug output with full traceback")

    return parser
