# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Modular Command Line Interface for the core framework.

This is the main CLI entry point that has been refactored for modularity.
Most functionality has been moved to specialized modules in utils.cli.commands.
"""

import atexit
import sys

from sysagent.utils.cli.commands import get_command_function
from sysagent.utils.cli.parsers import create_argument_parser
from sysagent.utils.core.dependencies import get_dependency_manager
from sysagent.utils.infrastructure import download_github_repo
from sysagent.utils.logging import cleanup_logging, configure_logging


def check_and_report_dependencies() -> bool:
    """
    Check system dependencies and report issues with actionable instructions.
    Uses the modular dependency management system.

    Returns:
        bool: True if all dependencies are satisfied, False otherwise
    """

    manager = get_dependency_manager()
    return manager.check_and_report_dependencies()


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    # Import at function level to avoid circular imports
    import os
    import sys

    # Set CLI package context for profile discovery
    # This ensures each CLI (esq, test-esq, sysagent) only shows its own profiles
    if len(sys.argv) > 0:
        cli_command = os.path.basename(sys.argv[0])
        # Map CLI command to package name
        if cli_command in ["esq", "test-esq", "sysagent"]:
            os.environ["SYSAGENT_CLI_PACKAGE"] = cli_command

    # Create and parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle verbose/debug flags - check both global and subcommand usage
    command_index = sys.argv.index(args.command) if args.command and args.command in sys.argv else len(sys.argv)
    global_args = sys.argv[:command_index]

    global_verbose = "-v" in global_args or "--verbose" in global_args
    global_debug = "-d" in global_args or "--debug" in global_args

    # Merge global and subcommand flags
    verbose = global_verbose or getattr(args, "verbose", False)
    debug = global_debug or getattr(args, "debug", False)

    # Configure logging based on command line options
    configure_logging(verbose=verbose, debug=debug)

    # Skip dependency verification for deps command or commands that don't need it
    skip_deps = args.command == "deps" or not hasattr(args, "command")

    # Skip GitHub repo downloads for commands that don't need them or clean them
    skip_downloads = args.command in ["deps", "clean"] or not hasattr(args, "command")

    if not skip_deps:
        # Check dependencies with detailed reporting
        if not check_and_report_dependencies():
            return 1

    # Configure the dependencies and environment (skip for deps and clean commands)
    if not skip_downloads:
        download_github_repo()

    # Set up a signal handler to clean up log handlers on exit
    atexit.register(cleanup_logging)

    # Route to appropriate command handler
    try:
        if args.command == "run":
            run_tests = get_command_function("run_tests")
            return run_tests(
                profile_name=args.profile,
                suite_name=args.suite,
                sub_suite_name=args.sub_suite,
                test_name=args.test,
                verbose=verbose,
                debug=debug,
                suites_dir=args.suites_dir,
                skip_system_check=args.skip_system_check,
                no_cache=args.no_cache,
                filters=args.filter,
                run_all_profiles=args.all,
                extra_args=[],  # No extra args from command line
            )
        elif args.command == "info":
            run_system_info = get_command_function("run_system_info")
            return run_system_info(verbose=verbose, debug=debug)
        elif args.command == "list":
            list_available_items = get_command_function("list_available_items")
            return list_available_items(verbose=verbose, debug=debug)
        elif args.command == "clean":
            clean_data_dir = get_command_function("clean_data_dir")
            return clean_data_dir(
                clean_cache=args.cache,
                clean_thirdparty=args.thirdparty,
                clean_data=args.data,
                clean_all=args.all,
                cache_only=args.cache_only,
                thirdparty_only=args.thirdparty_only,
                data_only=args.data_only,
                verbose=verbose,
                debug=debug,
            )
        elif args.command == "report":
            generate_report = get_command_function("generate_report")
            return generate_report(
                report_name=args.report_name,
                report_version=args.report_version,
                force=args.force,
                debug=debug,
            )
        elif args.command == "summary":
            generate_summary = get_command_function("generate_summary")
            return generate_summary(output_file=args.output_file, verbose=verbose, debug=debug)
        elif args.command == "deps":
            if args.list:
                list_dependencies = get_command_function("list_dependencies")
                return list_dependencies(verbose=verbose, debug=debug)
            else:
                check_dependencies = get_command_function("check_dependencies")
                dependency_names = args.check if args.check is not None else None
                return check_dependencies(
                    dependency_names=dependency_names,
                    show_instructions=args.instructions,
                    verbose=verbose,
                    debug=debug,
                )
        else:
            parser.print_help()
            return 0
    except Exception as e:
        # Import logging here to avoid potential issues during startup
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Command execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
