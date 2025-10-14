# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Dependency management command for the CLI.

Provides commands to check, validate, and get installation instructions
for system dependencies using the improved modular dependency system.
"""

import logging
from typing import List

from sysagent.utils.core.dependencies import DependencyStatus, get_dependency_manager
from sysagent.utils.logging import setup_command_logging

logger = logging.getLogger(__name__)


def check_dependencies(
    dependency_names: List[str] = None, show_instructions: bool = False, verbose: bool = False, debug: bool = False
) -> int:
    """
    Check system dependencies and optionally show installation instructions.

    Args:
        dependency_names: Specific dependencies to check (default: all)
        show_instructions: Whether to show installation instructions for missing deps
        verbose: Whether to show detailed output
        debug: Whether to show debug level logs

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        # Set up logging for this operation
        setup_command_logging("deps", verbose=verbose, debug=debug)

        # Get dependency manager
        dm = get_dependency_manager()

        if dependency_names:
            # Check specific dependencies
            logger.info(f"Checking specific dependencies: {', '.join(dependency_names)}")
            results = {}
            for name in dependency_names:
                if name in dm.config.dependencies:
                    results[name] = dm.check_dependency(name)
                else:
                    from sysagent.utils.core.dependencies.schema import DependencyCheckResult

                    results[name] = DependencyCheckResult(
                        name=name, status=DependencyStatus.ERROR, message=f"Unknown dependency: {name}"
                    )
        else:
            # Check all dependencies
            logger.info("Checking all system dependencies...")
            results = dm.check_all_dependencies()

        # Display results with improved status labels
        print("\nDEPENDENCY CHECK RESULTS")
        print("=" * 50)

        installed_count = 0
        missing_count = 0
        manual_count = 0

        for name, result in results.items():
            # Determine status icon and color based on result status
            if result.status == DependencyStatus.INSTALLED:
                status_icon = "[OK]"
                installed_count += 1
            elif result.status == DependencyStatus.MANUAL_REQUIRED:
                status_icon = "[MANUAL]"
                manual_count += 1
            elif result.status == DependencyStatus.MISSING:
                status_icon = "[MISSING]"
                missing_count += 1
            elif result.status == DependencyStatus.NEEDS_CONFIGURATION:
                status_icon = "[CONFIG]"
                missing_count += 1
            elif result.status == DependencyStatus.ERROR:
                status_icon = "[ERROR]"
                missing_count += 1
            else:
                status_icon = "[UNKNOWN]"
                missing_count += 1

            print(f"{status_icon} {name}: {result.message}")

            # Show additional details if available
            if result.details and verbose:
                print(f"    Details: {result.details}")

        print("\n" + "=" * 50)
        summary_parts = []
        if installed_count > 0:
            summary_parts.append(f"{installed_count} installed")
        if manual_count > 0:
            summary_parts.append(f"{manual_count} manual setup")
        if missing_count > 0:
            summary_parts.append(f"{missing_count} missing")

        print(f"Summary: {', '.join(summary_parts)}")

        # Show installation instructions if requested
        if show_instructions:
            if dependency_names:
                # Show instructions for specified dependencies
                print("\n")
                instructions = dm.generate_installation_instructions(dependency_names)
                print(instructions)
            else:
                # Show instructions for all dependencies
                all_deps = list(dm.config.dependencies.keys())
                print("\n")
                instructions = dm.generate_installation_instructions(all_deps)
                print(instructions)
        elif missing_count > 0 and not dependency_names:
            # Show installation instructions for missing dependencies only
            missing_deps = [
                name
                for name, result in results.items()
                if result.status
                in [DependencyStatus.MISSING, DependencyStatus.NEEDS_CONFIGURATION, DependencyStatus.ERROR]
            ]

            if missing_deps:
                print("\n")
                instructions = dm.generate_installation_instructions(missing_deps)
                print(instructions)

        # Exit code: 0 if no missing dependencies, 1 if any missing
        return 0 if missing_count == 0 else 1

    except Exception as e:
        logger.error(f"Error checking dependencies: {e}", exc_info=debug)
        return 1


def list_dependencies(verbose: bool = False, debug: bool = False) -> int:
    """
    List all available dependencies and their status.

    Args:
        verbose: Whether to show detailed output
        debug: Whether to show debug level logs

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        # Set up logging for this operation
        setup_command_logging("deps", verbose=verbose, debug=debug)

        # Get dependency manager
        dm = get_dependency_manager()

        # Check all dependencies
        results = dm.check_all_dependencies()

        print("\nAVAILABLE DEPENDENCIES")
        print("=" * 50)

        for name, result in results.items():
            dependency = dm.config.dependencies[name]

            # Determine status icon
            if result.status == DependencyStatus.INSTALLED:
                status_icon = "✓"
            elif result.status == DependencyStatus.MANUAL_REQUIRED:
                status_icon = "⚠"
            else:
                status_icon = "✗"

            print(f"{status_icon} {name}")
            print(f"    Description: {dependency.description}")
            print(f"    Status: {result.message}")
            print(f"    Required: {'Yes' if dependency.required else 'No'}")
            print(f"    Priority: {dependency.priority}")

            if dependency.package_source:
                print(f"    Package: {dependency.package_source}")

            if verbose and result.details:
                print(f"    Details: {result.details}")

            print()

        print(f"Total dependencies: {len(results)}")

        return 0

    except Exception as e:
        logger.error(f"Error listing dependencies: {e}", exc_info=debug)
        return 1


def list_dependencies(verbose: bool = False, debug: bool = False) -> int:
    """
    List all available dependencies and their information.

    Args:
        verbose: Whether to show detailed information
        debug: Whether to show debug level logs

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        # Set up logging for this operation
        setup_command_logging("deps", verbose=verbose, debug=debug)

        # Get dependency manager
        dm = get_dependency_manager()

        print("\nAVAILABLE DEPENDENCIES")
        print("=" * 50)

        # Sort by priority
        sorted_deps = sorted(dm.config.dependencies.items(), key=lambda x: x[1].priority)

        for name, dependency in sorted_deps:
            required_text = "Required" if dependency.required else "Optional"
            print(f"\n{name.upper()} ({required_text})")
            print(f"  Description: {dependency.description}")
            print(f"  Priority: {dependency.priority}")

            if dependency.dependencies:
                print(f"  Depends on: {', '.join(dependency.dependencies)}")

        # Show groups
        if dm.config.groups:
            print("\n\nDEPENDENCY GROUPS")
            print("=" * 50)

            for group_name, group in dm.config.groups.items():
                optional_text = " (Optional)" if group.optional else ""
                print(f"\n{group.name.upper()}{optional_text}")
                print(f"  Description: {group.description}")
                print(f"  Dependencies: {', '.join(group.dependencies)}")

        return 0

    except Exception as e:
        logger.error(f"Error listing dependencies: {e}", exc_info=debug)
        return 1
