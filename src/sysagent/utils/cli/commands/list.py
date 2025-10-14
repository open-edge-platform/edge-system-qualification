# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
List command implementation.

Handles listing all available profiles with their descriptions,
types, and optionally their test structures.
"""
import os
import logging
from collections import defaultdict

from sysagent.utils.config import list_profiles
from sysagent.utils.logging import setup_command_logging
from sysagent.utils.cli.helpers import get_test_names_from_profile

logger = logging.getLogger(__name__)


def list_available_items(verbose: bool = False, debug: bool = False) -> int:
    """
    List available profiles.

    Lists all profiles by reading the 'name' field from each YAML configuration file.
    The 'name' field must contain the profile identifier (e.g., "profile.qualification.edge-connectivity-standard").

    Args:
        verbose: Whether to show more detailed output
        debug: Whether to show debug level logs

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        # Set up logging for this operation
        setup_command_logging("list", verbose=verbose, debug=debug)

        print("Available Test Suite Profiles:")

        # Use list_profiles to get all profile files (excluding examples)
        profiles_dict = list_profiles(include_examples=False)

        all_profiles = []
        for profile_type, items in profiles_dict.items():
            for item in items:
                configs = item.get("configs")
                path = item.get("path")
                if not configs or not configs.get("name"):
                    logger.warning(f"Missing profile identifier in name field of {path}")
                    continue
                all_profiles.append({
                    "name": configs["name"],
                    "description": configs.get("description", "No description"),
                    "type": profile_type,
                    "configs": configs,
                    "file_path": path,
                    "filename": os.path.basename(path)
                })

        # Sort profiles by name
        all_profiles.sort(key=lambda x: x["name"])

        # Display all profiles in a standardized, indented summary format
        if all_profiles:
            for idx, profile in enumerate(all_profiles, 1):
                print(f"\nProfile {idx}:")
                print(f"  Name: {profile['name']}")
                print(f"  Type: {profile['type']}")
                print(f"  Description: {profile['description']}")

                # Extract and display test information - unified for all profile types
                if verbose:
                    test_names = get_test_names_from_profile(profile["configs"])
                    if test_names:
                        print(f"  Tests:")
                        suite_map = defaultdict(lambda: defaultdict(list))
                        for suite_name, sub_suite_name, test_name in test_names:
                            suite_map[suite_name][sub_suite_name].append(test_name)
                        for suite_name in sorted(suite_map):
                            print(f"    Suite: {suite_name}")
                            for sub_suite_name in sorted(suite_map[suite_name]):
                                print(f"      Sub-suite: {sub_suite_name}")
                                for test_name in sorted(suite_map[suite_name][sub_suite_name]):
                                    print(f"        - {test_name}")
                    else:
                        print("  Tests: None listed.")
        else:
            print("\nNo profiles found. Please check the configuration directories.")

        if not verbose:
            print("\nUse the --verbose flag to see detailed profiles with test suites.")
        return 0

    except Exception as e:
        logger.error(f"Error listing available items: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1
