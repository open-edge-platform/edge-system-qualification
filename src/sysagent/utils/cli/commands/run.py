# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test run command implementation.

Handles running tests based on profiles, suites, or specific test cases
with comprehensive validation and reporting capabilities.
"""

import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List

from sysagent.utils.cli.filters import parse_filters
from sysagent.utils.cli.handlers import handle_interrupt
from sysagent.utils.config import filter_profile_by_tier, get_suite_directory, list_profiles, setup_data_dir
from sysagent.utils.core import shared_state
from sysagent.utils.logging import setup_command_logging
from sysagent.utils.reporting import CoreResultsSummaryGenerator, TestSummaryTableGenerator
from sysagent.utils.testing import (
    add_test_paths_to_args,
    cleanup_pytest_cache,
    create_profile_pytest_args,
    create_pytest_args,
    run_pytest,
)

# Import will be done dynamically to avoid circular imports

logger = logging.getLogger(__name__)


def run_tests(
    profile_name: str = None,
    suite_name: str = None,
    sub_suite_name: str = None,
    test_name: str = None,
    verbose: bool = False,
    debug: bool = False,
    suites_dir: str = None,
    skip_system_check: bool = False,
    no_cache: bool = False,
    filters: List[str] = None,
    force: bool = False,
    no_mask: bool = False,
    set_prompt: List[str] = None,
    extra_args: List[str] = None,
    run_all_profiles: bool = None,
    qualification_only: bool = None,
) -> int:
    """
    Run tests based on profiles, suites, or specific test cases.

    Generic sysagent behavior:
    - Default run (no flags): Runs all profile types (qualifications, suites, verticals)
    - Specific profile (-p): Runs specified profile
    - Suite/test run: Runs specified suite/test

    Args:
        profile_name: Profile name to run
        suite_name: Name of the suite to run
        sub_suite_name: Name of the sub-suite to run (requires suite_name)
        test_name: Name of the test to run (requires sub_suite_name)
        verbose: Whether to enable medium traceback (--tb=short)
        debug: Whether to enable full traceback and debug logs (--tb=long)
        suites_dir: Custom directory containing test suites (overrides default location)
        skip_system_check: Whether to skip system requirement validation
        no_cache: Whether to run tests without using cached results
        filters: List of filter expressions in format "key=value" to filter tests
        force: Whether to skip interactive prompts (ignored in generic sysagent)
        no_mask: Whether to disable masking of data in system information
        set_prompt: List of prompt overrides (ignored in generic sysagent)
        extra_args: Additional pytest arguments to pass
        run_all_profiles: Whether to run all profile types. Ignored in generic sysagent.
        qualification_only: Whether to run only qualification profiles. Ignored in generic sysagent.

    Returns:
        int: Exit code (0 for success, non-zero for failure)

    Execution Flow:
        1. If profile_name specified: Run that profile
        2. If suite_name specified: Run that suite
        3. Otherwise: Run all profiles (no filtering, no prompts)
    """
    # Parse prompt overrides from CLI
    prompt_overrides = {}
    if set_prompt:
        for override in set_prompt:
            if "=" in override:
                prompt_name, answer = override.split("=", 1)
                prompt_overrides[prompt_name.strip()] = answer.strip()
                logger.info(f"CLI prompt override: {prompt_name.strip()}={answer.strip()}")
            else:
                logger.warning(f"Invalid --set-prompt format: {override} (expected PROMPT=ANSWER)")

    if no_mask:
        os.environ["CORE_MASK_DATA"] = "false"

    # Reset the interrupt flags at the start using the shared_state module
    shared_state.INTERRUPT_OCCURRED = False
    shared_state.INTERRUPT_SIGNAL = None
    shared_state.INTERRUPT_SIGNAL_NAME = "Unknown"

    # Register the global interrupt handler
    original_sigint_handler = signal.signal(signal.SIGINT, handle_interrupt)
    if "ACTIVE_PROFILE" in os.environ:
        del os.environ["ACTIVE_PROFILE"]

    if "ACTIVE_PROFILE_HIGHEST_TIER" in os.environ:
        del os.environ["ACTIVE_PROFILE_HIGHEST_TIER"]

    if sub_suite_name and not suite_name:
        logger.error("Error: --sub-suite option requires --suite option to be specified")
        return 1

    if test_name and not sub_suite_name:
        logger.error("Error: --test option requires --sub-suite option to be specified")
        return 1

    # Parse and validate filters
    parsed_filters = {}
    if filters:
        try:
            parsed_filters = parse_filters(filters)
            logger.info(f"Applying test filters: {parsed_filters}")
        except ValueError as e:
            logger.error(f"Invalid filter format: {e}")
            return 1

        # Filters can only be used with profile-based execution
        if not profile_name:
            logger.error("Error: --filter option can only be used with --profile option")
            return 1

    data_dir = setup_data_dir()

    if suites_dir:
        if not os.path.isdir(suites_dir):
            logger.error(f"Custom suites directory does not exist: {suites_dir}")
            return 1
        os.environ["CORE_SUITES_PATH"] = os.path.abspath(suites_dir)
        logger.info(f"Using custom suites directory: {suites_dir}")

    setup_command_logging("run", verbose=verbose, debug=debug, data_dir=data_dir)
    os.environ["CORE_DATA_DIR"] = data_dir

    if no_cache:
        os.environ["CORE_NO_CACHE"] = "1"
        logger.info("Running tests with no cache enabled")

    if extra_args is None:
        extra_args = []

    pytest_args = create_pytest_args(data_dir, verbose, debug, extra_args)

    result_code = 0
    tests_ran = False
    interrupt_occurred = False

    try:
        # Option 1: If a profile name is provided, run that specific profile (no prompt)
        if profile_name:
            result_code, tests_ran = _run_profile_tests(
                profile_name, pytest_args, skip_system_check, data_dir, verbose, debug, parsed_filters, force
            )

        # Option 2: If a suite name is provided, run that specific suite (no prompt)
        elif suite_name:
            result_code, tests_ran = _run_suite_tests(suite_name, sub_suite_name, test_name, pytest_args)

        # Option 3: Default run behavior - runs all profiles (generic sysagent)
        else:
            result_code, tests_ran = _run_all_profiles(
                skip_system_check,
                data_dir,
                verbose,
                debug,
                force,
                prompt_overrides,
            )

    except KeyboardInterrupt:
        logger.warning("Main test execution interrupted by user. Proceeding to report generation.")
        interrupt_occurred = True
        tests_ran = True
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)

        # Clean up filter environment variable
        if "CORE_TEST_FILTERS" in os.environ:
            del os.environ["CORE_TEST_FILTERS"]

        # Check if any interrupt was detected
        if interrupt_occurred or shared_state.INTERRUPT_OCCURRED:
            logger.warning("Test execution was interrupted by user")

        if tests_ran:
            _generate_test_reports(data_dir, verbose, debug)
            # Determine final exit code based on test summary (only fail on broken tests)
            result_code = _determine_final_exit_code(data_dir, result_code)

    return result_code


def _run_profile_tests(
    profile_name: str,
    pytest_args: List[str],
    skip_system_check: bool,
    data_dir: str,
    verbose: bool = False,
    debug: bool = False,
    filters: Dict[str, Any] = None,
    force: bool = False,
) -> tuple:
    """Run tests for a specific profile.

    Args:
        force: Whether to skip interactive prompts

    Returns:
        tuple: (exit_code, tests_ran) where tests_ran indicates if pytest actually executed
    """
    # Import dependency resolver
    from sysagent.utils.config import expand_profile_with_dependencies, get_profile_dependencies

    # Get all available profiles with their configs
    all_profiles_data = list_profiles(include_examples=True)
    all_profiles_dict = {}

    for profile_type, profiles in all_profiles_data.items():
        for profile in profiles:
            configs = profile.get("configs")
            if configs:
                profile_name_key = configs.get("name")
                if profile_name_key:
                    all_profiles_dict[profile_name_key] = configs

    # Check if profile exists
    if profile_name not in all_profiles_dict:
        logger.error(f"Profile not found: {profile_name}")
        return 1, False

    # Check profile type - only run system validation for qualification profiles
    profile_config = all_profiles_dict[profile_name]
    profile_labels = profile_config.get("params", {}).get("labels", {})
    profile_type = profile_labels.get("type", "")

    # Generic sysagent: No CPU validation (package-specific implementations handle this)
    # ESQ overrides the entire run command with its own CPU validation logic
    logger.debug(f"Running profile: {profile_name} (type: {profile_type})")

    # Expand profile with dependencies (dependencies first, then the profile)
    try:
        execution_order = expand_profile_with_dependencies(profile_name, all_profiles_dict)

        # Log dependency information
        dependencies = get_profile_dependencies(all_profiles_dict[profile_name])
        if dependencies:
            logger.info(f"Profile '{profile_name}' has dependencies: {', '.join(dependencies)}")
            logger.info("Execution order:")
            for i, prof in enumerate(execution_order, 1):
                prefix = "  └─" if i == len(execution_order) else "  ├─"
                suffix = " (requested)" if prof == profile_name else ""
                logger.info(f"{prefix} {prof}{suffix}")

    except Exception as e:
        logger.error(f"Failed to resolve dependencies for profile '{profile_name}': {e}")
        return 1, False

    # Execute profiles in dependency order
    final_exit_code = 0
    tests_ran = False

    for current_profile_name in execution_order:
        result_code, profile_tests_ran = _run_single_profile(
            current_profile_name,
            pytest_args,
            skip_system_check,
            data_dir,
            verbose,
            debug,
            filters if current_profile_name == profile_name else None,  # Only apply filters to requested profile
        )

        tests_ran = tests_ran or profile_tests_ran

        # Track the worst exit code from dependency profiles
        # Note: We continue execution even if dependencies fail, as the main profile
        # may need to run (e.g., to summarize results from failed/passed dependencies)
        if result_code != 0 and current_profile_name != profile_name:
            logger.warning(
                f"Dependency profile '{current_profile_name}' completed with exit code {result_code}. "
                f"Continuing to execute main profile '{profile_name}'."
            )
            # Don't update final_exit_code yet - let main profile determine final status
        elif result_code != 0 and current_profile_name == profile_name:
            # Only set non-zero exit code if the main profile itself fails
            final_exit_code = result_code

    return final_exit_code, tests_ran


def _run_single_profile(
    profile_name: str,
    pytest_args: List[str],
    skip_system_check: bool,
    data_dir: str,
    verbose: bool = False,
    debug: bool = False,
    filters: Dict[str, Any] = None,
) -> tuple:
    """Run a single profile without dependency resolution.

    Returns:
        tuple: (exit_code, tests_ran) where tests_ran indicates if pytest actually executed
    """
    logger.info(f"Running profile: {profile_name}")
    os.environ["ACTIVE_PROFILE"] = profile_name

    # Store filters in environment for access by pytest plugins
    if filters:
        import json

        os.environ["CORE_TEST_FILTERS"] = json.dumps(filters)
        logger.debug(f"Applied test filters: {filters}")

    # Use the profile name to find the profile configuration
    all_profiles = list_profiles(include_examples=True)
    profile_configs = None

    for profile_type, profiles in all_profiles.items():
        for profile in profiles:
            configs = profile.get("configs")
            if configs and configs.get("name") == profile_name:
                profile_configs = configs
                break
        if profile_configs:
            break

    if not profile_configs:
        logger.error(f"Profile not found: {profile_name}")
        return 1, False

    # Validate profile requirements if not explicitly skipped
    if not skip_system_check:
        from sysagent.utils.testing.profile_validator import (
            validate_filtered_profile_requirements,
            validate_profile_requirements,
        )

        if filters:
            validation_result = validate_filtered_profile_requirements(
                profile_configs, filters, profile_name=profile_name
            )
        else:
            validation_result = validate_profile_requirements(profile_configs, profile_name=profile_name)

        if not validation_result.get("passed", False):
            return 1, False

    # Get the profile tier and filter configuration
    from sysagent.utils.testing.tier_validator import get_highest_matching_tier

    profile_highest_tier = get_highest_matching_tier(profile_configs)
    if profile_highest_tier:
        logger.info(f"Highest passed tier for profile '{profile_name}': {profile_highest_tier}")
        os.environ["ACTIVE_PROFILE_HIGHEST_TIER"] = profile_highest_tier
        profile_configs = filter_profile_by_tier(profile_configs, profile_highest_tier)

    # Verify that the profile has valid suites section after filtering
    profile_suites = profile_configs.get("suites", [])
    if not profile_suites:
        logger.error(f"No suites remaining after tier filtering for profile: {profile_name}")
        return 1, False

    # Get profile-level venv config (may be None)
    profile_venv_config = profile_configs.get("params", {}).get("venv")

    # Collect test paths grouped by venv configuration
    venv_groups = _collect_test_paths_from_suites(profile_suites, profile_venv_config)

    if not venv_groups:
        logger.warning(f"No test files found for profile: {profile_name}")
        return 1, False

    # Flatten all test paths for initial validation
    all_test_paths = []
    for test_paths, _, _ in venv_groups.values():
        all_test_paths.extend(test_paths)

    if all_test_paths:
        pytest_args = add_test_paths_to_args(pytest_args, all_test_paths)
    else:
        logger.warning(f"No test files found for profile: {profile_name}")

    # Check if interrupt has already occurred
    if shared_state.INTERRUPT_OCCURRED:
        logger.warning("Interrupt detected before running profile")

    logger.info(f"Running pytest for profile {profile_name}")

    # Run tests for each venv configuration group
    overall_exit_code = 0

    for venv_key, (test_paths, venv_config, suite_path) in venv_groups.items():
        if not test_paths:
            continue

        venv_enabled = venv_config.get("enabled", False)
        requirements_file_rel = venv_config.get("requirements_file")
        python_version = venv_config.get("python_version")
        venv_timeout = venv_config.get("timeout", 7200.0)

        # Log venv group info
        if venv_enabled:
            logger.info(f"Running {len(test_paths)} test(s) with venv (requirements: {requirements_file_rel})")
        else:
            logger.info(f"Running {len(test_paths)} test(s) without venv")

        # Create pytest args for this group
        group_pytest_args = create_pytest_args(data_dir, verbose, debug)
        group_pytest_args = add_test_paths_to_args(group_pytest_args, test_paths)

        try:
            if venv_enabled:
                if not requirements_file_rel:
                    logger.warning("Venv enabled but no requirements_file specified, running without venv")
                    exit_code = run_pytest(group_pytest_args)
                else:
                    # Resolve requirements file path relative to suite directory
                    requirements_file = os.path.join(suite_path, requirements_file_rel)

                    if not os.path.exists(requirements_file):
                        logger.error(f"Requirements file not found: {requirements_file}")
                        overall_exit_code = 1
                        continue

                    logger.info(f"Running tests in isolated venv with requirements from: {requirements_file}")
                    logger.info(f"Venv timeout configured: {venv_timeout}s ({venv_timeout / 3600:.2f} hours)")

                    # Run pytest with venv
                    from sysagent.utils.testing.pytest_config import run_pytest_with_venv

                    exit_code = run_pytest_with_venv(
                        pytest_args=group_pytest_args,
                        suite_path=suite_path,
                        requirements_file=requirements_file,
                        data_dir=data_dir,
                        python_version=python_version,
                        force=False,
                        timeout=venv_timeout,
                    )
            else:
                # Run pytest normally without venv
                exit_code = run_pytest(group_pytest_args)

            # Track worst exit code
            if exit_code != 0:
                overall_exit_code = exit_code

        except KeyboardInterrupt:
            logger.warning("Test execution interrupted by user. Stopping all tests.")
            return 130, True

    return overall_exit_code, True


def _run_suite_tests(suite_name: str, sub_suite_name: str, test_name: str, pytest_args: List[str]) -> tuple:
    """Run tests for a specific suite, sub-suite, or test.

    Returns:
        tuple: (exit_code, tests_ran) where tests_ran indicates if pytest actually executed
    """
    logger.info(f"Running suite: {suite_name}")
    suite_path = get_suite_directory(suite_name)
    if not suite_path:
        logger.error(f"Suite not found: {suite_name}")
        return 1, False

    if sub_suite_name:
        sub_suite_path = os.path.join(suite_path, sub_suite_name)
        logger.info(f"Running sub-suite: {sub_suite_name}")
        if not os.path.exists(sub_suite_path):
            logger.error(f"Sub-suite not found: {sub_suite_name}")
            return 1, False

        if test_name:
            test_path = os.path.join(sub_suite_path, f"{test_name}.py")
            logger.info(f"Running test: {test_name}")
            if not os.path.exists(test_path):
                logger.error(f"Test file not found: {test_name}")
                return 1, False
            pytest_args = add_test_paths_to_args(pytest_args, [test_path])
        else:
            pytest_args = add_test_paths_to_args(pytest_args, [sub_suite_path])
    else:
        pytest_args = add_test_paths_to_args(pytest_args, [suite_path])

    if shared_state.INTERRUPT_OCCURRED:
        logger.warning("Interrupt detected before running test suite")

    logger.info(f"Running pytest with args: {pytest_args}")
    try:
        exit_code = run_pytest(pytest_args)
        return exit_code, True
    except KeyboardInterrupt:
        logger.warning("Test execution interrupted by user. Stopping all tests.")
        return 130, True


def _run_all_profiles(
    skip_system_check: bool,
    data_dir: str,
    verbose: bool,
    debug: bool,
    force: bool = False,
    prompt_overrides: dict = None,
) -> tuple:
    """Run all available profiles - generic sysagent behavior without prompts or CPU validation.

    Generic behavior: No prompts, no CPU validation checks.
    Always runs all profile types (qualifications, suites, verticals).
    Package-specific implementations (e.g., ESQ) should override the entire run command.

    Args:
        skip_system_check: Whether to skip system requirement validation (ignored - always skipped)
        data_dir: Data directory path
        verbose: Whether to enable verbose output
        debug: Whether to enable debug output
        force: Ignored (no prompts in generic implementation)
        prompt_overrides: Ignored (no prompts in generic implementation)

    Returns:
        tuple: (exit_code, tests_ran) where tests_ran indicates if any pytest actually executed
    """
    from sysagent.utils.config import (
        expand_profile_with_dependencies,
        get_profile_dependencies,
        resolve_profile_dependencies,
        validate_profile_dependencies,
    )

    all_profiles = list_profiles(include_examples=False)
    logger.debug(f"Found {sum(len(profiles) for profiles in all_profiles.values())} profiles")

    # Build complete profiles dictionary (all available profiles)
    complete_profiles_dict = {}
    complete_profile_items_map = {}  # Map profile_name -> (profile_type, profile)

    for profile_type, profiles in all_profiles.items():
        for profile in profiles:
            configs = profile.get("configs")
            if configs:
                profile_name = configs.get("name")
                if profile_name:
                    complete_profiles_dict[profile_name] = configs
                    complete_profile_items_map[profile_name] = (profile_type, profile)

    # Simplified generic behavior: no prompts, no CPU validation, no filtering
    # Package-specific implementations (e.g., ESQ) should override this command
    logger.info("Running all profile types (qualifications, suites, verticals)")

    # First pass: collect all available profiles
    requested_profile_names = []

    for profile_type, profiles in all_profiles.items():
        for profile in profiles:
            configs = profile.get("configs")
            if configs:
                profile_name = configs.get("name")
                if profile_name:
                    requested_profile_names.append(profile_name)

    if not requested_profile_names:
        logger.error("No profiles found")
        return 1, False

    # Second pass: expand each requested profile with its dependencies
    all_profiles_to_run = set()

    for profile_name in requested_profile_names:
        try:
            # Expand profile with dependencies (returns list in execution order)
            expanded_profiles = expand_profile_with_dependencies(profile_name, complete_profiles_dict)

            # Log dependencies if they exist
            dependencies = get_profile_dependencies(complete_profiles_dict[profile_name])
            if dependencies:
                logger.debug(f"Profile '{profile_name}' depends on: {', '.join(dependencies)}")

            # Add all profiles (dependencies + requested) to the set
            all_profiles_to_run.update(expanded_profiles)

        except Exception as e:
            logger.error(f"Failed to resolve dependencies for profile '{profile_name}': {e}")
            # Still add the profile itself even if dependency resolution fails
            all_profiles_to_run.add(profile_name)

    # Build final profile items and dict from the complete set
    all_profile_items = []
    all_profiles_dict = {}

    for profile_name in all_profiles_to_run:
        if profile_name in complete_profile_items_map:
            profile_type, profile = complete_profile_items_map[profile_name]
            all_profile_items.append((profile_type, profile))
            all_profiles_dict[profile_name] = complete_profiles_dict[profile_name]

    if not all_profile_items:
        logger.error("No valid profiles to run after dependency resolution")
        return 1, False

    # Validate profile dependencies
    dep_errors = validate_profile_dependencies(all_profiles_dict)
    if dep_errors:
        logger.warning("Profile dependency validation warnings:")
        for error in dep_errors:
            logger.warning(f"  - {error}")

    # Resolve execution order based on dependencies
    try:
        execution_order = resolve_profile_dependencies(all_profiles_dict)
        logger.info("Profile execution order (respecting dependencies):")
        for i, profile_name in enumerate(execution_order, 1):
            prefix = "  └─" if i == len(execution_order) else "  ├─"
            logger.info(f"{prefix} {profile_name}")
    except Exception as e:
        logger.error(f"Failed to resolve profile dependencies: {e}")
        logger.info("Falling back to alphabetical order")
        execution_order = sorted(all_profiles_dict.keys())

    # Validate all profiles if not explicitly skipped
    valid_profiles, failed_profiles = _validate_all_profiles(all_profile_items, skip_system_check)

    if failed_profiles:
        logger.info("")
        logger.info("═" * 70)
        logger.info("Profile Validation Summary")
        logger.info("═" * 70)
        logger.info(f"Failed profiles ({len(failed_profiles)}):")
        for name in failed_profiles:
            logger.info(f"  ✗ {name}")
        logger.info("")
        logger.error("Some profiles failed validation. Aborting test run.")
        return 1, False

    if not valid_profiles:
        logger.error("No valid profiles found after validation. Aborting test run.")
        return 1, False

    # Create mapping of profile names to (profile_type, profile) tuples
    valid_profiles_map = {}
    for profile_type, profile in valid_profiles:
        configs = profile.get("configs")
        if configs:
            profile_name = configs.get("name")
            if profile_name:
                valid_profiles_map[profile_name] = (profile_type, profile)

    # Run profiles in dependency order (only those that are valid)
    logger.info(f"Running tests for {len(valid_profiles)} valid profiles in dependency order")
    result = 0
    executed_profiles = set()

    for profile_name in execution_order:
        # Only run if profile is valid
        if profile_name in valid_profiles_map:
            # Skip if already executed (in case of duplicate handling)
            if profile_name in executed_profiles:
                continue

            profile_type, profile = valid_profiles_map[profile_name]
            result = _run_single_profile_in_batch(profile, data_dir, verbose, debug)
            executed_profiles.add(profile_name)

    logger.info(f"All profiles processed. Results: {result}")
    return result, True


def _validate_all_profiles(all_profile_items, skip_system_check: bool):
    """Validate all profiles and return valid and failed lists."""
    from sysagent.utils.testing.profile_validator import validate_profile_requirements

    valid_profiles = []
    failed_profiles = []

    for profile_type, profile in all_profile_items:
        profile_configs = profile.get("configs")
        profile_path = profile.get("path")
        profile_name = profile_configs.get("name") if profile_configs else None

        if not profile_name:
            logger.error(f"No 'name' field found in profile configs: {profile_path}")
            failed_profiles.append(profile_path or "Unknown")
            continue

        if not skip_system_check:
            # Pass profile name for better context in validation messages
            validation_result = validate_profile_requirements(profile_configs, profile_name=profile_name)
            if not validation_result.get("passed", False):
                failed_profiles.append(profile_name)
                continue

        valid_profiles.append((profile_type, profile))

    return valid_profiles, failed_profiles


def _run_single_profile_in_batch(profile, data_dir: str, verbose: bool, debug: bool) -> int:
    """Run a single profile in batch mode with proper cleanup."""
    profile_configs = profile.get("configs")
    profile_name = profile_configs.get("name")

    # Clean up environment between profile runs
    try:
        cleanup_pytest_cache()
        _cleanup_modules()
        _reload_config_module()
    except Exception as e:
        logger.warning(f"Error cleaning environment between profile runs: {e}")

    # Set environment variables
    if "ACTIVE_PROFILE" in os.environ:
        del os.environ["ACTIVE_PROFILE"]
    os.environ["ACTIVE_PROFILE"] = profile_name

    # Get profile tier and filter configuration
    from sysagent.utils.testing.tier_validator import get_highest_matching_tier

    profile_highest_tier = get_highest_matching_tier(profile_configs)

    if "ACTIVE_PROFILE_HIGHEST_TIER" in os.environ:
        del os.environ["ACTIVE_PROFILE_HIGHEST_TIER"]
    if profile_highest_tier:
        logger.info(f"Highest passed tier for profile '{profile_name}': {profile_highest_tier}")
        os.environ["ACTIVE_PROFILE_HIGHEST_TIER"] = profile_highest_tier
        profile_configs = filter_profile_by_tier(profile_configs, profile_highest_tier)

    # Verify suites exist after filtering
    profile_suites = profile_configs.get("suites", [])
    if not profile_suites:
        logger.debug(f"No suites remaining after tier filtering for profile: {profile_name}")
        return 1

    # Get profile-level venv config (may be None)
    profile_venv_config = profile_configs.get("params", {}).get("venv")

    # Collect test paths grouped by venv configuration
    venv_groups = _collect_test_paths_from_suites(profile_suites, profile_venv_config)

    if not venv_groups:
        logger.error(f"No tests found for profile: {profile_name}")
        return 1

    if shared_state.INTERRUPT_OCCURRED:
        logger.warning("Interrupt detected before running profile single profile in batch")

    # Run tests for each venv configuration group
    overall_result = 0

    for venv_key, (test_paths, venv_config, suite_path) in venv_groups.items():
        if not test_paths:
            continue

        venv_enabled = venv_config.get("enabled", False)
        requirements_file_rel = venv_config.get("requirements_file")
        python_version = venv_config.get("python_version")
        venv_timeout = venv_config.get("timeout", 7200.0)

        # Log venv group info
        if venv_enabled:
            logger.info(f"Running {len(test_paths)} test(s) with venv (requirements: {requirements_file_rel})")
        else:
            logger.info(f"Running {len(test_paths)} test(s) without venv")

        # Create pytest args for this group
        profile_pytest_args = create_profile_pytest_args(data_dir, profile_name, verbose, debug)
        profile_pytest_args = add_test_paths_to_args(profile_pytest_args, test_paths)

        try:
            if venv_enabled:
                if not requirements_file_rel:
                    logger.warning("Venv enabled but no requirements_file specified, running without venv")
                    result = run_pytest(profile_pytest_args)
                else:
                    # Resolve requirements file path relative to suite directory
                    requirements_file = os.path.join(suite_path, requirements_file_rel)

                    if not os.path.exists(requirements_file):
                        logger.error(f"Requirements file not found: {requirements_file}")
                        overall_result = 1
                        continue

                    logger.info(f"Running tests in isolated venv with requirements from: {requirements_file}")
                    logger.info(f"Venv timeout configured: {venv_timeout}s ({venv_timeout / 3600:.2f} hours)")

                    # Run pytest with venv
                    from sysagent.utils.testing.pytest_config import run_pytest_with_venv

                    result = run_pytest_with_venv(
                        pytest_args=profile_pytest_args,
                        suite_path=suite_path,
                        requirements_file=requirements_file,
                        data_dir=data_dir,
                        python_version=python_version,
                        force=False,
                        timeout=venv_timeout,
                    )
            else:
                # Run pytest normally without venv
                result = run_pytest(profile_pytest_args)

            # Track worst result
            if result != 0:
                overall_result = result

        except KeyboardInterrupt:
            logger.warning("Test execution interrupted by user. Stopping all tests.")
            return 130

    # Log final result
    if overall_result == 0:
        logger.info(f"Profile passed: {profile_name}")
    else:
        logger.error(f"Profile failed: {profile_name}")

    return overall_result


def _collect_test_paths_from_suites(suites, profile_venv_config=None) -> Dict[str, tuple]:
    """Collect test file paths from suite configurations, grouped by venv configuration.

    Returns:
        Dict mapping venv config key to tuple of (test_paths, venv_config, suite_path)
        The venv config key is a tuple of (enabled, requirements_file, python_version, timeout)
    """
    from sysagent.utils.config import get_suite_directory

    # Dictionary to group tests by venv configuration
    # Key: (enabled, requirements_file, python_version, timeout)
    # Value: (test_paths, venv_config_dict, suite_path)
    venv_groups = {}

    for suite in suites:
        suite_name = suite.get("name")
        suite_path = get_suite_directory(suite_name)
        if not suite_path:
            logger.warning(f"Suite not found: {suite_name}")
            continue

        for sub_suite in suite.get("sub_suites", []):
            sub_suite_name = sub_suite.get("name", "")
            sub_suite_path = os.path.join(suite_path, sub_suite_name)
            if not os.path.exists(sub_suite_path) or not os.path.isdir(sub_suite_path):
                logger.warning(f"Sub-suite folder not found: {sub_suite_path}")
                continue

            # Get venv config for this sub_suite (with fallback to profile-level config)
            venv_config = _get_venv_config_for_subsuite(sub_suite, profile_venv_config)

            # Create venv config key for grouping
            venv_key = (
                venv_config.get("enabled", False),
                venv_config.get("requirements_file"),
                venv_config.get("python_version"),
                venv_config.get("timeout", 7200.0),
            )

            # Initialize group if not exists
            if venv_key not in venv_groups:
                venv_groups[venv_key] = ([], venv_config, os.path.join(suite_path, sub_suite_name))

            # Collect test paths for this sub_suite
            tests_config = sub_suite.get("tests", {})
            for test_name, test_config in tests_config.items():
                test_file = f"{test_name}.py"
                test_path = os.path.join(sub_suite_path, test_file)
                if os.path.exists(test_path):
                    venv_groups[venv_key][0].append(test_path)
                    logger.debug(f"Adding test to venv group {venv_key}: {test_path}")
                else:
                    logger.warning(f"Test file not found: {test_path}")

    return venv_groups


def _get_venv_config_for_subsuite(sub_suite: Dict[str, Any], profile_venv_config: Dict[str, Any]) -> Dict[str, Any]:
    """Get venv configuration for a sub_suite, with fallback to profile-level config.

    Args:
        sub_suite: Sub-suite configuration dictionary
        profile_venv_config: Profile-level venv configuration (can be None)

    Returns:
        Dict with venv configuration (enabled, requirements_file, python_version, timeout)
    """
    # Check if sub_suite has its own venv config (under params.venv for consistency)
    subsuite_params = sub_suite.get("params", {})
    subsuite_venv = subsuite_params.get("venv")

    if subsuite_venv is not None:
        # Sub-suite has its own venv config - use it
        return {
            "enabled": subsuite_venv.get("enabled", False),
            "requirements_file": subsuite_venv.get("requirements_file"),
            "python_version": subsuite_venv.get("python_version"),
            "timeout": subsuite_venv.get("timeout", 7200.0),
        }
    elif profile_venv_config:
        # No sub-suite config, use profile-level config
        return profile_venv_config
    else:
        # No venv config at all - disabled by default
        return {"enabled": False, "requirements_file": None, "python_version": None, "timeout": 7200.0}


def _determine_final_exit_code(data_dir: str, pytest_exit_code: int) -> int:
    """
    Determine the final exit code based on test summary.

    Returns:
        0 if tests ran successfully (even with failed tests)
        1 only if there are broken tests or critical errors

    Args:
        data_dir: Data directory containing test results
        pytest_exit_code: Original pytest exit code

    Returns:
        int: Final exit code (0 for success, 1 for failure)
    """
    summary_path = os.path.join(data_dir, "results", "core", "test_summary.json")

    # If no summary exists, return the original pytest exit code
    if not os.path.exists(summary_path):
        logger.warning("No test summary found. Using pytest exit code.")
        return pytest_exit_code

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        # Check for broken tests
        broken_count = summary_data.get("summary", {}).get("status_counts", {}).get("broken", 0)

        if broken_count > 0:
            logger.debug(f"Found {broken_count} broken test(s). Returning exit code 1.")
            return 1

        # If no broken tests, return success even if there are failed tests
        failed_count = summary_data.get("summary", {}).get("status_counts", {}).get("failed", 0)
        if failed_count > 0:
            logger.debug(f"Found {failed_count} failed test(s), but no broken tests. Returning exit code 0.")

        return 0

    except Exception as e:
        logger.warning(f"Failed to read test summary: {e}. Using pytest exit code.")
        return pytest_exit_code


def _cleanup_modules():
    """Clean up cached modules between profile runs."""
    for module_name in list(sys.modules.keys()):
        # Skip shared_state module to preserve interrupt state between profile runs
        if module_name == "sysagent.utils.core.shared_state":
            continue
        if module_name.startswith("sysagent.suites") or module_name.startswith("sysagent.utils"):
            if module_name in sys.modules:
                del sys.modules[module_name]


def _reload_config_module():
    """Reload the config module to ensure fresh state."""
    import importlib

    importlib.invalidate_caches()
    from sysagent.utils import config as config_module

    importlib.reload(config_module)


def _generate_test_reports(data_dir: str, verbose: bool, debug: bool):
    """Generate comprehensive test reports including summary, logs, and Allure report."""
    try:
        # Step 1: Generate and save test results summary
        logger.info("Generating test results summary")
        summary_generator = CoreResultsSummaryGenerator(data_dir)

        summary_filepath = None
        summary_data = None
        try:
            summary_filepath = summary_generator.generate_and_save_summary(verbose=verbose)

            # Load the summary data for later display
            with open(summary_filepath, "r", encoding="utf-8") as f:
                summary_data = json.load(f)

            relative_summary_path = os.path.relpath(summary_filepath, os.getcwd())
            logger.info(f"Test results summary saved to: {relative_summary_path}")

        except Exception as e:
            logger.warning(f"Failed to generate test results summary: {e}")

        # Step 2: Flush all log handlers
        _flush_all_loggers()
        time.sleep(0.1)  # Small delay for file system

        # Step 3: Attach logs, summaries, and system info
        _attach_test_artifacts(verbose, debug)

        # Step 4: Display summary tables
        _display_summary_tables(summary_data, summary_filepath, verbose, debug)

        # Step 5: Generate Allure report
        from sysagent.utils.cli.commands.report import generate_report

        report_result = generate_report(debug=debug)
        if report_result != 0:
            logger.warning("Allure report generation failed or incomplete.")

    except Exception as e:
        logger.error(f"Error generating Allure report: {e}")


def _flush_all_loggers():
    """Flush all log handlers to ensure logs are written."""
    for handler in logger.handlers:
        try:
            if hasattr(handler, "stream") and not handler.stream.closed:
                handler.flush()
        except (ValueError, AttributeError, OSError):
            pass

    for name, log_obj in logging.Logger.manager.loggerDict.items():
        if isinstance(log_obj, logging.Logger):
            for handler in log_obj.handlers:
                try:
                    if hasattr(handler, "stream") and not handler.stream.closed:
                        handler.flush()
                except (ValueError, AttributeError, OSError):
                    pass


def _attach_test_artifacts(verbose: bool, debug: bool):
    """Attach logs, summaries, and system information to the report."""
    from sysagent.utils.cli.commands.attach import attach_logs, attach_summaries, attach_system

    logger.debug("Attaching logs to report")
    log_attached = attach_logs(verbose=verbose, debug=debug)
    if log_attached != 0:
        logger.warning(f"Log attachment failed with status code: {log_attached}")

    logger.debug("Attaching summaries to report")
    summary_attached = attach_summaries(verbose=verbose, debug=debug)
    if summary_attached != 0:
        logger.warning(f"Summary attachment failed with status code: {summary_attached}")

    logger.debug("Attaching system information to report")
    system_attached = attach_system(verbose=verbose, debug=debug)
    if system_attached != 0:
        logger.warning(f"System information attachment failed with status code: {system_attached}")


def _display_summary_tables(summary_data, summary_filepath: str, verbose: bool, debug: bool):
    """Display summary tables if test results exist."""
    if not summary_data:
        return

    try:
        summary_info = summary_data.get("summary", {})
        total_tests = summary_info.get("total_tests", 0)

        if total_tests > 0:
            table_generator = TestSummaryTableGenerator(summary_data)
            summary_table = table_generator.generate_summary_table()
            detailed_table = table_generator.generate_detailed_test_table()

            if summary_table:
                if verbose or debug:
                    logger.info("\n\n" + summary_table + "\n" + detailed_table)
                else:
                    logger.info("\n\n" + summary_table)
        else:
            if summary_filepath:
                relative_summary_path = os.path.relpath(summary_filepath, os.getcwd())
                logger.info(f"No test results found to display. Summary saved to: {relative_summary_path}")
    except Exception as e:
        logger.warning(f"Failed to display summary table: {e}")
