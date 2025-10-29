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
from sysagent.utils.system import SystemInfoCache
from sysagent.utils.testing import (
    add_test_paths_to_args,
    cleanup_pytest_cache,
    create_profile_pytest_args,
    create_pytest_args,
    run_pytest,
    validate_pytest_args,
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
    extra_args: List[str] = None,
) -> int:
    """
    Run tests based on a profile or specific suite/test.

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
        extra_args: Additional pytest arguments to pass

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
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
    system_info_cache = SystemInfoCache(os.path.join(data_dir, "cache"))

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
        # Option 1: If a profile name is provided, load the profile configuration
        if profile_name:
            result_code, tests_ran = _run_profile_tests(
                profile_name, pytest_args, skip_system_check, data_dir, parsed_filters
            )

        # Option 2: If a suite name is provided, run the specified suite
        elif suite_name:
            result_code, tests_ran = _run_suite_tests(suite_name, sub_suite_name, test_name, pytest_args)

        # Option 3: Run all profiles if no specific profile or suite is provided
        else:
            result_code, tests_ran = _run_all_profiles(skip_system_check, data_dir, verbose, debug)

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
            logger.warning("Test execution was interrupted by user. Generating report with partial results.")

        if tests_ran:
            _generate_test_reports(data_dir, verbose, debug)
            # Determine final exit code based on test summary (only fail on broken tests)
            result_code = _determine_final_exit_code(data_dir, result_code)

    return result_code


def _run_profile_tests(
    profile_name: str, pytest_args: List[str], skip_system_check: bool, data_dir: str, filters: Dict[str, Any] = None
) -> tuple:
    """Run tests for a specific profile.

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
    profile_name: str, pytest_args: List[str], skip_system_check: bool, data_dir: str, filters: Dict[str, Any] = None
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
        logger.info(f"Applied test filters: {filters}")

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
        from sysagent.utils.testing.profile_validator import validate_profile_requirements

        validation_result = validate_profile_requirements(profile_configs)
        if not validation_result.get("passed", False):
            logger.error(f"Profile validation failed for {profile_name}")
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

    # Collect test paths from profile suites
    test_paths = _collect_test_paths_from_suites(profile_suites)

    if test_paths:
        pytest_args = add_test_paths_to_args(pytest_args, test_paths)
    else:
        logger.warning(f"No test files found for profile: {profile_name}")

    # Check if interrupt has already occurred
    if shared_state.INTERRUPT_OCCURRED:
        logger.warning("Interrupt detected before running profile")

    logger.info(f"Running pytest with args: {pytest_args}")
    try:
        exit_code = run_pytest(pytest_args)
        return exit_code, True
    except KeyboardInterrupt:
        logger.warning("Test execution interrupted by user. Stopping all tests.")
        return 130, True


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


def _run_all_profiles(skip_system_check: bool, data_dir: str, verbose: bool, debug: bool) -> tuple:
    """Run all available profiles.

    Returns:
        tuple: (exit_code, tests_ran) where tests_ran indicates if any pytest actually executed
    """
    from sysagent.utils.config import resolve_profile_dependencies, validate_profile_dependencies

    all_profiles = list_profiles(include_examples=False)
    logger.debug(f"Found {sum(len(profiles) for profiles in all_profiles.values())} profiles")

    all_profile_items = []
    all_profiles_dict = {}

    for profile_type, profiles in all_profiles.items():
        for profile in profiles:
            configs = profile.get("configs")
            if configs:
                profile_name = configs.get("name")
                if profile_name:
                    all_profile_items.append((profile_type, profile))
                    all_profiles_dict[profile_name] = configs

    if not all_profile_items:
        logger.error("No profiles found")
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
        logger.info("Profile Validation Summary")
        logger.info(f"Failed profiles ({len(failed_profiles)}):")
        for name in failed_profiles:
            logger.info(f"  ✗ {name}")
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
            validation_result = validate_profile_requirements(profile_configs)
            if not validation_result.get("passed", False):
                logger.error(f"Profile validation failed for {profile_name}")
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

    # Create pytest args and collect test paths
    profile_pytest_args = create_profile_pytest_args(data_dir, profile_name, verbose, debug)
    test_paths = _collect_test_paths_from_suites(profile_suites)

    if test_paths:
        profile_pytest_args = add_test_paths_to_args(profile_pytest_args, test_paths)

    if not validate_pytest_args(profile_pytest_args):
        logger.error(f"No tests found for profile: {profile_name}")
        return 1

    if shared_state.INTERRUPT_OCCURRED:
        logger.warning("Interrupt detected before running profile single profile in batch")

    # Run the tests
    try:
        logger.info(f"Running pytest for profile {profile_name}")
        result = run_pytest(profile_pytest_args)
        if result == 0:
            logger.info(f"Profile passed: {profile_name}")
        else:
            logger.error(f"Profile failed: {profile_name}")
        return result
    except KeyboardInterrupt:
        logger.warning("Test execution interrupted by user. Stopping all tests.")
        return 130


def _collect_test_paths_from_suites(suites) -> List[str]:
    """Collect test file paths from suite configurations."""
    test_paths = []

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

            tests_config = sub_suite.get("tests", {})
            for test_name, test_config in tests_config.items():
                test_file = f"{test_name}.py"
                test_path = os.path.join(sub_suite_path, test_file)
                if os.path.exists(test_path):
                    test_paths.append(test_path)
                    logger.info(f"Adding test: {test_path}")
                else:
                    logger.warning(f"Test file not found: {test_path}")

    return test_paths


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
            logger.error(f"Found {broken_count} broken test(s). Returning exit code 1.")
            return 1

        # If no broken tests, return success even if there are failed tests
        failed_count = summary_data.get("summary", {}).get("status_counts", {}).get("failed", 0)
        if failed_count > 0:
            logger.info(f"Found {failed_count} failed test(s), but no broken tests. Returning exit code 0.")

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
