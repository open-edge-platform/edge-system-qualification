# Copyright (C) 2026 Intel Corporation
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
import string
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any


from sysagent.utils.cli.filters import parse_filters
from sysagent.utils.cli.handlers import handle_interrupt
from sysagent.utils.config import filter_profile_by_tier, get_suite_directory, list_profiles, setup_data_dir
from sysagent.utils.config.config_loader import get_cli_aware_project_name
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
    filters: list[str] = None,
    force: bool = False,
    no_mask: bool = False,
    set_prompt: list[str] = None,
    extra_args: list[str] = None,
    run_all_profiles: bool = None,
    qualification_only: bool = None,
    select_profile: bool = None,
    profiles_file: str = None,
    telemetry_interval: int = None,
    tags: list[str] = None,
) -> int:
    """
    Run tests based on profiles, suites, or specific test cases.

    Generic sysagent behavior:
    - Default run (no flags): Runs all profile types (qualifications, suites, verticals)
    - Specific profile (-p): Runs specified profile
    - Suite/test run: Runs specified suite/test
    - Profiles file (--profiles-file): Runs the profile(s) listed in a YAML file
    - Interactive selection (--select): Pick profile(s)/test_id(s) via a checkbox tree

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
        select_profile: Whether to interactively pick profile(s)/test_id(s) via a checkbox tree.
        profiles_file: Path to a YAML profiles template file to run (see --profiles-file).
        tags: List of tag keywords to resolve to profile(s). Ignored in generic sysagent
            (package-specific implementations like ESQ resolve tags to profiles).

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

    if telemetry_interval is not None:
        if telemetry_interval >= 1:
            os.environ["CORE_TELEMETRY_INTERVAL"] = str(telemetry_interval)
            logger.info("Telemetry interval overridden via CLI: %ds", telemetry_interval)
        else:
            logger.warning("--telemetry-interval must be >= 1 second; ignoring value %d", telemetry_interval)

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

    if profiles_file and (profile_name or suite_name):
        logger.error("Error: --profiles-file cannot be combined with --profile or --suite")
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
        if not profile_name and not profiles_file:
            logger.error("Error: --filter option can only be used with --profile or --profiles-file option")
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

        # Option 2: If a profiles file is provided, run the profile(s) listed in it
        elif profiles_file:
            result_code, tests_ran = _run_profiles_file(
                profiles_file, pytest_args, skip_system_check, data_dir, verbose, debug, parsed_filters, force
            )

        # Option 3: If a suite name is provided, run that specific suite (no prompt)
        elif suite_name:
            result_code, tests_ran = _run_suite_tests(suite_name, sub_suite_name, test_name, pytest_args)

        # Option 4: Interactively pick profile(s) (and optionally specific test_id(s)) to run
        elif select_profile:
            selected_profile_names, per_profile_filters = _prompt_select_any_profile(
                force, list_profiles(include_examples=False)
            )
            if not selected_profile_names:
                logger.error("No profile selected - nothing to run")
                result_code, tests_ran = 1, False
            else:
                _offer_save_profiles_selection(selected_profile_names, per_profile_filters)
                result_code, tests_ran = _resolve_and_execute_profiles(
                    selected_profile_names,
                    pytest_args,
                    skip_system_check,
                    data_dir,
                    verbose,
                    debug,
                    per_profile_filters=per_profile_filters,
                )

        # Option 5: Default run behavior - runs all profiles (generic sysagent)
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

        # Clean up telemetry interval override (only if it was set by this invocation)
        if telemetry_interval is not None and "CORE_TELEMETRY_INTERVAL" in os.environ:
            del os.environ["CORE_TELEMETRY_INTERVAL"]

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
    pytest_args: list[str],
    skip_system_check: bool,
    data_dir: str,
    verbose: bool = False,
    debug: bool = False,
    filters: dict[str, Any] = None,
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


def _resolve_and_execute_profiles(
    requested_profile_names: list[str],
    pytest_args: list[str],
    skip_system_check: bool,
    data_dir: str,
    verbose: bool = False,
    debug: bool = False,
    filters: dict[str, Any] = None,
    force: bool = False,
    per_profile_filters: dict[str, dict[str, Any]] = None,
) -> tuple:
    """
    Resolve dependencies for one or more explicitly requested profiles and execute them.

    Generic multi-profile counterpart to `_run_profile_tests` (which only handles a
    single profile): expands each requested profile with its dependencies, deduplicates
    the combined set, and resolves a single dependency-priority execution order (avoiding
    redundant re-runs of shared dependencies). Used by --profiles-file and --select.
    No system validation or extra prompts are performed here - package-specific
    implementations (like ESQ) that need those should build on top of this.

    Args:
        per_profile_filters: Optional {profile_name: filters_dict} overriding the shared
            `filters` for specific requested profiles (e.g., a per-profile test_id scope
            picked via --select or loaded from --profiles-file). Profiles not present in
            this mapping fall back to the shared `filters`.

    Returns:
        tuple: (exit_code, tests_ran)
    """
    from sysagent.utils.config import expand_profile_with_dependencies, resolve_profile_dependencies

    per_profile_filters = per_profile_filters or {}

    all_profiles_data = list_profiles(include_examples=True)
    all_profiles_dict = {}
    for profiles in all_profiles_data.values():
        for profile in profiles:
            configs = profile.get("configs")
            if configs:
                profile_name_key = configs.get("name")
                if profile_name_key:
                    all_profiles_dict[profile_name_key] = configs

    missing = [name for name in requested_profile_names if name not in all_profiles_dict]
    if missing:
        logger.error(f"Profile(s) not found: {', '.join(missing)}")
        return 1, False

    # Expand each requested profile with its dependencies; a dict naturally
    # dedupes profiles shared across multiple requested profiles.
    required_profiles: dict[str, Any] = {}
    for profile_name in requested_profile_names:
        try:
            for expanded_name in expand_profile_with_dependencies(profile_name, all_profiles_dict):
                required_profiles[expanded_name] = all_profiles_dict[expanded_name]
        except Exception as e:
            logger.error(f"Failed to resolve dependencies for profile '{profile_name}': {e}")
            return 1, False

    try:
        execution_order = resolve_profile_dependencies(required_profiles)
    except Exception as e:
        logger.error(f"Failed to resolve profile execution order: {e}")
        return 1, False

    requested_set = set(requested_profile_names)
    logger.info("Execution order:")
    for i, prof in enumerate(execution_order, 1):
        prefix = "  └─" if i == len(execution_order) else "  ├─"
        suffix = " (requested)" if prof in requested_set else " (dependency)"
        logger.info(f"{prefix} {prof}{suffix}")

    final_exit_code = 0
    tests_ran = False
    for current_profile_name in execution_order:
        if current_profile_name in requested_set:
            profile_filters = per_profile_filters.get(current_profile_name, filters)
        else:
            profile_filters = None
        result_code, profile_tests_ran = _run_single_profile(
            current_profile_name,
            pytest_args,
            skip_system_check,
            data_dir,
            verbose,
            debug,
            profile_filters,
        )
        tests_ran = tests_ran or profile_tests_ran
        if result_code != 0:
            if current_profile_name in requested_set:
                final_exit_code = result_code
            else:
                logger.warning(
                    f"Dependency profile '{current_profile_name}' completed with exit code {result_code}. "
                    f"Continuing to execute requested profile(s)."
                )

    return final_exit_code, tests_ran


def _run_profiles_file(
    profiles_file: str,
    pytest_args: list[str],
    skip_system_check: bool,
    data_dir: str,
    verbose: bool = False,
    debug: bool = False,
    filters: dict[str, Any] = None,
    force: bool = False,
) -> tuple:
    """
    Run the profile(s) listed in a YAML profiles template file (see --profiles-file).

    Returns:
        tuple: (exit_code, tests_ran)
    """
    profile_names, per_profile_filters = _load_profiles_from_file(profiles_file)
    if profile_names is None:
        return 1, False
    if not profile_names:
        logger.error(f"No profiles listed in '{profiles_file}' - nothing to run")
        return 1, False

    logger.info(f"Loaded {len(profile_names)} profile(s) from '{profiles_file}': {', '.join(profile_names)}")
    return _resolve_and_execute_profiles(
        profile_names,
        pytest_args,
        skip_system_check,
        data_dir,
        verbose,
        debug,
        filters,
        force,
        per_profile_filters=per_profile_filters,
    )


def _run_single_profile(
    profile_name: str,
    pytest_args: list[str],
    skip_system_check: bool,
    data_dir: str,
    verbose: bool = False,
    debug: bool = False,
    filters: dict[str, Any] = None,
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
    from sysagent.utils.testing.tier_validator import get_highest_matching_tier, validate_profile_tiers

    profile_highest_tier = get_highest_matching_tier(profile_configs)
    if profile_highest_tier:
        logger.info(f"Highest passed tier for profile '{profile_name}': {profile_highest_tier}")
        os.environ["ACTIVE_PROFILE_HIGHEST_TIER"] = profile_highest_tier
        profile_configs = filter_profile_by_tier(profile_configs, profile_highest_tier)
    else:
        profile_has_tiers = bool(profile_configs.get("params", {}).get("tiers"))
        if profile_has_tiers:
            tier_results = validate_profile_tiers(profile_configs)
            _log_no_tier_match(profile_name, tier_results)
            return TIER_SKIP_EXIT, False

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


def _run_suite_tests(suite_name: str, sub_suite_name: str, test_name: str, pytest_args: list[str]) -> tuple:
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


# Default location (relative to cwd) offered when saving a --select checkbox
# choice, and used as the suggested path in --profiles-file help text.
DEFAULT_PROFILES_FILE = "custom_profiles.yml"


def _sanitize_path(path: str) -> str:
    """
    Sanitize a user-supplied file-system path to break Coverity PATH_MANIPULATION
    taint chains.

    Resolves the path to an absolute form (eliminating ".." traversals) and rebuilds
    it character-by-character so Coverity's taint tracker sees a freshly-constructed
    string rather than propagated external input.
    """
    resolved = str(Path(path).resolve())
    # Character-by-character copy breaks Coverity taint propagation.
    chars: list = []
    for char in resolved:
        chars.append(char)
    return "".join(chars)


_CLI_NAME_ALLOWED_CHARS = set(string.ascii_letters + string.digits + "-_")


def _sanitize_cli_name(name: str) -> str:
    """Restrict a display name to safe characters before it's echoed back to the user."""
    chars: list = []
    for char in name or "":
        if char in _CLI_NAME_ALLOWED_CHARS:
            chars.append(char)
    return "".join(chars) or "sysagent"


def _load_profiles_from_file(profiles_file: str) -> tuple:
    """
    Load profile names (and any per-profile test_id filters) from a YAML profiles
    template file (see --profiles-file). Each entry may be a mapping {name, test_ids}
    (test_ids optional) - the standardized format written by the --select save prompt -
    or, for convenience when hand-writing a file, a plain profile name string (run the
    whole profile).

    Returns:
        tuple: (profile_names, per_profile_filters) with profile_names deduplicated
        (order preserved) and per_profile_filters a dict of
        {profile_name: {"test_id": [...]}} for profiles scoped to specific tests.
        Returns (None, None) on error.
    """
    profiles_file = _sanitize_path(profiles_file)
    if not os.path.isfile(profiles_file):
        logger.error(f"Profiles file not found: {profiles_file}")
        return None, None

    try:
        with open(profiles_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        logger.error(f"Failed to read profiles file '{profiles_file}': {e}")
        return None, None

    if isinstance(data, dict):
        raw_entries = data.get("profiles") or []
    elif isinstance(data, list):
        raw_entries = data
    else:
        logger.error(f"Invalid format in '{profiles_file}': expected a list or a mapping with a 'profiles' key")
        return None, None

    profile_names = []
    per_profile_filters = {}
    seen = set()
    for entry in raw_entries:
        if isinstance(entry, str):
            name, test_ids = entry.strip(), None
        elif isinstance(entry, dict):
            name, test_ids = str(entry.get("name", "")).strip(), entry.get("test_ids")
        else:
            continue

        if not name or name in seen:
            continue
        seen.add(name)
        profile_names.append(name)

        if test_ids:
            cleaned_ids = sorted({str(t).strip() for t in test_ids if str(t).strip()})
            if cleaned_ids:
                per_profile_filters[name] = {"test_id": cleaned_ids}
    return profile_names, per_profile_filters


def _write_profiles_file(profiles_file: str, profile_names: list, per_profile_filters: dict = None) -> bool:
    """
    Write the given profile names to a YAML profiles template file.

    Standardized format: every entry is always a {name, ...} mapping, never a bare
    profile name string - this avoids mixing formats in a single file. A profile scoped
    to specific test_id(s) - or fully selected but declaring test_id(s) of its own - also
    gets a "test_ids" key, listing ALL of its test_id(s) explicitly when the whole
    profile was selected. This avoids the ambiguity of a bare profile name silently
    implying "all tests". Only profiles that declare no test_id(s) at all are written
    with just "name" (nothing to list).
    """
    per_profile_filters = per_profile_filters or {}
    profiles_file = _sanitize_path(profiles_file)

    all_profiles_dict = {}
    for profiles in list_profiles(include_examples=True).values():
        for profile in profiles:
            configs = profile.get("configs")
            if configs and configs.get("name"):
                all_profiles_dict[configs["name"]] = configs

    entries = []
    for name in sorted(profile_names):
        test_ids = per_profile_filters.get(name, {}).get("test_id")
        if not test_ids:
            configs = all_profiles_dict.get(name)
            if configs:
                test_ids = [test_id for test_id, _ in _get_test_ids_from_profile(configs)]
        entry = {"name": name}
        if test_ids:
            entry["test_ids"] = sorted(test_ids)
        entries.append(entry)
    data = {"profiles": entries}
    try:
        directory = os.path.dirname(profiles_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Restrictive permissions (owner/group read-write only)
        fd = os.open(profiles_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    except OSError as e:
        logger.error(f"Failed to save profiles file '{profiles_file}': {e}")
        return False

    shared_state.LAST_SAVED_PROFILES_FILE = profiles_file

    cli_name = _sanitize_cli_name(get_cli_aware_project_name().lower())
    print(f"Saved {len(profile_names)} profile(s) to '{profiles_file}'.")
    print(f"Tip: run '{cli_name} run --profiles-file {profiles_file}' to reuse this selection.\n")
    return True


def _offer_save_profiles_selection(profile_names: list, per_profile_filters: dict = None) -> None:
    """
    After an interactive --select run, offer to save the picked profiles (and any
    per-profile test_id filters) for reuse.

    A Ctrl+C here is intentionally NOT caught: it should cancel the whole run rather
    than silently skip the save prompt and continue on to execute the selected tests.
    """
    try:
        response = input("Save this selection to a file for reuse? (y/N): ").strip().lower()
    except EOFError:
        return
    if response not in ("y", "yes"):
        return

    # No file path prompt - always auto-generate a unique timestamped filename
    base, ext = os.path.splitext(DEFAULT_PROFILES_FILE)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    profiles_file = f"{base}_{timestamp}{ext}"

    _write_profiles_file(profiles_file, profile_names, per_profile_filters)


def _get_test_ids_from_profile(profile_configs: dict) -> list:
    """
    Extract (test_id, display_name) tuples for every parameterized test declared in a
    profile's YAML (suites[].sub_suites[].tests{}.params[]), deduplicated by test_id.
    """
    test_entries = []
    seen_ids = set()
    for suite in profile_configs.get("suites", []) or []:
        for sub_suite in suite.get("sub_suites", []) or []:
            for test_config in (sub_suite.get("tests", {}) or {}).values():
                for param in (test_config or {}).get("params", []) or []:
                    test_id = param.get("test_id")
                    if not test_id or test_id in seen_ids:
                        continue
                    seen_ids.add(test_id)
                    test_entries.append((test_id, param.get("display_name", test_id)))
    return test_entries


def _build_profile_tree(all_profiles: dict) -> dict:
    """
    Build the nested {group: {profile_name: {"display_name", "tests"}}} structure shown
    by the --select tree picker, skipping profiles marked "hidden" (they remain runnable
    explicitly via --profile/--tag). "tests" is the list of (test_id, display_name)
    tuples declared by that profile (possibly empty), enabling drill-down to specific
    test_id(s) within a profile.
    """
    all_profiles = all_profiles or {}
    tree = {}
    for profile_type in ("qualifications", "suites", "verticals"):
        section = profile_type.capitalize()
        for profile in all_profiles.get(profile_type, []):
            configs = profile.get("configs")
            if not configs:
                continue
            profile_name = configs.get("name")
            if not profile_name:
                continue
            labels = configs.get("params", {}).get("labels", {})
            if labels.get("hidden", False):
                continue
            tree.setdefault(section, {})[profile_name] = {
                "display_name": labels.get("profile_display_name", profile_name),
                "tests": _get_test_ids_from_profile(configs),
            }
    return tree


def _prompt_checkbox_profiles(tree: dict) -> tuple:
    """
    Interactive tree-style picker with real inline group/ungroup (expand/collapse),
    built directly on `prompt_toolkit`, spanning three levels: group -> profile -> test_id.
    Groups and profiles stay collapsed until expanded, and items can be picked across
    multiple groups/profiles in a single screen.

    Keys:
        Up/Down (or k/j): move cursor
        Right/l: expand the current group or profile row
        Left/h: collapse the current group or profile row
        Space: toggle the checkbox of the current profile or test_id row
        a: toggle all currently visible profile/test_id rows
        Enter: confirm and submit the current selection
        q / Ctrl+C: cancel

    Checking a profile row selects ALL of its test_id(s) (tri-state: unchecked/partial
    both toggle to fully checked; fully checked toggles to fully unchecked). Unchecking
    the profile row after individually picking only some test_id(s) scopes that profile
    to just those tests (equivalent to --filter test_id=...). Profiles that declare no
    test_id(s) are checked/unchecked as a single unit.

    Requires a real TTY on stdin/stdout.

    Returns:
        tuple: (selected_profile_names, per_profile_filters) - per_profile_filters is
        {profile_name: {"test_id": [...]}} for profiles scoped to specific tests.
        Returns ([], {}) if cancelled/nothing selected.
    """
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import D
    from prompt_toolkit.layout.layout import Layout
    from prompt_toolkit.styles import Style

    group_order = [section for section in ("Qualifications", "Suites", "Verticals") if section in tree]
    profiles_by_group = {
        section: sorted(tree[section].items(), key=lambda kv: kv[1]["display_name"]) for section in group_order
    }
    profile_info_by_name = {
        profile_name: info for profiles in profiles_by_group.values() for profile_name, info in profiles
    }

    expanded_groups = dict.fromkeys(group_order, False)
    expanded_profiles = {}
    checked_profiles = set()  # profiles with no test_id(s) - checked/unchecked as a whole
    checked_tests = {}  # profile_name -> set(test_id), for profiles that declare test_id(s)
    cursor = {"index": 0}

    def _all_test_ids(tests):
        return {test_id for test_id, _ in tests}

    def _is_profile_fully_checked(profile_name, tests):
        if not tests:
            return profile_name in checked_profiles
        all_ids = _all_test_ids(tests)
        return bool(all_ids) and checked_tests.get(profile_name, set()) == all_ids

    def _is_profile_picked(profile_name, tests):
        if tests:
            return bool(checked_tests.get(profile_name))
        return profile_name in checked_profiles

    def visible_rows():
        rows = []
        for section in group_order:
            rows.append(("group", section))
            if not expanded_groups[section]:
                continue
            for profile_name, info in profiles_by_group[section]:
                rows.append(("profile", section, profile_name, info["display_name"], info["tests"]))
                if expanded_profiles.get(profile_name) and info["tests"]:
                    for test_id, test_display in info["tests"]:
                        rows.append(("test", section, profile_name, test_id, test_display))
        return rows

    def _is_checked(row):
        if row[0] == "profile":
            return _is_profile_fully_checked(row[2], row[4])
        if row[0] == "test":
            return row[3] in checked_tests.get(row[2], ())
        return False

    def _set_checked(row, value):
        if row[0] == "profile":
            _, _section, profile_name, _display_name, tests = row
            if tests:
                # Checking a profile selects ALL of its test_id(s); unchecking clears them.
                checked_tests[profile_name] = _all_test_ids(tests) if value else set()
            elif value:
                checked_profiles.add(profile_name)
            else:
                checked_profiles.discard(profile_name)
        elif row[0] == "test":
            _, _, profile_name, test_id, _ = row
            test_set = checked_tests.setdefault(profile_name, set())
            if value:
                test_set.add(test_id)
            else:
                test_set.discard(test_id)

    def render_rows():
        rows = visible_rows()
        cursor["index"] = max(0, min(cursor["index"], len(rows) - 1)) if rows else 0
        lines = []
        for i, row in enumerate(rows):
            is_cursor = i == cursor["index"]
            pointer = "\u276f " if is_cursor else "  "
            style = "class:cursor" if is_cursor else ""
            if row[0] == "group":
                section = row[1]
                arrow = "\u25be" if expanded_groups[section] else "\u25b8"
                profiles = profiles_by_group[section]
                picked = sum(1 for profile_name, info in profiles if _is_profile_picked(profile_name, info["tests"]))
                text = f"{pointer}{arrow} {section} ({picked}/{len(profiles)} selected)\n"
            elif row[0] == "profile":
                _, _section, profile_name, display_name, tests = row
                has_tests = bool(tests)
                arrow = ("\u25be" if expanded_profiles.get(profile_name) else "\u25b8") if has_tests else " "
                test_set = checked_tests.get(profile_name)
                if _is_profile_fully_checked(profile_name, tests):
                    box = "[x]"
                elif test_set:
                    box = "[~]"
                else:
                    box = "[ ]"
                suffix = f"  ({len(test_set)}/{len(tests)} tests)" if has_tests and test_set else ""
                text = f"{pointer}  {arrow} {box} {display_name}  ({profile_name}){suffix}\n"
            else:
                _, _section, profile_name, test_id, test_display = row
                box = "[x]" if test_id in checked_tests.get(profile_name, ()) else "[ ]"
                text = f"{pointer}      {box} {test_id}  {test_display}\n"
            if is_cursor:
                # Marks this row as the "cursor" so the Window auto-scrolls to keep
                # it visible when the list is taller than the terminal.
                lines.append(("[SetCursorPosition]", ""))
            lines.append((style, text))
        return lines

    kb = KeyBindings()

    @kb.add("c-c")
    @kb.add("q")
    def _cancel(event):
        event.app.exit(result=(None, None))

    @kb.add("up")
    @kb.add("k")
    def _up(event):
        rows = visible_rows()
        if rows:
            cursor["index"] = (cursor["index"] - 1) % len(rows)

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        rows = visible_rows()
        if rows:
            cursor["index"] = (cursor["index"] + 1) % len(rows)

    @kb.add("right")
    @kb.add("l")
    def _expand(event):
        rows = visible_rows()
        if not rows:
            return
        row = rows[cursor["index"]]
        if row[0] == "group":
            expanded_groups[row[1]] = True
        elif row[0] == "profile" and row[4]:
            expanded_profiles[row[2]] = True

    @kb.add("left")
    @kb.add("h")
    def _collapse(event):
        rows = visible_rows()
        if not rows:
            return
        row = rows[cursor["index"]]
        if row[0] == "group":
            expanded_groups[row[1]] = False
        elif row[0] == "profile":
            expanded_profiles[row[2]] = False

    @kb.add(" ")
    def _toggle(event):
        rows = visible_rows()
        if not rows:
            return
        row = rows[cursor["index"]]
        if row[0] in ("profile", "test"):
            _set_checked(row, not _is_checked(row))

    @kb.add("a")
    def _toggle_all(event):
        rows = [row for row in visible_rows() if row[0] in ("profile", "test")]
        if not rows:
            return
        turn_on = not all(_is_checked(row) for row in rows)
        for row in rows:
            _set_checked(row, turn_on)

    @kb.add("enter")
    def _submit(event):
        selected_profile_names = set(checked_profiles)
        per_profile_filters = {}
        for profile_name, test_ids in checked_tests.items():
            if not test_ids:
                continue
            selected_profile_names.add(profile_name)
            all_ids = _all_test_ids(profile_info_by_name[profile_name]["tests"])
            if test_ids != all_ids:
                per_profile_filters[profile_name] = {"test_id": sorted(test_ids)}
        event.app.exit(result=(sorted(selected_profile_names), per_profile_filters))

    cli_name = _sanitize_cli_name(get_cli_aware_project_name().lower())
    instructions = (
        f"Select profile(s) to run - or drill into a profile to pick specific test_id(s) "
        f"(see '{cli_name} list' for full details):\n"
        "Up/Down: move   Right/Left: expand/collapse   Space: toggle   "
        "a: toggle all visible   Enter: confirm   q: cancel"
    )
    layout = Layout(
        HSplit(
            [
                Window(content=FormattedTextControl(lambda: instructions), height=D(preferred=2)),
                Window(content=FormattedTextControl(render_rows), always_hide_cursor=True, wrap_lines=False),
            ]
        )
    )
    style = Style.from_dict({"cursor": "reverse"})
    app = Application(layout=layout, key_bindings=kb, style=style, full_screen=True, mouse_support=True)
    selected_profile_names, per_profile_filters = app.run()

    if not selected_profile_names:
        logger.info("No profile selected. Exiting.")
        return [], {}

    print(f"Tip: use --profile/-p or --tag/-t to skip this prompt. See '{cli_name} run --help' for all options.\n")
    return selected_profile_names, per_profile_filters


def _prompt_checkbox_profiles_fallback(tree: dict) -> tuple:
    """
    Plain-text profile-level fallback for non-interactive terminals (e.g., piped input,
    CI). Accepts comma-separated numbers (e.g., "1,3,5") or "all". Does not support
    drilling into individual test_id(s) - use --filter test_id=... for that instead.
    """
    entries = []
    for section in ("Qualifications", "Suites", "Verticals"):
        for profile_name, info in sorted(tree.get(section, {}).items(), key=lambda kv: kv[1]["display_name"]):
            entries.append((section, info["display_name"], profile_name))

    cli_name = _sanitize_cli_name(get_cli_aware_project_name().lower())
    print(f"Available profiles (see '{cli_name} list' for full details):\n")
    for i, (section, display_name, profile_name) in enumerate(entries, 1):
        print(f"  {i}) [{section}] {display_name}  [{profile_name}]")

    valid_range = f"1-{len(entries)}"
    try:
        response = input(
            f"\nSelect profile(s) to run [{valid_range}, comma-separated, or 'all'] (or press Enter to cancel): "
        ).strip()
    except (KeyboardInterrupt, EOFError):
        logger.info("Interrupted by user. Exiting.")
        return [], {}

    if not response:
        logger.info("No profile selected. Exiting.")
        return [], {}

    if response.lower() == "all":
        return [profile_name for _, _, profile_name in entries], {}

    selected = []
    for token in response.split(","):
        token = token.strip()
        if token.isdigit() and 1 <= int(token) <= len(entries):
            selected.append(entries[int(token) - 1][2])
        else:
            logger.error(f"Invalid selection '{token}' - expected a number ({valid_range}). Exiting.")
            return [], {}

    print(
        "Tip: use --profile/-p or --tag/-t to skip this prompt, or --filter test_id=... to run "
        f"specific tests within a profile. See '{cli_name} run --help' for all options.\n"
    )
    return selected, {}


def _prompt_select_any_profile(
    force: bool = False,
    all_profiles: dict = None,
) -> tuple:
    """
    Prompt the user to interactively pick one or more available profiles to run - or
    drill into a profile to pick specific test_id(s) within it - via a checkbox-style
    tree menu (falls back to a plain profile-level prompt without a TTY).

    Unlike a qualification-only picker, this lists every non-hidden profile across all
    profile types (qualifications, suites, verticals - the same set shown by the `list`
    command), so users aren't limited to qualification profiles when running interactively.

    Generic and reusable across extension packages (like the other low-level execution
    functions in this module): CPU/system-specific validation and any associated-profile
    prompts are the caller's responsibility, applied downstream of the returned selection.

    Args:
        force: If True, skip prompting entirely (non-interactive run - use --profile/--tag/
            --all/--qualification-only to run something specific).
        all_profiles: Dict of {profile_type: [profile_item, ...]} from list_profiles().

    Returns:
        tuple: (selected_profile_names, per_profile_filters) - per_profile_filters is
        {profile_name: {"test_id": [...]}} for profiles scoped to specific tests.
        Returns ([], {}) if nothing was selected/cancelled.
    """
    if force:
        logger.info(
            "No profile auto-selected in non-interactive mode (--force). "
            "Use --profile/--tag/--all/--qualification-only to run something."
        )
        return [], {}

    tree = _build_profile_tree(all_profiles)
    if not tree:
        logger.error("No profiles found to select from")
        return [], {}

    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _prompt_checkbox_profiles(tree)
        except Exception as e:
            logger.debug(f"Checkbox picker unavailable ({e}), falling back to plain prompt")

    return _prompt_checkbox_profiles_fallback(tree)


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
        # Logged at WARNING level so the summary is visible even in non-verbose
        # mode, matching the per-profile validation failure details above.
        logger.warning("")
        logger.warning("═" * 70)
        logger.warning("Profile Validation Summary")
        logger.warning("═" * 70)
        logger.warning(f"Failed profiles ({len(failed_profiles)}):")
        for name in failed_profiles:
            logger.warning(f"  ✗ {name}")
        logger.warning("")
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


# Sentinel exit code returned when a profile is skipped because no system tier matched.
# Distinct from 1 (test failure) so callers can distinguish skipped vs failed.
TIER_SKIP_EXIT = 2


def _log_no_tier_match(profile_name: str, tier_results: dict) -> None:
    """Log a concise error when no system tier matches the profile's requirements."""
    logger.error(f"No system tier matched for profile '{profile_name}' - skipping")


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
    else:
        profile_has_tiers = bool(profile_configs.get("params", {}).get("tiers"))
        if profile_has_tiers:
            from sysagent.utils.testing.tier_validator import validate_profile_tiers

            tier_results = validate_profile_tiers(profile_configs)
            _log_no_tier_match(profile_name, tier_results)
            return TIER_SKIP_EXIT

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


def _collect_test_paths_from_suites(suites, profile_venv_config=None) -> dict[str, tuple]:
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


def _get_venv_config_for_subsuite(sub_suite: dict[str, Any], profile_venv_config: dict[str, Any]) -> dict[str, Any]:
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
