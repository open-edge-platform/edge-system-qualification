# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
ESQ-specific run command implementation.

This module provides a complete, self-contained implementation of the run command
with ESQ-specific CPU validation prompts and Intel processor requirements.
Reuses low-level helpers from sysagent for generic test execution.
"""

import logging
import os
import signal
import sys
from typing import Any, Dict, List

# Import sysagent's low-level execution functions (generic, reusable)
from sysagent.utils.cli.commands.run import (
    _determine_final_exit_code,
    _generate_test_reports,
    _run_single_profile,
    _run_single_profile_in_batch,
    _run_suite_tests,
    _validate_all_profiles,
)

# Import low-level helpers from sysagent
from sysagent.utils.cli.filters import parse_filters
from sysagent.utils.cli.handlers import handle_interrupt
from sysagent.utils.config import list_profiles, setup_data_dir
from sysagent.utils.core import shared_state
from sysagent.utils.logging import setup_command_logging

# Import system info for CPU validation
from sysagent.utils.system.cache import SystemInfoCache
from sysagent.utils.system.cpu import is_generation_supported
from sysagent.utils.testing import create_pytest_args

logger = logging.getLogger(__name__)


# ESQ-specific: List of unsupported Intel processor generations
#
# PRODUCT COLLECTION REFERENCE:
# CPU Detection returns: (codename, generation, product_collection, segment)
#
# USAGE EXAMPLES:
# To block all Core Ultra Series 1:
#   "Core Ultra (Series 1)"
#
# To block only 4th Gen Xeon Scalable (but allow 4th Gen Xeon Workstation):
#   ("4th Gen", "Xeon Scalable")
#
# To block 3rd Gen Ice Lake-SP servers (but allow desktop variants if they existed):
#   ("3rd Gen", "Xeon Scalable", "server")
#
# To block specific codename (most precise - RECOMMENDED for Xeon W):
#   {"codename": "Tiger Lake-W", "product_collection": "Workstation"}
#
UNSUPPORTED_GENERATIONS = [
    # Core Ultra - older series
    "Core Ultra (Series 1)",  # Meteor Lake
    # Traditional Core - 14th Gen and older
    "14th Gen Core",  # Raptor Lake Refresh (RPL-S Refresh)
    "13th Gen Core",  # Raptor Lake
    "12th Gen Core",  # Alder Lake
    "11th Gen Core",  # Tiger Lake/Rocket Lake
    "10th Gen Core",  # Ice Lake/Comet Lake
    "9th Gen Core",  # Coffee Lake Refresh
    "8th Gen Core",  # Coffee Lake, Kaby Lake Refresh
    "7th Gen Core",  # Kaby Lake, Amber Lake, Whiskey Lake
    "6th Gen Core",  # Skylake
    "5th Gen Core",  # Broadwell
    "4th Gen Core",  # Haswell, Devil's Canyon
    "3rd Gen Core",  # Ivy Bridge
    "2nd Gen Core",  # Sandy Bridge, Westmere
    "1st Gen Core",  # Nehalem
    # Pre-Core i Series (before 2008)
    "Core 2",  # Core (2006), Penryn (2007) - Merom, Conroe, Kentsfield
    "Core (Yonah)",  # Enhanced Pentium M (2006) - First "Intel Core" branding
    "Pentium",  # Pentium 4, Pentium D, Pentium M, Pentium III, Pentium II, Pentium Pro, Pentium Gold, Pentium Silver
    "Celeron",  # All Celeron processors (entry-level)
    # Entry-level processors
    {"codename": "Alder Lake-N", "product_collection": "N-series"},  # Alder Lake-N only
    # Atom X-series (Embedded/IoT) - legacy products before x7000
    "Atom x6000",  # Elkhart Lake
    "Atom x5000",  # Apollo Lake
    "Atom (Cherry Trail)",  # Cherry Trail
    "Atom (Bay Trail)",  # Bay Trail - includes C-series (Avoton, Rangeley)
    "Atom (Avoton)",  # Avoton (C2xxx series - Silvermont)
    "Atom (Airmont)",  # Airmont architecture
    "Atom (Silvermont)",  # Silvermont architecture
    "Atom (Goldmont)",  # Goldmont architecture - Apollo Lake, Denverton
    "Atom (Goldmont Plus)",  # Goldmont Plus architecture - Gemini Lake
    # Atom Z-series (Mobile/Tablet) - all legacy
    "Atom Z-series",  # Clover Trail, Merrifield, Moorefield
    # Xeon W (Legacy Workstation) - pre-Sapphire Rapids generations
    # Using dict format with explicit codenames for consistency and precision
    # Product Collection: "Workstation"
    # Brand pattern: "Xeon W-" (uppercase W) with model numbers
    # Excludes: Sapphire Rapids WS ("w[digit]-" lowercase) which returns "4th Gen Xeon Workstation"
    # Dict format: {"codename": "X", "product_collection": "Y"}
    # Note: These entries block BOTH workstation AND embedded segments
    #       - Regular workstation: W-11955M (segment=workstation)
    #       - Embedded variants: W-11865MRE, W-1390E, W-3375RE (segment=embedded)
    #       Because segment is not specified in dict, it's not checked - only codename and product_collection
    {"codename": "Tiger Lake-W", "product_collection": "Workstation"},  # W-11xxx (e.g., W-11955M, W-11865MRE)
    {"codename": "Rocket/Comet Lake-W", "product_collection": "Workstation"},  # W-1xxx (e.g., W-1390, W-1390E)
    {"codename": "Ice/Cascade Lake-W", "product_collection": "Workstation"},  # W-3xxx (e.g., W-3375, W-3375RE)
    {"codename": "Cascade/Skylake-W", "product_collection": "Workstation"},  # W-2xxx (e.g., W-2295)
    {"codename": "Unknown Xeon W", "product_collection": "Workstation"},  # Fallback pattern
    # Additional string format entries for generation-based matching
    # (used when detection returns generation string instead of codename matching)
    "Tiger Lake Xeon W",  # Xeon W Tiger Lake variants (W-11xxx series)
    "Rocket/Comet Lake Xeon W",  # Xeon W Rocket/Comet Lake variants (W-1xxx series)
    "Ice/Cascade Lake Xeon W",  # Xeon W Ice/Cascade Lake variants (W-3xxx series)
    "Cascade/Skylake Xeon W",  # Xeon W Cascade/Skylake variants (W-2xxx series)
    "Legacy Xeon W",  # Fallback for unidentified Xeon W
    "Tiger Lake",  # Generic Tiger Lake processors (also covers non-W Tiger Lake)
    "Xeon W",  # Generic Xeon W processors not matched above
    # Xeon Scalable - Server product collection (4th Gen and older)
    # Using tuple format for scalability: (generation, product_collection)
    # This allows blocking specific product collections within a generation
    ("4th Gen", "Xeon Scalable"),  # Sapphire Rapids-SP/HBM (but allows 4th Gen Xeon Workstation)
    ("3rd Gen", "Xeon Scalable"),  # Ice Lake-SP
    ("2nd Gen", "Xeon Scalable"),  # Cascade Lake
    ("1st Gen", "Xeon Scalable"),  # Skylake-SP
    # Catch-all for unrecognized processors
    # Conservative approach: if we can't identify the processor, don't qualify it
    # This ensures only known, validated processors pass qualification
    "Unknown",  # Any processor with Unknown generation (detection failed or very old CPU)
    # Example of most specific format (currently not used, but available):
    # ("4th Gen", "Xeon Scalable", "server")  # Only blocks 4th Gen Xeon Scalable servers
]


def _check_system_validation_esq(force: bool = False, mode: str = "qualification") -> tuple:
    """
    ESQ-specific CPU validation for qualification profiles.

    Validates that the system meets Intel processor requirements for
    AI Edge System qualification. Behavior depends on run mode:
    - "all": If unsupported, offer to continue with remaining profiles only
    - "qualification": If unsupported, exit immediately

    Args:
        force: If True, skip interactive prompts
        mode: Run mode - "all" or "qualification"

    Returns:
        tuple: (is_cpu_supported, should_continue, skip_qualification)
    """
    try:
        # Load hardware info using SystemInfoCache
        cache = SystemInfoCache()
        hw_info = cache.get_hardware_info()

        if not hw_info or "cpu" not in hw_info:
            logger.debug("No hardware cache found, skipping CPU validation")
            return True, True, False

        cpu_info = hw_info.get("cpu", {})
        generation_info = cpu_info.get("generation_info", {})
        cpu_generation = generation_info.get("generation", "Unknown")
        product_collection = generation_info.get("product_collection")
        segment = generation_info.get("segment")
        codename = generation_info.get("codename")
        cpu_brand = cpu_info.get("brand", "Unknown")

        # Check for developer mode
        developer_mode = os.environ.get("DEVELOPER_MODE", "0").lower() in ["1", "true", "yes"]
        if developer_mode:
            logger.warning("[DEVELOPER MODE] System validation bypassed")
            return True, True, False

        # Check if CPU is supported for ESQ qualification
        # Now supports dict format with codename: {"codename": "X", "product_collection": "Y"}
        is_supported = is_generation_supported(
            cpu_generation,
            supported_generations=None,  # Auto-support new generations
            unsupported_generations=UNSUPPORTED_GENERATIONS,
            product_collection=product_collection,
            segment=segment,
            codename=codename,
        )

        if is_supported:
            # CPU supported - show informational message
            message = f"""
System: {cpu_brand} - {cpu_generation}
""".strip()
            logger.info(message)
            return True, True, False
        else:
            # CPU not supported - behavior depends on mode
            if mode == "all":
                # --all mode: Offer to continue with remaining profiles
                message = f"""
System: {cpu_brand} - {cpu_generation}

System NOT supported for qualification profiles.

Refer to the documentation for supported hardware and system requirements.
However, you can still run remaining profiles.
""".strip()
                print(message)

                if not force:
                    try:
                        response = input("Continue with remaining profiles? (Y/n) ").strip().lower()
                        should_continue = response in ["y", "yes", ""]
                        if should_continue:
                            logger.info("Continuing with remaining profiles (qualification skipped)")
                            return False, True, True  # Skip qualification
                        else:
                            logger.info("Execution cancelled by user")
                            return False, False, True
                    except (KeyboardInterrupt, EOFError):
                        logger.info("Execution cancelled")
                        return False, False, True
                else:
                    # Force mode: continue with remaining profiles
                    logger.warning("Continuing with remaining profiles (--force flag, qualification skipped)")
                    return False, True, True
            else:
                # qualification mode: Exit immediately
                message = f"""
System: {cpu_brand} - {cpu_generation}

System NOT supported for qualification profiles.

Refer to the documentation for supported hardware and system requirements.
""".strip()
                print(message)
                return False, False, True

    except Exception as e:
        logger.debug(f"Error in system validation: {e}")
        return True, True, False  # Continue on error


def _prompt_run_configuration_esq(force: bool = False, vertical_profile_names: list = None) -> tuple:
    """
    ESQ-specific unified prompt for run configuration.

    Handles both system validation and profile selection in a single prompt
    that adapts based on Intel CPU compatibility status.

    Args:
        force: If True, skip prompt and use default behavior
        vertical_profile_names: List of vertical profile names

    Returns:
        tuple: (is_cpu_supported, should_continue, skip_qualification, skip_vertical)
    """
    try:
        # Load hardware info using SystemInfoCache
        cache = SystemInfoCache()
        hw_info = cache.get_hardware_info()

        if not hw_info or "cpu" not in hw_info:
            logger.debug("No hardware cache found, assuming compatible system")
            return True, True, False, False

        cpu_info = hw_info.get("cpu", {})
        generation_info = cpu_info.get("generation_info", {})
        cpu_generation = generation_info.get("generation", "Unknown")
        product_collection = generation_info.get("product_collection")
        segment = generation_info.get("segment")
        codename = generation_info.get("codename")
        cpu_brand = cpu_info.get("brand", "Unknown")

        # Check for developer mode
        developer_mode = os.environ.get("DEVELOPER_MODE", "0").lower() in ["1", "true", "yes"]
        if developer_mode:
            logger.warning("[DEVELOPER MODE] System compatibility check bypassed")
            return True, True, False, False

        # Check if CPU is supported for ESQ qualification
        # Now supports dict format with codename: {"codename": "X", "product_collection": "Y"}
        is_supported = is_generation_supported(
            cpu_generation,
            supported_generations=None,
            unsupported_generations=UNSUPPORTED_GENERATIONS,
            product_collection=product_collection,
            segment=segment,
            codename=codename,
        )

        if is_supported:
            logger.debug(f"CPU generation '{cpu_generation}' is supported for qualification profiles")
        else:
            logger.debug(f"CPU generation '{cpu_generation}' is NOT supported for qualification profiles")

        # If force flag is set, use default behavior without prompting
        if force:
            if is_supported:
                # Supported: default is to run all profiles (qualification + vertical)
                logger.info(
                    "System supported for qualification (--force flag). "
                    "Running all profiles (qualification + vertical)."
                )
                return True, True, False, False
            else:
                # Not supported: default is to continue with remaining profiles
                logger.warning(
                    "System not supported for qualification (--force flag). Continuing with remaining profiles."
                )
                return False, True, True, False

        # Format vertical profile list
        vertical_list = ""
        if vertical_profile_names:
            vertical_list = "\n".join(f"    - {name}" for name in sorted(vertical_profile_names))

        # Select appropriate message based on CPU support status
        if is_supported:
            message = f"""
System: {cpu_brand} - {cpu_generation}

The following test profiles will be executed:
  • Qualification profile
  • Vertical profiles

Available vertical profiles:
{vertical_list if vertical_list else "    None"}

""".strip()
            prompt_text = "Skip vertical profiles? (y/N)"
        else:
            message = f"""
System: {cpu_brand} - {cpu_generation}

System NOT supported for qualification profiles.

Refer to the documentation for supported hardware and system requirements.
However, you can still run remaining profiles.

Available vertical profiles:
{vertical_list if vertical_list else "    None"}

""".strip()
            prompt_text = "Skip vertical profiles? (y/N)"

        print(message)

        try:
            response = input(prompt_text + " ").strip().lower()

            if is_supported:
                # Supported system: asking about vertical profiles (negative question, default=no)
                skip_vertical = response in ["y", "yes"]
                logger.info(f"User chose to {'skip' if skip_vertical else 'include'} vertical profiles")
                return True, True, False, skip_vertical
            else:
                # Not supported: asking whether to skip (negative question, default=no/continue)
                should_skip = response in ["y", "yes"]
                if should_skip:
                    logger.info("User chose to skip vertical profiles. Exiting.")
                    return False, False, True, False
                else:
                    logger.info("User chose to continue with vertical profiles")
                    return False, True, True, False

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Exiting.")
            return False, False, True, False
        except EOFError:
            logger.info("EOF encountered, using default option")
            if is_supported:
                return True, True, False, False  # Include vertical
            else:
                return False, False, True, False  # Exit

    except Exception as e:
        logger.warning(f"Failed to process run configuration: {e}")
        return True, True, False, False


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
    run_all_profiles: bool = False,
    qualification_only: bool = False,
    force: bool = False,
    no_mask: bool = False,
    set_prompt: List[str] = None,
    extra_args: List[str] = None,
) -> int:
    """
    ESQ-specific run command with Intel processor validation.

    Implements complete test execution flow with ESQ-specific prompts
    for CPU compatibility and qualification requirements.

    Args:
        Same as sysagent.utils.cli.commands.run.run_tests

    Returns:
        int: Exit code (0 for success, non-zero for failure)
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

    # Reset interrupt flags
    shared_state.INTERRUPT_OCCURRED = False
    shared_state.INTERRUPT_SIGNAL = None
    shared_state.INTERRUPT_SIGNAL_NAME = "Unknown"

    # Register global interrupt handler
    original_sigint_handler = signal.signal(signal.SIGINT, handle_interrupt)
    if "ACTIVE_PROFILE" in os.environ:
        del os.environ["ACTIVE_PROFILE"]
    if "ACTIVE_PROFILE_HIGHEST_TIER" in os.environ:
        del os.environ["ACTIVE_PROFILE_HIGHEST_TIER"]

    # Validate arguments
    if sub_suite_name and not suite_name:
        logger.error("Error: --sub-suite option requires --suite option to be specified")
        return 1
    if test_name and not sub_suite_name:
        logger.error("Error: --test option requires --sub-suite option to be specified")
        return 1

    # Parse filters
    parsed_filters = {}
    if filters:
        try:
            parsed_filters = parse_filters(filters)
            logger.info(f"Applying test filters: {parsed_filters}")
        except ValueError as e:
            logger.error(f"Invalid filter format: {e}")
            return 1
        if not profile_name:
            logger.error("Error: --filter option can only be used with --profile option")
            return 1

    # Setup directories and logging
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
        # Route to appropriate execution mode
        if profile_name:
            # Run specific profile with ESQ validation
            result_code, tests_ran = _run_profile_tests_esq(
                profile_name, pytest_args, skip_system_check, data_dir, verbose, debug, parsed_filters, force
            )
        elif suite_name:
            # Run suite/test directly (no prompts)
            result_code, tests_ran = _run_suite_tests(suite_name, sub_suite_name, test_name, pytest_args)
        else:
            # Run all profiles with ESQ prompts
            result_code, tests_ran = _run_all_profiles_esq(
                skip_system_check,
                data_dir,
                verbose,
                debug,
                run_all_profiles,
                qualification_only,
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

        # Check for interrupts
        if interrupt_occurred or shared_state.INTERRUPT_OCCURRED:
            logger.warning("Test execution was interrupted by user")

        if tests_ran:
            _generate_test_reports(data_dir, verbose, debug)
            result_code = _determine_final_exit_code(data_dir, result_code)

    return result_code


def _run_profile_tests_esq(
    profile_name: str,
    pytest_args: List[str],
    skip_system_check: bool,
    data_dir: str,
    verbose: bool = False,
    debug: bool = False,
    filters: Dict[str, Any] = None,
    force: bool = False,
) -> tuple:
    """
    ESQ-specific profile execution with Intel CPU validation.

    Returns:
        tuple: (exit_code, tests_ran)
    """
    from sysagent.utils.config import expand_profile_with_dependencies, get_profile_dependencies

    # Get all available profiles
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

    # ESQ-specific: Check if qualification profile and validate CPU
    profile_config = all_profiles_dict[profile_name]
    profile_labels = profile_config.get("params", {}).get("labels", {})
    profile_type = profile_labels.get("type", "")

    # Only run CPU validation for qualification profiles
    if profile_type == "qualification" and not skip_system_check:
        is_cpu_supported, should_continue, skip_qual = _check_system_validation_esq(force, mode="qualification")
        if not should_continue:
            return 1, False

    # Expand profile with dependencies
    try:
        execution_order = expand_profile_with_dependencies(profile_name, all_profiles_dict)
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

    # Execute profiles in dependency order (reuse sysagent's generic execution)
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
            filters if current_profile_name == profile_name else None,
        )
        tests_ran = tests_ran or profile_tests_ran
        if result_code != 0 and current_profile_name != profile_name:
            logger.warning(
                f"Dependency profile '{current_profile_name}' completed with exit code {result_code}. "
                f"Continuing to execute main profile '{profile_name}'."
            )
        elif result_code != 0 and current_profile_name == profile_name:
            final_exit_code = result_code

    return final_exit_code, tests_ran


def _run_all_profiles_esq(
    skip_system_check: bool,
    data_dir: str,
    verbose: bool,
    debug: bool,
    run_all_profiles: bool = False,
    qualification_only: bool = False,
    force: bool = False,
    prompt_overrides: dict = None,
) -> tuple:
    """
    ESQ-specific all-profiles execution with Intel CPU validation prompts.

    Returns:
        tuple: (exit_code, tests_ran)
    """
    # Import sysagent utilities for profile handling
    from sysagent.utils.config import (
        expand_profile_with_dependencies,
        get_profile_dependencies,
        resolve_profile_dependencies,
        validate_profile_dependencies,
    )

    all_profiles = list_profiles(include_examples=False)
    logger.debug(f"Found {sum(len(profiles) for profiles in all_profiles.values())} profiles")

    # Build complete profiles dictionary (reuse sysagent pattern)
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

    # Collect vertical profile names for prompt
    vertical_profile_names = []
    for profile_type, profiles in all_profiles.items():
        for profile in profiles:
            configs = profile.get("configs")
            if configs:
                profile_name = configs.get("name")
                if profile_name:
                    params = configs.get("params", {})
                    labels = params.get("labels", {})
                    profile_label_type = labels.get("type", "")
                    is_vertical = profile_type == "verticals" or profile_label_type == "vertical"
                    if is_vertical:
                        vertical_profile_names.append(profile_name)

    # Initialize defaults
    skip_vertical_profiles = False
    skip_qualification = False

    # ESQ-specific prompt handling
    if run_all_profiles:
        # --all flag: Run all profile types with ESQ validation
        if not skip_system_check:
            is_cpu_supported, should_continue, skip_qual = _check_system_validation_esq(force, mode="all")
            if not should_continue:
                return 1, False
            skip_qualification = skip_qual

        include_all_types = True
        skip_vertical_profiles = False
        logger.info("Running all profile types (qualifications, verticals, suites)")

    elif qualification_only:
        # --qualification-only: Run qualification with ESQ validation
        if not skip_system_check:
            is_cpu_supported, should_continue, skip_qual = _check_system_validation_esq(force, mode="qualification")
            if not should_continue:
                return 1, False

        include_all_types = False
        skip_vertical_profiles = True
        logger.info("Running qualification profiles only")

    else:
        # Default mode: Show ESQ unified prompt
        include_all_types = False
        try:
            is_cpu_supported, should_continue, skip_qual, skip_vert = _prompt_run_configuration_esq(
                force, vertical_profile_names
            )
            skip_qualification = skip_qual
            skip_vertical_profiles = skip_vert

            if not should_continue:
                logger.info("Exiting as requested")
                return 1, False

            if skip_qualification and skip_vertical_profiles:
                logger.error("Both qualification and vertical profiles skipped - nothing to run")
                return 1, False
            elif skip_qualification:
                logger.warning("CPU not supported - running suite and vertical profiles only")
            elif skip_vertical_profiles:
                logger.info("Running qualification profiles only (vertical profiles skipped by user)")
            else:
                logger.info("Running qualification and vertical profiles")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            sys.exit(1)

    # Collect requested profiles based on filter
    requested_profile_names = []
    for profile_type, profiles in all_profiles.items():
        for profile in profiles:
            configs = profile.get("configs")
            if configs:
                profile_name = configs.get("name")
                if profile_name:
                    params = configs.get("params", {})
                    labels = params.get("labels", {})
                    profile_label_type = labels.get("type", "")
                    is_qualification = profile_type == "qualifications" or profile_label_type == "qualification"

                    if include_all_types:
                        if is_qualification and skip_qualification:
                            logger.debug(
                                f"Skipping qualification profile due to platform incompatibility: {profile_name}"
                            )
                            continue
                        requested_profile_names.append(profile_name)
                    else:
                        is_vertical = profile_type == "verticals" or profile_label_type == "vertical"
                        if is_qualification and not skip_qualification:
                            requested_profile_names.append(profile_name)
                        elif is_vertical and not skip_vertical_profiles:
                            requested_profile_names.append(profile_name)

    if not requested_profile_names:
        logger.warning("No profiles selected for execution")
        return 0, False

    # Expand profiles with dependencies (reuse sysagent logic)
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

    # Build final profile items and dict from the complete set (reuse sysagent pattern)
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

    # Validate profile dependencies (reuse sysagent utility)
    dep_errors = validate_profile_dependencies(all_profiles_dict)
    if dep_errors:
        logger.warning("Profile dependency validation warnings:")
        for error in dep_errors:
            logger.warning(f"  - {error}")

    # Resolve execution order based on dependencies (CORRECT argument order)
    try:
        execution_order = resolve_profile_dependencies(all_profiles_dict)  # Fixed: only pass profiles dict
        logger.info("Profile execution order (respecting dependencies):")
        for i, profile_name in enumerate(execution_order, 1):
            prefix = "  └─" if i == len(execution_order) else "  ├─"
            logger.info(f"{prefix} {profile_name}")
    except Exception as e:
        logger.error(f"Failed to resolve profile dependencies: {e}")
        logger.info("Falling back to alphabetical order")
        execution_order = sorted(all_profiles_dict.keys())

    # Validate all profiles (reuse sysagent's generic validator)
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

    # Create mapping of profile names to (profile_type, profile) tuples (reuse sysagent pattern)
    valid_profiles_map = {}
    for profile_type, profile in valid_profiles:
        configs = profile.get("configs")
        if configs:
            profile_name = configs.get("name")
            if profile_name:
                valid_profiles_map[profile_name] = (profile_type, profile)

    # Run profiles in dependency order (only those that are valid) (reuse sysagent's batch execution)
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
