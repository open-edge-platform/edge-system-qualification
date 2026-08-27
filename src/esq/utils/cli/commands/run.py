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
from typing import Any

# Import sysagent's low-level execution functions (generic, reusable)
from sysagent.utils.cli.commands.run import (
    TIER_SKIP_EXIT,
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
    # Specific Celeron series (older generations)
    "Celeron J-series",  # Celeron J-series (desktop, 2013-2021): J1xxx-J6xxx (Bay Trail through Elkhart Lake)
    "Celeron N-series",  # Celeron N-series (mobile, 4-digit, 2013-2021): N2xxx-N6xxx
    "Celeron G-series",  # Celeron G-series (desktop, 2011-2022): Gxxx-G6xxx (Sandy Bridge through Alder Lake)
    # Specific Pentium series (older generations)
    "Pentium Silver J-series",  # Pentium Silver J-series (desktop, 2017-2019): J4xxx-J5xxx (Apollo Lake, Gemini Lake)
    "Pentium Silver N-series",  # Pentium Silver N-series (mobile, 2020-2021): N5xxx-N6xxx (Jasper Lake)
    "Pentium Gold G-series",  # Pentium Gold G-series (desktop, 2017-2022): Gxxxx-G7xxx (Kaby Lake throughlder Lake)
    # Entry-level processors
    {
        "codename": "Alder Lake-N",
        "product_collection": "N-series",
    },  # Alder Lake-N (2022): Intel N-series (N305, N200, N100, N97, N95, N50)
    {
        "codename": "Alder Lake-N",
        "product_collection": "Atom",
    },  # Alder Lake-N (2022): Atom x7000 E-suffix (x7425E, x7213E, x7211E)
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
    # ============================================================================
    # XEON 6 (2024+) - PARTIAL SUPPORT
    # ============================================================================
    # Xeon 6 is Intel's current-generation server/workstation processor line
    # Launched: April 2024 (replaces "Xeon Scalable" branding)
    # Product Collection: "Xeon 6"
    # Microarchitectures:
    #   - Granite Rapids (P-cores): SUPPORTED - Server (6XXXP) and Workstation (678X, 677X)
    #   - Sierra Forest (E-cores): UNSUPPORTED - Server (6XXXE) - efficiency cores
    # SUPPORT POLICY:
    #   - Granite Rapids P-cores: QUALIFIED for ESQ testing (AI edge workloads)
    #   - Sierra Forest E-cores: NOT QUALIFIED (designed for cloud density workloads)
    # ============================================================================
    {"codename": "Sierra Forest"},  # Xeon 6 E-core processors (6XXXE models) - NOT supported for edge AI qualification
    # Xeon Scalable - Server product collection (4th Gen and older)
    # Using tuple format for scalability: (generation, product_collection)
    # This allows blocking specific product collections within a generation
    ("4th Gen", "Xeon Scalable"),  # Sapphire Rapids-SP/HBM (but allows 4th Gen Xeon Workstation)
    ("3rd Gen", "Xeon Scalable"),  # Ice Lake-SP
    ("2nd Gen", "Xeon Scalable"),  # Cascade Lake
    ("1st Gen", "Xeon Scalable"),  # Skylake-SP
    # ============================================================================
    # FORWARD COMPATIBILITY - "Unknown" generation is NO LONGER unsupported
    # ============================================================================
    # NOTE: Removed "Unknown" from unsupported list to allow CLI to run on newer
    # processors that haven't been added to detection logic yet. The CPU detection
    # now attempts to infer if a CPU is newer (e.g., Core Ultra Series 4+, newer
    # Xeon 6/7/8 models) and assumes such processors are supported. Intel has moved
    # to "Core Ultra" branding, so no 15th Gen+ Core i-series processors are expected.
    # Only explicitly listed old generations are blocked. This ensures forward
    # compatibility with future Intel platforms while maintaining qualification
    # rigor for known older platforms.
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


def _prompt_run_configuration_esq(
    force: bool = False,
    vertical_profile_names: list = None,
    qualification_profiles: list = None,
) -> tuple:
    """
    ESQ-specific unified prompt for run configuration.

    Lists available qualification profiles for selection (scalable to any number of
    qualification types, instead of always assuming a single hardcoded profile) alongside
    vertical profile inclusion, adapting based on Intel CPU compatibility status. Kept
    concise - full options remain available via `esq run --help` (--all, --profile, --tag).

    There is no implicit "default" qualification profile. Exactly one profile must be
    explicitly chosen by number. Any other response terminates the run rather than
    silently falling back to a preset profile, since each qualification profile can
    have different dependencies/system setup requirements.

    Vertical profile inclusion is opt-in per qualification profile (via the profile's
    `vertical_profiles` param, a list of vertical profile names) rather than a single
    yes/no that applies to every qualification profile. Only the selected qualification's
    own associated verticals (if any) are offered; if it declares none, nothing is prompted
    and no vertical profiles are included.

    Args:
        force: If True, skip prompt (non-interactive). Qualification only runs when
            explicitly requested via --profile/--tag/--qualification-only.
        vertical_profile_names: List of all vertical profile names (used only for the
            unsupported-system fallback prompt, where no qualification is selected).
        qualification_profiles: List of dicts describing selectable qualification profiles
            (profiles with the "hidden" label set are already excluded by the caller):
            {"name", "display_name", "tags", "vertical_profiles"}

    Returns:
        tuple: (is_cpu_supported, should_continue, selected_qualification_names, selected_vertical_names)
        selected_qualification_names is a list containing zero or exactly one qualification
        profile name to run. selected_vertical_names is the list of vertical profile names
        (possibly empty) to run alongside it.
    """
    qualification_profiles = qualification_profiles or []

    try:
        # Load hardware info using SystemInfoCache
        cache = SystemInfoCache()
        hw_info = cache.get_hardware_info()

        if not hw_info or "cpu" not in hw_info:
            logger.debug("No hardware cache found, assuming compatible system")
            return True, True, [], []

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
            return True, True, [], []

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

        # If force flag is set, skip prompting entirely. There is no implicit default
        # qualification profile: use --profile/--tag/--qualification-only to run one.
        if force:
            if is_supported:
                logger.info(
                    "System supported for qualification (--force flag). No qualification profile auto-selected; "
                    "use --profile/--tag/--qualification-only to run one. Continuing with vertical profiles."
                )
                return True, True, [], []
            else:
                logger.warning(
                    "System not supported for qualification (--force flag). Continuing with remaining profiles."
                )
                return False, True, [], []

        vertical_list = (
            "\n".join(f"    - {name}" for name in sorted(vertical_profile_names))
            if vertical_profile_names
            else "    None"
        )

        print(f"System: {cpu_brand} - {cpu_generation}\n")

        if not is_supported:
            print(
                "System NOT supported for qualification profiles.\n"
                "Refer to the documentation for supported hardware and system requirements.\n"
                "You can still run remaining (vertical) profiles.\n"
            )
            print(f"Available vertical profiles:\n{vertical_list}\n")
            try:
                response = input("Continue with vertical profiles? (Y/n) ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                logger.info("Interrupted by user. Exiting.")
                return False, False, [], []

            if response in ["n", "no"]:
                logger.info("User chose to skip vertical profiles. Exiting.")
                return False, False, [], []
            logger.info("User chose to continue with vertical profiles")
            return False, True, [], list(vertical_profile_names or [])

        # Supported system: the user must pick exactly ONE qualification profile to run,
        # since different profiles can have different dependencies/system setup. There is
        # no default and no multi-select here - an invalid response terminates the run
        # instead of guessing. Use --all/-a to run everything for data collection instead.
        selected_qualification_names = []
        selected_qualification = None
        if qualification_profiles:
            print("Qualification profiles:")
            for i, q in enumerate(qualification_profiles, 1):
                print(f"  {i}) {q['display_name']}")

            valid_range = f"1-{len(qualification_profiles)}"
            try:
                response = input(f"\nSelect a qualification profile [{valid_range}]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                logger.info("Interrupted by user. Exiting.")
                return False, False, [], []

            if response.isdigit() and 1 <= int(response) <= len(qualification_profiles):
                selected_qualification = qualification_profiles[int(response) - 1]
                selected_qualification_names = [selected_qualification["name"]]
            else:
                logger.error(f"Invalid selection '{response}' - expected a number ({valid_range}). Exiting.")
                return True, False, [], []

        # Vertical profiles are opt-in per qualification profile. Only prompt for the
        # verticals the selected qualification itself declares; skip silently (no
        # prompt, nothing included) if it declares none.
        selected_vertical_names = []
        associated_verticals = selected_qualification.get("vertical_profiles", []) if selected_qualification else []
        if associated_verticals:
            associated_list = "\n".join(f"    - {name}" for name in sorted(associated_verticals))
            print(f"\nVertical profiles for this qualification:\n{associated_list}\n")
            try:
                response = input("Include these vertical profile(s) as well? (Y/n) ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                logger.info("Interrupted by user. Exiting.")
                return True, False, [], []

            if response in ["n", "no"]:
                logger.info("User chose to skip the associated vertical profile(s)")
            else:
                selected_vertical_names = associated_verticals
                logger.info("User chose to include the associated vertical profile(s)")

        print("Tip: use --profile/-p or --tag/-t to skip this prompt. See 'esq run --help' for all options.\n")
        return True, True, selected_qualification_names, selected_vertical_names

    except Exception as e:
        logger.warning(f"Failed to process run configuration: {e}")
        return True, True, [], []


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
    run_all_profiles: bool = False,
    qualification_only: bool = False,
    force: bool = False,
    no_mask: bool = False,
    set_prompt: list[str] = None,
    extra_args: list[str] = None,
    telemetry_interval: int = None,
    tags: list[str] = None,
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

    if telemetry_interval is not None:
        if telemetry_interval >= 1:
            os.environ["CORE_TELEMETRY_INTERVAL"] = str(telemetry_interval)
            logger.info("Telemetry interval overridden via CLI: %ds", telemetry_interval)
        else:
            logger.warning("--telemetry-interval must be >= 1 second; ignoring value %d", telemetry_interval)

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
    if profile_name and tags:
        logger.error("Error: --profile and --tag/-t options cannot be used together")
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
        if not profile_name and not tags:
            logger.error("Error: --filter option can only be used with --profile or --tag option")
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
                profile_name,
                pytest_args,
                skip_system_check,
                data_dir,
                verbose,
                debug,
                parsed_filters,
                force,
                qualification_only,
            )
        elif tags:
            # Run profile(s) resolved from tag(s) with ESQ validation
            result_code, tests_ran = _run_tagged_profiles_esq(
                tags,
                pytest_args,
                skip_system_check,
                data_dir,
                verbose,
                debug,
                parsed_filters,
                force,
                qualification_only,
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

        # Clean up telemetry interval override (only if set by this invocation)
        if telemetry_interval is not None and "CORE_TELEMETRY_INTERVAL" in os.environ:
            del os.environ["CORE_TELEMETRY_INTERVAL"]

        # Check for interrupts
        if interrupt_occurred or shared_state.INTERRUPT_OCCURRED:
            logger.warning("Test execution was interrupted by user")

        # TIER_SKIP_EXIT is the specific sentinel for "no tier matched" - distinct from
        # validation failure (exit 1) or other early aborts.
        all_skipped = result_code == TIER_SKIP_EXIT

        # Generate reports when tests ran, OR when tier-skip (so system hardware info is
        # available for review). Do NOT generate for other failures (e.g. profile validation).
        if tests_ran or all_skipped:
            _generate_test_reports(data_dir, verbose, debug)
            result_code = _determine_final_exit_code(data_dir, result_code)

        if all_skipped:
            logger.error("No qualification tests ran - all profile(s) were skipped: no system tier matched")
            logger.error("Check system hardware compatibility: run 'esq info' or review the report above")
            result_code = 1

    return result_code


def _run_profile_tests_esq(
    profile_name: str,
    pytest_args: list[str],
    skip_system_check: bool,
    data_dir: str,
    verbose: bool = False,
    debug: bool = False,
    filters: dict[str, Any] = None,
    force: bool = False,
    qualification_only: bool = False,
) -> tuple:
    """
    ESQ-specific profile execution with Intel CPU validation.

    Returns:
        tuple: (exit_code, tests_ran)
    """
    return _resolve_and_execute_profiles_esq(
        [profile_name], pytest_args, skip_system_check, data_dir, verbose, debug, filters, force, qualification_only
    )


def _run_tagged_profiles_esq(
    tags: list[str],
    pytest_args: list[str],
    skip_system_check: bool,
    data_dir: str,
    verbose: bool = False,
    debug: bool = False,
    filters: dict[str, Any] = None,
    force: bool = False,
    qualification_only: bool = False,
) -> tuple:
    """
    ESQ-specific execution of profile(s) resolved from short tag/keyword(s).

    Multiple tags may resolve to multiple profiles; results are deduplicated
    and dependency-resolved (with proper priority ordering) via the same
    shared executor used for explicit --profile runs.

    Returns:
        tuple: (exit_code, tests_ran)
    """
    from sysagent.utils.config import get_profiles_matching_tags

    all_profiles_data = list_profiles(include_examples=True)
    all_profiles_dict = {}
    for profile_type, profiles in all_profiles_data.items():
        for profile in profiles:
            configs = profile.get("configs")
            if configs:
                profile_name_key = configs.get("name")
                if profile_name_key:
                    all_profiles_dict[profile_name_key] = configs

    matched_profiles = get_profiles_matching_tags(tags, all_profiles_dict)
    if not matched_profiles:
        logger.error(f"No profiles found matching tag(s): {', '.join(tags)}")
        return 1, False

    logger.info(f"Tag(s) {', '.join(tags)} matched profile(s): {', '.join(matched_profiles)}")
    return _resolve_and_execute_profiles_esq(
        matched_profiles,
        pytest_args,
        skip_system_check,
        data_dir,
        verbose,
        debug,
        filters,
        force,
        qualification_only,
    )


def _resolve_and_execute_profiles_esq(
    requested_profile_names: list[str],
    pytest_args: list[str],
    skip_system_check: bool,
    data_dir: str,
    verbose: bool = False,
    debug: bool = False,
    filters: dict[str, Any] = None,
    force: bool = False,
    qualification_only: bool = False,
) -> tuple:
    """
    Resolve dependencies for one or more explicitly requested profiles and execute them.

    Shared by --profile and --tag execution: expands each requested profile with its
    dependencies, deduplicates the combined set, and resolves a single dependency-priority
    execution order (avoiding redundant re-runs of shared dependencies). Filters only apply
    to the explicitly requested profiles, not to their dependencies.

    Returns:
        tuple: (exit_code, tests_ran)
    """
    from sysagent.utils.config import expand_profile_with_dependencies, resolve_profile_dependencies

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

    # Check that all requested profiles exist
    missing = [name for name in requested_profile_names if name not in all_profiles_dict]
    if missing:
        logger.error(f"Profile(s) not found: {', '.join(missing)}")
        return 1, False

    # ESQ-specific: Validate CPU once if any requested profile is a qualification
    # profile that opts-in to the system compatibility check.
    requires_system_compatibility_check = any(
        all_profiles_dict[name].get("params", {}).get("labels", {}).get("type") == "qualification"
        and all_profiles_dict[name].get("params", {}).get("labels", {}).get("system_compatibility", False)
        for name in requested_profile_names
    )
    if requires_system_compatibility_check and not skip_system_check:
        is_cpu_supported, should_continue, skip_qual = _check_system_validation_esq(force, mode="qualification")
        if not should_continue:
            return 1, False

    # Pre-validate each requested profile's own system requirements before expanding
    # dependencies, so a dependency profile doesn't run pointlessly when the profile
    # that actually needs it can't meet its own requirements.
    viable_requested_names = requested_profile_names
    if not skip_system_check:
        from sysagent.utils.testing.profile_validator import validate_profile_requirements

        viable_requested_names = []
        for profile_name in requested_profile_names:
            validation_result = validate_profile_requirements(
                all_profiles_dict[profile_name], profile_name=profile_name
            )
            if validation_result.get("passed", False):
                viable_requested_names.append(profile_name)
            else:
                logger.error(
                    f"Profile '{profile_name}' does not meet system requirements - skipping it and its dependencies"
                )

        if not viable_requested_names:
            logger.error("No requested profile(s) meet system requirements - nothing to run")
            return 1, False

    for profile_name in list(viable_requested_names):
        labels = all_profiles_dict[profile_name].get("params", {}).get("labels", {})
        if labels.get("type") != "qualification":
            continue
        associated_verticals = [
            name
            for name in (all_profiles_dict[profile_name].get("params", {}).get("vertical_profiles") or [])
            if name in all_profiles_dict and name not in viable_requested_names
        ]
        if not associated_verticals:
            continue

        display_name = labels.get("profile_display_name", profile_name)
        if qualification_only:
            logger.info(f"Skipping vertical profile(s) for '{display_name}' (--qualification-only flag)")
            continue
        if force:
            logger.info(f"Including vertical profile(s) for '{display_name}' (--force flag)")
            viable_requested_names.extend(associated_verticals)
            continue

        associated_list = "\n".join(f"    - {name}" for name in sorted(associated_verticals))
        print(f"\nVertical profiles for '{display_name}':\n{associated_list}\n")
        try:
            response = input("Include these vertical profile(s) as well? (Y/n) ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            logger.info("Interrupted by user. Exiting.")
            return 1, False

        if response in ["n", "no"]:
            logger.info(f"Skipping the associated vertical profile(s) for '{display_name}'")
        else:
            logger.info(f"Including the associated vertical profile(s) for '{display_name}'")
            viable_requested_names.extend(associated_verticals)

    # Expand each requested profile with its dependencies; a dict naturally
    # dedupes profiles shared across multiple requested profiles.
    required_profiles: dict[str, Any] = {}
    for profile_name in viable_requested_names:
        try:
            for expanded_name in expand_profile_with_dependencies(profile_name, all_profiles_dict):
                required_profiles[expanded_name] = all_profiles_dict[expanded_name]
        except Exception as e:
            logger.error(f"Failed to resolve dependencies for profile '{profile_name}': {e}")
            return 1, False

    # Resolve a single execution order across the combined set (dependencies first)
    try:
        execution_order = resolve_profile_dependencies(required_profiles)
    except Exception as e:
        logger.error(f"Failed to resolve profile execution order: {e}")
        return 1, False

    requested_set = set(viable_requested_names)
    logger.info("Execution order:")
    for i, prof in enumerate(execution_order, 1):
        prefix = "  └─" if i == len(execution_order) else "  ├─"
        suffix = " (requested)" if prof in requested_set else " (dependency)"
        logger.info(f"{prefix} {prof}{suffix}")

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
            filters if current_profile_name in requested_set else None,
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
        get_profile_tags,
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

    # Collect vertical profile names and qualification profile metadata for prompting.
    # Qualification metadata is collected generically (not hardcoded to one profile) so
    # the interactive prompt scales automatically as new qualification profiles are added.
    vertical_profile_names = []
    qualification_profiles_meta = []
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
                    is_qualification = profile_type == "qualifications" or profile_label_type == "qualification"
                    if is_vertical:
                        vertical_profile_names.append(profile_name)
                    elif is_qualification and not labels.get("hidden", False):
                        # "hidden" is a generic, reusable label (not prompt-listing-specific)
                        # for excluding a profile from interactive discovery while still
                        # allowing it to run explicitly via --profile/--tag.
                        qualification_profiles_meta.append(
                            {
                                "name": profile_name,
                                "display_name": labels.get("profile_display_name", profile_name),
                                "tags": get_profile_tags(configs),
                                # Vertical profiles this qualification opts in to prompting for.
                                # Absent/empty means no vertical prompt for this qualification.
                                "vertical_profiles": params.get("vertical_profiles") or [],
                            }
                        )
    qualification_profiles_meta.sort(key=lambda q: q["display_name"])

    # Initialize defaults
    skip_vertical_profiles = False
    skip_qualification = False
    # Explicit qualification selection from the interactive default prompt. Remains None
    # for --all/--qualification-only flag-driven runs so their existing behavior (based on
    # skip_qualification gating) is left untouched.
    interactive_qualification_selection = None
    # Specific vertical profile names accepted alongside the interactive qualification
    # selection (opt-in per qualification profile, not a blanket all-verticals toggle).
    interactive_vertical_selection = []

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
        # --qualification-only: Run qualification profiles with ESQ validation.
        # Use mode="all" so that on an unsupported CPU the user is offered the
        # chance to continue running qualification profiles that do NOT require
        # the ESQ-level system compatibility check.
        if not skip_system_check:
            is_cpu_supported, should_continue, skip_qual = _check_system_validation_esq(force, mode="all")
            if not should_continue:
                return 1, False
            skip_qualification = skip_qual

        include_all_types = False
        skip_vertical_profiles = True
        logger.info("Running qualification profiles only")

    else:
        # Default mode: Show ESQ unified prompt
        include_all_types = False
        try:
            (
                is_cpu_supported,
                should_continue,
                selected_qualification_names,
                selected_vertical_names,
            ) = _prompt_run_configuration_esq(force, vertical_profile_names, qualification_profiles_meta)
            interactive_qualification_selection = selected_qualification_names
            interactive_vertical_selection = selected_vertical_names
            skip_qualification = not selected_qualification_names
            skip_vertical_profiles = not selected_vertical_names

            if not should_continue:
                logger.info("Exiting as requested")
                return 1, False

            if skip_qualification:
                # No qualification selected, so no other profiles are available to
                # include either (they're offered alongside a selected qualification).
                logger.error("No profile selected - nothing to run")
                return 1, False
            elif skip_vertical_profiles:
                logger.info("Running selected profile only")
            else:
                logger.info("Running selected profile and its additional profile(s)")

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

                    # A qualification profile explicitly selected via the interactive
                    # prompt is included regardless of which prompt path built the list.
                    explicitly_selected_qualification = (
                        interactive_qualification_selection is not None
                        and is_qualification
                        and profile_name in interactive_qualification_selection
                    )

                    if include_all_types:
                        # skip_qualification only applies to profiles that opted-in
                        # to the ESQ-level UNSUPPORTED_GENERATIONS system compatibility check
                        if is_qualification and skip_qualification and labels.get("system_compatibility", False):
                            logger.debug(
                                f"Skipping qualification profile due to platform incompatibility: {profile_name}"
                            )
                            continue
                        requested_profile_names.append(profile_name)
                    elif interactive_qualification_selection is not None:
                        # Default interactive-prompt path: use the explicit qualification
                        # selection, and only the specific vertical profiles accepted
                        # alongside it (not a blanket all-verticals toggle).
                        is_vertical = profile_type == "verticals" or profile_label_type == "vertical"
                        if explicitly_selected_qualification or (
                            is_vertical and profile_name in interactive_vertical_selection
                        ):
                            requested_profile_names.append(profile_name)
                    else:
                        is_vertical = profile_type == "verticals" or profile_label_type == "vertical"
                        # Include qualification profiles that either pass the system compatibility
                        # check or do not opt-in to it
                        requires_sys_compat = labels.get("system_compatibility", False)
                        if (
                            is_qualification
                            and (not skip_qualification or not requires_sys_compat)
                            or is_vertical
                            and not skip_vertical_profiles
                        ):
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
    tests_actually_ran = False

    for profile_name in execution_order:
        # Only run if profile is valid
        if profile_name in valid_profiles_map:
            # Skip if already executed (in case of duplicate handling)
            if profile_name in executed_profiles:
                continue

            profile_type, profile = valid_profiles_map[profile_name]
            profile_result = _run_single_profile_in_batch(profile, data_dir, verbose, debug)
            executed_profiles.add(profile_name)
            if profile_result != TIER_SKIP_EXIT:
                tests_actually_ran = True
                result = profile_result
            elif result == 0:
                # Only record tier-skip as error if no other result yet
                result = 1

    logger.info(f"All profiles processed. Results: {result}")
    # Signal tier-skip-all with the sentinel so run_tests can distinguish it from
    # validation failures (which return 1, not TIER_SKIP_EXIT)
    if not tests_actually_ran and result != 0:
        return TIER_SKIP_EXIT, False
    return result, tests_actually_ran
