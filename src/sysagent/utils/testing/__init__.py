# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Testing utilities package.

Provides utilities for test execution, validation, and configuration
including pytest configuration, profile validation, and tier validation.
"""

# Import local modules directly without backward compatibility
from .parameterization import (
    generate_test_id,
    parameterize_test,
)
from .profile_validator import (
    ProfileValidationError,
    validate_profile_requirements,
)
from .pytest_config import (
    add_test_paths_to_args,
    cleanup_pytest_cache,
    configure_pytest_environment,
    create_profile_pytest_args,
    create_pytest_args,
    run_pytest,
    validate_pytest_args,
)
from .system_validator import (
    SystemValidator,
    check_system_ready_for_tests,
    validate_system_requirements,
)
from .tier_validator import (
    TierValidationError,
    get_highest_matching_tier,
    get_test_params_tiers,
    log_tier_validation_results,
    validate_profile_tier_configuration,
    validate_profile_tiers,
    validate_test_tiers,
)

__all__ = [
    # From pytest_config
    "configure_pytest_environment",
    "create_profile_pytest_args",
    "create_pytest_args",
    "run_pytest",
    "cleanup_pytest_cache",
    "validate_pytest_args",
    "add_test_paths_to_args",
    # From profile_validator
    "validate_profile_requirements",
    "ProfileValidationError",
    # From tier_validator
    "validate_profile_tiers",
    "validate_test_tiers",
    "get_highest_matching_tier",
    "get_test_params_tiers",
    "validate_profile_tier_configuration",
    "log_tier_validation_results",
    "TierValidationError",
    # From parameterization
    "generate_test_id",
    "parameterize_test",
    # From system_validator
    "SystemValidator",
    "validate_system_requirements",
    "check_system_ready_for_tests",
]
