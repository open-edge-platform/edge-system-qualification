# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Profile dependency resolution utilities.

This module provides functionality to resolve profile dependencies and
determine the correct execution order for profiles with prerequisites.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ProfileDependencyError(Exception):
    """Exception raised when profile dependency resolution fails."""

    pass


class CircularDependencyError(ProfileDependencyError):
    """Exception raised when circular dependencies are detected."""

    pass


class MissingDependencyError(ProfileDependencyError):
    """Exception raised when a required dependency profile is not found."""

    pass


def get_profile_dependencies(profile_configs: Dict[str, Any]) -> List[str]:
    """
    Extract dependency list from profile configuration.

    Args:
        profile_configs: Profile configuration dictionary

    Returns:
        List of profile names that this profile depends on
    """
    params = profile_configs.get("params", {})
    depends_on = params.get("depends_on", [])

    # Ensure it's a list
    if isinstance(depends_on, str):
        depends_on = [depends_on]
    elif not isinstance(depends_on, list):
        depends_on = []

    return depends_on


def resolve_profile_dependencies(
    profiles: Dict[str, Dict[str, Any]], requested_profiles: List[str] = None
) -> List[str]:
    """
    Resolve profile dependencies and return execution order.

    Uses topological sorting to determine the correct order to execute profiles
    based on their dependencies. Profiles with no dependencies are executed first,
    followed by profiles that depend on them.

    Args:
        profiles: Dictionary mapping profile names to their configurations
        requested_profiles: Optional list of specific profiles to resolve.
                          If None, all profiles are resolved.

    Returns:
        List of profile names in execution order (dependencies first)

    Raises:
        CircularDependencyError: If circular dependencies are detected
        MissingDependencyError: If a dependency profile is not found
    """
    # If specific profiles requested, filter to those
    if requested_profiles:
        profile_subset = {name: profiles[name] for name in requested_profiles if name in profiles}
        profiles = profile_subset

    # Build dependency graph
    dependency_graph = {}
    for profile_name, profile_config in profiles.items():
        dependencies = get_profile_dependencies(profile_config)
        dependency_graph[profile_name] = dependencies

    # Validate all dependencies exist
    all_profile_names = set(profiles.keys())
    for profile_name, dependencies in dependency_graph.items():
        for dep in dependencies:
            if dep not in all_profile_names:
                logger.warning(
                    f"Profile '{profile_name}' depends on '{dep}' which is not available. Dependency will be skipped."
                )

    # Perform topological sort using Kahn's algorithm
    sorted_profiles = []
    in_degree = {name: 0 for name in dependency_graph}

    # Calculate in-degrees
    for profile_name, dependencies in dependency_graph.items():
        for dep in dependencies:
            if dep in in_degree:
                in_degree[profile_name] += 1

    # Queue profiles with no dependencies
    queue = [name for name, degree in in_degree.items() if degree == 0]

    while queue:
        # Sort queue to ensure deterministic order
        queue.sort()
        current = queue.pop(0)
        sorted_profiles.append(current)

        # Reduce in-degree for dependents
        for profile_name, dependencies in dependency_graph.items():
            if current in dependencies and profile_name in in_degree:
                in_degree[profile_name] -= 1
                if in_degree[profile_name] == 0:
                    queue.append(profile_name)

    # Check for circular dependencies
    if len(sorted_profiles) != len(dependency_graph):
        remaining = set(dependency_graph.keys()) - set(sorted_profiles)
        raise CircularDependencyError(f"Circular dependency detected among profiles: {', '.join(remaining)}")

    return sorted_profiles


def expand_profile_with_dependencies(profile_name: str, all_profiles: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Expand a single profile to include all its dependencies in execution order.

    Args:
        profile_name: Name of the profile to expand
        all_profiles: Dictionary of all available profiles

    Returns:
        List of profile names to execute (dependencies first, then the profile)

    Raises:
        MissingDependencyError: If the profile or any dependency is not found
    """
    if profile_name not in all_profiles:
        raise MissingDependencyError(f"Profile '{profile_name}' not found")

    # Get all dependencies recursively
    to_process = [profile_name]
    all_required = set()
    processed = set()

    while to_process:
        current = to_process.pop(0)

        if current in processed:
            continue

        processed.add(current)

        if current not in all_profiles:
            logger.warning(f"Dependency profile '{current}' not found, skipping")
            continue

        all_required.add(current)

        # Add dependencies to process
        dependencies = get_profile_dependencies(all_profiles[current])
        for dep in dependencies:
            if dep not in processed:
                to_process.append(dep)

    # Build subset with only required profiles
    required_profiles = {name: all_profiles[name] for name in all_required if name in all_profiles}

    # Resolve execution order
    execution_order = resolve_profile_dependencies(required_profiles)

    return execution_order


def validate_profile_dependencies(profiles: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Validate all profile dependencies and return list of issues.

    Args:
        profiles: Dictionary mapping profile names to their configurations

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    for profile_name, profile_config in profiles.items():
        dependencies = get_profile_dependencies(profile_config)

        for dep in dependencies:
            if dep not in profiles:
                errors.append(f"Profile '{profile_name}' depends on '{dep}' which does not exist")

    # Check for circular dependencies
    try:
        resolve_profile_dependencies(profiles)
    except CircularDependencyError as e:
        errors.append(str(e))

    return errors


def get_dependency_tree(profile_name: str, all_profiles: Dict[str, Dict[str, Any]], indent: int = 0) -> str:
    """
    Generate a text representation of a profile's dependency tree.

    Args:
        profile_name: Name of the profile
        all_profiles: Dictionary of all available profiles
        indent: Current indentation level

    Returns:
        String representation of the dependency tree
    """
    if profile_name not in all_profiles:
        return "  " * indent + f"❌ {profile_name} (NOT FOUND)\n"

    tree = "  " * indent + f"📦 {profile_name}\n"

    dependencies = get_profile_dependencies(all_profiles[profile_name])
    for dep in dependencies:
        tree += get_dependency_tree(dep, all_profiles, indent + 1)

    return tree
