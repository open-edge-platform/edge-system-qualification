# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Improved dependency management system with modular design.

This module provides the main DependencyManager class that coordinates
dependency checking, validation, and installation guidance with proper
status reporting and package-level configuration support.
"""

import logging

from .loader import DependencyConfigLoader
from .schema import Dependency, DependencyCheckResult, DependencyStatus, DependencyType
from .validator import DependencyValidator

logger = logging.getLogger(__name__)


class DependencyManager:
    """
    Improved dependency management system with modular design.

    Features:
    - Package-level configuration merging
    - Proper status reporting for manual setup dependencies
    - Modular validation and installation management
    - Under 500 lines of code per file
    """

    def __init__(self, config_loader: DependencyConfigLoader | None = None):
        """
        Initialize the dependency manager.

        Args:
            config_loader: Optional custom config loader
        """
        self.config_loader = config_loader or DependencyConfigLoader()
        self.validator = DependencyValidator()
        self.config = self.config_loader.load_default_config()

    def check_dependency(self, name: str) -> DependencyCheckResult:
        """
        Check if a specific dependency is installed and working.

        Args:
            name: Name of the dependency to check

        Returns:
            DependencyCheckResult with detailed status information
        """
        if name not in self.config.dependencies:
            return DependencyCheckResult(
                name=name, status=DependencyStatus.ERROR, message=f"Unknown dependency: {name}"
            )

        dependency = self.config.dependencies[name]
        return self.validator.validate_dependency(dependency)

    def check_all_dependencies(self, include_functional: bool = False) -> dict[str, DependencyCheckResult]:
        """
        Check all dependencies and return their detailed status.

        Args:
            include_functional: Whether to also run functional_check (runtime
                sanity/smoke test) dependencies. These are excluded by default
                because they run real (if lightweight) workloads - e.g. an
                OpenVINO device compile/inference smoke test - and should not
                slow down every CLI command. Use ``check_functional_dependencies``
                or the prerun check module to run them explicitly (e.g. before
                the CLI's run command executes profiles/tests).

        Returns:
            Dictionary mapping dependency name to DependencyCheckResult
        """
        results = {}

        # Check in priority order if install_order is defined
        check_order = (
            self.config.install_order if self.config.install_order else sorted(self.config.dependencies.keys())
        )

        for name in check_order:
            if name in self.config.dependencies:
                dependency = self.config.dependencies[name]
                if not include_functional and self._is_functional_only(dependency):
                    continue
                results[name] = self.check_dependency(name)

        return results

    def _is_functional_only(self, dependency: Dependency) -> bool:
        """Whether this dependency's current-OS installation is a functional_check (runtime smoke test)."""
        installation = self.validator._get_installation_for_os(dependency)
        return bool(installation and installation.install_method == DependencyType.FUNCTIONAL_CHECK)

    def get_functional_dependency_names(self, active_flags: set[str] | None = None) -> list[str]:
        """Get names of dependencies configured as functional_check (runtime sanity/smoke tests).

        Returns names in install_order (if defined), with any remaining
        functional dependencies appended alphabetically.

        Args:
            active_flags: Profile requirement flags (e.g. {"docker_required"}) known
                to be relevant to the profile(s)/test(s) about to run. A dependency
                whose ``requires_flag`` isn't in this set is excluded. Pass None (the
                default) when the relevant profile(s) aren't known ahead of time -
                every functional check is then included, same as before flag-gating
                existed. Dependencies with no ``requires_flag`` are always included.
        """
        ordered_names = (
            list(self.config.install_order) if self.config.install_order else sorted(self.config.dependencies.keys())
        )
        remaining = sorted(n for n in self.config.dependencies if n not in ordered_names)
        ordered_names.extend(remaining)

        names = [
            name
            for name in ordered_names
            if name in self.config.dependencies and self._is_functional_only(self.config.dependencies[name])
        ]
        if active_flags is None:
            return names
        return [
            name
            for name in names
            if not self.config.dependencies[name].requires_flag
            or self.config.dependencies[name].requires_flag in active_flags
        ]

    def check_functional_dependencies(self, active_flags: set[str] | None = None) -> dict[str, DependencyCheckResult]:
        """Check only functional_check (runtime sanity/smoke test) dependencies.

        Args:
            active_flags: See ``get_functional_dependency_names``.

        Returns:
            Dictionary mapping dependency name to DependencyCheckResult
        """
        return {name: self.check_dependency(name) for name in self.get_functional_dependency_names(active_flags)}

    def get_missing_dependencies(self) -> list[str]:
        """Get list of missing required dependencies."""
        results = self.check_all_dependencies()
        missing = []

        for name, result in results.items():
            dependency = self.config.dependencies.get(name)
            if (
                dependency
                and dependency.required
                and result.status in [DependencyStatus.MISSING, DependencyStatus.ERROR]
            ):
                missing.append(name)

        return missing

    def get_manual_setup_dependencies(self) -> list[str]:
        """Get list of dependencies requiring manual setup."""
        results = self.check_all_dependencies()
        manual = []

        for name, result in results.items():
            if result.status == DependencyStatus.MANUAL_REQUIRED:
                manual.append(name)

        return manual

    def generate_installation_instructions(self, dependency_names: list[str] | None = None) -> str:
        """
        Generate installation instructions for dependencies.

        Args:
            dependency_names: Specific dependencies to generate instructions for

        Returns:
            Formatted installation instructions
        """
        if dependency_names is None:
            dependency_names = list(self.config.dependencies.keys())

        instructions = []
        instructions.append("INSTALLATION INSTRUCTIONS")
        instructions.append("=" * 50)

        # Group dependencies by installation method for efficiency
        grouped_deps = self._group_dependencies_by_method(dependency_names)

        for dep_type, deps in grouped_deps.items():
            if not deps:
                continue

            instructions.append(f"\n{dep_type.value.replace('_', ' ').title()} Dependencies:")
            instructions.append("-" * 40)

            for dep_name in deps:
                dependency = self.config.dependencies[dep_name]
                installation = self.validator._get_installation_for_os(dependency)

                if installation:
                    instructions.append(f"\n{dep_name}:")
                    instructions.append(f"  Description: {dependency.description}")

                    if installation.documentation_url:
                        instructions.append(f"  Documentation: {installation.documentation_url}")

                    if installation.notes:
                        instructions.append("  Notes:")
                        for note in installation.notes:
                            instructions.append(f"    - {note}")

                    # Add package-specific information for system packages
                    if installation.packages and dep_type.value == "system_package":
                        # Get only missing packages for this dependency
                        missing_packages = self._get_missing_packages(dependency)
                        if missing_packages:
                            packages_str = " ".join(missing_packages)
                            instructions.append(
                                f"  Missing system packages: {packages_str}. "
                                "Refer to project documentation to install them."
                            )

        return "\n".join(instructions)

    def check_and_report_dependencies(self, show_manual_deps: bool = False) -> bool:
        """
        Check system dependencies and provide concise feedback for missing ones.

        Args:
            show_manual_deps: Whether to show manual setup dependencies in output

        Returns:
            bool: True if all required dependencies are satisfied, False otherwise
        """
        results = self.check_all_dependencies()

        missing_deps = []
        manual_deps = []
        failed_deps = []

        for name, result in results.items():
            dependency = self.config.dependencies.get(name)
            if dependency and dependency.required:
                if result.status == DependencyStatus.MISSING:
                    missing_deps.append(name)
                elif result.status == DependencyStatus.ERROR:
                    failed_deps.append(name)
                elif result.status == DependencyStatus.MANUAL_REQUIRED:
                    manual_deps.append(name)

        # If there are missing or failed dependencies, show concise information
        if missing_deps or failed_deps:
            print("\nDEPENDENCY CHECK FAILED")
            print("=" * 50)

            for name, result in results.items():
                dependency = self.config.dependencies.get(name)
                if dependency and dependency.required:
                    # Only show auto-validatable dependencies in the main output
                    if result.status == DependencyStatus.MANUAL_REQUIRED and not show_manual_deps:
                        continue

                    status_symbol = "✗" if result.status in [DependencyStatus.MISSING, DependencyStatus.ERROR] else "✓"
                    print(f"[{status_symbol}] {name}: {result.message}")

            # Show simplified installation instructions only for actionable dependencies
            actionable_deps = missing_deps + failed_deps
            if actionable_deps:
                # Convert dependency names to Dependency objects
                dependency_objects = [
                    self.config.dependencies[name] for name in actionable_deps if name in self.config.dependencies
                ]
                print(f"\n{self._generate_simplified_instructions(dependency_objects)}")

            return False

        return True

    def _get_missing_packages(self, dependency: Dependency) -> list[str]:
        """
        Get list of missing packages for a dependency.

        Args:
            dependency: The dependency to check

        Returns:
            List of missing package names
        """
        missing_packages = []

        for installation in dependency.installations:
            if installation.install_method == DependencyType.SYSTEM_PACKAGE and installation.packages:
                # Use the validator to check individual packages
                for package in installation.packages:
                    if not self.validator._check_system_package(package):
                        missing_packages.append(package)

        return missing_packages

    def _generate_simplified_instructions(self, dependencies: list[Dependency]) -> str:
        """
        Generate simplified, actionable installation instructions.

        Args:
            dependencies: Dependencies that need installation

        Returns:
            Simplified installation instructions
        """
        instructions = []
        instructions.append("INSTALL MISSING DEPENDENCIES")
        instructions.append("=" * 50)

        # Handle system packages specially with a single command
        system_package_deps = []
        other_deps = []

        for dependency in dependencies:
            installation = self.validator._get_installation_for_os(dependency)
            if installation and installation.install_method == DependencyType.SYSTEM_PACKAGE:
                system_package_deps.append(dependency)
            else:
                other_deps.append(dependency)

        # Generate consolidated system package installation command
        if system_package_deps:
            all_missing_packages = []
            for dependency in system_package_deps:
                missing_packages = self._get_missing_packages(dependency)
                all_missing_packages.extend(missing_packages)

            # Remove duplicates while preserving order
            unique_packages = []
            seen = set()
            for pkg in all_missing_packages:
                if pkg not in seen:
                    unique_packages.append(pkg)
                    seen.add(pkg)

            if unique_packages:
                packages_str = " ".join(unique_packages)
                instructions.append(f"Missing system packages: {packages_str}")
                instructions.append("Refer to project documentation to install the required system packages.\n")

        # Handle other dependency types (manual setup, etc.) - but skip them for CLI simplicity
        # They can be shown via the CLI's `deps` command for full details

        return "\n".join(instructions)

    def _group_dependencies_by_method(self, dependency_names: list[str]) -> dict:
        """Group dependencies by their installation method."""
        from .schema import DependencyType

        grouped = {dep_type: [] for dep_type in DependencyType}

        for name in dependency_names:
            if name in self.config.dependencies:
                dependency = self.config.dependencies[name]
                installation = self.validator._get_installation_for_os(dependency)
                if installation:
                    grouped[installation.install_method].append(name)

        return grouped

    def verify_system_dependencies(self) -> bool:
        """
        Verify that all required system dependencies are satisfied.

        Returns:
            True if all required dependencies are satisfied, False otherwise
        """
        results = self.check_all_dependencies()

        for name, result in results.items():
            dependency = self.config.dependencies.get(name)
            if dependency and dependency.required:
                logger.info(f"Dependency '{name}': {result.status.value} - {result.message}")
                # Consider manual_required as satisfied if it's a manual setup dependency
                if result.status in [DependencyStatus.MISSING, DependencyStatus.ERROR]:
                    return False

        return True

    def reload_config(self):
        """Reload dependency configuration from files."""
        self.config = self.config_loader.load_default_config()

    def add_package_config(self, package_name: str, config_path: str | None = None):
        """
        Add or reload configuration for a specific package.

        Args:
            package_name: Name of the package
            config_path: Optional path to config file
        """
        self.config_loader.load_package_config(package_name, config_path)
        self.reload_config()


# Global instance for backward compatibility
_dependency_manager = None


def get_dependency_manager() -> DependencyManager:
    """Get the global dependency manager instance."""
    global _dependency_manager
    if _dependency_manager is None:
        _dependency_manager = DependencyManager()
    return _dependency_manager
