# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Improved dependency management system with modular design.

This module provides the main DependencyManager class that coordinates
dependency checking, validation, and installation guidance with proper
status reporting and package-level configuration support.
"""
import logging
from typing import Dict, List, Optional, Tuple

from .schema import DependencyConfig, DependencyStatus, DependencyCheckResult, Dependency, DependencyType
from .loader import DependencyConfigLoader
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
    
    def __init__(self, config_loader: Optional[DependencyConfigLoader] = None):
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
                name=name,
                status=DependencyStatus.ERROR,
                message=f"Unknown dependency: {name}"
            )
        
        dependency = self.config.dependencies[name]
        return self.validator.validate_dependency(dependency)
    
    def check_all_dependencies(self) -> Dict[str, DependencyCheckResult]:
        """
        Check all dependencies and return their detailed status.
        
        Returns:
            Dictionary mapping dependency name to DependencyCheckResult
        """
        results = {}
        
        # Check in priority order if install_order is defined
        check_order = self.config.install_order if self.config.install_order else sorted(self.config.dependencies.keys())
        
        for name in check_order:
            if name in self.config.dependencies:
                results[name] = self.check_dependency(name)
        
        return results
    
    def get_missing_dependencies(self) -> List[str]:
        """Get list of missing required dependencies."""
        results = self.check_all_dependencies()
        missing = []
        
        for name, result in results.items():
            dependency = self.config.dependencies.get(name)
            if dependency and dependency.required and result.status in [DependencyStatus.MISSING, DependencyStatus.ERROR]:
                missing.append(name)
        
        return missing
    
    def get_manual_setup_dependencies(self) -> List[str]:
        """Get list of dependencies requiring manual setup."""
        results = self.check_all_dependencies()
        manual = []
        
        for name, result in results.items():
            if result.status == DependencyStatus.MANUAL_REQUIRED:
                manual.append(name)
        
        return manual
    
    def generate_installation_instructions(self, dependency_names: Optional[List[str]] = None) -> str:
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
                    
                    # Add package-specific installation commands for system packages
                    if installation.packages and dep_type.value == "system_package":
                        # Get only missing packages for this dependency
                        missing_packages = self._get_missing_packages(dependency)
                        if missing_packages:
                            packages_str = " ".join(missing_packages)
                            instructions.append(f"  Install command: sudo apt-get update && sudo apt-get install -y {packages_str}")
        
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
                dependency_objects = [self.config.dependencies[name] for name in actionable_deps 
                                    if name in self.config.dependencies]
                print(f"\n{self._generate_simplified_instructions(dependency_objects)}")
            
            return False
        
        return True

    def _get_missing_packages(self, dependency: Dependency) -> List[str]:
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

    def _generate_simplified_instructions(self, dependencies: List[Dependency]) -> str:
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
                instructions.append(f"Run this command to install missing system packages:\n")
                instructions.append(f"sudo apt update && sudo apt install -y {packages_str}\n")
        
        # Handle other dependency types (manual setup, etc.) - but skip them for CLI simplicity
        # They can be shown in `esq deps` command for full details
        
        return "\n".join(instructions)
    
    def _group_dependencies_by_method(self, dependency_names: List[str]) -> Dict:
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
    
    def add_package_config(self, package_name: str, config_path: Optional[str] = None):
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
