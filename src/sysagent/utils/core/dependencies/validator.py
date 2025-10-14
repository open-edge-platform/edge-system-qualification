# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Dependency validator with improved status reporting.

This module provides validation logic for dependencies with proper
status classification, especially for manual setup dependencies.
"""
import os
import platform
import shutil
import logging
from typing import Tuple, List, Optional

from sysagent.utils.core.process import run_command
from .schema import (
    Dependency, DependencyInstallation, DependencyType, 
    OSType, DependencyStatus, DependencyCheckResult
)

logger = logging.getLogger(__name__)


class DependencyValidator:
    """Validates dependency installation status with improved reporting."""
    
    def __init__(self):
        self.current_os = self._detect_os()
    
    def _detect_os(self) -> OSType:
        """Detect the current operating system."""
        system = platform.system().lower()
        if system == "linux":
            return OSType.LINUX
        elif system == "windows":
            return OSType.WINDOWS
        elif system == "darwin":
            return OSType.MACOS
        else:
            logger.warning(f"Unsupported OS: {system}, defaulting to Linux")
            return OSType.LINUX
    
    def validate_dependency(self, dependency: Dependency) -> DependencyCheckResult:
        """
        Validate a dependency and return detailed status.
        
        Args:
            dependency: Dependency to validate
            
        Returns:
            DependencyCheckResult with detailed status information
        """
        # Find installation for current OS
        installation = self._get_installation_for_os(dependency)
        if not installation:
            return DependencyCheckResult(
                name=dependency.name,
                status=DependencyStatus.ERROR,
                message=f"No installation instructions available for {self.current_os.value}"
            )
        
        # Handle different installation methods
        if installation.install_method == DependencyType.MANUAL_SETUP:
            return self._validate_manual_setup(dependency, installation)
        elif installation.install_method == DependencyType.SYSTEM_PACKAGE:
            return self._validate_system_packages(dependency, installation)
        elif installation.install_method == DependencyType.DOCKER_SETUP:
            return self._validate_docker_setup(dependency, installation)
        else:
            return self._validate_generic(dependency, installation)
    
    def _get_installation_for_os(self, dependency: Dependency) -> Optional[DependencyInstallation]:
        """Get installation instructions for the current OS."""
        for installation in dependency.installations:
            if installation.os_type == self.current_os:
                return installation
        return None
    
    def _validate_manual_setup(self, dependency: Dependency, installation: DependencyInstallation) -> DependencyCheckResult:
        """Validate manual setup dependencies with proper status."""
        # For manual setup, we can't automatically validate installation
        # Check if there's any validation defined, otherwise mark as needing manual setup
        
        if installation.validation:
            # Try to run validation if provided
            try:
                is_valid, message = self._run_validation(installation.validation)
                if is_valid:
                    return DependencyCheckResult(
                        name=dependency.name,
                        status=DependencyStatus.INSTALLED,
                        message=message
                    )
                else:
                    return DependencyCheckResult(
                        name=dependency.name,
                        status=DependencyStatus.MANUAL_REQUIRED,
                        message=f"Manual setup required - {message}",
                        details=f"See documentation: {installation.documentation_url}" if installation.documentation_url else None
                    )
            except Exception as e:
                logger.debug(f"Validation failed for {dependency.name}: {e}")
        
        # No validation available - this is a manual setup dependency
        return DependencyCheckResult(
            name=dependency.name,
            status=DependencyStatus.MANUAL_REQUIRED,
            message="Manual setup required (cannot auto-validate)",
            details=f"See documentation: {installation.documentation_url}" if installation.documentation_url else None
        )
    
    def _validate_system_packages(self, dependency: Dependency, installation: DependencyInstallation) -> DependencyCheckResult:
        """Validate system package dependencies."""
        if not installation.packages:
            return DependencyCheckResult(
                name=dependency.name,
                status=DependencyStatus.ERROR,
                message="No packages defined for system package dependency"
            )
        
        # Use dpkg validation for Debian/Ubuntu systems
        if installation.validation and hasattr(installation.validation, 'check_method') and installation.validation.check_method == "dpkg":
            return self._validate_dpkg_packages(dependency, installation)
        
        # Fallback to generic package validation
        return self._validate_packages_generic(dependency, installation)
    
    def _validate_dpkg_packages(self, dependency: Dependency, installation: DependencyInstallation) -> DependencyCheckResult:
        """Validate packages using dpkg (Debian/Ubuntu)."""
        packages_to_check = installation.packages
        missing_packages = []
        installed_packages = []
        
        for package in packages_to_check:
            if self._check_system_package(package):
                installed_packages.append(package)
            else:
                missing_packages.append(package)
        
        total_packages = len(packages_to_check)
        installed_count = len(installed_packages)
        missing_count = len(missing_packages)
        
        if missing_count == 0:
            return DependencyCheckResult(
                name=dependency.name,
                status=DependencyStatus.INSTALLED,
                message=f"All {installed_count}/{total_packages} system packages installed",
                total_packages=total_packages,
                installed_packages=installed_count
            )
        else:
            return DependencyCheckResult(
                name=dependency.name,
                status=DependencyStatus.MISSING,
                message=f"Missing {missing_count}/{total_packages} system packages",
                missing_packages=missing_packages,
                total_packages=total_packages,
                installed_packages=installed_count
            )

    def _check_system_package(self, package_name: str) -> bool:
        """
        Check if a single system package is installed.
        
        Args:
            package_name: Name of the package to check
            
        Returns:
            bool: True if package is installed, False otherwise
        """
        try:
            result = run_command(
                f"dpkg-query -W -f='${{Status}}' {package_name}",
                timeout=10
            )
            # Check if package is installed and configured
            return result.success and "install ok installed" in result.stdout
        except Exception:
            return False
    
    def _validate_packages_generic(self, dependency: Dependency, installation: DependencyInstallation) -> DependencyCheckResult:
        """Generic package validation using command availability."""
        if installation.validation and installation.validation.check_commands:
            missing_packages = []
            installed_packages = []
            
            for check_cmd in installation.validation.check_commands:
                try:
                    result = run_command(check_cmd, timeout=10)
                    if result.success:
                        pkg_name = check_cmd.split()[0]
                        installed_packages.append(pkg_name)
                    else:
                        pkg_name = check_cmd.split()[0]
                        missing_packages.append(pkg_name)
                except Exception:
                    pkg_name = check_cmd.split()[0]
                    missing_packages.append(pkg_name)
            
            total_packages = len(installation.validation.check_commands)
            installed_count = len(installed_packages)
            missing_count = len(missing_packages)
            
            if missing_count == 0:
                return DependencyCheckResult(
                    name=dependency.name,
                    status=DependencyStatus.INSTALLED,
                    message=f"All {installed_count}/{total_packages} packages available",
                    total_packages=total_packages,
                    installed_packages=installed_count
                )
            else:
                return DependencyCheckResult(
                    name=dependency.name,
                    status=DependencyStatus.MISSING,
                    message=f"Missing {missing_count}/{total_packages} packages",
                    missing_packages=missing_packages,
                    total_packages=total_packages,
                    installed_packages=installed_count
                )
        
        # No validation available - try basic command check
        if shutil.which(dependency.name):
            return DependencyCheckResult(
                name=dependency.name,
                status=DependencyStatus.INSTALLED,
                message=f"{dependency.name} is available in PATH"
            )
        else:
            return DependencyCheckResult(
                name=dependency.name,
                status=DependencyStatus.MISSING,
                message=f"{dependency.name} not found in PATH"
            )
    
    def _validate_docker_setup(self, dependency: Dependency, installation: DependencyInstallation) -> DependencyCheckResult:
        """Validate Docker setup dependency."""
        # Check if docker command is available
        if not shutil.which("docker"):
            return DependencyCheckResult(
                name=dependency.name,
                status=DependencyStatus.MISSING,
                message="Docker command not found in PATH"
            )
        
        # Try to run docker version to check if daemon is running
        try:
            result = run_command("docker version", timeout=10)
            if result.success:
                return DependencyCheckResult(
                    name=dependency.name,
                    status=DependencyStatus.INSTALLED,
                    message="Docker is installed and running"
                )
            else:
                return DependencyCheckResult(
                    name=dependency.name,
                    status=DependencyStatus.NEEDS_CONFIGURATION,
                    message="Docker installed but daemon not accessible",
                    details="Check if Docker daemon is running and user has permissions"
                )
        except Exception as e:
            return DependencyCheckResult(
                name=dependency.name,
                status=DependencyStatus.ERROR,
                message=f"Error checking Docker status: {e}"
            )
    
    def _validate_generic(self, dependency: Dependency, installation: DependencyInstallation) -> DependencyCheckResult:
        """Generic validation for other dependency types."""
        if installation.validation:
            try:
                is_valid, message = self._run_validation(installation.validation)
                status = DependencyStatus.INSTALLED if is_valid else DependencyStatus.MISSING
                return DependencyCheckResult(
                    name=dependency.name,
                    status=status,
                    message=message
                )
            except Exception as e:
                return DependencyCheckResult(
                    name=dependency.name,
                    status=DependencyStatus.ERROR,
                    message=f"Validation error: {e}"
                )
        
        # No validation - try basic command check
        if shutil.which(dependency.name):
            return DependencyCheckResult(
                name=dependency.name,
                status=DependencyStatus.INSTALLED,
                message=f"{dependency.name} is available"
            )
        else:
            return DependencyCheckResult(
                name=dependency.name,
                status=DependencyStatus.MISSING,
                message=f"{dependency.name} not found"
            )
    
    def _run_validation(self, validation) -> Tuple[bool, str]:
        """Run validation check and return result."""
        # Check file exists
        if validation.check_file_exists:
            if os.path.exists(validation.check_file_exists):
                return True, f"Required file exists: {validation.check_file_exists}"
            else:
                return False, f"Required file missing: {validation.check_file_exists}"
        
        # Check service running
        if validation.check_service_running:
            try:
                result = run_command(f"systemctl is-active {validation.check_service_running}", timeout=10)
                if result.success and "active" in result.stdout:
                    return True, f"Service {validation.check_service_running} is running"
                else:
                    return False, f"Service {validation.check_service_running} is not running"
            except Exception:
                return False, f"Cannot check service {validation.check_service_running}"
        
        # Run check command
        if validation.check_command:
            try:
                result = run_command(validation.check_command, timeout=10)
                
                if result.returncode != validation.expected_return_code:
                    return False, f"Command failed: {validation.check_command}"
                
                if validation.expected_output_contains:
                    if validation.expected_output_contains not in result.stdout:
                        return False, f"Expected output not found in command: {validation.check_command}"
                
                return True, "Validation command passed"
                
            except Exception as e:
                return False, f"Validation command error: {e}"
        
        return False, "No validation method defined"
