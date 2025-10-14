# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Dependency configuration loader with package-level merging support.

This module handles loading and merging dependency configurations from
multiple packages dynamic        return installationme discovery mechanism as 
software.yml configurations for system information tracking.
"""
import os
import yaml
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from .schema import (
    DependencyConfig, Dependency, DependencyInstallation, 
    DependencyGroup, DependencyType, OSType, DependencyValidation,
    InstallCommand, PostInstallStep
)

logger = logging.getLogger(__name__)


class DependencyConfigLoader:
    """Loads and merges dependency configurations from multiple packages dynamically."""
    
    def __init__(self):
        self.loaded_configs: Dict[str, DependencyConfig] = {}
    
    def _discover_dependency_configs(self) -> List[Path]:
        """
        Discover dependency configuration files using the same approach as software.yml.
        
        Returns:
            List of paths to dependencies.yml files found across packages
        """
        dependency_config_paths = []
        
        try:
            # Import the discovery function from config loader
            from sysagent.utils.config.config_loader import discover_entrypoint_paths
            
            # Search for configs in all discovered packages
            config_paths = discover_entrypoint_paths("configs")
            
            for config_path in config_paths:
                # Look for dependencies.yml in core subfolder
                deps_file = config_path / "core" / "dependencies.yml"
                if deps_file.exists():
                    dependency_config_paths.append(deps_file)
                    logger.debug(f"Found dependency config: {deps_file}")
                
                # Also check direct dependencies.yml in configs folder
                deps_file_direct = config_path / "dependencies.yml"
                if deps_file_direct.exists():
                    dependency_config_paths.append(deps_file_direct)
                    logger.debug(f"Found dependency config: {deps_file_direct}")
        
        except Exception as e:
            logger.warning(f"Failed to discover dependency configs via entrypoints: {e}")
            # Fallback to basic discovery
            dependency_config_paths = self._fallback_discovery()
        
        return dependency_config_paths
    
    def _fallback_discovery(self) -> List[Path]:
        """Fallback discovery method when entrypoint discovery fails."""
        paths = []
        
        # Look for src/ directory structure
        src_dir = Path("src")
        if src_dir.exists():
            for package_dir in src_dir.iterdir():
                if package_dir.is_dir() and not package_dir.name.startswith('.'):
                    # Check for core/dependencies.yml
                    deps_file = package_dir / "configs" / "core" / "dependencies.yml"
                    if deps_file.exists():
                        paths.append(deps_file)
                    
                    # Check for direct dependencies.yml
                    deps_file_direct = package_dir / "configs" / "dependencies.yml"
                    if deps_file_direct.exists():
                        paths.append(deps_file_direct)
        
        return paths
    
    def load_all_package_configs(self) -> Dict[str, DependencyConfig]:
        """
        Load dependency configurations from all discovered packages.
        
        Returns:
            Dictionary mapping package names to their dependency configurations
        """
        config_files = self._discover_dependency_configs()
        loaded_configs = {}
        
        for config_file in config_files:
            # Extract package name from path
            package_name = self._extract_package_name(config_file)
            
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    raw_config = yaml.safe_load(f)
                
                config = self._parse_config(raw_config, package_name)
                loaded_configs[package_name] = config
                self.loaded_configs[package_name] = config
                logger.info(f"Loaded dependency config for package {package_name} from {config_file}")
                
            except Exception as e:
                logger.error(f"Failed to load dependency config from {config_file}: {e}")
        
        return loaded_configs
    
    def _extract_package_name(self, config_path: Path) -> str:
        """Extract package name from configuration file path."""
        # Try to find package name from path structure
        # Expected: .../src/package_name/configs/... or .../package_name/configs/...
        parts = config_path.parts
        
        for i, part in enumerate(parts):
            if part == "configs" and i > 0:
                return parts[i-1]
        
        # Fallback: use parent directory name
        return config_path.parent.parent.name if config_path.parent.name == "core" else config_path.parent.name
    
    def load_package_config(self, package_name: str, config_path: Optional[str] = None) -> Optional[DependencyConfig]:
        """
        Load dependency configuration for a specific package.
        
        Args:
            package_name: Name of the package
            config_path: Optional path to config file, otherwise auto-discovered
            
        Returns:
            DependencyConfig if loaded successfully, None otherwise
        """
        if config_path is None:
            # Use dynamic discovery instead of hardcoded paths
            config_files = self._discover_dependency_configs()
            
            # Find config file for this specific package
            target_config = None
            for config_file in config_files:
                if self._extract_package_name(config_file) == package_name:
                    target_config = str(config_file)
                    break
            
            if not target_config:
                logger.debug(f"No dependency config found for package {package_name}")
                return None
            
            config_path = target_config
            
        if not config_path or not os.path.exists(config_path):
            logger.debug(f"No dependency config found for package {package_name}")
            return None
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            config = self._parse_config(raw_config, package_name)
            self.loaded_configs[package_name] = config
            logger.info(f"Loaded dependency config for package {package_name} from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load dependency config for {package_name} from {config_path}: {e}")
            return None
    
    def _discover_config_path(self, package_name: str) -> Optional[str]:
        """Discover the dependency config path for a package."""
        # Try multiple possible locations
        possible_paths = [
            f"src/{package_name}/configs/core/dependencies.yml",
            f"src/{package_name}/configs/dependencies.yml", 
            f"{package_name}/configs/core/dependencies.yml",
            f"{package_name}/configs/dependencies.yml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _parse_config(self, raw_config: Dict[str, Any], package_name: str) -> DependencyConfig:
        """Parse raw YAML config into structured objects."""
        config = DependencyConfig()
        config.version = raw_config.get("version", "1.0")
        config.install_order = raw_config.get("install_order", [])
        config.package_source = package_name
        
        # Parse dependencies
        for name, dep_data in raw_config.get("dependencies", {}).items():
            dependency = self._parse_dependency(name, dep_data, package_name)
            config.dependencies[name] = dependency
        
        # Parse groups
        for name, group_data in raw_config.get("groups", {}).items():
            group = DependencyGroup(
                name=group_data.get("name", name),
                description=group_data.get("description", ""),
                dependencies=group_data.get("dependencies", []),
                optional=group_data.get("optional", False)
            )
            config.groups[name] = group
        
        return config
    
    def _parse_dependency(self, name: str, data: Dict[str, Any], package_name: str) -> Dependency:
        """Parse a single dependency from configuration data."""
        dependency = Dependency(
            name=name,
            description=data.get("description", ""),
            required=data.get("required", True),
            priority=data.get("priority", 100),
            dependencies=data.get("dependencies", []),
            conflicts_with=data.get("conflicts_with", []),
            min_version=data.get("min_version"),
            max_version=data.get("max_version"),
            package_source=package_name
        )
        
        # Parse installations
        for install_data in data.get("installations", []):
            installation = self._parse_installation(install_data)
            dependency.installations.append(installation)
        
        return dependency
    
    def _parse_installation(self, data: Dict[str, Any]) -> DependencyInstallation:
        """Parse installation instructions from configuration data."""
        installation = DependencyInstallation(
            os_type=OSType(data["os_type"]),
            install_method=DependencyType(data["install_method"]),
            notes=data.get("notes", []),
            documentation_url=data.get("documentation_url"),
            packages=data.get("packages", [])
        )
        
        # Parse validation
        validation_data = data.get("validation")
        if validation_data:
            validation = DependencyValidation(
                check_command=validation_data.get("check_command", ""),
                expected_return_code=validation_data.get("expected_return_code", 0),
                expected_output_contains=validation_data.get("expected_output_contains"),
                check_file_exists=validation_data.get("check_file_exists"),
                check_service_running=validation_data.get("check_service_running"),
                check_method=validation_data.get("check_method"),
                packages_check=validation_data.get("packages_check", [])
            )
            
            if "check_commands" in validation_data:
                validation.check_commands = validation_data["check_commands"]
            
            installation.validation = validation
        
        return installation
    
    def merge_configs(self, package_names: Optional[List[str]] = None) -> DependencyConfig:
        """
        Merge dependency configurations from multiple packages.
        
        Args:
            package_names: Optional list of specific package names to merge. 
                          If None, discovers and merges all available packages.
            
        Returns:
            Merged DependencyConfig
        """
        merged_config = DependencyConfig()
        
        if package_names is None:
            # Load all discovered package configs
            self.load_all_package_configs()
            package_names = list(self.loaded_configs.keys())
        else:
            # Load configs for specified packages
            for package_name in package_names:
                if package_name not in self.loaded_configs:
                    self.load_package_config(package_name)
        
        # Sort packages for consistent merging order (sysagent first if present)
        sorted_package_names = self._sort_packages_for_merging(package_names)
        
        # Merge configurations with smart dependency merging
        for package_name in sorted_package_names:
            if package_name in self.loaded_configs:
                package_config = self.loaded_configs[package_name]
                
                # Smart dependency merging (merge packages within same dependency)
                for dep_name, dependency in package_config.dependencies.items():
                    if dep_name in merged_config.dependencies:
                        # Merge with existing dependency
                        merged_config.dependencies[dep_name] = self._merge_dependencies(
                            merged_config.dependencies[dep_name], 
                            dependency
                        )
                    else:
                        # Add new dependency
                        merged_config.dependencies[dep_name] = dependency
                
                # Merge groups
                for group_name, group in package_config.groups.items():
                    merged_config.groups[group_name] = group
                
                # Merge install order (append unique items)
                for item in package_config.install_order:
                    if item not in merged_config.install_order:
                        merged_config.install_order.append(item)
        
        logger.info(f"Merged {len(merged_config.dependencies)} dependencies from {len(sorted_package_names)} packages")
        return merged_config
    
    def _merge_dependencies(self, base_dependency: Dependency, extension_dependency: Dependency) -> Dependency:
        """
        Merge two dependencies with the same name, combining packages and configurations.
        
        Args:
            base_dependency: Base dependency (usually from core package)
            extension_dependency: Extension dependency to merge into base
            
        Returns:
            Merged dependency with combined packages and smart configuration merging
        """
        # Start with a copy of the base dependency
        merged = Dependency(
            name=base_dependency.name,
            description=extension_dependency.description,  # Use extension description for specificity
            required=base_dependency.required or extension_dependency.required,  # OR logic for required
            priority=min(base_dependency.priority, extension_dependency.priority),  # Lower number = higher priority
            dependencies=list(set(base_dependency.dependencies + extension_dependency.dependencies)),
            conflicts_with=list(set(base_dependency.conflicts_with + extension_dependency.conflicts_with)),
            min_version=extension_dependency.min_version or base_dependency.min_version,
            max_version=extension_dependency.max_version or base_dependency.max_version,
            package_source=f"{base_dependency.package_source or 'unknown'}+{extension_dependency.package_source or 'unknown'}"
        )
        
        # Merge installations by OS type
        merged_installations = {}
        
        # Start with base installations
        for installation in base_dependency.installations:
            merged_installations[installation.os_type] = installation
        
        # Merge extension installations
        for ext_installation in extension_dependency.installations:
            if ext_installation.os_type in merged_installations:
                # Merge installations for the same OS
                base_installation = merged_installations[ext_installation.os_type]
                merged_installations[ext_installation.os_type] = self._merge_installations(
                    base_installation, ext_installation
                )
            else:
                # Add new OS installation
                merged_installations[ext_installation.os_type] = ext_installation
        
        merged.installations = list(merged_installations.values())
        return merged
    
    def _merge_installations(self, base_installation: DependencyInstallation, 
                           extension_installation: DependencyInstallation) -> DependencyInstallation:
        """
        Merge two dependency installations for the same OS.
        
        Args:
            base_installation: Base installation configuration
            extension_installation: Extension installation to merge
            
        Returns:
            Merged installation with combined packages and configurations
        """
        # Ensure same OS and installation method
        if (base_installation.os_type != extension_installation.os_type or 
            base_installation.install_method != extension_installation.install_method):
            logger.warning(f"Cannot merge installations with different OS types or methods")
            return extension_installation  # Return extension as fallback
        
        # Merge packages (for system_package dependencies)
        merged_packages = list(base_installation.packages)
        for pkg in extension_installation.packages:
            if pkg not in merged_packages:
                merged_packages.append(pkg)
        
        # Merge commands
        merged_commands = list(base_installation.commands)
        for cmd in extension_installation.commands:
            if cmd not in merged_commands:
                merged_commands.append(cmd)
        
        # Merge post-install steps
        merged_post_install = list(base_installation.post_install_steps)
        for step in extension_installation.post_install_steps:
            if step not in merged_post_install:
                merged_post_install.append(step)
        
        # Merge notes
        merged_notes = list(base_installation.notes)
        for note in extension_installation.notes:
            if note not in merged_notes:
                merged_notes.append(note)
        
        # Create merged installation
        merged_installation = DependencyInstallation(
            os_type=base_installation.os_type,
            install_method=base_installation.install_method,
            commands=merged_commands,
            post_install_steps=merged_post_install,
            validation=extension_installation.validation or base_installation.validation,  # Prefer extension validation
            notes=merged_notes,
            documentation_url=extension_installation.documentation_url or base_installation.documentation_url,
            packages=merged_packages
        )
        
        return merged_installation
    
    def _sort_packages_for_merging(self, package_names: List[str]) -> List[str]:
        """Sort packages for consistent merging order, prioritizing core packages."""
        # Put sysagent first as it's the core package
        sorted_packages = []
        
        if 'sysagent' in package_names:
            sorted_packages.append('sysagent')
        
        # Add remaining packages in alphabetical order
        remaining = [pkg for pkg in package_names if pkg != 'sysagent']
        remaining.sort()
        sorted_packages.extend(remaining)
        
        return sorted_packages
    
    def load_default_config(self) -> DependencyConfig:
        """Load the default merged configuration for all discovered packages."""
        # Use dynamic discovery instead of hardcoded packages
        return self.merge_configs()  # None means discover all packages
    
    def _discover_packages(self) -> List[str]:
        """
        Discover packages that have dependency configurations.
        
        DEPRECATED: Use load_all_package_configs() instead.
        This method is kept for backward compatibility.
        """
        config_files = self._discover_dependency_configs()
        packages = []
        
        for config_file in config_files:
            package_name = self._extract_package_name(config_file)
            if package_name not in packages:
                packages.append(package_name)
        
        return self._sort_packages_for_merging(packages)
