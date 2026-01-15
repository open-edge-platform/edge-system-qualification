# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Infrastructure utilities package.

Provides utilities for infrastructure management including Docker operations,
Node.js setup, and third-party dependency management with conditional imports.
"""

# Conditional imports - only import if available
try:
    from .docker_client import DockerClient
except ImportError:
    # Define dummy classes/functions for missing dependencies
    class DockerClient:
        def __init__(self):
            raise ImportError("Docker package not available.")


try:
    from .node import (
        NODE_VERSION,
        extract_nodejs_archive,
        get_node_binary_paths,
        get_node_env_vars,
        get_yarn_binary_path,
        install_nodejs,
        install_yarn_global,
        is_nodejs_installed,
        setup_nodejs,
        verify_nodejs_installation,
    )
except ImportError:
    # Node utilities might have dependencies
    pass

try:
    from .setup_dependency import (
        cleanup_dependency,
        download_and_prepare_audio,
        download_file,
        download_github_repo,
        extract_zip_file,
        get_dependency_info,
        list_installed_dependencies,
        setup_all_dependencies,
        setup_allure,
        setup_dependency,
        setup_edge_developer_kit,
        setup_node,
        verify_dependency_installation,
    )
except ImportError:
    # Setup dependency utilities might need requests
    pass

try:
    from .network import (
        check_internet_connectivity,
        check_network_restrictions,
        get_cached_network_restrictions,
        get_preferred_download_source,
        test_connectivity,
    )
except ImportError:
    # Network utilities might have dependencies
    pass

# Import new modular dependency system
from sysagent.utils.core.dependencies import get_dependency_manager

# Export all available functions and classes
__all__ = [
    # Docker client
    "DockerClient",
    # Node utilities
    "NODE_VERSION",
    "is_nodejs_installed",
    "get_node_binary_paths",
    "get_yarn_binary_path",
    "install_yarn_global",
    "extract_nodejs_archive",
    "install_nodejs",
    "setup_nodejs",
    "verify_nodejs_installation",
    "get_node_env_vars",
    # Setup dependency utilities
    "download_file",
    "extract_zip_file",
    "download_and_prepare_audio",
    "setup_allure",
    "setup_edge_developer_kit",
    "setup_node",
    "setup_dependency",
    "setup_all_dependencies",
    "verify_dependency_installation",
    "cleanup_dependency",
    "list_installed_dependencies",
    "get_dependency_info",
    "download_github_repo",
    # Network utilities
    "check_internet_connectivity",
    "check_network_restrictions",
    "get_cached_network_restrictions",
    "get_preferred_download_source",
    "test_connectivity",
    # Dependency system
    "get_dependency_manager",
]
