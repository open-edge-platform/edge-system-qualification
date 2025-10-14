# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Infrastructure utilities package.

Provides utilities for infrastructure management including Docker operations,
Node.js setup, and third-party dependency management with conditional imports.
"""

# Conditional imports - only import if available
try:
    from .docker_client import *
except ImportError:
    # Define dummy classes/functions for missing dependencies
    class DockerClient:
        def __init__(self):
            raise ImportError("Docker package not available. Install with: pip install docker")

try:
    from .node import *
except ImportError:
    # Node utilities might have fewer dependencies, but handle gracefully
    pass

try:
    from .setup_dependency import *
except ImportError:
    # Setup dependency utilities might need requests, handle gracefully
    pass

# Import new modular dependency system
from sysagent.utils.core.dependencies import get_dependency_manager

# Re-export all functions and classes that are available
__all__ = [
    # From docker_client (if available)
    'DockerClient',
    
    # From new dependency system
    'get_dependency_manager'
]

# Dynamically add to __all__ based on what was successfully imported
import sys
current_module = sys.modules[__name__]

# Add node functions if available
if hasattr(current_module, 'NODE_VERSION'):
    __all__.extend([
        'NODE_VERSION',
        'is_nodejs_installed',
        'get_node_binary_paths',
        'get_yarn_binary_path',
        'install_yarn_global',
        'extract_nodejs_archive',
        'install_nodejs',
        'setup_nodejs',
        'verify_nodejs_installation',
        'get_node_env_vars',
    ])

# Add setup_dependency functions if available
if hasattr(current_module, 'setup_dependency'):
    __all__.extend([
        'download_file',
        'extract_zip_file',
        'download_and_prepare_audio',
        'setup_allure',
        'setup_edge_developer_kit',
        'setup_node',
        'setup_dependency',
        'setup_all_dependencies',
        'verify_dependency_installation',
        'cleanup_dependency',
        'list_installed_dependencies',
        'get_dependency_info',
        'download_github_repo',
    ])
