# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# version will be injected by setuptools_scm
try:
    from ._version import version as __version__
except ImportError:
    from importlib.metadata import version
    __version__ = version("esq")