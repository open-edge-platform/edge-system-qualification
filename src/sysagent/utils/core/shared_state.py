# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Shared global state for the sysagent module.
This module contains global state that needs to be shared across different modules.
"""

# Flag to track if an interrupt has occurred (generic for any signal)
INTERRUPT_OCCURRED = False
INTERRUPT_SIGNAL = None
INTERRUPT_SIGNAL_NAME = "Unknown"
