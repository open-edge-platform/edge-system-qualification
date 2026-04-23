# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
ESQ-specific telemetry package.

ESQ telemetry modules are registered into the sysagent registry via the
``sysagent_telemetry`` entry point, which points directly to
``esq.utils.telemetry.modules``.  Registration happens automatically when the
collector loads extension modules — no explicit import is needed here.
"""
