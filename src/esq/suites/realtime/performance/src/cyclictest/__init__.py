# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""cyclictest sub-package for RT performance tests.

Splits the cyclictest orchestration logic into focused, reusable modules:

- ``utils``:    Input sanitization, duration/affinity helpers, RT binary
                resolution, timer-migration reads, Allure attachment helpers.
                Shareable across future RT performance tools.

- ``cat``:      Intel L3 CAT cache partition helpers — prerequisite detection,
                partition setup/restore, and report formatting.
                Shareable across future RT performance tools.

- ``cstate``:   CPU C-state disable/restore helpers via cpuidle sysfs.
                Shareable across future RT performance tools.

- ``commands``: Build validated cyclictest and stress-ng command lists from
                profile parameters.

- ``runner``:   Launch cyclictest with concurrent stress-ng, manage a tqdm
                progress bar, and clean up processes unconditionally.

- ``parse``:    Parse the cyclictest native JSON output, derive summary
                metrics, and build the latency distribution chart.
"""

from .cat import (
    check_cat_prerequisites,
    format_cat_report,
    read_cat_partitioned_passive,
    restore_cat_partition,
    setup_cat_partition,
)
from .commands import build_cyclictest_command, build_stress_command
from .cstate import (
    check_cstate_write_access,
    disable_rt_cstates,
    format_cstate_opt_report,
    read_cstate_disabled_passive,
    restore_rt_cstates,
)
from .parse import build_histogram_chart, derive_summary_metrics, parse_cyclictest_json
from .runner import run_cyclictest_with_stress
from .utils import (
    DURATION_ENV_VAR,
    attach_json_file,
    check_cpu_isolation,
    count_cores_in_affinity,
    duration_to_metric,
    get_session_rt_dir,
    parse_core_set,
    parse_duration_to_seconds,
    read_timer_migration,
    resolve_cyclic_duration,
    resolve_rt_binary,
    safe_int,
    sanitize_affinity,
    sanitize_duration,
)

__all__ = [
    # utils
    "DURATION_ENV_VAR",
    "attach_json_file",
    "check_cpu_isolation",
    "count_cores_in_affinity",
    "duration_to_metric",
    "get_session_rt_dir",
    "parse_core_set",
    "parse_duration_to_seconds",
    "read_timer_migration",
    "resolve_cyclic_duration",
    "resolve_rt_binary",
    "safe_int",
    "sanitize_affinity",
    "sanitize_duration",
    # commands
    "build_cyclictest_command",
    "build_stress_command",
    # runner
    "run_cyclictest_with_stress",
    # parse
    "build_histogram_chart",
    "derive_summary_metrics",
    "parse_cyclictest_json",
    # cat
    "check_cat_prerequisites",
    "format_cat_report",
    "read_cat_partitioned_passive",
    "restore_cat_partition",
    "setup_cat_partition",
    # cstate
    "check_cstate_write_access",
    "disable_rt_cstates",
    "format_cstate_opt_report",
    "read_cstate_disabled_passive",
    "restore_rt_cstates",
]
