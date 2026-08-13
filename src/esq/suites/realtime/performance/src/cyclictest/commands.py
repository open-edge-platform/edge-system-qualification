# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Command builders for cyclictest and stress-ng.

Constructs validated, taint-free command lists for both tools from profile
parameters.  No ``shell=True`` is used anywhere — all arguments are passed as
discrete list elements.
"""

import logging

from .utils import (
    build_stress_affinity_all_except,
    count_cores_in_affinity,
    parse_duration_to_seconds,
    resolve_rt_binary,
    safe_int,
    sanitize_affinity,
)

logger = logging.getLogger(__name__)


def build_cyclictest_command(
    configs: dict,
    duration: str,
    json_output_path: str,
    use_chrt: bool = True,
) -> list[str]:
    """Build the validated cyclictest command list from profile parameters.

    Prepends ``chrt -f <priority>`` when *use_chrt* is True so the process
    enters SCHED_FIFO before setting up timer threads.
    """
    affinity = sanitize_affinity(configs.get("cyclic_cpu_affinity"))
    # When no affinity is configured, default to a single thread; otherwise
    # derive the thread count from the number of pinned cores.
    threads = count_cores_in_affinity(affinity) if affinity else 1
    priority = max(min(safe_int(configs.get("cyclic_priority", 99), 99), 99), 1)
    interval_us = max(safe_int(configs.get("cyclic_interval_us", 250), 250), 1)
    histogram_us = max(safe_int(configs.get("cyclic_histogram_us", 700), 700), 1)
    mlock = bool(configs.get("cyclic_mlock", True))
    main_affinity_raw = configs.get("cyclic_main_affinity")
    main_affinity = sanitize_affinity(main_affinity_raw) if main_affinity_raw is not None else None

    # Break taint chain on the output path before embedding in command
    safe_json_path = "".join(ch for ch in str(json_output_path) if ch.isprintable() and ch not in "'\";|&`$")

    # Wrap with chrt so the process enters SCHED_FIFO before any cyclictest
    # code runs.  Prefer the session copy; fall back to system chrt.
    command: list[str] = []
    if use_chrt:
        command += [resolve_rt_binary("chrt"), "-f", str(priority)]

    command.append(resolve_rt_binary("cyclictest"))
    # -a is only added when cyclic_cpu_affinity is explicitly configured;
    # omitting it lets cyclictest run on any available CPU (baseline mode).
    if affinity:
        command.append(f"-a{affinity}")
    command += [
        f"-t{threads}",
        f"-p{priority}",
        f"-i{interval_us}",
        f"-h{histogram_us}",
        "-q",
        f"-D{duration}",
        f"--json={safe_json_path}",
    ]
    if mlock:
        command.append("-m")
    if main_affinity is not None:
        command.append(f"--mainaffinity={main_affinity}")

    return command


def build_stress_command(
    configs: dict,
    cyclic_affinity: str,
    duration: str,
) -> list[str] | None:
    """Build the stress-ng command to run concurrently with cyclictest.

    Returns None when ``stress_enabled`` is False.
    """
    if not bool(configs.get("stress_enabled", True)):
        return None

    all_except = bool(configs.get("stress_all_except_cyclic", False))
    if all_except:
        raw_affinity = build_stress_affinity_all_except(cyclic_affinity)
    else:
        raw_affinity = str(configs.get("stress_cpu_affinity", "0-1"))

    affinity = sanitize_affinity(raw_affinity)
    workers_cfg = safe_int(configs.get("stress_cpu_workers", 0), 0)
    workers = workers_cfg if workers_cfg > 0 else count_cores_in_affinity(affinity)
    cpu_load = max(min(safe_int(configs.get("stress_cpu_load", 100), 100), 100), 1)

    # Run stress slightly longer than cyclictest so it is always active during
    # the measurement window; kill it explicitly after cyclictest finishes.
    duration_s = parse_duration_to_seconds(duration) + 60
    stress_timeout = f"{duration_s}s"

    return [
        "stress-ng",
        "--taskset",
        affinity,
        "--cpu",
        str(workers),
        "--cpu-load",
        str(cpu_load),
        "--timeout",
        stress_timeout,
    ]
