# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Real-Time Performance — cyclictest wakeup latency measurement.

Runs ``cyclictest`` pinned to isolated CPU cores (default: cores 2–3) while
``stress-ng`` loads the remaining cores, creating realistic scheduling pressure
for a worst-case latency measurement.

Key metrics: ``max_latency_us``, ``avg_latency_us``, ``min_latency_us``.

CAT optimization (optional): when ``cat_enabled: true``, partitions L3 cache
via Intel CAT MSR registers before the run and restores state on completion.
Requires ``system-setup-rt.sh`` MSR Tools module.

C-state optimization (optional): when ``cstate_disable_enabled: true``,
disables deep C-states on RT cores before the run to eliminate wake-from-sleep
jitter.  Requires ``system-setup-rt.sh`` Kernel Tuning module.

Permissions: ``cyclictest`` requires session copies with ``cap_sys_nice`` and
``cap_ipc_lock`` created by ``system-setup-rt.sh`` Real-Time Latency Tools
module in ``/run/user/<UID>/esq/`` (cleared on reboot).  The test skips with
an instructional message when the session copies are absent.

Implementation details are split across the ``src/cyclictest/`` sub-package:
  - ``utils``    Sanitization, duration, affinity, RT binary, timer migration
  - ``commands`` cyclictest and stress-ng command builders
  - ``runner``   Process execution and progress display
  - ``parse``    JSON output parsing and chart generation
  - ``cat``      Intel L3 CAT cache partition (shareable RT optimization)
  - ``cstate``   CPU C-state disable/restore (shareable RT optimization)
"""

import logging
import os
import shutil
import time
from pathlib import Path

import allure
import pytest
from sysagent.utils.config import ensure_dir_permissions
from sysagent.utils.core import Metrics, Result, run_command

from esq.suites.realtime.performance.src.cyclictest import (
    attach_json_file,
    build_cyclictest_command,
    build_histogram_chart,
    build_stress_command,
    check_cat_prerequisites,
    check_cpu_isolation,
    check_cstate_write_access,
    derive_summary_metrics,
    disable_rt_cstates,
    duration_to_metric,
    format_cat_report,
    format_cstate_opt_report,
    get_session_rt_dir,
    parse_core_set,
    parse_cyclictest_json,
    parse_duration_to_seconds,
    read_cat_partitioned_passive,
    read_cstate_disabled_passive,
    read_timer_migration,
    resolve_cyclic_duration,
    resolve_rt_binary,
    restore_cat_partition,
    restore_rt_cstates,
    run_cyclictest_with_stress,
    safe_int,
    sanitize_affinity,
    setup_cat_partition,
)

logger = logging.getLogger(__name__)


@allure.title("Real-Time Latency - cyclictest")
def test_cyclictest(
    request,
    configs,
    cached_result,
    cache_result,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """Measure worst-case RT wakeup latency with cyclictest under stress load.

    Runs cyclictest pinned to isolated cores while stress-ng saturates the
    remaining cores.  The maximum latency (``max_latency_us``) across all RT
    threads is the primary key metric; lower values indicate better real-time
    capability.
    """
    # ================================================================
    # STEP 1: Extract Parameters
    # ================================================================
    test_name = request.node.name.split("[")[0]
    test_id = configs.get("test_id", test_name)
    test_display_name = configs.get("display_name", "RT Latency (cyclictest)")

    description = configs.get("description")
    if description:
        allure.dynamic.description(description)

    is_qualification = configs.get("labels", {}).get("type") == "qualification"

    duration = resolve_cyclic_duration(configs)
    duration_seconds = parse_duration_to_seconds(duration)

    cyclic_affinity = sanitize_affinity(configs.get("cyclic_cpu_affinity"))
    # cyclic_cpu_affinity is mandatory — running without CPU pinning produces
    # non-deterministic results because cyclictest threads can migrate freely.
    # Profiles must explicitly name the RT core(s) to measure.
    if not cyclic_affinity:
        _outcome = pytest.fail if is_qualification else pytest.skip
        _outcome(
            "cyclic_cpu_affinity is not configured. "
            "Set cyclic_cpu_affinity in the profile to specify which CPU core(s) "
            "to pin RT measurement threads to."
        )
    cstate_disable_enabled: bool = bool(configs.get("cstate_disable_enabled", False))
    cstate_max_latency_us: int = safe_int(configs.get("cstate_max_latency_us", 0), 0)

    # Resolve the set of cyclic cores once (reused for isolation check and metrics).
    cyclic_core_set: set[int] = parse_core_set(cyclic_affinity)
    _, isolated_cyclic_cores = check_cpu_isolation(cyclic_core_set)

    logger.info(
        "Starting cyclictest: test=%s, duration=%s (%ss), cores=%s",
        test_id,
        duration,
        duration_seconds,
        cyclic_affinity,
    )

    # ================================================================
    # STEP 2: Validate System Requirements
    # ================================================================
    validate_system_requirements_from_configs(configs)

    # Resolve output directory (Coverity: break taint chain on CORE_DATA_DIR)
    core_data_dir_tainted = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
    core_data_resolved = str(Path(core_data_dir_tainted).resolve())
    core_data_dir = "".join(ch for ch in core_data_resolved)

    expected_base = Path(os.getcwd()).resolve()
    if not Path(core_data_dir).resolve().is_relative_to(expected_base):
        core_data_dir = os.path.join(os.getcwd(), "esq_data")

    results_dir = os.path.join(core_data_dir, "data", "suites", "realtime", "performance", "results", test_id)
    results_resolved = str(Path(results_dir).resolve())
    results_dir = "".join(ch for ch in results_resolved)

    os.makedirs(results_dir, mode=0o770, exist_ok=True)
    ensure_dir_permissions(results_dir, uid=os.getuid(), gid=os.getgid(), mode=0o770)

    output_json_path = os.path.join(results_dir, f"{test_id}_cyclictest_results.json")

    # cyclictest has an internal MAX_PATH buffer of 256 bytes; deep workspace
    # paths frequently exceed that limit, causing strncpy to silently truncate
    # the path so the JSON file is written under a different name than Python
    # later tries to read.
    _cyclic_stage_dir = os.path.join(Path.home(), ".cache", "esq", "cyclictest")
    os.makedirs(_cyclic_stage_dir, mode=0o700, exist_ok=True)
    cyclic_json_path = os.path.join(_cyclic_stage_dir, f"{test_id}_{os.getpid()}_cyclictest_results.json")
    logger.debug("cyclictest JSON staged at: %s (final: %s)", cyclic_json_path, output_json_path)

    # Verify session RT binaries (placed by system-setup-rt.sh Real-Time
    # Latency Tools module). System binaries lack the required file capabilities.
    session_dir = get_session_rt_dir()
    session_cyclic = resolve_rt_binary("cyclictest")
    if session_cyclic is None:
        _outcome = pytest.fail if is_qualification else pytest.skip
        _outcome(
            f"RT session binaries not configured — cyclictest not found at "
            f"{session_dir}/cyclictest. "
            f"Refer to the installation guide to run system-setup-rt.sh."
        )

    session_chrt = resolve_rt_binary("chrt")
    use_chrt = session_chrt is not None
    if not use_chrt:
        logger.info(
            "Session chrt not found at %s/chrt; cyclictest will run without chrt wrapper",
            session_dir,
        )

    # CAT optimization prerequisite check
    cat_enabled: bool = bool(configs.get("cat_enabled", False))
    cat_prereq_ctx: dict = {}
    if cat_enabled:
        cat_ok, cat_skip_reason, cat_prereq_ctx = check_cat_prerequisites()
        if not cat_ok:
            _outcome = pytest.fail if is_qualification else pytest.skip
            _outcome(cat_skip_reason)
        logger.info(
            "CAT prerequisites met: %d total L3 ways available, ref CPU %d; partition will be applied before the run",
            cat_prereq_ctx["total_ways"],
            cat_prereq_ctx["ref_cpu"],
        )

    # C-state optimization prerequisite check
    if cstate_disable_enabled:
        cstate_ok, cstate_skip_reason = check_cstate_write_access(sorted(parse_core_set(cyclic_affinity)))
        if not cstate_ok:
            _outcome = pytest.fail if is_qualification else pytest.skip
            _outcome(cstate_skip_reason)
        logger.info(
            "C-state disable prerequisites met on RT cores %s (max_latency_us=%d); "
            "C-states will be disabled before the run",
            cyclic_affinity,
            cstate_max_latency_us,
        )

    # Compute timeout from the configured duration.
    # Profile can override with an explicit `timeout` value; otherwise a
    # 15-minute setup/teardown buffer is added to the duration.
    timeout = max(
        safe_int(configs.get("timeout", duration_seconds + 900), duration_seconds + 900),
        duration_seconds + 60,
    )
    logger.debug("Effective duration: %s (%ds), timeout: %ds", duration, duration_seconds, timeout)

    cyclic_command = build_cyclictest_command(configs, duration, cyclic_json_path, use_chrt=use_chrt)
    stress_command = build_stress_command(configs, cyclic_affinity, duration)

    stress_available = bool(
        configs.get("stress_enabled", True)
        and run_command(["which", "stress-ng"], timeout=5)
        and run_command(["which", "stress-ng"], timeout=5).returncode == 0
    )
    if configs.get("stress_enabled", True) and not stress_available:
        logger.warning("stress-ng not found; running cyclictest without load stress")
        stress_command = None

    # ================================================================
    # Outcome tracking
    # ================================================================
    result = None
    test_failed = False
    test_interrupted = False
    failure_message = ""

    try:
        # ============================================================
        # STEP 3: Prepare Assets/Dependencies (none required)
        # ============================================================
        prepare_test(
            test_name=test_name,
            prepare_func=lambda: Result(
                name=f"{test_id} - Asset Preparation",
                metadata={"status": "completed"},
            ),
            configs=configs,
            name="Assets",
        )

        # ============================================================
        # STEP 4: Execute Test Logic (with caching)
        # ============================================================
        def _execute_logic():
            cat_baseline: dict | None = None
            cat_partition_info: dict | None = None
            cstate_saved_config: dict[int, dict[int, int]] | None = None
            cstate_applied: list[str] = []
            cstate_failed: list[str] = []
            timer_migration_value: int | None = read_timer_migration()

            try:
                rt_cpu_ids = sorted(parse_core_set(cyclic_affinity))

                if cat_enabled and cat_prereq_ctx:
                    cat_baseline, cat_partition_info = setup_cat_partition(
                        rt_cpu_ids=rt_cpu_ids,
                        configs=configs,
                        prereq_ctx=cat_prereq_ctx,
                    )

                if cstate_disable_enabled:
                    cstate_saved_config, cstate_applied, cstate_failed = disable_rt_cstates(
                        rt_cpu_ids, cstate_max_latency_us
                    )
                    if cstate_applied:
                        logger.info("C-states disabled on RT cores: %s", cstate_applied)
                    elif cstate_failed:
                        logger.warning(
                            "C-state disable: all writes failed — run system-setup-rt.sh "
                            "Kernel Tuning module. Failures: %s",
                            cstate_failed,
                        )
                    else:
                        logger.info("C-state disable: no changes needed")

                logger.info(
                    "Timer migration state at test start: %s",
                    "disabled (RT-safe)"
                    if timer_migration_value == 0
                    else f"enabled (value={timer_migration_value})"
                    if timer_migration_value is not None
                    else "unavailable",
                )

                # Remove any stale staged file from a previous interrupted run.
                # cyclic_json_path is deterministic; if it exists before the runner
                # starts, it belongs to a prior run whose shutil.move() never ran.
                # Removing it ensures the JSON parsed below belongs to this run only.
                if os.path.exists(cyclic_json_path):
                    try:
                        os.unlink(cyclic_json_path)
                    except OSError:
                        pass

                execution_start = time.monotonic()

                run_info = run_cyclictest_with_stress(
                    cyclic_command=cyclic_command,
                    stress_command=stress_command,
                    duration_seconds=duration_seconds,
                    timeout=timeout,
                )

                actual_duration_s = round(time.monotonic() - execution_start, 2)
                dur_value, dur_unit = duration_to_metric(int(actual_duration_s))

                # Parse directly from the staged path so the result is independent
                # of whether the archive move below succeeds or fails.
                threads, top_meta, parse_error = parse_cyclictest_json(cyclic_json_path)
                was_interrupted = run_info.get("interrupted", False)

                if os.path.exists(cyclic_json_path):
                    try:
                        shutil.move(cyclic_json_path, output_json_path)
                        logger.debug(
                            "Archived cyclictest JSON: %s -> %s",
                            cyclic_json_path,
                            output_json_path,
                        )
                    except OSError as _move_err:
                        logger.warning(
                            "Failed to archive cyclictest JSON to %s: %s",
                            output_json_path,
                            _move_err,
                        )

                timer_migration_metric = Metrics(
                    value=float(1 if timer_migration_value == 0 else 0),
                    unit="",
                    is_key_metric=False,
                )

                # Passive system-state metrics — read the kernel/hardware state as
                # it is during the cyclictest measurement window (after any
                # optimizations applied by this test, before they are restored).
                rt_cores_isolated_metric = Metrics(
                    value=float(1 if isolated_cyclic_cores and isolated_cyclic_cores == cyclic_core_set else 0),
                    unit="",
                    is_key_metric=False,
                )
                rt_cstate_disabled_metric = Metrics(
                    value=float(read_cstate_disabled_passive(rt_cpu_ids)),
                    unit="",
                    is_key_metric=False,
                )
                cat_partitioned_metric = Metrics(
                    value=float(read_cat_partitioned_passive()),
                    unit="",
                    is_key_metric=False,
                )

                # Case 1: Process did not succeed AND no usable JSON output
                if not run_info["success"] and (parse_error or not threads):
                    error_msg = f"cyclictest exited with code {run_info['returncode']}" + (
                        f": {run_info['stderr'][:300]}" if run_info.get("stderr") else ""
                    )
                    attach_json_file(output_json_path, f"{test_id}_cyclictest_results.json")
                    res = Result(
                        name=f"{test_id} - {test_display_name}",
                        metadata={"status": False, "interrupted": was_interrupted},
                        extended_metadata={"message": error_msg},
                        metrics={
                            "max_latency_us": Metrics(value=-1.0, unit="\u00b5s", is_key_metric=True),
                            "avg_latency_us": Metrics(value=-1.0, unit="\u00b5s", is_key_metric=False),
                            "min_latency_us": Metrics(value=-1.0, unit="\u00b5s", is_key_metric=False),
                            "run_duration": Metrics(value=dur_value, unit=dur_unit, is_key_metric=False),
                            "timer_migration_disabled": timer_migration_metric,
                            "rt_cores_isolated": rt_cores_isolated_metric,
                            "rt_cstate_disabled": rt_cstate_disabled_metric,
                            "cat_partitioned": cat_partitioned_metric,
                        },
                    )
                    res.update_timestamps()
                    res.metadata["total_duration_seconds"] = actual_duration_s
                    return res

                # Case 2: JSON parse failed despite a clean exit
                if parse_error:
                    error_msg = f"cyclictest ran but JSON output could not be parsed: {parse_error}"
                    logger.error(error_msg)
                    res = Result(
                        name=f"{test_id} - {test_display_name}",
                        metadata={"status": False, "interrupted": was_interrupted},
                        extended_metadata={"message": error_msg},
                        metrics={
                            "max_latency_us": Metrics(value=-1.0, unit="\u00b5s", is_key_metric=True),
                            "avg_latency_us": Metrics(value=-1.0, unit="\u00b5s", is_key_metric=False),
                            "min_latency_us": Metrics(value=-1.0, unit="\u00b5s", is_key_metric=False),
                            "run_duration": Metrics(value=dur_value, unit=dur_unit, is_key_metric=False),
                            "timer_migration_disabled": timer_migration_metric,
                            "rt_cores_isolated": rt_cores_isolated_metric,
                            "rt_cstate_disabled": rt_cstate_disabled_metric,
                            "cat_partitioned": cat_partitioned_metric,
                        },
                    )
                    res.update_timestamps()
                    res.metadata["total_duration_seconds"] = actual_duration_s
                    return res

                # Case 3: Data available — full run or partial from interrupt/timeout
                if not run_info["success"]:
                    result_status = False
                    result_message = (
                        f"cyclictest {'interrupted' if was_interrupted else 'terminated'} "
                        f"after {actual_duration_s}s \u2014 partial results captured "
                        f"(exit code {run_info['returncode']})"
                    )
                    logger.warning(result_message)
                else:
                    result_status = True
                    result_message = "cyclictest completed successfully"

                summary = derive_summary_metrics(threads)
                max_lat = summary["max_latency_us"]
                avg_lat = summary["avg_latency_us"]
                min_lat = summary["min_latency_us"]
                total_cycles = summary["total_cycles"]

                logger.info(
                    "cyclictest complete: max=%s\u00b5s, avg=%s\u00b5s, min=%s\u00b5s, cycles=%d (%d threads)",
                    max_lat,
                    avg_lat,
                    min_lat,
                    total_cycles,
                    len(threads),
                )

                thread_results = [
                    {
                        "id": t["id"],
                        "cpu": t["cpu"],
                        "node": t["node"],
                        "min_us": t["min_us"],
                        "avg_us": t["avg_us"],
                        "max_us": t["max_us"],
                        "cycles": t["cycles"],
                        "histogram": t["histogram"],
                    }
                    for t in threads
                ]

                histogram_chart = build_histogram_chart(
                    threads,
                    run_info={
                        "Run Duration": duration,
                        "Avg Latency": f"{avg_lat}\u00b5s",
                        "Min Latency": f"{min_lat}\u00b5s",
                        "Timer Migration Disabled": "Yes" if timer_migration_metric.value == 1.0 else "No",
                        "RT Cores Isolated": "Yes" if rt_cores_isolated_metric.value == 1.0 else "No",
                        "C-States Disabled": "Yes" if rt_cstate_disabled_metric.value == 1.0 else "No",
                        "CAT Partitioned": "Yes" if cat_partitioned_metric.value == 1.0 else "No",
                    },
                )

                ext_meta: dict = {
                    "message": result_message,
                    "configured_duration": duration,
                    "actual_duration_s": actual_duration_s,
                    "cyclic_command": " ".join(cyclic_command),
                    "stress_command": " ".join(stress_command) if stress_command else "disabled",
                    "output_dir": results_dir,
                    "cyclictest_results": {
                        "num_threads": top_meta.get("num_threads", len(threads)),
                        "resolution_ns": top_meta.get("resolution_ns", 0),
                        "rt_test_version": top_meta.get("rt_test_version", ""),
                        "threads": thread_results,
                    },
                    "charts": [histogram_chart] if histogram_chart else [],
                }

                if cat_partition_info is not None:
                    ext_meta["cat_baseline"] = {
                        "clos0_mask": f"0x{cat_baseline['clos0_mask']:x}",
                        "clos_rt_mask_before": (
                            f"0x{cat_baseline['clos_rt_mask_orig']:x}"
                            if cat_baseline.get("clos_rt_mask_orig") is not None
                            else "default"
                        ),
                        "rt_cpu_pqr_before": {
                            str(k): f"0x{v:x}" if v is not None else "default"
                            for k, v in cat_baseline.get("rt_cpu_pqr_orig", {}).items()
                        },
                    }
                    ext_meta["cat_partition"] = cat_partition_info
                    ext_meta["cat_report"] = format_cat_report(cat_baseline, cat_partition_info)

                if cstate_saved_config is not None:
                    ext_meta["cstate_report"] = format_cstate_opt_report(
                        cstate_saved_config, cstate_applied, rt_cpu_ids, cstate_failed
                    )

                res = Result(
                    name=f"{test_id} - {test_display_name}",
                    metadata={"status": result_status, "interrupted": was_interrupted},
                    extended_metadata=ext_meta,
                    metrics={
                        "max_latency_us": Metrics(value=float(max_lat), unit="\u00b5s", is_key_metric=True),
                        "avg_latency_us": Metrics(value=float(avg_lat), unit="\u00b5s", is_key_metric=False),
                        "min_latency_us": Metrics(value=float(min_lat), unit="\u00b5s", is_key_metric=False),
                        "total_cycles": Metrics(value=float(total_cycles), unit="cycles", is_key_metric=False),
                        "thread_count": Metrics(value=float(len(threads)), unit="", is_key_metric=False),
                        "run_duration": Metrics(value=dur_value, unit=dur_unit, is_key_metric=False),
                        "timer_migration_disabled": timer_migration_metric,
                        "rt_cores_isolated": rt_cores_isolated_metric,
                        "rt_cstate_disabled": rt_cstate_disabled_metric,
                        "cat_partitioned": cat_partitioned_metric,
                    },
                )
                res.parameters["Cyclic Cores"] = cyclic_affinity
                res.parameters["Duration"] = duration
                if stress_command:
                    res.parameters["Stress Command"] = " ".join(stress_command)
                if cat_partition_info:
                    res.parameters["CAT RT CLOS"] = str(cat_partition_info.get("rt_clos", "?"))
                    res.parameters["CAT RT Ways"] = str(cat_partition_info.get("rt_ways", "?"))
                if cstate_saved_config is not None:
                    n_disabled = len(cstate_applied)
                    res.parameters["C-State Opt"] = (
                        f"{n_disabled} state(s) disabled" if n_disabled else "already disabled"
                    )
                res.update_timestamps()
                res.metadata["total_duration_seconds"] = actual_duration_s
                return res

            finally:
                # Restore in reverse setup order: C-states first, then CAT
                if cstate_saved_config is not None:
                    restore_rt_cstates(cstate_saved_config)
                if cat_baseline is not None:
                    restore_cat_partition(cat_baseline)

        # Isolate cache key per resolved duration override so runs with
        # different ENV_SUITE_CYCLICTEST_DURATION values are never shared.
        cache_configs = {**configs, "cyclic_duration": duration}

        result = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=_execute_logic,
            test_name=test_name,
            configs=configs,
            cache_configs=cache_configs,
        )

        # Attach optimization reports to Allure (fresh runs and cache hits)
        if cat_enabled and result and result.extended_metadata:
            cat_report_txt = result.extended_metadata.get("cat_report")
            if cat_report_txt:
                allure.attach(
                    cat_report_txt,
                    name="cat_partition_report.txt",
                    attachment_type=allure.attachment_type.TEXT,
                )

        if cstate_disable_enabled and result and result.extended_metadata:
            cstate_report_txt = result.extended_metadata.get("cstate_report")
            if cstate_report_txt:
                allure.attach(
                    cstate_report_txt,
                    name="cstate_opt_report.txt",
                    attachment_type=allure.attachment_type.TEXT,
                )

        if result.metadata.get("interrupted"):
            test_interrupted = True
            failure_message = result.extended_metadata.get(
                "message", "cyclictest interrupted \u2014 partial results captured"
            )
        elif not result.metadata.get("status", False):
            test_failed = True
            failure_message = result.extended_metadata.get("message", f"{test_display_name} failed")

    except KeyboardInterrupt:
        test_interrupted = True
        failure_message = "Interrupt detected during cyclictest execution"
        logger.error(failure_message)

    except Exception as error:
        test_failed = True
        failure_message = f"Unexpected error during cyclictest: {error}"
        logger.error(failure_message, exc_info=True)

    # ================================================================
    # Ensure a result object always exists for reporting
    # ================================================================
    if result is None:
        dur_v, dur_u = duration_to_metric(duration_seconds)
        result = Result(
            name=f"{test_id} - {test_display_name}",
            metadata={"status": False},
            extended_metadata={"message": failure_message or "cyclictest did not complete"},
            metrics={
                "max_latency_us": Metrics(value=-1.0, unit="\u00b5s", is_key_metric=True),
                "avg_latency_us": Metrics(value=-1.0, unit="\u00b5s", is_key_metric=False),
                "min_latency_us": Metrics(value=-1.0, unit="\u00b5s", is_key_metric=False),
                "run_duration": Metrics(value=dur_v, unit=dur_u, is_key_metric=False),
            },
        )

    # ================================================================
    # STEP 5: Validate Results Against KPIs (qualification profiles only)
    # ================================================================
    try:
        validate_test_results(
            results=result,
            configs=configs,
            get_kpi_config=get_kpi_config,
            test_name=test_name,
        )
    except Exception as validation_error:
        logger.error("KPI validation failed: %s", validation_error)

    # ================================================================
    # STEP 6: Generate Summary (always runs)
    # ================================================================
    try:
        summarize_test_results(
            results=result,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
    except Exception as summary_error:
        logger.error("Summarization failed: %s", summary_error, exc_info=True)

    cache_result(result)

    logger.info("Test completed: %s", test_display_name)

    # ================================================================
    # STEP 7: Surface the Outcome
    # ================================================================
    if test_interrupted:
        if is_qualification:
            pytest.fail(failure_message)
        else:
            raise RuntimeError(failure_message)
    if test_failed:
        pytest.fail(failure_message)
