# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
import os
from pathlib import Path

from esq.suites.ai.gen.src.chatqna_core import (
    ChatQnAComposeManager,
    append_performance_row,
    build_runtime_env,
    ensure_chatqna_report_paths,
    generate_performance_graphs,
    generate_presentation_csv,
    get_selected_services,
    render_nginx_config,
    resolve_runtime_paths,
    run_chatqna_benchmark,
    wait_for_service_health,
    write_scenario_metadata,
)
from sysagent.utils.core import Metrics, Result

try:
    _allure = importlib.import_module("allure")
except ImportError:
    _allure = None

logger = logging.getLogger(__name__)

METRIC_DECIMAL_PLACES = 2


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def _as_float(value: object, default: float = -1.0) -> float:
    try:
        return float(str(value))
    except Exception:
        return default


def _as_list_of_dicts(value: object) -> list:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _round_metric(value: object, default: float = -1.0) -> float:
    metric_value = _as_float(value, default=default)
    if metric_value < 0:
        return default
    return round(metric_value, METRIC_DECIMAL_PLACES)


def _pick_best_scenario(scenario_results: list) -> dict:
    if not scenario_results:
        return {}

    def _rank(entry: dict) -> tuple:
        status = 1 if _as_bool(entry.get("status", False), default=False) else 0
        success_rate = _as_float(entry.get("query_success_rate", 0.0), default=0.0)
        tokens_per_second = _as_float(entry.get("estimated_output_tokens_per_second", -1.0), default=-1.0)
        p95_latency = _as_float(entry.get("p95_query_latency_ms", -1.0), default=-1.0)
        latency_score = -p95_latency if p95_latency >= 0.0 else float("-inf")
        return (status, success_rate, tokens_per_second, latency_score)

    best_entry = max(scenario_results, key=_rank)
    return dict(best_entry) if isinstance(best_entry, dict) else {}


def test_chatqna_core(
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
    """Chat Question-and-Answer Core sample application skeleton for ESQ."""

    test_name = request.node.name.split("[")[0]
    test_display_name = configs.get("display_name", test_name)
    test_id = configs.get("test_id", test_name)
    timeout = int(configs.get("timeout", 1800))
    compose_project_name = str(configs.get("compose_project_name", "esq-chatqna-core"))
    preserve_environment = _as_bool(configs.get("preserve_environment", False), default=False)
    include_ui = _as_bool(configs.get("include_ui", False), default=False)
    pull_images = _as_bool(configs.get("pull_images", True), default=True)

    core_data_dir = Path(os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))).resolve()
    suite_assets_root = core_data_dir / "data" / "suites" / "ai" / "gen" / "chatqna_core" / "assets"
    resolved_paths = resolve_runtime_paths(configs=configs, suite_assets_root=str(suite_assets_root))

    # Make resolved benchmark inputs visible to helper code and cache metadata.
    configs["corpus_dir"] = resolved_paths["corpus_dir"]
    configs["prompt_file"] = resolved_paths["prompt_file"]
    configs["model_config_path"] = resolved_paths["model_config_path"]
    configs["compose_file"] = resolved_paths["compose_file"]

    validate_system_requirements_from_configs(configs)

    runtime_env = build_runtime_env(configs=configs, resolved_paths=resolved_paths)
    compose_manager = ChatQnAComposeManager(
        compose_file=resolved_paths["compose_file"],
        project_name=compose_project_name,
        timeout=timeout,
    )

    # Idempotent guard: container teardown should happen exactly once even if
    # pytest receives SIGINT (Ctrl+C) before the finally block runs.
    _teardown_done = [False]

    def _teardown_compose() -> None:
        if _teardown_done[0] or preserve_environment:
            return
        _teardown_done[0] = True
        try:
            compose_manager.bring_down(env=runtime_env)
        except Exception as _err:
            logger.warning("ChatQnA compose bring_down failed during teardown: %s", _err)

    request.addfinalizer(_teardown_compose)

    results = None

    def prepare_assets() -> Result:
        compose_manager.validate_runtime()
        compose_manager.validate_paths()
        if include_ui:
            render_nginx_config(resolved_paths=resolved_paths, env=runtime_env)
        if pull_images:
            compose_manager.pull_images(env=runtime_env)
        return Result(
            metadata={
                "status": True,
                "compose_file": resolved_paths["compose_file"],
                "compose_project_name": compose_project_name,
                "reference_root": resolved_paths["reference_root"],
            }
        )

    try:
        prepare_test(test_name=test_name, prepare_func=prepare_assets, configs=configs, name="Assets")

        # Captured by execute_logic so per-scenario CSV/JSON artifacts can be
        # built without persisting the full list in Result.metadata.
        run_scenario_results: list = []

        def execute_logic() -> Result:
            scenario_matrix = _as_list_of_dicts(configs.get("scenario_matrix", []))
            pulled_image_signatures = set()

            def run_single_scenario(case_overrides: dict) -> dict:
                case_configs = dict(configs)
                # Drop outer-injected backend-dependent paths so each scenario
                # re-resolves them (e.g. picks ollama_template vs openvino_template).
                if isinstance(case_overrides, dict):
                    for backend_dependent_key in ("model_config_path", "compose_file"):
                        if backend_dependent_key not in case_overrides:
                            case_configs.pop(backend_dependent_key, None)
                    case_configs.update(case_overrides)

                case_test_id = str(case_configs.get("test_id", test_id))
                case_display_name = str(case_configs.get("display_name", test_display_name))
                case_timeout = int(case_configs.get("timeout", timeout))
                case_compose_project_name = str(case_configs.get("compose_project_name", compose_project_name))
                case_preserve_environment = _as_bool(
                    case_configs.get("preserve_environment", preserve_environment),
                    default=preserve_environment,
                )
                case_include_ui = _as_bool(case_configs.get("include_ui", include_ui), default=include_ui)
                case_pull_images = _as_bool(case_configs.get("pull_images", pull_images), default=pull_images)

                case_resolved_paths = resolve_runtime_paths(
                    configs=case_configs,
                    suite_assets_root=str(suite_assets_root),
                )
                case_configs["corpus_dir"] = case_resolved_paths["corpus_dir"]
                case_configs["prompt_file"] = case_resolved_paths["prompt_file"]
                case_configs["model_config_path"] = case_resolved_paths["model_config_path"]
                case_configs["compose_file"] = case_resolved_paths["compose_file"]

                case_runtime_env = build_runtime_env(configs=case_configs, resolved_paths=case_resolved_paths)
                case_compose_manager = ChatQnAComposeManager(
                    compose_file=case_resolved_paths["compose_file"],
                    project_name=case_compose_project_name,
                    timeout=case_timeout,
                )

                case_selected_services = get_selected_services(case_configs)
                startup_timeout = int(case_configs.get("startup_timeout_seconds", 2700))
                startup_poll = int(case_configs.get("startup_poll_interval_seconds", 15))
                startup_log_timeout = int(case_configs.get("startup_log_timeout_seconds", min(600, startup_timeout)))
                health_url = str(case_configs.get("health_url", "http://127.0.0.1:8888/health"))
                startup_attempts = max(1, _as_int(case_configs.get("startup_attempts", 2), default=2))

                service_startup_seconds = -1.0
                startup_error = None

                signature = (
                    case_resolved_paths["compose_file"],
                    case_runtime_env.get("COMPOSE_PROFILES", ""),
                    case_runtime_env.get("BACKEND_TAG", ""),
                    case_runtime_env.get("UI_TAG", ""),
                )

                try:
                    case_compose_manager.validate_runtime()
                    case_compose_manager.validate_paths()
                    if case_include_ui:
                        render_nginx_config(resolved_paths=case_resolved_paths, env=case_runtime_env)
                    if case_pull_images and signature not in pulled_image_signatures:
                        case_compose_manager.pull_images(env=case_runtime_env)
                        pulled_image_signatures.add(signature)

                    for attempt in range(1, startup_attempts + 1):
                        if not case_preserve_environment:
                            case_compose_manager.bring_down(env=case_runtime_env)

                        case_compose_manager.bring_up(services=case_selected_services, env=case_runtime_env)

                        try:
                            services_ready = case_compose_manager.wait_for_services_ready(
                                services=case_selected_services,
                                env=case_runtime_env,
                                timeout_seconds=120,
                                poll_interval_seconds=5,
                            )
                            if not services_ready:
                                raise RuntimeError(f"Compose services did not start in time: {case_selected_services}")

                            backend_service = str(case_configs.get("backend_service", case_selected_services[0]))
                            log_ready = case_compose_manager.wait_for_container_log_ready(
                                service_name=backend_service,
                                ready_log_marker="Application startup complete.",
                                env=case_runtime_env,
                                timeout_seconds=max(1, min(startup_log_timeout, startup_timeout)),
                                poll_interval_seconds=startup_poll,
                            )

                            if log_ready:
                                logger.info(
                                    "Log marker found for '%s'; confirming via HTTP health (%ds timeout).",
                                    backend_service,
                                    startup_timeout,
                                )
                            else:
                                logger.warning(
                                    (
                                        "Log marker not found for '%s' within %ds; "
                                        "falling back to HTTP health poll (%ds timeout)."
                                    ),
                                    backend_service,
                                    max(1, min(startup_log_timeout, startup_timeout)),
                                    startup_timeout,
                                )

                            service_startup_seconds = wait_for_service_health(
                                health_url=health_url,
                                timeout_seconds=startup_timeout,
                                poll_interval_seconds=5,
                            )
                            startup_error = None
                            break
                        except Exception as attempt_error:
                            startup_error = attempt_error
                            logger.warning(
                                "ChatQnA startup attempt %d/%d failed for scenario '%s': %s",
                                attempt,
                                startup_attempts,
                                case_test_id,
                                attempt_error,
                            )
                            if attempt < startup_attempts:
                                logger.info("Retrying ChatQnA startup after compose restart...")
                                continue
                            raise

                    if startup_error is not None:
                        raise startup_error

                    benchmark_results = run_chatqna_benchmark(
                        configs=case_configs,
                        api_base_url=str(case_configs.get("api_base_url", "http://127.0.0.1:8888")),
                    )

                    successful_queries = _as_int(benchmark_results.get("successful_queries", 0), default=0)
                    failed_queries = _as_int(benchmark_results.get("failed_queries", 0), default=0)
                    final_status = successful_queries > 0 and failed_queries == 0

                    return {
                        "test_id": case_test_id,
                        "display_name": case_display_name,
                        "status": final_status,
                        "backend_runtime": str(case_configs.get("backend_runtime", "openvino")),
                        "compute_device": str(case_configs.get("compute_device", "cpu")),
                        "service_startup_seconds": _round_metric(service_startup_seconds),
                        "document_ingestion_seconds": _round_metric(
                            benchmark_results.get("document_ingestion_seconds", -1.0),
                            default=-1.0,
                        ),
                        "p50_query_latency_ms": _round_metric(
                            benchmark_results.get("p50_query_latency_ms", -1.0),
                            default=-1.0,
                        ),
                        "p95_query_latency_ms": _round_metric(
                            benchmark_results.get("p95_query_latency_ms", -1.0),
                            default=-1.0,
                        ),
                        "ttft_ms": _round_metric(benchmark_results.get("ttft_ms", -1.0), default=-1.0),
                        "query_success_rate": _round_metric(
                            benchmark_results.get("query_success_rate", 0.0),
                            default=0.0,
                        ),
                        "successful_queries": successful_queries,
                        "failed_queries": failed_queries,
                        "corpus_document_count": _as_int(benchmark_results.get("corpus_document_count", 0), default=0),
                        "prompt_count": _as_int(benchmark_results.get("prompt_count", 0), default=0),
                        "ref_p50_query_latency_ms": _round_metric(
                            case_configs.get("ref_p50_query_latency_ms", -1.0),
                            default=-1.0,
                        ),
                        "ref_p95_query_latency_ms": _round_metric(
                            case_configs.get("ref_p95_query_latency_ms", -1.0),
                            default=-1.0,
                        ),
                        "ref_ttft_ms": _round_metric(case_configs.get("ref_ttft_ms", -1.0), default=-1.0),
                    }
                finally:
                    if not case_preserve_environment:
                        try:
                            case_compose_manager.bring_down(env=case_runtime_env)
                        except Exception as teardown_error:
                            logger.warning(
                                "ChatQnA compose bring_down failed after scenario '%s': %s",
                                case_test_id,
                                teardown_error,
                            )

            if scenario_matrix:
                scenario_results = []
                for case in scenario_matrix:
                    case_test_id = str(case.get("test_id", test_id)) if isinstance(case, dict) else str(test_id)
                    case_display_name = (
                        str(case.get("display_name", case_test_id))
                        if isinstance(case, dict)
                        else str(test_display_name)
                    )

                    try:
                        scenario_results.append(run_single_scenario(case))
                    except Exception as scenario_error:
                        logger.exception(
                            "Consolidated scenario '%s' failed; continuing with remaining scenarios.",
                            case_test_id,
                        )

                        case_configs = dict(configs)
                        if isinstance(case, dict):
                            case_configs.update(case)
                        case_resolved_paths = resolve_runtime_paths(
                            configs=case_configs,
                            suite_assets_root=str(suite_assets_root),
                        )

                        scenario_results.append(
                            {
                                "test_id": case_test_id,
                                "display_name": case_display_name,
                                "status": False,
                                "error": str(scenario_error),
                                "backend_runtime": str(case_configs.get("backend_runtime", "openvino")),
                                "compute_device": str(case_configs.get("compute_device", "cpu")),
                                "service_startup_seconds": -1.0,
                                "document_ingestion_seconds": -1.0,
                                "p50_query_latency_ms": -1.0,
                                "p95_query_latency_ms": -1.0,
                                "ttft_ms": -1.0,
                                "query_success_rate": 0.0,
                                "successful_queries": 0,
                                "failed_queries": 0,
                                "corpus_document_count": 0,
                                "prompt_count": 0,
                                "ref_p50_query_latency_ms": case_configs.get("ref_p50_query_latency_ms", -1.0),
                                "ref_p95_query_latency_ms": case_configs.get("ref_p95_query_latency_ms", -1.0),
                                "ref_ttft_ms": case_configs.get("ref_ttft_ms", -1.0),
                            }
                        )

                best_scenario = _pick_best_scenario(scenario_results)
                aggregate_status = all(_as_bool(item.get("status", False), default=False) for item in scenario_results)
                run_scenario_results.clear()
                run_scenario_results.extend(scenario_results)

                return Result.from_test_config(
                    configs=configs,
                    parameters={
                        "Display Name": test_display_name,
                        "Scenario Count": len(scenario_results),
                        "Compose File": best_scenario.get("compose_file", resolved_paths["compose_file"]),
                        "Model Config": best_scenario.get("model_config_path", resolved_paths["model_config_path"]),
                        "Corpus Dir": best_scenario.get("corpus_dir", resolved_paths["corpus_dir"]),
                        "Prompt File": best_scenario.get("prompt_file", resolved_paths["prompt_file"]),
                    },
                    metrics={
                        "service_startup_seconds": Metrics(
                            unit="secs",
                            value=_as_float(best_scenario.get("service_startup_seconds", -1.0), default=-1.0),
                        ),
                        "document_ingestion_seconds": Metrics(
                            unit="secs",
                            value=_as_float(best_scenario.get("document_ingestion_seconds", -1.0), default=-1.0),
                        ),
                        "p50_query_latency_ms": Metrics(
                            unit="ms",
                            value=_as_float(best_scenario.get("p50_query_latency_ms", -1.0), default=-1.0),
                        ),
                        "p95_query_latency_ms": Metrics(
                            unit="ms",
                            value=_as_float(best_scenario.get("p95_query_latency_ms", -1.0), default=-1.0),
                            is_key_metric=True,
                        ),
                        "ttft_ms": Metrics(
                            unit="ms",
                            value=_as_float(best_scenario.get("ttft_ms", -1.0), default=-1.0),
                        ),
                        "query_success_rate": Metrics(
                            unit="ratio",
                            value=_as_float(best_scenario.get("query_success_rate", 0.0), default=0.0),
                        ),
                    },
                    metadata={
                        "status": aggregate_status,
                        "is_consolidated": True,
                        "consolidated_scenario_count": len(scenario_results),
                        "backend_runtime": best_scenario.get("backend_runtime", configs.get("backend_runtime", "")),
                        "compute_device": best_scenario.get("compute_device", configs.get("compute_device", "")),
                        "successful_queries": _as_int(best_scenario.get("successful_queries", 0), default=0),
                        "failed_queries": _as_int(best_scenario.get("failed_queries", 0), default=0),
                    },
                )

            single_scenario = run_single_scenario({})
            return Result.from_test_config(
                configs=configs,
                parameters={
                    "Display Name": test_display_name,
                    "Backend Runtime": single_scenario.get("backend_runtime", "openvino"),
                    "Compute Device": single_scenario.get("compute_device", "cpu"),
                    "Include UI": include_ui,
                    "Compose File": single_scenario.get("compose_file", resolved_paths["compose_file"]),
                    "Model Config": single_scenario.get("model_config_path", resolved_paths["model_config_path"]),
                    "Corpus Dir": single_scenario.get("corpus_dir", resolved_paths["corpus_dir"]),
                    "Prompt File": single_scenario.get("prompt_file", resolved_paths["prompt_file"]),
                },
                metrics={
                    "service_startup_seconds": Metrics(
                        unit="secs",
                        value=_as_float(single_scenario.get("service_startup_seconds", -1.0), default=-1.0),
                    ),
                    "document_ingestion_seconds": Metrics(
                        unit="secs",
                        value=_as_float(single_scenario.get("document_ingestion_seconds", -1.0), default=-1.0),
                    ),
                    "p50_query_latency_ms": Metrics(
                        unit="ms",
                        value=_as_float(single_scenario.get("p50_query_latency_ms", -1.0), default=-1.0),
                    ),
                    "p95_query_latency_ms": Metrics(
                        unit="ms",
                        value=_as_float(single_scenario.get("p95_query_latency_ms", -1.0), default=-1.0),
                        is_key_metric=True,
                    ),
                    "ttft_ms": Metrics(
                        unit="ms",
                        value=_as_float(single_scenario.get("ttft_ms", -1.0), default=-1.0),
                    ),
                    "query_success_rate": Metrics(
                        unit="ratio",
                        value=_as_float(single_scenario.get("query_success_rate", 0.0), default=0.0),
                    ),
                },
                metadata={
                    "status": _as_bool(single_scenario.get("status", False), default=False),
                    "backend_runtime": single_scenario.get("backend_runtime", configs.get("backend_runtime", "")),
                    "compute_device": single_scenario.get("compute_device", configs.get("compute_device", "")),
                    "successful_queries": _as_int(single_scenario.get("successful_queries", 0), default=0),
                    "failed_queries": _as_int(single_scenario.get("failed_queries", 0), default=0),
                    "model_config_path": single_scenario.get("model_config_path", resolved_paths["model_config_path"]),
                    "corpus_dir": single_scenario.get("corpus_dir", resolved_paths["corpus_dir"]),
                    "prompt_file": single_scenario.get("prompt_file", resolved_paths["prompt_file"]),
                },
            )

        results = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=execute_logic,
            test_name=test_name,
            configs=configs,
            cache_configs={
                "test_id": test_id,
                "backend_runtime": configs.get("backend_runtime", "openvino"),
                "compute_device": configs.get("compute_device", "cpu"),
                "include_ui": include_ui,
                "compose_profile": configs.get("compose_profile", "OPENVINO"),
                "model_config_path": resolved_paths["model_config_path"],
                "corpus_dir": resolved_paths["corpus_dir"],
                "prompt_file": resolved_paths["prompt_file"],
                "backend_tag": configs.get("backend_tag", "core_2026.1.0"),
            },
        )

        validate_test_results(
            test_name=test_name,
            results=results,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )

        try:
            is_consolidated = bool(_as_list_of_dicts(configs.get("scenario_matrix", [])))
            report_paths = ensure_chatqna_report_paths(str(core_data_dir), consolidated=is_consolidated)

            # Individual scenario runs should always produce a single-row CSV and
            # single-run plots. Clear prior individual artifacts before writing.
            if not is_consolidated:
                cleanup_paths = [
                    report_paths["csv_path"],
                    report_paths["presentation_csv_path"],
                    report_paths["metadata_json_path"],
                    report_paths["p50_plot_path"],
                    report_paths["p95_plot_path"],
                    report_paths["ttft_plot_path"],
                    report_paths["combined_plot_path"],
                    str(Path(report_paths["report_dir"]) / "chatqna_core_combined_latency.png"),
                ]
                for artifact_path in cleanup_paths:
                    artifact_file = Path(artifact_path)
                    if artifact_file.exists():
                        artifact_file.unlink()

            scenario_results = run_scenario_results

            for scenario_entry in scenario_results:
                append_performance_row(
                    csv_path=report_paths["csv_path"],
                    row_data={
                        "test_id": scenario_entry.get("test_id", test_id),
                        "display_name": scenario_entry.get("display_name", test_display_name),
                        "backend_runtime": scenario_entry.get(
                            "backend_runtime",
                            configs.get("backend_runtime", "openvino"),
                        ),
                        "compute_device": scenario_entry.get("compute_device", configs.get("compute_device", "cpu")),
                        "model_config": Path(str(scenario_entry.get("model_config_path", ""))).name,
                        "corpus_id": Path(str(scenario_entry.get("corpus_dir", ""))).name,
                        "prompt_set_id": Path(str(scenario_entry.get("prompt_file", ""))).stem,
                        "service_startup_seconds": scenario_entry.get("service_startup_seconds", -1.0),
                        "document_ingestion_seconds": scenario_entry.get("document_ingestion_seconds", -1.0),
                        "p50_query_latency_ms": scenario_entry.get("p50_query_latency_ms", -1.0),
                        "p95_query_latency_ms": scenario_entry.get("p95_query_latency_ms", -1.0),
                        "ttft_ms": scenario_entry.get("ttft_ms", -1.0),
                        "query_success_rate": scenario_entry.get("query_success_rate", 0.0),
                        "ref_platform": configs.get("platform", ""),
                        "ref_p50_query_latency_ms": scenario_entry.get("ref_p50_query_latency_ms", -1.0),
                        "ref_p95_query_latency_ms": scenario_entry.get("ref_p95_query_latency_ms", -1.0),
                        "ref_ttft_ms": scenario_entry.get("ref_ttft_ms", -1.0),
                        "status": (
                            "passed" if _as_bool(scenario_entry.get("status", False), default=False) else "failed"
                        ),
                    },
                )

            if not is_consolidated:
                append_performance_row(
                    csv_path=report_paths["csv_path"],
                    row_data={
                        "test_id": test_id,
                        "display_name": test_display_name,
                        "backend_runtime": results.metadata.get(
                            "backend_runtime",
                            configs.get("backend_runtime", "openvino"),
                        ),
                        "compute_device": results.metadata.get(
                            "compute_device",
                            configs.get("compute_device", "cpu"),
                        ),
                        "model_config": Path(
                            str(results.metadata.get("model_config_path", resolved_paths["model_config_path"]))
                        ).name,
                        "corpus_id": Path(str(results.metadata.get("corpus_dir", resolved_paths["corpus_dir"]))).name,
                        "prompt_set_id": Path(
                            str(results.metadata.get("prompt_file", resolved_paths["prompt_file"]))
                        ).stem,
                        "service_startup_seconds": results.metrics["service_startup_seconds"].value,
                        "document_ingestion_seconds": results.metrics["document_ingestion_seconds"].value,
                        "p50_query_latency_ms": results.metrics["p50_query_latency_ms"].value,
                        "p95_query_latency_ms": results.metrics["p95_query_latency_ms"].value,
                        "ttft_ms": results.metrics["ttft_ms"].value,
                        "query_success_rate": results.metrics["query_success_rate"].value,
                        "ref_platform": configs.get("platform", ""),
                        "ref_p50_query_latency_ms": configs.get("ref_p50_query_latency_ms", -1.0),
                        "ref_p95_query_latency_ms": configs.get("ref_p95_query_latency_ms", -1.0),
                        "ref_ttft_ms": configs.get("ref_ttft_ms", -1.0),
                        "status": "passed" if results.metadata.get("status", False) else "failed",
                    },
                )
            generate_presentation_csv(
                csv_path=report_paths["csv_path"],
                presentation_csv_path=report_paths["presentation_csv_path"],
            )
            plot_files = generate_performance_graphs(
                csv_path=report_paths["csv_path"],
                p50_plot_path=report_paths["p50_plot_path"],
                p95_plot_path=report_paths["p95_plot_path"],
                ttft_plot_path=report_paths["ttft_plot_path"],
            )

            scenario_metadata = {
                "test_id": test_id,
                "display_name": test_display_name,
                "backend_runtime": results.metadata.get("backend_runtime", configs.get("backend_runtime", "openvino")),
                "compute_device": results.metadata.get("compute_device", configs.get("compute_device", "cpu")),
                "successful_queries": results.metadata.get("successful_queries", 0),
                "failed_queries": results.metadata.get("failed_queries", 0),
                "is_consolidated": _as_bool(results.metadata.get("is_consolidated", False), default=False),
                "consolidated_scenario_count": _as_int(
                    results.metadata.get("consolidated_scenario_count", 0),
                    default=0,
                ),
                "scenario_results": scenario_results,
            }
            write_scenario_metadata(report_paths["metadata_json_path"], scenario_metadata)

            if _allure is not None:
                presentation_csv_path = Path(report_paths["presentation_csv_path"])
                if presentation_csv_path.exists():
                    _allure.attach(
                        presentation_csv_path.read_text(encoding="utf-8"),
                        name="Chat Q&A Core Performance CSV",
                        attachment_type=_allure.attachment_type.CSV,
                    )

                metadata_json_path = Path(report_paths["metadata_json_path"])
                if metadata_json_path.exists():
                    _allure.attach(
                        metadata_json_path.read_text(encoding="utf-8"),
                        name="Chat Q&A Core Scenario Metadata",
                        attachment_type=_allure.attachment_type.JSON,
                    )

                for plot_path in plot_files:
                    plot_file = Path(plot_path)
                    if plot_file.exists():
                        _allure.attach(
                            plot_file.read_bytes(),
                            name=f"Chat Q&A Core Plot - {plot_file.name}",
                            attachment_type=_allure.attachment_type.PNG,
                        )
        except Exception as report_error:
            logger.warning("Failed to update Chat Q&A Core performance report artifacts: %s", report_error)

    finally:
        _teardown_compose()  # no-op if already called by addfinalizer
        if results is None:
            results = Result(metadata={"status": False, "error": "Execution did not complete"})

        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
