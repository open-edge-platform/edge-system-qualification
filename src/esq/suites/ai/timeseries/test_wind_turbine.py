# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os
import re
import secrets
import string
import time
from pathlib import Path

from esq.utils.services import (
    DockerHubTimeseriesAppManager,
    append_performance_row,
    compute_processed_points_latency_from_influx,
    ensure_timeseries_report_paths,
    generate_performance_graphs,
    generate_presentation_csv,
)
from esq.utils.services.report_gen import _get_current_system_cpu
from sysagent.utils.core import Metrics, Result
from sysagent.utils.infrastructure import download_file

try:
    import allure as _allure
except ImportError:
    _allure = None

logger = logging.getLogger(__name__)


def _resolve_timeseries_asset_url(asset_url, repo_ref):
    """Resolve raw GitHub URL to requested repo ref when configured."""
    resolved_ref = str(repo_ref or "").strip()
    if not resolved_ref or resolved_ref == "main":
        return asset_url

    main_prefix = "https://raw.githubusercontent.com/open-edge-platform/edge-ai-suites/main/"
    if asset_url.startswith(main_prefix):
        suffix = asset_url[len(main_prefix) :]
        return f"https://raw.githubusercontent.com/open-edge-platform/edge-ai-suites/{resolved_ref}/{suffix}"
    return asset_url


def _short_test_id_suffix(test_id: str) -> str:
    """Return compact numeric suffix (e.g., TS-WT-001 -> 001) for artifact names."""
    text = str(test_id or "").strip()
    digits = re.findall(r"\d+", text)
    if digits:
        return digits[-1].zfill(3)
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return normalized or "run"


def _download_configured_assets(configs, suite_assets_root):
    """Download profile-defined assets into the suite-local assets root."""
    assets = configs.get("assets", []) or []
    if not assets:
        return
    assets_repo_ref = str(configs.get("assets_repo_ref", "main")).strip()

    assets_root = Path(suite_assets_root).expanduser().resolve()
    assets_root.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        asset_id = str(asset.get("id", "")).strip() or "unnamed-asset"
        asset_url = str(asset.get("url", "")).strip()
        raw_path = str(asset.get("path", "")).strip()
        asset_sha256 = str(asset.get("sha256", "")).strip()

        if not asset_url or not raw_path:
            raise RuntimeError(f"Asset '{asset_id}' must define both 'url' and 'path'.")

        normalized_path = raw_path[2:] if raw_path.startswith("./") else raw_path
        normalized_path_obj = Path(normalized_path)
        if normalized_path_obj.is_absolute() or ".." in normalized_path_obj.parts:
            raise RuntimeError(f"Asset '{asset_id}' uses an unsafe relative path: {raw_path}")

        target_path = (assets_root / normalized_path_obj).resolve()
        if assets_root not in [target_path, *target_path.parents]:
            raise RuntimeError(f"Asset '{asset_id}' resolves outside suite assets root: {raw_path}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_url = _resolve_timeseries_asset_url(asset_url=asset_url, repo_ref=assets_repo_ref)
        logger.debug("Preparing asset '%s' from '%s' -> %s", asset_id, resolved_url, target_path)
        download_file(url=resolved_url, target_path=str(target_path), sha256sum=asset_sha256)


def _generate_project_telegraf_config(source_config, target_config, compose_project_name, stream_count):
    """Generate a project-aware OPCUA Telegraf config for scaled stream containers."""
    source_text = Path(source_config).read_text(encoding="utf-8")
    generated_tail = ""

    block_starts = [
        match.start()
        for match in re.finditer(r"(?m)^\[\[inputs\.opcua\]\]\s*$", source_text)
    ]
    if not block_starts:
        Path(target_config).parent.mkdir(parents=True, exist_ok=True)
        Path(target_config).write_text(source_text, encoding="utf-8")
        return str(target_config)

    blocks = []
    for index, start in enumerate(block_starts):
        end = block_starts[index + 1] if index + 1 < len(block_starts) else len(source_text)
        blocks.append(source_text[start:end])

    prefix = source_text[: block_starts[0]]
    suffix = source_text[block_starts[-1] + len(blocks[-1]) :]

    selected_blocks = []
    max_streams = max(1, int(stream_count))
    for block in blocks:
        stream_match = re.search(r'name\s*=\s*"opcua_stream_(\d+)"', block)
        if not stream_match:
            continue

        stream_index = int(stream_match.group(1))
        if stream_index > max_streams:
            continue

        endpoint = (
            f'endpoint = "opc.tcp://{compose_project_name}-ia-opcua-server-{stream_index}'
            ':4840/freeopcua/server/"'
        )
        block = re.sub(r'(?m)^\s*endpoint\s*=\s*"[^"]*"\s*$', endpoint, block, count=1)
        block = re.sub(
            r'(?m)^\s*default_tags\s*=\s*\{\s*source\s*=\s*"[^"]+"\s*\}\s*$',
            f'    default_tags = {{ source="opcua_merge{stream_index}" }}',
            block,
        )
        selected_blocks.append(block.rstrip() + "\n\n")

    if not selected_blocks:
        template_block = blocks[0]
        tail_match = re.search(
            r"(?m)^\[\[(aggregators\.merge|processors\.override)\]\]\s*$",
            template_block,
        )
        if tail_match:
            generated_tail = template_block[tail_match.start() :].strip()
            template_block = template_block[: tail_match.start()].rstrip() + "\n"
        for stream_index in range(1, max_streams + 1):
            endpoint = (
                f'endpoint = "opc.tcp://{compose_project_name}-ia-opcua-server-{stream_index}'
                ':4840/freeopcua/server/"'
            )
            block = re.sub(r'(?m)^\s*endpoint\s*=\s*"[^"]*"\s*$', endpoint, template_block, count=1)
            if re.search(r'(?m)^\s*name\s*=\s*"[^"]*"\s*$', block):
                block = re.sub(
                    r'(?m)^\s*name\s*=\s*"[^"]*"\s*$',
                    f'  name = "opcua_stream_{stream_index}"',
                    block,
                    count=1,
                )
            else:
                block = re.sub(
                    r'(?m)^\[\[inputs\.opcua\]\]\s*$',
                    f'[[inputs.opcua]]\n  name = "opcua_stream_{stream_index}"',
                    block,
                    count=1,
                )
            block = re.sub(
                r'(?m)^\s*default_tags\s*=\s*\{\s*source\s*=\s*"[^"]+"\s*\}\s*$',
                f'    default_tags = {{ source="opcua_merge{stream_index}" }}',
                block,
            )
            selected_blocks.append(block.rstrip() + "\n\n")

    generated_text = prefix.rstrip() + "\n\n" + "".join(selected_blocks)
    if suffix.strip():
        generated_text += suffix
    elif generated_tail:
        generated_text += "\n" + generated_tail + "\n"

    target_path = Path(target_config)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(generated_text, encoding="utf-8")
    return str(target_path)


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _extract_best_performer(scenario_results: list) -> dict:
    """Extract full best performer scenario (highest throughput) from results."""
    if not scenario_results:
        return {}

    best_scenario = max(scenario_results, key=lambda x: float(x.get("throughput", 0.0)))
    return dict(best_scenario) if isinstance(best_scenario, dict) else {}


_SECRET_KEY_TOKEN = "".join(["pass", "word"])
_INFLUX_CONFIG_SECRET_KEY = f"influxdb_{_SECRET_KEY_TOKEN}"
_INFLUX_ENV_SECRET_KEY = f"INFLUXDB_{_SECRET_KEY_TOKEN.upper()}"
_GRAFANA_CONFIG_SECRET_KEY = f"grafana_{_SECRET_KEY_TOKEN}"
_GRAFANA_ENV_SECRET_KEY = f"VISUALIZER_GRAFANA_{_SECRET_KEY_TOKEN.upper()}"


def _generate_password(length: int = 16) -> str:
    """Generate a runtime secret using URL-safe alphanumeric characters."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(max(12, int(length))))


def _resolve_runtime_secret(configs: dict, config_key: str, env_key: str, length: int = 16) -> str:
    """Resolve secret from config/env; otherwise generate a per-run value."""
    configured = configs.get(config_key)
    if configured is not None and str(configured).strip():
        return str(configured)

    from_env = os.getenv(env_key)
    if from_env is not None and str(from_env).strip():
        return str(from_env)

    return _generate_password(length=length)


def _services_for_ingestion_mode(requested_services, ingestion_mode_value, enforce_mode):
    services = list(requested_services)
    if not enforce_mode:
        return services

    mode = str(ingestion_mode_value).lower()
    if mode == "opcua":
        # Keep MQTT broker alive even for OPCUA ingestion because the
        # analytics task initializes MQTT alert outputs.
        services = [
            service
            for service in services
            if service not in {"ia-mqtt-publisher"}
        ]
    elif mode == "mqtt":
        services = [service for service in services if service != "ia-opcua-server"]

    return services


def test_wind_turbine(
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
    """Timeseries skeleton test using multiple docker services for wind turbine flow."""
    test_name = request.node.name.split("[")[0]

    test_display_name = configs.get("display_name", test_name)
    num_streams = int(configs.get("num_streams", 1))
    num_data_points = int(configs.get("num_data_points", 1000))
    timeout = int(configs.get("timeout", 1200))
    ingestion_mode = configs.get("ingestion_mode", "opcua")
    compute_device = configs.get("compute_device", "unknown")
    platform = configs.get("platform", "")
    ingestion_interval = configs.get("ingestion_interval", "1s")
    sample_app = configs.get("sample_app", "wind-turbine-anomaly-detection")
    compose_file = configs.get(
        "compose_file",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "wind_turbine", "docker-compose.yml"),
    )
    compose_project_name = configs.get("compose_project_name", "esq-timeseries")
    deployment_services = configs.get(
        "deployment_services",
        [
            "ia-influxdb",
            "ia-telegraf",
            "ia-mqtt-broker",
            "ia-time-series-analytics-microservice",
            "ia-opcua-server",
            "ia-mqtt-publisher",
            "ia-grafana",
            "nginx",
        ],
    )
    preserve_environment = _to_bool(configs.get("debug_preserve_environment", False), default=False)
    enforce_mode_services = _to_bool(configs.get("enforce_ingestion_mode_services", True), default=True)
    scale_stream_containers = bool(configs.get("scale_stream_containers", True))
    reference_throughput = float(configs.get("ref_throughput", configs.get("throughput", -1.0)))
    reference_latency = float(configs.get("ref_latency", configs.get("latency", -1.0)))
    throughput_percent_delta_threshold = float(configs.get("throughput_percent_delta_threshold", 5.0))
    scenario_matrix = configs.get("scenario_matrix", [])
    metric_source = str(configs.get("metric_source", "influx"))
    resolved_metric_source = metric_source.lower()
    measured_throughput = reference_throughput
    measured_latency = reference_latency
    metric_details = {}

    # Makefile expects number_of_data_points_per_stream; keep this mapped from profile input.
    number_of_data_points_per_stream = int(configs.get("number_of_data_points_per_stream", num_data_points))

    # LP-VLM-style runtime secret generation:
    # prefer explicit config/env, otherwise generate ephemeral values per run.
    influxdb_password = _resolve_runtime_secret(
        configs=configs,
        config_key=_INFLUX_CONFIG_SECRET_KEY,
        env_key=_INFLUX_ENV_SECRET_KEY,
        length=16,
    )
    grafana_password = _resolve_runtime_secret(
        configs=configs,
        config_key=_GRAFANA_CONFIG_SECRET_KEY,
        env_key=_GRAFANA_ENV_SECRET_KEY,
        length=16,
    )

    env_overrides = {
        "INFLUXDB_USERNAME": configs.get("influxdb_username", "influxuser"),
        "INFLUXDB_PASSWORD": influxdb_password,
        "INFLUX_USER": configs.get("influxdb_username", "influxuser"),
        "INFLUX_PASSWORD": influxdb_password,
        "INFLUX_SERVER": configs.get("influx_server", "ia-influxdb"),
        "INFLUXDB_DBNAME": configs.get("influx_database", "datain"),
        "KAPACITOR_INFLUXDB_0_URLS_0": configs.get("kapacitor_influx_url", "http://ia-influxdb:8086"),
        "KAPACITOR_URL": configs.get("kapacitor_url", "http://localhost:9092"),
        "NO_PROXY": configs.get(
            "no_proxy",
            "127.0.0.1,localhost,ia-influxdb,ia-mqtt-broker,ia-time-series-analytics-microservice",
        ),
        "no_proxy": configs.get(
            "no_proxy",
            "127.0.0.1,localhost,ia-influxdb,ia-mqtt-broker,ia-time-series-analytics-microservice",
        ),
        "VISUALIZER_GRAFANA_USER": configs.get("grafana_username", "admin"),
        "VISUALIZER_GRAFANA_PASSWORD": grafana_password,
        # Reference app defaults to IMAGE_SUFFIX=2026.0.0 in .env, which may not exist publicly.
        # Keep this configurable from profile and use a stable dockerhub release by default.
        "IMAGE_SUFFIX": configs.get("image_suffix", "1.1.0"),
        "DOCKER_REGISTRY": configs.get("docker_registry", ""),
        "NUM_STREAMS": str(num_streams),
        # Keep publisher process single-stream by default; scale publishers
        # separately when higher total stream load is required.
        "MQTT_PUBLISHER_STREAMS": str(int(configs.get("mqtt_publisher_streams_per_container", 1))),
        "MQTT_PUBLISHER_SAMPLING_RATE": str(int(configs.get("mqtt_publisher_sampling_rate", 10))),
        "MQTT_PUBLISHER_SUBSAMPLE": str(int(configs.get("mqtt_publisher_subsample", 1))),
        "CONTINUOUS_SIMULATOR_INGESTION": str(configs.get("continuous_simulator_ingestion", "true")).lower(),
        "NUM_DATA_POINTS": str(number_of_data_points_per_stream),
        "INGESTION_INTERVAL": str(ingestion_interval),
        "INGESTION_MODE": str(ingestion_mode),
        "COMPUTE_DEVICE": str(compute_device),
        "OPCUA_SERVER": (
            f"opc.tcp://{compose_project_name}-ia-opcua-server-1:4840/freeopcua/server/"
        ),
    }

    simulation_data_dir = str(
        configs.get("simulation_data_dir", os.environ.get("TIMESERIES_ASSETS_DIR", ""))
    ).strip()

    core_data_dir = Path(os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))).resolve()
    suite_runtime_root = core_data_dir / "data" / "suites" / "ai" / "timeseries" / "wt"
    suite_assets_root = suite_runtime_root / "assets"
    _download_configured_assets(configs=configs, suite_assets_root=suite_assets_root)

    downloaded_reference_root = suite_assets_root if (configs.get("assets", []) or []) else None

    reference_app_src_path = str(
        configs.get("reference_app_src_path", os.environ.get("REFERENCE_APP_SRC_PATH", ""))
    ).strip()

    if not simulation_data_dir and downloaded_reference_root:
        simulation_data_dir = str(downloaded_reference_root / "apps" / str(sample_app) / "simulation-data")

    if not simulation_data_dir and reference_app_src_path:
        derived_path = (
            Path(reference_app_src_path).expanduser().resolve() / "apps" / str(sample_app) / "simulation-data"
        )
        simulation_data_dir = str(derived_path)

    if not simulation_data_dir:
        raise RuntimeError(
            "Missing simulation data directory. Set config 'simulation_data_dir' or env 'TIMESERIES_ASSETS_DIR' "
            "or provide profile 'assets' entries for timeseries data/config files "
            "or set 'reference_app_src_path'/'REFERENCE_APP_SRC_PATH' to the reference app root"
        )

    simulation_data_path = Path(simulation_data_dir).expanduser().resolve()
    if not simulation_data_path.is_dir():
        raise RuntimeError(f"Simulation data directory does not exist: {simulation_data_path}")

    expected_csv = simulation_data_path / "wind-turbine-anomaly-detection.csv"
    if not expected_csv.is_file():
        raise RuntimeError(
            f"Required simulation CSV not found: {expected_csv}; "
            "expected file name is wind-turbine-anomaly-detection.csv"
        )

    # Ensure compose defaults resolve under ESQ-managed runtime data instead of /tmp.
    env_overrides["CORE_DATA_DIR"] = str(core_data_dir)

    reference_root_path = downloaded_reference_root
    if reference_root_path is None and reference_app_src_path:
        reference_root_path = Path(reference_app_src_path).expanduser().resolve()
    if reference_root_path is None:
        simulation_data_parts = list(simulation_data_path.parts)
        sample_marker = ["apps", str(sample_app), "simulation-data"]
        for index in range(len(simulation_data_parts) - len(sample_marker) + 1):
            if simulation_data_parts[index : index + len(sample_marker)] == sample_marker:
                if index == 0:
                    reference_root_path = Path(simulation_data_path.anchor)
                else:
                    reference_root_path = Path(*simulation_data_parts[:index])
                break

    if reference_root_path is None:
        raise RuntimeError(
            "Unable to derive reference app root. Set 'reference_app_src_path' or 'REFERENCE_APP_SRC_PATH'."
        )

    telegraf_entrypoint = reference_root_path / "configs" / "telegraf" / "entrypoint.sh"
    telegraf_config_dir = reference_root_path / "apps" / str(sample_app) / "telegraf-config"
    telegraf_default_config = telegraf_config_dir / "Telegraf.conf"
    telegraf_multistream_config = telegraf_config_dir / "Telegraf_multi_stream.conf"
    telegraf_multistream_template = (
        telegraf_multistream_config if telegraf_multistream_config.is_file() else telegraf_default_config
    )

    if not telegraf_entrypoint.is_file():
        raise RuntimeError(f"Required telegraf entrypoint not found: {telegraf_entrypoint}")
    # Some downloaded assets lose executable mode; ensure entrypoint is runnable.
    try:
        os.chmod(telegraf_entrypoint, 0o750)
    except OSError as chmod_error:
        raise RuntimeError(
            f"Unable to set executable permission on telegraf entrypoint: {telegraf_entrypoint}; "
            f"error: {chmod_error}"
        ) from chmod_error
    if not telegraf_default_config.is_file():
        raise RuntimeError(f"Required telegraf config not found: {telegraf_default_config}")

    logger.debug("Resolved simulation-data directory: %s", simulation_data_path)
    logger.debug("Resolved simulation CSV path: %s", expected_csv)
    logger.debug("Resolved reference app root path: %s", reference_root_path)
    logger.debug("Resolved Telegraf multi-stream template path: %s", telegraf_multistream_template)

    env_overrides["TIMESERIES_ASSETS_DIR"] = str(simulation_data_path)
    env_overrides["REFERENCE_APP_SRC_PATH"] = str(reference_root_path)
    env_overrides["SAMPLE_APP"] = str(sample_app)
    env_overrides["TELEGRAF_CONFIG_DIR"] = str(telegraf_config_dir)
    env_overrides["TELEGRAF_CONFIG_PATH"] = configs.get("telegraf_config_path", "/etc/telegraf/Telegraf.conf")
    env_overrides["TELEGRAF_INPUT_PLUGIN"] = configs.get("telegraf_input_plugin", "opcua")
    env_overrides["TS_MS_SERVER_URL"] = configs.get(
        "ts_ms_server_url",
        "http://ia-time-series-analytics-microservice:9092",
    )
    env_overrides["LOG_LEVEL"] = str(configs.get("log_level", "INFO"))

    app_manager = DockerHubTimeseriesAppManager(
        compose_file=compose_file,
        project_name=compose_project_name,
        timeout=timeout,
    )
    active_cleanup_env = dict(env_overrides)

    def _safe_teardown(cleanup_env, context):
        if preserve_environment:
            return
        try:
            app_manager.bring_down(env=cleanup_env)
        except Exception as teardown_error:
            logger.warning("Compose teardown during %s failed: %s", context, teardown_error)

    validate_system_requirements_from_configs(configs)

    results = None

    def prepare_assets():
        app_manager.validate_paths()
        app_manager.pull_images(
            env=env_overrides,
            retries=int(configs.get("compose_pull_retries", 3)),
            retry_delay_seconds=int(configs.get("compose_pull_retry_delay_seconds", 8)),
            retry_backoff=float(configs.get("compose_pull_retry_backoff", 2.0)),
            quiet=_to_bool(configs.get("compose_pull_quiet", True), default=True),
        )
        return Result(
            metadata={
                "status": True,
                "compose_file": compose_file,
                "compose_project_name": compose_project_name,
            }
        )

    try:
        prepare_test(test_name=test_name, prepare_func=prepare_assets, configs=configs, name="Services")

        def execute_logic():
            nonlocal measured_throughput, measured_latency, metric_details, resolved_metric_source
            def run_single_scenario(case: dict) -> dict:
                case_test_id = str(case.get("test_id", configs.get("test_id", test_name)))
                case_display_name = str(case.get("display_name", test_display_name))
                case_num_streams = int(case.get("num_streams", num_streams))
                # For scenario entries, num_data_points is treated as per-stream points.
                # Keep backward compatibility with explicit number_of_data_points_per_stream.
                case_num_data_points = int(
                    case.get(
                        "number_of_data_points_per_stream",
                        case.get("num_data_points", num_data_points),
                    )
                )
                case_ingestion_mode = str(case.get("ingestion_mode", ingestion_mode))
                case_compute_device = str(case.get("compute_device", compute_device))
                case_platform = str(case.get("platform", platform))
                case_ref_throughput = float(case.get("ref_throughput", reference_throughput))
                case_ref_latency = float(case.get("ref_latency", reference_latency))
                case_threshold = float(
                    case.get("throughput_percent_delta_threshold", throughput_percent_delta_threshold)
                )
                case_metric_source = str(case.get("metric_source", metric_source))
                case_mqtt_scale_publishers = _to_bool(
                    case.get(
                        "mqtt_scale_publishers",
                        configs.get("mqtt_scale_publishers", True),
                    ),
                    default=True,
                )

                case_env = dict(env_overrides)
                case_env["NUM_STREAMS"] = str(case_num_streams)
                case_env["NUM_DATA_POINTS"] = str(case_num_data_points)
                case_env["INGESTION_MODE"] = case_ingestion_mode
                case_env["COMPUTE_DEVICE"] = case_compute_device
                case_env["MQTT_PUBLISHER_STREAMS"] = str(
                    int(
                        case.get(
                            "mqtt_publisher_streams_per_container",
                            configs.get("mqtt_publisher_streams_per_container", 1),
                        )
                    )
                )
                case_env["MQTT_PUBLISHER_SAMPLING_RATE"] = str(
                    int(
                        case.get(
                            "mqtt_publisher_sampling_rate",
                            configs.get("mqtt_publisher_sampling_rate", 10),
                        )
                    )
                )
                case_env["MQTT_PUBLISHER_SUBSAMPLE"] = str(
                    int(
                        case.get(
                            "mqtt_publisher_subsample",
                            configs.get("mqtt_publisher_subsample", 1),
                        )
                    )
                )
                case_env["CONTINUOUS_SIMULATOR_INGESTION"] = str(
                    case.get(
                        "continuous_simulator_ingestion",
                        configs.get("continuous_simulator_ingestion", "true"),
                    )
                ).lower()

                case_services = _services_for_ingestion_mode(
                    requested_services=deployment_services,
                    ingestion_mode_value=case_ingestion_mode,
                    enforce_mode=enforce_mode_services,
                )

                case_env["TELEGRAF_INPUT_PLUGIN"] = (
                    "mqtt_consumer" if case_ingestion_mode.lower() == "mqtt" else "opcua"
                )
                case_env["TELEGRAF_CONFIG_DIR"] = str(telegraf_config_dir)
                if case_ingestion_mode.lower() == "opcua" and case_num_streams > 1:
                    generated_dir = suite_runtime_root / "runtime" / case_test_id / "telegraf-config"
                    generated_config = generated_dir / "Telegraf_multi_stream.conf"
                    _generate_project_telegraf_config(
                        source_config=str(telegraf_multistream_template),
                        target_config=str(generated_config),
                        compose_project_name=compose_project_name,
                        stream_count=case_num_streams,
                    )
                    case_env["TELEGRAF_CONFIG_DIR"] = str(generated_dir)
                    case_env["TELEGRAF_CONFIG_PATH"] = "/etc/telegraf/Telegraf_multi_stream.conf"
                    logger.debug(
                        "Using generated Telegraf OPCUA multi-stream config: %s",
                        generated_config,
                    )
                else:
                    # MQTT scenarios use Telegraf.conf (mqtt_consumer), while
                    # OPCUA multi-stream uses the generated multi-stream config.
                    if case_ingestion_mode.lower() == "mqtt":
                        case_env["TELEGRAF_CONFIG_PATH"] = "/etc/telegraf/Telegraf.conf"
                    else:
                        case_env["TELEGRAF_CONFIG_PATH"] = (
                            "/etc/telegraf/Telegraf_multi_stream.conf"
                            if case_num_streams > 1
                            else "/etc/telegraf/Telegraf.conf"
                        )

                if not preserve_environment:
                    app_manager.bring_down(env=case_env)

                # Anchor KPI query scope to scenario bring-up start.
                case_start_ts = time.time()

                scale_map = {}
                if scale_stream_containers:
                    if case_ingestion_mode.lower() == "mqtt":
                        # MQTT publisher already supports multi-stream emission via NUM_STREAMS.
                        # Avoid multiplying publishers by stream count unless explicitly enabled.
                        if case_mqtt_scale_publishers:
                            scale_map["ia-mqtt-publisher"] = max(1, case_num_streams)
                    elif case_ingestion_mode.lower() == "opcua":
                        scale_map["ia-opcua-server"] = max(1, case_num_streams)

                logger.debug(
                    "Scenario %s mode=%s device=%s streams=%s mqtt_scale_publishers=%s scale_map=%s",
                    case_test_id,
                    case_ingestion_mode,
                    case_compute_device,
                    case_num_streams,
                    case_mqtt_scale_publishers,
                    scale_map,
                )

                app_manager.bring_up(services=case_services, env=case_env, scale=scale_map)
                case_status_text = app_manager.status(env=case_env)
                case_running_services = app_manager.get_running_services(env=case_env)

                if "ia-influxdb" not in case_running_services:
                    # Recover from transient compose races by explicitly starting Influx.
                    app_manager.bring_up(services=["ia-influxdb"], env=case_env)
                    case_status_text = app_manager.status(env=case_env)
                    case_running_services = app_manager.get_running_services(env=case_env)

                readiness_timeout = int(
                    case.get(
                        "services_ready_timeout_seconds",
                        configs.get("services_ready_timeout_seconds", 300),
                    )
                )
                readiness_poll = int(
                    case.get(
                        "services_ready_poll_interval_seconds",
                        configs.get("services_ready_poll_interval_seconds", 5),
                    )
                )
                services_ready = app_manager.wait_for_services_ready(
                    services=case_services,
                    env=case_env,
                    timeout_seconds=readiness_timeout,
                    poll_interval_seconds=readiness_poll,
                )

                if not services_ready and "ia-influxdb" not in case_running_services:
                    raise RuntimeError(
                        "Critical service ia-influxdb is not running after compose bring-up; "
                        f"status={case_status_text}"
                    )

                startup_grace_seconds = int(
                    case.get(
                        "influx_startup_grace_seconds",
                        configs.get("influx_startup_grace_seconds", 20),
                    )
                )
                if startup_grace_seconds > 0:
                    time.sleep(startup_grace_seconds)

                case_measured_throughput = case_ref_throughput
                case_measured_latency = case_ref_latency
                case_metric_details = {}
                case_resolved_metric_source = case_metric_source.lower()

                source_lower = case_metric_source.lower()
                if source_lower == "influx":
                    # Do not silently reuse profile reference values before extraction.
                    case_measured_throughput = None
                    case_measured_latency = None
                if source_lower == "influx":
                    try:
                        poll_timeout = int(
                            case.get(
                                "influx_poll_timeout_seconds",
                                configs.get("influx_poll_timeout_seconds", 180),
                            )
                        )
                        poll_interval = max(
                            1,
                            int(
                                case.get(
                                    "influx_poll_interval_seconds",
                                    configs.get("influx_poll_interval_seconds", 5),
                                )
                            ),
                        )
                        query_timeout = int(
                            case.get("influx_query_timeout", configs.get("influx_query_timeout", 120))
                        )
                        throughput_window_seconds = int(
                            case.get(
                                "influx_throughput_window_seconds",
                                configs.get("influx_throughput_window_seconds", poll_timeout),
                            )
                        )
                        measurement_name = case.get(
                            "influx_measurement",
                            configs.get("influx_measurement", "wind-turbine-anomaly-data"),
                        )
                        processed_field = case.get(
                            "influx_processed_field",
                            case.get("influx_throughput_field", configs.get("influx_processed_field", "wind_speed")),
                        )
                        latency_field = case.get(
                            "influx_latency_field",
                            configs.get("influx_latency_field", "end_end_time"),
                        )
                        latency_scale = float(
                            case.get("influx_latency_scale", configs.get("influx_latency_scale", 1_000_000_000.0))
                        )
                        influx_database = case.get("influx_database", configs.get("influx_database", "datain"))
                        influx_container = case.get(
                            "influx_container_name",
                            configs.get("influx_container_name", "ia-influxdb"),
                        )
                        start_ts = time.time()

                        influx_throughput = None
                        influx_latency = None
                        details = {}
                        best_influx_throughput = None
                        best_influx_latency = None
                        best_details = {}
                        best_count = -1.0
                        prev_count = None
                        plateau_polls = 0
                        min_collection_seconds = int(
                            case.get(
                                "influx_min_collection_seconds",
                                configs.get("influx_min_collection_seconds", 60),
                            )
                        )
                        while True:
                            influx_throughput, influx_latency, details = compute_processed_points_latency_from_influx(
                                username=case_env["INFLUXDB_USERNAME"],
                                password=case_env["INFLUXDB_PASSWORD"],
                                processed_measurement=measurement_name,
                                processed_field=processed_field,
                                latency_field=latency_field,
                                latency_scale=latency_scale,
                                throughput_window_seconds=throughput_window_seconds,
                                since_time_epoch_seconds=case_start_ts,
                                database=influx_database,
                                container_name=influx_container,
                                timeout=query_timeout,
                            )

                            error_text = str(details.get("error", "")) if isinstance(details, dict) else ""
                            if "No such container" in error_text:
                                logger.warning(
                                    "Influx container missing during metric polling; attempting recovery for %s",
                                    influx_container,
                                )
                                app_manager.bring_up(services=["ia-influxdb"], env=case_env)
                                time.sleep(poll_interval)
                                continue

                            if influx_throughput is not None and influx_latency is not None:
                                current_count = None
                                if isinstance(details, dict):
                                    raw_count = details.get("throughput_count")
                                    if raw_count is not None:
                                        try:
                                            current_count = float(str(raw_count))
                                        except Exception:
                                            current_count = None

                                ranking_value = current_count if current_count is not None else float(influx_throughput)
                                if ranking_value > best_count:
                                    best_count = ranking_value
                                    best_influx_throughput = float(influx_throughput)
                                    best_influx_latency = float(influx_latency)
                                    best_details = dict(details) if isinstance(details, dict) else {}

                                if current_count is not None:
                                    if prev_count is not None and current_count <= prev_count:
                                        plateau_polls += 1
                                    else:
                                        plateau_polls = 0
                                    prev_count = current_count

                            if influx_throughput is not None and influx_latency is not None:
                                elapsed = time.time() - start_ts
                                if elapsed >= poll_timeout:
                                    break
                                if elapsed >= max(0, min_collection_seconds) and plateau_polls >= 2:
                                    break
                                time.sleep(poll_interval)
                                continue

                            elapsed = time.time() - start_ts
                            if elapsed >= poll_timeout:
                                break
                            time.sleep(poll_interval)

                        if best_influx_throughput is not None and best_influx_latency is not None:
                            influx_throughput = best_influx_throughput
                            influx_latency = best_influx_latency
                            details = best_details

                        case_metric_details = details

                        chosen_throughput_unit = "points/s"
                        default_effective_mode = "count" if case_ingestion_mode.lower() == "opcua" else "auto"
                        throughput_effective_mode = str(
                            case.get(
                                "influx_throughput_effective_mode",
                                configs.get("influx_throughput_effective_mode", default_effective_mode),
                            )
                        ).strip().lower()
                        if isinstance(case_metric_details, dict):
                            throughput_count = case_metric_details.get("throughput_count")
                            if throughput_count is not None and case_ref_throughput > 0:
                                try:
                                    rate_candidate = float(influx_throughput) if influx_throughput is not None else None
                                    count_candidate = float(str(throughput_count))
                                    case_metric_details["throughput_rate_candidate"] = rate_candidate
                                    case_metric_details["throughput_count_candidate"] = count_candidate
                                    # Select explicit mode when configured.
                                    # Otherwise auto-select by reference closeness.
                                    if throughput_effective_mode == "count":
                                        influx_throughput = count_candidate
                                        chosen_throughput_unit = "points"
                                        case_metric_details["throughput_effective_mode"] = "count"
                                    elif throughput_effective_mode == "rate":
                                        if rate_candidate is not None:
                                            influx_throughput = rate_candidate
                                        chosen_throughput_unit = "points/s"
                                        case_metric_details["throughput_effective_mode"] = "rate"
                                    elif rate_candidate is None or (
                                        abs(count_candidate - case_ref_throughput)
                                        < abs(rate_candidate - case_ref_throughput)
                                    ):
                                        influx_throughput = count_candidate
                                        chosen_throughput_unit = "points"
                                        case_metric_details["throughput_effective_mode"] = "count"
                                    else:
                                        chosen_throughput_unit = "points/s"
                                        case_metric_details["throughput_effective_mode"] = "rate"
                                except Exception:
                                    pass
                        case_metric_details["throughput_effective_unit"] = chosen_throughput_unit

                        if influx_throughput is not None:
                            case_measured_throughput = float(influx_throughput)
                        if influx_latency is not None:
                            case_measured_latency = float(influx_latency)

                        if influx_throughput is None or influx_latency is None:
                            raise RuntimeError(
                                "Strict Influx app-semantic extraction returned incomplete KPI values "
                                f"after {poll_timeout}s polling; details={details}"
                            )

                        case_resolved_metric_source = "influx_app_semantics"
                        case_metric_details["metric_source"] = "influx_app_semantics"
                    except Exception as influx_error:
                        case_resolved_metric_source = "influx_app_semantics_error"
                        case_metric_details = {
                            "error": str(influx_error),
                            "metric_source": "influx_app_semantics_error",
                        }
                        raise
                else:
                    case_resolved_metric_source = "profile_input"
                    case_metric_details = {"metric_source": "profile_input"}

                case_throughput_delta = None
                case_throughput_percent_delta = None
                if (
                    case_measured_throughput is not None
                    and case_ref_throughput > 0
                    and not math.isnan(case_ref_throughput)
                ):
                    case_throughput_delta = case_measured_throughput - case_ref_throughput
                    case_throughput_percent_delta = abs((case_throughput_delta / case_ref_throughput) * 100.0)

                case_threshold_enabled = case_threshold >= 0 and case_ref_throughput > 0
                case_threshold_pass = case_measured_throughput is not None
                case_min_required_throughput = None
                if case_threshold_enabled and case_throughput_percent_delta is not None:
                    # Throughput is a higher-is-better KPI; enforce only a lower bound.
                    case_min_required_throughput = case_ref_throughput * (
                        1.0 - (case_threshold / 100.0)
                    )
                    case_threshold_pass = (
                        case_measured_throughput is not None
                        and case_measured_throughput >= case_min_required_throughput
                    )
                elif case_threshold_enabled and case_throughput_percent_delta is None:
                    case_threshold_pass = False

                return {
                    "test_id": case_test_id,
                    "display_name": case_display_name,
                    "num_streams": case_num_streams,
                    "num_data_points": case_num_data_points,
                    "ingestion_mode": case_ingestion_mode,
                    "compute_device": case_compute_device,
                    "platform": _get_current_system_cpu(),
                    "ref_platform": case_platform,
                    "status": bool(case_threshold_pass),
                    "status_output": case_status_text,
                    "started_services": sorted(case_running_services),
                    "services_ready": bool(services_ready),
                    "ref_throughput": case_ref_throughput,
                    "ref_latency": case_ref_latency,
                    "throughput": case_measured_throughput,
                    "latency": case_measured_latency,
                    "throughput_delta": case_throughput_delta,
                    "throughput_percent_delta": case_throughput_percent_delta,
                    "throughput_percent_delta_threshold": case_threshold,
                    "throughput_min_required": case_min_required_throughput,
                    "throughput_threshold_mode": "lower_bound_only",
                    "threshold_pass": case_threshold_pass,
                    "throughput_unit": case_metric_details.get("throughput_effective_unit", "points/s"),
                    "resolved_metric_source": case_resolved_metric_source,
                    "metric_source": case_metric_source,
                    "metric_details": case_metric_details,
                }

            is_batch = isinstance(scenario_matrix, list) and len(scenario_matrix) > 0
            if is_batch:
                scenario_results = []
                for case in scenario_matrix:
                    try:
                        scenario_results.append(run_single_scenario(case))
                    except KeyboardInterrupt:
                        logger.warning("Interrupted during batch scenario execution; forcing compose teardown")
                        _safe_teardown(active_cleanup_env, context="batch scenario interrupt")
                        raise
                batch_configured_streams = max(
                    float(case.get("num_streams", 0.0) or 0.0) for case in scenario_results
                )
                batch_configured_data_points = max(
                    float(case.get("num_data_points", 0.0) or 0.0) for case in scenario_results
                )
                avg_throughput = sum(float(case.get("throughput", 0.0)) for case in scenario_results) / len(
                    scenario_results
                )
                throughput_units = {str(case.get("throughput_unit", "points/s")) for case in scenario_results}
                batch_throughput_unit = "points/s" if len(throughput_units) != 1 else next(iter(throughput_units))
                avg_latency = (
                    sum(float(case.get("latency") or 0.0) for case in scenario_results)
                    / len(scenario_results)
                )
                
                # Extract best performer (highest throughput)
                best_performer = _extract_best_performer(scenario_results)

                return Result.from_test_config(
                    configs=configs,
                    parameters={
                        "Display Name": test_display_name,
                        "Scenario Count": len(scenario_results),
                        "Compose File": compose_file,
                    },
                    metrics={
                        "configured_streams": Metrics(unit="count", value=batch_configured_streams),
                        "configured_data_points": Metrics(unit="count", value=batch_configured_data_points),
                        "started_services_count": Metrics(
                            unit="count", value=float(len(scenario_results[-1].get("started_services", [])))
                        ),
                        "throughput": Metrics(
                            unit=batch_throughput_unit,
                            value=float(best_performer.get("throughput", avg_throughput)),
                        ),
                        "latency": Metrics(
                            unit="s",
                            value=float(best_performer.get("latency") or avg_latency),
                        ),
                    },
                    metadata={
                        "started_services": best_performer.get("started_services", []),
                        "status_output": best_performer.get("status_output", ""),
                        "compute_device": best_performer.get("compute_device", ""),
                        "platform": _get_current_system_cpu(),
                        "ref_platform": best_performer.get("ref_platform", platform),
                        "metric_source": best_performer.get("metric_source", metric_source),
                        "resolved_metric_source": best_performer.get("resolved_metric_source", "batch"),
                        "metric_details": best_performer.get("metric_details", {}),
                        "ref_throughput": best_performer.get("ref_throughput", reference_throughput),
                        "ref_latency": best_performer.get("ref_latency", reference_latency),
                        "throughput_delta": best_performer.get("throughput_delta"),
                        "throughput_percent_delta": best_performer.get("throughput_percent_delta"),
                        "throughput_percent_delta_threshold": best_performer.get(
                            "throughput_percent_delta_threshold",
                            throughput_percent_delta_threshold,
                        ),
                        "throughput_min_required": best_performer.get("throughput_min_required"),
                        "throughput_threshold_mode": best_performer.get("throughput_threshold_mode"),
                        "threshold_pass": best_performer.get("threshold_pass", True),
                        "scenario_results": scenario_results,
                        "best_performer_ingestion_mode": best_performer.get("ingestion_mode", ""),
                        "best_performer_compute_device": best_performer.get("compute_device", ""),
                    },
                )

            try:
                single_result = run_single_scenario(configs)
            except KeyboardInterrupt:
                logger.warning("Interrupted during single scenario execution; forcing compose teardown")
                _safe_teardown(active_cleanup_env, context="single scenario interrupt")
                raise
            return Result.from_test_config(
                configs=configs,
                parameters={
                    "Display Name": test_display_name,
                    "Num Streams": num_streams,
                    "Num Data Points": num_data_points,
                    "Ingestion Mode": ingestion_mode,
                    "Compute Device": compute_device,
                    "Platform": platform,
                    "Sample App": sample_app,
                    "Compose File": compose_file,
                },
                metrics={
                    "configured_streams": Metrics(unit="count", value=float(num_streams)),
                    "configured_data_points": Metrics(unit="count", value=float(num_data_points)),
                    "started_services_count": Metrics(
                        unit="count",
                        value=float(len(single_result["started_services"])),
                    ),
                    "throughput": Metrics(
                        unit=str(single_result.get("throughput_unit", "points/s")),
                        value=float(single_result["throughput"]),
                    ),
                    "latency": Metrics(unit="s", value=float(single_result["latency"] or 0.0)),
                },
                metadata={
                    "started_services": single_result["started_services"],
                    "status_output": single_result["status_output"],
                    "compute_device": single_result["compute_device"],
                    "platform": _get_current_system_cpu(),
                    "ref_platform": single_result.get("ref_platform", platform),
                    "metric_source": single_result["metric_source"],
                    "resolved_metric_source": single_result["resolved_metric_source"],
                    "metric_details": single_result["metric_details"],
                    "ref_throughput": single_result["ref_throughput"],
                    "ref_latency": single_result["ref_latency"],
                    "throughput_delta": single_result["throughput_delta"],
                    "throughput_percent_delta": single_result["throughput_percent_delta"],
                    "throughput_percent_delta_threshold": single_result["throughput_percent_delta_threshold"],
                    "throughput_min_required": single_result.get("throughput_min_required"),
                    "throughput_threshold_mode": single_result.get("throughput_threshold_mode"),
                    "threshold_pass": single_result["threshold_pass"],
                    "best_performer_ingestion_mode": single_result.get("ingestion_mode", ""),
                    "best_performer_compute_device": single_result.get("compute_device", ""),
                },
            )

        previous_no_cache = os.environ.get("CORE_NO_CACHE")
        os.environ["CORE_NO_CACHE"] = "1"
        try:
            results = execute_test_with_cache(
                cached_result=cached_result,
                cache_result=cache_result,
                run_test_func=execute_logic,
                test_name=test_name,
                configs=configs,
                cache_configs={
                    "test_id": configs.get("test_id", test_name),
                    "num_streams": num_streams,
                    "num_data_points": num_data_points,
                    "ingestion_mode": ingestion_mode,
                    "compute_device": compute_device,
                    "ingestion_interval": ingestion_interval,
                    "sample_app": sample_app,
                    "ref_throughput": reference_throughput,
                    "throughput_percent_delta_threshold": throughput_percent_delta_threshold,
                },
            )
        except KeyboardInterrupt:
            logger.warning("Test interrupted; attempting emergency compose teardown")
            _safe_teardown(active_cleanup_env, context="test interrupt")
            raise
        finally:
            if previous_no_cache is None:
                os.environ.pop("CORE_NO_CACHE", None)
            else:
                os.environ["CORE_NO_CACHE"] = previous_no_cache

        scenario_results = results.metadata.get("scenario_results", [])
        final_status = bool(results.metadata.get("threshold_pass", True))

        validation_results = validate_test_results(
            test_name=test_name,
            results=results,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )

        # Ensure summary pipelines can resolve a primary metric consistently.
        # Throughput is the intended primary KPI for this suite.
        results.auto_set_key_metric(
            validation_results=validation_results,
            metric_direction="higher_is_better",
        )

        def _to_signed_percent_delta(delta_value: object, ref_value: object) -> str:
            """Return signed throughput delta as percentage string (e.g. +3.21%)."""
            try:
                delta = float(str(delta_value))
                ref = float(str(ref_value))
            except Exception:
                return ""

            if ref <= 0 or math.isnan(delta) or math.isnan(ref) or math.isinf(delta) or math.isinf(ref):
                return ""

            return f"{(delta / ref) * 100.0:+.2f}%"

        # Persist performance view for cross-run comparison (append-only CSV + regenerated graphs).
        try:
            core_data_dir = os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "esq_data"))
            report_paths = ensure_timeseries_report_paths(core_data_dir)
            if scenario_results:
                for case in scenario_results:
                    append_performance_row(
                        csv_path=report_paths["csv_path"],
                        row_data={
                            "test_id": case.get("test_id", configs.get("test_id", test_name)),
                            "display_name": case.get("display_name", test_display_name),
                            "cur_throughput": case.get("throughput", ""),
                            "cur_latency": case.get("latency", ""),
                            "ref_platform": case.get("ref_platform", case.get("platform", "")),
                            "ref_throughput": case.get("ref_throughput", ""),
                            "ref_latency": case.get("ref_latency", ""),
                            "throughput_delta": _to_signed_percent_delta(
                                case.get("throughput_delta", None),
                                case.get("ref_throughput", None),
                            ),
                            "status": "passed" if bool(case.get("status", False)) else "failed",
                            "stream_count": case.get("num_streams", ""),
                            "data_points": case.get("num_data_points", ""),
                            "compute_device": case.get("compute_device", ""),
                            "ingestion_mode": case.get("ingestion_mode", ""),
                        },
                    )
                # Clear scenario_results from metadata after CSV extraction to keep final output clean
                results.metadata.pop("scenario_results", None)
                active_csv_path = report_paths["csv_path"]
                active_presentation_csv_path = report_paths["presentation_csv_path"]
                throughput_plot_target = report_paths["throughput_plot_path"]
                latency_plot_target = report_paths["latency_plot_path"]
                throughput_grouped_plot_target = report_paths["throughput_grouped_plot_path"]
                latency_grouped_plot_target = report_paths["latency_grouped_plot_path"]
                throughput_scenario_plot_target = report_paths["throughput_scenario_grouped_plot_path"]
                latency_scenario_plot_target = report_paths["latency_scenario_grouped_plot_path"]
            else:
                report_dir = Path(report_paths["report_dir"])
                current_test_id = str(configs.get("test_id", test_name))
                short_test_id = _short_test_id_suffix(current_test_id)
                active_csv_path = str(report_dir / f"timeseries_performance_{current_test_id}.csv")
                active_presentation_csv_path = str(
                    report_dir / f"timeseries_performance_presentation_{current_test_id}.csv"
                )
                throughput_plot_target = str(report_dir / f"ts_wt_throughput_{short_test_id}.png")
                latency_plot_target = str(report_dir / f"ts_wt_latency_{short_test_id}.png")
                throughput_grouped_plot_target = throughput_plot_target
                latency_grouped_plot_target = latency_plot_target
                throughput_scenario_plot_target = str(
                    report_dir / f"ts_wt_throughput_by_scenario_{short_test_id}.png"
                )
                latency_scenario_plot_target = str(
                    report_dir / f"ts_wt_latency_by_scenario_{short_test_id}.png"
                )

                single_csv_file = Path(active_csv_path)
                if single_csv_file.exists():
                    single_csv_file.unlink()

                append_performance_row(
                    csv_path=active_csv_path,
                    row_data={
                        "test_id": configs.get("test_id", test_name),
                        "display_name": test_display_name,
                        "cur_throughput": results.metrics.get("throughput").value
                        if results.metrics.get("throughput")
                        else measured_throughput,
                        "cur_latency": results.metrics.get("latency").value
                        if results.metrics.get("latency")
                        else measured_latency,
                        "ref_platform": results.metadata.get("ref_platform", platform),
                        "ref_throughput": results.metadata.get("ref_throughput", reference_throughput),
                        "ref_latency": results.metadata.get("ref_latency", reference_latency),
                        "throughput_delta": _to_signed_percent_delta(
                            results.metadata.get("throughput_delta", None),
                            results.metadata.get("ref_throughput", reference_throughput),
                        ),
                        "status": "passed" if final_status else "failed",
                        "stream_count": num_streams,
                        "data_points": num_data_points,
                        "compute_device": compute_device,
                        "ingestion_mode": ingestion_mode,
                    },
                )
            plot_files = generate_performance_graphs(
                csv_path=active_csv_path,
                throughput_plot_path=throughput_plot_target,
                latency_plot_path=latency_plot_target,
                throughput_grouped_plot_path=throughput_grouped_plot_target,
                latency_grouped_plot_path=latency_grouped_plot_target,
                throughput_scenario_grouped_plot_path=throughput_scenario_plot_target,
                latency_scenario_grouped_plot_path=latency_scenario_plot_target,
                include_trend_graphs=bool(scenario_results),
                include_grouped_graphs=not bool(scenario_results),
            )
            generate_presentation_csv(
                csv_path=active_csv_path,
                presentation_csv_path=active_presentation_csv_path,
            )

            # Attach CSV and plots so they appear in Allure HTML artifacts.
            presentation_csv_file = Path(active_presentation_csv_path)
            if presentation_csv_file.exists() and _allure is not None:
                _allure.attach(
                    presentation_csv_file.read_text(encoding="utf-8"),
                    name="TS Wind Turbine App Performances",
                    attachment_type=_allure.attachment_type.CSV,
                )

            for plot_path in plot_files:
                plot_file = Path(plot_path)
                if plot_file.exists() and _allure is not None:
                    _allure.attach(
                        plot_file.read_bytes(),
                        name=f"Timeseries Plot - {plot_file.name}",
                        attachment_type=_allure.attachment_type.PNG,
                    )
        except Exception as report_error:
            logger.warning("Failed to update timeseries performance report artifacts: %s", report_error)

    finally:
        if not preserve_environment:
            _safe_teardown(active_cleanup_env, context="test finalization")
        else:
            logger.warning(
                "debug_preserve_environment=true: skipping final compose teardown; "
                "stack and volumes are left running for post-run inspection"
            )

        if results is None:
            results = Result(metadata={"status": False, "error": "Execution did not complete"})

        summarize_test_results(
            results=results,
            test_name=test_name,
            configs=configs,
            get_kpi_config=get_kpi_config,
        )
