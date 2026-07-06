# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import yaml
from sysagent.utils.infrastructure import download_file

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_FILENAME = "default_context.txt"
DEFAULT_PROMPTS_FILENAME = "default_prompts.txt"
DEFAULT_CORPUS_TEXT = (
    "Intel Edge AI sample context for ChatQnA Core benchmarking.\n"
    "This document is used as a fallback corpus when no custom corpus directory is configured.\n"
    "The benchmark validates ingestion, query latency, and TTFT measurements.\n"
)
DEFAULT_PROMPTS_TEXT = (
    "Summarize the key purpose of this benchmark context.\n"
    "What performance metrics are being validated by this suite?\n"
    "List the main steps in the ChatQnA Core benchmark flow.\n"
)


def _resolve_asset_url(asset_url: str, repo_ref: str) -> str:
    resolved_ref = str(repo_ref or "").strip()
    if not resolved_ref or resolved_ref == "main":
        return asset_url

    main_prefix = "https://raw.githubusercontent.com/open-edge-platform/edge-ai-libraries/main/"
    if asset_url.startswith(main_prefix):
        suffix = asset_url[len(main_prefix) :]
        return f"https://raw.githubusercontent.com/open-edge-platform/edge-ai-libraries/{resolved_ref}/{suffix}"
    return asset_url


def download_configured_assets(configs: Dict[str, object], suite_assets_root: str) -> Optional[Path]:
    raw_assets = configs.get("assets", [])
    if not isinstance(raw_assets, list) or not raw_assets:
        return None
    assets = raw_assets

    assets_root = Path(suite_assets_root).expanduser().resolve()
    assets_root.mkdir(parents=True, exist_ok=True)
    repo_ref = str(configs.get("assets_repo_ref", "main")).strip()

    for asset in assets:
        asset_id = str(asset.get("id", "unnamed-asset")).strip() or "unnamed-asset"
        asset_url = str(asset.get("url", "")).strip()
        asset_path = str(asset.get("path", "")).strip()
        asset_sha256 = str(asset.get("sha256", "")).strip()

        if not asset_url or not asset_path:
            raise RuntimeError(f"Asset '{asset_id}' must define both 'url' and 'path'.")

        normalized_path = asset_path[2:] if asset_path.startswith("./") else asset_path
        normalized_path_obj = Path(normalized_path)
        if normalized_path_obj.is_absolute() or ".." in normalized_path_obj.parts:
            raise RuntimeError(f"Asset '{asset_id}' uses an unsafe relative path: {asset_path}")

        target_path = (assets_root / normalized_path_obj).resolve()
        if assets_root not in [target_path, *target_path.parents]:
            raise RuntimeError(f"Asset '{asset_id}' resolves outside suite assets root: {asset_path}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_url = _resolve_asset_url(asset_url, repo_ref)
        logger.info("Preparing Chat Q&A Core asset '%s' from '%s' -> %s", asset_id, resolved_url, target_path)
        download_file(url=resolved_url, target_path=str(target_path), sha256sum=asset_sha256)

    return assets_root


def _resolve_path_from_config(configs: Dict[str, object], config_key: str, env_key: str) -> str:
    configured = str(configs.get(config_key, "")).strip()
    if configured:
        return configured
    return str(os.environ.get(env_key, "")).strip()


def _resolve_existing_file(path_value: str, description: str) -> Path:
    resolved_path = Path(path_value).expanduser().resolve()
    if not resolved_path.is_file():
        raise RuntimeError(f"Required {description} does not exist: {resolved_path}")
    return resolved_path


def _resolve_existing_directory(path_value: str, description: str) -> Path:
    resolved_path = Path(path_value).expanduser().resolve()
    if not resolved_path.is_dir():
        raise RuntimeError(f"Required {description} does not exist: {resolved_path}")
    return resolved_path


def _is_subpath(path_value: Path, parent: Path) -> bool:
    try:
        path_value.relative_to(parent)
        return True
    except ValueError:
        return False


def _prepare_default_benchmark_inputs(suite_assets_root: str) -> Dict[str, Path]:
    runtime_inputs_dir = Path(suite_assets_root).expanduser().resolve() / "runtime" / "benchmark_inputs"
    corpus_dir = runtime_inputs_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    default_corpus_file = corpus_dir / DEFAULT_CORPUS_FILENAME
    if not default_corpus_file.exists():
        default_corpus_file.write_text(DEFAULT_CORPUS_TEXT, encoding="utf-8")

    prompt_file = runtime_inputs_dir / DEFAULT_PROMPTS_FILENAME
    if not prompt_file.exists():
        prompt_file.write_text(DEFAULT_PROMPTS_TEXT, encoding="utf-8")

    return {"corpus_dir": corpus_dir, "prompt_file": prompt_file}


def _prepare_sizing_tool_benchmark_inputs(reference_root: Path, suite_assets_root: str) -> Optional[Dict[str, Path]]:
    sizing_root = reference_root / "tools" / "genai-applications-sizing"
    profiles_path = sizing_root / "profiles" / "profiles.yaml"
    sample_doc_candidates = [
        sizing_root / "data" / "file1.txt",
        sizing_root / "data" / "file.txt",
    ]

    if not profiles_path.is_file():
        return None

    sample_doc_path: Optional[Path] = None
    for candidate in sample_doc_candidates:
        if candidate.is_file():
            sample_doc_path = candidate
            break
    if sample_doc_path is None:
        return None

    try:
        profile_payload = yaml.safe_load(profiles_path.read_text(encoding="utf-8")) or {}
        profiles_section = profile_payload.get("profiles", {})
        chatqna_profile = profiles_section.get("chatqna_wsf", {})
        prompt_text = str(chatqna_profile.get("prompt", "")).strip()
    except Exception as exc:
        logger.warning("Failed to parse sizing-tool profile inputs from %s: %s", profiles_path, exc)
        return None

    if not prompt_text:
        return None

    runtime_inputs_dir = Path(suite_assets_root).expanduser().resolve() / "runtime" / "benchmark_inputs"
    corpus_dir = runtime_inputs_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    corpus_target = corpus_dir / "sizing_sample_context.txt"
    corpus_target.write_text(sample_doc_path.read_text(encoding="utf-8"), encoding="utf-8")

    prompt_file = runtime_inputs_dir / "sizing_prompts.txt"
    prompt_file.write_text(f"{prompt_text}\n", encoding="utf-8")

    return {"corpus_dir": corpus_dir, "prompt_file": prompt_file}


def resolve_runtime_paths(configs: Dict[str, object], suite_assets_root: str) -> Dict[str, str]:
    downloaded_root = download_configured_assets(configs=configs, suite_assets_root=suite_assets_root)

    reference_app_src_path = _resolve_path_from_config(
        configs,
        config_key="reference_app_src_path",
        env_key="CHATQNA_CORE_REFERENCE_APP_SRC_PATH",
    )
    if reference_app_src_path:
        reference_root = _resolve_existing_directory(reference_app_src_path, "reference app root")
    elif downloaded_root is not None:
        reference_root = downloaded_root
    else:
        raise RuntimeError(
            "Missing Chat Q&A Core reference app root. Set config 'reference_app_src_path' "
            "or env 'CHATQNA_CORE_REFERENCE_APP_SRC_PATH', or provide profile assets "
            "that contain the compose and config files."
        )

    compose_file_value = str(configs.get("compose_file", "")).strip()
    compose_path = (
        _resolve_existing_file(compose_file_value, "compose file")
        if compose_file_value
        else _resolve_existing_file(str(reference_root / "docker" / "compose.yaml"), "compose file")
    )

    backend_runtime = str(configs.get("backend_runtime", "openvino")).strip().lower()
    model_config_value = _resolve_path_from_config(configs, "model_config_path", "MODEL_CONFIG_PATH")
    if model_config_value:
        model_config_path = _resolve_existing_file(model_config_value, "model config")
    else:
        default_name = "ollama_template.yaml" if backend_runtime == "ollama" else "openvino_template.yaml"
        model_config_path = _resolve_existing_file(
            str(reference_root / "model_config" / "sample" / default_name),
            "default model config",
        )

    corpus_dir_value = _resolve_path_from_config(configs, "corpus_dir", "CHATQNA_CORE_CORPUS_DIR")
    prompt_file_value = _resolve_path_from_config(configs, "prompt_file", "CHATQNA_CORE_PROMPT_FILE")

    if not corpus_dir_value or not prompt_file_value:
        default_inputs = _prepare_sizing_tool_benchmark_inputs(reference_root, suite_assets_root)
        if default_inputs is None:
            default_inputs = _prepare_default_benchmark_inputs(suite_assets_root)

        logger.info(
            "Using default ChatQnA benchmark inputs because corpus_dir/prompt_file were not both configured. "
            "corpus_dir=%s, prompt_file=%s",
            default_inputs["corpus_dir"],
            default_inputs["prompt_file"],
        )

        if not corpus_dir_value:
            corpus_dir_value = str(default_inputs["corpus_dir"])
        if not prompt_file_value:
            prompt_file_value = str(default_inputs["prompt_file"])

    corpus_dir = _resolve_existing_directory(corpus_dir_value, "benchmark corpus directory")
    prompt_file = _resolve_existing_file(prompt_file_value, "benchmark prompt file")

    model_cache_value = _resolve_path_from_config(configs, "model_cache_path", "MODEL_CACHE_PATH")
    if model_cache_value:
        model_cache_path = Path(model_cache_value).expanduser().resolve()
    else:
        model_cache_path = Path(suite_assets_root).expanduser().resolve() / "runtime" / "model_cache"
    model_cache_path.mkdir(parents=True, exist_ok=True)

    suite_assets_path = Path(suite_assets_root).expanduser().resolve()
    managed_data_root = next(
        (parent for parent in [suite_assets_path, *suite_assets_path.parents] if parent.name == "data"),
        None,
    )
    if managed_data_root is not None and not _is_subpath(model_cache_path, managed_data_root):
        logger.warning(
            "ChatQnA model_cache_path is outside managed data dir (%s): %s. "
            "Artifacts at this path will not be removed by 'esq clean --all'.",
            managed_data_root,
            model_cache_path,
        )

    nginx_template_path = reference_root / "nginx_config" / "nginx.conf.template"
    resolved_paths = {
        "reference_root": str(reference_root),
        "compose_file": str(compose_path),
        "model_config_path": str(model_config_path),
        "model_cache_path": str(model_cache_path),
        "corpus_dir": str(corpus_dir),
        "prompt_file": str(prompt_file),
        "nginx_template_path": str(nginx_template_path),
        "nginx_output_path": str(reference_root / "nginx_config" / "nginx.conf"),
    }
    return resolved_paths
