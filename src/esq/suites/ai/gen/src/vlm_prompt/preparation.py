# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VLM Prompt preparation: model serving config, media assets and client venv."""

import hashlib
import json
import logging
import os
import shutil
from typing import Any
from urllib.parse import urlparse

import jinja2
import requests
from esq.utils.downloads import download_model
from esq.utils.servers.ovms import export_ovms_model
from esq.utils.servers.ovms.ovms_utils import (
    add_servable_to_config,
    text_generation_graph_template,
)
from sysagent.utils.core.venv import VenvManager, get_venv_manager

logger = logging.getLogger(__name__)

# Pinned because the default interpreter differs per OS (3.14 on Ubuntu 26.04) and numpy 2.5 needs >= 3.12
CLIENT_VENV_PYTHON_VERSION = "3.12"

# Files an OpenVINO VLM export must contain to be served by OVMS
REQUIRED_MODEL_FILES = ("openvino_language_model.xml", "openvino_language_model.bin")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_rest_workers(
    rest_workers: int | None,
    max_rest_workers: int | None,
    cpu_count: int | None = None,
) -> tuple[int | None, str]:
    """
    Decide the ``--rest_workers`` value for OVMS and describe it for the result.

    ``rest_workers`` unset leaves the value to OVMS, which starts one worker per CPU core. Each worker
    needs about 3 file descriptors, so on a system with many cores the default can exceed the open files
    limit (1024 on Ubuntu) and OVMS fails with "Too many open files". ``max_rest_workers`` is a failsafe:
    when the number of workers would exceed it, ``max_rest_workers`` is used instead. Without it the value
    is never limited.

    Args:
        rest_workers: Explicit number of workers, or None for the OVMS default.
        max_rest_workers: Highest number of workers allowed, or None for no limit.
        cpu_count: Number of CPUs of the system, which is the OVMS default. Detected when None.

    Returns:
        The value to pass to OVMS (None leaves the default) and a short description.
    """
    cpus = cpu_count or os.cpu_count() or 1
    effective = rest_workers or cpus  # the OVMS default is one worker per CPU
    if max_rest_workers and effective > max_rest_workers:
        source = f"requested {rest_workers}" if rest_workers else f"default of {cpus} CPUs"
        return max_rest_workers, f"{max_rest_workers} (limited, {source})"
    if rest_workers:
        return rest_workers, str(rest_workers)
    return None, f"default ({cpus})"


def setup_client_venv(core_data_dir: str, requirements_file: str) -> tuple[VenvManager, str]:
    """
    Create (or reuse) the isolated venv of the load generator.

    The venv name contains a hash of the requirements file, so changing the requirements creates a new
    venv. Venvs are kept in ``{core_data_dir}/venvs/``, shared with the other suites.

    Returns:
        The venv manager and the venv name.
    """
    manager = get_venv_manager(core_data_dir)
    venv_name = (
        f"vlm_prompt_client_py{CLIENT_VENV_PYTHON_VERSION.replace('.', '')}_"
        f"{manager.get_requirements_hash(requirements_file)[:8]}"
    )
    if manager.venv_exists(venv_name):
        logger.info(f"Reusing VLM client venv: {venv_name}")
        return manager, venv_name

    logger.info(f"Setting up VLM client venv: {venv_name}")
    success, message = manager.setup_named_venv(
        venv_name=venv_name,
        requirements_file=requirements_file,
        python_version=CLIENT_VENV_PYTHON_VERSION,
        install_timeout=900.0,
    )
    if not success:
        raise RuntimeError(f"Failed to set up VLM client venv '{venv_name}': {message}")
    return manager, venv_name


def prepare_model(
    model_id: str,
    models_dir: str,
    device_id: str,
    cache_size: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    enable_prefix_caching: bool,
    plugin_config: dict[str, Any],
    download_timeout: float,
    model_precision: str | None = None,
    quantization_params: str | None = None,
    export_timeout: float = 3600,
    data_dir: str | None = None,
) -> str:
    """
    Prepare an OpenVINO VLM for OVMS and generate the graph and config for it.

    With ``model_precision`` the source model is exported and quantized through the same
    OVMS export flow as the text generation suite (``export_ovms_model``). Without it,
    ``model_id`` must be a pre-quantized OpenVINO VLM, which is downloaded as is.

    The graph is always written here so the serving settings come from the profile.
    Only this model is registered in ``config_all.json``, so OVMS loads nothing else.

    Returns:
        Servable name to use in requests.
    """
    os.makedirs(models_dir, exist_ok=True)

    if model_precision:
        servable_name = _export_model(
            model_id=model_id,
            models_dir=models_dir,
            device_id=device_id,
            model_precision=model_precision,
            quantization_params=quantization_params,
            export_timeout=export_timeout,
            download_timeout=download_timeout,
            data_dir=data_dir,
        )
    else:
        servable_name = _download_prequantized_model(model_id, models_dir, download_timeout)
    model_dir = os.path.join(models_dir, servable_name)

    missing = [f for f in REQUIRED_MODEL_FILES if not os.path.isfile(os.path.join(model_dir, f))]
    if missing:
        raise RuntimeError(f"Model {model_id} is not an OpenVINO VLM export, missing: {', '.join(missing)}")

    graph = jinja2.Environment(loader=jinja2.BaseLoader, autoescape=True).from_string(text_generation_graph_template)
    graph_content = graph.render(
        model_path="./",
        target_device=device_id.upper(),
        plugin_config=json.dumps(plugin_config),
        enable_prefix_caching=enable_prefix_caching,
        cache_size=cache_size,
        max_num_seqs=str(max_num_seqs),
        max_num_batched_tokens=max_num_batched_tokens,
        dynamic_split_fuse=True,
        pipeline_type="VLM_CB",
    )
    with open(os.path.join(model_dir, "graph.pbtxt"), "w") as f:
        f.write(graph_content)

    config_path = os.path.join(models_dir, "config_all.json")
    if os.path.exists(config_path):
        os.remove(config_path)
    add_servable_to_config(config_path, servable_name, servable_name)
    return servable_name


def _export_model(
    model_id: str,
    models_dir: str,
    device_id: str,
    model_precision: str,
    quantization_params: str | None,
    export_timeout: float,
    download_timeout: float,
    data_dir: str | None,
) -> str:
    """
    Export and quantize a Hugging Face VLM with the shared OVMS export flow.

    ``export_ovms_model`` forces ``--task text-generation-with-past`` for LLMs unless the
    parameters already contain ``--task``, so a VLM export must pass
    ``--task image-text-to-text`` in ``quantization_params``. The export is skipped when a
    complete model for the same parameters already exists.
    """
    params = quantization_params or ""
    if "--task" not in params:
        params = f"--task image-text-to-text {params}".strip()

    quant_hash = hashlib.sha256(params.encode()).hexdigest()[:8]
    servable_name = f"{model_id.replace('/', '_')}_q{quant_hash}"
    model_dir = os.path.join(models_dir, servable_name)
    if all(os.path.isfile(os.path.join(model_dir, f)) for f in REQUIRED_MODEL_FILES):
        logger.info(f"Exported VLM already present at {model_dir}")
        return servable_name

    logger.info(f"Exporting VLM {model_id} ({model_precision}, {params})")
    _, export_duration, download_duration, _, exported_name = export_ovms_model(
        model_id_or_path=model_id,
        models_dir=models_dir,
        model_precision=model_precision,
        device_id=device_id,
        configs={"quantization_params": params},
        export_timeout=export_timeout,
        download_timeout=download_timeout,
        data_dir=data_dir,
    )
    logger.info(f"VLM export completed (download {download_duration:.0f}s, export {export_duration:.0f}s)")
    return exported_name


def _download_prequantized_model(model_id: str, models_dir: str, download_timeout: float) -> str:
    servable_name = model_id.replace("/", "_")
    model_dir = os.path.join(models_dir, servable_name)

    if not all(os.path.isfile(os.path.join(model_dir, f)) for f in REQUIRED_MODEL_FILES):
        tmp_dir = os.path.join(models_dir, f".tmp_{servable_name}")
        logger.info(f"Downloading VLM {model_id}")
        download_model(model_id=model_id, local_dir=tmp_dir, timeout=download_timeout)
        shutil.rmtree(model_dir, ignore_errors=True)
        os.makedirs(model_dir, exist_ok=True)
        for entry in os.listdir(tmp_dir):
            if entry.startswith("."):
                continue
            shutil.move(os.path.join(tmp_dir, entry), os.path.join(model_dir, entry))
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        logger.info(f"VLM {model_id} already present at {model_dir}")
    return servable_name


def prepare_media(media: list[dict[str, Any]], media_dir: str, timeout: float = 300) -> list[str]:
    """
    Download the input media (images and videos) declared in the profile.

    Each entry has ``url`` and optionally ``name`` and ``sha256``. A file that already exists is reused
    when its checksum matches ``sha256``, so a run without network access works after the first one.
    ``media_dir`` must only hold the media of one profile entry, because the load generator reads all files in it.
    """
    os.makedirs(media_dir, exist_ok=True)
    paths = []
    for entry in media:
        url = entry["url"]
        name = entry.get("name") or os.path.basename(urlparse(url).path)
        if not name:
            raise ValueError(f"Media entry needs a 'name' when the URL has no file name: {url}")
        target = os.path.join(media_dir, name)
        expected = entry.get("sha256")
        if expected and os.path.isfile(target) and _sha256(target) == expected:
            logger.info(f"Media {name} already present and verified")
            paths.append(target)
            continue
        logger.info(f"Downloading media {name} from {url}")
        with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}) as response:
            response.raise_for_status()
            with open(target, "wb") as f:
                f.writelines(response.iter_content(chunk_size=1024 * 1024))
        if expected and _sha256(target) != expected:
            raise RuntimeError(f"Checksum mismatch for media {name}")
        paths.append(target)
    return paths
