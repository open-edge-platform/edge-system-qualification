# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Container management for Whisper ASR benchmarking."""

import grp
import json
import logging
import os

import pytest

from sysagent.utils.infrastructure import DockerClient

logger = logging.getLogger(__name__)

_CONTAINER_MNT_DIR = "/mnt"
_RESULT_FILENAME = "whisper_result.json"


def _get_render_gid() -> int:
    """Return the render group GID, falling back to the current GID."""
    try:
        return grp.getgrnam("render").gr_gid
    except KeyError:
        logger.warning("'render' group not found; falling back to current GID")
        return os.getgid()


def _container_devices() -> list:
    devices = ["/dev/dri:/dev/dri"]
    if os.path.exists("/dev/accel"):
        devices.append("/dev/accel:/dev/accel")
    return devices


def build_whisper_image(
    docker_client: DockerClient,
    backend: str,
    containers_dir: str,
    image_tag: str,
) -> str:
    """Build the Whisper Docker image for *backend* and return the image tag.

    The build context is the ``containers/`` directory so that Dockerfiles can
    COPY files from both the backend subdirectory and the shared ``utils/``
    subdirectory.

    Args:
        docker_client: Connected DockerClient instance.
        backend: ``"openvino"`` or ``"pytorch"``.
        containers_dir: Absolute path to ``src/containers/``.
        image_tag: Tag to apply to the built image.

    Returns:
        The resolved image tag.
    """
    dockerfile = os.path.join(backend, "Dockerfile")

    # Skip build if the image already exists locally (avoids rebuilding across separate runs)
    # try:
    #     import docker as _docker
    #     docker_client.client.images.get(image_tag)
    #     logger.info("Whisper %s image already exists locally, skipping build: %s", backend, image_tag)
    #     return image_tag
    # except _docker.errors.ImageNotFound:
    #     pass

    logger.info("Building Whisper %s Docker image: %s", backend, image_tag)
    result = docker_client.build_image(
        path=containers_dir,
        tag=image_tag,
        dockerfile=dockerfile,
        extract_base_image_info=True,
    )
    if not result.get("image_id"):
        pytest.fail(f"Failed to build Whisper {backend} Docker image: {image_tag}")
    logger.info("Whisper %s image built successfully: %s", backend, image_tag)
    return image_tag


def run_openvino_export_container(
    docker_client: DockerClient,
    image_tag: str,
    container_name: str,
    data_dir: str,
    model_hf_id: str,
    model_dir: str,
    export_timeout: int = 1800,
) -> None:
    """Export a Whisper model to OpenVINO IR format inside a Docker container.

    Args:
        docker_client: Connected DockerClient instance.
        image_tag: Whisper-OpenVINO image tag.
        container_name: Name for the container.
        data_dir: Host-side suite data directory (mounted at /mnt).
        model_hf_id: HuggingFace model identifier.
        model_dir: Absolute host path for the exported model.
        export_timeout: Seconds to allow for export. Default 1800.
    """
    model_dir_rel = os.path.relpath(model_dir, data_dir)
    container_model_dir = os.path.join(_CONTAINER_MNT_DIR, model_dir_rel)

    cmd = (
        f"python3 /app/openvino_backend.py"
        f" --export-model"
        f" --model-hf-id {model_hf_id}"
        f" --model-dir {container_model_dir}"
        f" --export-timeout {export_timeout}"
    )

    volumes = {data_dir: {"bind": _CONTAINER_MNT_DIR, "mode": "rw"}}

    result = docker_client.run_container(
        name=container_name,
        image=image_tag,
        entrypoint="/bin/bash",
        command=["-c", cmd],
        volumes=volumes,
        user=f"{os.getuid()}:{os.getgid()}",
        timeout=export_timeout + 60,
        mode="batch",
    )

    if result.get("container_info", {}).get("exit_code") != 0:
        raise RuntimeError(
            f"Whisper OpenVINO export container failed: {result.get('container_logs_text', 'unknown error')}"
        )


def run_pytorch_prepare_container(
    docker_client: DockerClient,
    image_tag: str,
    container_name: str,
    data_dir: str,
    model_size: str,
    pytorch_model_dir: str,
    download_timeout: int = 3600,
) -> None:
    """Pre-download PyTorch Whisper model weights inside a Docker container.

    Args:
        docker_client: Connected DockerClient instance.
        image_tag: Whisper-PyTorch image tag.
        container_name: Name for the container.
        data_dir: Host-side suite data directory (mounted at /mnt).
        model_size: Whisper model size string.
        pytorch_model_dir: Absolute host path used as model download root.
        download_timeout: Seconds to allow for download. Default 3600.
    """
    model_dir_rel = os.path.relpath(pytorch_model_dir, data_dir)
    container_model_dir = os.path.join(_CONTAINER_MNT_DIR, model_dir_rel)

    cmd = (
        f"python3 /app/pytorch_backend.py"
        f" --prepare-model"
        f" --model-size {model_size}"
        f" --model-dir {container_model_dir}"
    )

    volumes = {data_dir: {"bind": _CONTAINER_MNT_DIR, "mode": "rw"}}

    result = docker_client.run_container(
        name=container_name,
        image=image_tag,
        entrypoint="/bin/bash",
        command=["-c", cmd],
        volumes=volumes,
        user=f"{os.getuid()}:{os.getgid()}",
        timeout=download_timeout + 60,
        mode="batch",
    )

    if result.get("container_info", {}).get("exit_code") != 0:
        raise RuntimeError(
            f"Whisper PyTorch model prepare container failed: {result.get('container_logs_text', 'unknown error')}"
        )


def run_openvino_whisper_container(
    docker_client: DockerClient,
    image_tag: str,
    container_name: str,
    data_dir: str,
    model_dir: str,
    ov_device: str,
    wav_file_path: str,
    warmup_runs: int,
    num_runs: int,
    reference_transcript: str,
    inference_timeout: int = 600,
) -> dict:
    """Run OpenVINO Whisper inference inside a Docker container.

    The container mounts *data_dir* at ``/mnt`` so all paths passed to the
    script must be expressed relative to that mount point.

    Args:
        docker_client: Connected DockerClient instance.
        image_tag: Whisper-OpenVINO image tag.
        container_name: Name for the container.
        data_dir: Host-side suite data directory (mounted at /mnt).
        model_dir: Absolute host path to the exported OpenVINO model.
        ov_device: OpenVINO device string, e.g. ``"CPU"`` or ``"GPU"``.
        wav_file_path: Absolute host path to the WAV audio file.
        warmup_runs: Number of warm-up passes.
        num_runs: Number of timed inference passes.
        reference_transcript: Ground-truth text for WER/CER.
        inference_timeout: Seconds to allow for inference. Default 600.

    Returns:
        Parsed result dict with keys ``metrics``, ``parameters``, ``metadata``.
    """
    result_filename = f"{container_name}_{_RESULT_FILENAME}"
    container_result_dir = os.path.join(_CONTAINER_MNT_DIR, "results")

    # Express paths relative to the container mount point
    model_dir_rel = os.path.relpath(model_dir, data_dir)
    wav_file_rel = os.path.relpath(wav_file_path, data_dir)
    container_model_dir = os.path.join(_CONTAINER_MNT_DIR, model_dir_rel)
    container_wav_file = os.path.join(_CONTAINER_MNT_DIR, wav_file_rel)
    container_output_file = os.path.join(container_result_dir, result_filename)

    cmd = (
        f"python3 /app/openvino_backend.py"
        f" --model-dir {container_model_dir}"
        f" --device {ov_device}"
        f" --wav-file {container_wav_file}"
        f" --data-dir {_CONTAINER_MNT_DIR}"
        f" --warmup-runs {warmup_runs}"
        f" --num-runs {num_runs}"
        f" --reference '{reference_transcript}'"
        f" --output-file {container_output_file}"
    )

    volumes = {data_dir: {"bind": _CONTAINER_MNT_DIR, "mode": "rw"}}

    os.makedirs(os.path.join(data_dir, "results"), exist_ok=True)
    os.chmod(os.path.join(data_dir, "results"), 0o770)

    result = docker_client.run_container(
        name=container_name,
        image=image_tag,
        entrypoint="/bin/bash",
        command=["-c", cmd],
        volumes=volumes,
        devices=_container_devices(),
        group_add=[_get_render_gid(), os.getgid()],
        user=f"{os.getuid()}:{os.getgid()}",
        result_file=result_filename,
        container_result_file_dir=container_result_dir,
        timeout=inference_timeout,
        mode="batch",
    )

    if result.get("container_info", {}).get("exit_code") != 0:
        raise RuntimeError(
            f"Whisper OpenVINO container failed: {result.get('container_logs_text', 'unknown error')}"
        )

    return _parse_container_result(result, result_filename, data_dir)


def run_pytorch_whisper_container(
    docker_client: DockerClient,
    image_tag: str,
    container_name: str,
    data_dir: str,
    model_size: str,
    torch_device: str,
    pytorch_model_dir: str,
    wav_file_path: str,
    warmup_runs: int,
    num_runs: int,
    reference_transcript: str,
    inference_timeout: int = 600,
) -> dict:
    """Run PyTorch Whisper inference inside a Docker container.

    Args:
        docker_client: Connected DockerClient instance.
        image_tag: Whisper-PyTorch image tag.
        container_name: Name for the container.
        data_dir: Host-side suite data directory (mounted at /mnt).
        model_size: Whisper model size string, e.g. ``"base"``.
        torch_device: PyTorch device string: ``"cpu"`` or ``"xpu"``.
        pytorch_model_dir: Absolute host path used as model download root.
        wav_file_path: Absolute host path to the WAV audio file.
        warmup_runs: Number of warm-up passes.
        num_runs: Number of timed inference passes.
        reference_transcript: Ground-truth text for WER/CER.
        inference_timeout: Seconds to allow for inference. Default 600.

    Returns:
        Parsed result dict with keys ``metrics``, ``parameters``, ``metadata``.
    """
    result_filename = f"{container_name}_{_RESULT_FILENAME}"
    container_result_dir = os.path.join(_CONTAINER_MNT_DIR, "results")

    model_dir_rel = os.path.relpath(pytorch_model_dir, data_dir)
    wav_file_rel = os.path.relpath(wav_file_path, data_dir)
    container_model_dir = os.path.join(_CONTAINER_MNT_DIR, model_dir_rel)
    container_wav_file = os.path.join(_CONTAINER_MNT_DIR, wav_file_rel)
    container_output_file = os.path.join(container_result_dir, result_filename)

    cmd = (
        f"python3 /app/pytorch_backend.py"
        f" --model-size {model_size}"
        f" --torch-device '{torch_device}'"
        f" --model-dir {container_model_dir}"
        f" --wav-file {container_wav_file}"
        f" --warmup-runs {warmup_runs}"
        f" --num-runs {num_runs}"
        f" --reference '{reference_transcript}'"
        f" --output-file {container_output_file}"
    )

    volumes = {data_dir: {"bind": _CONTAINER_MNT_DIR, "mode": "rw"}}

    os.makedirs(os.path.join(data_dir, "results"), exist_ok=True)
    os.chmod(os.path.join(data_dir, "results"), 0o770)

    result = docker_client.run_container(
        name=container_name,
        image=image_tag,
        entrypoint="/bin/bash",
        command=["-c", cmd],
        volumes=volumes,
        devices=_container_devices(),
        group_add=[_get_render_gid(), os.getgid()],
        user=f"{os.getuid()}:{os.getgid()}",
        result_file=result_filename,
        container_result_file_dir=container_result_dir,
        timeout=inference_timeout,
        mode="batch",
    )

    if result.get("container_info", {}).get("exit_code") != 0:
        raise RuntimeError(
            f"Whisper PyTorch container failed: {result.get('container_logs_text', 'unknown error')}"
        )

    return _parse_container_result(result, result_filename, data_dir)


def _parse_container_result(container_result: dict, result_filename: str, data_dir: str) -> dict:
    """Extract and parse the JSON result written by the backend script."""
    # DockerClient may surface the file content via result_json or result_text
    if container_result.get("result_json"):
        return container_result["result_json"]

    if container_result.get("result_text"):
        try:
            return json.loads(container_result["result_text"])
        except json.JSONDecodeError:
            pass

    # Fall back to reading from the mounted results directory
    result_path = os.path.join(data_dir, "results", result_filename)
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    raise RuntimeError(
        f"Container completed but result file not found: {result_path}"
    )
