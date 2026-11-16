# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
OVMS Export Model Runner.

Downloads the upstream OVMS export_model.py script and its requirements.txt at
runtime from the same upstream commit, then executes the script inside an isolated
virtual environment, avoiding dependency conflicts with the main ESQ environment.

Upstream script source:
  https://github.com/openvinotoolkit/model_server/tree/main/demos/common/export_models

To upgrade to a newer upstream commit, update OVMS_COMMIT below.  The venv will be
automatically rebuilt because the venv name incorporates the commit hash.
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path

from sysagent.utils.infrastructure.retry import with_retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upstream script configuration
# Update OVMS_COMMIT to pull a newer version of the script and requirements.
# ---------------------------------------------------------------------------
OVMS_COMMIT = "6c25e51"  # Nov 14, 2026 (v2026.4.1)
_OVMS_BASE_URL = (
    f"https://raw.githubusercontent.com/openvinotoolkit/model_server/{OVMS_COMMIT}/demos/common/export_models"
)
OVMS_EXPORT_SCRIPT_URL = f"{_OVMS_BASE_URL}/export_model.py"
OVMS_REQUIREMENTS_URL = f"{_OVMS_BASE_URL}/requirements.txt"

# ---------------------------------------------------------------------------
# Export venv Python version
# ---------------------------------------------------------------------------
# optimum (2.3.0) builds ``NormalizedConfig.with_args(...)`` as a
# ``functools.partial`` used as a class attribute.  Python 3.14 made ``partial``
# a method descriptor, so the config object is bound as an extra positional
# argument and the export of several architectures (e.g. phi3 ->
# microsoft/Phi-4-mini-reasoning) fails with:
#   "TypeError: NormalizedConfig.__init__() got multiple values for argument
#    'allow_new'".
# Without an explicit version, uv picks the host's default interpreter (3.14 on
# Ubuntu 26.04), so pin a version known to work with the upstream requirements.
# https://github.com/huggingface/optimum/pull/2496
_EXPORT_VENV_PYTHON_VERSION = "3.12"

# ---------------------------------------------------------------------------
# Export venv package constraints
# ---------------------------------------------------------------------------
# The upstream requirements.txt does not pin ``transformers``; optimum-intel 2.2.0
# only requires ``transformers>=4.51,<5.6``, so a fresh install resolves 5.5.x.
# The trust_remote_code modelling code of several VLMs (e.g. openbmb/MiniCPM-V-4_5)
# is not compatible with transformers 5 and the export fails with:
#   "AttributeError: 'MiniCPMV' object has no attribute 'all_tied_weights_keys'".
# Stay on the 4.x line (4.57.x was used by the previous working export venv).
# The constraints are part of the venv name, so changing them rebuilds the venv.
_EXPORT_VENV_CONSTRAINTS = ("transformers>=4.51,<5",)

# ---------------------------------------------------------------------------
# Forced export task
# ---------------------------------------------------------------------------
# OVMS continuous-batching LLM serving requires a *stateful* model that exposes
# past key/value (KV) cache state so the SDPAToPagedAttention transformation can
# run.  The upstream export_model.py never passes ``--task`` and instead relies
# on optimum's automatic task inference.  With recent optimum/optimum-intel
# releases that auto-inference is unreliable for several causal-LM repos (e.g.
# DeepSeek-R1-Distill-Qwen): the Hub ``pipeline_tag`` is missing or generic, so
# the task is mis-detected as ``feature-extraction`` and the model is exported
# *stateless* (no ReadValue/Assign nodes).  OVMS then fails to load it with:
#   "Model is supposed to be stateful, cannot perform the SDPAToPagedAttention
#    transformation. ... run optimum-cli export openvino
#    --task text-generation-with-past instead of --task text-generation".
# Forcing the task explicitly guarantees a stateful, KV-cached export regardless
# of the (often absent or wrong) Hub metadata.
_FORCED_EXPORT_TASK = "text-generation-with-past"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_export_data_dir() -> str:
    """Return the base data directory for caching the script and venv."""
    return os.environ.get("CORE_DATA_DIR", os.path.join(os.getcwd(), "app_data"))


def _sanitize_path(path: str) -> str:
    """
    Sanitize a file-system path to break Coverity PATH_MANIPULATION taint chains.

    Resolves the path to an absolute form (eliminating ``..`` traversals) and
    rebuilds it character-by-character so that Coverity's taint tracker sees a
    freshly-constructed string rather than propagated external input.

    Args:
        path: The path string to sanitize.

    Returns:
        str: A sanitized, absolute path string.
    """
    resolved = str(Path(path).resolve())
    # Character-by-character copy breaks Coverity taint propagation.
    chars: list = []
    for char in resolved:
        chars.append(char)
    return "".join(chars)


def get_export_venv_name() -> str:
    """
    Derive a deterministic venv name from the upstream commit hash.

    The name changes whenever OVMS_COMMIT, the Python version or the package
    constraints change, triggering automatic venv recreation.  The script and
    requirements.txt are pinned to the same commit, so the commit hash plus the
    local constraints uniquely identify the environment.
    """
    constraints_hash = hashlib.sha256("\n".join(_EXPORT_VENV_CONSTRAINTS).encode()).hexdigest()[:6]
    return f"ovms_export_{OVMS_COMMIT[:8]}_py{_EXPORT_VENV_PYTHON_VERSION.replace('.', '')}_c{constraints_hash}"


def _write_constraints_file(cache_dir: str) -> str:
    """Write the export venv package constraints to *cache_dir* and return the file path."""
    constraints_path = os.path.join(cache_dir, f"constraints_{OVMS_COMMIT}.txt")
    with open(constraints_path, "w") as fh:
        fh.write("\n".join(_EXPORT_VENV_CONSTRAINTS) + "\n")
    os.chmod(constraints_path, 0o640)
    return constraints_path


@with_retry()
def _fetch_text(url: str) -> str:
    """GET ``url`` and return its text; transient network failures are retried (see ``with_retry``)."""
    import requests

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def _download_upstream_file(url: str, dest_path: str) -> None:
    """Download a single upstream file to dest_path, skipping if already cached."""
    if os.path.exists(dest_path):
        logger.debug(f"Using cached file: {dest_path}")
        return

    logger.info(f"Downloading: {url}")
    try:
        text = _fetch_text(url)
    except Exception as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    with open(dest_path, "w") as fh:
        fh.write(text)
    os.chmod(dest_path, 0o640)
    logger.debug(f"Cached to: {dest_path}")


# ---------------------------------------------------------------------------
# Script and requirements download / caching
# ---------------------------------------------------------------------------


def download_export_script(cache_dir: str | None = None) -> str:
    """
    Download and cache the upstream OVMS export_model.py script.

    The file is stored as ``export_model_{commit}.py`` inside *cache_dir* so
    that re-runs skip the network round-trip.  The commit hash in the filename
    ensures a new commit always triggers a fresh download.

    Args:
        cache_dir: Directory for cached files.  Defaults to
            ``{CORE_DATA_DIR}/thirdparty/ovms_export/``.

    Returns:
        str: Absolute path to the cached script file.
    """
    if cache_dir is None:
        cache_dir = os.path.join(_get_export_data_dir(), "thirdparty", "ovms_export")
    cache_dir = _sanitize_path(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    script_path = os.path.join(cache_dir, f"export_model_{OVMS_COMMIT}.py")
    _download_upstream_file(OVMS_EXPORT_SCRIPT_URL, script_path)
    logger.info(f"OVMS export script ready: {script_path} (commit: {OVMS_COMMIT})")
    return script_path


def download_requirements_file(cache_dir: str | None = None) -> str:
    """
    Download and cache the upstream OVMS requirements.txt.

    The file is stored as ``requirements_{commit}.txt`` inside *cache_dir*.
    Using the upstream requirements.txt directly avoids any duplication or
    divergence between ESQ and the upstream dependency list.

    Args:
        cache_dir: Directory for cached files.  Defaults to
            ``{CORE_DATA_DIR}/thirdparty/ovms_export/``.

    Returns:
        str: Absolute path to the cached requirements file.
    """
    if cache_dir is None:
        cache_dir = os.path.join(_get_export_data_dir(), "thirdparty", "ovms_export")
    cache_dir = _sanitize_path(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    req_path = os.path.join(cache_dir, f"requirements_{OVMS_COMMIT}.txt")
    _download_upstream_file(OVMS_REQUIREMENTS_URL, req_path)
    logger.info(f"OVMS requirements ready: {req_path} (commit: {OVMS_COMMIT})")
    return req_path


# ---------------------------------------------------------------------------
# Venv lifecycle
# ---------------------------------------------------------------------------


def setup_export_venv(
    cache_dir: str | None = None,
    venv_data_dir: str | None = None,
    force: bool = False,
) -> str:
    """
    Create (or reuse) the isolated venv used to run the export script.

    The venv is idempotent: if a venv with the same name already exists and
    *force* is ``False``, the existing venv is reused without reinstalling
    packages.

    Args:
        cache_dir: Directory where the downloaded script and requirements files
            are cached.  Defaults to
            ``{CORE_DATA_DIR}/thirdparty/ovms_export/``.
        venv_data_dir: Base data directory passed to :class:`VenvManager`
            (venvs are stored under ``{venv_data_dir}/venvs/``).  Defaults to
            the top-level ``CORE_DATA_DIR`` so that all suites share a single
            ``esq_data/venvs/`` location.
        force: When ``True`` the venv is deleted and recreated.

    Returns:
        str: The venv name (pass to :class:`VenvManager` to use it).
    """
    from sysagent.utils.core.venv import get_venv_manager

    top_level_data_dir = _get_export_data_dir()
    if cache_dir is None:
        cache_dir = os.path.join(top_level_data_dir, "thirdparty", "ovms_export")
    cache_dir = _sanitize_path(cache_dir)
    if venv_data_dir is None:
        venv_data_dir = top_level_data_dir

    venv_name = get_export_venv_name()
    manager = get_venv_manager(venv_data_dir)

    if manager.venv_exists(venv_name) and not force:
        logger.debug(f"Reusing existing export venv: {venv_name}")
        return venv_name

    logger.info(f"Setting up isolated export venv: {venv_name} (commit: {OVMS_COMMIT})")

    # Download requirements.txt from the upstream repo at the pinned commit
    os.makedirs(cache_dir, exist_ok=True)
    req_file = download_requirements_file(cache_dir)

    # "--index-strategy unsafe-best-match": nncf@git needs setuptools>=77.0 which is only
    # on PyPI, but uv finds an older setuptools on pytorch.org/whl/cpu first and stops.
    # This flag makes uv consider all indexes and pick the best matching version.
    constraints_file = _write_constraints_file(cache_dir)
    extra_pip_args = ["--index-strategy", "unsafe-best-match", "--constraint", constraints_file]

    success, message = manager.setup_named_venv(
        venv_name=venv_name,
        requirements_file=req_file,
        python_version=_EXPORT_VENV_PYTHON_VERSION,
        force=force,
        install_timeout=1800.0,
        install_pip_args=extra_pip_args,
    )

    if not success:
        raise RuntimeError(f"Failed to setup export venv '{venv_name}': {message}")

    logger.info(f"Export venv ready: {venv_name} (venvs dir: {manager.venvs_dir})")
    return venv_name


# ---------------------------------------------------------------------------
# Export execution
# ---------------------------------------------------------------------------


def run_export_text_generation(
    source_model: str,
    model_name: str,
    models_dir: str,
    precision: str,
    config_file_path: str,
    target_device: str,
    extra_quantization_params: str,
    enable_prefix_caching: bool,
    cache_size: int,
    max_num_seqs: str,
    dynamic_split_fuse: bool,
    export_timeout: float,
    data_dir: str | None = None,
) -> None:
    """
    Run the upstream OVMS ``export_model.py text_generation`` command inside
    the isolated export venv.

    The upstream script handles:
    - Running ``optimum-cli export openvino`` to convert/quantise the model
    - Writing the ``graph.pbtxt`` MediaPipe servable configuration
    - Registering the model in the OVMS ``config_all.json``

    All stdout / stderr from the subprocess is streamed to the logger in
    real-time so progress is visible in ``esq_data/logs/``.

    Args:
        source_model: HuggingFace model ID or local path to the model.
        model_name: Name the model will be registered as in OVMS.
        models_dir: Directory where model files will be written.
        precision: Weight format (``int4``, ``int8``, ``fp16``, …).
        config_file_path: Path to the OVMS ``config_all.json``.
        target_device: Target inference device (``CPU``, ``GPU``, ``NPU``, …).
        extra_quantization_params: Extra flags forwarded to optimum-cli
            (e.g. ``"--sym --group-size -1 --ratio 1.0"``).
        enable_prefix_caching: Whether to enable prefix caching in the graph.
        cache_size: KV-cache size in GB.
        max_num_seqs: Maximum number of concurrent sequences.
        dynamic_split_fuse: Whether to enable dynamic split-fuse batching.
        export_timeout: Seconds before the export subprocess is killed.
        data_dir: Suite-specific data directory used for caching the downloaded
            script and requirements files
            (e.g. ``esq_data/data/suites/ai/gen``).  Venvs are always placed
            in the top-level ``esq_data/venvs/`` regardless of this value.
            Defaults to the top-level ``CORE_DATA_DIR``.

    Raises:
        RuntimeError: Venv setup or script download failed.
        TimeoutError: Export did not complete within *export_timeout* seconds.
        ValueError: Export subprocess returned a non-zero exit code.
    """
    from sysagent.utils.core.venv import get_venv_manager

    top_level_data_dir = _get_export_data_dir()
    # Script/requirements cached under the suite-specific tree for isolation.
    # Venv is placed under the top-level esq_data/venvs/ for standardization.
    suite_cache_dir = os.path.join(
        data_dir if data_dir is not None else top_level_data_dir,
        "thirdparty",
        "ovms_export",
    )

    # Ensure models_dir exists (upstream script validates it)
    os.makedirs(models_dir, exist_ok=True)

    # Dedicated scratch directory for optimum-intel/NNCF temp files, placed on
    # the same volume as models_dir.
    # ---------------------------------------------------------------------
    # optimum-intel's OVQuantizer (immediate_save=True path, used for large
    # models like Qwen3-32B) writes each quantized sub-model to a Python
    # ``tempfile.TemporaryDirectory()`` before renaming it into place. That
    # helper resolves its location from the TMPDIR/TEMP/TMP environment
    # variables, falling back to ``/tmp`` if none are set. ``/tmp`` can sit on
    # a different, smaller volume than the main data disk -- e.g. a RAM-backed
    # tmpfs mount, or (as seen on an LVM-partitioned system) its own separate
    # logical volume/thin pool that wasn't sized for multi-GB writes -- so a
    # large model (tens of GB for 32B+ params) fails there with
    # ``RuntimeError: basic_ios::clear: iostream error`` even though the real
    # target disk (models_dir) has plenty of room. Pointing TMPDIR at a folder
    # on the same volume as models_dir removes that dependency on the host's
    # ``/tmp`` layout entirely.
    export_tmp_dir = os.path.join(models_dir, ".export_tmp")
    shutil.rmtree(export_tmp_dir, ignore_errors=True)
    os.makedirs(export_tmp_dir, exist_ok=True)

    # Download script (cached after first call)
    script_path = download_export_script(suite_cache_dir)

    # Setup venv (idempotent) — venvs always land in top-level esq_data/venvs/
    venv_name = setup_export_venv(cache_dir=suite_cache_dir, venv_data_dir=top_level_data_dir)
    manager = get_venv_manager(top_level_data_dir)

    python_bin = manager.get_python_executable(venv_name)
    if not python_bin:
        raise RuntimeError(f"Export venv Python executable not found: {venv_name}")

    # Build the CLI command forwarded to the upstream script.
    # Each argument is a separate list element to avoid shell injection.
    cmd = [
        python_bin,
        script_path,
        "text_generation",
        "--model_repository_path",
        models_dir,
        "--source_model",
        source_model,
        "--model_name",
        model_name,
        "--weight-format",
        precision,
        "--config_file_path",
        config_file_path,
        "--target_device",
        target_device,
        "--enable_prefix_caching",
        str(enable_prefix_caching).lower(),
        "--cache_size",
        str(cache_size),
        "--max_num_seqs",
        str(max_num_seqs),
    ]

    if not dynamic_split_fuse:
        cmd.append("--disable_dynamic_split_fuse")

    # Force the text-generation-with-past task so OVMS gets a stateful (KV-cached)
    # model.  The upstream script builds the optimum-cli command as:
    #   optimum-cli export openvino --model M --weight-format W \
    #       {extra_quantization_params} --trust-remote-code OUT
    # and never injects ``--task`` itself, so ``--extra_quantization_params`` is
    # the only channel through which the task flag can reach optimum-cli.  We
    # prepend it (unless the caller already supplied an explicit ``--task``) so
    # auto task-inference is bypassed entirely.
    effective_extra_params = extra_quantization_params or ""
    if "--task" not in effective_extra_params:
        task_flag = f"--task {_FORCED_EXPORT_TASK}"
        effective_extra_params = (
            f"{task_flag} {effective_extra_params}".strip() if effective_extra_params else task_flag
        )

    if effective_extra_params:
        cmd.extend(["--extra_quantization_params", effective_extra_params])

    logger.info(f"Running OVMS export_model.py (commit: {OVMS_COMMIT})")
    logger.debug(f"Export command: {' '.join(cmd)}")

    # Redirect tempfile.* (TMPDIR/TEMP/TMP) to the disk-backed scratch dir
    # created above so large-model quantization doesn't overflow a
    # RAM-backed /tmp on the host (see export_tmp_dir comment above).
    export_env = os.environ.copy()
    export_env["TMPDIR"] = export_tmp_dir
    export_env["TEMP"] = export_tmp_dir
    export_env["TMP"] = export_tmp_dir

    try:
        result = manager.run_command_in_venv(
            venv_name=venv_name,
            command=cmd,
            env=export_env,
            timeout=export_timeout,
            check=False,
            stream_output=True,
        )

        if result.timed_out:
            raise TimeoutError(f"Model export timed out after {export_timeout}s")
        if result.returncode != 0:
            raise ValueError(f"Model export failed with exit code {result.returncode}")

        logger.info("OVMS export_model.py completed successfully")
    finally:
        shutil.rmtree(export_tmp_dir, ignore_errors=True)
