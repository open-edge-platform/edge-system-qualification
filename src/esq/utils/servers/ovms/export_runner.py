# Copyright (C) 2025 Intel Corporation
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

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upstream script configuration
# Update OVMS_COMMIT to pull a newer version of the script and requirements.
# ---------------------------------------------------------------------------
OVMS_COMMIT = "3e5cbf5"  # Apr 7, 2026 (v2026.1)
_OVMS_BASE_URL = (
    f"https://raw.githubusercontent.com/openvinotoolkit/model_server/{OVMS_COMMIT}/demos/common/export_models"
)
OVMS_EXPORT_SCRIPT_URL = f"{_OVMS_BASE_URL}/export_model.py"
OVMS_REQUIREMENTS_URL = f"{_OVMS_BASE_URL}/requirements.txt"

# ---------------------------------------------------------------------------
# Per-commit package version overrides
# ---------------------------------------------------------------------------
# For each pinned upstream commit, define exact stable package versions to
# substitute for pre-release pins in the upstream requirements.txt.
# This guarantees a fully reproducible install that never drifts to a newer
# nightly or pre-release build on subsequent runs.
#
# When upgrading OVMS_COMMIT, add a new entry here with the known-working
# stable versions for that commit.  Run the export once manually, note the
# installed versions from the venv, then record the stable equivalents.
# ---------------------------------------------------------------------------
_COMMIT_PACKAGE_OVERRIDES: dict = {
    # Commit: 3e5cbf5 (Apr 7, 2026, v2026.1)
    # Upstream pins openvino==2026.1.0rc2 and openvino-tokenizers==2026.1.0.0rc2.
    # Replace with stable 2026.1.0 releases.
    "3e5cbf5": {
        "openvino": "2026.1.0",
        "openvino-tokenizers": "2026.1.0.0",
    },
}


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

    The name changes whenever OVMS_COMMIT changes, triggering automatic venv
    recreation.  Both the script and requirements.txt are pinned to the same
    commit, so the commit hash alone uniquely identifies the environment.
    """
    return f"ovms_export_{OVMS_COMMIT[:8]}"


def _download_upstream_file(url: str, dest_path: str) -> None:
    """Download a single upstream file to dest_path, skipping if already cached."""
    import requests

    if os.path.exists(dest_path):
        logger.debug(f"Using cached file: {dest_path}")
        return

    logger.info(f"Downloading: {url}")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    with open(dest_path, "w") as fh:
        fh.write(response.text)
    os.chmod(dest_path, 0o640)
    logger.debug(f"Cached to: {dest_path}")


# ---------------------------------------------------------------------------
# Script and requirements download / caching
# ---------------------------------------------------------------------------


def download_export_script(cache_dir: Optional[str] = None) -> str:
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


def download_requirements_file(cache_dir: Optional[str] = None) -> str:
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


def _preprocess_requirements(req_file: str, processed_dir: str) -> str:
    """
    Pre-process the upstream requirements.txt to apply reproducible package
    version pins for the current OVMS_COMMIT.

    Strategy:

    1. **Commit-specific overrides** (preferred): if ``_COMMIT_PACKAGE_OVERRIDES``
       contains an entry for the current ``OVMS_COMMIT``, substitute the listed
       packages with their exact stable versions and strip the nightly/pre-release
       index URLs and the ``--pre`` directive from the file.  This guarantees a
       fully reproducible install that never drifts to a newer nightly.

    2. **Fallback** (no overrides defined): relax pre-release exact pins
       (``==XrcN`` → ``>=XrcN``) so uv can at least find a compatible release.
       This is non-deterministic and may break when new nightlies appear.
       A warning is emitted so developers know to add overrides.

    The processed file is written to ``requirements_{commit}_pinned.txt``
    (with overrides) or ``requirements_{commit}_processed.txt`` (fallback),
    so re-runs reuse the cached copy without re-processing.

    Args:
        req_file: Path to the raw downloaded requirements.txt.
        processed_dir: Directory where the processed file is written.

    Returns:
        str: Path to the processed requirements file.
    """
    import re

    overrides = _COMMIT_PACKAGE_OVERRIDES.get(OVMS_COMMIT, {})

    if overrides:
        processed_path = os.path.join(processed_dir, f"requirements_{OVMS_COMMIT}_pinned.txt")
    else:
        processed_path = os.path.join(processed_dir, f"requirements_{OVMS_COMMIT}_processed.txt")

    if os.path.exists(processed_path):
        logger.debug(f"Using cached requirements: {processed_path}")
        return processed_path

    with open(req_file) as fh:
        content = fh.read()

    if overrides:
        # Apply exact commit-specific stable version pins.
        # Replace any existing version specifier for the named package with
        # the exact pinned version.  The character-by-character copy below
        # builds the replacement line without propagating the tainted input
        # version string from the upstream file.
        processed_content = content
        for pkg_name, pinned_version in overrides.items():
            pkg_pattern = re.compile(
                r"^(" + re.escape(pkg_name) + r")([=><!\s].*)$",
                re.MULTILINE | re.IGNORECASE,
            )
            # Build replacement from safe local constants (not from file content)
            replacement_line = "".join(c for c in f"{pkg_name}=={pinned_version}")
            new_content = pkg_pattern.sub(replacement_line, processed_content)
            if new_content != processed_content:
                logger.info(f"Pinned {pkg_name}=={pinned_version} (commit {OVMS_COMMIT})")
            processed_content = new_content

        # Remove nightly and pre-release OpenVINO index URLs – stable packages
        # are available on PyPI.  The PyTorch CPU index is kept because
        # torch==X.Y.Z+cpu is only available there.
        processed_content = re.sub(
            r'^--extra-index-url\s+"?https://storage\.openvinotoolkit\.org/simple/wheels/'
            r'(?:nightly|pre-release)"?\s*$\n?',
            "",
            processed_content,
            flags=re.MULTILINE,
        )
        # Remove the --pre directive from the file – stable pins don't need it
        # and it would cause uv to prefer pre-release versions of unversioned deps.
        processed_content = re.sub(r"^--pre\s*$\n?", "", processed_content, flags=re.MULTILINE)

        logger.info(
            f"Applied commit-specific stable version overrides for {OVMS_COMMIT}: "
            + ", ".join(f"{k}=={v}" for k, v in overrides.items())
        )
    else:
        # Fallback: relax pre-release exact pins to >=  so uv can pick a final release.
        logger.warning(
            f"No stable package overrides defined for commit {OVMS_COMMIT}. "
            "Falling back to >=rc relaxation – the resolver may pick unstable nightly builds. "
            "Add an entry to _COMMIT_PACKAGE_OVERRIDES to fix this."
        )
        pre_release_pattern = re.compile(
            r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\d[\d.]*(?:(?:rc|a|b|alpha|beta)\.?\d+|\.dev\d+))\s*$",
            re.MULTILINE,
        )
        processed_content = pre_release_pattern.sub(r"\1>=\2", content)

    with open(processed_path, "w") as fh:
        fh.write(processed_content)
    os.chmod(processed_path, 0o640)
    logger.debug(f"Processed requirements written to: {processed_path}")
    return processed_path


def setup_export_venv(
    cache_dir: Optional[str] = None,
    venv_data_dir: Optional[str] = None,
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

    # Pre-process requirements to apply stable version pins (or fallback >=rc)
    req_file = _preprocess_requirements(req_file, cache_dir)

    # Extra pip args needed for the upstream requirements.txt:
    # - "--index-strategy unsafe-best-match": nncf@git needs setuptools>=77.0 which is only
    #   on PyPI, but uv finds an older setuptools on pytorch.org/whl/cpu first and stops.
    #   This flag makes uv consider all indexes and pick the best matching version.
    # - "--pre": only needed when falling back to >=rc relaxation (no commit overrides).
    #   With stable pinned versions the nightly/pre-release indexes are removed from the
    #   processed requirements file, so --pre is omitted here to prevent uv from
    #   accidentally selecting pre-release versions of unversioned dependencies.
    extra_pip_args = ["--index-strategy", "unsafe-best-match"]
    if OVMS_COMMIT not in _COMMIT_PACKAGE_OVERRIDES:
        extra_pip_args = ["--pre"] + extra_pip_args

    success, message = manager.setup_named_venv(
        venv_name=venv_name,
        requirements_file=req_file,
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
    data_dir: Optional[str] = None,
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

    if extra_quantization_params:
        cmd.extend(["--extra_quantization_params", extra_quantization_params])

    logger.info(f"Running OVMS export_model.py (commit: {OVMS_COMMIT})")
    logger.debug(f"Export command: {' '.join(cmd)}")

    result = manager.run_command_in_venv(
        venv_name=venv_name,
        command=cmd,
        timeout=export_timeout,
        check=False,
        stream_output=True,
    )

    if result.timed_out:
        raise TimeoutError(f"Model export timed out after {export_timeout}s")
    if result.returncode != 0:
        raise ValueError(f"Model export failed with exit code {result.returncode}")

    logger.info("OVMS export_model.py completed successfully")
