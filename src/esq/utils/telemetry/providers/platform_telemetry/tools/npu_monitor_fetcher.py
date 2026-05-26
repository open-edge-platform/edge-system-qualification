# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime fetcher for the upstream Intel npu-monitor-tool.

The upstream tool lives under ``tools/npu-monitor-tool/`` of the
``open-edge-platform/edge-ai-libraries`` repository and is the canonical
reference implementation for Intel NPU telemetry collection.  Rather
than vendoring a copy we sparse-clone it into a user-scoped cache
directory on first use and refresh on demand.

Public API:

* ``ensure_tool(...)`` — return the absolute path to
  ``npu-monitor-tool.py``, performing a sparse ``git clone`` into the
  cache directory if needed.  Returns ``None`` (and logs at DEBUG) when
  the clone cannot be completed (no network, ``git`` missing, etc.).

The fetcher is intentionally **idempotent** and **non-fatal**: any
failure simply makes the upstream-tool path unavailable, allowing the
caller to fall back to other NPU collectors.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess  # nosec B404 # For runtime git sparse-clone of npu-monitor-tool
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_URL = "https://github.com/open-edge-platform/edge-ai-libraries.git"
_SPARSE_PATH = "tools/npu-monitor-tool"
_TOOL_FILENAME = "npu-monitor-tool.py"

# ESQ FW guideline: all runtime downloads live under the test data root
# (``esq_data/data/...``) so they are co-located with models, results and
# other suite artefacts and get cleaned up by ``esq clean``.  The exact
# data root is provided by the runner via ``CORE_DATA_DIR``; when that is
# not set (e.g. running unit tests outside the CLI) we fall back to a
# ``<cwd>/esq_data/data`` location to keep the layout consistent.
_RUNTIME_SUBDIR = ("system", "platform_telemetry", "npu-monitor-tool")


def _resolve_data_root() -> Path:
    override = os.environ.get("ESQ_PLATFORM_TELEMETRY_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()

    core_data_dir = os.environ.get("CORE_DATA_DIR")
    if core_data_dir:
        # CORE_DATA_DIR points at <project>_data (e.g. esq_data); runtime
        # artefacts live under its ``data/`` subtree per FW convention.
        return Path(core_data_dir).expanduser().resolve() / "data"

    return (Path.cwd() / "esq_data" / "data").resolve()


def _tool_dir() -> Path:
    return _resolve_data_root().joinpath(*_RUNTIME_SUBDIR)


def _have_git() -> bool:
    return shutil.which("git") is not None


def _run_git(args: list[str], cwd: Path, timeout: float) -> bool:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("npu-monitor-tool: git %s failed: %s", " ".join(args), exc)
        return False
    if proc.returncode != 0:
        logger.debug(
            "npu-monitor-tool: git %s exited %d: %s",
            " ".join(args),
            proc.returncode,
            (proc.stderr or proc.stdout or "").strip().splitlines()[-1:],
        )
        return False
    return True


def _sparse_clone(target: Path, timeout_s: float) -> bool:
    """Perform a sparse-checkout clone of just the npu-monitor-tool folder."""
    target.mkdir(parents=True, exist_ok=True)

    # Initialise an empty repo with sparse-checkout enabled.
    if not _run_git(["init"], cwd=target, timeout=timeout_s):
        return False
    if not _run_git(["remote", "add", "origin", _REPO_URL], cwd=target, timeout=timeout_s):
        # remote add fails idempotently when re-running; ignore in that case
        pass
    info_dir = target / ".git" / "info"
    try:
        info_dir.mkdir(parents=True, exist_ok=True)
        (info_dir / "sparse-checkout").write_text(f"{_SPARSE_PATH}/\n", encoding="utf-8")
    except OSError as exc:
        logger.debug("npu-monitor-tool: sparse-checkout setup failed: %s", exc)
        return False
    if not _run_git(["config", "core.sparseCheckout", "true"], cwd=target, timeout=timeout_s):
        return False
    if not _run_git(
        ["fetch", "--depth=1", "origin", "main"], cwd=target, timeout=timeout_s
    ):
        return False
    if not _run_git(["checkout", "main"], cwd=target, timeout=timeout_s):
        return False
    return True


def _resolve_tool_path(target: Path) -> Optional[Path]:
    """Locate npu-monitor-tool.py inside a freshly-cloned sparse tree."""
    candidates = [
        target / _SPARSE_PATH / _TOOL_FILENAME,
        target / _TOOL_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def ensure_tool(timeout_s: float = 30.0) -> Optional[Path]:
    """Return path to ``npu-monitor-tool.py``, cloning on first use.

    Returns ``None`` if cloning is not possible (no ``git`` on PATH,
    network unreachable, etc.).
    """
    target = _tool_dir()
    existing = _resolve_tool_path(target)
    if existing is not None:
        return existing

    if not _have_git():
        logger.debug("npu-monitor-tool: 'git' not on PATH; cannot fetch upstream tool")
        return None

    # If a previous attempt left a partial directory behind, clear it
    # before re-cloning so init/remote/fetch start from a clean slate.
    if target.exists():
        try:
            shutil.rmtree(target)
        except OSError as exc:
            logger.debug("npu-monitor-tool: cannot clear stale cache %s: %s", target, exc)
            return None

    logger.info("[platform_telemetry] fetching npu-monitor-tool into %s ...", target)
    if not _sparse_clone(target, timeout_s=timeout_s):
        logger.info(
            "[platform_telemetry] npu-monitor-tool fetch failed; falling back to PMT collector"
        )
        return None

    resolved = _resolve_tool_path(target)
    if resolved is None:
        logger.debug(
            "npu-monitor-tool: clone succeeded but %s not found", _TOOL_FILENAME
        )
        return None

    # Restrict the cloned tool to owner+group rwx/rx; it runs only under
    # the test user account, world bits are not required.
    try:
        os.chmod(resolved, 0o750)
    except OSError:
        pass
    return resolved
