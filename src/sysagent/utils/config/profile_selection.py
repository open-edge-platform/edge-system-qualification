# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Profile selections shared by ``run --profiles-file`` and ``run --smoke``.

A selection is a list of entries, each a profile name string (whole profile) or a mapping
``{name, test_ids}`` (only those test IDs). It is returned as
``(profile_names, per_profile_filters)``, the shape the profile executor takes.

Smoke files
-----------
Each sub-suite may ship a ``smoke.yml`` next to its tests. It lists the smoke tests: the
smallest set of tests that still shows the sub-suite works on every device it supports,
chosen so they finish quickly (smallest model, shortest duration). Entries are grouped by
test name::

    tests:
      test_text_generation:
        - name: profile.horizontal.ai.gen-llm
          test_ids: [GEN-LLM-001, GEN-LLM-002]

``run --smoke <suite>/<sub_suite>[/<test_name>]`` (or ``all``) runs them. The path is
resolved inside the installed packages, so no file-system path is needed.
"""

import logging
from pathlib import Path

import yaml

from .config import get_suite_directory
from .config_loader import discover_entrypoint_paths

logger = logging.getLogger(__name__)

SMOKE_FILE = "smoke.yml"
SMOKE_ALL = "all"


class ProfileSelection(tuple):
    """
    ``(profile_names, per_profile_filters)`` plus ``no_dependencies``.

    Unpacks like the plain two-item tuple callers already use; ``no_dependencies`` names the
    requested profiles whose ``depends_on`` profiles must not be added (entry
    ``dependencies: false``, and every smoke entry).
    """

    def __new__(cls, profile_names: list, per_profile_filters: dict, no_dependencies=()):
        selection = super().__new__(cls, (profile_names, per_profile_filters))
        selection.no_dependencies = frozenset(no_dependencies)
        return selection


def parse_profile_entries(raw_entries: list) -> ProfileSelection:
    """
    Turn selection entries into ``(profile_names, per_profile_filters)``.

    Each entry is a profile name string or a mapping ``{name, test_ids, dependencies}``. The first
    entry for a profile wins; profile names keep their order. ``dependencies: false`` runs the
    profile without the profiles in its ``depends_on``.
    """
    profile_names = []
    per_profile_filters = {}
    no_dependencies = set()
    seen = set()
    for entry in raw_entries or []:
        if isinstance(entry, str):
            name, test_ids, with_deps = entry.strip(), None, True
        elif isinstance(entry, dict):
            name, test_ids = str(entry.get("name", "")).strip(), entry.get("test_ids")
            with_deps = entry.get("dependencies", True) is not False
        else:
            continue

        if not name or name in seen:
            continue
        seen.add(name)
        profile_names.append(name)
        if not with_deps:
            no_dependencies.add(name)

        if test_ids:
            cleaned_ids = sorted({str(t).strip() for t in test_ids if str(t).strip()})
            if cleaned_ids:
                per_profile_filters[name] = {"test_id": cleaned_ids}
    return ProfileSelection(profile_names, per_profile_filters, no_dependencies)


def _read_smoke_file(path: Path) -> dict:
    """Return the ``tests`` mapping of a smoke file, or None when it is unreadable or malformed."""
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.error(f"Failed to read smoke file '{path}': {e}")
        return None
    tests = data.get("tests") if isinstance(data, dict) else None
    if not isinstance(tests, dict):
        logger.error(f"Invalid smoke file '{path}': expected a 'tests' mapping of test name to entries")
        return None
    return tests


def _smoke_targets(smoke: str) -> list:
    """Resolve a --smoke value to ``[(smoke_file, test_name or None)]``, or None on error."""
    value = smoke.strip().strip("/").replace("\\", "/")
    if value == SMOKE_ALL:
        files = sorted(
            {f for base in discover_entrypoint_paths("suites") if base.is_dir() for f in base.glob(f"*/*/{SMOKE_FILE}")}
        )
        if not files:
            logger.error("No smoke files found in the installed test suites")
            return None
        return [(f, None) for f in files]

    parts = [p for p in value.split("/") if p]
    if len(parts) not in (2, 3):
        logger.error(f"Invalid --smoke value '{smoke}': use <suite>/<sub_suite>[/<test_name>] or '{SMOKE_ALL}'")
        return None
    suite_dir = get_suite_directory(parts[0])
    if not suite_dir:
        logger.error(f"Suite not found: {parts[0]}")
        return None
    sub_suite_dir = Path(suite_dir) / parts[1]
    if not sub_suite_dir.is_dir():
        logger.error(f"Sub-suite not found: {parts[0]}/{parts[1]}")
        return None
    smoke_file = sub_suite_dir / SMOKE_FILE
    if not smoke_file.is_file():
        logger.error(f"No {SMOKE_FILE} in {parts[0]}/{parts[1]}; run one of its profiles instead (see 'list')")
        return None
    test_name = parts[2].removesuffix(".py") if len(parts) == 3 else None
    return [(smoke_file, test_name)]


def load_smoke_selection(smoke: str) -> tuple:
    """
    Load the smoke tests for a --smoke value as ``(profile_names, per_profile_filters)``.

    Entries for the same profile are merged: test IDs are combined, and an entry without
    test IDs (whole profile) wins. Smoke runs never add the ``depends_on`` profiles: those
    exist to fill a vertical report, not to make the smoke tests work. Returns
    ``(None, None)`` on error.
    """
    targets = _smoke_targets(smoke)
    if targets is None:
        return None, None

    merged = {}  # profile name -> set of test IDs, or None for the whole profile
    for smoke_file, test_name in targets:
        tests = _read_smoke_file(smoke_file)
        if tests is None:
            return None, None
        if test_name and test_name not in tests:
            listed = ", ".join(sorted(tests)) or "none"
            logger.error(f"{test_name} is not listed in {smoke_file} (listed: {listed})")
            return None, None
        for name in [test_name] if test_name else sorted(tests):
            for entry in tests.get(name) or []:
                if not isinstance(entry, dict) or not entry.get("name"):
                    continue
                profile = str(entry["name"]).strip()
                ids = entry.get("test_ids")
                if profile in merged and merged[profile] is None:
                    continue
                merged[profile] = None if not ids else (merged.get(profile) or set()) | {str(i) for i in ids}

    if not merged:
        logger.error(f"No smoke tests listed for '{smoke}'")
        return None, None
    entries = [
        {"name": p, "test_ids": sorted(ids) if ids else None, "dependencies": False}
        for p, ids in sorted(merged.items())
    ]
    return parse_profile_entries(entries)
