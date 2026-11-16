# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MkDocs hook that generates profile tables from the profile YAML files.

Why this exists
----------------
Each test suite page under ``docs/suites/`` lists the profiles
that run the suite, and the profile pages under ``docs/profiles/`` list the
profiles of each type with the suites they run. Written by hand, these lists go
stale whenever a profile is added, renamed, or changed. This hook reads the
shipped profile YAML files at build time, so the lists always match the code.

Directives
----------
Each directive is an HTML comment alone on its own line, so it stays invisible
in a plain Markdown preview::

    <!-- esq:suite-profiles test_dlstreamer -->

expands to the profiles that use the test suite ``test_dlstreamer``, each
linked to its YAML file, followed by their test cases. A page with this
directive is the documentation page of that test suite. Several test names
can be given, separated by spaces, for a page that covers related tests::

    <!-- esq:profiles -->

expands to a table of all profiles, with links to the pages of the test suites
they run. Add a profile type to list only the profiles of that type::

    <!-- esq:profiles vertical -->
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import yaml

log = logging.getLogger("mkdocs.hooks.esq_profiles")

_SUITE_RE = re.compile(r"^[ \t]*<!--\s*esq:suite-profiles\s+([\w\s]+?)\s*-->[ \t]*$", re.MULTILINE)
_PROFILES_RE = re.compile(r"^[ \t]*<!--\s*esq:profiles(?:[ \t]+(\w+))?\s*-->[ \t]*$", re.MULTILINE)

_TYPE_ORDER = {"qualification": 0, "vertical": 1, "horizontal": 2}

_profiles: list[dict] = []
_suite_pages: dict[str, str] = {}
_repo_url = ""


def _load_profiles(repo_root: Path) -> list[dict]:
    """Read the shipped profiles. Example profiles and profiles with the ``docs_exclude`` label are skipped."""
    profiles = []
    for path in sorted(repo_root.glob("src/*/configs/profiles/**/*.yml")):
        if "examples" in path.parts:
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as err:
            log.warning("Cannot parse profile %s: %s", path, err)
            continue
        if not data.get("name"):
            continue
        labels = (data.get("params") or {}).get("labels") or {}
        if labels.get("docs_exclude"):
            continue
        tests: dict[str, list[tuple[str, str]]] = {}
        for suite in data.get("suites") or []:
            for sub_suite in suite.get("sub_suites") or []:
                for test_name, test in (sub_suite.get("tests") or {}).items():
                    cases = tests.setdefault(test_name, [])
                    for entry in (test or {}).get("params") or []:
                        if not isinstance(entry, dict):
                            continue
                        case = (str(entry.get("test_id", "")), str(entry.get("display_name", "")))
                        if case not in cases:
                            cases.append(case)
        profiles.append(
            {
                "name": data["name"],
                "description": str(data.get("description", "")).strip(),
                "type": str(labels.get("type", path.parent.name)).lower(),
                "path": path.relative_to(repo_root).as_posix(),
                "tests": tests,
            }
        )
    profiles.sort(key=lambda p: (_TYPE_ORDER.get(p["type"], 9), p["name"]))
    return profiles


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _profile_link(profile: dict) -> str:
    return f"[`{profile['name']}`]({_repo_url}/blob/main/{profile['path']})"


def _render_suite_profiles(test_names: list[str]) -> str:
    users = [p for p in _profiles if any(t in p["tests"] for t in test_names)]
    if not users:
        return "_No shipped profile uses this test suite yet._\n"
    lines = ["| Profile | Type | Test cases |", "|---------|------|------------|"]
    for profile in users:
        count = sum(len(profile["tests"].get(t, [])) for t in test_names)
        lines.append(f"| {_profile_link(profile)} | {profile['type'].capitalize()} | {count} |")
    lines.append("")
    lines += ["Expand a profile to see the test cases it runs with this suite.", ""]
    for profile in users:
        cases = [case for t in test_names for case in profile["tests"].get(t, [])]
        lines += [
            '<details markdown="1">',
            f"<summary><code>{profile['name']}</code></summary>",
            "",
            "| Test ID | Test case |",
            "|---------|-----------|",
        ]
        lines += [f"| {_cell(test_id)} | {_cell(name)} |" for test_id, name in cases]
        lines += ["", "</details>", ""]
    return "\n".join(lines)


def _render_profiles(page_src: str, profile_type: str | None) -> str:
    lines = ["| Profile | Type | Description | Test suites |", "|---------|------|-------------|-------------|"]
    for profile in _profiles:
        if profile_type and profile["type"] != profile_type.lower():
            continue
        suites = []
        for test_name in profile["tests"]:
            target = _suite_pages.get(test_name)
            if target:
                rel = os.path.relpath(target, os.path.dirname(page_src)).replace(os.sep, "/")
                suites.append(f"[`{test_name}`]({rel})")
            else:
                suites.append(f"`{test_name}`")
        lines.append(
            f"| {_profile_link(profile)} | {profile['type'].capitalize()} | "
            f"{_cell(profile['description'])} | {', '.join(suites)} |"
        )
    return "\n".join(lines) + "\n"


def on_config(config, **kwargs):
    global _profiles, _repo_url
    _repo_url = (config.get("repo_url") or "").rstrip("/")
    repo_root = Path(config["config_file_path"]).resolve().parent
    _profiles = _load_profiles(repo_root)
    return config


def on_files(files, config, **kwargs):
    """Find the page of each test suite from its suite-profiles directive."""
    _suite_pages.clear()
    for file in files:
        if not file.src_uri.endswith(".md") or not file.abs_src_path:
            continue
        text = Path(file.abs_src_path).read_text(encoding="utf-8")
        for match in _SUITE_RE.finditer(text):
            for test_name in match.group(1).split():
                _suite_pages[test_name] = file.src_uri
    used = {t for p in _profiles for t in p["tests"]}
    for test_name in sorted(used - set(_suite_pages)):
        log.info("Test suite %s is used by a profile but has no documentation page", test_name)
    return files


def on_page_markdown(markdown, page, config, files, **kwargs):
    markdown = _SUITE_RE.sub(lambda m: _render_suite_profiles(m.group(1).split()), markdown)
    return _PROFILES_RE.sub(lambda m: _render_profiles(page.file.src_uri, m.group(1)), markdown)
