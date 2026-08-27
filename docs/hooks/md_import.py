# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MkDocs hook that expands ``@import "relative/path.md"`` directives.

Why this exists
----------------
Quick-start pages under ``docs/getting-started/`` reuse a set of shared
snippet files (``docs/includes/quick-start/*.md``) via a file-transclusion
directive. MkDocs needs some mechanism to expand that directive at build
time, but authors also open these ``.md`` files directly in VS Code for
single-page preview and Markdown-to-PDF export, which does not run through
MkDocs at all.

The VS Code "Markdown Preview Enhanced" extension has built-in, native
support for a file-import directive of the form::

    @import "relative/path.md"

(see https://shd101wyy.github.io/markdown-preview-enhanced/#/file-imports),
resolved relative to the importing file's own directory - the same
convention used for ordinary Markdown relative links/images.

This hook implements the exact same directive and resolution rule for
MkDocs, so a single ``@import "..."`` line renders correctly in both:

- ``mkdocs build`` / ``mkdocs serve`` (expanded here), and
- VS Code's single-page preview and PDF export (expanded natively by the
  Markdown Preview Enhanced extension).

Directive rules
----------------
The directive must be alone on its own line (optionally indented), e.g.::

    @import "../../../includes/quick-start/system-drivers.md"

If the referenced file cannot be found, the directive is left untouched
(rather than silently dropped) so a broken reference stays visible in the
rendered output instead of disappearing without a trace.
"""

from __future__ import annotations

import re
from pathlib import Path

_IMPORT_RE = re.compile(r'^[ \t]*@import\s+"([^"]+)"[ \t]*$', re.MULTILINE)

# Guards against runaway/circular @import chains.
_MAX_DEPTH = 5


def _expand(markdown: str, base_dir: Path, depth: int = 0) -> str:
    if depth > _MAX_DEPTH:
        return markdown

    def _replace(match: re.Match[str]) -> str:
        target = (base_dir / match.group(1)).resolve()
        if not target.is_file():
            return match.group(0)
        content = target.read_text(encoding="utf-8")
        return _expand(content, target.parent, depth + 1)

    return _IMPORT_RE.sub(_replace, markdown)


def on_page_markdown(markdown, page, config, files, **kwargs):
    """MkDocs hook entry point: expand @import directives before parsing."""
    base_dir = Path(page.file.abs_src_path).parent
    return _expand(markdown, base_dir)
