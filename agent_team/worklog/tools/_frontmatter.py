# Copyright 2026- blackjax-devs.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared frontmatter parser for build_index, lint_worklog, and normalize_tags.

Minimal YAML-subset parser — handles the schema we use (key: value strings,
key: [list, items], key:\\n  - item\\n  - item). No nested mappings.
Keeps the tools dependency-free (no PyYAML) for the frontmatter parsing layer.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


FM_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def parse_frontmatter(text: str) -> dict[str, Any] | None:
    """Parse YAML frontmatter from ``text``. Returns dict or None if absent.

    Supported value forms:
      key: scalar          (string, becomes Python string)
      key: null            (becomes Python None)
      key: []              (becomes empty list)
      key: [a, b, c]       (becomes list)
      key:                 (becomes list, parsed from following indented `- item` lines)
        - item1
        - item2
    """
    m = FM_RE.match(text)
    if not m:
        return None
    body = m.group(1)
    result: dict[str, Any] = {}
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "":
            # Block list follows
            items: list[str] = []
            j = i + 1
            while j < len(lines) and lines[j].startswith("  -"):
                item = lines[j][3:].strip()
                items.append(item)
                j += 1
            result[key] = items
            i = j
        elif val == "null":
            result[key] = None
            i += 1
        elif val == "[]":
            result[key] = []
            i += 1
        elif val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            if not inner:
                result[key] = []
            else:
                result[key] = [s.strip() for s in inner.split(",")]
            i += 1
        else:
            result[key] = val
            i += 1
    return result


def file_body(text: str) -> str:
    """Return the body text (everything after the frontmatter block)."""
    m = FM_RE.match(text)
    if not m:
        return text
    return text[m.end():]


def walk_worklog(repo_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Yield (path, frontmatter_dict) for every md file with valid frontmatter.

    Files whose stem starts with ``_`` are skipped — convention for template
    scaffolding (e.g. ``decisions/_template.md``) that should not be indexed
    or linted as real entries.
    """
    out: list[tuple[Path, dict[str, Any]]] = []
    worklog = repo_root / "worklog"
    candidates: list[Path] = []
    top = repo_root / "WORKLOG.md"
    if top.exists():
        candidates.append(top)
    if worklog.exists():
        for p in sorted(worklog.rglob("*.md")):
            if p.stem.startswith("_"):
                continue  # template / private scaffolding
            candidates.append(p)
    for path in candidates:
        text = path.read_text()
        fm = parse_frontmatter(text)
        if fm is not None:
            out.append((path, fm))
    return out
