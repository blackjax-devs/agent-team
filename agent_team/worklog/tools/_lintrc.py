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
"""Load per-org worklog hook configuration from worklog/.lintrc.yaml.

All sections are optional; missing keys return safe defaults.

Schema (worklog/.lintrc.yaml):

    # Paths relative to repo root that are exempt from frontmatter requirements.
    # Default: [worklog/README.md, worklog/INDEX.md]
    exempt_paths:
      - worklog/README.md
      - worklog/INDEX.md

    # Path prefixes treated as structural sibling-repo references in related:/supersedes:
    # fields. These paths are accepted without a filesystem existence check.
    # Default: [] (no prefixes; every related: path is checked on disk)
    sibling_repo_prefixes:
      - "org-repo/"
      - "another-repo/"

    # Tag normalization — applied by the worklog-tags hook.
    # All keys optional; if absent, the hook passes tags through unchanged.
    tag_substitutions:  # old-tag: new-tag
      old-name: new-name
    tag_drops:          # tags to remove entirely
      - obsolete-tag
    tag_splits:         # one tag -> multiple tags
      compound-tag:
        - part-a
        - part-b
"""
from __future__ import annotations

from pathlib import Path


def load_lintrc(root: Path) -> dict:
    """Load worklog/.lintrc.yaml; return empty dict if absent or unparseable."""
    lintrc_path = root / "worklog" / ".lintrc.yaml"
    if not lintrc_path.exists():
        return {}
    try:
        import yaml  # PyYAML — installed as a dep of agent-team
        data = yaml.safe_load(lintrc_path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        import sys
        print(f"Warning: could not parse {lintrc_path}: {exc}", file=sys.stderr)
        return {}


def get_exempt_paths(root: Path, cfg: dict) -> set[Path]:
    """Return the set of paths exempt from frontmatter requirements."""
    defaults = ["worklog/README.md", "worklog/INDEX.md"]
    raw = cfg.get("exempt_paths", defaults)
    return {root / p for p in raw}


def get_sibling_prefixes(cfg: dict) -> tuple[str, ...]:
    """Return tuple of path prefixes treated as sibling-repo structural refs."""
    raw = cfg.get("sibling_repo_prefixes", [])
    return tuple(raw)


def get_extra_search_roots(root: Path, cfg: dict) -> list[Path]:
    """Return additional directories to search when resolving ``related:`` paths.

    Useful when the worklog root is a subdirectory of a larger workspace and
    other files in that workspace are referenced in frontmatter.

    Paths in ``extra_search_roots`` are resolved relative to ``root`` (so ``..``
    means the parent directory). Example lintrc entry:

        extra_search_roots:
          - "../.."    # workspace root (two levels above --root)
    """
    raw = cfg.get("extra_search_roots", [])
    result: list[Path] = []
    for entry in raw:
        p = Path(entry)
        if not p.is_absolute():
            p = (root / p).resolve()
        result.append(p)
    return result


def resolve_root(root_arg: str | None) -> Path:
    """Resolve the repo root from a CLI arg or default to the current directory."""
    if root_arg is not None:
        p = Path(root_arg)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve()
    return Path.cwd()
