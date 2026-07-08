#!/usr/bin/env python3
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
"""
normalize_tags — apply canonical tag vocabulary from worklog/.lintrc.yaml.

Reads every markdown file under worklog/ + the top-level WORKLOG.md, parses
the YAML frontmatter's ``tags:`` list, applies the substitution and drop rules
from ``worklog/.lintrc.yaml``, deduplicates, and writes back.

Tag normalization is configured in the ``worklog/.lintrc.yaml`` under
``tag_substitutions``, ``tag_drops``, and ``tag_splits`` keys. If no lintrc
is found, the tool passes tags through unchanged (idempotent).

Idempotent: re-running on already-normalized files is a no-op.

Run:
    worklog-tags [--root DIR]            # rewrite files in-place
    worklog-tags [--root DIR] --check    # exit non-zero on any drift (pre-commit mode)
    worklog-tags [--root DIR] --dry-run  # print proposed changes without writing

Usage:
    worklog-tags [--root DIR] [--check] [--dry-run]
    python -m agent_team.worklog.tools.normalize_tags [--root DIR] [--check]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _lintrc import load_lintrc, resolve_root  # noqa: E402

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
TAGS_RE = re.compile(r"^tags:\s*\[(.*?)\]\s*$", re.MULTILINE)


def load_tag_config(cfg: dict) -> tuple[dict[str, str | None], dict[str, list[str]]]:
    """Return (substitution_map, split_map) from lintrc cfg.

    substitution_map: {old_tag: new_tag_or_None_to_drop}
    split_map: {compound_tag: [part1, part2, ...]}
    """
    subst: dict[str, str | None] = {}
    for old, new in (cfg.get("tag_substitutions") or {}).items():
        subst[old] = new if isinstance(new, str) else None
    for drop in cfg.get("tag_drops") or []:
        subst[drop] = None
    split: dict[str, list[str]] = {}
    for compound, parts in (cfg.get("tag_splits") or {}).items():
        if isinstance(parts, list):
            split[compound] = parts
    return subst, split


def normalize_tags(
    tags: list[str],
    subst: dict[str, str | None],
    split: dict[str, list[str]],
) -> list[str]:
    out: list[str] = []
    for t in tags:
        t = t.strip()
        if not t:
            continue
        if t in split:
            out.extend(split[t])
            continue
        if t in subst:
            replacement = subst[t]
            if replacement is None:
                continue  # dropped
            out.append(replacement)
        else:
            out.append(t)
    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def walk(root: Path) -> list[Path]:
    paths: list[Path] = []
    worklog = root / "worklog"
    if worklog.exists():
        for p in worklog.rglob("*.md"):
            paths.append(p)
    wl = root / "WORKLOG.md"
    if wl.exists():
        paths.append(wl)
    return sorted(paths)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=None,
        help=(
            "Repo root containing worklog/ and WORKLOG.md. "
            "Defaults to the current working directory (correct for pre-commit)."
        ),
    )
    ap.add_argument("--check", action="store_true", help="Exit non-zero if any file would change")
    ap.add_argument("--dry-run", action="store_true", help="Print proposed changes without writing")
    args = ap.parse_args()

    root = resolve_root(args.root)
    cfg = load_lintrc(root)
    subst, split = load_tag_config(cfg)

    changed_paths: list[tuple[Path, list[str], list[str]]] = []
    rewrites: dict[Path, str] = {}

    for p in walk(root):
        text = p.read_text()
        fm_match = FRONTMATTER_RE.match(text)
        if not fm_match:
            continue
        fm_body = fm_match.group(1)
        tags_match = TAGS_RE.search(fm_body)
        if not tags_match:
            continue
        old_tags = [t.strip() for t in tags_match.group(1).split(",") if t.strip()]
        new_tags = normalize_tags(old_tags, subst, split)
        if old_tags == new_tags:
            continue
        new_tags_str = "tags: [" + ", ".join(new_tags) + "]"
        new_fm_body = fm_body[: tags_match.start()] + new_tags_str + fm_body[tags_match.end() :]
        new_text = "---\n" + new_fm_body + "\n---\n" + text[fm_match.end() :]
        changed_paths.append((p, old_tags, new_tags))
        rewrites[p] = new_text

    if args.check:
        if changed_paths:
            print(f"{len(changed_paths)} files have un-normalized tags:")
            for p, old, new in changed_paths:
                try:
                    display = p.relative_to(root)
                except ValueError:
                    display = p
                print(f"  {display}: {old} -> {new}")
            return 1
        print("All tags normalized.")
        return 0

    if args.dry_run:
        print(f"Would change {len(changed_paths)} files:")
        for p, old, new in changed_paths:
            try:
                display = p.relative_to(root)
            except ValueError:
                display = p
            print(f"  {display}:")
            print(f"    old: {old}")
            print(f"    new: {new}")
        return 0

    for p, new_text in rewrites.items():
        p.write_text(new_text)

    print(f"Rewrote {len(changed_paths)} files.")
    for p, old, new in changed_paths:
        try:
            display = p.relative_to(root)
        except ValueError:
            display = p
        print(f"  {display}: {old} -> {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
