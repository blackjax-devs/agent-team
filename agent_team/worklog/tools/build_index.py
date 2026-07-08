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
build_index — machine-build worklog/INDEX.md from frontmatter.

Walks all worklog/*.md files (plus the top-level WORKLOG.md), reads their
YAML frontmatter, and emits ``worklog/INDEX.md`` containing:

  * Recent changes — last 30 days, sorted by date desc
  * By status — SUPERSEDED, PARTIALLY_SUPERSEDED, DEFERRED, DRAFT sections
  * By tag — all CURRENT files grouped by tag, sorted by frequency
  * By model — case-studies grouped by model (when ``model:`` field is set)
  * Orphans — files no other file's ``related:`` references
  * Broken cross-references

Run:
    worklog-index [--root DIR]          # auto-fix (pre-commit default)
    worklog-index [--root DIR] --check  # read-only drift detection

Default (no flags): writes INDEX.md if changed, then exits 1 so the pre-commit
hook fails and the user can ``git add worklog/INDEX.md`` and re-commit. Exits 0
if already up to date (formatter/black pattern). ``--check``: read-only mode.

Usage:
    worklog-index [--root DIR] [--check]
    python -m agent_team.worklog.tools.build_index [--root DIR] [--check]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _frontmatter import walk_worklog  # noqa: E402
from _lintrc import get_sibling_prefixes, load_lintrc, resolve_root  # noqa: E402

WINDOW_DAYS = 30


def related_target_exists(r: str, root: Path, sibling_prefixes: tuple[str, ...]) -> bool:
    """A ``related:`` target is valid if URL, sibling-repo path, or resolves on disk.

    Mirrors lint_worklog.related_target_exists — both must agree on the resolution
    rule so the lint and build_index 'Broken cross-references' section agree on
    what counts as broken.
    """
    if r.startswith(("http://", "https://")):
        return True
    if sibling_prefixes and r.startswith(sibling_prefixes):
        return True
    return (root / r).exists()


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def parse_date(s: str | None) -> _dt.date | None:
    if not s:
        return None
    try:
        return _dt.date.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def build_index_text(
    files: list[tuple[Path, dict]], root: Path, sibling_prefixes: tuple[str, ...]
) -> str:
    # Anchor the recent-window on the NEWEST frontmatter date in the corpus, not
    # wall-clock date.today(). Otherwise the generated INDEX (and the --check drift
    # result) depends on the RUN date: a committed INDEX generated on day X goes stale
    # the moment CI runs on day X+n. Anchoring on the corpus makes build_index a pure
    # function of the committed files. Fallback to today() only for an empty corpus.
    _dates = [d for d in (parse_date(fm.get("date")) for _, fm in files) if d]
    anchor = max(_dates) if _dates else _dt.date.today()
    cutoff = anchor - _dt.timedelta(days=WINDOW_DAYS)

    by_path = {rel(p, root): (p, fm) for p, fm in files}

    # ---- collect cross-reference graph ----
    inbound: dict[str, set[str]] = defaultdict(set)
    for p, fm in files:
        for r in fm.get("related", []) or []:
            inbound[r].add(rel(p, root))

    # ---- categorise ----
    recent: list[tuple[Path, dict]] = []
    by_status: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    by_tag: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    by_model: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    orphans: list[tuple[Path, dict]] = []
    broken_related: list[tuple[Path, str]] = []
    broken_supersession: list[tuple[Path, str, str]] = []

    for p, fm in files:
        status = (fm.get("status") or "").strip()
        date = parse_date(fm.get("date"))
        model = fm.get("model")
        tags = fm.get("tags") or []

        if date and date >= cutoff:
            recent.append((p, fm))

        if status and status != "CURRENT":
            by_status[status].append((p, fm))

        if status == "CURRENT":
            for t in tags:
                by_tag[t].append((p, fm))

        if isinstance(model, str) and model not in ("null", "None"):
            by_model[model].append((p, fm))

        if rel(p, root) not in inbound:
            orphans.append((p, fm))

        for r in fm.get("related", []) or []:
            if not related_target_exists(r, root, sibling_prefixes):
                broken_related.append((p, r))

        for s in fm.get("supersedes", []) or []:
            target = by_path.get(s)
            if target is None:
                broken_supersession.append((p, s, "missing"))
                continue
            target_status = (target[1].get("status") or "").strip()
            if target_status not in {"SUPERSEDED", "PARTIALLY_SUPERSEDED"}:
                broken_supersession.append(
                    (
                        p,
                        s,
                        f"target status is {target_status!r}, "
                        f"expected SUPERSEDED/PARTIALLY_SUPERSEDED",
                    )
                )

    # ---- render ----
    lines: list[str] = []
    lines.append("# Worklog Index")
    lines.append("")
    lines.append(
        "*Machine-built by `worklog-index` (agent-team) — "
        "do not edit by hand; re-run after worklog/ changes.*"
    )
    lines.append("")
    lines.append(
        f"**Files indexed**: {len(files)}  |  "
        f"**Recent window**: last {WINDOW_DAYS} days  |  "
        f"**Top-level WORKLOG.md**: see [`../WORKLOG.md`](../WORKLOG.md) for the active dashboard."
    )
    lines.append("")

    # Recent
    lines.append("## Recent changes (last 30 days)")
    lines.append("")
    if not recent:
        lines.append("*(no files modified in the last 30 days)*")
    else:
        for p, fm in sorted(
            recent,
            key=lambda x: (parse_date(x[1].get("date")) or _dt.date.min, rel(x[0], root)),
            reverse=True,
        ):
            date = fm.get("date", "?")
            tags = ", ".join(fm.get("tags") or []) or "-"
            status = fm.get("status", "?")
            r = rel(p, root)
            lines.append(
                f"- `{date}`  [`{r}`]({r.replace('worklog/', '')}) "
                f"  · status={status} · tags=[{tags}]"
            )
    lines.append("")

    # By status
    lines.append("## By status (non-CURRENT)")
    lines.append("")
    for status in ("SUPERSEDED", "PARTIALLY_SUPERSEDED", "DEFERRED", "DRAFT"):
        items = by_status.get(status, [])
        if not items:
            continue
        lines.append(f"### {status} ({len(items)})")
        lines.append("")
        for p, fm in sorted(items, key=lambda x: rel(x[0], root)):
            sups = ", ".join(fm.get("supersedes", []) or []) or "-"
            r = rel(p, root)
            lines.append(f"- [`{r}`]({r.replace('worklog/', '')})  · supersedes=[{sups}]")
        lines.append("")

    # By tag
    lines.append("## By tag (CURRENT only)")
    lines.append("")
    lines.append("Tags sorted by frequency (descending).")
    lines.append("")
    sorted_tags = sorted(by_tag.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for tag, items in sorted_tags:
        lines.append(f"### `{tag}` ({len(items)})")
        lines.append("")
        for p, fm in sorted(items, key=lambda x: rel(x[0], root)):
            other_tags = [t for t in (fm.get("tags") or []) if t != tag]
            extras = (
                (" · also tagged: " + ", ".join(f"`{t}`" for t in other_tags))
                if other_tags
                else ""
            )
            r = rel(p, root)
            lines.append(f"- [`{r}`]({r.replace('worklog/', '')}){extras}")
        lines.append("")

    # By model
    if by_model:
        lines.append("## By model")
        lines.append("")
        for model, items in sorted(by_model.items()):
            lines.append(f"### `{model}` ({len(items)})")
            lines.append("")
            for p, fm in sorted(items, key=lambda x: rel(x[0], root)):
                r = rel(p, root)
                lines.append(f"- [`{r}`]({r.replace('worklog/', '')})")
            lines.append("")

    # Orphans
    lines.append("## Orphans — no inbound `related:` references")
    lines.append("")
    lines.append(
        "Files no other file points at. Some are legitimately top-level "
        "(WORKLOG.md, READMEs); others may indicate discoverability gaps."
    )
    lines.append("")
    actionable_orphans = [
        (p, fm)
        for p, fm in orphans
        if p.name not in {"WORKLOG.md", "README.md", "INDEX.md"}
    ]
    if not actionable_orphans:
        lines.append("*(no actionable orphans)*")
    else:
        for p, _ in sorted(actionable_orphans, key=lambda x: rel(x[0], root)):
            r = rel(p, root)
            lines.append(f"- [`{r}`]({r.replace('worklog/', '')})")
    lines.append("")

    # Broken cross-references
    lines.append("## Broken cross-references")
    lines.append("")
    if not broken_related and not broken_supersession:
        lines.append("*(none — all `related:` and `supersedes:` paths resolve cleanly)*")
    else:
        if broken_related:
            lines.append("### Broken `related:` (target doesn't exist)")
            lines.append("")
            for p, target in broken_related:
                lines.append(f"- `{rel(p, root)}` -> missing `{target}`")
            lines.append("")
        if broken_supersession:
            lines.append("### Broken `supersedes:` (target missing or wrong status)")
            lines.append("")
            for p, target, reason in broken_supersession:
                lines.append(f"- `{rel(p, root)}` -> `{target}` ({reason})")
            lines.append("")

    return "\n".join(lines) + "\n"


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
    ap.add_argument(
        "--check",
        action="store_true",
        help=(
            "Read-only mode: exit non-zero if INDEX.md is stale without modifying it. "
            "The pre-commit hook uses default (no flag) auto-fix mode."
        ),
    )
    args = ap.parse_args()

    root = resolve_root(args.root)
    cfg = load_lintrc(root)
    sibling_prefixes = get_sibling_prefixes(cfg)
    index_path = root / "worklog" / "INDEX.md"

    files = walk_worklog(root)
    new_text = build_index_text(files, root, sibling_prefixes)

    if args.check:
        if not index_path.exists():
            print(f"INDEX.md missing at {index_path}; run without --check to create it.")
            return 1
        on_disk = index_path.read_text()
        if on_disk != new_text:
            print("INDEX.md is STALE — re-run `worklog-index` to refresh.")
            return 1
        print(f"INDEX.md is up to date ({len(files)} files indexed).")
        return 0

    on_disk = index_path.read_text() if index_path.exists() else None
    if on_disk == new_text:
        print(f"INDEX.md is up to date ({len(files)} files indexed).")
        return 0

    index_path.write_text(new_text)
    print(
        f"INDEX.md was out of date — updated ({len(files)} files indexed).\n"
        f"Please `git add worklog/INDEX.md` and re-commit."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
