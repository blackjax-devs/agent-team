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
lint_worklog — schema-enforcement linter for worklog frontmatter.

Validates every file under worklog/ + WORKLOG.md:

  * Has YAML frontmatter (required fields: status, date, tags, model, author,
    supersedes, related).
  * ``status`` is one of the valid enum values.
  * ``date`` parses as YYYY-MM-DD.
  * ``tags`` is a list.
  * ``related:`` paths exist on disk or are configured as sibling-repo refs.
  * ``supersedes:`` paths exist AND target has SUPERSEDED/PARTIALLY_SUPERSEDED status.
  * Case-studies files appear in their per-model README (orphan detection).

Per-org configuration lives in ``worklog/.lintrc.yaml`` (see _lintrc.py for schema).

Exit codes:  0 = all checks pass; 1 = one or more violations.

Usage:
    worklog-lint [--root DIR] [--verbose]
    python -m agent_team.worklog.tools.lint_worklog [--root DIR] [--verbose]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

# Allow running as a script without installation
sys.path.insert(0, str(Path(__file__).parent))
from _frontmatter import walk_worklog  # noqa: E402
from _lintrc import get_exempt_paths, get_sibling_prefixes, load_lintrc, resolve_root  # noqa: E402


REQUIRED_FIELDS = ["status", "date", "tags", "model", "author", "supersedes", "related"]
VALID_STATUSES = {"CURRENT", "CLOSED", "SUPERSEDED", "PARTIALLY_SUPERSEDED", "DEFERRED", "DRAFT"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def related_target_exists(r: str, root: Path, sibling_prefixes: tuple[str, ...]) -> bool:
    """A ``related:`` entry is valid if it's a URL, a sibling-repo path, or resolves on disk.

    Resolution order:
      * URLs (http://, https://) are accepted unchecked.
      * Paths that match a configured sibling_repo_prefix are accepted as structural
        references without a filesystem check (the sibling repos may not be
        checked out alongside the consuming org's private config in CI).
      * Paths are checked relative to ``root``.
    """
    if r.startswith(("http://", "https://")):
        return True
    if sibling_prefixes and r.startswith(sibling_prefixes):
        return True
    return (root / r).exists()


def rel(p: Path, root: Path) -> str:
    return str(p.relative_to(root))


def lint_file(
    path: Path,
    fm: dict | None,
    all_files: dict[str, dict],
    root: Path,
    sibling_prefixes: tuple[str, ...],
) -> list[str]:
    """Return list of error strings for this file. Empty = clean."""
    errors: list[str] = []
    pfx = f"{rel(path, root)}: "

    if fm is None:
        errors.append(pfx + "missing YAML frontmatter")
        return errors

    for field in REQUIRED_FIELDS:
        if field not in fm:
            errors.append(pfx + f"missing required field {field!r}")

    status = fm.get("status")
    if status and status not in VALID_STATUSES:
        errors.append(
            pfx + f"invalid status {status!r}; expected one of {sorted(VALID_STATUSES)}"
        )

    date_val = fm.get("date")
    if date_val and not DATE_RE.match(str(date_val)):
        errors.append(pfx + f"date {date_val!r} doesn't match YYYY-MM-DD")
    elif date_val:
        try:
            _dt.date.fromisoformat(date_val)
        except (TypeError, ValueError):
            errors.append(pfx + f"date {date_val!r} is malformed")

    tags = fm.get("tags")
    if tags is not None and not isinstance(tags, list):
        errors.append(pfx + f"tags must be a list (got {type(tags).__name__})")

    for r in fm.get("related") or []:
        if not related_target_exists(r, root, sibling_prefixes):
            errors.append(pfx + f"related: target {r!r} does not exist")

    for s in fm.get("supersedes") or []:
        if s not in all_files:
            errors.append(pfx + f"supersedes: target {s!r} not found in worklog")
            continue
        target_status = all_files[s].get("status", "")
        if target_status not in {"SUPERSEDED", "PARTIALLY_SUPERSEDED"}:
            errors.append(
                pfx
                + f"supersedes: target {s!r} has status {target_status!r}, "
                f"expected SUPERSEDED or PARTIALLY_SUPERSEDED "
                f"(when X supersedes Y, Y's status must reflect that)"
            )

    return errors


def lint_case_study_inclusion(files: list[tuple[Path, dict]], root: Path) -> list[str]:
    """Every case-studies file should appear in its per-model README.

    The check is loose: README must mention the file's filename anywhere in the body.
    """
    errors: list[str] = []
    case_study_files = [
        (p, fm)
        for p, fm in files
        if "case-studies" in p.parts and p.name not in {"README.md", "INDEX.md"}
    ]
    for path, _ in case_study_files:
        readme = path.parent / "README.md"
        if not readme.exists():
            errors.append(f"{rel(path, root)}: per-model README {rel(readme, root)} missing")
            continue
        if path.name not in readme.read_text():
            errors.append(
                f"{rel(path, root)}: not mentioned in {rel(readme, root)} "
                "(per-model README should index every case study)"
            )
    return errors


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
    ap.add_argument("--verbose", action="store_true", help="Print each file as it's checked")
    args = ap.parse_args()

    root = resolve_root(args.root)
    cfg = load_lintrc(root)
    exempt_paths = get_exempt_paths(root, cfg)
    sibling_prefixes = get_sibling_prefixes(cfg)

    files = walk_worklog(root)
    all_files = {rel(p, root): fm for p, fm in files}

    candidate_paths: list[Path] = []
    top = root / "WORKLOG.md"
    if top.exists():
        candidate_paths.append(top)
    worklog_dir = root / "worklog"
    if worklog_dir.exists():
        for p in sorted(worklog_dir.rglob("*.md")):
            if p in exempt_paths:
                continue
            if p.stem.startswith("_"):
                continue
            candidate_paths.append(p)

    indexed_paths = {p for p, _ in files}
    all_errors: list[str] = []
    for p in candidate_paths:
        if p not in indexed_paths and p not in exempt_paths:
            all_errors.append(
                f"{rel(p, root)}: missing YAML frontmatter (back-fill incomplete)"
            )

    for path, fm in files:
        errs = lint_file(path, fm, all_files, root, sibling_prefixes)
        all_errors.extend(errs)
        if args.verbose and not errs:
            print(f"  ok  {rel(path, root)}")

    case_study_errs = lint_case_study_inclusion(files, root)
    all_errors.extend(case_study_errs)

    if all_errors:
        print(f"\nLint errors ({len(all_errors)}):\n")
        for e in all_errors:
            print(f"  x {e}")
        print(f"\n{len(all_errors)} error(s) — fix or add exempt_paths to worklog/.lintrc.yaml.")
        return 1

    print(f"All {len(files)} worklog files pass schema lint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
