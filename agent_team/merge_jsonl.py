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
"""Chronologically read (and optionally merge) the team's jsonl audit log(s).

By default reads the deployment's audit log at ``$SAGENT_DATA_DIR/main.jsonl``
— the channel's operator posts + inter-agent ``sagent_send`` records, i.e. the
end-of-day source. Pass explicit paths to merge several logs chronologically.

Usage:
    agent-team-merge                           # the SAGENT_DATA_DIR audit log → stdout
    agent-team-merge --output merged.jsonl     # write to a file
    agent-team-merge --since 2026-06-01        # only records on/after a date
    agent-team-merge a/main.jsonl b/main.jsonl # merge explicit inputs

Each emitted record is the original ``{ts, from, to, body}`` plus a
``"source"`` field (the input's tag — its parent dir name, or stem) so
multi-input merges stay attributable. ``--source`` filters to one tag.

Designed for the end-of-day routine and ad-hoc forensics — fast, no external
deps, no in-memory cap beyond the OS file-open limit.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import argparse
import heapq
import json
import os
import sys


# Honor SAGENT_DATA_DIR (the same env the running server uses) so the default
# log path tracks the real audit log. This mirrors
# ``agent_team.mcp_sagent.delivery._resolve_data_dir``: env var if set, else the
# launch cwd (NOT the package install dir, which is read-only site-packages when
# installed).
_DATA_DIR = Path(
    os.environ.get("SAGENT_DATA_DIR", str(Path.cwd()))
).expanduser()
_DEFAULT_LOG = _DATA_DIR / "main.jsonl"


def _iter_records(path: Path, source_tag: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(ts, record)`` pairs from a jsonl file with the source stamped."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines silently (a few have historically crept
                # in from credit-balance error responses).
                continue
            if not isinstance(rec, dict) or "ts" not in rec:
                continue
            rec.setdefault("source", source_tag)
            yield rec["ts"], rec


def merge(
    sources: list[tuple[Path, str]],
    *,
    since: str | None = None,
    only_source: str | None = None,
) -> Iterator[dict[str, Any]]:
    """K-way merge across sources by ``ts``.

    ``ts`` is ISO-8601 with a uniform precision (``YYYY-MM-DDTHH:MM:SS.mmmZ``),
    so lexicographic string ordering matches temporal ordering — no
    parsing needed in the hot path.
    """
    streams = [_iter_records(p, tag) for p, tag in sources]
    for ts, rec in heapq.merge(*streams, key=lambda pair: pair[0]):
        if since is not None and ts < since:
            continue
        if only_source is not None and rec.get("source") != only_source:
            continue
        yield rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Explicit jsonl paths to merge (default: the SAGENT_DATA_DIR audit log).",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write merged stream to this path (default: stdout).",
    )
    ap.add_argument(
        "--since",
        help="Drop records with ts < this prefix (e.g. '2026-06-01').",
    )
    ap.add_argument(
        "--source",
        help="Filter to records from one source tag.",
    )
    args = ap.parse_args()

    if args.paths:
        # Tag each explicit input by its parent directory name, else its stem.
        sources = [(p, p.parent.name or p.stem) for p in args.paths]
    else:
        sources = [(_DEFAULT_LOG, _DATA_DIR.name or "audit")]

    stream = merge(sources, since=args.since, only_source=args.source)
    out = (
        sys.stdout if args.output is None else open(args.output, "w", encoding="utf-8")
    )
    try:
        for rec in stream:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
