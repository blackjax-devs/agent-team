# worklog/ — coordination and knowledge layer

The `worklog/` directory is the agent team's shared, durable memory: process
decisions, recurring discoveries, in-flight task state, and a machine-built index.
It is designed to work **across sessions** — agents read it at session start to
reconstruct context without relying on conversation memory.

This README covers layout conventions, frontmatter schema, and tooling. Copy it
into your own `worklog/` when bootstrapping a new org config.

## Layout

```
worklog/
├── README.md                              # this file
├── INDEX.md                               # machine-built — do not edit by hand
├── TAGS.md                                # canonical tag vocabulary (org-specific)
├── .lintrc.yaml                           # per-org hook configuration (optional)
├── threads/<branch-or-slug>.md            # one file per active work thread
│   └── _archive/                          # closed threads (moved on merge)
├── decisions/YYYY-MM-DD-<slug>.md         # process/architecture decisions
└── lessons/
    ├── tool-harness/<slug>.md             # harness/tool quirks
    ├── code-patterns/<slug>.md            # recurring code-level patterns
    ├── process/<slug>.md                  # cross-task process lessons
    └── subagent-behaviour/<slug>.md       # agent communication patterns
```

## When to add what

| Situation | File |
|-----------|------|
| Starting a new branch / task | `threads/<branch>.md` |
| A non-trivial process or architecture decision was made | `decisions/YYYY-MM-DD-<slug>.md` |
| A recurring tooling, code, or agent quirk is worth capturing | `lessons/<category>/<slug>.md` |
| A thread closes (PR merged, work abandoned) | move file to `threads/_archive/` |
| A lesson supersedes an older one | new lesson file + `supersedes:` frontmatter linking the old one; old one gets status `SUPERSEDED` |

## Conventions

- **One file = one topic.** Multiple lessons in one file create merge conflicts.
- **Append-only for `decisions/` and `lessons/`** once committed. Corrections go
  into a new file with `supersedes:` linking the old one.
- **Dated slugs for decisions and lessons.** Format: `YYYY-MM-DD-<short-slug>.md`.
- **`threads/` uses the bare branch name** as the filename slug.

## YAML frontmatter (required on every file)

Every markdown file under `worklog/` (except `README.md` and `INDEX.md`) must open
with this YAML frontmatter block:

```yaml
---
status: CURRENT         # CURRENT | CLOSED | SUPERSEDED | PARTIALLY_SUPERSEDED | DEFERRED | DRAFT
date: 2026-01-01        # YYYY-MM-DD; from filename prefix or first-write date
tags: [tag-a, tag-b]   # 3-5 lowercase-hyphenated tags from worklog/TAGS.md
model: null             # or a model/project slug; used for case-study grouping
author: tl              # workstream slug ("tl" for single-contributor threads)
supersedes: []          # relative paths to files this file replaces
related:                # relative paths or URLs to related files
  - worklog/decisions/...
  - WORKLOG.md
---
```

**Status semantics:**

| Value | Meaning |
|-------|---------|
| `CURRENT` | Accurate as of last edit (default) |
| `CLOSED` | Task complete, no further action |
| `SUPERSEDED` | Fully replaced by another file (link it in `supersedes:`) |
| `PARTIALLY_SUPERSEDED` | Core lesson still holds; specific prescription refined elsewhere |
| `DEFERRED` | Captured as future work, not yet active |
| `DRAFT` | Work in progress |

## Tooling (from `agent-team`)

Three pre-commit hooks are provided by `blackjax-devs/agent-team`:

| Hook ID | What it does |
|---------|--------------|
| `worklog-lint` | Validates frontmatter fields, status enum, date format, cross-references |
| `worklog-index` | Rebuilds `worklog/INDEX.md` when worklog files change (formatter pattern) |
| `worklog-tags` | Fails if any tag is not normalized per the lintrc substitution map |

Install in your `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/blackjax-devs/agent-team
    rev: v0.1.0  # pin to a release tag
    hooks:
      - id: worklog-lint
      - id: worklog-index
      - id: worklog-tags
```

If your `worklog/` is not at the repo root (e.g. it lives under `project/`), pass
`--root` to each hook:

```yaml
      - id: worklog-lint
        args: [--root, project]
      - id: worklog-index
        args: [--root, project]
      - id: worklog-tags
        args: [--root, project, --check]
```

Run manually:

```bash
worklog-lint [--root DIR] [--verbose]
worklog-index [--root DIR] [--check]
worklog-tags [--root DIR] [--check]
```

## Per-org configuration: `worklog/.lintrc.yaml`

All tool behavior can be overridden via `worklog/.lintrc.yaml` (relative to the
`--root`). All keys are optional:

```yaml
# Paths (relative to --root) exempt from the frontmatter requirement.
# Default: [worklog/README.md, worklog/INDEX.md]
exempt_paths:
  - worklog/README.md
  - worklog/INDEX.md

# Path prefixes treated as structural sibling-repo refs in related:/supersedes:.
# These paths are accepted without a filesystem check (useful in CI where
# sibling repos are not checked out).
# Default: [] (all related: paths must exist on disk)
sibling_repo_prefixes:
  - "my-other-repo/"

# Tag normalization (worklog-tags hook only).
# tag_substitutions: {old: new}  — renames
# tag_drops: [tag, ...]           — removes entirely
# tag_splits: {compound: [a, b]} — expands to multiple tags
tag_substitutions:
  old-name: new-name
tag_drops:
  - obsolete-tag
tag_splits:
  compound-tag:
    - part-a
    - part-b
```

## Bootstrapping a new org

1. Copy this `README.md` into your repo's `worklog/`.
2. Copy `TAGS.template.md` to `worklog/TAGS.md` and edit the vocabulary.
3. Create `worklog/.lintrc.yaml` with your org's sibling prefixes.
4. Copy `pre-commit.template.yaml` to `.pre-commit-config.yaml` and set the `rev:`.
5. Create `WORKLOG.md` at the repo root (the active dashboard — pointers only,
   no content). Add the required frontmatter.
6. Run `worklog-index` to generate the initial `worklog/INDEX.md`.
7. Install pre-commit: `pip install pre-commit && pre-commit install`.
