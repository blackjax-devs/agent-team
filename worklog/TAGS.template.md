# Canonical tag vocabulary — TEMPLATE

Copy this file to `worklog/TAGS.md` in your org's worklog and replace the
example vocabulary with your own tags.

Tags are applied via `tags: [...]` in worklog frontmatter. The `worklog-tags`
hook enforces the substitution/drop rules in `worklog/.lintrc.yaml` — add them
there, not here (this file is human-readable documentation; the machine-readable
normalization lives in the lintrc).

## Style rules

- **lowercase, hyphen-separated** — `code-pattern`, `tool-harness`. Exception:
  code symbols that have underscores (e.g. a specific function name).
- **singular form** — `recert` not `recertification`.
- **no document-type tags** — `decision`, `index`, `dashboard` are implicit in
  path/filename; don't repeat them as tags.
- **no status tags** — `closed`, `fix`, `bug-fix` belong in the `status:` field.
- **3–5 tags per file** is the right density. More than 6 means at least one
  is redundant.

## Vocabulary (fill in for your org)

### Projects
- `<your-project>` · `<another-project>`

### Tooling
- `tool-harness` — harness/tool behaviour (Edit, Write, Bash quirks, etc.)
- `git` — git operations, branch management, CI
- `uv` — dependency/package management
- `pre-commit` — hook configuration and execution

### Process
- `worklog` — worklog substrate, indexing, conventions
- `process` — recurring process patterns and lessons
- `subagent` — subagent behaviour, communication, spawn patterns
- `bg-task` — background job discipline, polling, sentinel files

### Workflow
- `plan` — planning/architecture documents
- `retrospective` — post-arc synthesis
- `deferred` — captured future work

### Add org-specific vocabulary below:
# (e.g. sampler families, model names, phase identifiers)
