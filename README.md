# ai-dev-team

A configurable, **Bayesian-aware multi-agent dev team**, built on
[sagent](https://github.com/rekursiv-ai/sagent) + the Claude CLI. The default
profile is a team for **building and debugging applications that use
[BlackJAX](https://github.com/blackjax-devs/blackjax)** — but the workspace,
roster, models, and prompts are all config, so you can point it at any project.

> **Status: pre-release scaffold (milestone 2, in progress).** Extracted from the
> BlackJAX team's internal channel and being generalized + decoupled. Not yet
> published; the quickstart below is provisional. See `DECOUPLING.md` for the
> remaining extraction work.

## Two ways to use it

| Surface | What | Best for |
|---|---|---|
| **Agent pack** (Claude Code native) | a Bayesian-aware set of agents (`tl`, `swe`, `statistician`, …) + methodology checklists dropped into your repo's `.claude/`. One entry agent fans out to subagents. No server. | "finalize this PR", "debug why this won't sample" |
| **Channel** (sagent server) | a persistent multi-agent team with a web UI, peer messaging, and a shared worklog. Configurable roster — the full team, or a single curated agent (e.g. just the statistician) as a debug surface. | ongoing coordination; a dedicated debugging surface |

Both share the same roles, methodology, and workspace config; only the runtime differs.

## Requirements
- The **`claude` CLI**, authenticated — the framework shells out to `claude --print`.
- Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/).
- Loopback-only HTTP; write-capable roles are sandboxed.

## Quickstart (channel) — provisional
```bash
uv sync
AI_DEV_TEAM_PROFILE_DIR=profiles/default \
  .venv/bin/python bin/serve.py --port 8767
# UI at http://127.0.0.1:8767/
```

## Profiles
Behavior is driven by a **profile dir** (`AI_DEV_TEAM_PROFILE_DIR`): a
[`team.toml`](profiles/default/team.toml) (roster, per-role model, workspace,
sandbox, session-id namespace, presets) + role prompt templates. The default
profile targets "an app that uses BlackJAX." The BlackJAX dev team itself runs
this same framework against a **private** profile (its monorepo + worklog).

Presets in the default profile: `dev-team` (the full five) and
`statistician-only` (a single curated debug agent).

## License
MIT — see [LICENSE](LICENSE).
