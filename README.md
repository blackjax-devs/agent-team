# agent-team

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

**No clone — one line** (runs the server straight from GitHub; sagent comes along
via the direct-URL dep, no extra flags):
```bash
uvx --from git+https://github.com/blackjax-devs/agent-team.git agent-team serve --port 8767
# UI at http://127.0.0.1:8767/
# persistent install instead:  uv tool install git+https://github.com/blackjax-devs/agent-team.git
```

**From a clone** (for development or a custom profile):
```bash
uv sync                        # installs the package + the `agent-team` console script
agent-team serve --port 8767   # or: python -m agent_team.serve --port 8767
```

> **Run it in `tmux`.** However you launch it, `serve` is a foreground,
> long-running process — start it inside a **tmux** (or `screen`) session so it
> survives terminal/SSH disconnects and you can detach (`Ctrl-b` then `d`) and
> reattach (`tmux attach -t agent-team`) without killing the team:
> ```bash
> tmux new -s agent-team
> # …run the serve command above, then Ctrl-b d to detach; Ctrl-C in the window stops it
> ```

The bundled **default profile** ships inside the package, so it works out of the
box from any cwd. To use a custom profile, set `AGENT_TEAM_PROFILE_DIR` to your
profile dir. Per-session audit data (`main.jsonl`, `sessions/`) lands in the
launch cwd by default; set `SAGENT_DATA_DIR` to relocate it.

`agent-team --help` prints usage and exits without booting the server.

## Profiles
Behavior is driven by a **profile dir** (`AGENT_TEAM_PROFILE_DIR`): a
[`team.toml`](agent_team/profiles/default/team.toml) (roster, per-role model,
workspace, sandbox, session-id namespace, presets) + role prompt templates. The
bundled default profile ships inside the package
(`agent_team/profiles/default`) and targets "an app that uses BlackJAX"; it is
used automatically when `AGENT_TEAM_PROFILE_DIR` is unset. The BlackJAX dev team
itself runs this same framework against a **private** profile (its monorepo +
worklog).

Presets in the default profile: `dev-team` (the full five) and
`statistician-only` (a single curated debug agent).

## License
MIT — see [LICENSE](LICENSE).
