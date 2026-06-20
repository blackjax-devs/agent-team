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

## Restart, recovery & large tapes

The channel persists each agent's full conversation (its sagent *tape*) to
`SAGENT_DATA_DIR/sessions/<role>.sagent/session.jsonl`, so stopping and
relaunching `serve` resumes every agent where it left off.

> **After any restart, give the team a few minutes to settle — don't touch
> anything in the meanwhile.** On boot each agent re-feeds its resumed tape to
> the `claude` CLI and re-establishes its MCP bridge — a warmup that can run for
> up to ~90s, after which agents recover on their first real turn (≈2–3 min
> total). **Don't send work or trigger another restart/slim during this
> window**: acting on a still-settling agent can crash its compaction and wedge
> it.

**Resume-slim** (the `slim` mode — materialize's fallback). The resumed tape is
trimmed to roughly its last `AGENT_TEAM_RESUME_KEEP` messages (default `120`),
snapped to a clean turn boundary, *before* it is re-fed to the CLI. This bounds
the first-turn re-feed so a large tape (a long-running coordinator can reach
megabytes) can't stall the MCP handshake on boot. It is deterministic (no model
call), the on-disk history is never rewritten, and small tapes are left
untouched. Set `AGENT_TEAM_RESUME_KEEP=0` to disable (resume the full tape), or
raise it to keep more context. **A server restart is therefore the reliable
recovery for an agent whose tape has grown too large to re-feed.**

**Resume mode (`AGENT_TEAM_RESUME_MODE`, default `materialize`).** Three modes:
- **`materialize`** (default) — instead of re-feeding the tape, write claude's
  session JSONL *directly* from the resumed context and resume it with a native
  `--resume`. There is no re-feed at all, so startup is effectively instant and
  each agent continues exactly mid-thread. It couples to claude's session-file
  format, so it is gated by a boot **drift-canary** (a throwaway `claude`
  round-trip diffed against the materializer's output); on drift — or if
  `claude` isn't available at boot — it falls back to `slim`.
- **`slim`** — the trimming described above: the deterministic fallback the
  canary degrades to, and selectable directly.
- **`full`** — resume the untrimmed tape (no slimming): the original fat-tape
  re-feed behaviour, kept as an escape hatch.

**Per-agent restart (debug console).** The console at `/debug` exposes three
intensities per agent (also `POST /api/restart {role, mode}`, loopback-only):

| mode | what it does | use when |
|---|---|---|
| `slim` | compacts the tape (summarize old turns, keep the recent thread); agent stays on-task | a long-but-healthy agent you don't want to interrupt |
| `soft` | clears history, preserves the inbox | a blunt reset |
| `reanchor` | clears history, then re-seeds the last couple of turns + a "re-sync with the lead before acting" directive | recovering a stuck/confused agent |

> `slim` runs a model call to summarize, so it only fits a **warm, settled**
> agent. The server **refuses** a live `slim` when the context is large
> (> `AGENT_TEAM_SLIM_MAX_MESSAGES`) or a turn is in flight — re-feeding a big
> tape inside the compact call is exactly what wedges an agent. For a very
> large or unresponsive tape, prefer a server restart (resume-slim) or
> `reanchor`.

Related env knobs:
- `AGENT_TEAM_RESUME_MODE` (default `materialize`) — `materialize` (native `--resume`, canary-gated, effectively-instant startup) | `slim` (fallback) | `full`.
- `AGENT_TEAM_RESUME_KEEP` (default `120`) — messages kept per tape on resume (slim mode/fallback); `0` resumes the full tape.
- `AGENT_TEAM_SLIM_MAX_MESSAGES` (default `250`) — max resolved-context size at which a live `slim` is allowed; above it the server refuses and points at a restart.
- `AGENT_TEAM_MCP_CONNECT_TIMEOUT_SEC` (default `25`) — seconds boot waits for the CLI's MCP bridge before respawning.
- `AGENT_TEAM_COMPACT_TRIGGER` (default `0.80`) — context-utilization fraction at which an agent auto-compacts mid-run.

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
