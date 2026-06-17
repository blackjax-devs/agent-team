# examples

A clean clone can point the team at a real workspace using `sample-app/`.

## `sample-app/`

A minimal one-file BlackJAX project (`model.py`): a Normal-mean posterior
sampled with NUTS, plus a one-line `README.md`. It contains a **mild,
intentional bug** — `step_size=5.0` is far too large for the tightly-peaked
posterior, so acceptance collapses and the chain barely mixes. That is the
thing the team is meant to debug.

Run it standalone to see the symptom (low acceptance, poor estimate):

```bash
cd examples/sample-app
uv run python model.py
```

## Pointing the team at it

`sample-app/profile/team.toml` is a tiny, **self-contained** profile: it carries
its own copy of the statistician role prompt under `profile/roles/` (the default
profile now ships inside the installed `agent_team` package, so there is no
repo-root `profiles/default` to point back into). It sets
`roster = "statistician-only"` and `[workspace].root = "."`, so it spins up a
single statistician against whatever dir you launch from.

Two resolution rules to keep straight:

- `AGENT_TEAM_PROFILE_DIR` selects the **profile** (roster, models, prompts).
- `[workspace].root` resolves relative to the **launch cwd**, so launch from
  `examples/sample-app` to make that dir the workspace.

### Channel (sagent server) — statistician-only against `sample-app`

```bash
# from the repo root, after `uv sync` (installs the `agent-team` console script):
cd examples/sample-app
AGENT_TEAM_PROFILE_DIR=profile \
  ../../.venv/bin/agent-team --port 8767
# UI at http://127.0.0.1:8767/ — a single statistician debug surface,
# workspace = examples/sample-app (model.py).
```

`AGENT_TEAM_PROFILE_DIR=profile` is relative to the `examples/sample-app`
launch cwd; an absolute path works too. (`../../.venv/bin/agent-team` is just
the console script in the repo venv; once installed on `PATH` you can call
`agent-team` directly, or `python -m agent_team.serve`.)

### Verify the profile resolves (no model turns)

```bash
cd examples/sample-app
AGENT_TEAM_PROFILE_DIR=profile python -c "
from agent_team.team_profile import load_profile
p = load_profile()
print(p.roster, p.workspace_root, p.role_prompts['statistician'])
"
# -> ['statistician'] .../examples/sample-app .../examples/sample-app/profile/roles/statistician.md
```

### Reuse the full default roster instead

Drop `AGENT_TEAM_PROFILE_DIR` while still launching from
`examples/sample-app`; resolution falls back to the **bundled default profile**
(shipped inside the `agent_team` package), giving you the full
`tl/swe/junior-swe/statistician/tech-writer` team against the same workspace.
