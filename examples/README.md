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

## Pointing the team at it (documented reuse, no profile duplication)

`sample-app/profile/team.toml` is a tiny profile that **reuses the default
profile's role prompts** (its `roles.statistician.prompt` resolves back into
`profiles/default/roles/`). It sets `roster = "statistician-only"` and
`[workspace].root = "."`, so it spins up a single statistician against whatever
dir you launch from.

Two resolution rules to keep straight:

- `AI_DEV_TEAM_PROFILE_DIR` selects the **profile** (roster, models, prompts).
- `[workspace].root` resolves relative to the **launch cwd**, so launch from
  `examples/sample-app` to make that dir the workspace.

### Channel (sagent server) — statistician-only against `sample-app`

```bash
# from the repo root, after `uv sync`:
cd examples/sample-app
AI_DEV_TEAM_PROFILE_DIR=profile \
  ../../.venv/bin/python ../../bin/serve.py --port 8767
# UI at http://127.0.0.1:8767/ — a single statistician debug surface,
# workspace = examples/sample-app (model.py).
```

`AI_DEV_TEAM_PROFILE_DIR=profile` is relative to the `examples/sample-app`
launch cwd; an absolute path works too.

### Verify the profile resolves (no model turns)

```bash
cd examples/sample-app
AI_DEV_TEAM_PROFILE_DIR=profile python -c "
import sys; sys.path.insert(0, '../..')
from team_profile import load_profile
p = load_profile()
print(p.roster, p.workspace_root, p.role_prompts['statistician'])
"
# -> ['statistician'] .../examples/sample-app .../profiles/default/roles/statistician.md
```

### Reuse the full default roster instead

Drop `AI_DEV_TEAM_PROFILE_DIR` (or set it to `../../profiles/default`) while
still launching from `examples/sample-app`; you get the full
`tl/swe/junior-swe/statistician/tech-writer` team against the same workspace.
